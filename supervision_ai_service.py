#!/usr/bin/env python3
"""
supervision_ai_service.py — SOTA 2025 VMS AI Service
=======================================================
Công nghệ:
  - YOLO11n (ONNX/GPU) — phát hiện xe, người, khuôn mặt
  - ByteTrack (Supervision) — tracking đa đối tượng
  - PaddleOCR v5 ONNX INLINE (không HTTP) — nhận diện biển số
  - CLAHE + Super-Resolution + Deskew + Perspective — preprocessing SOTA
  - Multi-attempt strategy — thử 5 cách, chọn confidence cao nhất
  - Per-model lock — không block nhau giữa các model
  - NVIDIA GPU acceleration — CUDA cho cả YOLO và OCR
"""

import os
import sys

# ── Ensure user site-packages (onnxruntime-gpu) is found BEFORE ultralytics import
import site as _site
_user_site = _site.getusersitepackages()
if _user_site and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import time
import datetime
import threading

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import psycopg2
from ultralytics import YOLO
import supervision as sv

# ─── Inline OCR (không qua HTTP, tốc độ cao nhất) ───────────────────────────
# Import trực tiếp từ ai_ocr_service.py để loại bỏ hoàn toàn HTTP overhead
_service_dir = os.path.dirname(os.path.abspath(__file__))
if _service_dir not in sys.path:
    sys.path.insert(0, _service_dir)

try:
    from ai_ocr_service import (
        create_ocr_engine,
        recognize_plate,
        OCR_ENGINE as _placeholder,
    )
    import ai_ocr_service as _ocr_module
    _OCR_IMPORT_OK = True
except ImportError as _e:
    print(f"[AI SERVICE WARNING] Could not import ai_ocr_service: {_e}", flush=True)
    _OCR_IMPORT_OK = False

# ─── Runtime configuration ───────────────────────────────────────────────────
SERVER_URL              = os.environ.get("VMS_SERVER_URL", "http://127.0.0.1:8080")
DB_DSN                  = os.environ.get(
    "DB_CONN_STR",
    "host=localhost port=5432 dbname=smart_monitoring user=postgres password=585858",
)
MAX_CAMERAS             = max(0, int(os.environ.get("AI_MAX_CAMERAS", "0")))
MAX_TRACK_ATTEMPTS      = max(1, int(os.environ.get("AI_MAX_TRACK_ATTEMPTS", "8")))
EVENT_COOLDOWN_SECONDS  = max(1, int(os.environ.get("AI_EVENT_COOLDOWN_SECONDS", "20")))
PLATE_RETRY_CONF_THRESH = float(os.environ.get("AI_PLATE_RETRY_CONF", "0.70"))
PROCESS_FPS             = float(os.environ.get("AI_PROCESS_FPS", "25"))   # 25 FPS for smooth tracking
FRAME_INTERVAL          = 1.0 / PROCESS_FPS

# ─── Global state ────────────────────────────────────────────────────────────
g_running       = True
active_threads  = {}
event_dedup     = {}
event_dedup_lock = threading.Lock()

g_ai_modules    = {}
g_ai_models     = {}
vehicle_model   = None
plate_model     = None
face_model      = None

def reload_yolo_models(new_models):
    global vehicle_model, plate_model, face_model, g_ai_models
    try:
        import onnxruntime as ort
        _providers = ort.get_available_providers()
        _use_gpu   = "CUDAExecutionProvider" in _providers
        
        if new_models.get('vehicle') and new_models['vehicle'] != g_ai_models.get('vehicle', ''):
            print(f"[AI SERVICE] Loading new vehicle model: {new_models['vehicle']}", flush=True)
            with vehicle_lock:
                vehicle_model = YOLO(f"models/{new_models['vehicle']}", task="detect")
                pass
            g_ai_models['vehicle'] = new_models['vehicle']
            
        if new_models.get('plate') and new_models['plate'] != g_ai_models.get('plate', ''):
            print(f"[AI SERVICE] Loading new plate model: {new_models['plate']}", flush=True)
            with plate_lock:
                plate_model = YOLO(f"models/{new_models['plate']}", task="detect")
                pass
            g_ai_models['plate'] = new_models['plate']
            
        if new_models.get('face') and new_models['face'] != g_ai_models.get('face', ''):
            print(f"[AI SERVICE] Loading new face model: {new_models['face']}", flush=True)
            with face_lock:
                face_model = YOLO(f"models/{new_models['face']}", task="detect")
                pass
            g_ai_models['face'] = new_models['face']
    except Exception as e:
        print(f"[AI SERVICE ERROR] Failed to reload YOLO models: {e}", flush=True)


# Per-model locks (tách riêng để tránh block nhau)
vehicle_lock = threading.Lock()
plate_lock   = threading.Lock()
face_lock    = threading.Lock()

import requests  # noqa: E402  (dùng cho push detection boxes lên C++ server)

# ─── Load YOLO Models ────────────────────────────────────────────────────────
print("[AI SERVICE] Loading YOLO models...", flush=True)
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("[AI SERVICE] PyTorch CPU threads limited to 1 for optimization.", flush=True)
except Exception as e:
    print(f"[AI SERVICE] Failed to limit PyTorch threads: {e}")

try:
    # Thử GPU trước
    import onnxruntime as ort
    _providers = ort.get_available_providers()
    _use_gpu   = "CUDAExecutionProvider" in _providers
    if _use_gpu:
        print("[AI SERVICE] NVIDIA GPU detected — CUDA enabled for all inference", flush=True)
    else:
        print("[AI SERVICE] No CUDA GPU — using CPU inference", flush=True)

    reload_yolo_models({
        "vehicle": "yolo11n.onnx",
        "plate": "best_plateSegment.onnx",
        "face": "yolo11n-face.onnx"
    })
    print("[AI SERVICE] Initial YOLO models loaded.", flush=True)
except Exception as e:
    print(f"[AI SERVICE ERROR] Failed to initialize models: {e}", flush=True)

# ─── Load Inline OCR Engine ──────────────────────────────────────────────────
if _OCR_IMPORT_OK:
    print("[AI SERVICE] Loading PaddleOCR v5 ONNX engine (inline, no HTTP)...", flush=True)
    try:
        _ocr_module.OCR_ENGINE = create_ocr_engine(use_gpu=_use_gpu)
        print("[AI SERVICE] PaddleOCR v5 engine ready.", flush=True)
    except Exception as _e:
        print(f"[AI SERVICE WARNING] Inline OCR failed: {_e}. Will fallback to HTTP OCR.", flush=True)
        _OCR_IMPORT_OK = False


# ─── Helpers ─────────────────────────────────────────────────────────────────

def save_crop(img: np.ndarray, prefix: str, cam_id: str) -> str:
    if img is None or img.size == 0:
        return ""
    try:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            gaussian = cv2.GaussianBlur(img_clahe, (5, 5), 1.0)
            img_enhanced = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
        except Exception:
            img_enhanced = img
            
        os.makedirs("snapshots/detections", exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ms  = int(time.time() * 1000) % 1000
        save_crop.counter += 1
        filename = (
            f"snapshots/detections/{prefix}_{cam_id}_{ts}_{ms}_{save_crop.counter}.jpg"
        )
        cv2.imwrite(filename, img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filename
    except Exception as exc:
        print(f"[AI ERROR] Failed to save crop: {exc}", flush=True)
        return ""

save_crop.counter = 0


def ensure_database_schema():
    try:
        with psycopg2.connect(DB_DSN, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_text "
                    "VARCHAR(20) DEFAULT ''"
                )
                cur.execute(
                    "ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_image_path "
                    "VARCHAR(500) DEFAULT ''"
                )
        print("[AI SERVICE] Database schema is ready.", flush=True)
    except Exception as error:
        print(f"[AI SERVICE WARNING] Could not prepare event columns: {error}", flush=True)


def should_save_event(camera_id: str, object_type: str, plate_text: str, tracker_id: int = 0) -> bool:
    now       = time.monotonic()
    event_key = (camera_id, object_type, plate_text or "-", tracker_id)
    with event_dedup_lock:
        previous = event_dedup.get(event_key, 0.0)
        if now - previous < EVENT_COOLDOWN_SECONDS:
            return False
        event_dedup[event_key] = now
        if len(event_dedup) > 2048:
            cutoff = now - EVENT_COOLDOWN_SECONDS
            for k, ts in list(event_dedup.items()):
                if ts < cutoff:
                    event_dedup.pop(k, None)
    return True


def do_ocr(plate_crop: np.ndarray):
    """
    Nhận diện biển số.
    Ưu tiên: inline PaddleOCR v5 → HTTP fallback (nếu inline lỗi).
    Trả về (plate_text, plate_confidence).
    """
    if plate_crop is None or plate_crop.size == 0:
        return "", 0.0

    # Phóng to biển số bằng OpenCV để OCR nhận diện dễ hơn (phóng 2.5x)
    h, w = plate_crop.shape[:2]
    if h > 0 and w > 0:
        plate_crop = cv2.resize(plate_crop, (int(w * 2.5), int(h * 2.5)), interpolation=cv2.INTER_CUBIC)

    # ── Inline OCR (SOTA 2025 multi-attempt) ──
    if _OCR_IMPORT_OK and _ocr_module.OCR_ENGINE is not None:
        try:
            text, conf = recognize_plate(plate_crop)
            return text, conf
        except Exception as exc:
            print(f"[AI SERVICE] Inline OCR error: {exc}", flush=True)

    # ── HTTP Fallback (tương thích ngược) ──
    try:
        import base64
        _, buffer  = cv2.imencode(".jpg", plate_crop)
        img_b64    = base64.b64encode(buffer).decode("utf-8")
        ocr_res    = requests.post(
            "http://127.0.0.1:8765/ocr",
            json={"image_b64": img_b64, "ts": time.time()},
            timeout=(0.5, 6.0),
        )
        if ocr_res.status_code == 200:
            data = ocr_res.json()
            if data.get("success") or data.get("text"):
                return data.get("text", ""), float(data.get("confidence", 0.0))
    except Exception as exc:
        print(f"[AI SERVICE] HTTP OCR fallback error: {exc}", flush=True)

    return "", 0.0


def save_detection_to_db(
    camera_id, object_type, plate_text, plate_conf, identity_label,
    identity_conf, vehicle_img, plate_img, face_img, bx, by, bw, bh, location, tracker_id=0
):
    if not should_save_event(camera_id, object_type, plate_text, tracker_id):
        return
    try:
        with psycopg2.connect(DB_DSN, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO detections
                    (camera_id, object_type, plate_text, plate_confidence,
                     identity_label, identity_confidence,
                     vehicle_image_path, plate_image_path, face_image_path,
                     bbox_x, bbox_y, bbox_w, bbox_h, location_name)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (camera_id, object_type, plate_text, plate_conf,
                     identity_label, identity_conf,
                     vehicle_img, plate_img, face_img,
                     bx, by, bw, bh, location),
                )
                if object_type == "person":
                    cur.execute(
                        "INSERT INTO events (camera_id, event_type, description, "
                        "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                        (camera_id, "person_detected", "Nguoi di bo", 0.8, vehicle_img),
                    )
                elif object_type.startswith("vehicle_"):
                    desc = (f"Phat hien phuong tien: {plate_text}"
                            if plate_text else "Phat hien phuong tien")
                    cur.execute(
                        "INSERT INTO events (camera_id, event_type, description, "
                        "confidence, snapshot_path, plate_text, plate_image_path) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (camera_id, "vehicle_detected", desc, 0.8,
                         vehicle_img, plate_text, plate_img),
                    )
                    if plate_text:
                        cur.execute(
                            "INSERT INTO events (camera_id, event_type, description, "
                            "confidence, snapshot_path, plate_text, plate_image_path) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (camera_id, "plate_recognized",
                             f"Bien so xe: {plate_text}", plate_conf,
                             plate_img or vehicle_img, plate_text, plate_img),
                        )
                elif object_type.startswith("face"):
                    cur.execute(
                        "INSERT INTO events (camera_id, event_type, description, "
                        "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                        (camera_id, "face_recognized",
                         identity_label or "Nhan dien khuon mat",
                         identity_conf or 0.8, face_img or vehicle_img),
                    )
                elif object_type == "motion":
                    cur.execute(
                        "INSERT INTO events (camera_id, event_type, description, "
                        "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                        (camera_id, "motion_detected", "Phat hien chuyen dong",
                         1.0, vehicle_img),
                    )
                conn.commit()
    except psycopg2.errors.UndefinedColumn:
        # Fallback: chèn không có plate_text/plate_image_path columns
        try:
            with psycopg2.connect(DB_DSN, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    if object_type == "person":
                        cur.execute(
                            "INSERT INTO events (camera_id, event_type, description, "
                            "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                            (camera_id, "person_detected", "Nguoi di bo", 0.8, vehicle_img),
                        )
                    elif object_type.startswith("vehicle_"):
                        cur.execute(
                            "INSERT INTO events (camera_id, event_type, description, "
                            "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                            (camera_id, "vehicle_detected",
                             f"Phat hien phuong tien: {plate_text or ''}",
                             0.8, vehicle_img),
                        )
                    elif object_type == "motion":
                        cur.execute(
                            "INSERT INTO events (camera_id, event_type, description, "
                            "confidence, snapshot_path) VALUES (%s,%s,%s,%s,%s)",
                            (camera_id, "motion_detected", "Phat hien chuyen dong",
                             1.0, vehicle_img),
                        )
                    conn.commit()
        except Exception as exc:
            print(f"[AI ERROR] DB fallback save error: {exc}", flush=True)
    except Exception as exc:
        print(f"[AI ERROR] DB save error: {exc}", flush=True)


# ─── Per-Camera Processing Thread ────────────────────────────────────────────

def process_camera(cam_id: str, rtsp_url: str, location_name: str):
    global g_running
    print(f"[AI SERVICE] Starting camera: {cam_id} ({rtsp_url})", flush=True)

    tracker         = sv.ByteTrack()
    processed_tracks = {}   # tracker_id → plate_text or "processed"
    track_attempts   = {}   # tracker_id → int (how many attempts made)
    track_conf_cache = {}   # tracker_id → best confidence so far

    fgbg            = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    cap             = cv2.VideoCapture(rtsp_url)
    last_frame_time = 0.0
    reconnect_time  = 0.0

    while g_running and cam_id in active_threads:
        # ── Reconnect logic ──────────────────────────────────────────────────
        if not cap.isOpened():
            if time.time() - reconnect_time > 5:
                print(f"[AI SERVICE] Reconnecting {cam_id}...", flush=True)
                cap = cv2.VideoCapture(rtsp_url)
                reconnect_time = time.time()
            time.sleep(1.0)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1.0)
            continue

        # ── Frame rate control ───────────────────────────────────────────────
        now = time.time()
        if now - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now

        h, w = frame.shape[:2]

        # ── 0. Motion Detection ──────────────────────────────────────────────
        fgmask = fgbg.apply(frame)
        motion_area = cv2.countNonZero(fgmask)
        if motion_area > 1500:
            if should_save_event(cam_id, "motion", "", 0):
                motion_path = save_crop(frame, "motion", cam_id)
                save_detection_to_db(
                    cam_id, "motion", "", 0.0, "", 0.0,
                    motion_path, "", "", 0, 0, w, h, location_name, 0
                )

        # ── 1. Vehicle Detection (YOLO11n) ───────────────────────────────────
        enable_vehicle = g_ai_modules.get('vehicle', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        enable_plate   = g_ai_modules.get('plate', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        enable_face    = g_ai_modules.get('face', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        
        boxes_payload = []
        if not enable_vehicle or vehicle_model is None:
            time.sleep(FRAME_INTERVAL)
            continue
            
        with vehicle_lock:
            results    = vehicle_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Classes: 0=person, 2=car, 3=motorbike, 5=bus, 7=truck
        target_classes = [0, 2, 3, 5, 7]
        detections = detections[np.isin(detections.class_id, target_classes)]
        detections = detections[detections.confidence >= 0.45]
        detections = tracker.update_with_detections(detections)

        boxes_payload = []

        # ── 2. Per-detection processing ──────────────────────────────────────
        for xyxy, mask, confidence, class_id, tracker_id, data in detections:
            tracker_id = int(tracker_id)
            class_id   = int(class_id)

            bx1 = max(0, int(xyxy[0]))
            by1 = max(0, int(xyxy[1]))
            bx2 = min(w, int(xyxy[2]))
            by2 = min(h, int(xyxy[3]))
            bw  = bx2 - bx1
            bh  = by2 - by1
            if bw < 15 or bh < 15:
                continue

            # Human-readable label
            label_map = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
            label = label_map.get(class_id, "vehicle")

            # Hiển thị biển số đã nhận diện nếu có
            cached_plate = processed_tracks.get(tracker_id)
            if cached_plate and cached_plate != "processed":
                label = cached_plate

            boxes_payload.append({
                "x": bx1, "y": by1, "w": bw, "h": bh,
                "label": label,
                "confidence": float(confidence),
            })

            # ── Xử lý track mới (chưa nhận diện đủ) ─────────────────────────
            if cached_plate and cached_plate != "processed":
                continue   # Đã đọc được biển số tốt → skip

            # Smart padding engine
            padding_map = {"person": 0.15, "car": 0.12, "motorbike": 0.15, "bus": 0.10, "truck": 0.10}
            pad_ratio = padding_map.get(label, 0.12)
            margin_x = int(bw * pad_ratio)
            margin_y = int(bh * pad_ratio)
            cx1 = max(0, bx1 - margin_x)
            cy1 = max(0, by1 - margin_y)
            cx2 = min(w, bx2 + margin_x)
            cy2 = min(h, by2 + margin_y)
            obj_crop = frame[cy1:cy2, cx1:cx2]

            # ── Vehicle: nhận diện biển số ────────────────────────────────────
            if class_id in (2, 3, 5, 7):
                # Plate detection (YOLO11n-plate)
                with plate_lock:
                    plate_results = plate_model(obj_crop, verbose=False)[0]
                plate_dets = sv.Detections.from_ultralytics(plate_results)

                plate_text = ""
                plate_conf = 0.0
                plate_path = ""

                if len(plate_dets) > 0:
                    # Lấy plate có confidence cao nhất (không chỉ plate đầu tiên)
                    best_plate_idx = int(np.argmax(plate_dets.confidence))
                    pxxy = plate_dets.xyxy[best_plate_idx]
                    pbw = int(pxxy[2] - pxxy[0])
                    pbh = int(pxxy[3] - pxxy[1])
                    pmx = int(pbw * 0.05)
                    pmy = int(pbh * 0.05)
                    px1  = max(0, int(pxxy[0]) - pmx)
                    py1  = max(0, int(pxxy[1]) - pmy)
                    px2  = min(obj_crop.shape[1], int(pxxy[2]) + pmx)
                    py2  = min(obj_crop.shape[0], int(pxxy[3]) + pmy)

                    plate_crop = obj_crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        # ── SOTA Multi-attempt OCR ────────────────────────────
                        plate_text, plate_conf = do_ocr(plate_crop)
                        if plate_text:
                            plate_path = save_crop(plate_crop, "plate", cam_id)

                # Adaptive retry: tiếp tục thử nếu confidence thấp
                current_best_conf = track_conf_cache.get(tracker_id, 0.0)
                if plate_conf > current_best_conf and plate_text:
                    track_conf_cache[tracker_id] = plate_conf
                    processed_tracks[tracker_id] = plate_text
                    print(
                        f"[AI SERVICE] {cam_id} → Plate: {plate_text} "
                        f"(conf={plate_conf:.3f})",
                        flush=True,
                    )

                attempt_count = track_attempts.get(tracker_id, 0) + 1
                track_attempts[tracker_id] = attempt_count

                # Điều kiện dừng:
                # a) Đã đọc được và confidence đủ cao → dừng
                # b) Đã thử đủ MAX_TRACK_ATTEMPTS lần → dừng
                if plate_text and plate_conf >= PLATE_RETRY_CONF_THRESH:
                    # Confidence tốt → không cần thử thêm
                    pass
                elif attempt_count >= MAX_TRACK_ATTEMPTS:
                    # Hết lượt thử → đánh dấu processed
                    if tracker_id not in processed_tracks:
                        processed_tracks[tracker_id] = "processed"
                else:
                    # Còn lần thử → không continue để cho lần sau cơ hội đọc lại
                    pass

                # Lưu vào DB (chỉ khi vừa đọc được biển số lần này)
                if plate_text and plate_path:
                    veh_path = save_crop(obj_crop, "vehicle", cam_id)
                    save_detection_to_db(
                        cam_id, "vehicle_" + label_map.get(class_id, "vehicle"),
                        plate_text, plate_conf, "", 0.0,
                        veh_path, plate_path, "",
                        bx1, by1, bw, bh, location_name, tracker_id
                    )
                elif attempt_count >= MAX_TRACK_ATTEMPTS and tracker_id not in [
                    k for k, v in processed_tracks.items() if v != "processed"
                ]:
                    # Hết lần thử, không đọc được biển → vẫn lưu vehicle event
                    if should_save_event(cam_id, "vehicle_" + label, "", tracker_id):
                        veh_path = save_crop(obj_crop, "vehicle", cam_id)
                        save_detection_to_db(
                            cam_id, "vehicle_" + label,
                            "", 0.0, "", 0.0,
                            veh_path, "", "",
                            bx1, by1, bw, bh, location_name, tracker_id
                        )

            # ── Person: phát hiện khuôn mặt ──────────────────────────────────
            elif class_id == 0:
                if tracker_id in processed_tracks:
                    continue
                processed_tracks[tracker_id] = "processed"

                with face_lock:
                    face_results = face_model(obj_crop, verbose=False)[0]
                face_dets = sv.Detections.from_ultralytics(face_results)
                face_dets = face_dets[face_dets.confidence >= 0.45]

                face_path  = ""
                face_label = ""
                face_conf  = 0.0

                if len(face_dets) > 0:
                    fxxy = face_dets.xyxy[0]
                    fx1  = max(0, int(fxxy[0]))
                    fy1  = max(0, int(fxxy[1]))
                    fx2  = min(obj_crop.shape[1], int(fxxy[2]))
                    fy2  = min(obj_crop.shape[0], int(fxxy[3]))
                    face_crop = obj_crop[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        face_path = save_crop(face_crop, "face", cam_id)

                pers_path = save_crop(obj_crop, "person", cam_id)
                save_detection_to_db(
                    cam_id, "person", "", 0.0,
                    face_label, face_conf, pers_path, "", face_path,
                    bx1, by1, bw, bh, location_name, tracker_id
                )

        # ── 3. Push boxes lên C++ server (Web UI overlay) ────────────────────
        payload = {
            "camera_id":    cam_id,
            "frame_width":  int(w),
            "frame_height": int(h),
            "boxes":        boxes_payload,
        }
        try:
            requests.post(
                f"{SERVER_URL}/api/detections",
                json=payload,
                timeout=(0.3, 1.0),
            )
        except Exception:
            pass

    cap.release()
    print(f"[AI SERVICE] Thread stopped: {cam_id}", flush=True)


# ─── Camera Monitor ───────────────────────────────────────────────────────────

def monitor_cameras():
    global g_running, active_threads
    print("[AI SERVICE] Starting camera monitor...", flush=True)

    while g_running:
        try:
            # 1. Kiểm tra AI có bật không
            global_enabled = True
            try:
                ctrl_res = requests.get(
                    f"{SERVER_URL}/api/ai/control", timeout=(0.5, 2.0)
                )
                if ctrl_res.status_code == 200:
                    global_enabled = ctrl_res.json().get("enabled", True)
            except Exception:
                pass


            global g_ai_modules
            try:
                mod_res = requests.get(f"{SERVER_URL}/api/ai/modules", timeout=(0.5, 2.0))
                if mod_res.status_code == 200:
                    g_ai_modules = mod_res.json().get('modules', mod_res.json())
            except Exception: pass
            
            try:
                modl_res = requests.get(f"{SERVER_URL}/api/ai/models", timeout=(0.5, 2.0))
                if modl_res.status_code == 200:
                    sel = modl_res.json().get("selected", {})
                    if sel: reload_yolo_models(sel)
            except Exception: pass

            if not global_enabled:
                if active_threads:
                    print("[AI SERVICE] Global AI disabled. Stopping all threads.", flush=True)
                    active_threads.clear()
                time.sleep(3.0)
                continue

            # 2. Lấy danh sách camera ID từ C++ server
            ai_cam_ids = []
            try:
                res = requests.get(
                    f"{SERVER_URL}/api/ai/cameras", timeout=(0.5, 2.0)
                )
                if res.status_code == 200:
                    ai_cam_ids = sorted(set(res.json()))
                    if MAX_CAMERAS > 0:
                        ai_cam_ids = ai_cam_ids[:MAX_CAMERAS]
            except Exception as exc:
                print(f"[AI SERVICE WARNING] Failed to fetch cameras: {exc}", flush=True)

            # 3. Query DB để lấy RTSP URL
            current_cameras = {}
            if ai_cam_ids:
                try:
                    with psycopg2.connect(DB_DSN, connect_timeout=3) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT camera_id, rtsp_url, name FROM cameras "
                                "WHERE camera_id = ANY(%s) AND is_active = true",
                                (ai_cam_ids,),
                            )
                            rows = cur.fetchall()
                    current_cameras = {r[0]: (r[1], r[2]) for r in rows}
                except Exception as exc:
                    print(f"[AI SERVICE WARNING] DB query failed: {exc}", flush=True)

            # 4. Start new threads
            for cam_id, (rtsp_url, name) in current_cameras.items():
                if cam_id not in active_threads:
                    local_rtsp = f"rtsp://127.0.0.1:8554/{cam_id}_hd"
                    t = threading.Thread(
                        target=process_camera,
                        args=(cam_id, local_rtsp, name),
                        daemon=True,
                    )
                    active_threads[cam_id] = t
                    t.start()

            # 5. Stop removed threads
            for cam_id in list(active_threads.keys()):
                if cam_id not in current_cameras:
                    print(f"[AI SERVICE] Stopping: {cam_id}", flush=True)
                    del active_threads[cam_id]

        except Exception as err:
            print(f"[AI SERVICE MONITOR ERROR] {err}", flush=True)

        time.sleep(3.0)


# ─── Entry Point ──────────────────────────────────────────────────────────────


# ─── Test Frame API (Mini HTTP Server) ───────────────────────────────────────
class TestFrameHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Tắt log ồn ào
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/test-frame':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                img_b64 = data.get("image_b64", "")
                img_bytes = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                boxes_payload = []
                results_payload = []
                if frame is not None:
                    enable_vehicle = g_ai_modules.get('vehicle', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
                    enable_plate   = g_ai_modules.get('plate', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
                    enable_face    = g_ai_modules.get('face', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True

                    # Xe / Người
                    if enable_vehicle and vehicle_model is not None:
                        with vehicle_lock:
                            res = vehicle_model(frame, verbose=False)[0]
                        dets = sv.Detections.from_ultralytics(res)
                        for xyxy, conf, class_id in zip(dets.xyxy, dets.confidence, dets.class_id):
                            if class_id not in [0, 2, 3, 5, 7] or conf < 0.45: continue
                            bx1, by1, bx2, by2 = map(int, xyxy)
                            bw, bh = bx2 - bx1, by2 - by1
                            if bw < 15 or bh < 15: continue
                            
                            label_map = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
                            label = label_map.get(int(class_id), "vehicle")
                            
                            boxes_payload.append({
                                "x": bx1, "y": by1, "w": bw, "h": bh,
                                "label": label,
                                "confidence": float(conf)
                            })
                            
                            if int(class_id) in [2, 3, 5, 7] and enable_plate and plate_model is not None:
                                margin_x, margin_y = int(bw * 0.15), int(bh * 0.15)
                                cy1, cy2 = max(0, by1 - margin_y), min(frame.shape[0], by2 + margin_y)
                                cx1, cx2 = max(0, bx1 - margin_x), min(frame.shape[1], bx2 + margin_x)
                                obj_crop = frame[cy1:cy2, cx1:cx2]
                                if obj_crop.size > 0:
                                    with plate_lock:
                                        pres = plate_model(obj_crop, verbose=False)[0]
                                    pdets = sv.Detections.from_ultralytics(pres)
                                    if len(pdets) > 0:
                                        best_p = int(np.argmax(pdets.confidence))
                                        pxxy = pdets.xyxy[best_p]
                                        py1, py2 = max(0, int(pxxy[1])), min(obj_crop.shape[0], int(pxxy[3]))
                                        px1, px2 = max(0, int(pxxy[0])), min(obj_crop.shape[1], int(pxxy[2]))
                                        pcrop = obj_crop[py1:py2, px1:px2]
                                        ptext, pconf = do_ocr(pcrop)
                                        if ptext:
                                            boxes_payload[-1]["label"] = ptext
                                            results_payload.append({
                                                "type": "plate",
                                                "text": ptext,
                                                "confidence": float(pconf),
                                                "vehicle_type": label
                                            })
                                            
                    # Khuôn mặt
                    if enable_face and face_model is not None:
                        with face_lock:
                            fres = face_model(frame, verbose=False)[0]
                        fdets = sv.Detections.from_ultralytics(fres)
                        for xyxy, conf, class_id in zip(fdets.xyxy, fdets.confidence, fdets.class_id):
                            if conf < 0.45: continue
                            bx1, by1, bx2, by2 = map(int, xyxy)
                            bw, bh = bx2 - bx1, by2 - by1
                            boxes_payload.append({
                                "x": bx1, "y": by1, "w": bw, "h": bh,
                                "label": "face",
                                "confidence": float(conf)
                            })
                            results_payload.append({
                                "type": "face",
                                "text": "Face Detected",
                                "confidence": float(conf)
                            })
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "boxes": boxes_payload,
                    "results": results_payload
                }).encode('utf-8'))
            except Exception as e:
                print(f"[AI SERVICE] Test frame error: {e}", flush=True)
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run_test_server():
    server = ThreadingHTTPServer(('0.0.0.0', 18081), TestFrameHandler)
    print("[AI SERVICE] Test API server listening on 0.0.0.0:18081", flush=True)
    server.serve_forever()

if __name__ == "__main__":

    print("=" * 60, flush=True)
    print("=== VMS SOTA 2025 AI Service ===", flush=True)
    print(f"=== GPU: {'CUDA Enabled' if _use_gpu else 'CPU Only'} ===", flush=True)
    print(f"=== OCR: {'Inline PaddleOCR v5' if _OCR_IMPORT_OK else 'HTTP Sidecar'} ===", flush=True)
    print(f"=== Process FPS: {PROCESS_FPS} | Max Track Attempts: {MAX_TRACK_ATTEMPTS} ===", flush=True)
    print("=" * 60, flush=True)

    ensure_database_schema()
    print(
        "[AI SERVICE] Capacity: "
        + (str(MAX_CAMERAS) if MAX_CAMERAS else "all enabled")
        + " cameras",
        flush=True,
    )


    test_srv_thread = threading.Thread(target=run_test_server, daemon=True)
    test_srv_thread.start()
    
    monitor_thread = threading.Thread(target=monitor_cameras, daemon=True)

    monitor_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        g_running = False
        print("[AI SERVICE] Shutting down.", flush=True)
