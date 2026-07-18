import re

with open('supervision_ai_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add imports
imports_addition = """
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
"""
code = code.replace('import threading\n', 'import threading\n' + imports_addition)

# 2. Add globals
globals_addition = """
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
                if _use_gpu: vehicle_model.overrides["device"] = "cuda:0"
            g_ai_models['vehicle'] = new_models['vehicle']
            
        if new_models.get('plate') and new_models['plate'] != g_ai_models.get('plate', ''):
            print(f"[AI SERVICE] Loading new plate model: {new_models['plate']}", flush=True)
            with plate_lock:
                plate_model = YOLO(f"models/{new_models['plate']}", task="detect")
                if _use_gpu: plate_model.overrides["device"] = "cuda:0"
            g_ai_models['plate'] = new_models['plate']
            
        if new_models.get('face') and new_models['face'] != g_ai_models.get('face', ''):
            print(f"[AI SERVICE] Loading new face model: {new_models['face']}", flush=True)
            with face_lock:
                face_model = YOLO(f"models/{new_models['face']}", task="detect")
                if _use_gpu: face_model.overrides["device"] = "cuda:0"
            g_ai_models['face'] = new_models['face']
    except Exception as e:
        print(f"[AI SERVICE ERROR] Failed to reload YOLO models: {e}", flush=True)

"""
code = code.replace('event_dedup_lock = threading.Lock()\n', 'event_dedup_lock = threading.Lock()\n' + globals_addition)

# 3. Replace YOLO model loading
yolo_load_original = """    vehicle_model = YOLO("models/yolo11n.onnx",       task="detect")
    plate_model   = YOLO("models/yolo11n-plate.onnx", task="detect")
    face_model    = YOLO("models/yolo11n-face.onnx",  task="detect")

    if _use_gpu:
        # Force ONNX Runtime providers cho ultralytics models
        vehicle_model.overrides["device"] = "cuda:0"
        plate_model.overrides["device"]   = "cuda:0"
        face_model.overrides["device"]    = "cuda:0"

    print("[AI SERVICE] All YOLO models loaded successfully.", flush=True)
except Exception as e:
    print(f"[AI SERVICE ERROR] Failed to load YOLO models: {e}", flush=True)
    sys.exit(1)"""

yolo_load_new = """    reload_yolo_models({
        "vehicle": "yolo11n.onnx",
        "plate": "best_plateSegment.onnx",
        "face": "yolo11n-face.onnx"
    })
    print("[AI SERVICE] Initial YOLO models loaded.", flush=True)
except Exception as e:
    print(f"[AI SERVICE ERROR] Failed to initialize models: {e}", flush=True)"""

code = code.replace(yolo_load_original, yolo_load_new)

# 4. In monitor_cameras, fetch models and modules
monitor_cams_addition = """
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
"""
code = code.replace('            if not global_enabled:', monitor_cams_addition + '\n            if not global_enabled:')


# 5. In process_camera, guard with enable_vehicle
code = code.replace(
    '# ── 1. Vehicle Detection (YOLO11n) ───────────────────────────────────\n        with vehicle_lock:',
    """# ── 1. Vehicle Detection (YOLO11n) ───────────────────────────────────
        enable_vehicle = g_ai_modules.get('vehicle', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        enable_plate   = g_ai_modules.get('plate', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        enable_face    = g_ai_modules.get('face', {}).get('enabled', True) if isinstance(g_ai_modules, dict) else True
        
        boxes_payload = []
        if not enable_vehicle or vehicle_model is None:
            time.sleep(FRAME_INTERVAL)
            continue
            
        with vehicle_lock:"""
)

# 6. Add HTTP server
http_server_code = """
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
                            if class_id not in [0, 2, 3, 5, 7]: continue
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
"""
code = code.replace('if __name__ == "__main__":', http_server_code)

start_thread_code = """
    test_srv_thread = threading.Thread(target=run_test_server, daemon=True)
    test_srv_thread.start()
    
    monitor_thread = threading.Thread(target=monitor_cameras, daemon=True)
"""
code = code.replace('    monitor_thread = threading.Thread(target=monitor_cameras, daemon=True)', start_thread_code)

with open('supervision_ai_service.py', 'w', encoding='utf-8') as f:
    f.write(code)
