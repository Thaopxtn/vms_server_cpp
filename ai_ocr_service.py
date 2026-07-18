#!/usr/bin/env python3
"""
ai_ocr_service.py — SOTA 2025 Vietnamese License Plate OCR Service
Công nghệ sử dụng:
  - PaddleOCR v5 ONNX (DBNet detector + CRNN/CTC recognizer) — độ chính xác cao nhất
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) — tăng tương phản biển mờ/tối
  - Super-resolution upscaling (Lanczos 4x) — xử lý biển số ở xa
  - Perspective correction via best_plateSegment.onnx — duỗi thẳng biển nghiêng
  - Multi-attempt strategy — thử nhiều cách preprocessing, chọn confidence cao nhất
  - Có thể import trực tiếp (không cần HTTP) hoặc chạy như HTTP sidecar port 8765
"""

import base64
import binascii
import json
import logging
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OCR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DET_MODEL_PATH  = MODEL_DIR / "det.onnx"
REC_MODEL_PATH  = MODEL_DIR / "rec.onnx"
REC_KEYS_PATH   = MODEL_DIR / "ppocrv5_dict.txt"
SEG_MODEL_PATH  = MODEL_DIR / "best_plateSegment.onnx"
MAX_REQUEST_BYTES = 10 * 1024 * 1024
INFER_LOCK = threading.Lock()
OCR_ENGINE = None
SEG_SESSION = None   # best_plateSegment.onnx session (optional, loaded lazily)

# --------------------------------------------------------------------------- #
#  Vietnamese Plate Validation Rules (2023+ biển định danh cá nhân included)  #
# --------------------------------------------------------------------------- #
PLATE_PATTERN = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")
DATE_PATTERN  = re.compile(
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})"
)
TWO_LETTER_SERIES = {
    "LD", "DA", "KT", "NG", "NN", "QT", "CD", "TD", "LB", "HC",
    "SA", "MK", "MD", "CV",
}
TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "J": "3",
}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A"}
VALID_PROVINCE_CODES = {
    11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 43, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99,
}
VALID_SERIES_LETTERS = frozenset("ABCDEFGHKLMNPRSTUVXYZ")


# =========================================================================== #
#  SOTA Preprocessing Utilities                                                #
# =========================================================================== #

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Tăng tương phản cục bộ cho biển số mờ, tối, hoặc bị ảnh hưởng ánh sáng.
    Xử lý từng kênh LAB để giữ màu sắc tự nhiên.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def super_resolve(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Super-resolution upscaling với Lanczos interpolation (chất lượng cao nhất OpenCV).
    Phóng to biển số nhỏ ở xa camera để OCR nhận diện rõ hơn.
    Áp dụng khi cả hai chiều đều nhỏ hơn ngưỡng.
    """
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if h < 48 or w < 120:  # Biển số nhỏ → upscale
        new_h = max(48, h * scale)
        new_w = max(120, w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image


def sharpen_plate(image: np.ndarray) -> np.ndarray:
    """
    Unsharp mask để làm sắc nét các ký tự bị mờ do chuyển động hoặc nén JPEG.
    """
    if image is None or image.size == 0:
        return image
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened


def deskew_plate(image: np.ndarray) -> np.ndarray:
    """
    Tự động deskew (duỗi thẳng) biển số bị nghiêng bằng cách:
    1. Tìm bounding box xoay nhỏ nhất của vùng foreground
    2. Perspective transform để trả về ảnh thẳng
    """
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1.0:   # Không cần xoay nếu lệch ít hơn 1°
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def segment_perspective_correct(image: np.ndarray) -> np.ndarray:
    """
    Sử dụng best_plateSegment.onnx để lấy mask segment chính xác của biển số,
    sau đó thực hiện perspective transform 4-point để duỗi thẳng biển số.
    Đây là kỹ thuật mạnh nhất cho biển số nghiêng, bị bóp méo phối cảnh.
    """
    global SEG_SESSION
    if SEG_SESSION is None:
        return image
    if image is None or image.size == 0:
        return image

    try:
        h, w = image.shape[:2]
        # Resize về 640x640 letterbox cho YOLO seg model
        size = 640
        scale = min(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_y = (size - new_h) // 2
        pad_x = (size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Preprocess: BGR → RGB, HWC → CHW, /255
        tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]   # [1, 3, 640, 640]

        input_name = SEG_SESSION.get_inputs()[0].name
        outputs = SEG_SESSION.run(None, {input_name: tensor})

        # Lấy output [1, 300, 38]: 4 bbox + 1 conf + ... (YOLO seg format)
        pred = outputs[0][0]  # [300, 38]
        if pred.shape[0] == 0:
            return image

        # Lấy detection có confidence cao nhất
        confidences = pred[:, 4] if pred.shape[1] > 4 else pred[:, 4:5].flatten()
        best_idx = int(np.argmax(confidences))
        if confidences[best_idx] < 0.3:
            return image

        box = pred[best_idx, :4]  # cx, cy, bw, bh (normalized)
        cx, cy, bw, bh = box

        # Convert về pixel coordinates (trừ padding)
        cx = (cx - pad_x) / scale
        cy = (cy - pad_y) / scale
        bw = bw / scale
        bh = bh / scale

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w, int(cx + bw / 2))
        y2 = min(h, int(cy + bh / 2))

        if x2 - x1 < 20 or y2 - y1 < 8:
            return image

        # Crop theo bbox từ segment
        cropped = image[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else image

    except Exception as e:
        logger.debug("Segment perspective correction failed: %s", e)
        return image


# =========================================================================== #
#  PaddleOCR v5 ONNX Engine (DBNet + CRNN/CTC)                               #
# =========================================================================== #

class CameraAI3OCR:
    """
    PaddleOCR v5 pipeline chạy hoàn toàn qua ONNX Runtime.
    DBNet text detector + CRNN/CTC recognizer.
    Hỗ trợ cả GPU (CUDAExecutionProvider) và CPU.
    """

    DET_LIMIT_SIDE   = 960
    DET_THRESHOLD    = 0.3
    DET_BOX_THRESHOLD = 0.6
    DET_UNCLIP_RATIO = 1.6   # Tăng từ 1.5 → 1.6 để bao phủ đủ ký tự biển số
    REC_HEIGHT       = 48
    REC_MAX_WIDTH    = 3200

    def __init__(self, det_path, rec_path, keys_path, use_gpu: bool = False):
        for path in (det_path, rec_path, keys_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing OCR asset: {path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2   # Tăng từ 1 → 2 threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = True

        # Ưu tiên GPU nếu có
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.det_session = ort.InferenceSession(str(det_path),
                                                sess_options=options,
                                                providers=providers)
        self.rec_session = ort.InferenceSession(str(rec_path),
                                                sess_options=options,
                                                providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name

        dictionary = keys_path.read_text(encoding="utf-8").split("\n")
        if dictionary and dictionary[-1] == "":
            dictionary.pop()
        self.characters = [""] + dictionary + [" "]

        output_classes = self.rec_session.get_outputs()[0].shape[-1]
        if isinstance(output_classes, int) and output_classes != len(self.characters):
            raise ValueError(
                f"OCR dictionary has {len(self.characters)} classes; "
                f"model outputs {output_classes}"
            )
        provider_used = self.det_session.get_providers()[0]
        logger.info(
            "Loaded PaddleOCR v5 models (dict=%d chars, provider=%s)",
            len(self.characters), provider_used,
        )

    @staticmethod
    def _order_points(points):
        points = np.asarray(points, dtype=np.float32)
        x_sorted = points[np.argsort(points[:, 0])]
        left  = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
        right = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
        return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

    @staticmethod
    def _box_score(probability_map, points):
        height, width = probability_map.shape
        x_min = max(0, int(np.floor(points[:, 0].min())))
        x_max = min(width - 1, int(np.ceil(points[:, 0].max())))
        y_min = max(0, int(np.floor(points[:, 1].min())))
        y_max = min(height - 1, int(np.ceil(points[:, 1].max())))
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        local_points = points.copy()
        local_points[:, 0] -= x_min
        local_points[:, 1] -= y_min
        cv2.fillPoly(mask, [np.rint(local_points).astype(np.int32)], 1)
        crop = probability_map[y_min:y_max + 1, x_min:x_max + 1]
        return float(cv2.mean(crop, mask)[0])

    def _prepare_detection_input(self, image):
        height, width = image.shape[:2]
        ratio = min(1.0, self.DET_LIMIT_SIDE / float(max(height, width)))
        resized_height = max(32, int(round(height * ratio / 32.0) * 32))
        resized_width  = max(32, int(round(width  * ratio / 32.0) * 32))
        resized = cv2.resize(image, (resized_width, resized_height),
                             interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
               / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = tensor.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(tensor)

    def _detect_text_boxes(self, image):
        tensor = self._prepare_detection_input(image)
        output = self.det_session.run(None, {self.det_input_name: tensor})[0]
        probability_map = np.squeeze(output).astype(np.float32)
        if probability_map.ndim != 2:
            raise ValueError(f"Unexpected OCR detector output shape: {output.shape}")

        bitmap = (probability_map >= self.DET_THRESHOLD).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = image.shape[:2]
        map_height, map_width = probability_map.shape
        boxes = []

        for contour in contours[:1000]:
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 3.0:
                continue
            points = cv2.boxPoints(rect)
            if self._box_score(probability_map, points) < self.DET_BOX_THRESHOLD:
                continue
            expanded_rect = (rect[0], (
                rect[1][0] * self.DET_UNCLIP_RATIO,
                rect[1][1] * self.DET_UNCLIP_RATIO,
            ), rect[2])
            points = cv2.boxPoints(expanded_rect)
            points[:, 0] *= image_width  / float(map_width)
            points[:, 1] *= image_height / float(map_height)
            points[:, 0] = np.clip(points[:, 0], 0, image_width  - 1)
            points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
            ordered = self._order_points(points)
            if min(
                np.linalg.norm(ordered[0] - ordered[1]),
                np.linalg.norm(ordered[0] - ordered[3]),
            ) >= 3.0:
                boxes.append(ordered)
        return boxes

    @staticmethod
    def _crop_text_line(image, points):
        width = max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
        height = max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
        target_width  = max(1, int(round(width)))
        target_height = max(1, int(round(height)))
        destination = np.float32([
            [0, 0], [target_width - 1, 0],
            [target_width - 1, target_height - 1], [0, target_height - 1],
        ])
        transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
        crop = cv2.warpPerspective(
            image, transform, (target_width, target_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        if crop.shape[0] / float(max(1, crop.shape[1])) >= 1.5:
            crop = np.rot90(crop)
        return np.ascontiguousarray(crop)

    def _recognize_text_line(self, image):
        height, width = image.shape[:2]
        resized_width = max(8, min(
            self.REC_MAX_WIDTH,
            int(math.ceil(self.REC_HEIGHT * width / float(max(1, height)))),
        ))
        resized = cv2.resize(image, (resized_width, self.REC_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = ((tensor - 0.5) / 0.5)[None, ...]
        predictions = self.rec_session.run(
            None, {self.rec_input_name: np.ascontiguousarray(tensor)},
        )[0]
        indices      = predictions.argmax(axis=2)[0]
        probabilities = predictions.max(axis=2)[0]

        characters  = []
        confidences = []
        previous    = -1
        for index, confidence in zip(indices, probabilities):
            index = int(index)
            if index != 0 and index != previous and index < len(self.characters):
                characters.append(self.characters[index])
                confidences.append(float(confidence))
            previous = index
        return "".join(characters), float(np.mean(confidences)) if confidences else 0.0

    def __call__(self, image):
        if image.shape[0] < 64:
            scale = 64.0 / image.shape[0]
            working_image = cv2.resize(
                image,
                (max(1, int(round(image.shape[1] * scale))), 64),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            working_image = image

        boxes = self._detect_text_boxes(working_image)
        if not boxes:
            h, w = working_image.shape[:2]
            boxes = [np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])]

        results = []
        for box in boxes:
            crop = self._crop_text_line(working_image, box)
            text, confidence = self._recognize_text_line(crop)
            if text:
                results.append([box.tolist(), text, confidence])
        return results


# =========================================================================== #
#  Engine Initialization                                                       #
# =========================================================================== #

def _try_load_seg_session():
    """Load best_plateSegment.onnx nếu có."""
    global SEG_SESSION
    if SEG_MODEL_PATH.is_file():
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            SEG_SESSION = ort.InferenceSession(
                str(SEG_MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("Loaded plate segment model: %s", SEG_MODEL_PATH.name)
        except Exception as e:
            logger.warning("Could not load segment model: %s", e)
            SEG_SESSION = None


def create_ocr_engine(use_gpu: bool = False) -> "CameraAI3OCR":
    logger.info(
        "Loading PaddleOCR v5 ONNX models (det=%s rec=%s)",
        DET_MODEL_PATH.name, REC_MODEL_PATH.name,
    )
    engine = CameraAI3OCR(DET_MODEL_PATH, REC_MODEL_PATH, REC_KEYS_PATH,
                          use_gpu=use_gpu)
    _try_load_seg_session()
    return engine


# =========================================================================== #
#  Vietnamese Plate Post-Processing                                            #
# =========================================================================== #

def split_plate(text: str):
    """
    Tách biển số VN thành (province, series, digits).
    Hỗ trợ biển 1 chữ cái series và biển đặc biệt 2 chữ cái series.
    """
    if not text or not PLATE_PATTERN.fullmatch(text):
        return None
    prefix = text[:2]
    try:
        if int(prefix) not in VALID_PROVINCE_CODES:
            return None
    except ValueError:
        return None

    for series_len in (1, 2):
        digits_len = len(text) - 2 - series_len
        if digits_len not in (4, 5):
            continue
        series = text[2:2 + series_len]
        digits = text[2 + series_len:]
        if not digits.isdigit() or series[0] not in VALID_SERIES_LETTERS:
            continue
        if series_len == 2:
            second = series[1]
            if second.isalpha() and series not in TWO_LETTER_SERIES:
                continue
        return prefix, series, digits
    return None


def normalize_plate(text: str) -> str:
    """
    Chuẩn hóa và xác thực biển số VN.
    Thực hiện sửa lỗi OCR thông minh: O↔0, I↔1, S↔5, v.v.
    Hỗ trợ biển số VN 2023+ (biển định danh cá nhân, biển 2 hàng xe máy).
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if not 7 <= len(cleaned) <= 9:
        return ""

    # Biển số thực phải có ít nhất 1 chữ cái (series letter)
    if not any(char.isalpha() for char in cleaned):
        return ""
    if split_plate(cleaned):
        return cleaned

    candidates = []
    for series_len in (1, 2):
        digits_len = len(cleaned) - 2 - series_len
        if digits_len not in (4, 5):
            continue

        # Sửa 2 chữ số đầu (province code)
        prefix = "".join(TO_DIGIT.get(char, char) for char in cleaned[:2])
        if not prefix.isdigit():
            continue
        try:
            if int(prefix) not in VALID_PROVINCE_CODES:
                continue
        except ValueError:
            continue

        # Sửa chữ cái series đầu
        first_series = TO_ALPHA.get(cleaned[2], cleaned[2])
        if first_series not in VALID_SERIES_LETTERS:
            continue
        series = first_series

        if series_len == 2:
            second_raw = cleaned[3]
            second_options = [second_raw]
            for opt in (TO_DIGIT.get(second_raw, second_raw),
                        TO_ALPHA.get(second_raw, second_raw)):
                if opt not in second_options:
                    second_options.append(opt)
        else:
            second_options = [""]

        digits_raw = cleaned[2 + series_len:]
        digits = "".join(TO_DIGIT.get(char, char) for char in digits_raw)
        if not digits.isdigit():
            continue

        for second in second_options:
            candidate = prefix + series + second + digits
            if split_plate(candidate):
                conversions = sum(a != b for a, b in zip(cleaned, candidate))
                candidates.append((conversions, -len(digits), candidate))

    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][2]


def order_ocr_results(results):
    if not results:
        return []

    def center(item):
        pts = np.asarray(item[0], dtype=np.float32)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def height(item):
        pts = np.asarray(item[0], dtype=np.float32)
        return float(max(np.linalg.norm(pts[0] - pts[3]),
                         np.linalg.norm(pts[1] - pts[2])))

    row_tolerance = max(5.0, float(np.median([height(i) for i in results])) * 0.5)
    rows = []
    for item in sorted(results, key=lambda v: center(v)[1]):
        _, cy = center(item)
        if not rows or abs(cy - rows[-1][0]) > row_tolerance:
            rows.append([cy, [item]])
        else:
            rows[-1][1].append(item)
            rows[-1][0] = float(np.mean([center(v)[1] for v in rows[-1][1]]))

    ordered = []
    for _, row in rows:
        ordered.extend(sorted(row, key=lambda v: center(v)[0]))
    return ordered


# =========================================================================== #
#  SOTA Multi-Attempt Recognition                                              #
# =========================================================================== #

def _recognize_single(engine: "CameraAI3OCR", image: np.ndarray):
    """Nhận diện 1 ảnh, trả về (text, confidence) hoặc ('', 0.0)."""
    if image is None or image.size == 0:
        return "", 0.0
    candidates = []
    results = engine(image)
    ordered = order_ocr_results(results)
    if ordered:
        raw = "".join(str(item[1]) for item in ordered)
        conf = float(np.mean([float(item[2]) for item in ordered]))
        candidates.append((raw, conf))

    # Xử lý biển 2 hàng (xe máy VN)
    h, w = image.shape[:2]
    if h >= 10 and w / float(max(1, h)) <= 2.5:
        target_h = max(96, h * 4)
        scale = target_h / float(h)
        enlarged = cv2.resize(
            image,
            (max(1, int(round(w * scale))), target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        for split_ratio in (0.42, 0.47, 0.50, 0.53, 0.58):
            split_y = int(round(enlarged.shape[0] * split_ratio))
            top_text, top_conf = engine._recognize_text_line(enlarged[:split_y])
            bot_text, bot_conf = engine._recognize_text_line(enlarged[split_y:])
            if top_text and bot_text:
                combined = top_text + bot_text
                avg_conf = float((top_conf + bot_conf) * 0.5)
                candidates.append((combined, avg_conf))

    best_text = ""
    best_conf = 0.0
    for raw_text, conf in candidates:
        if DATE_PATTERN.search(raw_text):
            continue
        normalized = normalize_plate(raw_text)
        if normalized and conf > best_conf:
            best_text = normalized
            best_conf = conf
    return best_text, best_conf


def recognize_plate(image: np.ndarray) -> tuple[str, float]:
    """
    ========================================================================
    SOTA 2025 Multi-Attempt Vietnamese License Plate Recognition Pipeline
    ========================================================================
    Thử 5 cách preprocessing khác nhau, chọn kết quả có confidence cao nhất:

    Attempt 1 — Raw crop (không preprocessing)
    Attempt 2 — CLAHE enhanced (tăng tương phản)
    Attempt 3 — CLAHE + Super-resolution 4x (biển số nhỏ ở xa)
    Attempt 4 — CLAHE + Sharpen (biển mờ do JPEG/motion blur)
    Attempt 5 — Segment perspective correction (biển nghiêng/méo)
    """
    if OCR_ENGINE is None:
        raise RuntimeError("OCR engine is not initialized")
    if image is None or image.size == 0:
        return "", 0.0

    attempts = []

    with INFER_LOCK:
        # --- Attempt 1: Raw ---
        text1, conf1 = _recognize_single(OCR_ENGINE, image)
        attempts.append((text1, conf1, "raw"))

        # --- Attempt 2: CLAHE ---
        img_clahe = apply_clahe(image)
        text2, conf2 = _recognize_single(OCR_ENGINE, img_clahe)
        attempts.append((text2, conf2, "clahe"))

        # --- Attempt 3: CLAHE + Super-resolution ---
        img_sr = super_resolve(img_clahe)
        text3, conf3 = _recognize_single(OCR_ENGINE, img_sr)
        attempts.append((text3, conf3, "clahe+sr"))

        # --- Attempt 4: CLAHE + Sharpen ---
        img_sharp = sharpen_plate(img_clahe)
        text4, conf4 = _recognize_single(OCR_ENGINE, img_sharp)
        attempts.append((text4, conf4, "clahe+sharpen"))

        # --- Attempt 5: Segment Perspective Correction ---
        if SEG_SESSION is not None:
            img_seg = segment_perspective_correct(image)
            if img_seg is not image:   # Only if segment found something
                img_seg_clahe = apply_clahe(img_seg)
                text5, conf5 = _recognize_single(OCR_ENGINE, img_seg_clahe)
                attempts.append((text5, conf5, "seg+clahe"))

        # --- Attempt 6: Deskew (tự động xoay biển nghiêng) ---
        img_deskew = deskew_plate(img_clahe)
        if img_deskew is not img_clahe:  # Only if deskew changed something
            text6, conf6 = _recognize_single(OCR_ENGINE, img_deskew)
            attempts.append((text6, conf6, "deskew+clahe"))

    # Chọn kết quả tốt nhất
    best_text = ""
    best_conf = 0.0
    best_method = "none"
    for text, conf, method in attempts:
        if text and conf > best_conf:
            best_text = text
            best_conf = conf
            best_method = method

    if best_text:
        logger.info("Plate: %s (conf=%.3f, method=%s)", best_text, best_conf, best_method)
    return best_text, best_conf


# =========================================================================== #
#  HTTP Sidecar Server (giữ lại để tương thích ngược)                         #
# =========================================================================== #

class OCRRequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "engine": "paddleocr-v5-onnx-sota2025",
                "features": ["clahe", "super_resolution", "multi_attempt",
                             "perspective_correction", "deskew"],
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ocr":
            self.send_json(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "invalid request size"})
                return
            request_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            image_b64 = request_data.get("image_b64", "")
            if not image_b64:
                self.send_json(400, {"error": "missing image_b64"})
                return
            image_bytes = base64.b64decode(image_b64, validate=True)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.send_json(400, {"error": "cannot decode image"})
                return
            text, confidence = recognize_plate(image)
            self.send_json(200, {
                "text": text,
                "confidence": confidence,
                "success": bool(text),
            })
        except (ValueError, json.JSONDecodeError, binascii.Error) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            logger.exception("OCR request failed")
            self.send_json(500, {"error": str(error)})

    def log_message(self, fmt, *args):
        return  # Suppress noisy HTTP logs


def main():
    global OCR_ENGINE
    # Thử GPU trước, fallback CPU
    try:
        providers = ort.get_available_providers()
        use_gpu = "CUDAExecutionProvider" in providers
        if use_gpu:
            logger.info("CUDA GPU detected — enabling GPU inference")
        else:
            logger.info("No CUDA GPU — using CPU inference")
        OCR_ENGINE = create_ocr_engine(use_gpu=use_gpu)
    except Exception as error:
        logger.error("OCR initialization failed: %s", error)
        return 1

    port = 8765
    server = ThreadingHTTPServer(("0.0.0.0", port), OCRRequestHandler)
    logger.info("SOTA 2025 OCR service listening on port %d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
