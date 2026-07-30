#!/usr/bin/env python3
"""
Digital Meter Reading Pipeline
================================
Enterprise water meter recognition: OBB/REG detection -> crop extraction -> OCR recognition -> post-processing.
Best accuracy: 65.2% (30/46) on enterprise test set.

Pipeline stages:
  1. Phase 1 (standard): REG + OBB detection at imgsz=640, multi-angle (0/90/180/270)
  2. OBB angle fallback: additional OBB detection for 90/180/270 rotations
  3. Phase 2 (fallback): Multi-scale detection (960/1280) + CLAHE enhancement for hard cases
  4. OCR scoring: Multi-candidate OCR with rotation/flip augmentation and confidence scoring
  5. Post-processing: pp_v2 - "88" prefix removal, period-separated format conversion, decimal placement

Models:
  - REG detection: YOLOv8s (delivery_best.pt)
  - OBB detection: YOLOv8n-OBB (yolov8n_obb_wordwheel.pt)
  - Recognition: OpenOCR SVTRv2 (generic Chinese OCR)

Author: AI Research Team
Date: 2026-07
"""

import os
import sys
import cv2
import re
import csv
import json
import time
import yaml
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ============================================================
# Image Utilities
# ============================================================

def rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return img
    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(img, rotation_map[angle])


def rotate_box(box: np.ndarray, angle: int, img_w: int, img_h: int) -> List[float]:
    """Rotate a REG bounding box [x1,y1,x2,y2] back to original coordinates."""
    x1, y1, x2, y2 = box
    if angle == 0:
        return [x1, y1, x2, y2]
    elif angle == 90:
        return [y1, img_h - x2, y2, img_h - x1]
    elif angle == 180:
        return [img_w - x2, img_h - y2, img_w - x1, img_h - y1]
    elif angle == 270:
        return [img_h - y2, x1, img_h - y1, x2]
    return [x1, y1, x2, y2]


def trim_88(crop: np.ndarray) -> np.ndarray:
    """
    Remove printed "88" prefix from crop by finding the first content column.
    Uses column brightness analysis to identify the start of actual digits.
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    col_means = gray.mean(axis=0)
    thresh = max(80, col_means.max() * 0.6)
    start = 0
    for i in range(min(200, w)):
        if col_means[i] > thresh:
            start = max(0, i - 5)
            break
    if start > 10:
        crop = crop[:, start:]
    return crop


def extract_obb(img: np.ndarray, corners: np.ndarray, angle_rad: float,
                expand: float = 1.15, right_shift: float = 0.0) -> Optional[np.ndarray]:
    """
    Extract OBB crop using rotation transform.

    Args:
        img: Source image
        corners: 4 corner points of OBB
        angle_rad: Rotation angle in radians
        expand: Expansion factor for crop region (default 1.15)
        right_shift: Optional rightward shift (fraction of box width)

    Returns:
        Cropped and rotated image, or None if invalid
    """
    pts = np.array(corners, dtype=np.float32)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    if right_shift > 0:
        shift_px = right_shift * np.linalg.norm(pts[1] - pts[0])
        cx += np.cos(angle_rad) * shift_px
        cy += np.sin(angle_rad) * shift_px

    wb = float(np.linalg.norm(pts[1] - pts[0]))
    hb = float(np.linalg.norm(pts[3] - pts[0]))
    hi, wi = img.shape[:2]
    ad = float(np.degrees(angle_rad))

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ad, 1.0)
    ca, sa = abs(M[0, 0]), abs(M[0, 1])
    nw = int(hi * sa + wi * ca)
    nh = int(hi * ca + wi * sa)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    rot = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC)

    cxn = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    cyn = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]

    if hb > wb:
        cw, ch = int(hb * expand), int(wb * expand)
    else:
        cw, ch = int(wb * expand), int(hb * expand)

    x1 = max(0, int(cxn - cw / 2))
    y1 = max(0, int(cyn - ch / 2))
    x2 = min(nw, int(cxn + cw / 2))
    y2 = min(nh, int(cyn + ch / 2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rot[y1:y2, x1:x2]
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


# ============================================================
# OCR and Post-processing
# ============================================================

def occp(crop: np.ndarray, ocr_model) -> Tuple[str, float]:
    """
    Run OCR with augmentation: tries RGB + inverted (if dark) x 0/180 flip.
    Returns the best (text, confidence) based on scoring.
    """
    if crop is None or crop.size == 0:
        return "", 0

    best_text = ""
    best_conf = 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()

    # Try RGB, and inverted if image is dark
    methods = [(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), "rgb")]
    if mean_val < 128:
        methods.append((cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR), "inv"))

    for proc_img, method_name in methods:
        for aug_img, aug_name in [(proc_img, ""),
                                  (cv2.rotate(proc_img, cv2.ROTATE_180), "_flip")]:
            try:
                result = ocr_model(img_numpy=aug_img)
                text = result[0].get("text", "") if result and len(result) > 0 else ""
                conf = result[0].get("score", 0) if result and len(result) > 0 else 0
            except Exception:
                continue

            digits = re.sub(r'[^0-9]', '', text)
            if len(digits) < 2:
                continue

            # Scoring: confidence + digit length bonus + period bonus - flip penalty
            ns = (2 if "." in text else 0) + (1 if 2 < len(text) < 8 else 0)
            fp = 0.2 if aug_name == "_flip" else 0
            score = conf + len(digits) / 20 - fp + ns * 0.5

            if score > best_conf:
                best_text = text
                best_conf = score

    return best_text, best_conf


def pp_v2(text: str) -> str:
    """
    Advanced post-processing for OCR output.

    Features:
    - Handles period-separated format ("08.36.6" -> "36.6")
    - Removes printed prefixes ("88", "08", "68", "98", "78", "58")
    - Proper decimal point placement based on digit count
    - Handles various OCR artifacts

    Args:
        text: Raw OCR output text

    Returns:
        Cleaned reading string with proper decimal placement
    """
    if not text:
        return ""

    sign = ""
    t = text
    if t.startswith("-"):
        sign = "-"
        t = t[1:]

    parts = t.split(".")

    # Handle period-separated format (e.g., "08.36.6")
    if len(parts) >= 3 and all(len(p) <= 2 for p in parts) and all(p.isdigit() or p == "" for p in parts):
        joined = "".join(p for p in parts if p)
        if not joined:
            return ""
        # Remove printed "88" prefix
        if len(joined) >= 4 and joined[:2] in ("88", "08", "68", "98", "78", "58"):
            joined = joined[2:]
        if len(joined) >= 4 and joined[0] == "8":
            joined = joined[1:]
        if not joined:
            return ""

        n = len(joined)
        # Format: XXX.YYY for n>=5, XX.YY for n=4, XY.Z for n=3
        if n >= 5:
            return sign + (joined[:n-3].lstrip("0") or "0") + "." + joined[n-3:]
        elif n == 4:
            return sign + joined[:2] + "." + joined[2:]
        elif n == 3:
            return sign + joined[:2] + "." + joined[2:]
        return sign + joined

    # Standard format processing
    digits = re.sub(r'[^0-9]', '', t)
    if not digits:
        return ""

    if digits.startswith("88") and len(digits) > 4 and not sign:
        digits = digits[2:]
    if sign == "" and digits.startswith("8") and len(digits) >= 4:
        digits = digits[1:]

    # Handle "0.8..." format (common OCR error)
    if "." in t and t.startswith("0.") and len(t) >= 4 and t[2] == "8":
        stripped = t[2:]
        digits2 = re.sub(r'[^0-9]', '', stripped)
        if digits2.startswith("8") and len(digits2) >= 3:
            digits = digits2[1:]
            return sign + digits

    # Handle "51" prefix (common for certain meter types)
    if len(digits) == 4 and digits.startswith("51") and sign == "":
        digits = "00" + digits[2:]

    if "." in t:
        p = t.split(".")
        if len(p) == 2 and p[0]:
            return sign + (p[0].lstrip("0") or "0") + "." + p[1]

    n = len(digits)
    if n >= 4 and digits[0] == "0":
        return sign + "0." + digits[1:]
    if n >= 5:
        return sign + (digits[:n-3].lstrip("0") or "0") + "." + digits[n-3:]

    return sign + digits


def compute_edit_distance(gt_text: str, pred_text: str) -> int:
    """
    Compute Levenshtein edit distance between ground truth and prediction.
    Both strings are stripped of non-digit characters for comparison.
    """
    if not pred_text or not gt_text:
        return 99
    gs = re.sub(r'[^0-9]', '', gt_text).lstrip("0") or "0"
    ps = re.sub(r'[^0-9]', '', pred_text).lstrip("0") or "0"
    m, n = len(gs), len(ps)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if i <= len(gs) and j <= len(ps) and gs[i-1] == ps[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    return dp[m][n]


# ============================================================
# Detection Pipeline
# ============================================================

class MeterReadingPipeline:
    """
    Main pipeline for water meter reading recognition.
    Combines REG detection, OBB detection, and OCR recognition.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with models from config."""
        cfg = load_config(config_path)

        # Detection models
        det_cfg = cfg["detection"]
        from ultralytics import YOLO
        self.det_reg = YOLO(det_cfg["reg_model"])
        self.det_obb = YOLO(det_cfg["obb_model"])

        # Recognition model
        from openocr import OpenOCR
        self.ocr = OpenOCR(
            task="rec",
            mode="server",
            backend="torch",
            use_gpu="auto" if cfg["device"] == "auto" else cfg["device"]
        )

        # Configuration
        self.conf_thresholds = det_cfg.get("conf_thresholds", [0.1, 0.05, 0.01])
        self.low_conf_thresholds = det_cfg.get("low_conf_thresholds", [0.01, 0.005, 0.001])

        # CLAHE for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process_image_phase1(self, img: np.ndarray) -> List[tuple]:
        """
        Phase 1: Standard detection at imgsz=640 with multi-angle.
        Uses REG detector first (4 angles), then OBB detector (4 angles).
        """
        h, w = img.shape[:2]
        candidates = []

        # --- REG detection (0/90/180/270) ---
        for angle in [0, 90, 180, 270]:
            rotated = rotate_img(img, angle)
            for conf_th in self.conf_thresholds:
                result = self.det_reg(rotated, imgsz=640, conf=conf_th, verbose=False)[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for bi in range(len(result.boxes)):
                    conf = float(result.boxes.conf[bi])
                    box = rotate_box(result.boxes.xyxy[bi].cpu().numpy(), angle, w, h)
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw, bh = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ex1 = max(0, int(cx - bw / 2))
                    ey1 = max(0, int(cy - bh / 2))
                    ex2 = min(w, int(cx + bw / 2))
                    ey2 = min(h, int(cy + bh / 2))
                    crop = img[ey1:ey2, ex1:ex2]
                    if crop is not None and crop.size > 0:
                        crop_trimmed = trim_88(crop)
                        if crop_trimmed is not None and crop_trimmed.size > 0:
                            candidates.append(("REG", conf, 1.0, crop_trimmed))

        # --- OBB detection (0/90/180/270) ---
        for angle in [0, 90, 180, 270]:
            rotated = rotate_img(img, angle)
            for conf_th in self.conf_thresholds:
                result = self.det_obb(rotated, imgsz=640, conf=conf_th, verbose=False)[0]
                if result.obb is None or len(result.obb) == 0:
                    continue
                confs = result.obb.conf.cpu().numpy()
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                xywhr = result.obb.xywhr.cpu().numpy()
                for bi in range(len(confs)):
                    conf = float(confs[bi])
                    corners = boxes[bi]
                    if corners.ndim == 1:
                        corners = corners.reshape(4, 2)
                    angle_rad = float(xywhr[bi][4])
                    pts = np.array(corners, dtype=np.float32)

                    # Rotate back corners to original image coordinates
                    if angle == 0:
                        rc = pts
                    elif angle == 90:
                        rc = np.column_stack([pts[:, 1], h - 1 - pts[:, 0]])
                    elif angle == 180:
                        rc = np.column_stack([w - 1 - pts[:, 0], h - 1 - pts[:, 1]])
                    elif angle == 270:
                        rc = np.column_stack([h - 1 - pts[:, 1], pts[:, 0]])

                    crop = extract_obb(img, rc, angle_rad, expand=1.15)
                    if crop is not None and crop.size > 0:
                        candidates.append(("OBB", conf, 1.15, crop))
                    crop2 = extract_obb(img, rc, angle_rad, expand=1.25)
                    if crop2 is not None and crop2.size > 0:
                        candidates.append(("OBB_X", conf, 1.25, crop2))

        return candidates

    def process_image_phase2(self, img: np.ndarray) -> List[tuple]:
        """
        Phase 2: Multi-scale fallback detection for hard cases.
        Tries larger input sizes (960/1280) and CLAHE enhancement.
        """
        h, w = img.shape[:2]
        candidates = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- REG at larger scales ---
        for imgsz in [960, 1280]:
            for angle in [0, 180]:
                rotated = rotate_img(img, angle)
                for conf_th in self.low_conf_thresholds[:2]:
                    result = self.det_reg(rotated, imgsz=imgsz, conf=conf_th, verbose=False)[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    for bi in range(len(result.boxes)):
                        conf = float(result.boxes.conf[bi])
                        box = rotate_box(result.boxes.xyxy[bi].cpu().numpy(), angle, w, h)
                        x1, y1, x2, y2 = [int(v) for v in box]
                        bw, bh = x2 - x1, y2 - y1
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        ex1 = max(0, int(cx - bw / 2))
                        ey1 = max(0, int(cy - bh / 2))
                        ex2 = min(w, int(cx + bw / 2))
                        ey2 = min(h, int(cy + bh / 2))
                        crop = img[ey1:ey2, ex1:ex2]
                        if crop is not None and crop.size > 0:
                            crop_trimmed = trim_88(crop)
                            if crop_trimmed is not None and crop_trimmed.size > 0:
                                candidates.append(("REG2", conf, 1.0, crop_trimmed))

        # --- CLAHE-enhanced REG ---
        enhanced = cv2.cvtColor(self.clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        for imgsz in [640, 1280]:
            for angle in [0, 90]:
                rotated = rotate_img(enhanced, angle)
                for conf_th in self.low_conf_thresholds[:2]:
                    result = self.det_reg(rotated, imgsz=imgsz, conf=conf_th, verbose=False)[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    for bi in range(len(result.boxes)):
                        conf = float(result.boxes.conf[bi])
                        box = rotate_box(result.boxes.xyxy[bi].cpu().numpy(), angle, w, h)
                        x1, y1, x2, y2 = [int(v) for v in box]
                        bw, bh = x2 - x1, y2 - y1
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        ex1 = max(0, int(cx - bw / 2))
                        ey1 = max(0, int(cy - bh / 2))
                        ex2 = min(w, int(cx + bw / 2))
                        ey2 = min(h, int(cy + bh / 2))
                        crop = img[ey1:ey2, ex1:ex2]
                        if crop is not None and crop.size > 0:
                            crop_trimmed = trim_88(crop)
                            if crop_trimmed is not None and crop_trimmed.size > 0:
                                candidates.append(("REG_C2", conf, 1.0, crop_trimmed))

        # --- OBB at 1280 ---
        for angle in [0, 180]:
            rotated = rotate_img(img, angle)
            for conf_th in self.low_conf_thresholds:
                result = self.det_obb(rotated, imgsz=1280, conf=conf_th, verbose=False)[0]
                if result.obb is None or len(result.obb) == 0:
                    continue
                confs = result.obb.conf.cpu().numpy()
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                xywhr = result.obb.xywhr.cpu().numpy()
                for bi in range(len(confs)):
                    conf = float(confs[bi])
                    corners = boxes[bi]
                    if corners.ndim == 1:
                        corners = corners.reshape(4, 2)
                    angle_rad = float(xywhr[bi][4])
                    pts = np.array(corners, dtype=np.float32)
                    rc = pts if angle == 0 else np.column_stack([h - 1 - pts[:, 1], pts[:, 0]])
                    crop = extract_obb(img, rc, angle_rad, expand=1.25)
                    if crop is not None and crop.size > 0:
                        candidates.append(("OBB2", conf, 1.25, crop))

        return candidates

    def score_candidates(self, candidates: List[tuple]) -> List[tuple]:
        """
        Run OCR on all detection candidates and score them.
        Returns sorted list of (score, det_type, reading) tuples.
        """
        scored = []
        for det_type, conf, expand, crop in candidates:
            text, ocr_conf = occp(crop, self.ocr)
            if not text:
                continue

            reading = pp_v2(text)

            # Filter out implausible zero readings
            if reading in ["00", "0.00", "000", "0.0", "0"]:
                if ocr_conf < 0.8:
                    continue
                if crop.shape[0] * crop.shape[1] > 5000:
                    continue

            # Score: OCR confidence + digit length bonus - expansion penalty
            expand_penalty = abs(expand - 1.15) * 0.15
            score = ocr_conf + len(re.sub(r'[^0-9]', '', reading)) / 20 - expand_penalty
            scored.append((score, det_type, reading))

        return scored

    def run_on_image(self, img: np.ndarray) -> Tuple[str, str, bool, int]:
        """
        Run full pipeline on a single image.

        Returns:
            (prediction_text, detection_type, is_match, edit_distance)
        """
        # Phase 1: Standard detection
        candidates = self.process_image_phase1(img)
        scored = self.score_candidates(candidates)

        # OBB angle fallback for 90/180/270 rotations
        if len(scored) == 0:
            h, w = img.shape[:2]
            for angle in [90, 180, 270]:
                rotated = rotate_img(img, angle)
                for conf_th in self.conf_thresholds:
                    result = self.det_obb(rotated, imgsz=640, conf=conf_th, verbose=False)[0]
                    if result.obb is None or len(result.obb) == 0:
                        continue
                    confs = result.obb.conf.cpu().numpy()
                    boxes = result.obb.xyxyxyxy.cpu().numpy()
                    xywhr = result.obb.xywhr.cpu().numpy()
                    for bi in range(len(confs)):
                        conf = float(confs[bi])
                        corners = boxes[bi]
                        if corners.ndim == 1:
                            corners = corners.reshape(4, 2)
                        angle_rad = float(xywhr[bi][4])
                        pts = np.array(corners, dtype=np.float32)
                        if angle == 90:
                            rc = np.column_stack([pts[:, 1], h - 1 - pts[:, 0]])
                        elif angle == 180:
                            rc = np.column_stack([w - 1 - pts[:, 0], h - 1 - pts[:, 1]])
                        elif angle == 270:
                            rc = np.column_stack([h - 1 - pts[:, 1], pts[:, 0]])
                        adj_angle = angle_rad + np.radians(angle)
                        crop = extract_obb(img, rc, adj_angle, expand=1.15)
                        if crop is not None and crop.size > 0:
                            text, ocr_conf = occp(crop, self.ocr)
                            if text:
                                reading = pp_v2(text)
                                scored.append((ocr_conf - 0.1, "OBBa", reading))

        # Phase 2: Multi-scale fallback
        p2_used = False
        if len(scored) == 0:
            candidates2 = self.process_image_phase2(img)
            scored2 = self.score_candidates(candidates2)
            if scored2:
                scored = scored2
                p2_used = True

        # Select best candidate
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) > 0:
            _, det_type, prediction = scored[0]
            if p2_used:
                det_type = det_type + "_P2"
        else:
            prediction = ""
            det_type = "NONE"

        return prediction, det_type

    @staticmethod
    def compute_metrics(gt_raw: str, prediction: str) -> Tuple[int, bool]:
        """Compute edit distance and match status."""
        gt_clean = re.sub(r'[^0-9.\-]', '', gt_raw) if gt_raw else ""
        try:
            gtv = float(gt_clean) if gt_clean else None
        except ValueError:
            gtv = None

        pred_clean = re.sub(r'[^0-9]', '', prediction) if prediction else ""
        if not pred_clean or gtv is None:
            return 99, False

        ed = compute_edit_distance(gt_clean, pred_clean)
        return ed, ed == 0


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Run pipeline on enterprise test dataset."""
    cfg = load_config()
    data_dir = cfg["data"]["data_dir"]
    output_dir = cfg["data"]["output_dir"]
    labels_file = cfg["data"]["labels_file"]

    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    gt_map = {}
    labels_path = os.path.join(data_dir, labels_file)
    with open(labels_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_map[row["filename"]] = row

    # Filter test images (clear images only)
    clear_label = cfg["data"].get("clear_label", "是")
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith((".jpg", ".png"))
        and f in gt_map
        and gt_map[f].get("human_clear", "") == clear_label
    ])

    if len(files) == 0:
        print("ERROR: No test images found!")
        sys.exit(1)

    print(f"Testing {len(files)} images...\n")

    # Initialize pipeline
    pipeline = MeterReadingPipeline(config_path)

    results = []
    t0 = time.time()

    for idx, fname in enumerate(files):
        img_path = os.path.join(data_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{idx+1}/{len(files)}] SKIP: Cannot read {fname}")
            continue

        gt_raw = gt_map[fname]["gt_reading"]
        prediction, det_type = pipeline.run_on_image(img)
        ed, match = MeterReadingPipeline.compute_metrics(gt_raw, prediction)

        results.append({
            "file": fname,
            "gt": gt_raw,
            "pred": prediction,
            "ed": ed,
            "match": match,
            "det_type": det_type,
        })

        status = "OK" if match else "XX"
        print(f"  [{idx+1}/{len(files)}] {det_type:10s} GT={gt_raw:>8s} "
              f"Pred={prediction:>20s} ED={ed:2d} {status}")

    elapsed = time.time() - t0
    total = len(results)
    ed0 = sum(1 for r in results if r["match"])
    ed1 = sum(1 for r in results if r["ed"] <= 1)

    print(f"\n=== Pipeline Complete ({elapsed:.0f}s) ===")
    print(f"Total: {total} | ED=0: {ed0}/{total} = {ed0/total*100:.1f}% "
          f"| ED<=1: {ed1}/{total} = {ed1/total*100:.1f}%")

    for r in results:
        if not r["match"]:
            print(f"  ERR {r['file'][:25]} GT={r['gt']:>8s} "
                  f"Pred={r['pred']:>20s} ED={r['ed']:2d} {r['det_type']}")

    # Save results
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    print("DONE!")


if __name__ == "__main__":
    main()
