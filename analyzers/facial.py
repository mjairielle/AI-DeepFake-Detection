"""
Deepfake Detection — Facial Analysis Module
Analyzes face regions for deepfake artifacts (skin smoothness, blending,
color mismatch, noise ratio, symmetry).
"""
import logging
from typing import Tuple, List

import numpy as np
import cv2

from models import FacialAnalysisParams
from analyzers.helpers import _safe_div

logger = logging.getLogger("deepfake_engine")


class FacialAnalyzer:
    """Analyzes face regions for deepfake artifacts."""

    def __init__(self, params: FacialAnalysisParams):
        self.p = params
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ---- public ----
    def analyze_image(self, bgr: np.ndarray) -> Tuple[float, List[str]]:
        flags = []
        scores = []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            return 0.0, ["no_face_detected"]

        for (x, y, w, h) in faces:
            roi = bgr[y:y+h, x:x+w]
            
            # Skin color heuristic to filter out false positive faces (like game textures)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Standard skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)
            skin_ratio = cv2.countNonZero(skin_mask) / (w * h)
            
            # If less than 5% of the bounding box is skin colored, it's probably a hallucinated face
            if skin_ratio < 0.05:
                continue

            s, f = self._analyze_face_roi(roi, bgr, x, y, w, h)
            scores.append(s)
            flags.extend(f)

        if not scores:
            return 0.0, ["no_face_detected"]

        return float(np.mean(scores)), flags

    def analyze_frames(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        if not frames:
            return 0.0, ["no_frames"]
        per_frame = [self.analyze_image(f) for f in frames]
        avg = float(np.mean([s for s, _ in per_frame]))
        all_flags = []
        for _, f in per_frame:
            all_flags.extend(f)
        return avg, list(set(all_flags))

    # ---- internals ----
    def _analyze_face_roi(self, roi, full, x, y, w, h):
        flags = []
        scores = []

        # Skin smoothness (over-smoothed = GAN)
        lap = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        variance = lap.var()
        smooth_score = 1.0 - min(variance / 500.0, 1.0)
        if smooth_score > self.p.skin_smoothness_anomaly_threshold:
            flags.append("skin_over_smoothed")
        scores.append(smooth_score)

        # Blending boundary check
        blend_score = self._check_blending(full, x, y, w, h)
        if blend_score > self.p.blending_gradient_threshold:
            flags.append("blending_boundary_detected")
        scores.append(blend_score)

        # Color mismatch face vs background
        cm = self._color_mismatch(roi, full, x, y, w, h)
        cm_norm = min(cm / 50.0, 1.0)
        if cm > self.p.color_mismatch_tolerance:
            flags.append("face_bg_color_mismatch")
        scores.append(cm_norm)

        # Noise ratio face vs background
        nr = self._noise_ratio(roi, full, x, y, w, h)
        if nr > self.p.face_bg_noise_ratio_threshold:
            flags.append("face_bg_noise_mismatch")
        scores.append(min(nr / 3.0, 1.0))

        # Symmetry check
        sym = self._symmetry_score(roi)
        if sym > self.p.landmark_symmetry_tolerance:
            flags.append("facial_asymmetry_anomaly")
        scores.append(min(sym / 0.5, 1.0))

        return float(np.mean(scores)), flags

    def _check_blending(self, img, x, y, w, h):
        ew = self.p.blending_edge_width_px
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, ew)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        border_pixels = edges[mask == 255]
        return float(np.mean(np.abs(border_pixels)) / 255.0) if len(border_pixels) else 0.0

    def _color_mismatch(self, roi, full, x, y, w, h):
        face_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(float)
        bg = full.copy()
        bg[y:y+h, x:x+w] = 0
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2Lab).astype(float)
        bg_mask = bg_lab.sum(axis=2) > 0
        if not bg_mask.any():
            return 0.0
        face_mean = face_lab.mean(axis=(0, 1))
        bg_mean = bg_lab[bg_mask].mean(axis=0)
        return float(np.linalg.norm(face_mean - bg_mean))

    def _noise_ratio(self, roi, full, x, y, w, h):
        face_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(float)
        face_noise = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        bg = full.copy()
        bg[y:y+h, x:x+w] = 0
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(float)
        bg_noise = cv2.Laplacian(bg_gray, cv2.CV_64F).var()
        return _safe_div(face_noise, bg_noise, 1.0)

    def _symmetry_score(self, roi):
        h, w = roi.shape[:2]
        left = roi[:, :w//2]
        right = cv2.flip(roi[:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        left, right = left[:, :min_w], right[:, :min_w]
        diff = np.abs(left.astype(float) - right.astype(float))
        return float(diff.mean() / 255.0)
