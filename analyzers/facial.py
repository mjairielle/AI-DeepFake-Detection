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

            # Fallback for monochrome / black & white portraits:
            # Faces in B&W or heavily desaturated images will have low saturation (S < 30).
            # Verify luminance is in typical facial range (V between 40 and 230).
            if skin_ratio < 0.05 and np.mean(hsv[:, :, 1]) < 30:
                desat_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 40) & (hsv[:, :, 2] < 230)
                desat_ratio = np.count_nonzero(desat_mask) / float(w * h)
                if desat_ratio > 0.30:
                    skin_ratio = desat_ratio

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
        smooth_score = 1.0 - min(variance / 350.0, 1.0)
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
        cm_norm = min(max(0.0, (cm - 15.0) / 35.0), 1.0)
        effective_cm_tol = max(self.p.color_mismatch_tolerance, 26.0)
        if cm > effective_cm_tol:
            flags.append("face_bg_color_mismatch")
        scores.append(cm_norm)

        # Noise ratio face vs background
        nr = self._noise_ratio(roi, full, x, y, w, h)
        if nr > self.p.face_bg_noise_ratio_threshold:
            flags.append("face_bg_noise_mismatch")
        scores.append(min(nr / 3.0, 1.0))

        # Symmetry check
        sym = self._symmetry_score(roi)
        effective_sym_tol = max(self.p.landmark_symmetry_tolerance, 0.24)
        if sym > effective_sym_tol:
            flags.append("facial_asymmetry_anomaly")
        scores.append(min(max(0.0, (sym - 0.15) / 0.35), 1.0))

        # Error Level Analysis (ELA) — detects splicing / manipulation
        ela_score = self._ela_analysis(roi)
        if ela_score > 0.55:
            flags.append("ela_manipulation_detected")
        scores.append(ela_score)

        # Pore / texture detail score (multi-scale frequency analysis)
        pore_score = self._pore_visibility(roi)
        if pore_score < self.p.pore_visibility_min_score:
            flags.append("low_skin_texture_detail")
        scores.append(1.0 - pore_score)  # higher = more suspicious (less pore detail)

        # Teeth detail analysis
        teeth_score = self._teeth_detail(roi)
        if teeth_score < self.p.teeth_detail_score_min:
            flags.append("teeth_detail_anomaly")
        scores.append(1.0 - teeth_score)

        top_scores = sorted(scores, reverse=True)
        max_score = top_scores[0]
        mean_score = float(np.mean(scores))
        if max_score > 0.4:
            return float(0.6 * max_score + 0.4 * mean_score), flags
        return mean_score, flags

    def _check_blending(self, img, x, y, w, h):
        ew = self.p.blending_edge_width_px
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, ew)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        border_pixels = edges[mask == 255]
        return float(np.mean(np.abs(border_pixels)) / 255.0) if len(border_pixels) else 0.0

    def _color_mismatch(self, roi, full, x, y, w, h):
        H, W = full.shape[:2]
        bg = full.copy()
        bg[y:y+h, x:x+w] = 0
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        mask = bg_gray > 0
        if not mask.any():
            return 0.0
        bg_pixels = bg_gray[mask]
        high_lum_ratio = np.count_nonzero(bg_pixels > 225) / float(len(bg_pixels))
        is_studio = (high_lum_ratio > 0.35) or (float(np.std(bg_pixels)) < 25.0)

        if is_studio:
            # On a studio backdrop, comparing face to white wall is invalid.
            # Check adjacent neck/chest skin below chin for donor mismatch instead.
            neck_y1 = min(y + h, H - 1)
            neck_y2 = min(y + int(h * 1.5), H)
            neck_x1 = max(x + int(w * 0.2), 0)
            neck_x2 = min(x + int(w * 0.8), W)

            if neck_y2 > neck_y1 and neck_x2 > neck_x1:
                neck_roi = full[neck_y1:neck_y2, neck_x1:neck_x2]
                hsv_neck = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                neck_skin_mask = cv2.inRange(hsv_neck, lower_skin, upper_skin)
                skin_count = np.count_nonzero(neck_skin_mask)
                total_neck = neck_roi.shape[0] * neck_roi.shape[1]

                if skin_count > (total_neck * 0.20) and skin_count > 100:
                    face_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(float)
                    neck_lab = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2Lab).astype(float)
                    face_mean = face_lab.mean(axis=(0, 1))
                    neck_mean = neck_lab[neck_skin_mask > 0].mean(axis=0)
                    return float(np.linalg.norm(face_mean - neck_mean))

            return 0.0

        face_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(float)
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

        if bg_noise < 15.0 or face_noise < 15.0:
            return 1.0

        return _safe_div(face_noise, bg_noise, 1.0)

    def _symmetry_score(self, roi):
        h, w = roi.shape[:2]
        left = roi[:, :w//2]
        right = cv2.flip(roi[:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        left, right = left[:, :min_w], right[:, :min_w]
        diff = np.abs(left.astype(float) - right.astype(float))
        return float(diff.mean() / 255.0)

    def _ela_analysis(self, roi: np.ndarray, quality: int = 90) -> float:
        """
        Error Level Analysis (ELA).
        Re-compresses ROI at specified quality and checks error distribution.
        Manipulated, spliced, or synthetic regions typically display abnormal error variance.
        """
        if roi.size == 0 or min(roi.shape[:2]) < 8:
            return 0.0
        success, encoded = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return 0.0
        resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if resaved is None:
            return 0.0
        diff = cv2.absdiff(roi, resaved).astype(np.float64)
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        ela_metric = min((diff_mean * 2.0 + diff_std * 3.0) / 40.0, 1.0)
        return float(np.clip(ela_metric, 0.0, 1.0))

    def _pore_visibility(self, roi: np.ndarray) -> float:
        """
        Multi-scale frequency analysis for natural skin micro-texture / pore visibility.
        Natural skin has fine micro-texture in high frequencies; GANs smooth or hallucinate.
        Returns a score in [0.0, 1.0], where higher = more natural micro-texture detected.
        """
        if roi.size == 0 or min(roi.shape[:2]) < 16:
            return 0.5
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Difference of Gaussians (DoG) for high-frequency micro-texture band
        g1 = cv2.GaussianBlur(gray, (3, 3), 0.5)
        g2 = cv2.GaussianBlur(gray, (7, 7), 1.5)
        dog = cv2.absdiff(g1, g2).astype(np.float64)
        texture_energy = np.var(dog)
        score = min(texture_energy / 35.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _teeth_detail(self, roi: np.ndarray) -> float:
        """
        Analyzes teeth detail in the mouth region.
        Deepfakes frequently produce merged, blurred, or textureless teeth.
        Returns score in [0.0, 1.0], where higher = well-defined teeth or neutral if mouth closed.
        """
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return 0.85
        mouth_y1, mouth_y2 = int(h * 0.65), int(h * 0.95)
        mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
        mouth = roi[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        if mouth.size == 0:
            return 0.85

        gray_mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        teeth_mask = gray_mouth > 165
        teeth_pixel_count = np.count_nonzero(teeth_mask)
        mouth_area = mouth.shape[0] * mouth.shape[1]
        teeth_ratio = teeth_pixel_count / max(mouth_area, 1)

        if teeth_ratio < 0.03:
            return 0.85

        sobel_x = cv2.Sobel(gray_mouth, cv2.CV_64F, 1, 0, ksize=3)
        teeth_edges = np.abs(sobel_x)[teeth_mask]
        if len(teeth_edges) == 0:
            return 0.85
        edge_strength = float(np.mean(teeth_edges))
        detail_score = min(edge_strength / 20.0, 1.0)
        return float(np.clip(detail_score, 0.0, 1.0))

