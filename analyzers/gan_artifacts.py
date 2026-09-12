"""
Deepfake Detection — GAN / AI Artifact Detection Module
Detects GAN-generated artifacts via FFT grid analysis, high-frequency
energy ratios, color entropy, checkerboard patterns, noise uniformity,
and edge sharpness.
"""
import logging
from typing import Tuple, List

import numpy as np
import cv2

from models import GANArtifactParams
from analyzers.helpers import _safe_div

logger = logging.getLogger("deepfake_engine")


class GANArtifactAnalyzer:
    def __init__(self, params: GANArtifactParams):
        self.p = params

    def analyze_image(self, bgr: np.ndarray) -> Tuple[float, List[str]]:
        flags = []
        scores = []

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # FFT grid artifact check
        if self.p.fft_grid_artifact_check:
            s, f = self._fft_check(gray)
            scores.append(s)
            flags.extend(f)

        # High-frequency energy ratio
        hf_score, hf_flags = self._high_freq_check(gray)
        scores.append(hf_score)
        flags.extend(hf_flags)

        # Color histogram entropy
        ent_score, ent_flags = self._color_entropy(bgr)
        scores.append(ent_score)
        flags.extend(ent_flags)

        # Checkerboard artifact
        if self.p.checkerboard_artifact_check:
            cb_score, cb_flags = self._checkerboard_check(gray)
            scores.append(cb_score)
            flags.extend(cb_flags)

        # Noise uniformity
        if self.p.noise_pattern_consistency_check:
            nu_score, nu_flags = self._noise_uniformity(gray)
            scores.append(nu_score)
            flags.extend(nu_flags)

        # Edge sharpness
        es_score, es_flags = self._edge_sharpness(gray)
        scores.append(es_score)
        flags.extend(es_flags)

        return float(np.mean(scores)), flags

    def _fft_check(self, gray):
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        h, w = magnitude.shape
        center = magnitude[h//2-2:h//2+2, w//2-2:w//2+2]
        periphery = magnitude.copy()
        periphery[h//4:3*h//4, w//4:3*w//4] = 0
        p_mean = periphery[periphery > 0].mean() if (periphery > 0).any() else 0
        ratio = _safe_div(p_mean, center.mean(), 0)
        score = min(ratio / 0.5, 1.0)
        flags = ["fft_grid_artifact"] if score > 0.6 else []
        return score, flags

    def _high_freq_check(self, gray):
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        total = mag.sum()
        r = min(crow, ccol) // 3
        mask = np.zeros_like(mag, dtype=bool)
        for i in range(rows):
            for j in range(cols):
                if (i - crow)**2 + (j - ccol)**2 > r**2:
                    mask[i, j] = True
        hf_energy = mag[mask].sum()
        ratio = _safe_div(hf_energy, total, 0.5)
        flags = []
        if ratio < self.p.high_freq_energy_ratio_max:
            flags.append("suppressed_high_frequency")
        score = 1.0 - min(ratio / 0.5, 1.0)
        return float(score), flags

    def _color_entropy(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        h_hist = h_hist / h_hist.sum()
        h_hist = h_hist[h_hist > 0]
        entropy = float(-np.sum(h_hist * np.log2(h_hist)))
        flags = []
        if entropy < self.p.color_histogram_entropy_min:
            flags.append("low_color_entropy")
        score = 1.0 - min(entropy / 7.0, 1.0)
        return score, flags

    def _checkerboard_check(self, gray):
        h, w = gray.shape
        block = 8
        diffs = []
        for i in range(0, h - 2*block, block):
            for j in range(0, w - 2*block, block):
                b1 = gray[i:i+block, j:j+block].mean()
                b2 = gray[i:i+block, j+block:j+2*block].mean()
                diffs.append(abs(b1 - b2))
        if not diffs:
            return 0.0, []
        pattern_score = float(np.std(diffs) / max(np.mean(diffs), 1e-5))
        score = min(pattern_score / 2.0, 1.0)
        flags = ["checkerboard_pattern"] if score > 0.5 else []
        return score, flags

    def _noise_uniformity(self, gray):
        h, w = gray.shape
        block = max(h, w) // 4
        if block < 8:
            return 0.0, []
        variances = []
        for i in range(0, h - block, block):
            for j in range(0, w - block, block):
                patch = gray[i:i+block, j:j+block]
                noise = cv2.Laplacian(patch, cv2.CV_64F).var()
                variances.append(noise)
        if len(variances) < 2:
            return 0.0, []
        uniformity = float(np.std(variances) / max(np.mean(variances), 1e-5))
        score = min(uniformity / 1.0, 1.0)
        flags = []
        if uniformity > self.p.noise_variance_uniformity_max:
            flags.append("noise_pattern_inconsistency")
        return score, flags

    def _edge_sharpness(self, gray):
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_ratio = edges.mean() / 255.0
        lo, hi = self.p.edge_response_sharpness_range
        flags = []
        if edge_ratio < lo:
            flags.append("edges_too_soft")
            score = (lo - edge_ratio) / lo
        elif edge_ratio > hi:
            flags.append("edges_too_sharp")
            score = (edge_ratio - hi) / (1.0 - hi)
        else:
            score = 0.0
        return min(float(score), 1.0), flags
