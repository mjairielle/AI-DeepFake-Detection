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

        # DCT block analysis
        if getattr(self.p, 'dct_block_analysis', True):
            dct_score, dct_flags = self._dct_check(gray)
            scores.append(dct_score)
            flags.extend(dct_flags)

        if not scores:
            return 0.0, flags
        # Use a blend of max signal and mean to avoid diluting a single strong
        # deepfake indicator across many clean sub-checks.
        top_scores = sorted(scores, reverse=True)
        max_score = top_scores[0]
        mean_score = float(np.mean(scores))
        if max_score > 0.4:
            combined = 0.5 * max_score + 0.5 * mean_score
        else:
            combined = 0.7 * mean_score + 0.3 * max_score
        return float(combined), flags

    def _fft_check(self, gray):
        """
        FFT grid artifact check.
        Compares peripheral spectral energy to central, detecting periodic
        grid-like artifacts typical of GAN upsampling layers.
        """
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        h, w = magnitude.shape
        center = magnitude[h//2-2:h//2+2, w//2-2:w//2+2]
        periphery = magnitude.copy()
        periphery[h//4:3*h//4, w//4:3*w//4] = 0
        p = periphery[periphery > 0]
        if len(p) == 0:
            return 0.0, []
        p_mean = p.mean()
        p_max = p.max()
        p_std = p.std()
        ratio = _safe_div(p_mean, center.mean(), 0)
        spike_metric = (p_max - p_mean) / max(p_std, 1e-5)

        score = min(max(0.0, (ratio - 0.40) / 0.15), 1.0)
        flags = []
        if ratio > 0.52 and spike_metric > 5.5:
            flags.append("fft_grid_artifact")
        return float(score), flags

    def _high_freq_check(self, gray):
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        total = mag.sum()
        r = min(crow, ccol) // 3
        y, x = np.ogrid[:rows, :cols]
        mask = ((y - crow)**2 + (x - ccol)**2) > r**2
        hf_energy = mag[mask].sum()
        ratio = _safe_div(hf_energy, total, 0.5)

        # Check if image has large uniform background (e.g. studio portrait)
        non_bg_mask = (gray > 15) & (gray < 240)
        bg_fraction = 1.0 - (np.count_nonzero(non_bg_mask) / float(rows * cols))
        effective_min = self.p.high_freq_energy_ratio_max * (0.6 if bg_fraction > 0.3 else 1.0)

        flags = []
        if ratio < effective_min:
            flags.append("suppressed_high_frequency")
        score = 1.0 - min(ratio / 0.5, 1.0)
        return float(score), flags

    def _color_entropy(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        # Mask out uniform monochrome backdrops (pure white/black studio walls)
        active_mask = ~(((s < 20) & (v > 220)) | (v < 20))
        if np.count_nonzero(active_mask) > (bgr.shape[0] * bgr.shape[1] * 0.1):
            h_vals = hsv[:, :, 0][active_mask]
            h_hist, _ = np.histogram(h_vals, bins=180, range=(0, 180))
        else:
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()

        h_hist = h_hist.astype(float)
        if h_hist.sum() == 0:
            return 0.0, []
        h_hist = h_hist / h_hist.sum()
        h_hist = h_hist[h_hist > 0]
        entropy = float(-np.sum(h_hist * np.log2(h_hist)))
        flags = []
        if entropy < self.p.color_histogram_entropy_min:
            flags.append("low_color_entropy")
        score = 1.0 - min(entropy / 7.0, 1.0)
        return float(score), flags

    def _checkerboard_check(self, gray):
        h, w = gray.shape
        block = 8
        if h < 2 * block or w < 2 * block:
            return 0.0, []

        n_rows = h // block
        n_cols = w // block
        block_means = np.zeros((n_rows, n_cols), dtype=np.float64)
        for i in range(n_rows):
            for j in range(n_cols):
                block_means[i, j] = gray[i*block:(i+1)*block, j*block:(j+1)*block].mean()

        i_idx, j_idx = np.ogrid[:n_rows, :n_cols]
        parity = ((-1.0) ** (i_idx + j_idx))
        centered = block_means - block_means.mean()
        std_val = np.std(block_means)
        if std_val < 1e-4:
            return 0.0, []

        checker_corr = abs(np.mean(parity * centered)) / std_val
        score = min(max(0.0, (checker_corr - 0.25) / 0.5), 1.0)
        flags = ["checkerboard_pattern"] if score > 0.5 else []
        return float(score), flags

    def _noise_uniformity(self, gray):
        h, w = gray.shape
        block = max(h, w) // 4
        if block < 8:
            return 0.0, []

        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        res = gray.astype(np.float64) - blurred.astype(np.float64)

        sigmas = []
        for i in range(0, h - block + 1, block):
            for j in range(0, w - block + 1, block):
                patch_gray = gray[i:i+block, j:j+block]
                if patch_gray.std() < 3.0:
                    continue
                patch_res = res[i:i+block, j:j+block]
                mad = np.median(np.abs(patch_res - np.median(patch_res)))
                sigma = 1.4826 * mad
                if sigma > 0.1:
                    sigmas.append(sigma)

        if len(sigmas) < 2:
            return 0.0, []

        mean_sigma = np.mean(sigmas)
        std_sigma = np.std(sigmas)
        uniformity = float(std_sigma / max(mean_sigma, 1e-4))

        score = min(max(0.0, (uniformity - 0.5) / 0.8), 1.0)
        flags = []
        if uniformity > self.p.noise_variance_uniformity_max and score > 0.5:
            flags.append("noise_pattern_inconsistency")
        return float(score), flags

    def _edge_sharpness(self, gray):
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        h, w = gray.shape
        non_bg_mask = (gray > 15) & (gray < 240)
        active_pixels = np.count_nonzero(non_bg_mask)
        if active_pixels > (h * w * 0.15):
            edge_ratio = np.count_nonzero(edges[non_bg_mask]) / float(active_pixels)
        else:
            edge_ratio = edges.mean() / 255.0

        lo, hi = self.p.edge_response_sharpness_range
        effective_lo = min(lo, 0.012)
        flags = []
        if edge_ratio < effective_lo:
            flags.append("edges_too_soft")
            score = (effective_lo - edge_ratio) / effective_lo
        elif edge_ratio > hi:
            flags.append("edges_too_sharp")
            score = (edge_ratio - hi) / (1.0 - hi)
        else:
            score = 0.0
        return min(float(score), 1.0), flags

    def _dct_check(self, gray: np.ndarray) -> Tuple[float, List[str]]:
        """
        8x8 DCT Block analysis.
        Detects double-JPEG compression anomalies, periodic grid traces,
        and abnormal AC coefficient distributions common in synthetic generators.
        """
        h, w = gray.shape
        block = 8
        if h < block * 2 or w < block * 2:
            return 0.0, []

        ratios = []
        step = max(block, (min(h, w) // 16))
        for i in range(0, h - block + 1, step):
            for j in range(0, w - block + 1, step):
                blk = gray[i:i+block, j:j+block].astype(np.float32)
                if np.std(blk) < 3.0:
                    continue  # Skip uniform flat background blocks
                dct_blk = cv2.dct(blk)
                ac_vals = dct_blk.flatten()[1:]
                std_ac = np.std(ac_vals)
                mean_ac = np.mean(np.abs(ac_vals))
                if mean_ac > 1e-3:
                    ratios.append(std_ac / mean_ac)

        if len(ratios) < 10:
            return 0.0, []

        avg_ratio = float(np.mean(ratios))
        flags = []
        score = 0.0
        if avg_ratio < 1.15:
            score = min((1.15 - avg_ratio) / 0.5, 1.0)
            flags.append("dct_coefficient_anomaly")
        elif avg_ratio > 3.6:
            score = min((avg_ratio - 3.6) / 2.0, 1.0)
            flags.append("dct_periodic_frequency_anomaly")

        return float(score), flags
