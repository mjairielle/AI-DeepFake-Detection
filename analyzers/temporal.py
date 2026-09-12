"""
Deepfake Detection — Temporal Analysis Module (video)
Detects flicker, optical flow inconsistency, frame duplication,
and double-compression artifacts across video frames.
"""
import logging
from typing import Tuple, List

import numpy as np
import cv2

from models import TemporalAnalysisParams
from analyzers.helpers import _safe_div

logger = logging.getLogger("deepfake_engine")


class TemporalAnalyzer:
    def __init__(self, params: TemporalAnalysisParams):
        self.p = params

    def analyze(self, frames: List[np.ndarray]) -> Tuple[float, List[str]]:
        if len(frames) < self.p.min_frames_required:
            return 0.0, ["insufficient_frames"]

        flags = []
        scores = []

        # Flicker detection
        flicker = self._detect_flicker(frames)
        if flicker > self.p.flicker_detection_threshold:
            flags.append("temporal_flicker_detected")
        scores.append(min(flicker / 0.3, 1.0))

        # Optical flow consistency
        flow_score = self._optical_flow_check(frames)
        if flow_score < self.p.optical_flow_consistency_min:
            flags.append("optical_flow_inconsistency")
        scores.append(1.0 - flow_score)

        # Frame duplication
        if self.p.frame_duplication_check:
            dup_ratio = self._duplicate_frames(frames)
            if dup_ratio > 0.15:
                flags.append("duplicate_frames_detected")
            scores.append(dup_ratio)

        # Compression artifacts
        comp = self._compression_check(frames)
        if comp > self.p.compression_artifact_sensitivity:
            flags.append("double_compression_artifacts")
        scores.append(comp)

        return float(np.mean(scores)), flags

    def _detect_flicker(self, frames):
        luminances = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
        diffs = np.abs(np.diff(luminances))
        return float(diffs.mean() / 255.0)

    def _optical_flow_check(self, frames):
        consistencies = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for f in frames[1::max(1, len(frames)//10)]:
            curr_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            consistency = 1.0 - min(mag.std() / 20.0, 1.0)
            consistencies.append(consistency)
            prev_gray = curr_gray
        return float(np.mean(consistencies)) if consistencies else 1.0

    def _duplicate_frames(self, frames):
        dups = 0
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i]).mean()
            if diff < 1.0:
                dups += 1
        return dups / max(len(frames)-1, 1)

    def _compression_check(self, frames):
        scores = []
        for f in frames[::max(1, len(frames)//5)]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
            dct = cv2.dct(gray[:8*(gray.shape[0]//8), :8*(gray.shape[1]//8)])
            high_freq = np.abs(dct[dct.shape[0]//2:, dct.shape[1]//2:]).mean()
            low_freq = np.abs(dct[:dct.shape[0]//2, :dct.shape[1]//2]).mean()
            ratio = _safe_div(high_freq, low_freq, 0.5)
            scores.append(min(ratio, 1.0))
        return float(np.mean(scores)) if scores else 0.0
