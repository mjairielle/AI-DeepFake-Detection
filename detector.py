import os
import time
import logging
import tempfile
from typing import Optional, Dict

import numpy as np
import cv2

from models import (
    DeepfakeDetectionConfig, DetectionResult,
    MediaType, Verdict, ManipulationType,
)
from analyzers import (
    FacialAnalyzer, TemporalAnalyzer, AudioAnalyzer,
    GANArtifactAnalyzer, MetadataAnalyzer, file_sha256,
)
from learning_engine import LearningEngine
from media_utils import detect_media_type, extract_frames, extract_audio_from_video

logger = logging.getLogger("deepfake_engine.detector")

class DeepfakeDetector:
    """
    Main orchestrator. Accepts a file path, routes to the correct
    analyzers based on media type, and returns a DetectionResult.
    """

    def __init__(self, config: Optional[DeepfakeDetectionConfig] = None):
        self.config = config or DeepfakeDetectionConfig()
        self.facial   = FacialAnalyzer(self.config.facial)
        self.temporal = TemporalAnalyzer(self.config.temporal)
        self.audio    = AudioAnalyzer(self.config.audio)
        self.gan      = GANArtifactAnalyzer(self.config.gan)
        self.metadata = MetadataAnalyzer(self.config.metadata)
        self.learner  = LearningEngine()

    def analyze(self, file_path: str) -> DetectionResult:
        t0 = time.perf_counter()
        file_hash = file_sha256(file_path)
        media_type = detect_media_type(file_path)
        logger.info("Analyzing %s [%s]  hash=%s...",
                     os.path.basename(file_path), media_type.value, file_hash[:12])

        # Size guard
        size_mb = os.path.getsize(file_path) / (1 << 20)
        if size_mb > self.config.max_file_size_mb:
            return self._error_result(media_type, file_hash, t0,
                                      "file_exceeds_max_size")

        if media_type == MediaType.IMAGE:
            return self._analyze_image(file_path, file_hash, t0)
        elif media_type == MediaType.VIDEO:
            return self._analyze_video(file_path, file_hash, t0)
        elif media_type == MediaType.AUDIO:
            return self._analyze_audio(file_path, file_hash, t0)
        else:
            return self._error_result(media_type, file_hash, t0,
                                      "unsupported_media_type")

    def _analyze_image(self, path, fhash, t0):
        bgr = cv2.imread(path)
        if bgr is None:
            return self._error_result(MediaType.IMAGE, fhash, t0,
                                      "image_load_failed")

        module_scores: Dict[str, float] = {}
        all_flags = []

        s, f = self.facial.analyze_image(bgr)
        module_scores["facial_analysis"] = s
        all_flags.extend(f)

        s, f = self.gan.analyze_image(bgr)
        module_scores["gan_artifacts"] = s
        all_flags.extend(f)

        s, f = self.metadata.analyze(path)
        module_scores["metadata_forensics"] = s
        all_flags.extend(f)

        # Heuristic for digital graphics, screenshots, and non-face images
        if "no_face_detected" in all_flags:
            all_flags.append("non_photographic_or_no_face")
            module_scores["gan_artifacts"] = 0.0
            module_scores["noise_analysis"] = 0.0
        else:
            module_scores["noise_analysis"] = module_scores.get("gan_artifacts", 0)

        weights = self.config.scoring.image_weights
        return self._build_result(
            MediaType.IMAGE, module_scores, weights,
            all_flags, fhash, t0,
        )

    def _analyze_video(self, path, fhash, t0):
        frames = extract_frames(path, self.config.temporal.analysis_fps)
        if not frames:
            return self._error_result(MediaType.VIDEO, fhash, t0,
                                      "no_frames_extracted")

        module_scores: Dict[str, float] = {}
        all_flags = []

        step = max(1, len(frames) // 20)
        key_frames = frames[::step]
        s, f = self.facial.analyze_frames(key_frames)
        module_scores["facial_analysis"] = s
        all_flags.extend(f)

        s, f = self.temporal.analyze(frames)
        module_scores["temporal_analysis"] = s
        all_flags.extend(f)

        gan_scores = []
        for kf in key_frames[:5]:
            gs, gf = self.gan.analyze_image(kf)
            gan_scores.append(gs)
            all_flags.extend(gf)
        module_scores["gan_artifacts"] = float(np.mean(gan_scores)) if gan_scores else 0.0

        with tempfile.TemporaryDirectory() as tmpd:
            audio_path = extract_audio_from_video(path, tmpd)
            if audio_path:
                s, f = self.audio.analyze(audio_path)
                module_scores["audio_analysis"] = s
                all_flags.extend(f)
            else:
                module_scores["audio_analysis"] = 0.0
                all_flags.append("no_audio_track")

        s, f = self.metadata.analyze(path)
        module_scores["metadata_forensics"] = s
        all_flags.extend(f)

        weights = self.config.scoring.video_weights
        return self._build_result(
            MediaType.VIDEO, module_scores, weights,
            all_flags, fhash, t0,
        )

    def _analyze_audio(self, path, fhash, t0):
        module_scores: Dict[str, float] = {}
        all_flags = []

        s, f = self.audio.analyze(path)
        module_scores["spectral_analysis"] = s
        module_scores["prosody_analysis"] = s
        module_scores["voice_clone_detect"] = s
        all_flags.extend(f)

        s2, f2 = self.metadata.analyze(path)
        module_scores["metadata_forensics"] = s2
        all_flags.extend(f2)

        weights = self.config.scoring.audio_weights
        return self._build_result(
            MediaType.AUDIO, module_scores, weights,
            all_flags, fhash, t0,
        )

    def _build_result(self, media_type, module_scores, weights,
                      flags, fhash, t0):
        adaptive_scores, learning_meta = self.learner.enhance_scores(module_scores)

        total_w = 0.0
        weighted_sum = 0.0
        for module, w in weights.items():
            score = adaptive_scores.get(module, 0.0)
            weighted_sum += score * w
            total_w += w

        probability = weighted_sum / total_w if total_w else 0.0
        probability = max(0.0, min(1.0, probability))

        score_vals = list(adaptive_scores.values())
        if len(score_vals) > 1:
            agreement = 1.0 - float(np.std(score_vals))
        else:
            agreement = 0.5
        confidence = max(0.0, min(1.0, agreement))

        if learning_meta.get("pattern_match"):
            confidence = min(1.0, confidence + 0.1)
            flags.append("matched_known_deepfake_pattern")

        sc = self.config.scoring
        if confidence < sc.min_confidence_for_verdict:
            verdict = "inconclusive"
        elif probability <= sc.authentic_max_score:
            verdict = Verdict.AUTHENTIC.value
        elif probability <= sc.suspicious_max_score:
            verdict = Verdict.SUSPICIOUS.value
        else:
            verdict = Verdict.DEEPFAKE.value

        manip = self._infer_manipulation(flags)

        elapsed = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            media_type=media_type.value,
            verdict=verdict,
            deepfake_probability=round(probability, 4),
            confidence=round(confidence, 4),
            manipulation_type=manip,
            module_scores={k: round(v, 4) for k, v in adaptive_scores.items()},
            flags=list(set(flags)),
            processing_time_ms=round(elapsed, 1),
            file_hash=fhash,
        )

    def _infer_manipulation(self, flags):
        flag_set = set(flags)
        if "blending_boundary_detected" in flag_set:
            return ManipulationType.FACE_SWAP.value
        if "face_bg_color_mismatch" in flag_set:
            return ManipulationType.FACE_SWAP.value
        if "monotone_pitch_detected" in flag_set:
            return ManipulationType.VOICE_CLONE.value
        if "ai_tool_in_exif" in flag_set or any("ai_signature" in f for f in flags):
            return ManipulationType.FULL_SYNTHESIS.value
        if "fft_grid_artifact" in flag_set:
            return ManipulationType.FULL_SYNTHESIS.value
        if "low_color_entropy" in flag_set:
            return ManipulationType.FULL_SYNTHESIS.value
        return ManipulationType.UNKNOWN.value

    def _error_result(self, media_type, fhash, t0, error_flag):
        elapsed = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            media_type=media_type.value,
            verdict="error",
            deepfake_probability=0.0,
            confidence=0.0,
            flags=[error_flag],
            processing_time_ms=round(elapsed, 1),
            file_hash=fhash,
        )
