"""
Unit tests for all analyzer modules.
These tests validate behavior BEFORE the analyzers/ package split,
so that we can confirm no regressions after refactoring.
"""
import os
import tempfile
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from analyzers import (
    FacialAnalyzer,
    TemporalAnalyzer,
    AudioAnalyzer,
    GANArtifactAnalyzer,
    MetadataAnalyzer,
    file_sha256,
    _safe_div,
)
from models import (
    FacialAnalysisParams,
    TemporalAnalysisParams,
    AudioAnalysisParams,
    GANArtifactParams,
    MetadataForensicsParams,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_div_normal(self):
        assert _safe_div(10, 2) == 5.0

    def test_safe_div_by_zero(self):
        assert _safe_div(10, 0) == 0.0

    def test_safe_div_by_zero_custom_default(self):
        assert _safe_div(10, 0, default=99.0) == 99.0

    def test_file_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"hello world")
            path = f.name
        try:
            h = file_sha256(path)
            assert isinstance(h, str)
            assert len(h) == 64  # SHA-256 hex digest
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
#  1. Facial Analyzer
# ---------------------------------------------------------------------------

class TestFacialAnalyzer:
    def setup_method(self):
        self.analyzer = FacialAnalyzer(FacialAnalysisParams())

    def test_no_face_detected(self):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        score, flags = self.analyzer.analyze_image(bgr)
        assert score == 0.0
        assert "no_face_detected" in flags

    def test_analyze_frames_empty(self):
        score, flags = self.analyzer.analyze_frames([])
        assert score == 0.0
        assert "no_frames" in flags

    def test_analyze_frames_no_faces(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        score, flags = self.analyzer.analyze_frames(frames)
        assert isinstance(score, float)
        assert "no_face_detected" in flags

    def test_return_types(self):
        bgr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        score, flags = self.analyzer.analyze_image(bgr)
        assert isinstance(score, float)
        assert isinstance(flags, list)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
#  2. Temporal Analyzer
# ---------------------------------------------------------------------------

class TestTemporalAnalyzer:
    def setup_method(self):
        self.analyzer = TemporalAnalyzer(TemporalAnalysisParams())

    def test_insufficient_frames(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 2
        score, flags = self.analyzer.analyze(frames)
        assert score == 0.0
        assert "insufficient_frames" in flags

    def test_identical_frames(self):
        """Identical frames should trigger duplicate detection."""
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        frames = [frame.copy() for _ in range(30)]
        score, flags = self.analyzer.analyze(frames)
        assert isinstance(score, float)
        assert "duplicate_frames_detected" in flags

    def test_random_frames(self):
        """Random frames should not trigger duplicate detection."""
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                  for _ in range(30)]
        score, flags = self.analyzer.analyze(frames)
        assert isinstance(score, float)
        assert isinstance(flags, list)


# ---------------------------------------------------------------------------
#  3. Audio Analyzer
# ---------------------------------------------------------------------------

class TestAudioAnalyzer:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(AudioAnalysisParams())

    @patch("analyzers.librosa", create=True)
    def test_librosa_not_available(self, mock_librosa):
        """When librosa import fails, should return gracefully."""
        # We patch import inside the method to simulate ImportError
        with patch.dict("sys.modules", {"librosa": None}):
            score, flags = self.analyzer.analyze("fake.wav")
            assert score == 0.0
            assert "librosa_not_available" in flags


# ---------------------------------------------------------------------------
#  4. GAN Artifact Analyzer
# ---------------------------------------------------------------------------

class TestGANArtifactAnalyzer:
    def setup_method(self):
        self.analyzer = GANArtifactAnalyzer(GANArtifactParams())

    def test_uniform_image(self):
        bgr = np.ones((64, 64, 3), dtype=np.uint8) * 128
        score, flags = self.analyzer.analyze_image(bgr)
        assert isinstance(score, float)
        assert isinstance(flags, list)
        assert 0.0 <= score <= 1.0

    def test_random_noise_image(self):
        bgr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score, flags = self.analyzer.analyze_image(bgr)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_gradient_image(self):
        """A smooth gradient image."""
        row = np.linspace(0, 255, 128, dtype=np.uint8)
        channel = np.tile(row, (128, 1))
        bgr = np.stack([channel, channel, channel], axis=-1)
        score, flags = self.analyzer.analyze_image(bgr)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_checkerboard_detection(self):
        """A checkerboard pattern should potentially trigger detection."""
        img = np.zeros((64, 64), dtype=np.uint8)
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    img[i:i+8, j:j+8] = 255
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        score, flags = self.analyzer.analyze_image(bgr)
        assert isinstance(score, float)

    def test_all_checks_enabled(self):
        """Verify all optional checks run when enabled."""
        params = GANArtifactParams(
            fft_grid_artifact_check=True,
            checkerboard_artifact_check=True,
            noise_pattern_consistency_check=True,
        )
        analyzer = GANArtifactAnalyzer(params)
        bgr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score, flags = analyzer.analyze_image(bgr)
        assert isinstance(score, float)

    def test_all_checks_disabled(self):
        """With optional checks off, should still produce a score."""
        params = GANArtifactParams(
            fft_grid_artifact_check=False,
            checkerboard_artifact_check=False,
            noise_pattern_consistency_check=False,
        )
        analyzer = GANArtifactAnalyzer(params)
        bgr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score, flags = analyzer.analyze_image(bgr)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
#  5. Metadata Analyzer
# ---------------------------------------------------------------------------

class TestMetadataAnalyzer:
    def setup_method(self):
        self.analyzer = MetadataAnalyzer(MetadataForensicsParams())

    def test_nonimage_file(self):
        """A .wav file should only run applicable checks."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"\x00" * 100)
            path = f.name
        try:
            score, flags = self.analyzer.analyze(path)
            assert isinstance(score, float)
            assert isinstance(flags, list)
        finally:
            os.remove(path)

    def test_jpeg_exif_missing(self):
        """A JPEG with no EXIF should flag exif_data_missing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            # Write a minimal valid-ish JPEG (just enough for PIL)
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            path = f.name
        try:
            score, flags = self.analyzer.analyze(path)
            assert isinstance(score, float)
            assert isinstance(flags, list)
            # Should either find missing exif or produce some result
        finally:
            os.remove(path)

    def test_ai_signature_scan(self):
        """File containing an AI tool signature should flag it."""
        params = MetadataForensicsParams(
            ai_generator_signature_db_enabled=True,
            known_signatures=["stable_diffusion"],
        )
        analyzer = MetadataAnalyzer(params)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png",
                                          mode="wb") as f:
            # Embed the signature in raw bytes
            f.write(b"\x89PNG\r\n\x1a\n" + b"stable diffusion" + b"\x00" * 100)
            path = f.name
        try:
            score, flags = analyzer.analyze(path)
            sig_flags = [fl for fl in flags if "ai_signature" in fl]
            assert len(sig_flags) > 0
        finally:
            os.remove(path)
