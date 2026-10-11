import pytest
import numpy as np
import cv2
import os

from detector import DeepfakeDetector
from models import MediaType, Verdict

@pytest.fixture
def detector():
    return DeepfakeDetector()

def test_detect_media_type_image(tmp_path):
    # Setup dummy image
    p = tmp_path / "test.jpg"
    p.write_text("")
    # detector.analyze should use media_utils.detect_media_type
    # But since it's an empty file, it might fail image loading.
    # We can mock detect_media_type or just use a real tiny image.
    pass

def test_detector_analyze_image(detector, tmp_path, monkeypatch):
    # Create a real small image
    img_path = str(tmp_path / "test.jpg")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)

    # Mock the analyzers to return deterministic values
    def mock_analyze_image(img_arg):
        return 0.9, ["mock_flag"]
    
    monkeypatch.setattr(detector.facial, "analyze_image", mock_analyze_image)
    monkeypatch.setattr(detector.gan, "analyze_image", mock_analyze_image)
    
    def mock_metadata_analyze(path_arg):
        return 0.9, ["meta_flag"]
    monkeypatch.setattr(detector.metadata, "analyze", mock_metadata_analyze)

    result = detector.analyze(img_path)
    assert result.media_type == MediaType.IMAGE.value
    # With high scores, verdict should be deepfake
    assert result.verdict == Verdict.DEEPFAKE.value
    assert "mock_flag" in result.flags
    assert "meta_flag" in result.flags

def test_detector_size_limit(detector, tmp_path):
    p = tmp_path / "large.jpg"
    with open(p, "wb") as f:
        # Just seek to make a sparse file
        f.seek((600 * 1024 * 1024) - 1)
        f.write(b"\0")
    
    result = detector.analyze(str(p))
    assert result.verdict == "error"
    assert "file_exceeds_max_size" in result.flags

def test_detector_authentic_studio_portrait(detector, tmp_path):
    """Ensure authentic portraits on plain/studio backdrops are classified as authentic."""
    p = tmp_path / "portrait.jpg"
    # Create an image with a white studio backdrop (240-250) and a natural subject
    img = np.ones((256, 256, 3), dtype=np.uint8) * 245
    # Add a person silhouette/skin region
    cv2.circle(img, (128, 110), 50, (140, 175, 220), -1)  # Face skin tone in BGR
    cv2.rectangle(img, (78, 150), (178, 256), (30, 30, 30), -1)  # Dark clothing
    # Add soft eye/mouth features
    cv2.circle(img, (110, 100), 5, (50, 40, 30), -1)
    cv2.circle(img, (146, 100), 5, (50, 40, 30), -1)
    cv2.ellipse(img, (128, 130), (15, 6), 0, 0, 180, (60, 50, 180), -1)
    # Natural camera sensor noise
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(p), noisy_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    result = detector.analyze(str(p))
    assert result.verdict == Verdict.AUTHENTIC.value
    assert result.deepfake_probability <= 0.30
    assert "face_bg_color_mismatch" not in result.flags
    assert "checkerboard_pattern" not in result.flags
    assert "fft_grid_artifact" not in result.flags
