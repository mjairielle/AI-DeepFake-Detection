import pytest
import json
from models import DeepfakeDetectionConfig, DetectionResult

def test_config_serialization():
    config = DeepfakeDetectionConfig()
    config_json = config.to_json()
    assert isinstance(config_json, str)
    
    parsed = json.loads(config_json)
    assert 'facial' in parsed
    assert 'audio' in parsed
    
    restored = DeepfakeDetectionConfig.from_json(config_json)
    assert restored.max_file_size_mb == config.max_file_size_mb

def test_detection_result():
    res = DetectionResult(
        media_type="image",
        verdict="authentic",
        deepfake_probability=0.1,
        confidence=0.9
    )
    d = res.to_dict()
    assert d["verdict"] == "authentic"
    assert d["media_type"] == "image"
