import pytest
import io
from api import app, detector
from models import DetectionResult

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert 'learning' in data

def test_config_endpoint(client):
    response = client.get('/api/config')
    assert response.status_code == 200
    data = response.get_json()
    assert 'config' in data

def test_analyze_no_file(client):
    response = client.post('/api/analyze')
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_analyze_unsupported_format(client):
    data = {
        'file': (io.BytesIO(b"dummy data"), 'test.txt')
    }
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert 'Unsupported format' in response.get_json()['error']

def test_analyze_mocked(client, monkeypatch):
    # Mock detector.analyze
    def mock_analyze(save_path):
        return DetectionResult(
            media_type="image",
            verdict="suspicious",
            deepfake_probability=0.6,
            confidence=0.8,
            manipulation_type="unknown",
            module_scores={"facial_analysis": 0.6},
            flags=["test_flag"],
            processing_time_ms=10.0,
            file_hash="mock_hash"
        )
    monkeypatch.setattr(detector, "analyze", mock_analyze)

    data = {
        # Valid JPEG magic bytes to bypass FileSanitizer magic bytes check
        'file': (io.BytesIO(b"\xff\xd8\xff\xdb\x00C\x00dummy"), 'test.jpg')
    }
    response = client.post('/api/analyze', data=data, content_type='multipart/form-data', environ_base={'REMOTE_ADDR': '10.0.0.1'})
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data['verdict'] == 'suspicious'
    assert res_data['original_filename'] == 'test.jpg'

def test_feedback_endpoint(client, monkeypatch):
    # Mock learner.submit_feedback
    def mock_submit(*args, **kwargs):
        return {"status": "mock_recorded"}
    monkeypatch.setattr(detector.learner, "submit_feedback", mock_submit)

    payload = {
        "file_hash": "mock_hash",
        "media_type": "image",
        "original_verdict": "suspicious",
        "corrected_verdict": "deepfake",
        "module_scores": {"facial_analysis": 0.6},
        "flags": ["test_flag"],
        "manipulation_type": "unknown"
    }
    response = client.post('/api/feedback', json=payload, environ_base={'REMOTE_ADDR': '10.0.0.2'})
    assert response.status_code == 200
    assert response.get_json() == {"status": "mock_recorded"}

def test_security_headers(client):
    response = client.get('/api/health')
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"

def test_rate_limiting(client, monkeypatch):
    # The analyze endpoint has a limit of 10 burst.
    # We will make 11 requests from a unique IP.
    for i in range(10):
        res = client.post('/api/analyze', environ_base={'REMOTE_ADDR': '10.0.0.3'})
        assert res.status_code == 400
    
    # The 11th request should be rate limited
    res = client.post('/api/analyze', environ_base={'REMOTE_ADDR': '10.0.0.3'})
    assert res.status_code == 429
    assert "Rate limit exceeded" in res.get_json()['error']

def test_xss_feedback_payload(client):
    payload = {
        "file_hash": "mock_hash",
        "media_type": "image",
        "original_verdict": "suspicious",
        "corrected_verdict": "deepfake",
        "module_scores": {"facial_analysis": 0.6},
        "flags": ["<script>alert('xss')</script>"],
        "manipulation_type": "unknown"
    }
    response = client.post('/api/feedback', json=payload, environ_base={'REMOTE_ADDR': '10.0.0.4'})
    assert response.status_code == 400
    assert "error" in response.get_json()
