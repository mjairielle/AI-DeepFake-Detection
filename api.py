import os
import uuid
import logging
from dataclasses import asdict

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from detector import DeepfakeDetector
from models import DeepfakeDetectionConfig
from media_utils import IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS
from security import (
    file_sanitizer, input_validator, rate_limiter,
    feedback_guard, apply_security_headers
)

logger = logging.getLogger("deepfake_engine.api")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

@app.before_request
def check_rate_limit():
    ip = request.remote_addr or "127.0.0.1"
    
    category = "query"
    if request.path.startswith("/api/analyze"):
        category = "analyze"
    elif request.path == "/api/feedback":
        category = "feedback"
    elif request.path == "/api/learning/retrain":
        category = "retrain"
        
    allowed, msg = rate_limiter.check(ip, category)
    if not allowed:
        return jsonify({"error": msg}), 429

@app.after_request
def set_security_headers(response):
    return apply_security_headers(response)

detector = DeepfakeDetector()

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health():
    learning = detector.learner.get_learning_status()
    return jsonify({
        "status": "ok",
        "engine": "deepfake_detection_v2_adaptive",
        "learning": learning,
    })

@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    """Return or update current detection configuration."""
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required."}), 400
        try:
            if data.get("_reset"):
                detector.config = DeepfakeDetectionConfig()
                detector.facial   = detector.facial.__class__(detector.config.facial)
                detector.temporal = detector.temporal.__class__(detector.config.temporal)
                detector.audio    = detector.audio.__class__(detector.config.audio)
                detector.gan      = detector.gan.__class__(detector.config.gan)
                detector.metadata = detector.metadata.__class__(detector.config.metadata)
                return jsonify({"status": "reset", "config": asdict(detector.config)})
            for group, params in data.items():
                if hasattr(detector.config, group) and isinstance(params, dict):
                    group_obj = getattr(detector.config, group)
                    for key, val in params.items():
                        if hasattr(group_obj, key):
                            current_val = getattr(group_obj, key)
                            if isinstance(current_val, bool):
                                val = bool(val) if isinstance(val, bool) else str(val).lower() == 'true'
                            elif isinstance(current_val, int) and not isinstance(current_val, bool):
                                val = int(float(val))
                            elif isinstance(current_val, float):
                                val = float(val)
                            elif isinstance(current_val, dict) and isinstance(val, dict):
                                updated_dict = dict(current_val)
                                for dk, dv in val.items():
                                    try:
                                        updated_dict[dk] = float(dv)
                                    except (ValueError, TypeError):
                                        updated_dict[dk] = dv
                                val = updated_dict
                            setattr(group_obj, key, val)
            # Propagate updated config parameters to analyzer modules
            detector.facial.p = detector.config.facial
            detector.temporal.p = detector.config.temporal
            detector.audio.p = detector.config.audio
            detector.gan.p = detector.config.gan
            detector.metadata.p = detector.config.metadata
            return jsonify({"status": "success", "config": asdict(detector.config)})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"config": asdict(detector.config)})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use 'file' field."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(uploaded.filename)[1].lower()
    all_exts = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS
    if ext not in all_exts:
        return jsonify({
            "error": f"Unsupported format '{ext}'.",
            "supported": sorted(all_exts),
        }), 400

    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    uploaded.save(save_path)

    is_safe, msg = file_sanitizer.validate(save_path, uploaded.filename)
    if not is_safe:
        os.remove(save_path)
        return jsonify({"error": msg}), 400

    try:
        result = detector.analyze(save_path)
        response = result.to_dict()
        response["original_filename"] = uploaded.filename
        logger.info("Result: %s  prob=%.2f  verdict=%s",
                     uploaded.filename, result.deepfake_probability, result.verdict)
        return jsonify(response)
    except Exception as e:
        logger.exception("Analysis failed for %s", uploaded.filename)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except OSError:
            pass

@app.route("/api/analyze/batch", methods=["POST"])
def analyze_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided. Use 'files' field."}), 400

    results = []
    for uploaded in files:
        ext = os.path.splitext(uploaded.filename)[1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        uploaded.save(save_path)

        is_safe, msg = file_sanitizer.validate(save_path, uploaded.filename)
        if not is_safe:
            results.append({
                "original_filename": uploaded.filename,
                "error": msg,
            })
            try:
                os.remove(save_path)
            except OSError:
                pass
            continue

        try:
            result = detector.analyze(save_path)
            r = result.to_dict()
            r["original_filename"] = uploaded.filename
            results.append(r)
        except Exception as e:
            results.append({
                "original_filename": uploaded.filename,
                "error": str(e),
            })
        finally:
            try:
                os.remove(save_path)
            except OSError:
                pass

    return jsonify({"results": results, "total": len(results)})

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required."}), 400

    ok, sanitized, msg = input_validator.validate_feedback_payload(data)
    if not ok:
        return jsonify({"error": msg}), 400

    ip = request.remote_addr or "127.0.0.1"
    ok, msg = feedback_guard.check(ip, sanitized["module_scores"], sanitized["corrected_verdict"])
    if not ok:
        return jsonify({"error": msg}), 400

    result = detector.learner.submit_feedback(
        record_id=uuid.uuid4().hex,
        file_hash=sanitized["file_hash"],
        media_type=sanitized.get("media_type", "unknown"),
        original_verdict=sanitized["original_verdict"],
        corrected_verdict=sanitized["corrected_verdict"],
        module_scores=sanitized["module_scores"],
        flags=sanitized.get("flags", []),
        metadata={"manipulation_type": sanitized.get("manipulation_type", "unknown")},
    )
    return jsonify(result)

@app.route("/api/learning/status", methods=["GET"])
def learning_status():
    return jsonify(detector.learner.get_learning_status())

@app.route("/api/learning/retrain", methods=["POST"])
def force_retrain():
    result = detector.learner.force_retrain()
    return jsonify(result)
