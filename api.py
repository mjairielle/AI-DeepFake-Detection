import os
import math
import time
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

MAX_BATCH_FILES = 20

_POSITIVE_INT_PARAMS = {
    "analysis_fps": 1,
    "min_frames_required": 2,
    "blending_edge_width_px": 1,
    "sample_rate_hz": 1000,
    "n_mfcc": 1,
    "max_file_size_mb": 1,
    "max_concurrent_analyses": 1,
}

_ALLOWED_GLOBAL_CONFIG_KEYS = {
    "max_file_size_mb",
    "gpu_acceleration",
    "batch_processing",
    "max_concurrent_analyses",
    "log_level",
}


@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    """Return or update current detection configuration."""
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return jsonify({"error": "JSON object body required."}), 400
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
                if group.startswith("_"):
                    continue
                if group in _ALLOWED_GLOBAL_CONFIG_KEYS:
                    current_val = getattr(detector.config, group)
                    if isinstance(current_val, bool):
                        val = bool(params) if isinstance(params, bool) else str(params).lower() == "true"
                    elif isinstance(current_val, int):
                        if isinstance(params, bool):
                            raise ValueError(f"'{group}' must be an integer")
                        fval = float(params)
                        if math.isnan(fval) or math.isinf(fval):
                            raise ValueError(f"'{group}' must be finite")
                        val = int(fval)
                        min_val = _POSITIVE_INT_PARAMS.get(group)
                        if min_val is not None and val < min_val:
                            raise ValueError(f"'{group}' must be >= {min_val}")
                    else:
                        val = str(params)
                    setattr(detector.config, group, val)
                    continue

                if hasattr(detector.config, group) and isinstance(params, dict):
                    group_obj = getattr(detector.config, group)
                    for key, val in params.items():
                        if key.startswith("_") or not hasattr(group_obj, key):
                            continue
                        current_val = getattr(group_obj, key)
                        if isinstance(current_val, bool):
                            val = bool(val) if isinstance(val, bool) else str(val).lower() == "true"
                        elif isinstance(current_val, int):
                            if isinstance(val, bool):
                                raise ValueError(f"'{group}.{key}' must be an integer")
                            fval = float(val)
                            if math.isnan(fval) or math.isinf(fval):
                                raise ValueError(f"'{group}.{key}' must be finite")
                            val = int(fval)
                            min_val = _POSITIVE_INT_PARAMS.get(key)
                            if min_val is not None and val < min_val:
                                raise ValueError(f"'{group}.{key}' must be >= {min_val}")
                        elif isinstance(current_val, float):
                            if isinstance(val, bool):
                                raise ValueError(f"'{group}.{key}' must be a float")
                            fval = float(val)
                            if math.isnan(fval) or math.isinf(fval):
                                raise ValueError(f"'{group}.{key}' must be finite")
                            val = fval
                        elif isinstance(current_val, tuple):
                            if not isinstance(val, (list, tuple)) or len(val) != len(current_val):
                                raise ValueError(f"'{group}.{key}' must be a {len(current_val)}-element sequence")
                            parsed = []
                            for item in val:
                                if isinstance(item, bool):
                                    raise ValueError(f"'{group}.{key}' elements must be numbers")
                                fi = float(item)
                                if math.isnan(fi) or math.isinf(fi):
                                    raise ValueError(f"'{group}.{key}' elements must be finite")
                                parsed.append(fi)
                            if key == "edge_response_sharpness_range" and not (0.0 < parsed[0] < parsed[1] < 1.0):
                                raise ValueError("edge_response_sharpness_range must satisfy 0 < lo < hi < 1")
                            val = tuple(parsed)
                        elif isinstance(current_val, dict):
                            if not isinstance(val, dict):
                                raise ValueError(f"'{group}.{key}' must be an object")
                            updated_dict = dict(current_val)
                            for dk, dv in val.items():
                                if isinstance(dv, bool):
                                    raise ValueError(f"'{group}.{key}.{dk}' must be a number")
                                fd = float(dv)
                                if math.isnan(fd) or math.isinf(fd):
                                    raise ValueError(f"'{group}.{key}.{dk}' must be finite")
                                if key.endswith("_weights") and fd < 0.0:
                                    raise ValueError(f"Weight '{dk}' must be >= 0")
                                updated_dict[dk] = fd
                            if key.endswith("_weights") and sum(updated_dict.values()) <= 0.0:
                                raise ValueError(f"'{group}.{key}' must sum to > 0")
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
    if len(files) > MAX_BATCH_FILES:
        return jsonify({"error": f"Too many files in batch (max {MAX_BATCH_FILES})."}), 400

    t0 = time.perf_counter()
    all_exts = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS
    results = []
    for uploaded in files:
        if not uploaded.filename:
            results.append({
                "original_filename": "",
                "error": "Empty filename.",
            })
            continue

        ext = os.path.splitext(uploaded.filename)[1].lower()
        if ext not in all_exts:
            results.append({
                "original_filename": uploaded.filename,
                "error": f"Unsupported format '{ext}'.",
            })
            continue

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

    total_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify({"results": results, "total": len(results), "total_time_ms": total_time_ms})

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Valid JSON body required."}), 400

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
