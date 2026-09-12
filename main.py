import os

from api import app, detector

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}

    print("=" * 60)
    print("  DEEPFAKE DETECTION ENGINE v2 - Adaptive Backend")
    print("=" * 60)
    print()
    print("  Detection Endpoints:")
    print("    GET  /api/health            - Health + learning status")
    print("    GET  /api/config            - Current config")
    print("    POST /api/analyze           - Analyze single file")
    print("    POST /api/analyze/batch     - Analyze multiple files")
    print()
    print("  Learning Endpoints:")
    print("    POST /api/feedback          - Submit correction feedback")
    print("    GET  /api/learning/status   - Learning engine status")
    print("    POST /api/learning/retrain  - Force classifier retrain")
    print()

    ls = detector.learner.get_learning_status()
    print(f"  Feedback records:  {ls['feedback']['total_records']}")
    print(f"  Stored patterns:   {ls['pattern_memory']['stored_patterns']}")
    print(f"  Classifier ready:  {ls['classifier']['is_ready']}")
    if ls['classifier']['is_ready']:
        print(f"  Classifier acc:    {ls['classifier']['accuracy']:.2%}")
    print()
    print(f"  Listening on http://{host}:{port}  (DEBUG={debug})")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)
