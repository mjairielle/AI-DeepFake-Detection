# AI Deepfake Detection

Local Flask app that scores images, video, and audio for likely AI manipulation. A small web UI (DeepScan AI) uploads a file, runs forensic heuristics, and can take human corrections so thresholds and a lightweight classifier adapt over time.

This is a **research / demo tool**, not a certified forensic product. Verdicts are heuristic scores, not proof that media is real or fake.

## Features

- **Image analysis** — facial consistency, GAN/frequency artifacts, metadata forensics
- **Video analysis** — sampled frames plus temporal checks
- **Audio analysis** — clone / synthesis oriented cues
- **Adaptive learning** — feedback endpoint updates patterns, thresholds, and an optional classifier
- **Upload hardening** — type checks, rate limits, and signed learning-store integrity

## Stack

| Layer | Tech |
| --- | --- |
| API & UI | Flask, Flask-CORS, static HTML/CSS |
| Vision / audio | OpenCV, NumPy, SciPy, Pillow, librosa |
| Learning | scikit-learn, SQLite (SQLAlchemy) |
| Tests | pytest |

## Project layout

```
├── main.py              # Dev server entrypoint
├── api.py               # HTTP routes
├── detector.py          # Orchestrates analyzers
├── models.py            # Config, verdicts, result types
├── security.py          # Upload, rate-limit, HMAC helpers
├── media_utils.py       # Type detection, frame/audio extract
├── analyzers/           # Facial, temporal, GAN, metadata, audio
├── learning_engine/     # Feedback, thresholds, classifier
├── static/              # Web UI
├── tests/               # pytest suite
└── requirements.txt
```

Runtime files (`learning_data/`, `uploads/`) are created locally and are not committed.

## Requirements

- Python 3.10+ (developed on 3.13)
- FFmpeg on `PATH` recommended for some video/audio paths

## Quick start

```bash
git clone https://github.com/mjairielle/AI-DeepFake-Detection.git
cd AI-DeepFake-Detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `5000` | HTTP port |
| `DEBUG` | `false` | Flask debug mode |

```bash
# Example: listen on all interfaces (do not enable DEBUG on a public host)
set HOST=0.0.0.0
python main.py
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Health and learning status |
| `GET` | `/api/config` | Current detector config |
| `POST` | `/api/analyze` | Multipart field `file` |
| `POST` | `/api/analyze/batch` | Multipart field `files` |
| `POST` | `/api/feedback` | JSON correction for a prior result |
| `GET` | `/api/learning/status` | Feedback / classifier stats |
| `POST` | `/api/learning/retrain` | Force classifier retrain |

Example:

```bash
curl -F "file=@sample.jpg" http://127.0.0.1:5000/api/analyze
```

Typical JSON fields: `verdict` (`authentic` / `suspicious` / `deepfake`), `deepfake_probability`, `confidence`, `manipulation_type`, `module_scores`, `flags`, `file_hash`, `processing_time_ms`.

## Tests

```bash
pytest tests -q
```

## Limitations

- Scores come from classical image/audio forensics, not a large pretrained deepfake model.
- Missing EXIF, heavy compression, or no detected face can look “suspicious” even for real media.
- Do not deploy the default debug-style setup to the public internet without auth, HTTPS, and a reverse proxy.

## License

MIT. See [LICENSE](LICENSE).
