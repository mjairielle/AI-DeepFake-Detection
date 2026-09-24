FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    DEBUG=false

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update \
    && apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY analyzers ./analyzers
COPY learning_engine ./learning_engine
COPY static ./static
COPY api.py detector.py main.py media_utils.py models.py security.py ./
COPY learning_data/.gitkeep ./learning_data/.gitkeep
COPY uploads/.gitkeep ./uploads/.gitkeep

RUN mkdir -p /app/learning_data /app/uploads \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker: SQLite learning store and in-process rate limits are not multi-process safe.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 180 api:app"]
