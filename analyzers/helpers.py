"""
Deepfake Detection — Helper Utilities
Shared across all analyzer modules.
"""
import hashlib


def _safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
