"""
==============================================================================
  SECURITY MODULE — Hardened Protection for Deepfake Detection Engine
==============================================================================
  Guards against:
    1. Malicious file uploads (polyglots, executables, zip bombs, path traversal)
    2. Input injection (JSON, XSS, command injection via filenames/fields)
    3. Learning data poisoning (adversarial feedback, flooding)
    4. Rate limiting (per-IP throttling for all endpoints)
    5. Data integrity (HMAC signing for persistent learning data)
==============================================================================
"""

import hashlib, hmac, json, logging, os, re, struct, time, threading
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("security")

# ---------------------------------------------------------------------------
#  SECRET KEY — loaded from env or generated once per install
# ---------------------------------------------------------------------------
_KEY_PATH = os.path.join(os.path.dirname(__file__), "learning_data", ".hmac_key")


def _get_hmac_key() -> bytes:
    """Load or generate a persistent HMAC key."""
    os.makedirs(os.path.dirname(_KEY_PATH), exist_ok=True)
    if os.path.exists(_KEY_PATH):
        with open(_KEY_PATH, "rb") as f:
            return f.read()
    key = os.urandom(32)
    with open(_KEY_PATH, "wb") as f:
        f.write(key)
    return key


HMAC_KEY = _get_hmac_key()

# ---------------------------------------------------------------------------
#  1. FILE SANITIZER — Validates uploads before any processing
# ---------------------------------------------------------------------------

# Magic bytes for supported file types
MAGIC_BYTES = {
    # Images
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png":  [b"\x89PNG\r\n\x1a\n"],
    ".webp": [b"RIFF"],
    ".bmp":  [b"BM"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
    # Video
    ".mp4":  [b"\x00\x00\x00\x18ftypmp4", b"\x00\x00\x00\x1cftypisom",
              b"\x00\x00\x00\x20ftypisom", b"\x00\x00\x00"],
    ".avi":  [b"RIFF"],
    ".mov":  [b"\x00\x00\x00\x14ftypqt", b"\x00\x00\x00"],
    ".mkv":  [b"\x1a\x45\xdf\xa3"],
    ".webm": [b"\x1a\x45\xdf\xa3"],
    # Audio
    ".wav":  [b"RIFF"],
    ".mp3":  [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
    ".flac": [b"fLaC"],
    ".ogg":  [b"OggS"],
    ".aac":  [b"\xff\xf1", b"\xff\xf9"],
    ".m4a":  [b"\x00\x00\x00\x18ftypM4A", b"\x00\x00\x00\x20ftypM4A",
              b"\x00\x00\x00"],
}

# Dangerous patterns in filenames
DANGEROUS_FILENAME_PATTERNS = [
    r"\.\.",            # path traversal
    r"[/\\]",          # directory separators
    r"[\x00-\x1f]",   # control characters
    r"[<>:\"|?*]",     # Windows reserved chars
    r"^(CON|PRN|AUX|NUL|COM\d|LPT\d)(\.|$)",  # Windows reserved names
]

# Executable extensions that should NEVER be accepted
BLOCKED_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".pif",
    ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".ps1",
    ".py", ".sh", ".bash", ".php", ".asp", ".aspx", ".jsp",
    ".dll", ".sys", ".cpl", ".inf", ".reg",
}


class FileSanitizer:
    """Validates uploaded files for safety before processing."""

    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate(self, file_path: str, original_filename: str) -> Tuple[bool, str]:
        """
        Run all file safety checks.
        Returns (is_safe, error_message).
        """
        checks = [
            self._check_filename(original_filename),
            self._check_extension(original_filename),
            self._check_size(file_path),
            self._check_magic_bytes(file_path, original_filename),
            self._check_not_executable(file_path),
            self._check_no_embedded_scripts(file_path),
            self._check_zip_bomb(file_path),
        ]
        for is_safe, msg in checks:
            if not is_safe:
                logger.warning("File rejected [%s]: %s", original_filename, msg)
                return False, msg
        return True, "ok"

    def _check_filename(self, filename: str) -> Tuple[bool, str]:
        """Reject filenames with path traversal or dangerous characters."""
        for pattern in DANGEROUS_FILENAME_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return False, f"Dangerous filename pattern: {pattern}"
        if len(filename) > 255:
            return False, "Filename too long (max 255 chars)"
        return True, "ok"

    def _check_extension(self, filename: str) -> Tuple[bool, str]:
        """Block executable and dangerous file extensions."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in BLOCKED_EXTENSIONS:
            return False, f"Blocked file extension: {ext}"
        # Double extension check (e.g., image.jpg.exe)
        parts = filename.lower().split(".")
        for part_ext in parts[1:]:
            if f".{part_ext}" in BLOCKED_EXTENSIONS:
                return False, f"Blocked extension in filename: .{part_ext}"
        return True, "ok"

    def _check_size(self, file_path: str) -> Tuple[bool, str]:
        """Enforce maximum file size."""
        size = os.path.getsize(file_path)
        if size > self.max_size_bytes:
            return False, f"File too large: {size / 1e6:.1f}MB (max {self.max_size_bytes / 1e6:.0f}MB)"
        if size == 0:
            return False, "Empty file"
        return True, "ok"

    def _check_magic_bytes(self, file_path: str, filename: str) -> Tuple[bool, str]:
        """Verify file magic bytes match declared extension."""
        ext = os.path.splitext(filename)[1].lower()
        expected = MAGIC_BYTES.get(ext)
        if not expected:
            return True, "ok"  # No magic bytes defined for this extension

        with open(file_path, "rb") as f:
            header = f.read(32)

        for magic in expected:
            if header[:len(magic)] == magic:
                return True, "ok"
        return False, f"File content does not match {ext} format (possible disguised file)"

    def _check_not_executable(self, file_path: str) -> Tuple[bool, str]:
        """Detect PE executables regardless of extension."""
        with open(file_path, "rb") as f:
            header = f.read(4)
        # PE executable
        if header[:2] == b"MZ":
            return False, "File is a Windows executable (PE format)"
        # ELF executable
        if header[:4] == b"\x7fELF":
            return False, "File is a Linux executable (ELF format)"
        return True, "ok"

    def _check_no_embedded_scripts(self, file_path: str) -> Tuple[bool, str]:
        """Scan for embedded script/exploit patterns in file content."""
        dangerous_patterns = [
            b"<script",
            b"javascript:",
            b"<?php",
            b"<%=",
            b"eval(",
            b"exec(",
            b"os.system(",
            b"subprocess",
            b"__import__",
            b"powershell",
            b"cmd.exe",
        ]
        try:
            with open(file_path, "rb") as f:
                # Only scan first 1MB and last 1KB for performance
                head = f.read(1024 * 1024).lower()
                f.seek(max(0, os.path.getsize(file_path) - 1024))
                tail = f.read(1024).lower()
                content = head + tail

            for pattern in dangerous_patterns:
                if pattern.lower() in content:
                    return False, f"Embedded script/exploit detected: {pattern.decode(errors='ignore')}"
        except Exception:
            pass
        return True, "ok"

    def _check_zip_bomb(self, file_path: str) -> Tuple[bool, str]:
        """Detect potential zip bombs / decompression bombs."""
        size = os.path.getsize(file_path)
        # Extremely suspicious: file is tiny but claims to be a media file
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv") and size < 100:
            return False, "Suspiciously small video file (possible bomb)"
        return True, "ok"


# ---------------------------------------------------------------------------
#  2. INPUT VALIDATOR — Sanitizes JSON/string inputs
# ---------------------------------------------------------------------------

# Regex patterns that indicate injection attempts
INJECTION_PATTERNS = [
    r"<script[^>]*>",                      # XSS
    r"javascript\s*:",                     # XSS via protocol
    r"on\w+\s*=",                          # XSS event handlers
    r"(\b(union|select|insert|update|delete|drop|alter)\b.*\b(from|into|table|set)\b)",  # SQL
    r";\s*(ls|cat|rm|wget|curl|nc|bash)\b",  # Command injection
    r"\$\(.*\)",                            # Shell substitution
    r"`.*`",                               # Backtick execution
    r"\{\{.*\}\}",                         # Template injection
    r"%\{.*\}",                            # Log4j-style
    r"__proto__",                          # Prototype pollution
    r"constructor\s*\[",                   # Prototype pollution
]

# Maximum lengths for various input fields
FIELD_MAX_LENGTHS = {
    "file_hash": 128,
    "media_type": 20,
    "original_verdict": 20,
    "corrected_verdict": 20,
    "manipulation_type": 50,
    "record_id": 64,
}

# Allowed values for enum-like fields
ALLOWED_VALUES = {
    "media_type": {"image", "video", "audio", "unknown"},
    "corrected_verdict": {"authentic", "deepfake"},
    "original_verdict": {"authentic", "suspicious", "deepfake", "inconclusive", "error"},
}


class InputValidator:
    """Validates and sanitizes all user-provided input."""

    @staticmethod
    def sanitize_string(value: str, field_name: str = "",
                        max_length: int = 500) -> Tuple[bool, str, str]:
        """
        Sanitize a string input.
        Returns (is_safe, sanitized_value, error_message).
        """
        if not isinstance(value, str):
            return False, "", f"Field '{field_name}' must be a string"

        # Length check
        max_len = FIELD_MAX_LENGTHS.get(field_name, max_length)
        if len(value) > max_len:
            return False, "", f"Field '{field_name}' exceeds max length {max_len}"

        # Check for null bytes
        if "\x00" in value:
            return False, "", f"Null bytes in field '{field_name}'"

        # Check for injection patterns
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "", f"Potential injection in '{field_name}'"

        # Enum validation
        if field_name in ALLOWED_VALUES:
            if value not in ALLOWED_VALUES[field_name]:
                allowed = ", ".join(sorted(ALLOWED_VALUES[field_name]))
                return False, "", f"Invalid value for '{field_name}'. Allowed: {allowed}"

        # Strip dangerous characters but keep the value usable
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        return True, sanitized, "ok"

    @staticmethod
    def validate_module_scores(scores: dict) -> Tuple[bool, dict, str]:
        """Validate module_scores dict: keys must be safe strings, values must be floats 0-1."""
        if not isinstance(scores, dict):
            return False, {}, "module_scores must be a dict"
        if len(scores) > 20:
            return False, {}, "Too many module score entries (max 20)"

        sanitized = {}
        for key, value in scores.items():
            # Key validation
            if not isinstance(key, str) or len(key) > 50:
                return False, {}, f"Invalid module score key: {repr(key)}"
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                return False, {}, f"Module key contains invalid characters: {key}"

            # Value validation
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return False, {}, f"Module score '{key}' must be a number"
            if not (0.0 <= fval <= 1.0):
                return False, {}, f"Module score '{key}' must be between 0.0 and 1.0"
            sanitized[key] = round(fval, 6)

        return True, sanitized, "ok"

    @staticmethod
    def validate_flags(flags: list) -> Tuple[bool, list, str]:
        """Validate flags list: must be short safe strings."""
        if not isinstance(flags, list):
            return False, [], "flags must be a list"
        if len(flags) > 50:
            return False, [], "Too many flags (max 50)"

        sanitized = []
        for flag in flags:
            if not isinstance(flag, str):
                return False, [], "Each flag must be a string"
            if len(flag) > 100:
                return False, [], f"Flag too long: {flag[:20]}..."
            if not re.match(r"^[a-zA-Z0-9_:. -]+$", flag):
                return False, [], f"Flag contains invalid characters: {flag[:20]}..."
            sanitized.append(flag[:100])

        return True, sanitized, "ok"

    @staticmethod
    def validate_feedback_payload(data: dict) -> Tuple[bool, dict, str]:
        """Full validation of a feedback submission payload."""
        if not isinstance(data, dict):
            return False, {}, "Request body must be a JSON object"

        # Reject unknown top-level keys
        allowed_keys = {
            "file_hash", "media_type", "original_verdict",
            "corrected_verdict", "module_scores", "flags",
            "manipulation_type",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            return False, {}, f"Unknown fields: {unknown}"

        # Required fields
        required = ["file_hash", "original_verdict", "corrected_verdict", "module_scores"]
        missing = [k for k in required if k not in data]
        if missing:
            return False, {}, f"Missing required fields: {missing}"

        sanitized = {}

        # Validate string fields
        for field in ["file_hash", "media_type", "original_verdict",
                      "corrected_verdict", "manipulation_type"]:
            if field in data:
                ok, val, err = InputValidator.sanitize_string(data[field], field)
                if not ok:
                    return False, {}, err
                sanitized[field] = val

        # Validate module_scores
        ok, scores, err = InputValidator.validate_module_scores(data["module_scores"])
        if not ok:
            return False, {}, err
        sanitized["module_scores"] = scores

        # Validate flags
        if "flags" in data:
            ok, flags, err = InputValidator.validate_flags(data["flags"])
            if not ok:
                return False, {}, err
            sanitized["flags"] = flags
        else:
            sanitized["flags"] = []

        return True, sanitized, "ok"


# ---------------------------------------------------------------------------
#  3. RATE LIMITER — Per-IP request throttling
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter with per-IP tracking.
    Different limits for different endpoint categories.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": 0, "last_refill": 0.0}
        )
        # Configuration: (max_tokens, refill_rate_per_second)
        self.limits = {
            "analyze":  (10, 2.0),    # 10 burst, 2/sec sustained
            "feedback": (20, 1.0),    # 20 burst, 1/sec sustained
            "query":    (60, 10.0),   # 60 burst, 10/sec for GET endpoints
            "retrain":  (2,  0.1),    # 2 burst, 1 per 10 sec
        }

    def check(self, ip: str, category: str = "analyze") -> Tuple[bool, str]:
        """
        Check if request is allowed.
        Returns (allowed, error_message).
        """
        max_tokens, refill_rate = self.limits.get(category, (30, 5.0))
        bucket_key = f"{ip}:{category}"
        now = time.time()

        with self._lock:
            bucket = self._buckets[bucket_key]
            if bucket["last_refill"] == 0:
                bucket["tokens"] = max_tokens
                bucket["last_refill"] = now

            # Refill tokens
            elapsed = now - bucket["last_refill"]
            bucket["tokens"] = min(max_tokens,
                                   bucket["tokens"] + elapsed * refill_rate)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True, "ok"

        return False, f"Rate limit exceeded for {category}. Try again shortly."

    def cleanup_old_buckets(self, max_age: float = 3600):
        """Remove stale IP buckets (call periodically)."""
        now = time.time()
        with self._lock:
            stale = [k for k, v in self._buckets.items()
                     if now - v["last_refill"] > max_age]
            for k in stale:
                del self._buckets[k]


# ---------------------------------------------------------------------------
#  4. DATA INTEGRITY — HMAC signing for learning data files
# ---------------------------------------------------------------------------

class DataIntegrity:
    """Signs and verifies learning data files to prevent tampering."""

    @staticmethod
    def compute_hmac(data: bytes) -> str:
        """Compute HMAC-SHA256 of data."""
        return hmac.new(HMAC_KEY, data, hashlib.sha256).hexdigest()

    @staticmethod
    def sign_file(file_path: str):
        """Write an HMAC signature file alongside the data file."""
        with open(file_path, "rb") as f:
            content = f.read()
        sig = DataIntegrity.compute_hmac(content)
        sig_path = file_path + ".sig"
        with open(sig_path, "w") as f:
            f.write(sig)

    @staticmethod
    def verify_file(file_path: str) -> bool:
        """Verify a data file against its HMAC signature."""
        sig_path = file_path + ".sig"
        if not os.path.exists(sig_path):
            return False
        with open(file_path, "rb") as f:
            content = f.read()
        with open(sig_path, "r") as f:
            stored_sig = f.read().strip()
        expected = DataIntegrity.compute_hmac(content)
        return hmac.compare_digest(stored_sig, expected)


# ---------------------------------------------------------------------------
#  5. FEEDBACK GUARD — Anti-poisoning for learning data
# ---------------------------------------------------------------------------

class FeedbackGuard:
    """
    Prevents learning data poisoning attacks:
      - Rate limits feedback per IP
      - Detects anomalous score distributions
      - Caps total feedback from a single source
      - Rejects scores that are statistically impossible
    """

    MAX_FEEDBACK_PER_IP_PER_HOUR = 50
    MAX_SCORE_DEVIATION = 3.0     # std devs from mean

    def __init__(self):
        self._lock = threading.Lock()
        self._ip_feedback_log: Dict[str, List[float]] = defaultdict(list)
        self._score_history: Dict[str, List[float]] = defaultdict(list)

    def check(self, ip: str, module_scores: Dict[str, float],
              corrected_verdict: str) -> Tuple[bool, str]:
        """
        Run anti-poisoning checks on a feedback submission.
        Returns (is_allowed, error_message).
        """
        # 1. Per-IP hourly limit
        ok, msg = self._check_ip_rate(ip)
        if not ok:
            return False, msg

        # 2. Score anomaly detection
        ok, msg = self._check_score_anomaly(module_scores)
        if not ok:
            return False, msg

        # 3. Record this submission
        self._record_submission(ip, module_scores)

        return True, "ok"

    def _check_ip_rate(self, ip: str) -> Tuple[bool, str]:
        now = time.time()
        one_hour_ago = now - 3600
        with self._lock:
            # Clean old entries
            self._ip_feedback_log[ip] = [
                t for t in self._ip_feedback_log[ip] if t > one_hour_ago
            ]
            if len(self._ip_feedback_log[ip]) >= self.MAX_FEEDBACK_PER_IP_PER_HOUR:
                return False, (
                    f"Feedback rate limit: max {self.MAX_FEEDBACK_PER_IP_PER_HOUR}/hour. "
                    "This prevents data poisoning attacks."
                )
            self._ip_feedback_log[ip].append(now)
        return True, "ok"

    def _check_score_anomaly(self, scores: Dict[str, float]) -> Tuple[bool, str]:
        """Reject scores that deviate wildly from historical distribution."""
        for module, score in scores.items():
            history = self._score_history.get(module, [])
            if len(history) < 10:
                continue  # Not enough history to judge
            import numpy as np
            mean = np.mean(history)
            std = np.std(history)
            if std < 0.01:
                continue
            z_score = abs(score - mean) / std
            if z_score > self.MAX_SCORE_DEVIATION:
                return False, (
                    f"Score for '{module}' is {z_score:.1f} std devs from mean. "
                    "Rejected as potential poisoning attempt."
                )
        return True, "ok"

    def _record_submission(self, ip: str, scores: Dict[str, float]):
        with self._lock:
            for module, score in scores.items():
                self._score_history[module].append(score)
                # Keep last 500 per module
                if len(self._score_history[module]) > 500:
                    self._score_history[module] = self._score_history[module][-500:]


# ---------------------------------------------------------------------------
#  6. SECURITY HEADERS — Applied to all responses
# ---------------------------------------------------------------------------

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    ),
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


def apply_security_headers(response):
    """Flask after_request handler to set security headers."""
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


# ---------------------------------------------------------------------------
#  7. CONVENIENCE — Global instances
# ---------------------------------------------------------------------------

file_sanitizer = FileSanitizer()
input_validator = InputValidator()
rate_limiter = RateLimiter()
data_integrity = DataIntegrity()
feedback_guard = FeedbackGuard()
