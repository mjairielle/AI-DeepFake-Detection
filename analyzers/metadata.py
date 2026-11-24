"""
Deepfake Detection — Metadata Forensics Module
Analyzes file metadata: EXIF data, double compression markers,
and known AI generator signatures.
"""
import os
import logging
from typing import Tuple, List

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from models import MetadataForensicsParams

logger = logging.getLogger("deepfake_engine")


class MetadataAnalyzer:
    def __init__(self, params: MetadataForensicsParams):
        self.p = params

    def analyze(self, file_path: str) -> Tuple[float, List[str]]:
        flags = []
        scores = []
        ext = os.path.splitext(file_path)[1].lower()

        # EXIF check (images)
        if self.p.check_exif_integrity and ext in (".jpg", ".jpeg", ".png", ".tiff", ".webp"):
            exif_score, exif_flags = self._check_exif(file_path)
            scores.append(exif_score)
            flags.extend(exif_flags)

        # Double compression check
        if self.p.check_double_compression and ext in (".jpg", ".jpeg"):
            dc_score, dc_flags = self._double_compression(file_path)
            scores.append(dc_score)
            flags.extend(dc_flags)

        # AI generator signature scan
        if self.p.ai_generator_signature_db_enabled:
            sig_score, sig_flags = self._scan_signatures(file_path)
            scores.append(sig_score)
            flags.extend(sig_flags)

        if not scores:
            return 0.0, ["no_metadata_checks_applicable"]

        return float(np.mean(scores)), flags

    def _check_exif(self, path):
        flags = []
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                if not exif:
                    flags.append("exif_data_missing")
                    return 0.4, flags

                exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}

                if self.p.check_software_tag:
                    sw = str(exif_dict.get("Software", "")).lower()
                    ai_tools = ["stable diffusion", "midjourney", "dall-e",
                                "comfyui", "automatic1111", "invoke"]
                    for tool in ai_tools:
                        if tool in sw:
                            flags.append(f"ai_tool_in_exif: {tool}")
                            return 0.9, flags

                return 0.0, flags
        except Exception:
            flags.append("exif_read_error")
            return 0.3, flags

    def _double_compression(self, path):
        try:
            with open(path, "rb") as f:
                data = f.read()

            # Walk top-level JPEG markers so embedded EXIF/COM thumbnails don't cause false positives
            if len(data) < 4 or data[:2] != b"\xff\xd8":
                return 0.0, []

            pos = 2
            dqt_count = 0
            eoi_pos = None
            n = len(data)
            while pos + 1 < n:
                if data[pos] != 0xFF:
                    pos += 1
                    continue
                while pos < n and data[pos] == 0xFF:
                    pos += 1
                if pos >= n:
                    break
                marker = data[pos]
                pos += 1
                if marker == 0xD9:  # EOI
                    eoi_pos = pos
                    break
                if marker == 0xDA:  # SOS — compressed scan data follows until EOI
                    eoi_idx = data.find(b"\xff\xd9", pos)
                    eoi_pos = (eoi_idx + 2) if eoi_idx != -1 else n
                    break
                if marker in (0x00, 0x01) or (0xD0 <= marker <= 0xD8):
                    continue
                if pos + 2 > n:
                    break
                seg_len = int.from_bytes(data[pos:pos + 2], "big")
                if seg_len < 2:
                    break
                if marker == 0xDB:  # DQT
                    dqt_count += 1
                pos += seg_len

            # Another JPEG SOI after EOI indicates concatenated/appended JPEG stream
            if eoi_pos is not None and b"\xff\xd8" in data[eoi_pos:]:
                return 0.6, ["multiple_jpeg_headers"]
            if self.p.quantization_table_analysis and dqt_count > 2:
                return 0.5, ["multiple_quantization_tables"]
            return 0.0, []
        except Exception:
            return 0.0, ["jpeg_analysis_error"]

    def _scan_signatures(self, path):
        try:
            file_size = os.path.getsize(path)
            chunk_size = 1 << 20  # 1 MB
            with open(path, "rb") as f:
                head = f.read(min(file_size, chunk_size))
                tail = b""
                if file_size > chunk_size:
                    f.seek(max(chunk_size, file_size - chunk_size))
                    tail = f.read(chunk_size)
            text = (head + b"\n" + tail).decode("utf-8", errors="ignore").lower()

            for sig in self.p.known_signatures:
                # 'rvc' is an audio tool; do not match against image files where random bytes mimic it
                if sig == "rvc" and not path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')):
                    continue
                readable = sig.replace("_", " ").replace("-", " ")
                if len(sig) <= 4:
                    import re
                    if re.search(r'(?:^|[\s"\'<>{}\[\]=:])' + re.escape(sig) + r'(?:$|[\s"\'<>{}\[\]=:])', text):
                        return 0.85, [f"ai_signature_found: {sig}"]
                else:
                    if readable in text or sig in text:
                        return 0.85, [f"ai_signature_found: {sig}"]
            return 0.0, []
        except Exception:
            return 0.0, []
