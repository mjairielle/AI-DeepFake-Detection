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
        if ext in (".jpg", ".jpeg", ".png", ".tiff", ".webp"):
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
            img = Image.open(path)
            exif = img._getexif()
            if exif is None:
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
            # Count JPEG SOI markers
            soi_count = data.count(b'\xff\xd8')
            if soi_count > 1:
                return 0.6, ["multiple_jpeg_headers"]
            # Check quantization tables
            qt_count = data.count(b'\xff\xdb')
            if qt_count > 2:
                return 0.5, ["multiple_quantization_tables"]
            return 0.0, []
        except Exception:
            return 0.0, ["jpeg_analysis_error"]

    def _scan_signatures(self, path):
        try:
            with open(path, "rb") as f:
                raw = f.read(min(os.path.getsize(path), 1 << 20))
            text = raw.decode("utf-8", errors="ignore").lower()

            for sig in self.p.known_signatures:
                readable = sig.replace("_", " ").replace("-", " ")
                # Require word boundaries for very short signatures like 'rvc' to avoid random binary matches
                if len(sig) <= 4:
                    import re
                    if re.search(r'\b' + re.escape(sig) + r'\b', text) or re.search(r'\b' + re.escape(readable) + r'\b', text):
                        return 0.85, [f"ai_signature_found: {sig}"]
                else:
                    if readable in text or sig in text:
                        return 0.85, [f"ai_signature_found: {sig}"]
            return 0.0, []
        except Exception:
            return 0.0, []
