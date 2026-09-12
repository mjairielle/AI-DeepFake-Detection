import os
import cv2
import logging
from typing import Optional
from models import MediaType

logger = logging.getLogger("deepfake_engine.media_utils")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}

def detect_media_type(path: str) -> MediaType:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTS:
        return MediaType.VIDEO
    if ext in AUDIO_EXTS:
        return MediaType.AUDIO
    raise ValueError(f"Unsupported file extension: {ext}")

def extract_frames(video_path: str, fps: int = 10,
                   max_frames: int = 300) -> list:
    """Sample frames from a video at the given FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(video_fps / fps), 1)
    frames = []
    idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames

def extract_audio_from_video(video_path: str, out_dir: str) -> Optional[str]:
    """Extract audio track to a temp WAV file using OpenCV (limited).
    For full extraction, ffmpeg would be used in production."""
    try:
        import subprocess
        audio_path = os.path.join(out_dir, "extracted_audio.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", audio_path],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(audio_path):
            return audio_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg not available — skipping audio extraction")
    return None
