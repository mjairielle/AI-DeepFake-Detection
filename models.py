"""
==============================================================================
  DEEPFAKE DETECTION ENGINE — Backend Parameters & Analysis Pipeline
==============================================================================
  Defines all detection parameters, thresholds, feature extractors, and
  scoring logic for identifying AI-generated / manipulated media content.
==============================================================================
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMS — Media Types & Verdict Labels
# ─────────────────────────────────────────────────────────────────────────────

class MediaType(Enum):
    """Supported media types for deepfake analysis."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Verdict(Enum):
    """Final classification labels."""
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    DEEPFAKE = "deepfake"


class ManipulationType(Enum):
    """Known deepfake manipulation categories."""
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    LIP_SYNC = "lip_sync"
    FULL_SYNTHESIS = "full_synthesis"
    VOICE_CLONE = "voice_clone"
    TEXT_TO_SPEECH = "text_to_speech"
    BACKGROUND_MANIPULATION = "background_manipulation"
    ATTRIBUTE_EDITING = "attribute_editing"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACIAL ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FacialAnalysisParams:
    """
    Parameters for face-level deepfake detection.
    These examine geometric, textural, and biological consistency of faces.
    """

    # — Landmark Consistency —
    num_landmarks: int = 68                       # facial landmark points (68 or 478 for mediapipe)
    landmark_jitter_threshold: float = 2.5        # max px deviation across consecutive frames
    landmark_symmetry_tolerance: float = 0.15     # acceptable asymmetry ratio (0 = perfect)

    # — Blending Boundary Detection —
    blending_edge_width_px: int = 12              # search width around face boundary
    blending_gradient_threshold: float = 0.35     # sharp gradient = likely paste boundary
    color_mismatch_tolerance: float = 20.0        # max ΔE (CIE-Lab) between face & background skin

    # — Eye & Gaze Analysis —
    blink_rate_min_hz: float = 0.15               # normal adult: 0.25 Hz (15/min); below is suspicious
    blink_rate_max_hz: float = 0.50               # above is suspicious
    gaze_consistency_threshold: float = 0.80      # correlation between L/R eye direction
    pupil_reflection_check: bool = True           # verify specular highlights match environment
    pupil_shape_circularity_min: float = 0.85     # deepfakes often produce irregular pupils

    # — Skin & Texture —
    skin_texture_frequency_bands: int = 5         # multi-scale frequency analysis layers
    pore_visibility_min_score: float = 0.40       # natural skin pore pattern detection
    skin_smoothness_anomaly_threshold: float = 0.70   # over-smoothed = GAN artifact
    teeth_detail_score_min: float = 0.50          # deepfakes often blur/merge teeth

    # — Facial Region Coherence —
    face_bg_noise_ratio_threshold: float = 1.8    # noise level ratio (face vs. background)
    ear_consistency_check: bool = True            # ears often distorted in face swaps
    hair_boundary_sharpness_max: float = 0.90     # unnatural hair-line cutoff


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL / VIDEO ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalAnalysisParams:
    """
    Parameters for frame-to-frame temporal consistency checks (video only).
    """

    # — Frame Sampling —
    analysis_fps: int = 10                        # frames per second to sample for analysis
    min_frames_required: int = 30                 # minimum frames needed for temporal analysis
    keyframe_interval: int = 5                    # every Nth frame is a keyframe for deep checks

    # — Flicker & Stability —
    flicker_detection_threshold: float = 0.12     # luminance variance between consecutive frames
    face_position_stability_px: float = 8.0       # max jitter of face bounding box center
    face_size_stability_ratio: float = 0.05       # max proportional size change per frame

    # — Motion & Flow —
    optical_flow_consistency_min: float = 0.75    # natural motion coherence score
    head_pose_smoothness_threshold: float = 5.0   # max degrees rotation change per frame
    expression_transition_smoothness: float = 0.80 # AU (Action Unit) transition coherence

    # — Temporal Artifacts —
    ghosting_detection_enabled: bool = True       # detect semi-transparent face overlays
    frame_duplication_check: bool = True           # detect repeated / held frames
    compression_artifact_sensitivity: float = 0.60 # detect double-compression artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIO ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioAnalysisParams:
    """
    Parameters for detecting AI-generated or cloned voice audio.
    """

    # — Spectral Analysis —
    sample_rate_hz: int = 16000                   # target sample rate for analysis
    n_mfcc: int = 40                              # Mel-frequency cepstral coefficients
    spectral_rolloff_threshold: float = 0.85      # frequency below which 85% energy sits
    spectral_flatness_max: float = 0.60           # overly flat spectrum = synthetic

    # — Prosody & Naturalness —
    pitch_range_min_hz: float = 75.0              # human speech fundamental frequency range
    pitch_range_max_hz: float = 600.0
    pitch_variance_min: float = 10.0              # monotone = suspicious (Hz std dev)
    speaking_rate_words_per_min_min: float = 80.0
    speaking_rate_words_per_min_max: float = 200.0

    # — Breath & Micro-Pauses —
    breath_pattern_detection: bool = True         # natural speech includes breathing artifacts
    min_breath_events_per_minute: float = 4.0     # fewer = likely synthetic
    micro_pause_naturalness_score_min: float = 0.50

    # — Voice Cloning Artifacts —
    formant_consistency_threshold: float = 0.80   # formant structure stability
    harmonic_to_noise_ratio_min: float = 15.0     # dB — lower = noisy / synthetic edge
    phase_continuity_threshold: float = 0.75      # phase coherence across windows
    vocoder_artifact_check: bool = True           # detect neural vocoder signatures

    # — Audio-Visual Sync (for video) —
    lip_sync_correlation_min: float = 0.70        # mouth movement ↔ audio alignment
    lip_sync_delay_tolerance_ms: float = 80.0     # max acceptable A/V offset


# ─────────────────────────────────────────────────────────────────────────────
# 5. GAN / AI ARTIFACT DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GANArtifactParams:
    """
    Parameters for detecting GAN-specific and diffusion-model artifacts
    at the pixel / frequency level.
    """

    # — Frequency Domain —
    fft_grid_artifact_check: bool = True          # periodic spectral peaks from upsampling
    dct_block_analysis: bool = True               # DCT coefficient distribution anomalies
    high_freq_energy_ratio_max: float = 0.25      # GANs often suppress high-frequency detail

    # — Noise Fingerprinting —
    noise_pattern_consistency_check: bool = True   # sensor noise should be spatially consistent
    prnu_analysis_enabled: bool = True             # Photo Response Non-Uniformity analysis
    noise_variance_uniformity_max: float = 0.30    # variance of local noise estimates

    # — Color & Tone —
    color_histogram_entropy_min: float = 4.0       # low entropy = flat / synthetic palette
    saturation_anomaly_threshold: float = 0.40     # over-/under-saturated regions
    white_balance_consistency_check: bool = True    # mixed lighting = compositing artifact

    # — Edge & Detail —
    edge_response_sharpness_range: tuple = (0.10, 0.90)  # too sharp or too soft = artifact
    checkerboard_artifact_check: bool = True        # common in transposed convolutions
    aliasing_detection_threshold: float = 0.20      # staircase edges in generated content


# ─────────────────────────────────────────────────────────────────────────────
# 6. METADATA & PROVENANCE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetadataForensicsParams:
    """
    Parameters for file-level metadata and provenance inspection.
    """

    # — EXIF / Metadata —
    check_exif_integrity: bool = True              # missing/stripped EXIF = suspicious
    check_software_tag: bool = True                # known AI tools in Software field
    check_creation_tool_signatures: bool = True    # e.g. "Stable Diffusion", "MidJourney"

    # — C2PA / Content Credentials —
    verify_c2pa_manifest: bool = True              # validate Content Credentials chain
    require_c2pa_for_trust: bool = False           # if True, no C2PA → automatic suspicion

    # — File Structure —
    check_double_compression: bool = True          # JPEG re-saved detection
    quantization_table_analysis: bool = True       # mismatched quant tables = editing
    container_format_consistency: bool = True       # e.g. MP4 structure anomalies

    # — Known Source Matching —
    ai_generator_signature_db_enabled: bool = True  # match against known AI model fingerprints
    known_signatures: list = field(default_factory=lambda: [
        "stable_diffusion_v1",
        "stable_diffusion_xl",
        "midjourney_v5",
        "midjourney_v6",
        "dall_e_3",
        "runway_gen2",
        "synthesia",
        "heygen",
        "eleven_labs",
        "bark_tts",
        "tortoise_tts",
        "so_vits_svc",
        "rvc",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 7. SCORING & THRESHOLD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    """
    Weights and thresholds for combining individual detector scores
    into a final deepfake probability and verdict.
    """

    # — Per-Module Weights (must sum to 1.0 for each media type) —
    image_weights: dict = field(default_factory=lambda: {
        "facial_analysis":    0.30,
        "gan_artifacts":      0.30,
        "metadata_forensics": 0.15,
        "noise_analysis":     0.25,
    })

    video_weights: dict = field(default_factory=lambda: {
        "facial_analysis":    0.25,
        "temporal_analysis":  0.25,
        "gan_artifacts":      0.20,
        "audio_analysis":     0.15,
        "metadata_forensics": 0.15,
    })

    audio_weights: dict = field(default_factory=lambda: {
        "spectral_analysis":  0.30,
        "prosody_analysis":   0.25,
        "voice_clone_detect": 0.30,
        "metadata_forensics": 0.15,
    })

    # — Verdict Thresholds (0.0 – 1.0 deepfake probability) —
    authentic_max_score: float = 0.30              # ≤ 0.30 → AUTHENTIC
    suspicious_max_score: float = 0.70             # 0.31 – 0.70 → SUSPICIOUS
    # > 0.70 → DEEPFAKE

    # — Confidence —
    min_confidence_for_verdict: float = 0.60       # below this → "inconclusive"
    ensemble_agreement_min: float = 0.65           # fraction of sub-models that must agree


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER CONFIGURATION — Aggregates All Parameter Groups
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepfakeDetectionConfig:
    """
    Master configuration object that bundles every parameter group
    required for the deepfake detection pipeline.
    """
    facial:    FacialAnalysisParams   = field(default_factory=FacialAnalysisParams)
    temporal:  TemporalAnalysisParams = field(default_factory=TemporalAnalysisParams)
    audio:     AudioAnalysisParams    = field(default_factory=AudioAnalysisParams)
    gan:       GANArtifactParams      = field(default_factory=GANArtifactParams)
    metadata:  MetadataForensicsParams = field(default_factory=MetadataForensicsParams)
    scoring:   ScoringConfig          = field(default_factory=ScoringConfig)

    # — Global Settings —
    max_file_size_mb: int = 500                    # reject files larger than this
    supported_image_formats: list = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "webp", "bmp", "tiff"
    ])
    supported_video_formats: list = field(default_factory=lambda: [
        "mp4", "avi", "mov", "mkv", "webm"
    ])
    supported_audio_formats: list = field(default_factory=lambda: [
        "wav", "mp3", "flac", "ogg", "aac", "m4a"
    ])
    gpu_acceleration: bool = True
    batch_processing: bool = True
    max_concurrent_analyses: int = 4
    log_level: str = "INFO"

    def to_json(self, indent: int = 2) -> str:
        """Serialize the entire config to a JSON string."""
        return json.dumps(asdict(self), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DeepfakeDetectionConfig":
        """Deserialize a JSON string back into a config object."""
        data = json.loads(json_str)
        return cls(
            facial=FacialAnalysisParams(**data.get("facial", {})),
            temporal=TemporalAnalysisParams(**data.get("temporal", {})),
            audio=AudioAnalysisParams(**data.get("audio", {})),
            gan=GANArtifactParams(**data.get("gan", {})),
            metadata=MetadataForensicsParams(**data.get("metadata", {})),
            scoring=ScoringConfig(**data.get("scoring", {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. ANALYSIS RESULT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Structured output from a deepfake analysis run.
    """
    media_type: str                                # image | video | audio
    verdict: str                                   # authentic | suspicious | deepfake
    deepfake_probability: float                    # 0.0 – 1.0
    confidence: float                              # 0.0 – 1.0
    manipulation_type: str = "unknown"             # detected manipulation category
    module_scores: dict = field(default_factory=dict)  # per-module breakdown
    flags: list = field(default_factory=list)       # human-readable warning flags
    processing_time_ms: float = 0.0
    file_hash: str = ""                            # SHA-256 of input file

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 10. QUICK DEMO — Print Default Configuration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = DeepfakeDetectionConfig()

    print("=" * 72)
    print("  DEEPFAKE DETECTION ENGINE - Default Parameters")
    print("=" * 72)

    sections = {
        "Facial Analysis":    config.facial,
        "Temporal Analysis":  config.temporal,
        "Audio Analysis":     config.audio,
        "GAN Artifact Detection": config.gan,
        "Metadata Forensics": config.metadata,
        "Scoring Config":     config.scoring,
    }

    for title, section in sections.items():
        print(f"\n{'-' * 72}")
        print(f"  {title}")
        print(f"{'-' * 72}")
        for key, value in asdict(section).items():
            print(f"    {key:45s} = {value}")

    print(f"\n{'-' * 72}")
    print(f"  Global Settings")
    print(f"{'-' * 72}")
    print(f"    {'max_file_size_mb':45s} = {config.max_file_size_mb}")
    print(f"    {'gpu_acceleration':45s} = {config.gpu_acceleration}")
    print(f"    {'max_concurrent_analyses':45s} = {config.max_concurrent_analyses}")
    print(f"    {'supported_image_formats':45s} = {config.supported_image_formats}")
    print(f"    {'supported_video_formats':45s} = {config.supported_video_formats}")
    print(f"    {'supported_audio_formats':45s} = {config.supported_audio_formats}")
    print(f"\n{'=' * 72}")
    print("  Configuration exported successfully.")
    print(f"{'=' * 72}")

    # Export to JSON for downstream use
    config_json = config.to_json()
    print(f"\n  JSON size: {len(config_json):,} characters")
