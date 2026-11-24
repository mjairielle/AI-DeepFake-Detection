"""
Deepfake Detection — Audio Analysis Module
Analyzes audio for synthetic speech indicators using spectral features,
pitch, MFCC regularity, zero-crossing rate, and harmonic-to-noise ratio.
"""
import logging
from typing import Tuple, List, Dict

import numpy as np

from models import AudioAnalysisParams
from analyzers.helpers import _safe_div

logger = logging.getLogger("deepfake_engine")


class AudioAnalyzer:
    def __init__(self, params: AudioAnalysisParams):
        self.p = params

    def analyze(self, audio_path: str) -> Tuple[float, List[str]]:
        overall, _, flags = self.analyze_detailed(audio_path)
        return overall, flags

    def analyze_detailed(self, audio_path: str) -> Tuple[float, Dict[str, float], List[str]]:
        empty_sub = {
            "spectral_analysis": 0.0,
            "prosody_analysis": 0.0,
            "voice_clone_detect": 0.0,
        }
        try:
            import librosa
        except ImportError:
            return 0.0, empty_sub, ["librosa_not_available"]

        flags = []

        try:
            y, sr = librosa.load(audio_path, sr=self.p.sample_rate_hz)
        except Exception as e:
            return 0.0, empty_sub, [f"audio_load_error: {e}"]

        if len(y) < sr:
            return 0.0, empty_sub, ["audio_too_short"]

        # Guard against pure silence (zero/near-zero energy)
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        if rms < 1e-4:
            return 0.0, empty_sub, ["audio_silent"]

        # Spectral flatness
        sf = librosa.feature.spectral_flatness(y=y)
        sf_mean = float(sf.mean())
        if sf_mean > self.p.spectral_flatness_max:
            flags.append("spectral_flatness_anomaly")
        sf_score = min(sf_mean / 1.0, 1.0)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        rolloff_norm = float(rolloff.mean() / (sr / 2))
        rolloff_score = 1.0 - min(max(rolloff_norm, 0.0), 1.0)

        # Pitch analysis
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]
        if len(pitch_vals) > 0:
            p_std = float(pitch_vals.std())
            if p_std < self.p.pitch_variance_min:
                flags.append("monotone_pitch_detected")
            pitch_score = 1.0 - min(p_std / 100.0, 1.0)
        else:
            flags.append("no_pitch_detected")
            pitch_score = 0.5

        # MFCC analysis — check for unnatural regularity
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.p.n_mfcc)
        mfcc_var = float(mfcc.var())
        mfcc_score = 1.0 - min(mfcc_var / 500.0, 1.0)
        if mfcc_score > 0.7:
            flags.append("mfcc_regularity_anomaly")

        # Zero crossing rate (synthetic speech often has smoother transitions)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_std = float(zcr.std())
        if zcr_std < 0.01:
            flags.append("low_zcr_variance")
        zcr_score = 1.0 - min(zcr_std / 0.05, 1.0)

        # Harmonic to noise ratio
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = _safe_div(np.abs(harmonic).mean(), np.abs(percussive).mean(), 1.0)
        hnr_db = 10 * np.log10(max(hnr, 1e-10))
        if hnr_db < self.p.harmonic_to_noise_ratio_min:
            flags.append("low_harmonic_to_noise")
        hnr_score = 1.0 - min(max(hnr_db, 0) / 30.0, 1.0)

        sub_scores = {
            "spectral_analysis": float(np.mean([sf_score, rolloff_score])),
            "prosody_analysis": float(np.mean([pitch_score, zcr_score])),
            "voice_clone_detect": float(np.mean([mfcc_score, hnr_score])),
        }
        overall = float(np.mean([sf_score, rolloff_score, pitch_score, mfcc_score, zcr_score, hnr_score]))
        return overall, sub_scores, flags
