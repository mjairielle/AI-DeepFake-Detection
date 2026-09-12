"""
Adaptive Thresholds — Auto-tune from feedback statistics
"""
import os
import json
import threading
from typing import Dict

import numpy as np

from learning_engine.config import DATA_DIR
from learning_engine.feedback import FeedbackRecord


class AdaptiveThresholds:
    """
    Tracks per-module score distributions for authentic vs. deepfake
    and nudges thresholds toward the optimal decision boundary via DB.
    """

    def __init__(self, path: str = None):
        from learning_engine.db import SessionLocal
        self.Session = SessionLocal
        self._lock = threading.Lock()
        
    @property
    def adjustments(self) -> Dict[str, float]:
        from learning_engine.db import ThresholdStatsDB
        result = {}
        with self.Session() as db:
            stats = db.query(ThresholdStatsDB).all()
            for s in stats:
                if s.adjustment > 0.0:
                    result[s.module_name] = s.adjustment
        return result

    @property
    def stats(self) -> Dict[str, Dict]:
        from learning_engine.db import ThresholdStatsDB
        result = {}
        with self.Session() as db:
            stats = db.query(ThresholdStatsDB).all()
            for s in stats:
                result[s.module_name] = {
                    "auth_scores": s.auth_scores,
                    "fake_scores": s.fake_scores
                }
        return result

    def update_from_feedback(self, feedback: FeedbackRecord):
        """Incorporate one feedback record into threshold statistics."""
        from learning_engine.db import ThresholdStatsDB
        is_fake = feedback.corrected_verdict == "deepfake"

        with self._lock:
            with self.Session() as db:
                for module, score in feedback.module_scores.items():
                    stat_record = db.query(ThresholdStatsDB).filter_by(module_name=module).first()
                    if not stat_record:
                        stat_record = ThresholdStatsDB(module_name=module)
                        stat_record.auth_scores = []
                        stat_record.fake_scores = []
                        db.add(stat_record)
                        
                    if is_fake:
                        f_scores = stat_record.fake_scores
                        f_scores.append(score)
                        stat_record.fake_scores = f_scores[-499:]
                    else:
                        a_scores = stat_record.auth_scores
                        a_scores.append(score)
                        stat_record.auth_scores = a_scores[-499:]

                    # Recompute adjustment for this module
                    a_s = stat_record.auth_scores
                    f_s = stat_record.fake_scores
                    if len(a_s) >= 3 and len(f_s) >= 3:
                        auth_mean = np.mean(a_s)
                        fake_mean = np.mean(f_s)
                        midpoint = float((auth_mean + fake_mean) / 2)
                        stat_record.adjustment = midpoint
                
                db.commit()

    def apply(self, module_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply learned adjustments to raw module scores."""
        adjusted = {}
        current_adj = self.adjustments
        for module, raw_score in module_scores.items():
            if module in current_adj:
                mid = current_adj[module]
                # Re-center score around learned midpoint
                # Scores above midpoint get pushed higher, below get pushed lower
                if mid > 0.01:
                    adjusted_score = raw_score + (raw_score - mid) * 0.3
                    adjusted[module] = max(0.0, min(1.0, adjusted_score))
                else:
                    adjusted[module] = raw_score
            else:
                adjusted[module] = raw_score
        return adjusted

    @property
    def summary(self) -> Dict:
        result = {}
        from learning_engine.db import ThresholdStatsDB
        with self.Session() as db:
            stats = db.query(ThresholdStatsDB).all()
            for s in stats:
                result[s.module_name] = {
                    "authentic_samples": len(s.auth_scores),
                    "deepfake_samples": len(s.fake_scores),
                    "adjustment": s.adjustment,
                }
        return result
