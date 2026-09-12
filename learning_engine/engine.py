"""
Learning Engine — Unified orchestrator
"""
import logging
from typing import Dict, List, Tuple

from learning_engine.feedback import FeedbackRecord, FeedbackStore
from learning_engine.thresholds import AdaptiveThresholds
from learning_engine.patterns import PatternMemory
from learning_engine.classifier import AdaptiveClassifier

logger = logging.getLogger("learning_engine")


class LearningEngine:
    """
    Orchestrates all adaptive components.  Plugs into the main detector
    to enhance scoring and learn from corrections.
    """

    def __init__(self):
        self.feedback_store = FeedbackStore()
        self.thresholds = AdaptiveThresholds()
        self.pattern_memory = PatternMemory()
        self.classifier = AdaptiveClassifier()
        logger.info(
            "LearningEngine initialized: %d feedback records, "
            "%d patterns, classifier_ready=%s",
            self.feedback_store.count,
            self.pattern_memory.count,
            self.classifier.is_ready,
        )

    def enhance_scores(self, module_scores: Dict[str, float]
                       ) -> Tuple[Dict[str, float], Dict]:
        """
        Apply all learned enhancements to raw module scores.
        Returns (adjusted_scores, learning_metadata).
        """
        meta = {}

        # 1. Apply threshold adjustments
        adjusted = self.thresholds.apply(module_scores)

        # 2. Check pattern memory
        pattern_match = self.pattern_memory.find_similar(module_scores)
        if pattern_match:
            meta["pattern_match"] = pattern_match
            # Boost scores if similar to known deepfake
            boost = pattern_match["similarity"] * 0.15
            adjusted = {k: min(1.0, v + boost) for k, v in adjusted.items()}

        # 3. ML classifier prediction (blended in)
        ml_pred = self.classifier.predict(module_scores)
        if ml_pred:
            meta["ml_prediction"] = ml_pred
            # Weighted blend: 70% rule-based, 30% ML
            ml_weight = min(0.3, ml_pred["ml_confidence"] * 0.4)
            ml_prob = ml_pred["ml_probability"]
            for k in adjusted:
                adjusted[k] = adjusted[k] * (1 - ml_weight) + ml_prob * ml_weight

        meta["adjustments_applied"] = len(self.thresholds.adjustments) > 0
        meta["total_feedback"] = self.feedback_store.count
        meta["patterns_stored"] = self.pattern_memory.count

        return adjusted, meta

    def submit_feedback(self, record_id: str, file_hash: str,
                        media_type: str, original_verdict: str,
                        corrected_verdict: str,
                        module_scores: Dict[str, float],
                        flags: List[str],
                        metadata: Dict = None) -> Dict:
        """
        Process user feedback: store it, update thresholds,
        update pattern memory, trigger retraining.
        """
        record = FeedbackRecord(
            record_id=record_id,
            file_hash=file_hash,
            media_type=media_type,
            original_verdict=original_verdict,
            corrected_verdict=corrected_verdict,
            module_scores=module_scores,
            flags=flags,
            metadata=metadata or {},
        )

        # 1. Store feedback
        self.feedback_store.add(record)

        # 2. Update adaptive thresholds
        self.thresholds.update_from_feedback(record)

        # 3. If user confirmed deepfake, add to pattern memory
        if corrected_verdict == "deepfake":
            manip_type = (metadata or {}).get("manipulation_type", "unknown")
            self.pattern_memory.add_pattern(module_scores, manip_type, flags)

        # 4. Trigger classifier retraining check
        self.classifier.notify_feedback(self.feedback_store)

        return {
            "status": "feedback_recorded",
            "total_feedback": self.feedback_store.count,
            "patterns_stored": self.pattern_memory.count,
            "classifier_ready": self.classifier.is_ready,
            "classifier_accuracy": self.classifier.accuracy,
            "threshold_adjustments": self.thresholds.summary,
        }

    def get_learning_status(self) -> Dict:
        """Return current state of all learning components."""
        return {
            "feedback": {
                "total_records": self.feedback_store.count,
            },
            "adaptive_thresholds": {
                "modules_adjusted": len(self.thresholds.adjustments),
                "details": self.thresholds.summary,
            },
            "pattern_memory": {
                "stored_patterns": self.pattern_memory.count,
            },
            "classifier": {
                "is_ready": self.classifier.is_ready,
                "accuracy": self.classifier.accuracy,
                "train_samples": self.classifier.train_size,
                "min_samples_needed": self.classifier.MIN_SAMPLES_TO_TRAIN,
            },
        }

    def force_retrain(self) -> Dict:
        """Manually trigger classifier retraining."""
        self.classifier.train(self.feedback_store)
        return {
            "status": "retrain_complete",
            "is_ready": self.classifier.is_ready,
            "accuracy": self.classifier.accuracy,
            "train_samples": self.classifier.train_size,
        }
