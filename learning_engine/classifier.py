"""
Adaptive Classifier — ML model retrained on feedback
"""
import os
import math
import logging
import threading
from typing import Dict, List, Optional

import numpy as np

from learning_engine.config import DATA_DIR
from learning_engine.feedback import FeedbackStore
from security import DataIntegrity

logger = logging.getLogger("learning_engine")


class AdaptiveClassifier:
    """
    Lightweight sklearn classifier (GradientBoosting) that retrains
    whenever enough new feedback accumulates.
    """

    MIN_SAMPLES_TO_TRAIN = 10    # need at least this many labeled samples
    RETRAIN_INTERVAL = 5         # retrain every N new feedback records

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(DATA_DIR, "classifier.pkl")
        self.model = None
        self.feature_keys: List[str] = []
        self._samples_since_train = 0
        self.is_ready = False
        self.accuracy = 0.0
        self.train_size = 0
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                if not DataIntegrity.verify_file(self.model_path):
                    logger.warning("Classifier signature verification failed for %s — refusing to load",
                                   self.model_path)
                    return
                import pickle
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["model"]
                self.feature_keys = saved.get("feature_keys", [])
                acc = float(saved.get("accuracy", 0.0))
                self.accuracy = 0.0 if math.isnan(acc) else acc
                self.train_size = saved.get("train_size", 0)
                self.is_ready = True
                logger.info("Loaded adaptive classifier (accuracy=%.2f, samples=%d)",
                            self.accuracy, self.train_size)
            except Exception as e:
                logger.warning("Could not load classifier: %s", e)

    def _save_model(self):
        import pickle
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_keys": self.feature_keys,
                "accuracy": self.accuracy,
                "train_size": self.train_size,
            }, f)
        DataIntegrity.sign_file(self.model_path)

    def notify_feedback(self, store: FeedbackStore):
        """Called after new feedback. Triggers retrain if needed."""
        self._samples_since_train += 1
        if (self._samples_since_train >= self.RETRAIN_INTERVAL
                and store.count >= self.MIN_SAMPLES_TO_TRAIN):
            self.train(store)
            self._samples_since_train = 0

    def train(self, store: FeedbackStore):
        """(Re)train the classifier on all available feedback."""
        X, y, feature_keys = store.get_labeled_data_with_keys()
        if len(X) < self.MIN_SAMPLES_TO_TRAIN:
            logger.info("Not enough samples to train (%d < %d)",
                         len(X), self.MIN_SAMPLES_TO_TRAIN)
            return

        # Need both classes
        if len(np.unique(y)) < 2:
            logger.info("Need both authentic and deepfake samples to train")
            return

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            with self._lock:
                clf = GradientBoostingClassifier(
                    n_estimators=min(100, max(20, len(X) // 2)),
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )

                # Train on full dataset
                clf.fit(X, y)

                # Cross-validated accuracy estimate when both classes have enough members
                class_counts = np.bincount(y.astype(int))
                min_class_count = int(class_counts.min())
                cv_folds = min(5, len(X) // 4, min_class_count)
                if len(X) >= 20 and cv_folds >= 2:
                    cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
                    cv_mean = float(np.nanmean(cv_scores))
                    self.accuracy = 0.0 if math.isnan(cv_mean) else round(cv_mean, 4)
                else:
                    # Conservative estimate for small or heavily imbalanced datasets (10-19 samples)
                    train_acc = float(clf.score(X, y))
                    self.accuracy = round(min(0.5, train_acc * 0.5), 4)

                self.model = clf
                self.train_size = len(X)
                self.feature_keys = feature_keys
                self.is_ready = True

                self._save_model()

            logger.info("Classifier retrained: samples=%d, accuracy=%.3f",
                         len(X), self.accuracy)

        except ImportError:
            logger.warning("scikit-learn not available for adaptive classifier")
        except Exception as e:
            logger.exception("Classifier training failed: %s", e)

    def predict(self, module_scores: Dict[str, float]) -> Optional[Dict]:
        """
        Get adaptive classifier's prediction.
        Returns {probability, verdict, confidence} or None if not ready.
        """
        if not self.is_ready or self.model is None or not module_scores:
            return None

        if self.feature_keys:
            if not any(k in module_scores for k in self.feature_keys):
                return None
            vec = np.array([module_scores.get(k, 0.0) for k in self.feature_keys], dtype=np.float64).reshape(1, -1)
        else:
            keys = sorted(module_scores.keys())
            vec = np.array([module_scores.get(k, 0.0) for k in keys], dtype=np.float64).reshape(1, -1)

        # Pad/trim to match training feature count
        expected = self.model.n_features_in_
        if vec.shape[1] < expected:
            vec = np.pad(vec, ((0, 0), (0, expected - vec.shape[1])))
        elif vec.shape[1] > expected:
            vec = vec[:, :expected]

        try:
            with self._lock:
                proba = self.model.predict_proba(vec)[0]

            fake_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            conf = 0.0 if math.isnan(self.accuracy) else round(self.accuracy, 4)
            return {
                "ml_probability": round(fake_prob, 4),
                "ml_verdict": "deepfake" if fake_prob > 0.5 else "authentic",
                "ml_confidence": conf,
                "train_samples": self.train_size,
            }
        except Exception as e:
            logger.warning("Classifier prediction failed: %s", e)
            return None
