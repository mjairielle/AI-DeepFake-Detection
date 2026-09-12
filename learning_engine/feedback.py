"""
Feedback Store — Persistent labeled dataset from user corrections
"""
import os
import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np

from learning_engine.config import DATA_DIR

logger = logging.getLogger("learning_engine")


@dataclass
class FeedbackRecord:
    """One piece of user feedback on an analysis result."""
    record_id: str
    file_hash: str
    media_type: str
    original_verdict: str
    corrected_verdict: str            # user says: authentic / deepfake
    module_scores: Dict[str, float]   # feature vector at time of analysis
    flags: List[str]
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def feature_vector(self) -> np.ndarray:
        """Convert module_scores to a fixed-order numeric vector."""
        keys = sorted(self.module_scores.keys())
        return np.array([self.module_scores[k] for k in keys], dtype=np.float64)


class FeedbackStore:
    """SQLAlchemy-backed store for user feedback."""

    def __init__(self, path: str = None):
        # path is ignored now, kept for backward compatibility in signature if needed
        from learning_engine.db import SessionLocal
        self.Session = SessionLocal

    def add(self, record: FeedbackRecord):
        from learning_engine.db import FeedbackDB
        record.timestamp = record.timestamp or time.time()
        
        db_record = FeedbackDB(
            record_id=record.record_id,
            file_hash=record.file_hash,
            media_type=record.media_type,
            original_verdict=record.original_verdict,
            corrected_verdict=record.corrected_verdict,
            timestamp=record.timestamp,
        )
        db_record.module_scores = record.module_scores
        db_record.flags = record.flags
        db_record.metadata_dict = record.metadata
        
        with self.Session() as db:
            db.add(db_record)
            db.commit()
            
        logger.info("Feedback stored in DB: hash=%s  corrected=%s",
                     record.file_hash[:12], record.corrected_verdict)

    @property
    def records(self) -> List[FeedbackRecord]:
        from learning_engine.db import FeedbackDB
        with self.Session() as db:
            db_records = db.query(FeedbackDB).all()
            
        result = []
        for r in db_records:
            result.append(FeedbackRecord(
                record_id=r.record_id,
                file_hash=r.file_hash,
                media_type=r.media_type,
                original_verdict=r.original_verdict,
                corrected_verdict=r.corrected_verdict,
                module_scores=r.module_scores,
                flags=r.flags,
                timestamp=r.timestamp,
                metadata=r.metadata_dict
            ))
        return result

    @property
    def count(self) -> int:
        from learning_engine.db import FeedbackDB
        with self.Session() as db:
            return db.query(FeedbackDB).count()

    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays for classifier training.
        y: 0=authentic, 1=deepfake."""
        recs = self.records
        if not recs:
            return np.empty((0, 0)), np.empty(0)

        X_list, y_list = [], []
        for rec in recs:
            vec = rec.feature_vector()
            if len(vec) == 0:
                continue
            X_list.append(vec)
            y_list.append(1.0 if rec.corrected_verdict == "deepfake" else 0.0)

        if not X_list:
            return np.empty((0, 0)), np.empty(0)

        # Pad to same length
        max_len = max(len(v) for v in X_list)
        X = np.array([np.pad(v, (0, max_len - len(v))) for v in X_list])
        y = np.array(y_list)
        return X, y
