"""
Pattern Memory — Fingerprint database of known deepfakes
"""
import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional

import numpy as np

from learning_engine.config import DATA_DIR

logger = logging.getLogger("learning_engine")


class PatternMemory:
    """
    Stores feature-vector fingerprints of confirmed deepfakes in DB.
    """

    def __init__(self, path: str = None):
        from learning_engine.db import SessionLocal
        self.Session = SessionLocal

    @property
    def patterns(self) -> List[Dict]:
        from learning_engine.db import PatternDB
        with self.Session() as db:
            db_patterns = db.query(PatternDB).order_by(PatternDB.id.desc()).limit(1000).all()
            
        result = []
        for p in db_patterns:
            result.append({
                "vector": p.vector,
                "keys": p.keys,
                "manipulation_type": p.manipulation_type,
                "flags": p.flags,
                "timestamp": p.timestamp
            })
        return result

    def add_pattern(self, module_scores: Dict[str, float],
                    manipulation_type: str, flags: List[str]):
        """Store a confirmed deepfake's feature vector."""
        from learning_engine.db import PatternDB
        keys = sorted(module_scores.keys())
        vector = [module_scores[k] for k in keys]
        
        db_pattern = PatternDB(
            manipulation_type=manipulation_type,
            timestamp=time.time()
        )
        db_pattern.vector = vector
        db_pattern.keys = keys
        db_pattern.flags = flags[:10]

        with self.Session() as db:
            db.add(db_pattern)
            db.commit()
            
        logger.info("Pattern stored: type=%s", manipulation_type)

    def find_similar(self, module_scores: Dict[str, float],
                     threshold: float = 0.85) -> Optional[Dict]:
        """Check if current scores are similar to any known deepfake pattern.
        Returns the best match or None."""
        current_patterns = self.patterns
        if not current_patterns:
            return None

        keys = sorted(module_scores.keys())
        query = np.array([module_scores.get(k, 0.0) for k in keys])

        best_sim = 0.0
        best_match = None

        for pat in current_patterns:
            pat_vec = np.array(pat["vector"])
            # Align vectors to same length
            max_len = max(len(query), len(pat_vec))
            q = np.pad(query, (0, max(0, max_len - len(query))))
            p = np.pad(pat_vec, (0, max(0, max_len - len(pat_vec))))

            # Cosine similarity
            dot = np.dot(q, p)
            norm = np.linalg.norm(q) * np.linalg.norm(p)
            sim = dot / norm if norm > 1e-10 else 0.0

            if sim > best_sim:
                best_sim = sim
                best_match = pat

        if best_sim >= threshold:
            return {
                "similarity": round(float(best_sim), 4),
                "matched_type": best_match["manipulation_type"],
                "matched_flags": best_match["flags"],
            }
        return None

    @property
    def count(self) -> int:
        from learning_engine.db import PatternDB
        with self.Session() as db:
            return db.query(PatternDB).count()
