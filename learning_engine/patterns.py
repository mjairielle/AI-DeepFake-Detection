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
        if not module_scores:
            return
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
        if not module_scores:
            return None
        current_patterns = self.patterns
        if not current_patterns:
            return None

        query_keys = set(module_scores.keys())
        query_vals = np.array([module_scores[k] for k in sorted(query_keys)], dtype=np.float64)

        # Authentic media with low overall scores should not match deepfake patterns
        if float(np.mean(query_vals)) < 0.20 and float(np.linalg.norm(query_vals)) < 0.25:
            return None

        best_sim = 0.0
        best_match = None

        for pat in current_patterns:
            pat_keys = pat.get("keys") or []
            pat_vec = pat.get("vector") or []
            if not pat_keys or len(pat_keys) != len(pat_vec):
                continue

            pat_map = dict(zip(pat_keys, pat_vec))
            pat_key_set = set(pat_keys)
            overlap = query_keys & pat_key_set
            # Require meaningful key overlap so different media modalities never falsely match
            if not overlap or len(overlap) < 0.5 * max(len(query_keys), len(pat_key_set)):
                continue

            aligned_keys = sorted(query_keys | pat_key_set)
            q = np.array([module_scores.get(k, 0.0) for k in aligned_keys], dtype=np.float64)
            p = np.array([pat_map.get(k, 0.0) for k in aligned_keys], dtype=np.float64)

            # Cosine similarity with directional fingerprint alignment
            dot = np.dot(q, p)
            q_norm = np.linalg.norm(q)
            p_norm = np.linalg.norm(p)
            norm = q_norm * p_norm
            cos_sim = dot / norm if norm > 1e-10 else 0.0

            # Only penalize if query has significantly lower energy than the deepfake pattern
            sim = cos_sim if q_norm >= p_norm else cos_sim * (q_norm / max(p_norm, 1e-5))

            if sim > best_sim:
                best_sim = sim
                best_match = pat

        if best_sim >= threshold and best_match is not None:
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
