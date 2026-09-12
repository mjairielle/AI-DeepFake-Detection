"""
Unit tests for the Learning Engine and its components.
"""
import os
import tempfile
import pytest
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from learning_engine.db import Base
import learning_engine.db

from learning_engine import (
    FeedbackRecord,
    FeedbackStore,
    AdaptiveThresholds,
    PatternMemory,
    AdaptiveClassifier,
    LearningEngine
)

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Setup an in-memory database for all tests in this file."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Monkeypatch the DB session factory used by the components
    monkeypatch.setattr(learning_engine.db, "SessionLocal", TestingSessionLocal)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

# --- FeedbackStore ---

def test_feedback_store():
    store = FeedbackStore()
    
    rec = FeedbackRecord(
        record_id="rec1", file_hash="hash1", media_type="image",
        original_verdict="authentic", corrected_verdict="deepfake",
        module_scores={"a": 0.1, "b": 0.9}, flags=["test_flag"]
    )
    store.add(rec)
    
    assert store.count == 1
    assert len(store.records) == 1
    
    # Check persistence
    store2 = FeedbackStore()
    assert store2.count == 1
    assert store2.records[0].corrected_verdict == "deepfake"
    
    # Check data formatting
    X, y = store.get_labeled_data()
    assert X.shape == (1, 2)
    assert y.shape == (1,)
    assert y[0] == 1.0  # deepfake

# --- AdaptiveThresholds ---

def test_adaptive_thresholds():
    thresh = AdaptiveThresholds()
    
    # 3 authentic feedbacks, 3 deepfake feedbacks to trigger adjustments
    for i in range(3):
        thresh.update_from_feedback(FeedbackRecord(
            "r1", "h", "image", "authentic", "authentic",
            {"mod1": 0.2}, []
        ))
    for i in range(3):
        thresh.update_from_feedback(FeedbackRecord(
            "r2", "h", "image", "authentic", "deepfake",
            {"mod1": 0.8}, []
        ))
        
    assert "mod1" in thresh.adjustments
    assert 0.4 < thresh.adjustments["mod1"] < 0.6  # Mean of 0.2 and 0.8 is 0.5
    
    # Apply
    adj = thresh.apply({"mod1": 0.8})
    assert adj["mod1"] > 0.8  # Should be pushed higher than original

# --- PatternMemory ---

def test_pattern_memory():
    mem = PatternMemory()
    
    mem.add_pattern({"mod1": 0.9, "mod2": 0.8}, "face_swap", ["flag1"])
    assert mem.count == 1
    
    # Find similar
    match = mem.find_similar({"mod1": 0.9, "mod2": 0.85})
    assert match is not None
    assert match["matched_type"] == "face_swap"
    
    # Different pattern shouldn't match (orthogonal vector)
    match2 = mem.find_similar({"mod1": 0.0, "mod2": 0.9})
    assert match2 is None

# --- AdaptiveClassifier ---

def test_adaptive_classifier(temp_dir):
    # Classifier still uses a pickle file for the ML model, so it needs temp_dir
    model_path = os.path.join(temp_dir, "model.pkl")
    store = FeedbackStore()
    clf = AdaptiveClassifier(model_path=model_path)
    
    # Minimum samples needed is 10
    clf.MIN_SAMPLES_TO_TRAIN = 4
    
    # Add samples
    for i in range(2):
        store.add(FeedbackRecord(f"r{i}", "h", "image", "a", "authentic", {"m1": 0.1}, []))
    for i in range(2):
        store.add(FeedbackRecord(f"r{i+2}", "h", "image", "a", "deepfake", {"m1": 0.9}, []))
        
    clf.train(store)
    
    if clf.is_ready: # Might be false if scikit-learn not available
        pred = clf.predict({"m1": 0.8})
        assert pred is not None
        assert "ml_probability" in pred

# --- LearningEngine ---

def test_learning_engine(temp_dir, monkeypatch):
    import learning_engine.classifier
    # Patch DATA_DIR only for the classifier (which saves model.pkl)
    monkeypatch.setattr(learning_engine.classifier, "DATA_DIR", temp_dir)
    
    engine = LearningEngine()
    
    res = engine.submit_feedback(
        "rec_id", "hash", "image", "authentic", "deepfake",
        {"mod1": 0.9}, ["test"]
    )
    
    assert res["status"] == "feedback_recorded"
    assert engine.feedback_store.count == 1
    
    scores, meta = engine.enhance_scores({"mod1": 0.9})
    assert isinstance(scores, dict)
    assert isinstance(meta, dict)
