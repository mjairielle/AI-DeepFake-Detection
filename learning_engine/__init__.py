"""
Adaptive Learning Engine — Self-Improving Deepfake Detection

This package re-exports the main LearningEngine and models to preserve
backward compatibility with existing imports.
"""

from learning_engine.feedback import FeedbackRecord, FeedbackStore
from learning_engine.thresholds import AdaptiveThresholds
from learning_engine.patterns import PatternMemory
from learning_engine.classifier import AdaptiveClassifier
from learning_engine.engine import LearningEngine

__all__ = [
    "FeedbackRecord",
    "FeedbackStore",
    "AdaptiveThresholds",
    "PatternMemory",
    "AdaptiveClassifier",
    "LearningEngine",
]
