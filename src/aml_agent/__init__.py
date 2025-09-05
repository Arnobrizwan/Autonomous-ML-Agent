"""
Autonomous ML Agent - Intelligent machine learning pipeline with LLM-guided optimization.

This package provides an end-to-end machine learning pipeline that automatically:
- Processes tabular data with intelligent preprocessing
- Trains multiple models with hyperparameter optimization
- Uses LLM guidance for intelligent search strategies
- Provides comprehensive model analysis and interpretability
- Exports production-ready models and services
"""

__version__ = "0.1.0"
__author__ = "AML Agent Team"

from .config import Config, load_config
from .types import (
    LeaderboardEntry,
    MetricType,
    ModelType,
    RunMetadata,
    SearchStrategy,
    TaskType,
    TrialResult,
)

__all__ = [
    "Config",
    "load_config",
    "TaskType",
    "MetricType",
    "SearchStrategy",
    "ModelType",
    "RunMetadata",
    "TrialResult",
    "LeaderboardEntry",
]
