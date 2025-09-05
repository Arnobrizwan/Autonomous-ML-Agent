"""
Model registry and training module for the Autonomous ML Agent.
"""

from .ensemble import EnsembleBuilder, create_ensemble
from .registries import ModelRegistry, get_model_factory
from .spaces import SearchSpaceGenerator, get_search_space
from .train_eval import ModelTrainer, evaluate_model

__all__ = [
    "ModelRegistry",
    "get_model_factory",
    "SearchSpaceGenerator",
    "get_search_space",
    "ModelTrainer",
    "evaluate_model",
    "EnsembleBuilder",
    "create_ensemble",
]
