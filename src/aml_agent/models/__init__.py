"""
Model registry and training module for the Autonomous ML Agent.
"""

from .ensemble import EnsembleBuilder, create_ensemble
from .registries import ModelRegistry, get_model_factory
from .spaces import SearchSpaceBuilder, get_search_space
from .train_eval import ModelTrainer

__all__ = [
    "ModelRegistry",
    "get_model_factory",
    "SearchSpaceBuilder",
    "get_search_space",
    "ModelTrainer",
    "EnsembleBuilder",
    "create_ensemble",
]
