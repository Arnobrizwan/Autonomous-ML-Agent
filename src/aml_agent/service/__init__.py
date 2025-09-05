"""
FastAPI service module for the Autonomous ML Agent.
"""

from .app import app, create_app
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    "create_app",
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
]
