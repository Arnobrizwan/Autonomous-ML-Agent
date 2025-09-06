"""
FastAPI service module for the Autonomous ML Agent.
"""

from .app import app
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
]
