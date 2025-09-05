"""
FastAPI service module for the Autonomous ML Agent.
"""

from .app import create_app, app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
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
