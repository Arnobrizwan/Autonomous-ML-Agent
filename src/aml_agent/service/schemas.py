"""
Pydantic schemas for the FastAPI service.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    data: Dict[str, Any] = Field(..., description="Input data for prediction")


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    prediction: float = Field(..., description="Model prediction")
    probabilities: Optional[List[float]] = Field(
        None, description="Prediction probabilities (if available)"
    )
    model_info: Dict[str, str] = Field(
        ..., description="Information about the model used"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    data: List[Dict[str, Any]] = Field(
        ..., description="List of input data for batch prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: List[float] = Field(..., description="List of model predictions")
    probabilities: Optional[List[List[float]]] = Field(
        None, description="List of prediction probabilities (if available)"
    )
    n_predictions: int = Field(..., description="Number of predictions made")
    model_info: Dict[str, str] = Field(
        ..., description="Information about the model used"
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    pipeline_info: Dict[str, Any] = Field(..., description="Pipeline information")
    metadata: Dict[str, Any] = Field(..., description="Model metadata")
    loaded: bool = Field(..., description="Whether a model is currently loaded")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
