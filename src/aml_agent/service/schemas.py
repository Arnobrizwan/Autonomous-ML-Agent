"""
Pydantic schemas for the FastAPI service.
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Single prediction request schema."""

    features: Dict[str, Union[str, int, float]] = Field(
        ..., description="Feature values"
    )
    run_id: str = Field(..., description="Run ID of the trained model")

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {"feature_0": 1.5, "feature_1": 2.3, "feature_2": 0.8},
                "run_id": "run_20241201_143022_abc123",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Single prediction response schema."""

    prediction: Union[int, float, List[float]] = Field(
        ..., description="Prediction result"
    )
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[List[float]] = Field(
        None, description="Class probabilities"
    )
    model_type: Optional[str] = Field(None, description="Model type used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "confidence": 0.85,
                "probabilities": [0.15, 0.85],
                "model_type": "random_forest",
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    data: Union[List[Dict[str, Union[str, int, float]]], str] = Field(
        ..., description="Batch data or CSV string"
    )
    run_id: str = Field(..., description="Run ID of the trained model")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    {"feature_0": 1.5, "feature_1": 2.3, "feature_2": 0.8},
                    {"feature_0": 2.1, "feature_1": 1.7, "feature_2": 1.2},
                ],
                "run_id": "run_20241201_143022_abc123",
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    predictions: List[Union[int, float, List[float]]] = Field(
        ..., description="Batch predictions"
    )
    confidences: Optional[List[float]] = Field(
        None, description="Prediction confidences"
    )
    probabilities: Optional[List[List[float]]] = Field(
        None, description="Class probabilities"
    )
    model_type: Optional[str] = Field(None, description="Model type used")
    n_predictions: int = Field(..., description="Number of predictions made")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [1, 0],
                "confidences": [0.85, 0.92],
                "probabilities": [[0.15, 0.85], [0.92, 0.08]],
                "model_type": "random_forest",
                "n_predictions": 2,
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Current timestamp")
    run_id: Optional[str] = Field(None, description="Current run ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-12-01T14:30:22Z",
                "run_id": "run_20241201_143022_abc123",
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    run_id: str = Field(..., description="Run ID")
    model_type: str = Field(..., description="Model type")
    task_type: str = Field(..., description="Task type")
    feature_names: List[str] = Field(..., description="Feature names")
    n_features: int = Field(..., description="Number of features")
    performance_metrics: Dict[str, float] = Field(
        ..., description="Performance metrics"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": "run_20241201_143022_abc123",
                "model_type": "random_forest",
                "task_type": "classification",
                "feature_names": ["feature_0", "feature_1", "feature_2"],
                "n_features": 3,
                "performance_metrics": {
                    "accuracy": 0.85,
                    "f1": 0.82,
                    "precision": 0.80,
                    "recall": 0.84,
                },
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Model not found",
                "detail": "Run ID 'invalid_run' does not exist",
                "timestamp": "2024-12-01T14:30:22Z",
            }
        }
    )
