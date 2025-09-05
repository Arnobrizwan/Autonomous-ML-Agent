"""
FastAPI application for the Autonomous ML Agent.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..logging import get_logger
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger()


class ModelService:
    """Service for managing and serving ML models."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.feature_names = []
        self.is_loaded = False

        # Load model and metadata
        self._load_model()

    def _load_model(self):
        """Load model and metadata from artifacts directory."""
        try:
            # Load model
            model_file = self.artifacts_dir / "model.joblib"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            self.model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")

            # Load preprocessor
            preprocessor_file = self.artifacts_dir / "preprocessor.joblib"
            if preprocessor_file.exists():
                self.preprocessor = joblib.load(preprocessor_file)
                logger.info(f"Loaded preprocessor from {preprocessor_file}")

            # Load metadata
            metadata_file = self.artifacts_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_file}")

            # Load feature names
            feature_names_file = self.artifacts_dir / "feature_names.json"
            if feature_names_file.exists():
                with open(feature_names_file, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded feature names: {self.feature_names}")
            else:
                # Fallback to model attributes
                if hasattr(self.model, "n_features_in_"):
                    self.feature_names = [
                        f"feature_{i}" for i in range(self.model.n_features_in_)
                    ]
                else:
                    self.feature_names = []
                logger.info(f"Using fallback feature names: {self.feature_names}")

            self.is_loaded = True
            logger.info("Model service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, features: Dict[str, Any]) -> PredictionResponse:
        """Make single prediction."""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Validate features
            self._validate_features(df)

            # Preprocess if available
            if self.preprocessor:
                df = self.preprocessor.transform(df)

            # Make prediction
            prediction = self.model.predict(df)[0]

            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(df)[0]
                probabilities = proba.tolist()

            # Calculate confidence
            confidence = None
            if probabilities:
                confidence = float(max(probabilities))

            return PredictionResponse(
                prediction=(
                    float(prediction)
                    if isinstance(prediction, (int, float))
                    else prediction.tolist()
                ),
                confidence=confidence,
                probabilities=probabilities,
                model_type=self.metadata.get("model_type", "unknown"),
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    def predict_batch(
        self, data: Union[List[Dict[str, Any]], str]
    ) -> BatchPredictionResponse:
        """Make batch predictions."""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            # Convert data to DataFrame
            if isinstance(data, str):
                # CSV string
                from io import StringIO

                df = pd.read_csv(StringIO(data))
            else:
                # List of dictionaries
                df = pd.DataFrame(data)

            # Validate features
            self._validate_features(df)

            # Preprocess if available
            if self.preprocessor:
                df = self.preprocessor.transform(df)

            # Make predictions
            predictions = self.model.predict(df)

            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(df)
                probabilities = proba.tolist()

            # Calculate confidences
            confidences = None
            if probabilities:
                confidences = [float(max(probs)) for probs in probabilities]

            # Convert predictions to appropriate format
            predictions_list = []
            for pred in predictions:
                if isinstance(pred, (int, float)):
                    predictions_list.append(float(pred))
                else:
                    predictions_list.append(pred.tolist())

            return BatchPredictionResponse(
                predictions=predictions_list,
                confidences=confidences,
                probabilities=probabilities,
                model_type=self.metadata.get("model_type", "unknown"),
                n_predictions=len(predictions),
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"Batch prediction failed: {str(e)}"
            )

    def _validate_features(self, df: pd.DataFrame):
        """Validate input features."""
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Check for extra features
        extra_features = set(df.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features provided: {extra_features}")

    def get_model_info(self) -> ModelInfoResponse:
        """Get model information."""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        return ModelInfoResponse(
            run_id=self.artifacts_dir.name,
            model_type=self.metadata.get("model_type", "unknown"),
            task_type=self.metadata.get("task_type", "unknown"),
            feature_names=self.feature_names,
            n_features=len(self.feature_names),
            performance_metrics=self.metadata.get("performance_metrics", {}),
        )


def create_app(artifacts_dir: Path) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Autonomous ML Agent API",
        description="Machine learning prediction service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model service
    try:
        model_service = ModelService(artifacts_dir)
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        model_service = None

    @app.get("/healthz", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status=(
                "healthy" if model_service and model_service.is_loaded else "unhealthy"
            ),
            version="0.1.0",
            timestamp=datetime.now().isoformat(),
            run_id=artifacts_dir.name if model_service else None,
        )

    @app.get("/info", response_model=ModelInfoResponse)
    async def get_model_info():
        """Get model information."""
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        return model_service.get_model_info()

    @app.post("/predict_one", response_model=PredictionResponse)
    async def predict_one(request: PredictionRequest):
        """Make single prediction."""
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        return model_service.predict_single(request.features)

    @app.post("/predict_batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make batch predictions."""
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        return model_service.predict_batch(request.data)

    @app.post("/predict_batch_file", response_model=BatchPredictionResponse)
    async def predict_batch_file(file: UploadFile = File(...), run_id: str = Form(...)):
        """Make batch predictions from uploaded CSV file."""
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if run_id != artifacts_dir.name:
            raise HTTPException(status_code=400, detail="Run ID mismatch")

        try:
            # Read CSV file
            contents = await file.read()
            csv_string = contents.decode("utf-8")

            return model_service.predict_batch(csv_string)

        except Exception as e:
            logger.error(f"File prediction failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"File prediction failed: {str(e)}"
            )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail, timestamp=datetime.now().isoformat()
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                timestamp=datetime.now().isoformat(),
            ).dict(),
        )

    return app


# Create default app instance
app = FastAPI(title="Autonomous ML Agent API", version="0.1.0")
