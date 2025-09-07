"""
FastAPI service for serving ML models.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..export.artifact import ArtifactExporter
from ..logging import get_logger
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger()

# Global variables for loaded model
loaded_pipeline = None
loaded_preprocessor = None
loaded_metadata = None
artifact_exporter = ArtifactExporter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting Autonomous ML Agent API service")
    yield
    logger.info("Shutting down Autonomous ML Agent API service")


app = FastAPI(
    title="Autonomous ML Agent API",
    description="API for serving machine learning models trained by the Autonomous ML Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", message="Autonomous ML Agent API is running", version="1.0.0"
    )


@app.get("/healthz")
async def healthz():
    """Kubernetes-style health check."""
    return {"status": "ok"}


@app.post("/load_model")
async def load_model(run_id: str = Form(...)):
    """Load a model from artifacts."""
    global loaded_pipeline, loaded_preprocessor, loaded_metadata

    try:
        # Load pipeline (model + preprocessor)
        loaded_pipeline = artifact_exporter.load_pipeline(run_id)

        # Also load the preprocessor separately if available
        try:
            loaded_preprocessor = artifact_exporter.load_preprocessor(run_id)
            logger.info(f"Preprocessor loaded successfully: {run_id}")
        except FileNotFoundError:
            loaded_preprocessor = None
            logger.info(f"No separate preprocessor found for: {run_id}")

        # Load metadata
        loaded_metadata = artifact_exporter.load_metadata(run_id)

        logger.info(f"Model loaded successfully: {run_id}")
        return {"message": f"Model {run_id} loaded successfully", "run_id": run_id}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model."""
    if loaded_pipeline is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        # Get pipeline info
        pipeline_info = {
            "model_type": type(loaded_pipeline).__name__,
            "n_steps": (
                len(loaded_pipeline.steps) if hasattr(loaded_pipeline, "steps") else 0
            ),
            "steps": (
                [step[0] for step in loaded_pipeline.steps]
                if hasattr(loaded_pipeline, "steps")
                else []
            ),
        }

        # Combine with metadata
        info = {
            "pipeline_info": pipeline_info,
            "metadata": loaded_metadata or {},
            "loaded": True,
        }

        return ModelInfoResponse(**info)

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a single prediction."""
    if loaded_pipeline is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.data])

        # Apply preprocessor if available
        if loaded_preprocessor is not None:
            data = loaded_preprocessor.transform(data)

        # Make prediction
        prediction = loaded_pipeline.predict(data)[0]

        # Get probabilities if available
        probabilities = None
        if hasattr(loaded_pipeline, "predict_proba"):
            proba = loaded_pipeline.predict_proba(data)[0]
            probabilities = proba.tolist()

        return PredictionResponse(
            prediction=float(prediction),
            probabilities=probabilities,
            model_info={
                "run_id": (
                    loaded_metadata.get("run_id", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
                "model_type": type(loaded_pipeline).__name__,
                "task_type": (
                    loaded_metadata.get("task_type", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
            },
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if loaded_pipeline is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame(request.data)

        # Make predictions
        predictions = loaded_pipeline.predict(data).tolist()

        # Get probabilities if available
        probabilities = None
        if hasattr(loaded_pipeline, "predict_proba"):
            proba = loaded_pipeline.predict_proba(data)
            probabilities = proba.tolist()

        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            n_predictions=len(predictions),
            model_info={
                "run_id": (
                    loaded_metadata.get("run_id", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
                "model_type": type(loaded_pipeline).__name__,
                "task_type": (
                    loaded_metadata.get("task_type", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
            },
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file."""
    if loaded_pipeline is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    try:
        # Read uploaded file
        content = await file.read()
        data = pd.read_csv(pd.io.common.StringIO(content.decode("utf-8")))

        # Make predictions
        predictions = loaded_pipeline.predict(data).tolist()

        # Get probabilities if available
        probabilities = None
        if hasattr(loaded_pipeline, "predict_proba"):
            proba = loaded_pipeline.predict_proba(data)
            probabilities = proba.tolist()

        # Prepare response data
        response_data = {
            "predictions": predictions,
            "probabilities": probabilities,
            "n_predictions": len(predictions),
            "model_info": {
                "run_id": (
                    loaded_metadata.get("run_id", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
                "model_type": type(loaded_pipeline).__name__,
                "task_type": (
                    loaded_metadata.get("task_type", "unknown")
                    if loaded_metadata
                    else "unknown"
                ),
            },
        }

        # Add predictions to original data
        data_with_predictions = data.copy()
        data_with_predictions["prediction"] = predictions

        if probabilities is not None:
            for i, prob in enumerate(probabilities[0] if probabilities else []):
                data_with_predictions[f"probability_class_{i}"] = [
                    p[i] for p in probabilities
                ]

        # Convert to CSV
        # csv_response = ...  # unused

        return JSONResponse(
            content=response_data,
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


@app.get("/available_models")
async def list_available_models():
    """List all available models in artifacts."""
    try:
        artifacts = artifact_exporter.list_artifacts()
        return {"models": artifacts}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/model/{run_id}/info")
async def get_model_info_by_id(run_id: str):
    """Get information about a specific model by run ID."""
    try:
        metadata = artifact_exporter.load_metadata(run_id)
        return {"run_id": run_id, "metadata": metadata}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {run_id} not found")
    except Exception as e:
        logger.error(f"Failed to get model info for {run_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/model/{run_id}/load")
async def load_model_by_id(run_id: str):
    """Load a specific model by run ID."""
    return await load_model(run_id)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Autonomous ML Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "predict_file": "/predict_file",
            "model_info": "/model_info",
            "available_models": "/available_models",
            "load_model": "/load_model",
        },
        "documentation": "/docs",
    }


def create_app(artifacts_dir: Path) -> FastAPI:
    """Create FastAPI app with loaded model from artifacts directory."""
    global loaded_pipeline, loaded_metadata

    # Load model and metadata from artifacts
    model_path = artifacts_dir / "model.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    if model_path.exists():
        import joblib

        loaded_pipeline = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    if metadata_path.exists():
        import json

        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
