"""
Utility functions for the Autonomous ML Agent.
"""

import hashlib
import json
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .logging import get_logger
from .types import DatasetProfile, MetricType, TaskType

logger = get_logger()


def generate_run_id() -> str:
    """Generate unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"run_{timestamp}_{random_suffix}"


def calculate_data_hash(data: pd.DataFrame) -> str:
    """Calculate hash of dataset for meta-learning."""
    # Convert to string representation for hashing
    data_str = data.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()


def detect_task_type(y: pd.Series, threshold: float = 0.05) -> TaskType:
    """
    Detect task type from target variable.

    Args:
        y: Target variable
        threshold: Threshold for considering as classification

    Returns:
        Detected task type
    """
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(y):
        return TaskType.CLASSIFICATION

    # Check if integer-like
    if y.dtype in [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]:
        unique_vals = y.nunique()
        if unique_vals <= 20:  # Likely classification
            return TaskType.CLASSIFICATION
        else:
            return TaskType.REGRESSION

    # Check if float with few unique values (classification)
    unique_vals = y.nunique()
    if unique_vals <= 20 and unique_vals / len(y) < threshold:
        return TaskType.CLASSIFICATION

    return TaskType.REGRESSION


def select_metric(
    task_type: TaskType, metric: MetricType = MetricType.AUTO
) -> MetricType:
    """
    Select appropriate metric based on task type.

    Args:
        task_type: Task type
        metric: Metric preference

    Returns:
        Selected metric type
    """
    if metric != MetricType.AUTO:
        return metric

    if task_type == TaskType.CLASSIFICATION:
        return MetricType.F1
    else:
        return MetricType.R2


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.CLASSIFICATION,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for model evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for classification)
        task_type: Task type

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if task_type == TaskType.CLASSIFICATION:
        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # AUC if probabilities available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:  # Multiclass
                    metrics["auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0

    else:
        # Regression metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


def profile_dataset(
    data: pd.DataFrame, target_column: Optional[str] = None
) -> DatasetProfile:
    """
    Profile dataset to extract metadata.

    Args:
        data: Input dataset
        target_column: Target column name

    Returns:
        Dataset profile
    """
    n_rows, n_cols = data.shape

    # Count column types
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

    # Text columns (short strings that might be text)
    text_cols = []
    for col in categorical_cols:
        if data[col].dtype == "object":
            avg_length = data[col].astype(str).str.len().mean()
            if avg_length > 20:  # Arbitrary threshold
                text_cols.append(col)

    # Remove text columns from categorical
    categorical_cols = [col for col in categorical_cols if col not in text_cols]

    # Calculate missing ratio
    missing_ratio = data.isnull().sum().sum() / (n_rows * n_cols)

    # Calculate class balance if target is specified
    class_balance = None
    if target_column and target_column in data.columns:
        target_series = data[target_column]
        if detect_task_type(target_series) == TaskType.CLASSIFICATION:
            class_counts = target_series.value_counts()
            class_balance = class_counts.min() / class_counts.max()

    # Determine task type
    task_type = None
    if target_column and target_column in data.columns:
        task_type = detect_task_type(data[target_column])

    # Feature columns (exclude target)
    feature_columns = [col for col in data.columns if col != target_column]

    # Calculate data hash
    data_hash = calculate_data_hash(data)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        n_datetime=len(datetime_cols),
        n_text=len(text_cols),
        missing_ratio=missing_ratio,
        class_balance=class_balance,
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns,
        data_hash=data_hash,
    )


def create_artifacts_dir(run_id: str, base_dir: str = "artifacts") -> Path:
    """
    Create artifacts directory for run.

    Args:
        run_id: Run ID
        base_dir: Base artifacts directory

    Returns:
        Path to artifacts directory
    """
    artifacts_dir = Path(base_dir) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_metadata(metadata: Dict[str, Any], artifacts_dir: Path) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        artifacts_dir: Artifacts directory
    """
    metadata_file = artifacts_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON file.

    Args:
        artifacts_dir: Artifacts directory

    Returns:
        Metadata dictionary
    """
    metadata_file = artifacts_dir / "metadata.json"
    if not metadata_file.exists():
        return {}

    with open(metadata_file, "r") as f:
        return json.load(f)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp_probabilities(probs: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Clamp probabilities to valid range [epsilon, 1-epsilon]."""
    return np.clip(probs, epsilon, 1 - epsilon)


def validate_prediction_input(
    data: Union[Dict, pd.DataFrame], expected_columns: List[str]
) -> bool:
    """
    Validate prediction input data.

    Args:
        data: Input data (dict or DataFrame)
        expected_columns: Expected column names

    Returns:
        True if valid
    """
    if isinstance(data, dict):
        data_columns = set(data.keys())
    else:
        data_columns = set(data.columns)

    expected_columns_set = set(expected_columns)

    # Check if all expected columns are present
    missing_columns = expected_columns_set - data_columns
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
        return False

    return True


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_score(score: float, metric: str) -> str:
    """Format score for display."""
    if metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "r2",
        "balanced_accuracy",
    ]:
        return f"{score:.4f}"
    elif metric in ["mae", "mse", "rmse"]:
        return f"{score:.6f}"
    else:
        return f"{score:.4f}"


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from various formats.

    Args:
        file_path: Path to data file
        **kwargs: Additional arguments for pandas readers

    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif suffix == ".json":
            return pd.read_json(file_path, **kwargs)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path, **kwargs)
        elif suffix == ".feather":
            return pd.read_feather(file_path, **kwargs)
        elif suffix == ".pickle":
            return pd.read_pickle(file_path, **kwargs)
        else:
            # Try to infer format from content
            return _infer_and_load(file_path, **kwargs)

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def _infer_and_load(file_path: Path, **kwargs) -> pd.DataFrame:
    """Infer format and load data."""
    try:
        # Try CSV first
        return pd.read_csv(file_path, **kwargs)
    except Exception:
        try:
            # Try JSON
            return pd.read_json(file_path, **kwargs)
        except Exception:
            try:
                # Try Parquet
                return pd.read_parquet(file_path, **kwargs)
            except Exception:
                raise ValueError(
                    f"Unable to load file {file_path}. "
                    "Supported formats: CSV, JSON, Parquet, Excel, Feather, Pickle"
                )


def save_data(data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Save data to various formats.

    Args:
        data: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for pandas writers
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if suffix == ".csv":
            data.to_csv(file_path, index=False, **kwargs)
        elif suffix == ".json":
            data.to_json(file_path, orient="records", **kwargs)
        elif suffix == ".parquet":
            data.to_parquet(file_path, index=False, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            data.to_excel(file_path, index=False, **kwargs)
        elif suffix == ".feather":
            data.to_feather(file_path, **kwargs)
        elif suffix == ".pickle":
            data.to_pickle(file_path, **kwargs)
        else:
            # Default to CSV
            data.to_csv(file_path, index=False, **kwargs)

        logger.info(f"Data saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise


def create_sample_data(
    n_samples: int = 100,
    n_features: int = 5,
    task_type: TaskType = TaskType.CLASSIFICATION,
) -> pd.DataFrame:
    """
    Create sample dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: Task type

    Returns:
        Sample DataFrame
    """
    np.random.seed(42)

    # Generate features
    feature_data = {}
    for i in range(n_features):
        feature_data[f"feature_{i}"] = np.random.randn(n_samples)

    # Generate target
    if task_type == TaskType.CLASSIFICATION:
        # Binary classification - use int64 for proper classification
        target = np.random.randint(0, 2, n_samples).astype(np.int64)
    else:
        # Regression
        target = np.random.randn(n_samples).astype(np.float64)

    feature_data["target"] = target

    return pd.DataFrame(feature_data)
