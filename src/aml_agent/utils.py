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
    feature_data: dict[str, np.ndarray] = {}
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


def create_ai_enhanced_sample_data(
    n_samples: int = 1000,
    n_features: int = 10,
    task_type: TaskType = TaskType.CLASSIFICATION,
    dataset_theme: str = "customer_analytics",
) -> pd.DataFrame:
    """
    Create AI-enhanced sample dataset with realistic features and names.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: Task type
        dataset_theme: Theme for the dataset (e.g., 'customer_analytics', 'financial', 'medical')

    Returns:
        Enhanced sample DataFrame with realistic column names and data
    """
    import os
    import random
    from typing import Optional

    np.random.seed(42)
    random.seed(42)

    # Define realistic feature names based on theme
    feature_templates = {
        "customer_analytics": [
            "age",
            "income",
            "spending_score",
            "credit_score",
            "tenure_months",
            "purchase_frequency",
            "avg_order_value",
            "last_login_days",
            "support_tickets",
            "satisfaction_rating",
            "loyalty_points",
            "referral_count",
            "device_type",
            "location_score",
            "engagement_score",
        ],
        "financial": [
            "account_balance",
            "credit_limit",
            "monthly_income",
            "debt_ratio",
            "payment_history",
            "credit_inquiries",
            "loan_amount",
            "interest_rate",
            "employment_years",
            "home_ownership",
            "loan_default_risk",
            "savings_rate",
            "investment_amount",
            "risk_tolerance",
            "financial_stress",
        ],
        "medical": [
            "age",
            "bmi",
            "blood_pressure",
            "cholesterol",
            "glucose_level",
            "exercise_hours",
            "sleep_hours",
            "stress_level",
            "family_history",
            "medication_count",
            "hospital_visits",
            "symptoms_count",
            "vital_signs",
            "lab_results",
            "treatment_duration",
        ],
        "ecommerce": [
            "product_rating",
            "price_sensitivity",
            "brand_loyalty",
            "seasonal_demand",
            "inventory_level",
            "marketing_spend",
            "conversion_rate",
            "bounce_rate",
            "session_duration",
            "cart_abandonment",
            "return_rate",
            "shipping_cost",
            "discount_usage",
            "repeat_purchase",
            "social_media_engagement",
        ],
    }

    # Get feature names for the theme
    available_features = feature_templates.get(
        dataset_theme, feature_templates["customer_analytics"]
    )
    selected_features = random.sample(
        available_features, min(n_features, len(available_features))
    )

    # Generate realistic data based on feature names
    feature_data = {}

    for feature in selected_features:
        if "age" in feature.lower():
            # Age: normally distributed around 35-45
            feature_data[feature] = np.random.normal(40, 15, n_samples).astype(int)
            feature_data[feature] = np.clip(feature_data[feature], 18, 80)
        elif (
            "income" in feature.lower()
            or "balance" in feature.lower()
            or "amount" in feature.lower()
        ):
            # Financial amounts: log-normal distribution
            feature_data[feature] = np.random.lognormal(8, 1, n_samples).astype(int)
        elif "score" in feature.lower() or "rating" in feature.lower():
            # Scores: uniform distribution 0-100
            feature_data[feature] = np.random.uniform(0, 100, n_samples).astype(int)
        elif (
            "count" in feature.lower()
            or "visits" in feature.lower()
            or "tickets" in feature.lower()
        ):
            # Counts: Poisson distribution
            feature_data[feature] = np.random.poisson(5, n_samples).astype(int)
        elif "ratio" in feature.lower() or "rate" in feature.lower():
            # Ratios: beta distribution
            feature_data[feature] = np.random.beta(2, 5, n_samples).astype(float)
        elif "level" in feature.lower() or "hours" in feature.lower():
            # Levels: normal distribution
            feature_data[feature] = np.random.normal(50, 20, n_samples).astype(float)
        else:
            # Default: normal distribution
            feature_data[feature] = np.random.normal(0, 1, n_samples)

    # Generate target based on task type
    if task_type == TaskType.CLASSIFICATION:
        # Create more realistic classification target
        if dataset_theme == "customer_analytics":
            # Churn prediction: based on multiple features
            churn_score = np.zeros(n_samples)
            for feature, values in feature_data.items():
                if "satisfaction" in feature.lower() or "rating" in feature.lower():
                    churn_score += (100 - values) * 0.3
                elif "tickets" in feature.lower() or "complaints" in feature.lower():
                    churn_score += values * 0.2
                elif "tenure" in feature.lower() or "loyalty" in feature.lower():
                    churn_score -= values * 0.1

            # Convert to binary classification
            target = (churn_score > np.percentile(churn_score, 70)).astype(int)
        else:
            # Generic binary classification
            target = np.random.randint(0, 2, n_samples)
    else:
        # Regression target
        if dataset_theme == "financial":
            # Price prediction based on features
            target = np.random.normal(1000, 300, n_samples)
        else:
            # Generic regression
            target = np.random.normal(0, 1, n_samples)

    feature_data["target"] = target.astype(
        np.int64 if task_type == TaskType.CLASSIFICATION else np.float64
    )

    # Add some realistic noise and correlations
    df = pd.DataFrame(feature_data)

    # Add some missing values for realism
    missing_ratio = 0.05
    for col in df.columns:
        if col != "target":
            mask = np.random.random(len(df)) < missing_ratio
            df.loc[mask, col] = np.nan

    return df


def generate_ai_dataset_description(
    dataset_theme: str = "customer_analytics",
    task_type: TaskType = TaskType.CLASSIFICATION,
    n_samples: int = 1000,
    n_features: int = 10,
) -> str:
    """
    Generate AI-powered dataset description using available APIs.

    Args:
        dataset_theme: Theme for the dataset
        task_type: Task type
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        Generated dataset description
    """
    import os

    # Try to use AI APIs if available
    try:
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            import openai

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""
            Generate a brief, professional description for a {task_type.value} dataset with the following characteristics:
            - Theme: {dataset_theme}
            - Samples: {n_samples}
            - Features: {n_features}
            
            The description should be 2-3 sentences explaining what this dataset represents and its potential use cases.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            content = response.choices[0].message.content
            return content.strip() if content else "Generated dataset description"

    except Exception as e:
        print(f"OpenAI API error: {e}")

    try:
        # Try Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-pro")

            prompt = f"Generate a brief description for a {task_type.value} dataset with {dataset_theme} theme, {n_samples} samples, and {n_features} features."

            response = model.generate_content(prompt)  # type: ignore
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                return "Generated dataset description"

    except Exception as e:
        print(f"Google Gemini API error: {e}")

    # Fallback to static descriptions
    descriptions = {
        "customer_analytics": f"This {task_type.value} dataset contains customer analytics data with {n_samples} samples and {n_features} features. It's designed for predicting customer behavior, churn analysis, and segmentation tasks.",
        "financial": f"This {task_type.value} dataset contains financial data with {n_samples} samples and {n_features} features. It's suitable for credit risk assessment, fraud detection, and financial modeling.",
        "medical": f"This {task_type.value} dataset contains medical data with {n_samples} samples and {n_features} features. It's designed for health outcome prediction, diagnosis assistance, and medical research.",
        "ecommerce": f"This {task_type.value} dataset contains e-commerce data with {n_samples} samples and {n_features} features. It's suitable for sales forecasting, recommendation systems, and market analysis.",
    }

    return descriptions.get(dataset_theme, descriptions["customer_analytics"])
