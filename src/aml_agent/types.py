"""
Type definitions for the Autonomous ML Agent.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TaskType(str, Enum):
    """Task type enumeration."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class MetricType(str, Enum):
    """Metric type enumeration."""

    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    AUC = "auc"
    BALANCED_ACCURACY = "balanced_accuracy"

    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"

    # Auto selection
    AUTO = "auto"


class SearchStrategy(str, Enum):
    """Hyperparameter search strategy."""

    RANDOM = "random"
    BAYES = "bayes"


class ModelType(str, Enum):
    """Model type enumeration."""

    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    KNN = "knn"
    MLP = "mlp"

    # Advanced ML Models
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class ImputationMethod(str, Enum):
    """Imputation method enumeration."""

    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    MOST_FREQUENT = "most_frequent"


class EncodingMethod(str, Enum):
    """Categorical encoding method."""

    ONEHOT = "onehot"
    TARGET = "target"
    ORDINAL = "ordinal"


class OutlierMethod(str, Enum):
    """Outlier detection method."""

    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class DatasetProfile:
    """Dataset profiling information."""

    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    missing_ratio: float
    class_balance: Optional[float] = None
    task_type: Optional[TaskType] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = None
    data_hash: Optional[str] = None


@dataclass
class RunMetadata:
    """Metadata for a training run."""

    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None
    dataset_profile: Optional[DatasetProfile] = None
    total_trials: int = 0
    best_score: Optional[float] = None
    best_model: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed", "timeout"


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    trial_id: int
    model_type: ModelType
    params: Dict[str, Any]
    score: float
    metric: MetricType
    cv_scores: List[float]
    fit_time: float
    predict_time: float
    timestamp: datetime
    status: str = "completed"  # "completed", "failed", "timeout"


@dataclass
class LeaderboardEntry:
    """Entry in the model leaderboard."""

    rank: int
    model_type: ModelType
    score: float
    metric: MetricType
    params: Dict[str, Any]
    cv_mean: float
    cv_std: float
    fit_time: float
    predict_time: float
    trial_id: int


@dataclass
class ModelCard:
    """Model card information."""

    model_name: str
    model_type: ModelType
    task_type: TaskType
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    top_features: List[str] = None
    limitations: List[str] = None
    recommendations: List[str] = None
    created_at: datetime = None


@dataclass
class PredictionRequest:
    """Single prediction request."""

    features: Dict[str, Any]
    run_id: str


@dataclass
class PredictionResponse:
    """Prediction response."""

    prediction: Union[int, float, List[float]]
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_type: Optional[str] = None


@dataclass
class BatchPredictionRequest:
    """Batch prediction request."""

    data: Union[pd.DataFrame, str]  # DataFrame or CSV string
    run_id: str


@dataclass
class BatchPredictionResponse:
    """Batch prediction response."""

    predictions: List[Union[int, float, List[float]]]
    confidences: Optional[List[float]] = None
    probabilities: Optional[List[List[float]]] = None
    model_type: Optional[str] = None


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""

    method: str  # "stacking", "blending", "voting"
    top_k: int
    meta_learner: Optional[str] = None
    weights: Optional[List[float]] = None


@dataclass
class BudgetClock:
    """Budget tracking for time-limited optimization."""

    start_time: datetime
    time_budget_seconds: float
    elapsed_seconds: float = 0.0

    def is_expired(self) -> bool:
        """Check if budget is expired."""
        return self.elapsed_seconds >= self.time_budget_seconds

    def remaining_seconds(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.time_budget_seconds - self.elapsed_seconds)

    def update_elapsed(self) -> None:
        """Update elapsed time."""
        self.elapsed_seconds = (datetime.now() - self.start_time).total_seconds()


@dataclass
class LLMConfig:
    """LLM configuration for planning and model cards."""

    enabled: bool = False
    provider: str = "openai"  # "openai", "gemini"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    handle_missing: bool = True
    impute_numeric: ImputationMethod = ImputationMethod.MEDIAN
    impute_categorical: ImputationMethod = ImputationMethod.MOST_FREQUENT
    encode_categorical: EncodingMethod = EncodingMethod.ONEHOT
    scale_features: bool = True
    handle_outliers: bool = True
    outlier_method: OutlierMethod = OutlierMethod.IQR
    datetime_expansion: bool = True


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    enabled: bool = True
    class_weight: Optional[str] = None
    early_stopping: bool = False
    validation_fraction: float = 0.1


@dataclass
class SearchSpace:
    """Hyperparameter search space definition."""

    model_type: ModelType
    space: Dict[str, Any]
    fixed_params: Dict[str, Any] = None


@dataclass
class PlannerProposal:
    """LLM planner proposal for optimization strategy."""

    candidate_models: List[ModelType]
    search_budgets: Dict[ModelType, int]
    metric: MetricType
    ensemble_strategy: Optional[EnsembleConfig] = None
    reasoning: Optional[str] = None
