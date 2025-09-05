"""
Configuration management for the Autonomous ML Agent.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

from .types import (
    TaskType,
    MetricType,
    SearchStrategy,
    ImputationMethod,
    EncodingMethod,
    OutlierMethod,
    PreprocessingConfig,
    ModelConfig,
    LLMConfig,
    EnsembleConfig,
)


class PreprocessingSettings(BaseModel):
    """Preprocessing configuration settings."""
    handle_missing: bool = True
    impute_numeric: ImputationMethod = ImputationMethod.MEDIAN
    impute_categorical: ImputationMethod = ImputationMethod.MOST_FREQUENT
    encode_categorical: EncodingMethod = EncodingMethod.ONEHOT
    scale_features: bool = True
    handle_outliers: bool = True
    outlier_method: OutlierMethod = OutlierMethod.IQR
    datetime_expansion: bool = True


class ModelSettings(BaseModel):
    """Model configuration settings."""
    logistic_regression: ModelConfig = ModelConfig(class_weight="balanced")
    linear_regression: ModelConfig = ModelConfig()
    random_forest: ModelConfig = ModelConfig(class_weight="balanced")
    gradient_boosting: ModelConfig = ModelConfig()
    knn: ModelConfig = ModelConfig()
    mlp: ModelConfig = ModelConfig(early_stopping=True, validation_fraction=0.1)


class SearchSpaceSettings(BaseModel):
    """Hyperparameter search space settings."""
    logistic_regression: Dict[str, Any] = Field(default_factory=lambda: {
        "C": [0.001, 1000],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["liblinear", "saga"]
    })
    linear_regression: Dict[str, Any] = Field(default_factory=lambda: {
        "fit_intercept": [True, False]
    })
    random_forest: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": [10, 200],
        "max_depth": [3, 20],
        "min_samples_split": [2, 20],
        "min_samples_leaf": [1, 10]
    })
    gradient_boosting: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": [10, 200],
        "learning_rate": [0.01, 0.3],
        "max_depth": [3, 10]
    })
    knn: Dict[str, Any] = Field(default_factory=lambda: {
        "n_neighbors": [3, 20],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    })
    mlp: Dict[str, Any] = Field(default_factory=lambda: {
        "hidden_layer_sizes": [[50], [100], [50, 50], [100, 50]],
        "activation": ["relu", "tanh"],
        "learning_rate": [0.001, 0.1],
        "alpha": [0.0001, 0.1]
    })


class LLMSettings(BaseModel):
    """LLM configuration settings."""
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000


class ExportSettings(BaseModel):
    """Export configuration settings."""
    save_pipeline: bool = True
    save_artifacts: bool = True
    generate_model_card: bool = True
    generate_plots: bool = True
    format: str = "joblib"


class APISettings(BaseModel):
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"


class EnvironmentSettings(BaseSettings):
    """Environment variable settings."""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    mlflow_tracking_uri: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="autonomous-ml-agent", env="MLFLOW_EXPERIMENT_NAME")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    default_data_path: str = Field(default="data/sample.csv", env="DEFAULT_DATA_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Config(BaseModel):
    """Main configuration class."""
    # Core settings
    data_path: str
    target: Optional[str] = None
    task_type: TaskType = TaskType.AUTO
    time_budget_seconds: int = 900
    max_trials: int = 60
    cv_folds: int = 5
    metric: Union[MetricType, str] = MetricType.AUTO
    search_strategy: SearchStrategy = SearchStrategy.BAYES
    enable_ensembling: bool = True
    top_k_for_ensemble: int = 3
    random_seed: int = 42
    use_mlflow: bool = False

    # Sub-configurations
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    search_spaces: SearchSpaceSettings = Field(default_factory=SearchSpaceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    api: APISettings = Field(default_factory=APISettings)

    @validator("data_path")
    def validate_data_path(cls, v):
        """Validate data path exists."""
        if not Path(v).exists():
            raise ValueError(f"Data path {v} does not exist")
        return v

    @validator("time_budget_seconds")
    def validate_time_budget(cls, v):
        """Validate time budget is positive."""
        if v <= 0:
            raise ValueError("Time budget must be positive")
        return v

    @validator("max_trials")
    def validate_max_trials(cls, v):
        """Validate max trials is positive."""
        if v <= 0:
            raise ValueError("Max trials must be positive")
        return v

    @validator("cv_folds")
    def validate_cv_folds(cls, v):
        """Validate CV folds is at least 2."""
        if v < 2:
            raise ValueError("CV folds must be at least 2")
        return v

    @validator("top_k_for_ensemble")
    def validate_top_k_ensemble(cls, v):
        """Validate top K for ensemble is positive."""
        if v <= 0:
            raise ValueError("Top K for ensemble must be positive")
        return v
    
    @validator("metric")
    def validate_metric(cls, v):
        """Convert string metrics to MetricType."""
        if isinstance(v, str):
            if v.lower() == "auto":
                return MetricType.AUTO
            try:
                return MetricType(v.lower())
            except ValueError:
                raise ValueError(f"Invalid metric: {v}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update config from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get model-specific configuration."""
        return getattr(self.models, model_type, ModelConfig())

    def get_search_space(self, model_type: str) -> Dict[str, Any]:
        """Get search space for model type."""
        return getattr(self.search_spaces, model_type, {})

    def is_model_enabled(self, model_type: str) -> bool:
        """Check if model is enabled."""
        model_config = self.get_model_config(model_type)
        return model_config.enabled

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        env_settings = EnvironmentSettings()
        return LLMConfig(
            enabled=self.llm.enabled,
            provider=self.llm.provider,
            model=self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            api_key=getattr(env_settings, f"{self.llm.provider}_api_key", None)
        )


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Load environment settings
    env_settings = EnvironmentSettings()
    
    # Override with environment variables
    if env_settings.default_data_path and "data_path" not in config_data:
        config_data["data_path"] = env_settings.default_data_path
    
    # Create config object
    config = Config(**config_data)
    
    return config


def create_default_config() -> Config:
    """Create default configuration."""
    return Config(
        data_path="data/sample.csv",
        target=None,
        task_type=TaskType.AUTO,
        time_budget_seconds=900,
        max_trials=60,
        cv_folds=5,
        metric=MetricType.AUTO,
        search_strategy=SearchStrategy.BAYES,
        enable_ensembling=True,
        top_k_for_ensemble=3,
        random_seed=42,
        use_mlflow=False
    )


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
