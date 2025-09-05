"""
Main preprocessing pipeline for the Autonomous ML Agent.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from ..logging import get_logger
from ..types import PreprocessingConfig
from .detectors import MissingValueDetector, OutlierDetector, TypeDetector
from .transformers import (
    CategoricalEncoder,
    DateTimeExpander,
    FeatureScaler,
    ImputationTransformer,
    OutlierHandler,
)

logger = get_logger()


class PreprocessingPipeline:
    """Complete preprocessing pipeline with intelligent type detection and advanced feature engineering."""

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        use_advanced_features: bool = True,
    ):
        self.config = config or PreprocessingConfig()
        self.use_advanced_features = use_advanced_features
        self.pipeline = None
        self.type_detector = TypeDetector()
        self.missing_detector = MissingValueDetector()
        self.outlier_detector = OutlierDetector(self.config.outlier_method)
        self.feature_names_ = []
        self.target_encoder = None
        self.is_fitted = False

        # Advanced feature engineering components
        self.advanced_pipeline = None
        if self.use_advanced_features:
            # Advanced features temporarily disabled due to feature name mismatch issues
            # TODO: Fix feature name handling in advanced transformers
            self.advanced_pipeline = None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PreprocessingPipeline":
        """
        Fit preprocessing pipeline on training data.

        Args:
            X: Feature DataFrame
            y: Target series (optional)

        Returns:
            Fitted pipeline
        """
        logger.info("Starting preprocessing pipeline fitting")

        # Detect types
        self.type_detector.detect_types(X)

        # Detect missing values
        self.missing_detector.detect_missing(X)

        # Detect outliers in numeric columns
        numeric_columns = self.type_detector.get_columns_by_type("numeric")
        if numeric_columns:
            self.outlier_detector.detect_outliers(X, numeric_columns)

        # Build pipeline
        self._build_pipeline(X, y)

        # Fit pipeline
        self.pipeline.fit(X, y)

        # Apply advanced feature engineering if enabled
        if self.advanced_pipeline is not None:
            logger.info("Applying advanced feature engineering...")
            # Apply advanced features to the preprocessed data
            X_preprocessed = self.pipeline.transform(X)
            X_advanced = self.advanced_pipeline.fit_transform(X_preprocessed, y)
            logger.info(
                f"Advanced features generated. Shape: {X.shape} -> {X_advanced.shape}"
            )

            # Update feature names with advanced features
            self.feature_names_ = self._get_feature_names(X_advanced)
        else:
            # Store feature names
            self.feature_names_ = self._get_feature_names(X)

        # Fit target encoder if needed
        encode_categorical = (
            self.config.encode_categorical.value
            if hasattr(self.config.encode_categorical, "value")
            else self.config.encode_categorical
        )
        if y is not None and encode_categorical == "target":
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y)

        self.is_fitted = True
        logger.info(
            f"Preprocessing pipeline fitted with {len(self.feature_names_)} features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through preprocessing pipeline.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        logger.info("Transforming data through preprocessing pipeline")

        # Apply pipeline
        X_transformed = self.pipeline.transform(X)

        # Convert to DataFrame
        if hasattr(self.pipeline, "get_feature_names_out"):
            feature_names = self.pipeline.get_feature_names_out()
        else:
            feature_names = self.feature_names_

        result = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        # Ensure all columns are numeric
        for col in result.columns:
            if result[col].dtype == "object":
                try:
                    result[col] = pd.to_numeric(result[col], errors="coerce")
                except:
                    pass

        # Fill any remaining NaN values with 0
        result = result.fillna(0)

        # Apply advanced feature engineering if enabled
        if self.advanced_pipeline is not None:
            logger.info("Applying advanced feature engineering to transformed data...")
            result = self.advanced_pipeline.transform(result)
            logger.info(f"Advanced features applied. Final shape: {result.shape}")

        logger.info(f"Transformed data shape: {result.shape}")
        return result

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform data in one step."""
        return self.fit(X, y).transform(X)

    def _build_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Build the preprocessing pipeline."""
        transformers = []

        # Get column types
        numeric_columns = self.type_detector.get_columns_by_type("numeric")
        categorical_columns = self.type_detector.get_columns_by_type("categorical")
        datetime_columns = self.type_detector.get_columns_by_type("datetime")

        # Imputation for numeric columns
        if numeric_columns and self.config.handle_missing:
            numeric_imputer = ImputationTransformer(
                numeric_strategy=self.config.impute_numeric,
                categorical_strategy=self.config.impute_categorical,
            )
            transformers.append(
                ("numeric_imputation", numeric_imputer, numeric_columns)
            )

        # Imputation for categorical columns
        if categorical_columns and self.config.handle_missing:
            categorical_imputer = ImputationTransformer(
                numeric_strategy=self.config.impute_numeric,
                categorical_strategy=self.config.impute_categorical,
            )
            transformers.append(
                ("categorical_imputation", categorical_imputer, categorical_columns)
            )

        # DateTime expansion
        if datetime_columns and self.config.datetime_expansion:
            datetime_expander = DateTimeExpander()
            transformers.append(
                ("datetime_expansion", datetime_expander, datetime_columns)
            )

        # Categorical encoding
        if categorical_columns:
            categorical_encoder = CategoricalEncoder(
                method=self.config.encode_categorical, max_categories=50
            )
            transformers.append(
                ("categorical_encoding", categorical_encoder, categorical_columns)
            )

        # Feature scaling
        if self.config.scale_features:
            # Apply scaling to all numeric columns (will be applied after imputation)
            feature_scaler = FeatureScaler()
            transformers.append(("feature_scaling", feature_scaler, numeric_columns))

        # Outlier handling
        if self.config.handle_outliers and numeric_columns:
            outlier_indices = self.outlier_detector.outlier_indices
            outlier_handler = OutlierHandler(
                method="clip", outlier_indices=outlier_indices
            )
            transformers.append(("outlier_handling", outlier_handler, numeric_columns))

        # Create column transformer
        if transformers:
            self.pipeline = ColumnTransformer(
                transformers=transformers, remainder="passthrough"
            )
        else:
            # If no transformers, create a simple passthrough
            from sklearn.preprocessing import FunctionTransformer

            self.pipeline = FunctionTransformer(lambda x: x)

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation."""
        if hasattr(self.pipeline, "get_feature_names_out"):
            return self.pipeline.get_feature_names_out().tolist()
        else:
            return [f"feature_{i}" for i in range(X.shape[1])]

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        return self.feature_names_

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing steps."""
        return {
            "column_types": self.type_detector.column_types,
            "missing_info": self.missing_detector.missing_info,
            "outlier_info": self.outlier_detector.outlier_info,
            "feature_names": self.feature_names_,
            "is_fitted": self.is_fitted,
        }

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Inverse transform target variable if it was encoded."""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y_encoded)
        return y_encoded

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transform target variable if needed."""
        if self.target_encoder is not None:
            return self.target_encoder.transform(y)
        return y.values


def create_preprocessing_pipeline(
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessingPipeline:
    """Create a preprocessing pipeline with given configuration."""
    return PreprocessingPipeline(config)


def detect_data_types(data: pd.DataFrame) -> Dict[str, str]:
    """Detect data types in a DataFrame."""
    detector = TypeDetector()
    return detector.detect_types(data)


def analyze_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data quality including missing values and outliers."""
    type_detector = TypeDetector()
    missing_detector = MissingValueDetector()
    outlier_detector = OutlierDetector()

    # Detect types
    column_types = type_detector.detect_types(data)

    # Detect missing values
    missing_info = missing_detector.detect_missing(data)

    # Detect outliers
    numeric_columns = type_detector.get_columns_by_type("numeric")
    outlier_info = {}
    if numeric_columns:
        outlier_info = outlier_detector.detect_outliers(data, numeric_columns)

    return {
        "column_types": column_types,
        "missing_info": missing_info,
        "outlier_info": outlier_info,
        "data_shape": data.shape,
        "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
