"""
Data transformers for preprocessing pipeline.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from ..logging import get_logger
from ..types import EncodingMethod, ImputationMethod

logger = get_logger()


class ImputationTransformer(BaseEstimator, TransformerMixin):
    """Handle missing value imputation for both numeric and categorical data."""

    def __init__(
        self,
        numeric_strategy: ImputationMethod = ImputationMethod.MEDIAN,
        categorical_strategy: ImputationMethod = ImputationMethod.MOST_FREQUENT,
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputation transformers."""
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Fit numeric imputer
        if numeric_cols:
            strategy_map = {
                ImputationMethod.MEDIAN: "median",
                ImputationMethod.MEAN: "mean",
                ImputationMethod.MODE: "most_frequent",
            }
            strategy = strategy_map.get(self.numeric_strategy, "median")
            self.numeric_imputer = SimpleImputer(strategy=strategy)
            if self.numeric_imputer is not None:
                self.numeric_imputer.fit(X[numeric_cols])

        # Fit categorical imputer
        if categorical_cols:
            strategy_map = {
                ImputationMethod.MOST_FREQUENT: "most_frequent",
                ImputationMethod.MODE: "most_frequent",
            }
            strategy = strategy_map.get(self.categorical_strategy, "most_frequent")
            self.categorical_imputer = SimpleImputer(strategy=strategy)
            if self.categorical_imputer is not None:
                self.categorical_imputer.fit(X[categorical_cols])

        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        # Transform numeric columns
        if self.numeric_imputer is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X_transformed[numeric_cols] = self.numeric_imputer.transform(
                    X[numeric_cols]
                )

        # Transform categorical columns
        if self.categorical_imputer is not None:
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if categorical_cols:
                X_transformed[categorical_cols] = self.categorical_imputer.transform(
                    X[categorical_cols]
                )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables using various methods."""

    def __init__(
        self,
        method: EncodingMethod = EncodingMethod.ONEHOT,
        max_categories: int = 50,
        handle_unknown: str = "ignore",
    ):
        self.method = method
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.encoders: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders for categorical columns."""
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            if self.method == EncodingMethod.ONEHOT:
                # Filter categories if too many
                unique_vals = X[col].value_counts()
                if len(unique_vals) > self.max_categories:
                    top_categories = unique_vals.head(
                        self.max_categories
                    ).index.tolist()
                    X[col] = X[col].where(X[col].isin(top_categories), "other")

                self.encoders[col] = OneHotEncoder(
                    handle_unknown=(
                        "ignore" if self.handle_unknown == "ignore" else "error"
                    ),
                    sparse_output=False,
                )
                self.encoders[col].fit(X[[col]])

            elif self.method == EncodingMethod.ORDINAL:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].astype(str))

        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            if col in self.encoders:
                if self.method == EncodingMethod.ONEHOT:
                    # Get one-hot encoded columns
                    encoded = self.encoders[col].transform(X[[col]])
                    feature_names = self.encoders[col].get_feature_names_out([col])

                    # Create DataFrame with encoded columns
                    encoded_df = pd.DataFrame(
                        encoded, columns=feature_names, index=X.index
                    )

                    # Drop original column and add encoded ones
                    X_transformed = X_transformed.drop(columns=[col])
                    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

                elif self.method == EncodingMethod.ORDINAL:
                    # Replace with encoded values
                    X_transformed[col] = self.encoders[col].transform(
                        X[col].astype(str)
                    )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Expand datetime columns into multiple features."""

    def __init__(self):
        self.datetime_columns = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit datetime expander."""
        self.datetime_columns = X.select_dtypes(include=["datetime64"]).columns.tolist()
        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime columns."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        for col in self.datetime_columns:
            if col in X.columns:
                # Extract datetime components
                X_transformed[f"{col}_year"] = X[col].dt.year
                X_transformed[f"{col}_month"] = X[col].dt.month
                X_transformed[f"{col}_day"] = X[col].dt.day
                X_transformed[f"{col}_dayofweek"] = X[col].dt.dayofweek
                X_transformed[f"{col}_hour"] = X[col].dt.hour
                X_transformed[f"{col}_minute"] = X[col].dt.minute

                # Drop original datetime column
                X_transformed = X_transformed.drop(columns=[col])

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale features using various scaling methods."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self.scaler: Optional[Any] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler."""
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        if self.scaler is not None:
            self.scaler.fit(X)
        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers in numeric data."""

    def __init__(self, method: str = "clip", outlier_indices: Optional[list] = None):
        self.method = method
        self.outlier_indices = outlier_indices if outlier_indices is not None else []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit outlier handler."""
        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        if self.method == "clip":
            # Clip outliers to IQR bounds
            for col in X.select_dtypes(include=[np.number]).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X_transformed[col] = X_transformed[col].clip(lower_bound, upper_bound)

        elif self.method == "remove":
            # Remove outlier rows
            if self.outlier_indices and len(self.outlier_indices) > 0:
                outlier_mask = X.index.isin(self.outlier_indices)
                X_transformed = X_transformed[~outlier_mask]

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from text columns."""

    def __init__(self, max_features: int = 100, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer: Optional[Any] = None
        self.text_columns: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit text feature extractor."""
        # Identify text columns (long string columns)
        self.text_columns = []
        for col in X.columns:
            if X[col].dtype == "object":
                avg_length = X[col].astype(str).str.len().mean()
                if avg_length > 20:  # Arbitrary threshold
                    self.text_columns.append(col)

        if self.text_columns:
            # Combine all text columns
            text_data = (
                X[self.text_columns].fillna("").astype(str).agg(" ".join, axis=1)
            )

            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                stop_words="english",
            )
            if self.vectorizer is not None:
                self.vectorizer.fit(text_data)

        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text columns."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        if self.text_columns and self.vectorizer is not None:
            # Combine text columns
            text_data = (
                X[self.text_columns].fillna("").astype(str).agg(" ".join, axis=1)
            )

            # Extract features
            text_features = self.vectorizer.transform(text_data)
            feature_names = [f"text_feature_{i}" for i in range(text_features.shape[1])]

            # Create DataFrame with text features
            text_df = pd.DataFrame(
                text_features.toarray(), columns=feature_names, index=X.index
            )

            # Remove original text columns and add features
            X_transformed = X_transformed.drop(columns=self.text_columns)
            X_transformed = pd.concat([X_transformed, text_df], axis=1)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select most important features."""

    def __init__(self, method: str = "variance", k: int = 100):
        self.method = method
        self.k = k
        self.selector: Optional[Any] = None
        self.selected_features: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit feature selector."""
        if self.method == "variance":
            from sklearn.feature_selection import VarianceThreshold

            self.selector = VarianceThreshold(threshold=0.01)
            if self.selector is not None:
                self.selector.fit(X)
                self.selected_features = X.columns[self.selector.get_support()].tolist()

        elif self.method == "k_best":
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression

            # Determine scoring function based on task type
            if y is not None:
                if len(np.unique(y)) < 10:  # Classification
                    score_func = f_classif
                else:  # Regression
                    score_func = f_regression
            else:
                score_func = f_classif

            self.selector = SelectKBest(
                score_func=score_func, k=min(self.k, X.shape[1])
            )
            if self.selector is not None:
                self.selector.fit(X, y)
                self.selected_features = X.columns[self.selector.get_support()].tolist()

        self.is_fitted = True
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        if self.selector is not None:
            X_selected = self.selector.transform(X)
            return pd.DataFrame(
                X_selected, columns=self.selected_features, index=X.index
            )
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before get_feature_names_out")

        if input_features is None:
            return self.feature_names_in_
        else:
            return input_features

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X).values
