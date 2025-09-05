"""
Data transformers for preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,
    TargetEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..types import ImputationMethod, EncodingMethod, OutlierMethod
from ..logging import get_logger

logger = get_logger()


class ImputationTransformer(BaseEstimator, TransformerMixin):
    """Handle missing values with different strategies."""
    
    def __init__(self, 
                 numeric_strategy: ImputationMethod = ImputationMethod.MEDIAN,
                 categorical_strategy: ImputationMethod = ImputationMethod.MOST_FREQUENT):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers on training data."""
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        
        # Detect column types
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create imputers
        if self.numeric_columns:
            self.numeric_imputer = SimpleImputer(
                strategy=self.numeric_strategy.value,
                keep_empty_features=True
            )
            self.numeric_imputer.fit(X[self.numeric_columns])
        
        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(
                strategy=self.categorical_strategy.value,
                keep_empty_features=True
            )
            self.categorical_imputer.fit(X[self.categorical_columns])
        
        # Output feature names are the same as input for imputation
        self.feature_names_out_ = self.feature_names_in_.copy()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        X_transformed = X.copy()
        
        if self.numeric_imputer and self.numeric_columns:
            X_transformed[self.numeric_columns] = self.numeric_imputer.transform(
                X[self.numeric_columns]
            )
        
        if self.categorical_imputer and self.categorical_columns:
            X_transformed[self.categorical_columns] = self.categorical_imputer.transform(
                X[self.categorical_columns]
            )
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        elif input_features is not None:
            return list(input_features)
        else:
            return [f"feature_{i}" for i in range(len(self.feature_names_in_))]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables."""
    
    def __init__(self, 
                 method: EncodingMethod = EncodingMethod.ONEHOT,
                 max_categories: int = 50,
                 target_encoder_cv: int = 5):
        self.method = method
        self.max_categories = max_categories
        self.target_encoder_cv = target_encoder_cv
        self.encoders = {}
        self.categorical_columns = []
        self.encoded_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders on training data."""
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.encoded_columns = []
        
        for col in self.categorical_columns:
            unique_vals = X[col].nunique()
            
            if unique_vals > self.max_categories:
                logger.warning(f"Column {col} has {unique_vals} unique values, "
                             f"exceeding max_categories={self.max_categories}")
                continue
            
            if self.method == EncodingMethod.ONEHOT:
                encoder = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    drop='first'
                )
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
                
                # Get encoded column names
                feature_names = encoder.get_feature_names_out([col])
                self.encoded_columns.extend(feature_names)
            
            elif self.method == EncodingMethod.TARGET and y is not None:
                encoder = TargetEncoder(
                    cv=self.target_encoder_cv,
                    random_state=42
                )
                encoder.fit(X[col], y)
                self.encoders[col] = encoder
                self.encoded_columns.append(col)
            
            elif self.method == EncodingMethod.ORDINAL:
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
                self.encoded_columns.append(col)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical data."""
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if self.method == EncodingMethod.ONEHOT:
                encoded = encoder.transform(X[[col]])
                feature_names = encoder.get_feature_names_out([col])
                
                # Add encoded columns
                for i, feature_name in enumerate(feature_names):
                    X_transformed[feature_name] = encoded[:, i]
                
                # Drop original column
                X_transformed = X_transformed.drop(columns=[col])
            
            elif self.method in [EncodingMethod.TARGET, EncodingMethod.ORDINAL]:
                X_transformed[col] = encoder.transform(X[col])
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.encoded_columns:
            return self.encoded_columns
        elif input_features is not None:
            return list(input_features)
        else:
            return [f"feature_{i}" for i in range(len(input_features) if input_features else 0)]


class DateTimeExpander(BaseEstimator, TransformerMixin):
    """Expand datetime columns into multiple features."""
    
    def __init__(self, 
                 expand_year: bool = True,
                 expand_month: bool = True,
                 expand_day: bool = True,
                 expand_dow: bool = True,
                 expand_hour: bool = True):
        self.expand_year = expand_year
        self.expand_month = expand_month
        self.expand_day = expand_day
        self.expand_dow = expand_dow
        self.expand_hour = expand_hour
        self.datetime_columns = []
        self.expanded_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit datetime expander."""
        self.datetime_columns = X.select_dtypes(include=['datetime64']).columns.tolist()
        self.expanded_columns = []
        
        for col in self.datetime_columns:
            if self.expand_year:
                self.expanded_columns.append(f"{col}_year")
            if self.expand_month:
                self.expanded_columns.append(f"{col}_month")
            if self.expand_day:
                self.expanded_columns.append(f"{col}_day")
            if self.expand_dow:
                self.expanded_columns.append(f"{col}_dow")
            if self.expand_hour:
                self.expanded_columns.append(f"{col}_hour")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime columns."""
        X_transformed = X.copy()
        
        for col in self.datetime_columns:
            dt_series = pd.to_datetime(X[col])
            
            if self.expand_year:
                X_transformed[f"{col}_year"] = dt_series.dt.year
            if self.expand_month:
                X_transformed[f"{col}_month"] = dt_series.dt.month
            if self.expand_day:
                X_transformed[f"{col}_day"] = dt_series.dt.day
            if self.expand_dow:
                X_transformed[f"{col}_dow"] = dt_series.dt.dayofweek
            if self.expand_hour:
                X_transformed[f"{col}_hour"] = dt_series.dt.hour
            
            # Drop original datetime column
            X_transformed = X_transformed.drop(columns=[col])
        
        return X_transformed


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scale numeric features."""
    
    def __init__(self, 
                 method: str = 'standard',
                 columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns
        self.scaler = None
        self.scaled_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler on training data."""
        if self.columns is None:
            self.scaled_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.scaled_columns = [col for col in self.columns if col in X.columns]
        
        if not self.scaled_columns:
            return self
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler.fit(X[self.scaled_columns])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling features."""
        X_transformed = X.copy()
        
        if self.scaler and self.scaled_columns:
            X_transformed[self.scaled_columns] = self.scaler.transform(
                X[self.scaled_columns]
            )
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is not None:
            return list(input_features)
        else:
            return [f"feature_{i}" for i in range(len(self.scaled_columns) if self.scaled_columns else 0)]


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers in numeric columns."""
    
    def __init__(self, 
                 method: str = 'clip',
                 outlier_indices: Optional[Dict[str, np.ndarray]] = None):
        self.method = method
        self.outlier_indices = outlier_indices or {}
        self.numeric_columns = []
        self.clip_values = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit outlier handler."""
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.method == 'clip':
            # Calculate clip values based on IQR
            for col in self.numeric_columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.clip_values[col] = {
                    'lower': Q1 - 1.5 * IQR,
                    'upper': Q3 + 1.5 * IQR
                }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers."""
        X_transformed = X.copy()
        
        if self.method == 'clip':
            for col in self.numeric_columns:
                if col in self.clip_values:
                    clip_vals = self.clip_values[col]
                    X_transformed[col] = X_transformed[col].clip(
                        lower=clip_vals['lower'],
                        upper=clip_vals['upper']
                    )
        
        elif self.method == 'remove':
            # Remove rows with outliers
            outlier_mask = np.zeros(len(X), dtype=bool)
            for col, indices in self.outlier_indices.items():
                if col in X.columns:
                    outlier_mask[indices] = True
            X_transformed = X_transformed[~outlier_mask]
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is not None:
            return list(input_features)
        else:
            return [f"feature_{i}" for i in range(len(self.numeric_columns) if self.numeric_columns else 0)]


class PreprocessingPipeline:
    """Complete preprocessing pipeline."""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline = None
        self.column_transformer = None
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit preprocessing pipeline."""
        # Create transformers
        transformers = []
        
        # Imputation
        imputation = ImputationTransformer(
            numeric_strategy=self.config.get('impute_numeric', ImputationMethod.MEDIAN),
            categorical_strategy=self.config.get('impute_categorical', ImputationMethod.MOST_FREQUENT)
        )
        
        # DateTime expansion
        if self.config.get('datetime_expansion', True):
            datetime_expander = DateTimeExpander()
            transformers.append(('datetime', datetime_expander, X.select_dtypes(include=['datetime64']).columns.tolist()))
        
        # Categorical encoding
        categorical_encoder = CategoricalEncoder(
            method=self.config.get('encode_categorical', EncodingMethod.ONEHOT),
            max_categories=self.config.get('max_categories', 50)
        )
        
        # Feature scaling
        if self.config.get('scale_features', True):
            feature_scaler = FeatureScaler(
                method=self.config.get('scaling_method', 'standard')
            )
        
        # Create column transformer
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Create pipeline
        pipeline_steps = [
            ('imputation', imputation),
            ('categorical_encoding', categorical_encoder)
        ]
        
        if self.config.get('scale_features', True):
            pipeline_steps.append(('scaling', feature_scaler))
        
        self.pipeline = Pipeline(pipeline_steps)
        
        # Fit pipeline
        X_processed = self.column_transformer.fit_transform(X)
        self.pipeline.fit(X_processed, y)
        
        # Store feature names
        self.feature_names_ = self._get_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through pipeline."""
        # Apply column transformer
        X_processed = self.column_transformer.transform(X)
        
        # Apply pipeline
        X_transformed = self.pipeline.transform(X_processed)
        
        # Convert to DataFrame
        if hasattr(self.pipeline, 'get_feature_names_out'):
            feature_names = self.pipeline.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation."""
        if hasattr(self.pipeline, 'get_feature_names_out'):
            return self.pipeline.get_feature_names_out().tolist()
        else:
            return [f"feature_{i}" for i in range(X.shape[1])]
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        return self.feature_names_
