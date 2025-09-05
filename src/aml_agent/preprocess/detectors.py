"""
Data type and quality detectors for preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..types import ImputationMethod, OutlierMethod
from ..logging import get_logger

logger = get_logger()


class TypeDetector:
    """Detect and categorize column types."""
    
    def __init__(self):
        self.column_types = {}
        self.datetime_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.text_columns = []
    
    def detect_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column types in dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary mapping column names to types
        """
        self.column_types = {}
        
        for col in data.columns:
            col_type = self._detect_column_type(data[col])
            self.column_types[col] = col_type
            
            # Categorize columns
            if col_type == 'datetime':
                self.datetime_columns.append(col)
            elif col_type == 'numeric':
                self.numeric_columns.append(col)
            elif col_type == 'categorical':
                self.categorical_columns.append(col)
            elif col_type == 'text':
                self.text_columns.append(col)
        
        logger.info(f"Detected column types: {self.column_types}")
        return self.column_types
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect type of a single column."""
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check for categorical (object with limited unique values)
        if series.dtype == 'object' or series.dtype.name == 'category':
            unique_ratio = series.nunique() / len(series)
            avg_length = series.astype(str).str.len().mean()
            
            # If many unique values and long strings, likely text
            if unique_ratio > 0.5 and avg_length > 20:
                return 'text'
            else:
                return 'categorical'
        
        return 'categorical'
    
    def get_columns_by_type(self, data_type: str) -> List[str]:
        """Get columns of specific type."""
        return [col for col, t in self.column_types.items() if t == data_type]


class MissingValueDetector:
    """Detect and analyze missing values."""
    
    def __init__(self):
        self.missing_info = {}
    
    def detect_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values in dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Missing value information
        """
        self.missing_info = {
            'total_missing': data.isnull().sum().sum(),
            'missing_ratio': data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
            'columns_with_missing': data.isnull().sum()[data.isnull().sum() > 0].to_dict(),
            'missing_patterns': self._analyze_missing_patterns(data)
        }
        
        logger.info(f"Missing values detected: {self.missing_info['total_missing']} total, "
                   f"{self.missing_info['missing_ratio']:.2%} ratio")
        return self.missing_info
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing values."""
        missing_data = data.isnull()
        
        # Check for completely missing columns
        completely_missing = missing_data.all().sum()
        
        # Check for rows with all missing values
        completely_missing_rows = missing_data.all(axis=1).sum()
        
        # Check for missing value correlation
        missing_corr = missing_data.corr()
        
        return {
            'completely_missing_columns': completely_missing,
            'completely_missing_rows': completely_missing_rows,
            'missing_correlation': missing_corr
        }
    
    def get_missing_columns(self) -> List[str]:
        """Get columns with missing values."""
        return list(self.missing_info.get('columns_with_missing', {}).keys())


class OutlierDetector:
    """Detect outliers in numeric columns."""
    
    def __init__(self, method: OutlierMethod = OutlierMethod.IQR):
        self.method = method
        self.outlier_info = {}
        self.outlier_indices = {}
    
    def detect_outliers(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Args:
            data: Input DataFrame
            numeric_columns: List of numeric column names
            
        Returns:
            Outlier information
        """
        self.outlier_info = {}
        self.outlier_indices = {}
        
        for col in numeric_columns:
            if col not in data.columns:
                continue
            
            # Skip boolean columns as they don't make sense for outlier detection
            if data[col].dtype == bool:
                continue
                
            series = data[col].dropna()
            if len(series) == 0:
                continue
            
            outliers = self._detect_column_outliers(series)
            self.outlier_indices[col] = outliers
            
            self.outlier_info[col] = {
                'n_outliers': len(outliers),
                'outlier_ratio': len(outliers) / len(series),
                'outlier_indices': outliers.tolist()
            }
        
        total_outliers = sum(info['n_outliers'] for info in self.outlier_info.values())
        logger.info(f"Detected {total_outliers} outliers across {len(numeric_columns)} numeric columns")
        
        return self.outlier_info
    
    def _detect_column_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers in a single column."""
        if self.method == OutlierMethod.IQR:
            return self._iqr_outliers(series)
        elif self.method == OutlierMethod.ZSCORE:
            return self._zscore_outliers(series)
        elif self.method == OutlierMethod.ISOLATION_FOREST:
            return self._isolation_forest_outliers(series)
        else:
            return np.array([])
    
    def _iqr_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.values
    
    def _zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.values
    
    def _isolation_forest_outliers(self, series: pd.Series, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        try:
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Get outlier indices
            outlier_indices = series.index[outlier_labels == -1]
            return outlier_indices.values
        except Exception as e:
            logger.warning(f"Isolation Forest failed for column {series.name}: {e}")
            return np.array([])
    
    def get_outlier_columns(self) -> List[str]:
        """Get columns with outliers."""
        return [col for col, info in self.outlier_info.items() if info['n_outliers'] > 0]
    
    def get_outlier_indices(self, column: str) -> np.ndarray:
        """Get outlier indices for specific column."""
        return self.outlier_indices.get(column, np.array([]))
