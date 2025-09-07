"""
Data type and quality detectors for preprocessing pipeline.
"""

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..logging import get_logger
from ..types import OutlierMethod

# LabelEncoder not used in this file


logger = get_logger()


class TypeDetector:
    """Detect data types in DataFrame columns."""

    def __init__(self):
        self.column_types = {}
        self.datetime_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
            r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        ]

    def detect_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column types in DataFrame.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping column names to types
        """
        self.column_types = {}

        for col in data.columns:
            col_type = self._detect_column_type(data[col])
            self.column_types[col] = col_type
            logger.debug(f"Column {col}: {col_type}")

        return self.column_types

    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect type of a single column."""
        # Check for datetime
        if self._is_datetime(series):
            return "datetime"

        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Check for categorical
        if self._is_categorical(series):
            return "categorical"

        # Check for text
        if self._is_text(series):
            return "text"

        # Default to categorical
        return "categorical"

    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if series contains datetime data."""
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Check for datetime patterns in string data
        if series.dtype == "object":
            sample_values = series.dropna().head(10)
            if len(sample_values) == 0:
                return False

            datetime_count = 0
            for value in sample_values:
                value_str = str(value)
                if any(
                    re.search(pattern, value_str) for pattern in self.datetime_patterns
                ):
                    datetime_count += 1

            # If more than 50% match datetime patterns
            return datetime_count / len(sample_values) > 0.5

        return False

    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if series is categorical."""
        # Check if already categorical
        if hasattr(
            pd.api.types, "is_categorical_dtype"
        ) and pd.api.types.is_categorical_dtype(series):
            return True

        # Check for object type with limited unique values
        if series.dtype == "object":
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.1  # Less than 10% unique values

        # Check for integer with few unique values
        if pd.api.types.is_integer_dtype(series):
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.05  # Less than 5% unique values

        return False

    def _is_text(self, series: pd.Series) -> bool:
        """Check if series contains text data."""
        if series.dtype != "object":
            return False

        # Check average string length
        sample_values = series.dropna().head(10)
        if len(sample_values) == 0:
            return False

        avg_length = sample_values.astype(str).str.len().mean()
        return avg_length > 20  # Average length > 20 characters

    def get_columns_by_type(self, col_type: str) -> List[str]:
        """Get columns of specific type."""
        return [col for col, type_ in self.column_types.items() if type_ == col_type]


class MissingValueDetector:
    """Detect missing values in DataFrame."""

    def __init__(self):
        self.missing_info = {}

    def detect_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values in DataFrame.

        Args:
            data: Input DataFrame

        Returns:
            Missing value information
        """
        self.missing_info = {
            "total_missing": data.isnull().sum().sum(),
            "missing_ratio": data.isnull().sum().sum()
            / (data.shape[0] * data.shape[1]),
            "columns_with_missing": [],
            "missing_by_column": {},
        }

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)

            if missing_count > 0:
                self.missing_info["columns_with_missing"].append(col)
                self.missing_info["missing_by_column"][col] = {
                    "count": int(missing_count),
                    "ratio": float(missing_ratio),
                }

        logger.info(
            f"Missing values detected: {self.missing_info['total_missing']} total"
        )
        return self.missing_info


class OutlierDetector:
    """Detect outliers in numeric columns."""

    def __init__(self, method: OutlierMethod = OutlierMethod.IQR):
        self.method = method
        self.outlier_indices: List[int] = []
        self.outlier_info: Dict[str, Any] = {}

    def detect_outliers(
        self, data: pd.DataFrame, numeric_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.

        Args:
            data: Input DataFrame
            numeric_columns: List of numeric column names

        Returns:
            Outlier information
        """
        self.outlier_info = {
            "method": self.method.value,
            "total_outliers": 0,
            "outliers_by_column": {},
        }

        for col in numeric_columns:
            if col not in data.columns:
                continue

            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            outliers = self._detect_column_outliers(col_data, col)
            self.outlier_info["outliers_by_column"][col] = {
                "count": len(outliers),
                "indices": outliers,
                "ratio": len(outliers) / len(col_data),
            }
            self.outlier_info["total_outliers"] += len(outliers)

            # Add to global outlier indices
            self.outlier_indices.extend(outliers)

        logger.info(f"Outliers detected: {self.outlier_info['total_outliers']} total")
        return self.outlier_info

    def _detect_column_outliers(self, series: pd.Series, col_name: str) -> List[int]:
        """Detect outliers in a single column."""
        if self.method == OutlierMethod.IQR:
            return self._iqr_outliers(series)
        elif self.method == OutlierMethod.ZSCORE:
            return self._zscore_outliers(series)
        elif self.method == OutlierMethod.ISOLATION_FOREST:
            return self._isolation_forest_outliers(series)
        else:
            return []

    def _iqr_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        # Ensure we have numeric data and handle edge cases
        if series.dtype == "bool" or series.dtype == "object":
            return []

        # Convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors="coerce")

        # Remove NaN values for quantile calculation
        clean_series = numeric_series.dropna()

        if len(clean_series) < 4:  # Need at least 4 points for IQR
            return []

        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:  # No variation in data
            return []

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = clean_series[
            (clean_series < lower_bound) | (clean_series > upper_bound)
        ]
        return outliers.index.tolist()

    def _zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.tolist()

    def _isolation_forest_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using Isolation Forest."""
        try:
            # Reshape for sklearn
            # X = ...  # unused
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            values = np.asarray(series.values)
            outlier_labels = iso_forest.fit_predict(values.reshape(-1, 1))
            outliers = series[outlier_labels == -1]
            return outliers.index.tolist()
        except Exception as e:
            logger.warning(f"Isolation Forest failed for {series.name}: {e}")
            return []


class DataQualityAnalyzer:
    """Comprehensive data quality analysis."""

    def __init__(self):
        self.type_detector = TypeDetector()
        self.missing_detector = MissingValueDetector()
        self.outlier_detector = OutlierDetector()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.

        Args:
            data: Input DataFrame

        Returns:
            Comprehensive quality report
        """
        logger.info("Starting comprehensive data quality analysis")

        # Detect types
        column_types = self.type_detector.detect_types(data)

        # Detect missing values
        missing_info = self.missing_detector.detect_missing(data)

        # Detect outliers in numeric columns
        numeric_columns = self.type_detector.get_columns_by_type("numeric")
        outlier_info = {}
        if numeric_columns:
            outlier_info = self.outlier_detector.detect_outliers(data, numeric_columns)

        # Calculate additional metrics
        quality_metrics = self._calculate_quality_metrics(data, column_types)

        return {
            "data_shape": data.shape,
            "column_types": column_types,
            "missing_info": missing_info,
            "outlier_info": outlier_info,
            "quality_metrics": quality_metrics,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
        }

    def _calculate_quality_metrics(
        self, data: pd.DataFrame, column_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """Calculate additional quality metrics."""
        metrics = {
            "numeric_columns": len(
                [t for t in column_types.values() if t == "numeric"]
            ),
            "categorical_columns": len(
                [t for t in column_types.values() if t == "categorical"]
            ),
            "datetime_columns": len(
                [t for t in column_types.values() if t == "datetime"]
            ),
            "text_columns": len([t for t in column_types.values() if t == "text"]),
            "duplicate_rows": data.duplicated().sum(),
            "duplicate_ratio": data.duplicated().sum() / len(data),
        }

        # Calculate cardinality for categorical columns
        categorical_cols = [
            col for col, t in column_types.items() if t == "categorical"
        ]
        if categorical_cols:
            cardinalities = {}
            for col in categorical_cols:
                cardinalities[col] = data[col].nunique()
            metrics["categorical_cardinalities"] = cardinalities

        return metrics


def detect_data_types(data: pd.DataFrame) -> Dict[str, str]:
    """Convenience function to detect data types."""
    detector = TypeDetector()
    return detector.detect_types(data)


def analyze_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for comprehensive data quality analysis."""
    analyzer = DataQualityAnalyzer()
    return analyzer.analyze(data)
