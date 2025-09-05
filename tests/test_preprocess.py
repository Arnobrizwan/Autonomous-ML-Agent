"""
Tests for preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.aml_agent.preprocess import (
    PreprocessingPipeline, TypeDetector, MissingValueDetector, 
    OutlierDetector, ImputationTransformer, CategoricalEncoder
)
from src.aml_agent.types import TaskType, ImputationMethod, EncodingMethod


class TestPreprocessing:
    """Test preprocessing components."""
    
    def test_type_detector(self):
        """Test type detection."""
        detector = TypeDetector()
        
        # Create test data
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'datetime': pd.date_range('2023-01-01', periods=5),
            'text': ['This is a very long text that should be detected as text type', 'Another very long text that should be detected as text type', 'This is also a very long text that should be detected as text type', 'Very long text here that should be detected as text type', 'This is a medium length text that should be detected as text type']
        })
        
        types = detector.detect_types(data)
        
        assert types['numeric'] == 'numeric'
        assert types['categorical'] == 'categorical'
        assert types['datetime'] == 'datetime'
        assert types['text'] == 'text'
    
    def test_missing_value_detector(self):
        """Test missing value detection."""
        detector = MissingValueDetector()
        
        # Create data with missing values
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['A', 'B', 'C', np.nan, 'E'],
            'col3': [1, 2, 3, 4, 5]  # No missing values
        })
        
        missing_info = detector.detect_missing(data)
        
        assert missing_info['total_missing'] == 2
        assert missing_info['missing_ratio'] == 2 / (3 * 5)  # 2 missing out of 15 total
        assert 'col1' in missing_info['columns_with_missing']
        assert 'col2' in missing_info['columns_with_missing']
        assert 'col3' not in missing_info['columns_with_missing']
    
    def test_outlier_detector(self):
        """Test outlier detection."""
        detector = OutlierDetector()
        
        # Create data with outliers
        data = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'with_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        })
        
        outlier_info = detector.detect_outliers(data, ['normal', 'with_outliers'])
        
        assert 'normal' in outlier_info
        assert 'with_outliers' in outlier_info
        assert outlier_info['normal']['n_outliers'] == 0
        assert outlier_info['with_outliers']['n_outliers'] > 0
    
    def test_imputation_transformer(self):
        """Test imputation transformer."""
        transformer = ImputationTransformer(
            numeric_strategy=ImputationMethod.MEDIAN,
            categorical_strategy=ImputationMethod.MOST_FREQUENT
        )
        
        # Create data with missing values
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', np.nan, 'A', 'C']
        })
        
        # Fit and transform
        transformed = transformer.fit_transform(data)
        
        assert not transformed.isnull().any().any()
        assert transformed['numeric'].dtype in ['int64', 'float64']
        assert transformed['categorical'].dtype == 'object'
    
    def test_categorical_encoder(self):
        """Test categorical encoding."""
        encoder = CategoricalEncoder(method=EncodingMethod.ONEHOT)
        
        # Create categorical data
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Z', 'Y']
        })
        
        # Fit and transform
        transformed = encoder.fit_transform(data)
        
        # Should have more columns after one-hot encoding
        assert transformed.shape[1] > data.shape[1]
        assert not transformed.isnull().any().any()
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        from src.aml_agent.config import PreprocessingConfig
        config = PreprocessingConfig(
            handle_missing=True,
            impute_numeric="mean",
            impute_categorical="most_frequent",
            encode_categorical="onehot",
            scale_features=False,
            handle_outliers=False,
            datetime_expansion=False
        )
        pipeline = PreprocessingPipeline(config=config)
        
        # Create test data
        data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, np.nan, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', np.nan, 'C'],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Fit and transform
        X_processed = pipeline.fit_transform(X, y)
        
        # Check results
        assert X_processed.shape[0] == X.shape[0]
        assert not X_processed.isnull().any().any()
        assert X_processed.shape[1] >= X.shape[1]  # May have more columns after encoding
        
        # Check feature names
        feature_names = pipeline.get_feature_names_out()
        assert len(feature_names) == X_processed.shape[1]
    
    def test_preprocessing_with_different_data_types(self):
        """Test preprocessing with various data types."""
        from src.aml_agent.config import PreprocessingConfig
        config = PreprocessingConfig(
            handle_missing=True,
            impute_numeric="mean",
            impute_categorical="most_frequent",
            encode_categorical="onehot",
            scale_features=False,
            handle_outliers=False,
            datetime_expansion=False
        )
        pipeline = PreprocessingPipeline(config=config)
        
        # Create complex test data
        data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['A', 'B', 'A', 'B', 'C'],
            'bool_col': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Should not raise any errors
        X_processed = pipeline.fit_transform(X, y)
        
        assert X_processed.shape[0] == X.shape[0]
        assert not X_processed.isnull().any().any()
    
    def test_preprocessing_with_missing_values(self):
        """Test preprocessing with missing values."""
        from src.aml_agent.config import PreprocessingConfig
        config = PreprocessingConfig(
            handle_missing=True,
            impute_numeric="mean",
            impute_categorical="most_frequent",
            encode_categorical="onehot",
            scale_features=False,
            handle_outliers=False,
            datetime_expansion=False
        )
        pipeline = PreprocessingPipeline(config=config)
        
        # Create data with missing values
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', np.nan, 'A', 'C'],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Should handle missing values
        X_processed = pipeline.fit_transform(X, y)
        
        assert X_processed.shape[0] == X.shape[0]
        assert not X_processed.isnull().any().any()
    
    def test_preprocessing_preserves_index(self):
        """Test that preprocessing preserves index."""
        pipeline = PreprocessingPipeline()
        
        # Create data with custom index
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'C'],
            'target': [0, 1, 0, 1, 0]
        }, index=['a', 'b', 'c', 'd', 'e'])
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        X_processed = pipeline.fit_transform(X, y)
        
        # Index should be preserved
        assert X_processed.index.equals(X.index)
    
    def test_preprocessing_info(self):
        """Test preprocessing info retrieval."""
        pipeline = PreprocessingPipeline()
        
        # Create test data
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'B', 'C'],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        pipeline.fit_transform(X, y)
        info = pipeline.get_preprocessing_info()
        
        assert 'column_types' in info
        assert 'missing_info' in info
        assert 'outlier_info' in info
        assert 'feature_names' in info
        assert 'is_fitted' in info
        assert info['is_fitted'] is True


if __name__ == "__main__":
    pytest.main([__file__])
