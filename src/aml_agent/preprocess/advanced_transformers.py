"""
Advanced preprocessing transformers for enhanced feature engineering.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk  # noqa: F401

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..logging import get_logger

logger = get_logger()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced text preprocessing transformer."""

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        use_sentiment: bool = True,
        use_length_features: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_sentiment = use_sentiment
        self.use_length_features = use_length_features
        self.vectorizer = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TextPreprocessor":
        """Fit the text preprocessor."""
        text_columns = self._get_text_columns(X)

        if not text_columns:
            logger.warning("No text columns found for preprocessing")
            return self

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words="english" if NLTK_AVAILABLE else None,
            lowercase=True,
            strip_accents="unicode",
        )

        # Fit on all text columns combined
        all_text = []
        for col in text_columns:
            text_data = X[col].fillna("").astype(str)
            all_text.extend(text_data.tolist())

        self.vectorizer.fit(all_text)
        self.is_fitted = True
        logger.info(f"Text preprocessor fitted on {len(text_columns)} text columns")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text data."""
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before transform")

        text_columns = self._get_text_columns(X)

        if not text_columns:
            return X

        result = X.copy()

        # Process each text column
        for col in text_columns:
            text_data = X[col].fillna("").astype(str)

            # TF-IDF features
            tfidf_features = self.vectorizer.transform(text_data)
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])],
                index=X.index,
            )
            result = pd.concat([result, tfidf_df], axis=1)

            # Additional text features
            if self.use_sentiment and TEXTBLOB_AVAILABLE:
                sentiment_features = self._extract_sentiment_features(text_data)
                for feature_name, feature_values in sentiment_features.items():
                    result[f"{col}_{feature_name}"] = feature_values

            if self.use_length_features:
                length_features = self._extract_length_features(text_data)
                for feature_name, feature_values in length_features.items():
                    result[f"{col}_{feature_name}"] = feature_values

        # Remove original text columns
        result = result.drop(columns=text_columns)

        logger.info(f"Text preprocessing completed. New shape: {result.shape}")
        return result

    def _get_text_columns(self, X) -> List[str]:
        """Identify text columns in the DataFrame."""
        if not hasattr(X, "columns"):
            # If X is a numpy array, return empty list
            return []

        text_columns = []
        for col in X.columns:
            if X[col].dtype == "object":
                # Check if column contains text-like data
                sample_values = X[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if values contain words (not just numbers or categories)
                    text_ratio = sum(
                        1
                        for val in sample_values
                        if isinstance(val, str) and len(val.split()) > 1
                    ) / len(sample_values)
                    if text_ratio > 0.3:  # 30% of values are multi-word
                        text_columns.append(col)
        return text_columns

    def _extract_sentiment_features(
        self, text_data: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Extract sentiment features from text."""
        features = {
            "sentiment_polarity": np.zeros(len(text_data)),
            "sentiment_subjectivity": np.zeros(len(text_data)),
        }

        for i, text in enumerate(text_data):
            try:
                blob = TextBlob(str(text))
                features["sentiment_polarity"][i] = blob.sentiment.polarity
                features["sentiment_subjectivity"][i] = blob.sentiment.subjectivity
            except Exception as e:
                logger.warning(f"Error processing sentiment for text {i}: {e}")
                features["sentiment_polarity"][i] = 0.0
                features["sentiment_subjectivity"][i] = 0.0

        return features

    def _extract_length_features(self, text_data: pd.Series) -> Dict[str, np.ndarray]:
        """Extract length-based features from text."""
        features = {
            "char_count": text_data.str.len().fillna(0).values,
            "word_count": text_data.str.split().str.len().fillna(0).values,
            "sentence_count": text_data.str.count(r"[.!?]+").fillna(0).values,
            "avg_word_length": (
                text_data.str.split().str.len() / text_data.str.len().replace(0, 1)
            )
            .fillna(0)
            .values,
        }
        return features


class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Advanced text embedding transformer using sentence transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_features: int = 384,
        use_sentence_transformers: bool = True,
        use_spacy_embeddings: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        self.model_name = model_name
        self.max_features = max_features
        self.use_sentence_transformers = use_sentence_transformers
        self.use_spacy_embeddings = use_spacy_embeddings
        self.spacy_model = spacy_model
        self.sentence_model = None
        self.spacy_model_obj = None
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "TextEmbeddingTransformer":
        """Fit the text embedding transformer."""
        text_columns = self._get_text_columns(X)

        if not text_columns:
            logger.warning("No text columns found for embedding")
            return self

        # Initialize sentence transformer model
        if self.use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.model_name)
                logger.info(f"Sentence transformer model '{self.model_name}' loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.use_sentence_transformers = False

        # Initialize spaCy model
        if self.use_spacy_embeddings and SPACY_AVAILABLE:
            try:
                self.spacy_model_obj = spacy.load(self.spacy_model)
                logger.info(f"spaCy model '{self.spacy_model}' loaded")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy_embeddings = False

        self.is_fitted = True
        logger.info(
            f"Text embedding transformer fitted on {len(text_columns)} text columns"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text data to embeddings."""
        if not self.is_fitted:
            raise ValueError("TextEmbeddingTransformer must be fitted before transform")

        text_columns = self._get_text_columns(X)

        if not text_columns:
            return X

        result = X.copy()

        # Process each text column
        for col in text_columns:
            text_data = X[col].fillna("").astype(str)

            # Generate sentence transformer embeddings
            if self.use_sentence_transformers and self.sentence_model is not None:
                try:
                    embeddings = self.sentence_model.encode(text_data.tolist())
                    # Limit to max_features
                    if embeddings.shape[1] > self.max_features:
                        embeddings = embeddings[:, : self.max_features]

                    # Create DataFrame with embeddings
                    embedding_cols = [
                        f"{col}_embed_{i}" for i in range(embeddings.shape[1])
                    ]
                    embedding_df = pd.DataFrame(
                        embeddings, columns=embedding_cols, index=X.index
                    )
                    result = pd.concat([result, embedding_df], axis=1)
                    logger.info(
                        f"Generated {embeddings.shape[1]} sentence transformer embeddings for {col}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate sentence transformer embeddings: {e}"
                    )

            # Generate spaCy embeddings
            if self.use_spacy_embeddings and self.spacy_model_obj is not None:
                try:
                    spacy_embeddings = []
                    for text in text_data:
                        doc = self.spacy_model_obj(text)
                        # Use average of word embeddings
                        if doc.has_vector:
                            spacy_embeddings.append(doc.vector[: self.max_features])
                        else:
                            spacy_embeddings.append(np.zeros(self.max_features))

                    spacy_embeddings = np.array(spacy_embeddings)
                    spacy_cols = [
                        f"{col}_spacy_{i}" for i in range(spacy_embeddings.shape[1])
                    ]
                    spacy_df = pd.DataFrame(
                        spacy_embeddings, columns=spacy_cols, index=X.index
                    )
                    result = pd.concat([result, spacy_df], axis=1)
                    logger.info(
                        f"Generated {spacy_embeddings.shape[1]} spaCy embeddings for {col}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate spaCy embeddings: {e}")

        # Remove original text columns
        result = result.drop(columns=text_columns)

        logger.info(f"Text embedding completed. New shape: {result.shape}")
        return result

    def _get_text_columns(self, X) -> List[str]:
        """Identify text columns in the DataFrame."""
        if not hasattr(X, "columns"):
            # If X is a numpy array, return empty list
            return []

        text_columns = []
        for col in X.columns:
            if X[col].dtype == "object":
                # Check if column contains text-like data
                sample_values = X[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if values contain words (not just numbers or categories)
                    text_ratio = sum(
                        1
                        for val in sample_values
                        if isinstance(val, str) and len(val.split()) > 1
                    ) / len(sample_values)
                    if text_ratio > 0.3:  # 30% of values are multi-word
                        text_columns.append(col)
        return text_columns


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate polynomial features for numeric columns."""

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_features: int = 100,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.poly_features = None
        self.feature_names_ = []
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PolynomialFeatureGenerator":
        """Fit the polynomial feature generator."""
        if hasattr(X, "select_dtypes"):
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If X is a numpy array, create string column names
            numeric_columns = [f"feature_{i}" for i in range(X.shape[1])]

        if not numeric_columns:
            logger.warning("No numeric columns found for polynomial features")
            return self

        # Limit features if too many
        if len(numeric_columns) > 10:  # Limit to prevent explosion
            numeric_columns = numeric_columns[:10]
            logger.warning(
                f"Limited polynomial features to {len(numeric_columns)} columns"
            )

        self.poly_features = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        # Fit on numeric columns only
        if hasattr(X, "iloc"):
            # Use column names for DataFrame
            X_numeric = X[numeric_columns]
        else:
            # For numpy arrays, use integer indices
            numeric_indices = list(range(len(numeric_columns)))
            X_numeric = X[:, numeric_indices]

        # Handle NaN values before polynomial features
        if hasattr(X_numeric, "isnull") and X_numeric.isnull().any().any():
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            if hasattr(X_numeric, "columns"):
                X_numeric = pd.DataFrame(
                    imputer.fit_transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index,
                )
            else:
                X_numeric = imputer.fit_transform(X_numeric)

        self.poly_features.fit(X_numeric)

        # Generate feature names
        self.feature_names_ = self.poly_features.get_feature_names_out(numeric_columns)

        # Limit features if too many
        if len(self.feature_names_) > self.max_features:
            # Keep most important features (simple heuristic)
            feature_importance = np.var(X_numeric, axis=0)
            top_indices = np.argsort(feature_importance)[-self.max_features :]
            self.feature_names_ = [self.feature_names_[i] for i in top_indices]
            logger.warning(
                f"Limited polynomial features to {len(self.feature_names_)} features"
            )

        self.is_fitted = True
        logger.info(
            f"Polynomial feature generator fitted with {len(self.feature_names_)} features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with polynomial features."""
        if not self.is_fitted:
            raise ValueError(
                "PolynomialFeatureGenerator must be fitted before transform"
            )

        if hasattr(X, "select_dtypes"):
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If X is a numpy array, create string column names
            numeric_columns = [f"feature_{i}" for i in range(X.shape[1])]

        if not numeric_columns:
            return X

        # Limit to same columns used in fit
        if len(numeric_columns) > 10:
            numeric_columns = numeric_columns[:10]

        # Generate polynomial features
        if hasattr(X, "iloc"):
            # Use column names for DataFrame
            X_numeric = X[numeric_columns]
        else:
            # For numpy arrays, use integer indices
            numeric_indices = list(range(len(numeric_columns)))
            X_numeric = X[:, numeric_indices]

        # Handle NaN values before polynomial features
        if hasattr(X_numeric, "isnull") and X_numeric.isnull().any().any():
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            if hasattr(X_numeric, "columns"):
                X_numeric = pd.DataFrame(
                    imputer.fit_transform(X_numeric),
                    columns=X_numeric.columns,
                    index=X_numeric.index,
                )
            else:
                X_numeric = imputer.fit_transform(X_numeric)

        poly_features = self.poly_features.transform(X_numeric)

        # Create DataFrame with polynomial features
        if hasattr(X, "index"):
            poly_df = pd.DataFrame(
                poly_features,
                columns=self.poly_features.get_feature_names_out(numeric_columns),
                index=X.index,
            )
        else:
            poly_df = pd.DataFrame(
                poly_features,
                columns=self.poly_features.get_feature_names_out(numeric_columns),
            )

        # Limit to selected features
        available_features = [f for f in self.feature_names_ if f in poly_df.columns]
        poly_df = poly_df[available_features]

        # Combine with original data
        if hasattr(X, "iloc"):
            result = pd.concat([X, poly_df], axis=1)
        else:
            # Convert numpy array to DataFrame first
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            result = pd.concat([X_df, poly_df], axis=1)

        logger.info(f"Polynomial features generated. New shape: {result.shape}")
        return result


class AdvancedOutlierDetector(BaseEstimator, TransformerMixin):
    """Advanced outlier detection and handling."""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        handling_method: str = "clip",  # "clip", "remove", "transform"
        robust_scaling: bool = True,
    ):
        self.method = method
        self.contamination = contamination
        self.handling_method = handling_method
        self.robust_scaling = robust_scaling
        self.outlier_detector = None
        self.scaler = None
        self.outlier_indices_ = set()
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AdvancedOutlierDetector":
        """Fit the outlier detector."""
        if hasattr(X, "select_dtypes"):
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If X is a numpy array, create string column names
            numeric_columns = [f"feature_{i}" for i in range(X.shape[1])]

        if not numeric_columns:
            logger.warning("No numeric columns found for outlier detection")
            return self

        # Initialize outlier detector
        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            self.outlier_detector = IsolationForest(
                contamination=self.contamination, random_state=42
            )
        elif self.method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor

            self.outlier_detector = LocalOutlierFactor(
                contamination=self.contamination, novelty=True
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

        # Fit detector
        if hasattr(X, "iloc"):
            # Use column names for DataFrame
            X_numeric = X[numeric_columns]
        else:
            # For numpy arrays, use integer indices
            numeric_indices = list(range(len(numeric_columns)))
            X_numeric = X[:, numeric_indices]
        outlier_labels = self.outlier_detector.fit_predict(X_numeric)
        self.outlier_indices_ = set(np.where(outlier_labels == -1)[0])

        # Initialize robust scaler if needed
        if self.robust_scaling:
            self.scaler = RobustScaler()
            self.scaler.fit(X_numeric)

        self.is_fitted = True
        logger.info(
            f"Advanced outlier detector fitted. Found {len(self.outlier_indices_)} outliers"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with outlier handling."""
        if not self.is_fitted:
            raise ValueError("AdvancedOutlierDetector must be fitted before transform")

        if hasattr(X, "select_dtypes"):
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If X is a numpy array, create string column names
            numeric_columns = [f"feature_{i}" for i in range(X.shape[1])]

        if not numeric_columns:
            return X

        result = X.copy()
        if hasattr(X, "iloc"):
            # Use column names for DataFrame
            X_numeric = X[numeric_columns]
        else:
            # For numpy arrays, use integer indices
            numeric_indices = list(range(len(numeric_columns)))
            X_numeric = X[:, numeric_indices]

        if self.handling_method == "clip":
            # Clip outliers to percentiles
            for col in numeric_columns:
                q1 = X_numeric[col].quantile(0.25)
                q3 = X_numeric[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                result[col] = result[col].clip(lower_bound, upper_bound)

        elif self.handling_method == "transform":
            # Apply robust scaling
            if self.scaler is not None:
                scaled_features = self.scaler.transform(X_numeric)
                for i, col in enumerate(numeric_columns):
                    result[col] = scaled_features[:, i]

        elif self.handling_method == "remove":
            # Remove outlier rows
            outlier_mask = np.zeros(len(X), dtype=bool)
            for idx in self.outlier_indices_:
                if idx < len(X):
                    outlier_mask[idx] = True
            result = result[~outlier_mask]

        logger.info(f"Outlier handling completed. New shape: {result.shape}")
        return result


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Advanced feature selection based on various criteria."""

    def __init__(
        self,
        method: str = "mutual_info",  # "mutual_info", "f_score", "variance", "correlation"
        k: int = 10,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
    ):
        self.method = method
        self.k = k
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.selected_features_ = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit the feature selector."""
        if y is None and self.method in ["mutual_info", "f_score"]:
            logger.warning(
                "Target required for mutual_info/f_score selection, using variance instead"
            )
            self.method = "variance"

        if hasattr(X, "select_dtypes"):
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # If X is a numpy array, create string column names
            numeric_columns = [f"feature_{i}" for i in range(X.shape[1])]

        if not numeric_columns:
            logger.warning("No numeric columns found for feature selection")
            return self

        if self.method == "mutual_info":
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )
            from sklearn.preprocessing import LabelEncoder

            # Encode target if categorical
            if y.dtype == "object":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y

            # Determine if classification or regression
            if len(np.unique(y_encoded)) < 20:  # Classification
                if hasattr(X, "iloc"):
                    scores = mutual_info_classif(X[numeric_columns], y_encoded)
                else:
                    numeric_indices = list(range(len(numeric_columns)))
                    scores = mutual_info_classif(X[:, numeric_indices], y_encoded)
            else:  # Regression
                if hasattr(X, "iloc"):
                    scores = mutual_info_regression(X[numeric_columns], y_encoded)
                else:
                    numeric_indices = list(range(len(numeric_columns)))
                    scores = mutual_info_regression(X[:, numeric_indices], y_encoded)

            # Select top k features
            top_indices = np.argsort(scores)[-self.k :]
            self.selected_features_ = [numeric_columns[i] for i in top_indices]

        elif self.method == "f_score":
            from sklearn.feature_selection import f_classif, f_regression
            from sklearn.preprocessing import LabelEncoder

            # Encode target if categorical
            if y.dtype == "object":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y

            # Determine if classification or regression
            if len(np.unique(y_encoded)) < 20:  # Classification
                if hasattr(X, "iloc"):
                    scores, _ = f_classif(X[numeric_columns], y_encoded)
                else:
                    numeric_indices = list(range(len(numeric_columns)))
                    scores, _ = f_classif(X[:, numeric_indices], y_encoded)
            else:  # Regression
                if hasattr(X, "iloc"):
                    scores, _ = f_regression(X[numeric_columns], y_encoded)
                else:
                    numeric_indices = list(range(len(numeric_columns)))
                    scores, _ = f_regression(X[:, numeric_indices], y_encoded)

            # Select top k features
            top_indices = np.argsort(scores)[-self.k :]
            self.selected_features_ = [numeric_columns[i] for i in top_indices]

        elif self.method == "variance":
            from sklearn.feature_selection import VarianceThreshold

            selector = VarianceThreshold(threshold=self.variance_threshold)
            if hasattr(X, "iloc"):
                selector.fit(X[numeric_columns])
            else:
                numeric_indices = list(range(len(numeric_columns)))
                selector.fit(X[:, numeric_indices])
            self.selected_features_ = [
                col
                for col, selected in zip(numeric_columns, selector.get_support())
                if selected
            ]

        elif self.method == "correlation":
            # Remove highly correlated features
            if hasattr(X, "iloc"):
                corr_matrix = X[numeric_columns].corr().abs()
            else:
                # For numpy arrays, convert to DataFrame first
                X_df = pd.DataFrame(
                    X[:, : len(numeric_columns)], columns=numeric_columns
                )
                corr_matrix = X_df.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column
                for column in upper_tri.columns
                if any(upper_tri[column] > self.correlation_threshold)
            ]

            self.selected_features_ = [
                col for col in numeric_columns if col not in to_drop
            ]

        # Limit to k features if more than k selected
        if len(self.selected_features_) > self.k:
            self.selected_features_ = self.selected_features_[: self.k]

        self.is_fitted = True
        logger.info(
            f"Feature selection completed. Selected {len(self.selected_features_)} features"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with selected features."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform")

        if not self.selected_features_:
            return X

        # Return only selected features
        result = X[self.selected_features_].copy()

        logger.info(f"Feature selection applied. New shape: {result.shape}")
        return result


class AdvancedPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """Advanced preprocessing pipeline with enhanced feature engineering."""

    def __init__(
        self,
        use_text_preprocessing: bool = True,
        use_text_embeddings: bool = True,
        use_polynomial_features: bool = True,
        use_advanced_outlier_detection: bool = True,
        use_feature_selection: bool = True,
        text_max_features: int = 1000,
        embedding_max_features: int = 384,
        poly_degree: int = 2,
        outlier_method: str = "isolation_forest",
        feature_selection_method: str = "mutual_info",
        feature_selection_k: int = 20,
    ):
        self.use_text_preprocessing = use_text_preprocessing
        self.use_text_embeddings = use_text_embeddings
        self.use_polynomial_features = use_polynomial_features
        self.use_advanced_outlier_detection = use_advanced_outlier_detection
        self.use_feature_selection = use_feature_selection
        self.text_max_features = text_max_features
        self.embedding_max_features = embedding_max_features
        self.poly_degree = poly_degree
        self.outlier_method = outlier_method
        self.feature_selection_method = feature_selection_method
        self.feature_selection_k = feature_selection_k

        self.text_preprocessor = None
        self.text_embedder = None
        self.poly_generator = None
        self.outlier_detector = None
        self.feature_selector = None
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AdvancedPreprocessingPipeline":
        """Fit the advanced preprocessing pipeline."""
        logger.info("Starting advanced preprocessing pipeline fitting")

        # Text preprocessing
        if self.use_text_preprocessing:
            self.text_preprocessor = TextPreprocessor(
                max_features=self.text_max_features
            )
            self.text_preprocessor.fit(X, y)

        # Text embeddings
        if self.use_text_embeddings:
            self.text_embedder = TextEmbeddingTransformer(
                max_features=self.embedding_max_features
            )
            self.text_embedder.fit(X, y)

        # Polynomial features
        if self.use_polynomial_features:
            self.poly_generator = PolynomialFeatureGenerator(degree=self.poly_degree)
            self.poly_generator.fit(X, y)

        # Advanced outlier detection
        if self.use_advanced_outlier_detection:
            self.outlier_detector = AdvancedOutlierDetector(method=self.outlier_method)
            self.outlier_detector.fit(X, y)

        # Feature selection
        if self.use_feature_selection:
            self.feature_selector = FeatureSelector(
                method=self.feature_selection_method, k=self.feature_selection_k
            )
            self.feature_selector.fit(X, y)

        self.is_fitted = True
        logger.info("Advanced preprocessing pipeline fitted successfully")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through advanced preprocessing pipeline."""
        if not self.is_fitted:
            raise ValueError(
                "AdvancedPreprocessingPipeline must be fitted before transform"
            )

        result = X.copy()

        # Apply text preprocessing
        if (
            self.text_preprocessor is not None
            and self.use_text_preprocessing
            and hasattr(self.text_preprocessor, "is_fitted")
            and self.text_preprocessor.is_fitted
        ):
            result = self.text_preprocessor.transform(result)
            # Ensure unique column names
            result.columns = [
                f"{col}_{i}" if col in result.columns[:i] else col
                for i, col in enumerate(result.columns)
            ]

        # Apply text embeddings
        if (
            self.text_embedder is not None
            and self.use_text_embeddings
            and hasattr(self.text_embedder, "is_fitted")
            and self.text_embedder.is_fitted
        ):
            result = self.text_embedder.transform(result)
            # Ensure unique column names
            result.columns = [
                f"{col}_{i}" if col in result.columns[:i] else col
                for i, col in enumerate(result.columns)
            ]

        # Apply polynomial features
        if (
            self.poly_generator is not None
            and self.use_polynomial_features
            and hasattr(self.poly_generator, "is_fitted")
            and self.poly_generator.is_fitted
        ):
            result = self.poly_generator.transform(result)
            # Ensure unique column names
            result.columns = [
                f"{col}_{i}" if col in result.columns[:i] else col
                for i, col in enumerate(result.columns)
            ]

        # Apply outlier detection
        if (
            self.outlier_detector is not None
            and self.use_advanced_outlier_detection
            and hasattr(self.outlier_detector, "is_fitted")
            and self.outlier_detector.is_fitted
        ):
            result = self.outlier_detector.transform(result)
            # Ensure unique column names
            result.columns = [
                f"{col}_{i}" if col in result.columns[:i] else col
                for i, col in enumerate(result.columns)
            ]

        # Apply feature selection
        if (
            self.feature_selector is not None
            and self.use_feature_selection
            and hasattr(self.feature_selector, "is_fitted")
            and self.feature_selector.is_fitted
        ):
            result = self.feature_selector.transform(result)
            # Ensure unique column names
            result.columns = [
                f"{col}_{i}" if col in result.columns[:i] else col
                for i, col in enumerate(result.columns)
            ]

        logger.info(f"Advanced preprocessing completed. Final shape: {result.shape}")
        return result
