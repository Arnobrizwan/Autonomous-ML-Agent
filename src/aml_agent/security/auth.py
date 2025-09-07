"""
Authentication and security utilities for the ML agent.
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..logging import get_logger

logger = get_logger()


class SecurityManager:
    """Manage security and authentication for the ML agent."""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.getenv(
            "AML_SECRET_KEY", self._generate_secret_key()
        )
        self.api_keys: dict[str, dict[str, Any]] = (
            {}
        )  # In production, use a proper database
        self.rate_limits: dict[str, dict[str, Any]] = {}  # Simple rate limiting

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)

    def generate_api_key(
        self, user_id: str, permissions: Optional[List[str]] = None
    ) -> str:
        """
        Generate an API key for a user.

        Args:
            user_id: Unique user identifier
            permissions: List of permissions for the API key

        Returns:
            Generated API key
        """
        if permissions is None:
            permissions = ["read", "predict"]

        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
        }

        logger.info(f"API key generated for user {user_id}")
        return api_key

    def validate_api_key(self, api_key: str, required_permission: str = "read") -> bool:
        """
        Validate an API key and check permissions.

        Args:
            api_key: API key to validate
            required_permission: Required permission level

        Returns:
            True if valid and authorized
        """
        if api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]

        # Check if key has required permission
        if required_permission not in key_info["permissions"]:
            logger.warning(
                f"API key {api_key[:8]}... lacks permission {required_permission}"
            )
            return False

        # Update usage info
        key_info["last_used"] = datetime.now().isoformat()
        key_info["usage_count"] += 1

        return True

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if successfully revoked
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"API key {api_key[:8]}... revoked")
            return True
        return False

    def check_rate_limit(
        self, identifier: str, max_requests: int = 100, window_minutes: int = 60
    ) -> bool:
        """
        Check if a request is within rate limits.

        Args:
            identifier: Unique identifier (IP, user_id, etc.)
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes

        Returns:
            True if within rate limits
        """
        now = time.time()
        window_start = now - (window_minutes * 60)

        # Clean old entries
        if identifier in self.rate_limits:
            self.rate_limits[identifier] = [
                timestamp
                for timestamp in self.rate_limits[identifier]
                if timestamp > window_start
            ]
        else:
            self.rate_limits[identifier] = []

        # Check if within limits
        if len(self.rate_limits[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False

        # Add current request
        self.rate_limits[identifier].append(now)
        return True

    def validate_input_data(
        self,
        data: Union[Dict, pd.DataFrame],
        expected_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate input data for security and correctness.

        Args:
            data: Input data to validate
            expected_columns: Expected column names

        Returns:
            Validation result dictionary
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()

            # Check for malicious content
            validation_result.update(self._check_malicious_content(df))

            # Check data types and ranges
            validation_result.update(self._check_data_types(df))

            # Check for expected columns
            if expected_columns:
                validation_result.update(
                    self._check_expected_columns(df, expected_columns)
                )

            # Check for data leakage indicators
            validation_result.update(self._check_data_leakage(df))

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Data validation error: {str(e)}")

        return validation_result

    def _check_malicious_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for potentially malicious content in data."""
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}

        # Check for SQL injection patterns
        sql_patterns = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
        for col in df.select_dtypes(include=["object"]).columns:
            for pattern in sql_patterns:
                if (
                    df[col]
                    .astype(str)
                    .str.contains(pattern, regex=False, na=False)
                    .any()
                ):
                    result["warnings"].append(
                        f"Potential SQL injection pattern in column {col}"
                    )

        # Check for script injection patterns
        script_patterns = ["<script", "javascript:", "vbscript:", "onload=", "onerror="]
        for col in df.select_dtypes(include=["object"]).columns:
            for pattern in script_patterns:
                if (
                    df[col]
                    .astype(str)
                    .str.contains(pattern, case=False, regex=False, na=False)
                    .any()
                ):
                    result["warnings"].append(
                        f"Potential script injection pattern in column {col}"
                    )

        return result

    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data types and ranges."""
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}

        for col in df.columns:
            # Check for extremely large values
            if df[col].dtype in ["int64", "float64"]:
                if df[col].abs().max() > 1e10:
                    result["warnings"].append(f"Very large values in column {col}")

            # Check for suspicious string lengths
            if df[col].dtype == "object":
                max_length = df[col].astype(str).str.len().max()
                if max_length > 10000:
                    result["warnings"].append(
                        f"Very long strings in column {col} (max: {max_length})"
                    )

        return result

    def _check_expected_columns(
        self, df: pd.DataFrame, expected_columns: List[str]
    ) -> Dict[str, Any]:
        """Check if data has expected columns."""
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            result["errors"].append(
                f"Missing expected columns: {list(missing_columns)}"
            )

        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            result["warnings"].append(f"Unexpected columns: {list(extra_columns)}")

        return result

    def _check_data_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for potential data leakage."""
        result: Dict[str, List[str]] = {"errors": [], "warnings": []}

        # Check for ID columns that might leak information
        id_columns = [
            col for col in df.columns if "id" in col.lower() or "key" in col.lower()
        ]
        if id_columns:
            result["warnings"].append(f"Potential ID columns detected: {id_columns}")

        # Check for timestamp columns
        timestamp_columns = [
            col for col in df.columns if "time" in col.lower() or "date" in col.lower()
        ]
        if timestamp_columns:
            result["warnings"].append(
                f"Timestamp columns detected: {timestamp_columns}"
            )

        return result

    def sanitize_input(
        self, data: Union[Dict, pd.DataFrame]
    ) -> Union[Dict, pd.DataFrame]:
        """
        Sanitize input data by removing or encoding potentially dangerous content.

        Args:
            data: Input data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if isinstance(value, str):
                    # Basic HTML encoding
                    sanitized[key] = value.replace("<", "&lt;").replace(">", "&gt;")
                else:
                    sanitized[key] = value
            return sanitized
        else:
            # For DataFrame, sanitize string columns
            df = data.copy()
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("<", "&lt;")
                    .str.replace(">", "&gt;")
                )
            return df

    def create_secure_hash(self, data: str) -> str:
        """Create a secure hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()

    def create_hmac_signature(self, data: str) -> str:
        """Create HMAC signature for data integrity."""
        return hmac.new(
            self.secret_key.encode(), data.encode(), hashlib.sha256
        ).hexdigest()

    def verify_hmac_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.create_hmac_signature(data)
        return hmac.compare_digest(expected_signature, signature)

    def get_security_report(self) -> Dict[str, Any]:
        """Get security status report."""
        return {
            "api_keys_count": len(self.api_keys),
            "rate_limits_active": len(self.rate_limits),
            "secret_key_configured": bool(self.secret_key),
            "timestamp": datetime.now().isoformat(),
        }

    def audit_log(
        self,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security-related actions."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": user_id,
            "details": details or {},
        }

        logger.info(f"Security audit: {json.dumps(log_entry)}")
