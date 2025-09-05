"""
Authentication and authorization for the Autonomous ML Agent.
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from ..logging import get_logger

logger = get_logger()


class APIKeyManager:
    """Manage API keys for authentication."""

    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.key_permissions: Dict[str, Set[str]] = {}

    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """
        Generate a new API key.

        Args:
            name: Name/description for the key
            permissions: List of permissions for the key

        Returns:
            Generated API key
        """
        if permissions is None:
            permissions = ["read", "predict"]

        # Generate secure random key
        api_key = secrets.token_urlsafe(32)

        # Store key metadata
        self.api_keys[api_key] = {
            "name": name,
            "created_at": datetime.now(),
            "last_used": None,
            "is_active": True,
            "permissions": set(permissions),
        }

        self.key_permissions[api_key] = set(permissions)

        logger.info(f"Generated API key for {name}")
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        if api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]

        # Check if key is active
        if not key_info["is_active"]:
            return False

        # Update last used
        key_info["last_used"] = datetime.now()

        return True

    def check_permission(self, api_key: str, permission: str) -> bool:
        """
        Check if API key has specific permission.

        Args:
            api_key: API key
            permission: Permission to check

        Returns:
            True if has permission, False otherwise
        """
        if not self.validate_api_key(api_key):
            return False

        return permission in self.key_permissions.get(api_key, set())

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if revoked, False if not found
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            logger.info(f"Revoked API key: {api_key[:8]}...")
            return True
        return False

    def list_api_keys(self) -> List[Dict]:
        """
        List all API keys (without the actual keys).

        Returns:
            List of key metadata
        """
        return [
            {
                "name": info["name"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "is_active": info["is_active"],
                "permissions": list(info["permissions"]),
            }
            for info in self.api_keys.values()
        ]


class RateLimiter:
    """Rate limiting for API endpoints."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed.

        Args:
            identifier: Unique identifier (IP, API key, etc.)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time
                for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []

        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds

        if identifier in self.requests:
            recent_requests = [
                req_time
                for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            return max(0, self.max_requests - len(recent_requests))

        return self.max_requests


class SecurityManager:
    """Main security manager."""

    def __init__(self):
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
        self.enabled = True

    def authenticate_request(
        self, api_key: Optional[str] = None, ip_address: Optional[str] = None
    ) -> bool:
        """
        Authenticate a request.

        Args:
            api_key: API key if provided
            ip_address: IP address for rate limiting

        Returns:
            True if authenticated, False otherwise
        """
        if not self.enabled:
            return True

        # Check rate limiting
        if ip_address and not self.rate_limiter.is_allowed(ip_address):
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return False

        # Check API key if provided
        if api_key and not self.api_key_manager.validate_api_key(api_key):
            logger.warning(f"Invalid API key: {api_key[:8] if api_key else 'None'}...")
            return False

        return True

    def check_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has permission."""
        return self.api_key_manager.check_permission(api_key, permission)

    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate new API key."""
        return self.api_key_manager.generate_api_key(name, permissions)

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        return self.api_key_manager.revoke_api_key(api_key)


# Global security manager instance
security_manager = SecurityManager()
