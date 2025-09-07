"""
Pytest configuration for the Autonomous ML Agent tests.
"""

import asyncio

import pytest

# Try to import pytest_asyncio, but don't fail if it's not available
try:
    import pytest_asyncio
    pytest_plugins = ["pytest_asyncio"]
except ImportError:
    pytest_plugins = []


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for anyio."""
    return "asyncio"
