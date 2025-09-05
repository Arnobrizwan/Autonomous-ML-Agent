# Optimized Dockerfile for Autonomous ML Agent
# Single-stage build for maximum reliability and speed

FROM python:3.11-slim

# Set labels for metadata
LABEL maintainer="Autonomous ML Agent Team" \
      version="1.0.0" \
      description="Autonomous ML Agent - Optimized Production Image" \
      org.opencontainers.image.source="https://github.com/Arnobrizwan/Autonomous-ML-Agent"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=3

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with retry and timeout handling
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=100 --retries=3 -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/artifacts

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add health check (using Python instead of curl to avoid system dependencies)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/healthz')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.aml_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
