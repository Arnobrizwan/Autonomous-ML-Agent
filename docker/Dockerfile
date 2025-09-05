# Ultra-Optimized Multi-Stage Dockerfile for Autonomous ML Agent
# Stage 1: Build dependencies and install packages
FROM python:3.11-slim AS builder

# Set environment variables for build stage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only essential Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    typer[all]==0.9.0 \
    rich==13.7.0 \
    pyyaml==6.0.1 \
    joblib==1.3.2 \
    click==8.1.7

# Install application
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime (ultra-slim)
FROM python:3.11-slim

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Copy only the virtual environment (no build tools)
COPY --from=builder /opt/venv /opt/venv

# Copy application code and install package
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
WORKDIR /app

# Install the package in the runtime environment
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/artifacts

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "aml_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000"]