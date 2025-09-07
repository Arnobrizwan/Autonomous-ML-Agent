# Ultra-Optimized Multi-Stage Dockerfile for Autonomous ML Agent
# Target: 50-70% size reduction (from ~833MB to ~250-400MB)
# Strategy: Multi-stage build, minimal runtime, optimized dependencies

# =============================================================================
# STAGE 1: Build Dependencies (Heavy, temporary)
# =============================================================================
FROM python:3.11-slim AS builder

# Set build environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UCF_FORCE_CONFFNEW=1

# Install minimal build dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/locale/* \
    && find /usr -name "*.pyc" -delete \
    && find /usr -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files first for better layer caching
COPY requirements.txt ./

# Install Python packages with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    httpx==0.25.2 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    pyyaml==6.0.1 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    typer[all]==0.9.0 \
    rich==13.7.0 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    xgboost==2.0.2 \
    lightgbm==4.1.0 \
    catboost==1.2.2 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    optuna==3.4.0 \
    shap==0.43.0 \
    eli5==0.13.0 \
    imbalanced-learn==0.11.0 \
    feature-engine==1.6.0 \
    mlxtend==0.22.0 && \
    pip install --no-cache-dir --prefer-binary --no-compile \
    plotly==5.17.0 \
    streamlit==1.28.1 \
    psutil==5.9.6 \
    openpyxl==3.1.2 \
    pyarrow==14.0.1

# Copy source code and install application
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-compile -e .

# Clean up build artifacts and optimize Python installation
RUN find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyo" -delete && \
    find /opt/venv -name "*.pyd" -delete && \
    find /opt/venv -name "test*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# STAGE 2: Runtime Dependencies (Minimal system packages)
# =============================================================================
FROM python:3.11-slim AS runtime-deps

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Install only essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    libgfortran5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/locale/* \
    && find /usr -name "*.pyc" -delete \
    && find /usr -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# STAGE 3: Production Runtime (Ultra-minimal)
# =============================================================================
FROM runtime-deps AS production

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code (minimal)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
WORKDIR /app

# Install application in production mode
RUN pip install --no-cache-dir --no-compile -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/artifacts /app/logs && \
    chmod 755 /app/data /app/artifacts /app/logs

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "aml_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# STAGE 4: Development (Includes dev tools)
# =============================================================================
FROM runtime-deps AS development

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and tests
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
WORKDIR /app

# Install application in development mode
RUN pip install --no-cache-dir --no-compile -e .

# Install development dependencies
RUN pip install --no-cache-dir --prefer-binary --no-compile \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    pytest-xdist==3.3.1 \
    black==23.11.0 \
    isort==5.12.0 \
    flake8==6.1.0

# Create necessary directories
RUN mkdir -p /app/data /app/artifacts /app/logs

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "aml_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# STAGE 5: Test (Includes test dependencies)
# =============================================================================
FROM runtime-deps AS test

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and tests
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
WORKDIR /app

# Install application
RUN pip install --no-cache-dir --no-compile -e .

# Install test dependencies
RUN pip install --no-cache-dir --prefer-binary --no-compile \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    pytest-xdist==3.3.1 \
    pytest-benchmark==4.0.0 \
    factory-boy==3.3.0 \
    faker==20.1.0 \
    responses==0.24.1

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]