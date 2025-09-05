#!/bin/bash
set -euo pipefail

# Production build script with optimizations
# Usage: ./scripts/build-prod.sh [tag] [push]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
TAG="${1:-aml-agent:latest}"
PUSH="${2:-false}"
DOCKERFILE="docker/Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Build production image with optimizations
build_production() {
    local tag="$1"
    local start_time=$(date +%s)
    
    log_info "Building production image with optimizations..."
    log_info "Tag: $tag"
    
    # Enable BuildKit for parallel builds and caching
    export DOCKER_BUILDKIT=1
    
    # Build with cache optimization
    if docker build \
        --target production \
        -f "$DOCKERFILE" \
        -t "$tag" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from "$tag" \
        --progress=plain \
        "$PROJECT_ROOT"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        log_success "Production build completed in ${minutes}m ${seconds}s"
        
        # Show image details
        log_info "Image details:"
        docker images "$tag" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        # Show layer sizes
        log_info "Layer analysis:"
        docker history "$tag" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
        
    else
        log_error "Production build failed!"
        exit 1
    fi
}

# Test production image
test_production() {
    local tag="$1"
    
    log_info "Testing production image..."
    
    # Test basic functionality
    if docker run --rm "$tag" python -c "import aml_agent; print('Import successful')"; then
        log_success "Import test passed"
    else
        log_error "Import test failed"
        exit 1
    fi
    
    # Test FastAPI app
    if docker run --rm "$tag" python -c "from aml_agent.service.app import create_app; print('FastAPI app creation successful')"; then
        log_success "FastAPI app test passed"
    else
        log_error "FastAPI app test failed"
        exit 1
    fi
    
    # Test health check
    log_info "Testing health check..."
    if timeout 30s docker run --rm -d --name aml-test "$tag" && \
       sleep 10 && \
       docker exec aml-test curl -f http://localhost:8000/healthz >/dev/null 2>&1; then
        log_success "Health check test passed"
        docker stop aml-test >/dev/null 2>&1 || true
    else
        log_error "Health check test failed"
        docker stop aml-test >/dev/null 2>&1 || true
        exit 1
    fi
}

# Push image to registry
push_image() {
    local tag="$1"
    
    log_info "Pushing image to registry..."
    
    if docker push "$tag"; then
        log_success "Image pushed successfully"
    else
        log_error "Failed to push image"
        exit 1
    fi
}

# Security scan
security_scan() {
    local tag="$1"
    
    log_info "Running security scan..."
    
    # Check if trivy is available
    if command -v trivy >/dev/null 2>&1; then
        log_info "Running Trivy security scan..."
        trivy image --severity HIGH,CRITICAL "$tag" || log_warning "Security scan found issues"
    else
        log_warning "Trivy not available, skipping security scan"
    fi
}

# Main execution
main() {
    log_info "Autonomous ML Agent Production Build"
    log_info "===================================="
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_docker
    
    # Build production image
    build_production "$TAG"
    
    # Test image
    test_production "$TAG"
    
    # Security scan
    security_scan "$TAG"
    
    # Push if requested
    if [ "$PUSH" = "true" ]; then
        push_image "$TAG"
    fi
    
    log_success "Production build process completed successfully!"
    log_info "Image: $TAG"
    log_info "Size: $(docker images --format "table {{.Size}}" "$TAG" | tail -n 1)"
}

# Run main function
main "$@"
