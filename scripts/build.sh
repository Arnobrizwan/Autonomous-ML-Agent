#!/bin/bash
set -euo pipefail

# Optimized build script for Autonomous ML Agent Docker image
# Usage: ./scripts/build.sh [tag] [--no-cache] [--test]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
TAG="${1:-aml-agent:latest}"
NO_CACHE="${2:-}"
TEST="${3:-true}"
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

# Build function with optimizations
build_image() {
    local tag="$1"
    local no_cache="$2"
    local start_time=$(date +%s)
    
    log_info "Starting optimized Docker build..."
    log_info "Tag: $tag"
    log_info "Dockerfile: $DOCKERFILE"
    
    # Enable BuildKit for parallel builds and caching
    export DOCKER_BUILDKIT=1
    
    # Build arguments
    local build_args=""
    if [ "$no_cache" = "--no-cache" ]; then
        build_args="--no-cache"
        log_info "Building without cache"
    else
        build_args="--cache-from $tag"
        log_info "Building with cache optimization"
    fi
    
    # Build the image with optimizations
    if docker build \
        $build_args \
        -f "$DOCKERFILE" \
        -t "$tag" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        "$PROJECT_ROOT"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        log_success "Build completed successfully in ${minutes}m ${seconds}s"
        
        # Show image details
        log_info "Image details:"
        docker images "$tag" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        # Show layer analysis
        log_info "Layer analysis:"
        docker history "$tag" --format "table {{.Size}}\t{{.CreatedBy}}" | head -5
        
    else
        log_error "Build failed!"
        exit 1
    fi
}

# Test function
test_image() {
    local tag="$1"
    
    log_info "Testing image: $tag"
    
    # Test import
    if docker run --rm "$tag" python -c "import aml_agent; print('Import successful')"; then
        log_success "Import test passed"
    else
        log_error "Import test failed"
        exit 1
    fi
    
    # Test FastAPI app creation
    if docker run --rm "$tag" python -c "from aml_agent.service.app import create_app; print('FastAPI app creation successful')"; then
        log_success "FastAPI app test passed"
    else
        log_error "FastAPI app test failed"
        exit 1
    fi
}

# Performance analysis
analyze_performance() {
    local tag="$1"
    
    log_info "Performance analysis:"
    
    # Image size
    local size=$(docker images --format "{{.Size}}" "$tag")
    log_info "Image size: $size"
    
    # Layer count
    local layers=$(docker history "$tag" --format "{{.CreatedBy}}" | wc -l)
    log_info "Layer count: $layers"
    
    # Largest layers
    log_info "Largest layers:"
    docker history "$tag" --format "table {{.Size}}\t{{.CreatedBy}}" | head -3
}

# Cleanup function
cleanup() {
    log_info "Cleaning up dangling images..."
    docker image prune -f
}

# Main execution
main() {
    log_info "Autonomous ML Agent Optimized Build Script"
    log_info "=========================================="
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_docker
    
    # Build image
    build_image "$TAG" "$NO_CACHE"
    
    # Test image if requested
    if [ "$TEST" = "true" ]; then
        test_image "$TAG"
    fi
    
    # Analyze performance
    analyze_performance "$TAG"
    
    # Cleanup
    cleanup
    
    log_success "Build process completed successfully!"
    log_info "To run the container:"
    log_info "  docker run --rm -p 8000:8000 $TAG"
    log_info "To run with docker-compose:"
    log_info "  docker-compose up"
}

# Run main function
main "$@"
