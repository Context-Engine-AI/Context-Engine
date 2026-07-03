#!/bin/bash

# Development Environment Setup Script for Remote Upload System
# This script sets up the development environment for testing the remote upload workflow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEV_WORKSPACE="${DEV_WORKSPACE:-./dev-workspace}"

# Functions
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
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    log_info "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_success "Docker Compose is available"
}

# Create development workspace directory structure
setup_workspace() {
    log_info "Setting up development workspace..."
    
    # Create main workspace directory
    mkdir -p "$DEV_WORKSPACE"
    mkdir -p "$DEV_WORKSPACE/.codebase"
    
    log_success "Development workspace created at $DEV_WORKSPACE"
    log_info "You can mount your existing repositories here for testing"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "Created .env from .env.example"
    else
        log_warning ".env file already exists, skipping creation"
    fi
    
    # Add dev-remote specific configurations if not already present
    if ! grep -q "HOST_INDEX_PATH=./dev-workspace" .env; then
        cat >> .env << 'EOF'

# Development Remote Upload Configuration
HOST_INDEX_PATH=./dev-workspace
DEV_REMOTE_MODE=1
DEV_REMOTE_DEBUG=1

# Upload Service Configuration (Development)
UPLOAD_SERVICE_HOST=0.0.0.0
UPLOAD_SERVICE_PORT=8002
UPLOAD_SERVICE_DEBUG=1

# Remote Upload Client Configuration
REMOTE_UPLOAD_ENABLED=1
REMOTE_UPLOAD_ENDPOINT=http://upload_service:8002
REMOTE_UPLOAD_MAX_RETRIES=3
REMOTE_UPLOAD_TIMEOUT=30
REMOTE_UPLOAD_DEBUG=1

# Development-specific settings
QDRANT_TIMEOUT=60
MAX_MICRO_CHUNKS_PER_FILE=200
INDEX_UPSERT_BATCH=128
INDEX_UPSERT_RETRIES=5
WATCH_DEBOUNCE_SECS=1.5
EOF
        log_success "Added dev-remote configuration to .env"
    else
        log_warning "Dev-remote configuration already exists in .env"
    fi
}

# Print usage information
print_usage() {
    log_info "Development environment setup complete!"
    echo
    echo "Quick Start:"
    echo "  1. Copy your repository to dev-workspace/your-repo-name"
    echo "  2. Run: make dev-remote-bootstrap"
    echo "  3. Test with: make dev-remote-test"
    echo
    echo "Available commands:"
    echo "  make dev-remote-up          - Start the dev-remote stack"
    echo "  make dev-remote-down        - Stop the dev-remote stack"
    echo "  make dev-remote-bootstrap   - Bootstrap the complete system"
    echo "  make dev-remote-test        - Test the remote upload workflow"
    echo "  make dev-remote-client      - Start remote upload client"
    echo "  make dev-remote-clean       - Clean up all dev-remote resources"
    echo
    echo "Service URLs:"
    echo "  Upload Service:     http://localhost:8004"
    echo "  Qdrant Dashboard:   http://localhost:6333"
    echo "  MCP Search:         http://localhost:8000"
    echo "  MCP Indexer:        http://localhost:8001"
    echo
    echo "Testing Workflow:"
    echo "  1. Place your code in: $DEV_WORKSPACE/your-repo"
    echo "  2. Start the stack: make dev-remote-bootstrap"
    echo "  3. Test upload: curl http://localhost:8004/health"
    echo "  4. Check status: curl 'http://localhost:8004/api/v1/delta/status?workspace_path=/work/your-repo'"
    echo
    echo "For remote upload testing:"
    echo "  1. Set REMOTE_UPLOAD_ENDPOINT=http://localhost:8004"
    echo "  2. Run: make watch-remote REMOTE_UPLOAD_ENDPOINT=http://localhost:8004"
    echo
    log_success "Ready to test the remote upload system!"
}

# Main execution
main() {
    log_info "Setting up development environment for remote upload system..."
    
    check_docker
    check_docker_compose
    setup_workspace
    create_env_file
    print_usage
    
    log_success "Development environment setup completed successfully!"
}

# Run main function
main "$@"