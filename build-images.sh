#!/bin/bash
# Docker Build Script for Context-Engine
# Builds all service images with custom registry tagging

set -euo pipefail

# Configuration
REGISTRY="192.168.96.61:30009/library"
PROJECT_NAME="context-engine"
TAG="${TAG:-latest}"

# Service mapping (service_name:dockerfile:final_image_name)
declare -A SERVICES=(
    ["memory"]="Dockerfile.mcp:${PROJECT_NAME}-memory"
    ["indexer"]="Dockerfile.mcp-indexer:${PROJECT_NAME}-indexer"
    ["indexer-service"]="Dockerfile.indexer:${PROJECT_NAME}-indexer-service"
    ["llamacpp"]="Dockerfile.llamacpp:${PROJECT_NAME}-llamacpp"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Build function
build_image() {
    local service=$1
    local dockerfile=$2
    local image_name=$3
    local full_image="${REGISTRY}/${image_name}:${TAG}"

    log_info "Building ${service} service..."
    log_info "Dockerfile: ${dockerfile}"
    log_info "Image: ${full_image}"

    if ! docker build \
        -f "${dockerfile}" \
        -t "${full_image}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .; then
        log_error "Failed to build ${service} image"
        return 1
    fi

    log_info "Successfully built ${service} image: ${full_image}"

    # Push if registry is accessible
    if [[ "${PUSH_IMAGES:-false}" == "true" ]]; then
        log_info "Pushing ${service} image..."
        if ! docker push "${full_image}"; then
            log_warn "Failed to push ${service} image (registry may be inaccessible)"
            return 1
        fi
        log_info "Successfully pushed ${service} image"
    fi

    echo "${full_image}"
}

# Main build process
main() {
    log_info "Starting Context-Engine Docker build process..."
    log_info "Registry: ${REGISTRY}"
    log_info "Tag: ${TAG}"
    log_info "Push enabled: ${PUSH_IMAGES:-false}"
    echo

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi

    # Check if Dockerfiles exist
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r dockerfile image_name <<< "${SERVICES[$service]}"
        if [[ ! -f "${dockerfile}" ]]; then
            log_error "Dockerfile not found: ${dockerfile}"
            exit 1
        fi
    done

    local built_images=()
    local failed_services=()

    # Build each service
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r dockerfile image_name <<< "${SERVICES[$service]}"

        if built_image=$(build_image "$service" "$dockerfile" "$image_name"); then
            built_images+=("$built_image")
        else
            failed_services+=("$service")
        fi
        echo
    done

    # Summary
    log_info "Build Summary:"
    log_info "Successfully built: ${#built_images[@]} images"
    for img in "${built_images[@]}"; do
        log_info "  ✓ ${img}"
    done

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Failed to build: ${#failed_services[@]} services"
        for service in "${failed_services[@]}"; do
            log_error "  ✗ ${service}"
        done
        exit 1
    fi

    log_info "All images built successfully!"

    # Generate updated kustomization.yaml
    cat > "deploy/kubernetes/kustomization-images.yaml" << 'EOF'
# Image overrides for Context-Engine Kubernetes deployment
# Use this with: kustomize build . --load-restrictor=LoadRestrictionsNone | kubectl apply -f -
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - namespace.yaml
  - configmap.yaml
  - qdrant.yaml
  - mcp-memory.yaml
  - mcp-indexer.yaml
  - mcp-http.yaml
  - indexer-services.yaml
  - llamacpp.yaml
  - ingress.yaml

images:
EOF

    # Add images to kustomization
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r dockerfile image_name <<< "${SERVICES[$service]}"
        full_image="${REGISTRY}/${image_name}:${TAG}"
        cat >> "deploy/kubernetes/kustomization-images.yaml" << EOF
  - name: ${image_name}
    newName: ${full_image%:*}  # Remove tag
    newTag: ${TAG}
EOF
    done

    cat >> "deploy/kubernetes/kustomization-images.yaml" << 'EOF'

# Common labels
commonLabels:
  app.kubernetes.io/name: context-engine
  app.kubernetes.io/component: kubernetes-deployment
  app.kubernetes.io/managed-by: kustomize

# Namespace override
namespace: context-engine
EOF

    log_info "Generated deploy/kubernetes/kustomization-images.yaml"
    log_info "To deploy: kustomize build deploy/kubernetes/ | kubectl apply -f -"
}

# Help function
show_help() {
    cat << EOF
Context-Engine Docker Build Script

Usage: $0 [OPTIONS]

Options:
  -t, --tag TAG          Set image tag (default: latest)
  -p, --push             Push images to registry after build
  -h, --help             Show this help message

Examples:
  $0                                    # Build with default tag
  $0 -t v1.0.0                         # Build with custom tag
  $0 --push                            # Build and push to registry
  TAG=dev-branch $0                    # Build using environment variable

Environment Variables:
  TAG                Image tag to use
  PUSH_IMAGES        Set to 'true' to push after build

Registry Configuration:
  Current registry: ${REGISTRY}
  To change: modify REGISTRY variable in script

Generated Files:
  - deploy/kubernetes/kustomization-images.yaml
    Contains image references for Kubernetes deployment

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            export PUSH_IMAGES=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"