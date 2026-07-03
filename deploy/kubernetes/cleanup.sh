#!/bin/bash

# Context-Engine Kubernetes Cleanup Script
# This script removes all Context-Engine resources from Kubernetes

set -e

# Configuration
NAMESPACE="context-engine"
FORCE=false

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

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Kubernetes connection verified"
}

# Check if namespace exists
check_namespace() {
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist"
        return 1
    fi
    return 0
}

# Show what will be deleted
show_deletion_plan() {
    log_info "The following resources will be deleted:"
    echo

    # Show current resources
    echo "Pods:"
    kubectl get pods -n $NAMESPACE 2>/dev/null || echo "  No pods found"
    echo
    echo "Services:"
    kubectl get services -n $NAMESPACE 2>/dev/null || echo "  No services found"
    echo
    echo "Deployments:"
    kubectl get deployments -n $NAMESPACE 2>/dev/null || echo "  No deployments found"
    echo
    echo "StatefulSets:"
    kubectl get statefulsets -n $NAMESPACE 2>/dev/null || echo "  No statefulsets found"
    echo
    echo "Jobs:"
    kubectl get jobs -n $NAMESPACE 2>/dev/null || echo "  No jobs found"
    echo
    echo "PersistentVolumeClaims:"
    kubectl get pvc -n $NAMESPACE 2>/dev/null || echo "  No PVCs found"
    echo
    echo "ConfigMaps:"
    kubectl get configmaps -n $NAMESPACE 2>/dev/null || echo "  No configmaps found"
    echo
    if kubectl get ingress -n $NAMESPACE &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n $NAMESPACE
        echo
    fi

    log_warning "This will permanently delete all data in Qdrant and any other persistent storage!"
}

confirm_cleanup() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    read -p "Are you sure you want to delete all Context-Engine resources? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
}

# Delete namespace and all resources
delete_namespace() {
    log_info "Deleting namespace: $NAMESPACE"
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    log_success "Namespace deleted"
}

# Wait for namespace deletion
wait_for_deletion() {
    log_info "Waiting for namespace deletion to complete..."

    local timeout=60
    local count=0

    while kubectl get namespace $NAMESPACE &> /dev/null; do
        if [[ $count -ge $timeout ]]; then
            log_warning "Namespace deletion is taking longer than expected"
            log_info "You may need to manually delete remaining resources"
            return 1
        fi

        echo -n "."
        sleep 1
        ((count++))
    done

    echo
    log_success "Namespace deletion completed"
}

# Force delete if needed
force_delete() {
    log_warning "Attempting to force delete remaining resources..."

    # Force delete any remaining pods
    kubectl delete pods --all -n $NAMESPACE --grace-period=0 --force 2>/dev/null || true

    # Force delete any remaining PVCs
    kubectl delete pvc --all -n $NAMESPACE --grace-period=0 --force 2>/dev/null || true

    log_success "Force delete completed"
}

# Verify cleanup
verify_cleanup() {
    log_info "Verifying cleanup..."

    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_error "Namespace $NAMESPACE still exists"
        return 1
    fi

    log_success "Cleanup completed successfully"
}

# Main cleanup function
main() {
    log_info "Starting Context-Engine Kubernetes cleanup"

    # Check prerequisites
    check_kubectl

    # Check if namespace exists
    if ! check_namespace; then
        log_success "Nothing to clean up - namespace $NAMESPACE does not exist"
        exit 0
    fi

    # Show what will be deleted
    show_deletion_plan

    # Ask for confirmation (unless forced)
    confirm_cleanup

    # Delete namespace
    delete_namespace

    # Wait for deletion
    if ! wait_for_deletion; then
        log_warning "Standard deletion incomplete, attempting force delete..."
        force_delete
    fi

    # Verify cleanup
    verify_cleanup

    log_success "Context-Engine cleanup completed!"
}

# Help function
show_help() {
    echo "Context-Engine Kubernetes Cleanup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -n, --namespace NAMESPACE     Kubernetes namespace (default: context-engine)"
    echo "  -f, --force                   Skip confirmation prompt"
    echo
    echo "Environment variables:"
    echo "  NAMESPACE=context-engine      Kubernetes namespace"
    echo
    echo "Examples:"
    echo "  $0                            # Interactive cleanup with confirmation"
    echo "  $0 --force                    # Cleanup without confirmation"
    echo "  $0 -n my-namespace            # Cleanup different namespace"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -f|--force|--force=true)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "qdrant.yaml" ]]; then
    log_error "Please run this script from the deploy/kubernetes directory"
    exit 1
fi

# Run main cleanup
main
