#!/bin/bash
# GPU Decoder Toggle Script
# Manages switching between Docker CPU-only and native GPU-accelerated decoders

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status    Show current decoder configuration"
    echo "  docker    Switch to Docker CPU-only decoder (stable)"
    echo "  gpu       Switch to native GPU-accelerated decoder (fast)"
    echo "  start     Start native GPU decoder on port 8081"
    echo "  stop      Stop native GPU decoder"
    echo "  test      Test current decoder configuration"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status                 # Check current setup"
    echo "  $0 gpu && $0 start        # Switch to GPU and start native server"
    echo "  $0 docker                 # Switch back to Docker CPU-only"
}

get_env_value() {
    local key="$1"
    local default="${2:-}"
    
    if [[ -f "$ENV_FILE" ]]; then
        grep "^${key}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '"' || echo "$default"
    else
        echo "$default"
    fi
}

set_env_value() {
    local key="$1"
    local value="$2"
    
    if [[ -f "$ENV_FILE" ]]; then
        # Update existing value or add new one
        if grep -q "^${key}=" "$ENV_FILE"; then
            sed -i.bak "s/^${key}=.*/${key}=${value}/" "$ENV_FILE"
        else
            echo "${key}=${value}" >> "$ENV_FILE"
        fi
    else
        echo "${key}=${value}" > "$ENV_FILE"
    fi
    echo -e "${GREEN}Set ${key}=${value}${NC}"
}

show_status() {
    echo -e "${BLUE}Current Decoder Configuration${NC}"
    echo ""
    
    local use_gpu=$(get_env_value "USE_GPU_DECODER" "0")
    local llamacpp_url=$(get_env_value "LLAMACPP_URL" "http://llamacpp:8080")
    
    if [[ "$use_gpu" == "1" ]]; then
        echo -e "Mode: ${GREEN}GPU-Accelerated${NC} (native llama.cpp with Metal)"
        echo -e "URL:  ${GREEN}http://host.docker.internal:8081${NC}"
        echo ""
        
        # Check if native server is running
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null | grep -q "200"; then
            echo -e "Status: ${GREEN}Native GPU server is running${NC}"
        else
            echo -e "Status: ${RED}Native GPU server is not running${NC}"
            echo -e "       ${YELLOW}Run: $0 start${NC}"
        fi
    else
        echo -e "Mode: ${YELLOW}Docker CPU-Only${NC} (stable, containerized)"
        echo -e "URL:  ${YELLOW}${llamacpp_url}${NC}"
        echo ""
        
        # Check if Docker container is running
        if docker ps --format '{{.Names}}' | grep -q "llama-decoder"; then
            echo -e "Status: ${GREEN}Docker container is running${NC}"
        else
            echo -e "Status: ${RED}Docker container is not running${NC}"
            echo -e "       ${YELLOW}Run: docker compose up llamacpp -d${NC}"
        fi
    fi
}

switch_to_docker() {
    echo -e "${BLUE}Switching to Docker CPU-only decoder${NC}"
    set_env_value "USE_GPU_DECODER" "0"
    echo ""
    echo -e "${GREEN}Switched to Docker mode${NC}"
    echo -e "   - Stable and containerized"
    echo -e "   - CPU-only inference"
    echo -e "   - Uses Docker service: llamacpp:8080"
    echo ""
    echo -e "${YELLOW}Restart your indexer services to apply changes:${NC}"
    echo -e "   docker compose restart mcp_indexer mcp_indexer_http"
}

switch_to_gpu() {
    echo -e "${BLUE}Switching to native GPU-accelerated decoder${NC}"
    set_env_value "USE_GPU_DECODER" "1"
    echo ""
    echo -e "${GREEN}Switched to GPU mode${NC}"
    echo -e "   - Metal GPU acceleration (Apple Silicon)"
    echo -e "   - Significantly faster inference"
    echo -e "   - Uses native server: localhost:8081"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "   1. Start native server: $0 start"
    echo -e "   2. Restart indexer: docker compose restart mcp_indexer mcp_indexer_http"
}

start_native_server() {
    echo -e "${BLUE}Starting native GPU-accelerated decoder${NC}"
    
    # Check if already running
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null | grep -q "200"; then
        echo -e "${YELLOW}Native server is already running on port 8081${NC}"
        return 0
    fi

    # Check if llama-server is available
    if ! command -v llama-server &> /dev/null; then
        echo -e "${RED}llama-server not found${NC}"
        echo -e "   Install with: brew install llama.cpp"
        return 1
    fi

    # Check if model exists
    local model_path="$PROJECT_ROOT/models/model.gguf"
    if [[ ! -f "$model_path" ]]; then
        echo -e "${RED}Model not found: $model_path${NC}"
        return 1
    fi

    echo -e "${GREEN}Starting native llama-server with GPU acceleration...${NC}"
    echo -e "   Model: $model_path"
    echo -e "   GPU Layers: 32"
    echo -e "   Port: 8081"
    echo ""
    
    # Start in background
    nohup llama-server \
        --model "$model_path" \
        --host 0.0.0.0 \
        --port 8081 \
        --n-gpu-layers 32 \
        --ctx-size 8192 \
        --no-warmup \
        > "$PROJECT_ROOT/llamacpp-gpu.log" 2>&1 &
    
    local pid=$!
    echo -e "${GREEN}Started native server (PID: $pid)${NC}"
    echo -e "   Logs: $PROJECT_ROOT/llamacpp-gpu.log"
    echo -e "   Health: http://localhost:8081/health"
    
    # Wait a moment and check if it started successfully
    sleep 3
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health 2>/dev/null | grep -q "200"; then
        echo -e "${GREEN}Server is healthy and ready${NC}"
    else
        echo -e "${YELLOW}Server may still be starting up. Check logs if issues persist.${NC}"
    fi
}

stop_native_server() {
    echo -e "${BLUE}Stopping native GPU decoder${NC}"
    
    # Find and kill llama-server processes on port 8081
    local pids=$(lsof -ti:8081 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo -e "${GREEN}Stopping processes: $pids${NC}"
        kill $pids
        sleep 2
        
        # Force kill if still running
        local remaining=$(lsof -ti:8081 2>/dev/null || true)
        if [[ -n "$remaining" ]]; then
            echo -e "${YELLOW}Force killing remaining processes: $remaining${NC}"
            kill -9 $remaining
        fi
        
        echo -e "${GREEN}Native server stopped${NC}"
    else
        echo -e "${YELLOW}No native server found running on port 8081${NC}"
    fi
}

test_decoder() {
    echo -e "${BLUE}Testing current decoder configuration${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    python test_gpu_switch.py
}

# Main command handling
case "${1:-help}" in
    "status")
        show_status
        ;;
    "docker")
        switch_to_docker
        ;;
    "gpu")
        switch_to_gpu
        ;;
    "start")
        start_native_server
        ;;
    "stop")
        stop_native_server
        ;;
    "test")
        test_decoder
        ;;
    "help"|*)
        usage
        ;;
esac
