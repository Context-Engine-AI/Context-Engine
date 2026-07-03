# Unified Context-Engine image for Kubernetes deployment
# Supports multiple roles: memory, indexer, watcher, llamacpp
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORK_ROOTS="/work,/app"

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps: reuse shared requirements file for consistency across services
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Copy scripts for all services
COPY scripts /app/scripts

# Create directories
WORKDIR /work

# Expose all necessary ports
EXPOSE 8000 8001 8002 8003 18000 18001 18002 18003

# Default to memory server
CMD ["python", "/app/scripts/mcp_memory_server.py"]