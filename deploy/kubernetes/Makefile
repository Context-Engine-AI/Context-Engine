# Context-Engine Kubernetes Deployment Makefile

# Configuration
NAMESPACE ?= context-engine
IMAGE_REGISTRY ?= context-engine
IMAGE_TAG ?= latest

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Context-Engine Kubernetes Deployment Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Prerequisites
.PHONY: check-kubectl
check-kubectl: ## Check if kubectl is available and cluster is accessible
	@which kubectl > /dev/null || (echo "kubectl not found. Please install kubectl." && exit 1)
	@kubectl cluster-info > /dev/null || (echo "Cannot connect to Kubernetes cluster." && exit 1)
	@echo "✓ Kubernetes connection verified"

# Deployment targets
.PHONY: deploy
deploy: check-kubectl ## Deploy all Context-Engine services
	./deploy.sh --namespace $(NAMESPACE) --registry $(IMAGE_REGISTRY) --tag $(IMAGE_TAG)

.PHONY: deploy-core
deploy-core: check-kubectl ## Deploy only core services (Qdrant + MCP servers)
	@echo "Deploying core services..."
	kubectl apply -f namespace.yaml
	kubectl apply -f configmap.yaml
	kubectl apply -f qdrant.yaml
	kubectl apply -f mcp-memory.yaml
	kubectl apply -f mcp-indexer.yaml

.PHONY: deploy-full
deploy-full: check-kubectl ## Deploy all services including optional ones
	./deploy.sh --namespace $(NAMESPACE) --registry $(IMAGE_REGISTRY) --tag $(IMAGE_TAG) --deploy-ingress

.PHONY: deploy-minimal
deploy-minimal: check-kubectl ## Deploy minimal setup (skip Llama.cpp and Ingress)
	./deploy.sh --namespace $(NAMESPACE) --registry $(IMAGE_REGISTRY) --tag $(IMAGE_TAG) --skip-llamacpp

# Kustomize targets
.PHONY: kustomize-build
kustomize-build: ## Build manifests with Kustomize
	kustomize build .

.PHONY: kustomize-apply
kustomize-apply: check-kubectl ## Apply manifests with Kustomize
	kustomize build . | kubectl apply -f -

.PHONY: kustomize-delete
kustomize-delete: check-kubectl ## Delete manifests with Kustomize
	kustomize build . | kubectl delete -f -
# Management targets
.PHONY: status
status: check-kubectl ## Show deployment status
	@echo "=== Namespace Status ==="
	kubectl get namespace $(NAMESPACE) || echo "Namespace $(NAMESPACE) not found"
	@echo ""
	@echo "=== Pods ==="
	kubectl get pods -n $(NAMESPACE) -o wide || echo "No pods found"
	@echo ""
	@echo "=== Services ==="
	kubectl get services -n $(NAMESPACE) || echo "No services found"
	@echo ""
	@echo "=== Deployments ==="
	kubectl get deployments -n $(NAMESPACE) || echo "No deployments found"
	@echo ""
	@echo "=== StatefulSets ==="
	kubectl get statefulsets -n $(NAMESPACE) || echo "No statefulsets found"
	@echo ""
	@echo "=== PersistentVolumeClaims ==="
	kubectl get pvc -n $(NAMESPACE) || echo "No PVCs found"
	@echo ""
	@echo "=== Jobs ==="
	kubectl get jobs -n $(NAMESPACE) || echo "No jobs found"

.PHONY: logs
logs: check-kubectl ## Show logs for core services (tail 100)
	@echo "=== Qdrant Logs ==="
	kubectl logs -f statefulset/qdrant -n $(NAMESPACE) --tail=100 || echo "Qdrant logs not available"
	@echo ""
	@echo "=== MCP Memory Logs ==="
	kubectl logs -f deployment/mcp-memory -n $(NAMESPACE) --tail=100 || echo "MCP Memory logs not available"
	@echo ""
	@echo "=== MCP Indexer Logs ==="
	kubectl logs -f deployment/mcp-indexer -n $(NAMESPACE) --tail=100 || echo "MCP Indexer logs not available"
	@echo ""
	@echo "=== Watcher Logs ==="
	kubectl logs -f deployment/watcher -n $(NAMESPACE) --tail=100 || echo "Watcher logs not available"

.PHONY: logs-service
logs-service: check-kubectl ## Show logs for specific service (usage: make logs-service SERVICE=mcp-memory)
	@if [ -z "$(SERVICE)" ]; then echo "Usage: make logs-service SERVICE=<service-name>"; exit 1; fi
	kubectl logs -f deployment/$(SERVICE) -n $(NAMESPACE) --tail=100 || kubectl logs -f statefulset/$(SERVICE) -n $(NAMESPACE) --tail=100 || kubectl logs -f job/$(SERVICE) -n $(NAMESPACE) --tail=100 || echo "Service $(SERVICE) not found"

.PHONY: shell
shell: check-kubectl ## Get a shell in a running pod (usage: make shell POD=mcp-memory-xxx)
	@if [ -z "$(POD)" ]; then echo "Usage: make shell POD=<pod-name>"; echo "Available pods:"; kubectl get pods -n $(NAMESPACE); exit 1; fi
	kubectl exec -it $(POD) -n $(NAMESPACE) -- /bin/bash || kubectl exec -it $(POD) -n $(NAMESPACE) -- /bin/sh

# Cleanup targets
.PHONY: cleanup
cleanup: check-kubectl ## Remove all Context-Engine resources
	./cleanup.sh --namespace $(NAMESPACE)

.PHONY: clean-force
clean-force: check-kubectl ## Force cleanup without confirmation
	./cleanup.sh --namespace $(NAMESPACE) --force

# Development targets
.PHONY: restart
restart: check-kubectl ## Restart all deployments
	kubectl rollout restart deployment -n $(NAMESPACE)
	kubectl rollout restart statefulset -n $(NAMESPACE)

.PHONY: restart-service
restart-service: check-kubectl ## Restart specific service (usage: make restart-service SERVICE=mcp-memory)
	@if [ -z "$(SERVICE)" ]; then echo "Usage: make restart-service SERVICE=<service-name>"; exit 1; fi
	kubectl rollout restart deployment/$(SERVICE) -n $(NAMESPACE) || kubectl rollout restart statefulset/$(SERVICE) -n $(NAMESPACE)

.PHONY: scale
scale: check-kubectl ## Scale a deployment (usage: make scale SERVICE=mcp-memory REPLICAS=3)
	@if [ -z "$(SERVICE)" ] || [ -z "$(REPLICAS)" ]; then echo "Usage: make scale SERVICE=<service-name> REPLICAS=<number>"; exit 1; fi
	kubectl scale deployment $(SERVICE) -n $(NAMESPACE) --replicas=$(REPLICAS)

# Port forwarding targets
.PHONY: port-forward
port-forward: check-kubectl ## Port forward all services
	@echo "Opening port forwards in background..."
	@kubectl port-forward -n $(NAMESPACE) service/qdrant 6333:6333 &
	@kubectl port-forward -n $(NAMESPACE) service/mcp-memory 8000:8000 &
	@kubectl port-forward -n $(NAMESPACE) service/mcp-indexer 8001:8001 &
	@echo "Port forwards started. Use 'make stop-port-forward' to stop."

.PHONY: port-forward-service
port-forward-service: check-kubectl ## Port forward specific service (usage: make port-forward-service SERVICE=qdrant LOCAL=6333 REMOTE=6333)
	@if [ -z "$(SERVICE)" ] || [ -z "$(LOCAL)" ] || [ -z "$(REMOTE)" ]; then echo "Usage: make port-forward-service SERVICE=<service-name> LOCAL=<local-port> REMOTE=<remote-port>"; exit 1; fi
	kubectl port-forward -n $(NAMESPACE) service/$(SERVICE) $(LOCAL):$(REMOTE)

.PHONY: stop-port-forward
stop-port-forward: ## Stop all port forwards
	pkill -f "kubectl port-forward" || echo "No port forwards found"

.PHONY: build-image
build-image: ## Build Docker image (requires Docker)
	docker build -t $(IMAGE_REGISTRY)/context-engine:$(IMAGE_TAG) ../../

.PHONY: push-image
push-image: build-image ## Push Docker image to registry
	docker push $(IMAGE_REGISTRY)/context-engine:$(IMAGE_TAG)

# Test targets
.PHONY: test-connection
test-connection: check-kubectl ## Test connectivity to all services
	@echo "Testing service connectivity..."
	@echo "Qdrant:"
	@kubectl run qdrant-test --image=curlimages/curl --rm -i --restart=Never -n $(NAMESPACE) -- curl -f http://qdrant.$(NAMESPACE).svc.cluster.local:6333/health || echo "Qdrant test failed"
	@echo "MCP Memory:"
	@kubectl run memory-test --image=curlimages/curl --rm -i --restart=Never -n $(NAMESPACE) -- curl -f http://mcp-memory.$(NAMESPACE).svc.cluster.local:8000/health || echo "MCP Memory test failed"
	@echo "MCP Indexer:"
	@kubectl run indexer-test --image=curlimages/curl --rm -i --restart=Never -n $(NAMESPACE) -- curl -f http://mcp-indexer.$(NAMESPACE).svc.cluster.local:8001/health || echo "MCP Indexer test failed"

# Configuration targets
.PHONY: show-config
show-config: ## Show current configuration
	@echo "Configuration:"
	@echo "  NAMESPACE: $(NAMESPACE)"
	@echo "  IMAGE_REGISTRY: $(IMAGE_REGISTRY)"
	@echo "  IMAGE_TAG: $(IMAGE_TAG)"
	@echo ""
	@echo "Quick start commands:"
	@echo "  make deploy              # Deploy all services"
	@echo "  make status              # Show deployment status"
	@echo "  make logs                # Show logs"
	@echo "  make cleanup             # Remove everything"

.PHONY: show-urls
show-urls: check-kubectl ## Show access URLs for services
	@echo "Service URLs (via NodePort):"
	@echo "  Qdrant:           http://<node-ip>:30333"
	@echo "  MCP Memory (SSE): http://<node-ip>:30800"
	@echo "  MCP Indexer (SSE): http://<node-ip>:30802"
	@echo "  MCP Memory (HTTP): http://<node-ip>:30804"
	@echo "  MCP Indexer (HTTP): http://<node-ip>:30806"
	@echo "  Llama.cpp:        http://<node-ip>:30808"
	@echo ""
	@echo "Service URLs (via port-forward):"
	@echo "  make port-forward # Then access via localhost ports"

# Advanced targets
.PHONY: watch-deployment
watch-deployment: check-kubectl ## Watch deployment progress
	watch kubectl get pods,services,deployments -n $(NAMESPACE)

.PHONY: describe-service
describe-service: check-kubectl ## Describe a service (usage: make describe-service SERVICE=mcp-memory)
	@if [ -z "$(SERVICE)" ]; then echo "Usage: make describe-service SERVICE=<service-name>"; echo "Available services:"; kubectl get services -n $(NAMESPACE); exit 1; fi
	kubectl describe service $(SERVICE) -n $(NAMESPACE)

.PHONY: events
events: check-kubectl ## Show recent events
	kubectl get events -n $(NAMESPACE) --sort-by=.metadata.creationTimestamp
