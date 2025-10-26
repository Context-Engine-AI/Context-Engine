# Delta Upload Format and Protocol Design

## Overview

This document specifies a delta upload format and protocol for real-time code ingestion in Context-Engine, designed to efficiently transmit only changed files from a local watch client to a remote upload service.

## 1. Delta Bundle Format Specification

### 1.1 Bundle Structure

A delta bundle is a tarball (`.tar.gz`) containing:

```
delta-bundle.tar.gz
├── manifest.json          # Bundle metadata and file operations
├── files/                 # Directory containing file content
│   ├── created/           # New files
│   ├── updated/           # Modified files
│   └── moved/            # Moved files (at destination)
└── metadata/             # File metadata and hashes
    ├── hashes.json        # Content hashes for all files
    └── operations.json   # Detailed operation metadata
```

### 1.2 Manifest Format (`manifest.json`)

```json
{
  "version": "1.0",
  "bundle_id": "uuid-v4",
  "workspace_path": "/absolute/path/to/workspace",
  "collection_name": "workspace-collection",
  "created_at": "2025-01-26T01:55:00.000Z",
  "sequence_number": 42,
  "parent_sequence": 41,
  "operations": {
    "created": 5,
    "updated": 3,
    "deleted": 2,
    "moved": 1
  },
  "total_files": 11,
  "total_size_bytes": 1048576,
  "compression": "gzip",
  "encoding": "utf-8"
}
```

### 1.3 File Operations Format

#### Created Files (`files/created/`)
- Path: `files/created/relative/path/to/file.ext`
- Content: Full file content
- Metadata: Stored in `metadata/operations.json`

#### Updated Files (`files/updated/`)
- Path: `files/updated/relative/path/to/file.ext`
- Content: Full file content (simpler than diff-based approach)
- Metadata: Stored in `metadata/operations.json`

#### Moved Files (`files/moved/`)
- Path: `files/moved/destination/path/to/file.ext`
- Content: Full file content at destination
- Metadata: Includes source path in `metadata/operations.json`

#### Deleted Files
- No content in bundle
- Metadata only in `metadata/operations.json`

### 1.4 Operations Metadata (`metadata/operations.json`)

```json
{
  "operations": [
    {
      "operation": "created",
      "path": "src/new_file.py",
      "relative_path": "src/new_file.py",
      "absolute_path": "/workspace/src/new_file.py",
      "size_bytes": 1024,
      "content_hash": "sha1:da39a3ee5e6b4b0d3255bfef95601890afd80709",
      "file_hash": "sha1:abc123...",
      "modified_time": "2025-01-26T01:55:00.000Z",
      "language": "python"
    },
    {
      "operation": "updated",
      "path": "src/existing.py",
      "relative_path": "src/existing.py",
      "absolute_path": "/workspace/src/existing.py",
      "size_bytes": 2048,
      "content_hash": "sha1:new_hash_value",
      "previous_hash": "sha1:old_hash_value",
      "file_hash": "sha1:def456...",
      "modified_time": "2025-01-26T01:55:00.000Z",
      "language": "python"
    },
    {
      "operation": "moved",
      "path": "src/new_location.py",
      "relative_path": "src/new_location.py",
      "absolute_path": "/workspace/src/new_location.py",
      "source_path": "src/old_location.py",
      "source_relative_path": "src/old_location.py",
      "source_absolute_path": "/workspace/src/old_location.py",
      "size_bytes": 1536,
      "content_hash": "sha1:same_hash_as_source",
      "file_hash": "sha1:ghi789...",
      "modified_time": "2025-01-26T01:55:00.000Z",
      "language": "python"
    },
    {
      "operation": "deleted",
      "path": "src/removed.py",
      "relative_path": "src/removed.py",
      "absolute_path": "/workspace/src/removed.py",
      "previous_hash": "sha1:deleted_file_hash",
      "file_hash": null,
      "modified_time": "2025-01-26T01:55:00.000Z",
      "language": "python"
    }
  ]
}
```

### 1.5 Hash Storage (`metadata/hashes.json`)

```json
{
  "workspace_path": "/workspace",
  "updated_at": "2025-01-26T01:55:00.000Z",
  "file_hashes": {
    "src/new_file.py": "sha1:abc123...",
    "src/existing.py": "sha1:def456...",
    "src/new_location.py": "sha1:ghi789..."
  }
}
```

## 2. HTTP API Contract

### 2.1 Upload Endpoint

```
POST /api/v1/delta/upload
Content-Type: multipart/form-data
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| bundle | File | Yes | Delta bundle tarball |
| workspace_path | String | Yes | Absolute workspace path |
| collection_name | String | No | Override collection name |
| sequence_number | Integer | No | Expected sequence number |
| force | Boolean | No | Force upload even if sequence mismatch |

#### Response Format

```json
{
  "success": true,
  "bundle_id": "uuid-v4",
  "sequence_number": 42,
  "processed_operations": {
    "created": 5,
    "updated": 3,
    "deleted": 2,
    "moved": 1,
    "skipped": 0,
    "failed": 0
  },
  "processing_time_ms": 1250,
  "indexed_points": 156,
  "collection_name": "workspace-collection",
  "next_sequence": 43
}
```

#### Error Response

```json
{
  "success": false,
  "error": {
    "code": "SEQUENCE_MISMATCH",
    "message": "Expected sequence 41, got 43",
    "expected_sequence": 41,
    "received_sequence": 43,
    "retry_after": 5000
  }
}
```

### 2.2 Status Endpoint

```
GET /api/v1/delta/status?workspace_path=/workspace
```

#### Response

```json
{
  "workspace_path": "/workspace",
  "collection_name": "workspace-collection",
  "last_sequence": 41,
  "last_upload": "2025-01-26T01:50:00.000Z",
  "pending_operations": 0,
  "status": "ready",
  "server_info": {
    "version": "1.0.0",
    "max_bundle_size_mb": 100,
    "supported_formats": ["tar.gz"]
  }
}
```

### 2.3 Recovery Endpoint

```
POST /api/v1/delta/recover
Content-Type: application/json
```

#### Request

```json
{
  "workspace_path": "/workspace",
  "from_sequence": 38,
  "to_sequence": 42
}
```

#### Response

```json
{
  "success": true,
  "recovered_bundles": [
    {
      "sequence": 39,
      "bundle_id": "uuid-39",
      "operations": {"created": 2, "updated": 1}
    }
  ],
  "next_sequence": 43
}
```

## 3. Change Detection Algorithm

### 3.1 Integration with Existing Hash Cache

The delta system leverages the existing hash-based caching in [`workspace_state.py`](scripts/workspace_state.py:304-310):

```python
def detect_file_changes(workspace_path: str, changed_paths: List[Path]) -> Dict[str, Any]:
    """
    Detect what type of changes occurred for each file path.
    
    Returns:
    {
        "created": [Path],
        "updated": [Path], 
        "deleted": [Path],
        "moved": [(source: Path, dest: Path)],
        "unchanged": [Path]
    }
    """
    changes = {
        "created": [],
        "updated": [],
        "deleted": [],
        "moved": [],
        "unchanged": []
    }
    
    for path in changed_paths:
        abs_path = str(path.resolve())
        cached_hash = get_cached_file_hash(workspace_path, abs_path)
        
        if not path.exists():
            # File was deleted
            if cached_hash:
                changes["deleted"].append(path)
        else:
            # File exists - calculate current hash
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                current_hash = hashlib.sha1(content).hexdigest()
                
                if not cached_hash:
                    # New file
                    changes["created"].append(path)
                elif cached_hash != current_hash:
                    # Modified file
                    changes["updated"].append(path)
                else:
                    # Unchanged (might be a move detection candidate)
                    changes["unchanged"].append(path)
                    
                # Update cache
                set_cached_file_hash(workspace_path, abs_path, current_hash)
            except Exception:
                # Skip files that can't be read
                continue
    
    # Detect moves by looking for files with same content hash
    # but different paths (requires additional tracking)
    changes["moved"] = detect_moves(changes["created"], changes["deleted"])
    
    return changes
```

### 3.2 Move Detection Algorithm

```python
def detect_moves(created_files: List[Path], deleted_files: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Detect file moves by matching content hashes between created and deleted files.
    """
    moves = []
    deleted_hashes = {}
    
    # Build hash map for deleted files
    for deleted_path in deleted_files:
        try:
            with open(deleted_path, 'rb') as f:
                content = f.read()
            file_hash = hashlib.sha1(content).hexdigest()
            deleted_hashes[file_hash] = deleted_path
        except Exception:
            continue
    
    # Match created files with deleted files by hash
    for created_path in created_files:
        try:
            with open(created_path, 'rb') as f:
                content = f.read()
            file_hash = hashlib.sha1(content).hexdigest()
            
            if file_hash in deleted_hashes:
                source_path = deleted_hashes[file_hash]
                moves.append((source_path, created_path))
                # Remove from consideration
                del deleted_hashes[file_hash]
        except Exception:
            continue
    
    return moves
```

### 3.3 Integration with ChangeQueue

The delta system integrates with the existing [`ChangeQueue`](scripts/watch_index.py:45-66) debouncing pattern:

```python
class DeltaChangeQueue(ChangeQueue):
    """Extended ChangeQueue that creates delta bundles."""
    
    def __init__(self, process_cb, workspace_path: str, upload_endpoint: str):
        super().__init__(process_cb)
        self.workspace_path = workspace_path
        self.upload_endpoint = upload_endpoint
        self.sequence_number = self._get_last_sequence()
    
    def _flush(self):
        """Override to create delta bundle before processing."""
        with self._lock:
            paths = list(self._paths)
            self._paths.clear()
            self._timer = None
        
        # Detect changes and create delta bundle
        changes = detect_file_changes(self.workspace_path, paths)
        if self._has_meaningful_changes(changes):
            bundle = self._create_delta_bundle(changes)
            self._upload_bundle(bundle)
        
        # Call original processing
        self._process_cb(paths)
    
    def _has_meaningful_changes(self, changes: Dict[str, List]) -> bool:
        """Check if changes warrant a delta upload."""
        total_changes = sum(len(files) for op, files in changes.items() if op != "unchanged")
        return total_changes > 0
```

## 4. Error Handling and Recovery Strategy

### 4.1 Retry Mechanism

```python
class DeltaUploadClient:
    def __init__(self, endpoint: str, max_retries: int = 3):
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.retry_delays = [1000, 2000, 5000]  # ms
    
    def upload_bundle(self, bundle_path: str, metadata: Dict) -> bool:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._send_bundle(bundle_path, metadata)
                if response["success"]:
                    return True
                
                # Handle specific error cases
                if response["error"]["code"] == "SEQUENCE_MISMATCH":
                    return self._handle_sequence_mismatch(response, metadata)
                
            except Exception as e:
                if attempt == self.max_retries:
                    self._log_failure(e, metadata)
                    return False
                
                # Wait before retry
                if attempt < len(self.retry_delays):
                    time.sleep(self.retry_delays[attempt] / 1000)
        
        return False
```

### 4.2 Sequence Number Recovery

```python
def _handle_sequence_mismatch(self, error_response: Dict, metadata: Dict) -> bool:
    """Handle sequence number mismatch by recovering missing bundles."""
    expected_seq = error_response["error"]["expected_sequence"]
    current_seq = metadata["sequence_number"]
    
    # Try to recover missing bundles
    recovery_response = self._request_recovery(
        metadata["workspace_path"],
        from_sequence=expected_seq,
        to_sequence=current_seq - 1
    )
    
    if recovery_response["success"]:
        # Apply recovered bundles locally
        for bundle_info in recovery_response["recovered_bundles"]:
            if not self._apply_recovered_bundle(bundle_info):
                return False
        
        # Retry original upload
        return self._send_bundle(metadata["bundle_path"], metadata)["success"]
    
    return False
```

### 4.3 Bundle Persistence

```python
class BundlePersistence:
    """Local persistence for delta bundles to enable recovery."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.bundle_dir = Path(workspace_path) / ".codebase" / "delta_bundles"
        self.bundle_dir.mkdir(exist_ok=True)
    
    def save_bundle(self, bundle_path: str, metadata: Dict) -> str:
        """Save bundle locally with metadata."""
        bundle_id = metadata["bundle_id"]
        saved_path = self.bundle_dir / f"{bundle_id}.tar.gz"
        metadata_path = self.bundle_dir / f"{bundle_id}.json"
        
        shutil.copy2(bundle_path, saved_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(saved_path)
    
    def get_pending_bundles(self) -> List[Dict]:
        """Get bundles that haven't been acknowledged by server."""
        pending = []
        for metadata_file in self.bundle_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                if not metadata.get("acknowledged", False):
                    pending.append(metadata)
            except Exception:
                continue
        return pending
```

## 5. Integration Points with Existing Code

### 5.1 Integration with watch_index.py

```python
# Modified IndexHandler to support delta uploads
class DeltaIndexHandler(IndexHandler):
    def __init__(self, root: Path, queue: ChangeQueue, client: QdrantClient, 
                 collection: str, delta_client: DeltaUploadClient):
        super().__init__(root, queue, client, collection)
        self.delta_client = delta_client
    
    def _maybe_enqueue(self, src_path: str):
        """Override to add delta queue processing."""
        super()._maybe_enqueue(src_path)
        # Delta processing happens in the extended ChangeQueue
    
    def on_deleted(self, event):
        """Override to handle deletions in delta system."""
        super().on_deleted(event)
        # Delta queue will handle the deletion processing
```

### 5.2 Integration with ingest_code.py

```python
# Extend ingest_code.py to process delta bundles
def process_delta_bundle(bundle_path: str, workspace_path: str, 
                       collection: str) -> Dict[str, Any]:
    """Process a delta bundle and update Qdrant collection."""
    
    # Extract bundle
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_path = Path(temp_dir)
        with tarfile.open(bundle_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        
        # Read manifest
        with open(extract_path / "manifest.json") as f:
            manifest = json.load(f)
        
        # Read operations
        with open(extract_path / "metadata" / "operations.json") as f:
            operations = json.load(f)["operations"]
        
        # Process each operation
        results = {"created": 0, "updated": 0, "deleted": 0, "moved": 0, "failed": 0}
        
        for op in operations:
            try:
                if op["operation"] == "created":
                    _process_created_file(extract_path, op, collection, results)
                elif op["operation"] == "updated":
                    _process_updated_file(extract_path, op, collection, results)
                elif op["operation"] == "deleted":
                    _process_deleted_file(op, collection, results)
                elif op["operation"] == "moved":
                    _process_moved_file(extract_path, op, collection, results)
            except Exception as e:
                results["failed"] += 1
                print(f"Failed to process {op['operation']} {op['path']}: {e}")
        
        # Update workspace state
        _update_workspace_state_from_delta(workspace_path, manifest, results)
        
        return results
```

### 5.3 Integration with workspace_state.py

```python
# Extend workspace_state.py for delta tracking
def get_last_delta_sequence(workspace_path: str) -> int:
    """Get the last processed delta sequence number."""
    state = get_workspace_state(workspace_path)
    return state.get("delta_state", {}).get("last_sequence", 0)

def update_delta_state(workspace_path: str, sequence: int, bundle_id: str) -> None:
    """Update delta processing state."""
    state = get_workspace_state(workspace_path)
    delta_state = state.get("delta_state", {})
    delta_state.update({
        "last_sequence": sequence,
        "last_bundle_id": bundle_id,
        "last_processed": datetime.now().isoformat()
    })
    update_workspace_state(workspace_path, {"delta_state": delta_state})
```

## 6. Implementation Roadmap

### Phase 1: Core Delta Format and API (Week 1)
1. Implement delta bundle creation and parsing
2. Create HTTP API endpoints for upload, status, and recovery
3. Implement basic error handling and response formats
4. Add unit tests for bundle format validation

### Phase 2: Change Detection Integration (Week 2)
1. Integrate with existing hash cache system
2. Implement move detection algorithm
3. Extend ChangeQueue for delta bundle creation
4. Add integration tests with watch_index.py

### Phase 3: Error Handling and Recovery (Week 3)
1. Implement retry mechanism with exponential backoff
2. Add sequence number recovery
3. Implement bundle persistence
4. Add comprehensive error logging and monitoring

### Phase 4: Production Integration (Week 4)
1. Integrate with ingest_code.py for bundle processing
2. Extend workspace_state.py for delta tracking
3. Add performance optimization and batching
4. Implement monitoring and alerting
5. Add end-to-end integration tests

### Phase 5: Performance and Scaling (Week 5)
1. Optimize bundle compression and size
2. Implement parallel processing for large bundles
3. Add bandwidth optimization for remote uploads
4. Performance testing and tuning

## 7. Configuration and Environment Variables

```bash
# Delta upload configuration
DELTA_UPLOAD_ENABLED=true
DELTA_UPLOAD_ENDPOINT=http://delta-server:8002/api/v1/delta
DELTA_MAX_BUNDLE_SIZE_MB=100
DELTA_BATCH_SIZE_FILES=50
DELTA_DEBOUNCE_SECS=2.0

# Retry and recovery
DELTA_MAX_RETRIES=3
DELTA_RETRY_DELAYS_MS=1000,2000,5000
DELTA_PERSIST_BUNDLES=true
DELTA_BUNDLE_RETENTION_DAYS=7

# Performance tuning
DELTA_COMPRESSION_LEVEL=6
DELTA_PARALLEL_UPLOADS=2
DELTA_CHUNK_SIZE_BYTES=1048576
```

## 8. Security Considerations

1. **Authentication**: Add API key or token-based authentication
2. **Authorization**: Validate workspace access permissions
3. **Input Validation**: Validate bundle format and file paths
4. **Rate Limiting**: Implement upload rate limits per workspace
5. **Audit Logging**: Log all delta operations for compliance

## 9. Monitoring and Observability

1. **Metrics**: Track bundle size, processing time, success rates
2. **Logging**: Structured logging for all delta operations
3. **Health Checks**: Endpoint health monitoring
4. **Alerting**: Alert on failed uploads or processing errors
5. **Dashboards**: Visual monitoring of delta system performance

This design provides a comprehensive foundation for implementing delta uploads in Context-Engine while leveraging existing infrastructure and maintaining compatibility with current file processing workflows.