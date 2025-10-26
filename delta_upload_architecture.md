# Delta Upload System Architecture

## System Overview

```mermaid
graph TB
    subgraph "Local Environment"
        FS[File System]
        W[watch_index.py]
        CQ[ChangeQueue]
        DC[Delta Creator]
        LC[Local Cache]
    end
    
    subgraph "Delta Upload Service"
        API[HTTP API]
        BP[Bundle Processor]
        Q[Qdrant Client]
        WS[Workspace State]
    end
    
    subgraph "Storage"
        S3[Bundle Storage]
        QDR[Qdrant DB]
    end
    
    FS --> W
    W --> CQ
    CQ --> DC
    DC --> LC
    DC --> API
    API --> BP
    BP --> Q
    BP --> WS
    Q --> QDR
    BP --> S3
```

## Delta Bundle Creation Flow

```mermaid
sequenceDiagram
    participant FS as File System
    participant W as watch_index.py
    participant CQ as ChangeQueue
    participant DC as Delta Creator
    participant LC as Local Cache
    participant API as Upload API
    
    FS->>W: File change event
    W->>CQ: Add path to queue
    CQ->>CQ: Debounce changes
    CQ->>DC: Flush batched changes
    DC->>LC: Check cached hashes
    LC-->>DC: Return cached hashes
    DC->>FS: Read file contents
    DC->>DC: Detect change types
    DC->>DC: Create delta bundle
    DC->>LC: Save bundle locally
    DC->>API: Upload bundle
    API-->>DC: Acknowledge receipt
    DC->>LC: Mark as acknowledged
```

## Change Detection Algorithm

```mermaid
flowchart TD
    A[Start: File Changes Detected] --> B[Get Cached Hashes]
    B --> C{File Exists?}
    C -->|No| D[File Deleted]
    C -->|Yes| E[Calculate Current Hash]
    E --> F{Hash Changed?}
    F -->|No| G[Unchanged]
    F -->|Yes| H{Has Cached Hash?}
    H -->|No| I[File Created]
    H -->|Yes| J[File Updated]
    D --> K[Add to Deleted List]
    I --> L[Add to Created List]
    J --> M[Add to Updated List]
    G --> N[Skip]
    K --> O[Detect Moves]
    L --> O
    M --> O
    N --> O
    O --> P[Create Delta Bundle]
```

## Error Recovery Flow

```mermaid
stateDiagram-v2
    [*] --> UploadBundle
    UploadBundle --> Success: Upload OK
    UploadBundle --> RetryableError: Network/Temp Error
    UploadBundle --> SequenceError: Sequence Mismatch
    UploadBundle --> FatalError: Permanent Failure
    
    RetryableError --> WaitRetry
    WaitRetry --> UploadBundle: Retry
    
    SequenceError --> RequestRecovery
    RequestRecovery --> ApplyRecovered: Recovery OK
    RequestRecovery --> FatalError: Recovery Failed
    ApplyRecovered --> UploadBundle
    
    Success --> [*]
    FatalError --> [*]
```

## Integration Points

```mermaid
graph LR
    subgraph "Existing Components"
        WI[watch_index.py]
        IC[ingest_code.py]
        WS[workspace_state.py]
        Q[Qdrant Client]
    end
    
    subgraph "New Delta Components"
        DQ[DeltaChangeQueue]
        DC[DeltaCreator]
        DS[DeltaService]
        DP[DeltaProcessor]
    end
    
    WI --> DQ
    DQ --> DC
    DC --> DS
    DS --> DP
    DP --> IC
    DP --> WS
    IC --> Q
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Client Side"
        A[File Changes] --> B[Change Detection]
        B --> C[Delta Bundle Creation]
        C --> D[Local Persistence]
        D --> E[HTTP Upload]
    end
    
    subgraph "Server Side"
        E --> F[Bundle Reception]
        F --> G[Validation]
        G --> H[Processing Queue]
        H --> I[File Operations]
        I --> J[Qdrant Updates]
        I --> K[State Updates]
    end
    
    subgraph "Recovery Flow"
        L[Sequence Mismatch] --> M[Recovery Request]
        M --> N[Missing Bundles]
        N --> O[Replay Operations]
    end
    
    F -.-> L
```

## Component Interactions

```mermaid
classDiagram
    class ChangeQueue {
        +add(Path)
        +_flush()
        -_lock: threading.Lock
        -_paths: Set[Path]
        -_timer: threading.Timer
    }
    
    class DeltaChangeQueue {
        +add(Path)
        +_flush()
        -detect_changes()
        -create_bundle()
        -upload_bundle()
    }
    
    class DeltaCreator {
        +create_bundle(changes)
        +detect_file_changes()
        +detect_moves()
        -calculate_hashes()
    }
    
    class DeltaService {
        +upload_bundle()
        +get_status()
        +recover_bundles()
        -validate_bundle()
        -process_operations()
    }
    
    class DeltaProcessor {
        +process_bundle()
        +process_created()
        +process_updated()
        +process_deleted()
        +process_moved()
    }
    
    ChangeQueue <|-- DeltaChangeQueue
    DeltaChangeQueue --> DeltaCreator
    DeltaCreator --> DeltaService
    DeltaService --> DeltaProcessor
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_FS[Local File System]
        DEV_WATCH[watch_index.py]
        DEV_DELTA[Delta Client]
        DEV_API[Local Delta API]
    end
    
    subgraph "Production Environment"
        PROD_FS[Shared File System]
        PROD_WATCH[watch_index.py Cluster]
        PROD_DELTA[Delta Client Cluster]
        LB[Load Balancer]
        PROD_API[Delta API Cluster]
        REDIS[Redis Cache]
        S3[Object Storage]
    end
    
    DEV_FS --> DEV_WATCH
    DEV_WATCH --> DEV_DELTA
    DEV_DELTA --> DEV_API
    
    PROD_FS --> PROD_WATCH
    PROD_WATCH --> PROD_DELTA
    PROD_DELTA --> LB
    LB --> PROD_API
    PROD_API --> REDIS
    PROD_API --> S3
```

This architecture provides a comprehensive view of how the delta upload system integrates with the existing Context-Engine infrastructure while providing scalability, reliability, and efficient change detection.