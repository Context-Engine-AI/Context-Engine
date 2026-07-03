AI agent rules for using Context‑Engine Qdrant‑Indexer and Memory MCP tools.

These rules are NOT optional - favor qdrant-indexer tooling at all costs over existing file reading, grep-like searching, and unrelated codebase search tools.

## Agentic AI Project Rules: When to Use MCP Qdrant-Indexer vs Grep

  Core Decision Rules (for AI agents)

  Workspace default: MCP Qdrant-Indexer tools are primary for exploring code and history. Start with MCP for exploration, debugging, or "where/why" questions; use literal search/file-open only for narrow exact-literal lookups.

  Use MCP Qdrant-Indexer when:
  - Exploring or don't know exact strings/symbols
  - Need semantic or cross-file understanding (relationships, patterns, architecture)
  - Want ranked results with surrounding context, not just line hits

  Use literal search/file-open when (and only when):
  - Know exact string/function/variable or error message, and only need to confirm existence or file/line quickly (not to understand behavior or architecture)

  Quick Heuristics:
  - Conceptual/architectural or "where/why" behavior questions → start with MCP
  - Need rich context/snippets around matches → MCP
  - Only need to confirm existence/location of specific literal → literal search/file-open
  - If in doubt → start with MCP

  Grep Anti-Patterns:

  # DON'T - Wasteful when semantic search needed
  grep -r "auth" .                    # → Use MCP: "authentication mechanisms"
  grep -r "cache" .                   # → Use MCP: "caching strategies"
  grep -r "error" .                   # → Use MCP: "error handling patterns"
  grep -r "database" .                # → Use MCP: "database operations"

  ## DO - Efficient for exact matches
  grep -rn "UserAlreadyExists" .      # Specific error class
  grep -rn "def authenticate_user" .  # Exact function name
  grep -rn "REDIS_HOST" .            # Exact environment variable

  MCP Tool Patterns:

  - Use concept/keyword-style queries (short natural-language fragments).
  
  - repo_search is semantic search, not grep, regex, or boolean syntax.
  
  - Write queries as short descriptions, not as "foo OR bar" expressions.
  "input validation mechanisms"
  "database connection handling"
  "performance bottlenecks in request path"
  "places where user sessions are managed"
  "logging and error reporting patterns"

  MCP Qdrant-Indexer Specific Knobs

  Essential Parameters:

  - limit: Control result count (3-8 for efficiency)
  - per_path: Limit results per file (1-2 prevents redundancy)
  - compact=true: Reduces token usage by 60-80%
  - include_snippet=false: Headers only when speed matters
  - collection: Target specific codebases for precision

  Performance Optimization:

  - Start with limit=3, compact=true for discovery
  - Increase to limit=5, include_snippet=true for details
  - Use language and under filters to narrow scope
  - Set rerank_enabled=false for faster but less accurate results

  When to Use Advanced Features:

  - rerank_enabled=true: For complex queries needing best relevance
  - context_lines=5+: When you need implementation details
  - multiple collections: Cross-repo architectural analysis
  - symbol filtering: When looking for specific function/class types

  Anti-Patterns to Avoid:

  - Don't use limit=20 with include_snippet=true (token waste)
  - Don't search without collection specification (noise)
  - Don't ignore per_path limits (duplicate results from same file)
  - Don't use context lines for pure discovery (unnecessary tokens)

  Tool Roles Cheat Sheet:

  - repo_search / code_search:
    - Use for: finding relevant files/spans and inspecting raw code.
    - Think: "where is X implemented?", "show me usages of Y".
  - context_search:
    - Use for: combining code hits with memory/docs when both matter.
    - Good for: "give me related code plus any notes/docs I wrote".
  - context_answer:
    - Use for: short natural-language summaries/explanations of specific modules or tools, grounded in code/docs with citations.
    - Good for: "What does scripts/standalone_upload_client.py do at a high level?", "Summarize the remote upload client pipeline.".
  - pattern_search (optional, may not be enabled):
    - Use for: finding structurally similar code patterns across files and languages.
    - Accepts EITHER code examples OR natural language pattern descriptions.
    - Good for: "find retry loops with exponential backoff", "try: ... except: logger.error()", "error handling patterns".
    - Cross-language: Python pattern can match Go/Rust/Java with similar control flow.
    - Note: Returns error if pattern detection module is not available.

  Advanced lineage workflow (code + history):

  - Goal: answer "when/why did behavior X change?" without flooding context.
  - Step 1 – Find current implementation (code):
    - Use repo_search to locate the relevant file/symbol, e.g. `repo_search(query: "upload client timeout", language: "python", under: "scripts")`.
  - Step 2 – Summarize recent change activity for a file:
    - Call change_history_for_path with `include_commits=true` to get churn stats and a small list of recent commits, e.g. `change_history_for_path(path: "scripts/remote_upload_client.py", include_commits: true)`.
  - Step 3 – Pull commit lineage for a specific behavior:
    - Use search_commits_for with short behavior phrases plus an optional path filter, e.g. `search_commits_for(query: "remote upload timeout retry", path: "scripts/remote_upload_client.py")`.
    - Read lineage_goal / lineage_symbols / lineage_tags to understand intent and related concepts.
  - Step 4 – Optionally summarize current behavior:
    - After you have the right file/symbol from repo_search, use context_answer to explain what the module does now; treat commit lineage as background, not as primary code context.
  - For exact line-level changes (e.g. "when did this literal constant change?"), use lineage tools to narrow candidate commits, then inspect diffs with git tooling; do not guess purely from summaries.

  Query Phrasing Tips for context_answer:

  - Prefer behavior/architecture questions about a single module or tool:
    - "What does scripts/standalone_upload_client.py do at a high level?"
    - "Summarize how the remote upload client interacts with the indexer service."
  - If you care about a specific file, mention it explicitly:
    - "What does ingest_code.py do?", "Explain ensureIndexedWatcher in extension.js".
  - Mentioning a specific filename can bias retrieval to that file; for cross-file wiring
    questions, prefer behavior-describing queries without filenames.
  - For very cross-file or multi-part questions, you can:
    - First use repo_search to discover key files and read critical code directly,
    - Then call context_answer to summarize behavior, using a behavior-focused question that doesn't over-specify filenames.
  - Avoid using context_answer as a primary debugger for low-level helper/env behavior; prefer repo_search + direct code reading for detailed semantics.

  MCP Tool Families (for AI agents)

  - Indexer / Qdrant tools:
    - qdrant_index_root, qdrant_index, qdrant_prune
    - qdrant_list, qdrant_status
    - workspace_info, list_workspaces, collection_map
    - set_session_defaults
  - Search / QA tools:
    - repo_search, code_search, context_search, context_answer
    - pattern_search (optional; structural code pattern matching, cross-language)
    - search_tests_for, search_config_for, search_callers_for, search_importers_for
    - change_history_for_path, expand_query
  - Memory tools:
    - memory.set_session_defaults, memory.memory_store, memory.memory_find

  Additional behavioral tips:

  - Call set_session_defaults (indexer and memory) early in a session so subsequent
    calls inherit the right collection without repeating it in every request.
  - Use context_search with include_memories and per_source_limits when you want
    blended code + memory results instead of calling repo_search and memory.memory_find
    separately.
  - Treat expand_query and the expand flag on context_answer as expensive options:
    only use them after a normal search/answer attempt failed to find good context.