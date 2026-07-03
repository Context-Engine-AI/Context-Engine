# Commit Indexing & Context Lineage: Goals and Status

## 1. Motivation

- **Historical context for agents**
  - Modern agents are good at reading current files but struggle when the answer is buried in months of commit history.
  - Goal: expose a compact, queryable view of *how and why* the code evolved, not just what it looks like now.
- **Complement, not replace, git log**
  - Humans and local tools can always use `git log` / `git show` / `git diff` directly.
  - Commit indexing and lineage should add value by:
    - Making history available to remote/agent clients that cannot run git.
    - Providing structured summaries and tags so agents can quickly find and explain relevant changes.

## 2. Current architecture (v1)

- **Commit harvesting (`scripts/ingest_history.py`)**
  - Walks git history with configurable filters (`--since`, `--until`, `--author`, `--path`, `--max-commits`).
  - For each commit:
    - Captures `commit_id`, `author_name`, `authored_date`, `message` (subject + body, redacted), and `files` touched.
    - Builds a short `document` and `information` string.
    - Embeds into Qdrant in the same collection as code, with metadata:
      - `language="git"`, `kind="git_message"`.
      - `symbol` / `symbol_path` = `commit_id`.
      - `files`, `repo`, `path=.git`, `ingested_at`, etc.

- **GLM-backed diff summarization (`generate_commit_summary`)**
  - Opt-in via `REFRAG_COMMIT_DESCRIBE=1` and decoder flags (`REFRAG_DECODER=1`, `REFRAG_RUNTIME=glm`).
  - For each commit, fetches a truncated `git show --stat --patch --unified=3 <sha>` and sends it to the decoder (GLM or llama.cpp).
  - Asks for compact JSON:
    - `goal`: short explanation of the commit’s intent / behavior change.
    - `symbols`: 1–6 key functions/flags/terms.
    - `tags`: 3–6 short keywords to aid retrieval.
  - On success, stores these as metadata on the commit point:
    - `lineage_goal`, `lineage_symbols`, `lineage_tags`.
  - On failure or when disabled, falls back gracefully and leaves these fields empty.

- **Indexer-facing Qdrant schema**
  - Commit points live in the same Qdrant collection as code spans (e.g. `Context-Engine-41e67959`).
  - This allows hybrid flows that combine code search and commit search within one collection.

## 3. MCP tools and usage

- **`search_commits_for` (indexer MCP)**
  - Purpose: search git commit history stored in Qdrant.
  - Filters:
    - Always restricts to `language="git"`, `kind="git_message"`.
    - Optional `path` filter: only keep commits whose `files` list contains the path substring.
    - Optional `query`: lexical match against a composite blob containing:
      - `message` + `information`.
      - `lineage_goal`, `lineage_symbols`, `lineage_tags`.
  - Output (per commit):
    - `commit_id`, `author_name`, `authored_date`, `message`, `files`.
    - `lineage_goal`, `lineage_symbols`, `lineage_tags`.
  - Dedupes by `commit_id` so each commit appears at most once per response.

- **`change_history_for_path(include_commits=true)` (indexer MCP)**
  - Base behavior:
    - Scans Qdrant points whose `metadata.path == <path>` (code index), summarizing:
      - `points_scanned`, `distinct_hashes`, `last_modified_min/max`, `ingested_min/max`, `churn_count_max`.
  - With `include_commits=true`:
    - Calls `search_commits_for(path=<path>)` and attaches a small list of recent commits:
      - Each entry includes commit metadata plus any `lineage_*` fields.
    - Dedupes by `commit_id` before attaching.
  - Intended usage:
    - Fast “what changed and how hot is this file?” view for agents.
    - Entry point for deeper lineage questions when combined with `repo_search` and git diffs.

## 4. Current experiments and evaluation

See:
- `cmds.md` for handy one-liner commands (curl, ingest, local GLM tests).
- `experiments.md` for a detailed “when/why did behavior X change?” recipe and worked examples.

Key experiments so far:

- **GLM summarization sanity-check**
  - Local script that:
    - Picks `HEAD` via `git rev-list --max-count=1 HEAD`.
    - Calls `commit_metadata` + `generate_commit_summary`.
  - Observed: with valid GLM API keys and flags, we get reasonable `goal/symbols/tags` for real commits.

- **Qdrant payload inspection**
  - Direct `curl` scroll over `Context-Engine-41e67959` for `language="git"`, `kind="git_message"`.
  - Verified commit points include:
    - Baseline metadata (message, files, etc.).
    - Newly-added `lineage_goal`, `lineage_symbols`, `lineage_tags` after reindexing.

- **MCP round-trip tests**
  - `search_commits_for(query="pseudo tag boost")` → surfaces the hybrid_search commit with clear lineage fields.
  - `search_commits_for(query="ctx script", path="vscode-extension/context-engine-uploader/scripts/ctx.py")` → surfaces the ctx cleanup commit and explains its intent.
  - `change_history_for_path(path="vscode-extension/context-engine-uploader/scripts/ctx.py", include_commits=true)` → returns a deduped list of relevant commits with lineage summaries.

These confirm the end-to-end path:
- Git → ingest_history → GLM → Qdrant → MCP → agent.

## 5. Target workflows (what we are aiming for)

Our north star is the "Context Lineage" behavior from the Augment blog:

- **Hero question:**
  - “When and why did behavior X change in file Y?”

- **Recommended agent flow:**
  1. **Localize X in current code**
     - Use `repo_search` to find the symbol / behavior in the current tree.
  2. **Shortlist commits about X**
     - Use `search_commits_for(path=<file>, query=<keywords about X>)` to get a compact list of relevant commits with `lineage_goal`/tags.
  3. **Try to answer "why" from summaries**
     - Many “why was this introduced/removed/renamed?” questions can be answered from `lineage_goal` plus minimal code context.
  4. **If necessary, pull diffs to answer "how"**
     - Use `git show --unified=3 <commit_id> -- <file>` (or a future MCP diff tool) and let the LLM explain the behavior change in detail.

This should:
- Reduce reliance on raw `git log` grepping in larger repos.
- Give agents a semantic, compact view of history they can reason over.

## 6. Open questions and future improvements

- **Prompt quality and consistency**
  - Are `lineage_goal` strings consistently helpful across many commits, or do they drift toward restating the subject line?
  - Do `lineage_symbols` and `lineage_tags` give agents enough hooks to connect history with current code (e.g., flags, functions, config keys)?

- **Search behavior and ranking**
  - How often does `search_commits_for` surface the right commit(s) in the top N for real questions?
  - Do we need semantic reranking or additional filters (date ranges, authors, etc.) in practice?

- **Higher-level `lineage_answer` helper**
  - Today: agents compose `repo_search` + `change_history_for_path(include_commits=true)` + `search_commits_for` + optional `context_answer` themselves.
  - Future: a thin MCP wrapper (e.g., `lineage_answer(query, path=...)`) could orchestrate those calls and ask the decoder to produce a short "when/why did this change" answer, returning both the text and the underlying commit/code citations.

- **Diff access for remote agents**
  - Today: local workflows can rely on `git show` from the shell.
  - Future: a small, token-conscious MCP tool like `get_commit_diff(commit_id, path, context_lines)` could make lineage usable from fully remote contexts when precise line-level inspection is required.

- **Remote git metadata via upload pipeline**
  - Current commit ingest assumes direct access to a local `.git` repo (`ingest_history.py` running alongside the indexer).
  - Future: the standalone upload client could optionally parse a compact git log view (e.g., JSON-ified commit metadata + diffs) and bundle it with delta uploads, with the upload_service and watcher feeding that into commit indexing for remote/non-git workspaces.

- **Docs and agent guidance**
  - CLAUDE.md (and related examples) should clearly document when to:
    - Prefer lineage summaries over raw diffs for "why" questions.
    - Fallback to `repo_search + git show` (or a future diff MCP tool) for detailed "how" questions.

This document is meant as the high-level tracker for commit indexing and context-lineage work. Use `cmds.md` for concrete commands and `experiments.md` for detailed workflows and notes on specific runs.
