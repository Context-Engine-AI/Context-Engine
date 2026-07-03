---

## 1. High-level agent recipe

Target question:  
**“When and why did behavior X change in file Y?”**

Recommended steps:

1. **Localize the behavior now (code search).**  
   Use [repo_search](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:1663:0-2468:5) to find the current implementation of X.

2. **Shortlist relevant commits (lineage search).**  
   Use [search_commits_for(path=<file>, query=<behavior keywords>)](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:2760:0-2898:65) to get a tiny set of candidate commits with `lineage_goal` / tags.

3. **Decide if lineage summary is enough.**  
   Sometimes `lineage_goal` already answers the “why” without diffs.

4. **If needed, pull small diffs and reason.**  
   For 1–2 chosen SHAs, fetch compact diffs for that file and let the LLM explain _how_ the behavior changed.

Everything else is just detail and guardrails around these four steps.

---

## 2. Step-by-step, with knobs

### Step 1: Localize behavior (repo_search)

- **Inputs:**
  - Behavior symbol: [ensureIndexedWatcher](cci:1://file:///home/coder/project/Context-Engine/vscode-extension/context-engine-uploader/extension.js:751:0-781:1), `index_repo`, etc.
  - Optional context: `"status bar"`, `"upload delta"`, etc.

- **Call pattern:**

  ```jsonc
  repo_search(
    query: "ensureIndexedWatcher status bar",
    under: "vscode-extension/context-engine-uploader",
    limit: 8,
    per_path: 2
  )
  ```

- **Goal:**
  - Identify:
    - Canonical file path, e.g. `"vscode-extension/context-engine-uploader/extension.js"`.
    - Specific function or symbol (span) you care about.

Agents should **store**:

- `target_file`: relative path under repo.
- `target_symbol` or a short description of the behavior X.

### Step 2: Shortlist commits (search_commits_for)

- **Normalize path:**
  - [search_commits_for](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:2760:0-2898:65) uses commit metadata `files` like:
    - `"scripts/ingest_code.py"`
    - `"vscode-extension/context-engine-uploader/scripts/ctx.py"`
  - So pass exactly that style: **no `/work/...` prefix**.

- **Call pattern:**

  ```jsonc
  search_commits_for(
    query: "ensureIndexedWatcher status bar",  // or a simpler keyword
    path: "vscode-extension/context-engine-uploader/extension.js",
    collection: "Context-Engine-41e67959",
    limit: 5,
    max_points: 1000
  )
  ```

- **What you get back:**

  ```json
  {
    "commit_id": "...",
    "message": "...",
    "files": ["..."],
    "lineage_goal": "short intent summary",
    "lineage_symbols": [...],
    "lineage_tags": [...]
  }
  ```

Agents should:

- Prefer commits where:
  - `files` includes the target file, and
  - `lineage_goal` / `lineage_symbols` mention relevant concepts.

This step is the **semantic pre-filter**: instead of scrolling `git log -- path`, you pick 1–3 promising SHAs.

### Step 3: Answer “why” from lineage, if possible

For many questions, you don’t need diffs at all:

- Example from your run:

  ```json
  {
    "commit_id": "e9d5...",
    "message": "Remove duplicate ctx script in extension ...",
    "lineage_goal": "Remove duplicate ctx script from extension as it's bundled at build time",
    "lineage_tags": ["cleanup","duplicate","build","extension","script"]
  }
  ```

For a question like:

> “Why did ctx.py disappear from the extension folder?”

An agent can answer almost entirely from:

- `lineage_goal` + `message` + filename, maybe with a tiny code/context snippet.

**Rule of thumb:**  
If the question is “why was this introduced/removed/renamed?”, try to answer from `lineage_goal` before reaching for diffs.

### Step 4: When you actually need diffs

Only when:

- The question is about **behavior changes** (“when did it start returning null?”, “when did it stop calling X?”), or
- `lineage_goal` is too high-level,

should you pull real diffs.

In this repo, a local agent can:

```bash
git show -p <commit_id> -- <target_file>
# or with smaller context:
git show --unified=3 <commit_id> -- <target_file>
```

Then:

- Extract only the hunks around the target symbol or lines (to save tokens).
- Ask the model:

  > “Given this diff for `<target_file>` and the current code for `<symbol>`, explain how the behavior of X changed.”

For a future remote/MCP world, this would be a natural small MCP tool:

- `get_commit_diff(commit_id, path, context_lines=3)` → returns only the relevant diff hunks as text.

But you don’t need that implemented yet to exercise the pattern locally.

---

## 3. Example in this repo (ctx.py cleanup)

Concrete run we just saw:

1. **Behavior:** “What happened to `ctx.py` in the VS Code extension?”

2. **repo_search (hypothetical):**

   - Find `vscode-extension/context-engine-uploader/scripts/ctx.py`.

3. **search_commits_for:**

   ```json
   search_commits_for(
     query: "ctx script build time",
     path: "vscode-extension/context-engine-uploader/scripts/ctx.py",
     limit: 3
   )
   ```

   One of the results:

   ```json
   {
     "commit_id": "e9d5...",
     "message": "Remove duplicate ctx script in extension - bundled at build time - ctx is available in-repo (scripts/ctx.py)",
     "lineage_goal": "Remove duplicate ctx script from extension as it's bundled at build time",
     "lineage_symbols": ["ctx.py","vscode-extension","build-time","bundled"],
     "lineage_tags": ["cleanup","duplicate","build","extension","script"]
   }
   ```

4. **Answer “why”**:

   - No diff needed: you can say:
     - It was removed as a duplicate because the script is bundled at build time and already available in-repo.
   - If you *also* want “how did the code change?”:
     - Pull `git show -p e9d5... -- vscode-extension/context-engine-uploader/scripts/ctx.py`.
     - Let the LLM confirm that the extension now uses the in-repo `scripts/ctx.py` and no longer ships a copy.

---

## 4. Where this goes next

We don’t need more MCP tools immediately; we have:

- [repo_search](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:1663:0-2468:5) → code now.
- [search_commits_for](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:2760:0-2898:65) → commit shortlist with lineage summaries.
- [change_history_for_path(include_commits=true)](cci:1://file:///home/coder/project/Context-Engine/scripts/mcp_indexer_server.py:2901:0-3015:56) → file-level view with recent commits.

Polish / next actions (conceptual, not coding yet):

- Encode this **4-step playbook** into an “advanced: context lineage” section in [CLAUDE.md](cci:7://file:///home/coder/project/CLAUDE.md:0:0-0:0) for agents.
- Later, if needed, introduce a tiny `get_commit_diff` MCP tool for remote setups; locally, continue to use `git show` directly.

If you want, next step we can actually draft that “advanced lineage workflow” section text for [CLAUDE.md](cci:7://file:///home/coder/project/CLAUDE.md:0:0-0:0), using the above structure but even more compressed for agents.

---

## 5. "Bad message" / good summary sanity check

Question: *Is GLM just parroting commit messages, or is it actually reading diffs and/or detailed bodies?*

### Commit under test

- **SHA:** `6adced4ed83adf75ad8f8c2649b4599a68fb53ae`
- **Subject:** `fix`
- **Body (excerpt):**
  - `What this fixes:`
  - `stopProcesses will not resolve prematurely`
  - `runSequence cannot start a new watcher while the previous one is still alive`
  - `Resilient to processes that ignore SIGTERM`
- **Files touched (relevant):**
  - `vscode-extension/context-engine-uploader/extension.js`

Diff (abridged) shows changes to `terminateProcess(proc, label)`:

- Introduces `termTimer` / `killTimer` and a `cleanup()` helper.
- Makes `finalize(reason)` idempotent and ensures timers are cleared.
- Hooks `exit` / `close` handlers into a shared `onExit` that calls `finalize` with exit status / signal.
- Keeps an initial `proc.kill()` (SIGTERM), then:
  - Waits `waitSigtermMs` (4s), then tries `proc.kill('SIGKILL')` and logs a message.
  - After an additional `waitSigkillMs` (2s), forces `finalize` with a “forced after X ms” reason.

### Lineage summary produced by GLM

From `search_commits_for(query="fix", path="", limit=5)` we see, for this SHA:

```json
{
  "commit_id": "6adced4ed83adf75ad8f8c2649b4599a68fb53ae",
  "message": "fix",
  "files": [".env.example", "vscode-extension/context-engine-uploader/extension.js"],
  "lineage_goal": "Fix process termination and watcher lifecycle issues",
  "lineage_symbols": [
    "SIGTERM",
    "SIGKILL",
    "watchProcess",
    "forceProcess"
  ],
  "lineage_tags": [
    "process-management",
    "termination",
    "watcher",
    "lifecycle",
    "signal-handling"
  ]
}
```

### Interpretation

- The **subject** alone (`fix`) is non-informative.
- The **body** gives some English hints about watcher behavior and SIGTERM resilience.
- The **diff** clearly shows:
  - A more robust termination sequence (SIGTERM → SIGKILL → forced finalize).
  - Explicit references to `watchProcess`, `forceProcess`, and signal names.

The GLM summary:

- Captures the high-level intent (`process termination and watcher lifecycle issues`).
- Names concrete symbols seen in the diff (`SIGTERM`, `SIGKILL`, `watchProcess`, `forceProcess`).
- Adds tags (`process-management`, `signal-handling`, etc.) that do not appear verbatim in the subject.

Conclusion for this case:

- `lineage_goal` is *not* just a restatement of the one-word subject; it reflects both the commit body and the structure of the diff.
- `lineage_symbols` / `lineage_tags` show that GLM is paying attention to changed identifiers and behavior, making this commit discoverable via queries like `"watcher lifecycle"`, `"SIGTERM"`, or `"process termination"` even though the subject is just `fix`.