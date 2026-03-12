# Context Engine Tool Reference

This file is a **Codex quick reference**, not the canonical parameter spec. For full semantics and examples, defer to `skills/context-engine/SKILL.md` and the live MCP tool schemas.

## Default Entry Points

| Tool | Reach for it when | Notes |
|------|-------------------|-------|
| `search` | You have a general codebase question, lookup, or exploration task | Default first tool. Auto-routes to code search, Q&A, tests, config, symbols, and import lookups. |
| `repo_search` / `code_search` | You need direct code-search control | Good for explicit filters like `language`, `under`, `path_glob`, `not_glob`, and `repo`. |
| `batch_search` | You have 2+ independent code searches | Prefer this over repeated `repo_search` calls when the searches do not depend on one another. |
| `context_answer` | You want an explanation grounded in retrieved code | Use when the output should be synthesized rather than raw hits. |
| `info_request` | You want a lightweight natural-language lookup | Useful when you want discovery with minimal parameters. |

## Symbol and Relationship Tools

| Tool | Reach for it when | Notes |
|------|-------------------|-------|
| `symbol_graph` | You need direct callers, callees, definitions, importers, subclasses, or base classes | Use first for symbol relationships. Supports `depth` for multi-hop caller/callee traversals. |
| `search_callers_for` | You want a quick heuristic caller search | Broader and less precise than `symbol_graph`. |
| `search_importers_for` | You want a quick heuristic importer search | Use when text-level import searching is sufficient. |
| `graph_query` | You need deeper impact, dependency, transitive, or cycle analysis | Use **only if the tool is actually available** in the environment. Otherwise combine `symbol_graph` with targeted `search`. |

## Search Specializations

| Tool | Reach for it when | Notes |
|------|-------------------|-------|
| `search_tests_for` | You want tests related to a feature or symbol | `search` can route here automatically. |
| `search_config_for` | You want config or settings files | `search` can route here automatically. |
| `pattern_search` | You want structurally similar code | Optional; availability depends on deployment. |
| `search_commits_for` | You want commit history or co-change prediction | Use `predict_related=true` for historically coupled files. |
| `change_history_for_path` | You want a file-level change summary | Optionally include recent commits. |

## Multi-Repo and Memory

| Tool | Reach for it when | Notes |
|------|-------------------|-------|
| `cross_repo_search` | The question spans multiple repos or collections | Prefer this over ad hoc cross-repo search chains. |
| `context_search` | You want code + stored notes together | Set `include_memories=true`. |
| `memory_store` / `memory_find` | You want to persist or recall non-code knowledge | Use for decisions, conventions, gotchas, and notes. |
| `qdrant_status` / `qdrant_list` | You are debugging search/index availability | Diagnostics only; do not assume indexing tools exist unless the live tool schema exposes them. |

## Common Filter Reminders

- `language` — narrow by language.
- `under` — restrict to a path prefix.
- `path_glob` / `not_glob` — include or exclude paths.
- `repo` — limit to repo names when supported.
- `include_snippet` — include code excerpts in results.
- `compact` / `output_format="toon"` — reduce token usage during discovery.

## Exploration Policy

- Prefer MCP tools over grep/file-open for cross-file exploration.
- Narrow grep/file-open usage is still okay for exact literal confirmation, exact path/file confirmation, or opening a file you already identified for editing.
- If this quick reference conflicts with the shared skill doc or the live tool schema, follow the shared skill doc or tool schema.

