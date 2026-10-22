---
name: context-engine
description: Search and understand the indexed codebase with Context Engine (hybrid semantic + lexical search, reranked, with a symbol graph). Use for any question about this codebase - where something is implemented, how it works, callers/definitions, every place that does X - before ls/find/Grep/Read.
---

# Context Engine

When the `context-engine` MCP server is connected, this workspace is indexed: for any question about the codebase, call `repo_search` FIRST, before ls/find/Grep/Read, and again whenever you move to a new area; for how/why questions that span several files use `context_answer`. Results are verified source with exact line ranges: treat them as already read. Use Grep only to confirm an exact literal in a file you already know, or to count. For questions asking for every/all sites, run one repo_search per distinct pattern (queries=[...]) and follow the refs. If the tools are listed as deferred, load them with ToolSearch first.

| Need | Tool |
|---|---|
| Find code by behavior, concept, error text or identifier | `repo_search` (`queries=[...]` for several at once; `under`, `language`, `path_glob` to narrow) |
| Explain how/why across several files, with citations | `context_answer` |
| Definition, callers, callees, importers, references of a symbol | `symbol_graph` (several names: `A\|B\|C`) |
| Multi-hop impact before a change | `graph_query` |
| Code that may live in another indexed repo | `cross_repo_search` |

Before editing, fetch every involved symbol in one call; before a rename or signature change, get every site with `symbol_graph` `query_type=references`. Search again only after learning something new.
