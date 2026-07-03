curl -s "http://localhost:6333/collections/Context-Engine-41e67959/points/scroll" -H "Content-Type: application/json" -d '{"filter":{"must":[{"key":"metadata.language","match":{"value":"git"}},{"key":"metadata.kind","match":{"value":"git_message"}}]},"limit":5,"with_payload":true,"with_vector":false}'




set -a; . .env; set +a; REFRAG_DECODER=1 REFRAG_RUNTIME=glm REFRAG_COMMIT_DESCRIBE=1 python3 - << 'PY'
from scripts.refrag_llamacpp import is_decoder_enabled, get_runtime_kind
from scripts.ingest_history import commit_metadata, generate_commit_summary, run

print('is_decoder_enabled:', is_decoder_enabled())
print('runtime:', get_runtime_kind())

sha = run('git rev-list --max-count=1 HEAD').strip()
md = commit_metadata(sha)

diff = run(f'git show --stat --patch --unified=3 {sha}')
print('Testing commit:', sha)
print('Files:', md.get('files'))

goal, symbols, tags = generate_commit_summary(md, diff)
print('goal:', repr(goal))
print('symbols:', symbols)
print('tags:', tags)
PY



Index commits:
set -a; . .env; set +a; COLLECTION_NAME=Context-Engine-41e67959 QDRANT_URL=http://localhost:6333 REFRAG_DECODER=1 REFRAG_RUNTIME=glm REFRAG_COMMIT_DESCRIBE=1 python3 -m scripts.ingest_history --since '6 months ago' --max-commits 10 --per-batch 10