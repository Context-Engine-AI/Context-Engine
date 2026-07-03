#!/usr/bin/env bash
set -euo pipefail
# Use Python stdlib to avoid curl dependency in the container
until python - <<'PY'
import os, sys, urllib.request
url = os.environ.get("QDRANT_URL", "http://localhost:6333")
if not url.endswith("/"):
    url += "/"
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        sys.exit(0 if getattr(r, "status", 200) < 500 else 1)
except Exception:
    sys.exit(1)
PY
do
  echo "Waiting for Qdrant at ${QDRANT_URL:-http://localhost:6333} ..."
  sleep 1
done
echo "Qdrant is up."

