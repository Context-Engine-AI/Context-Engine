// Minimal JSON-RPC over HTTP client for the remote Context-Engine MCP indexer

const DEFAULT_INDEXER_URL = process.env.CTXCE_INDEXER_URL || "http://localhost:8003/mcp";

export class IndexerClient {
  constructor(url) {
    this.url = url || DEFAULT_INDEXER_URL;
  }

  async call(method, params) {
    const id = Date.now().toString() + Math.random().toString(16).slice(2);
    const body = { jsonrpc: "2.0", id, method, params };

    const res = await fetch(this.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`[indexer] HTTP ${res.status}: ${await res.text()}`);
    }

    const json = await res.json();
    if (json.error) {
      throw json.error;
    }
    return json.result;
  }
}
