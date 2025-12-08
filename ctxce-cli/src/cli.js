// CLI entrypoint for ctxce

import process from "node:process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runMcpServer } from "./mcpServer.js";

export async function runCli() {
  const argv = process.argv.slice(2);
  const cmd = argv[0];

  if (cmd === "mcp-serve") {
    // TODO: add proper argument parsing; PoC uses cwd as workspace
    const workspace = process.cwd();
    const indexerUrl = process.env.CTXCE_INDEXER_URL || "http://localhost:8003/mcp";

    // eslint-disable-next-line no-console
    console.error(`[ctxce] Starting MCP bridge: workspace=${workspace}, indexerUrl=${indexerUrl}`);
    await runMcpServer({ workspace, indexerUrl });
    return;
  }

  // Default help
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const binName = "ctxce";

  // eslint-disable-next-line no-console
  console.error(`Usage: ${binName} mcp-serve`);
  process.exit(1);
}
