// CLI entrypoint for ctxce

import process from "node:process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runMcpServer } from "./mcpServer.js";

export async function runCli() {
  const argv = process.argv.slice(2);
  const cmd = argv[0];

  if (cmd === "mcp-serve") {
    // Minimal flag parsing for PoC: allow passing workspace/root and indexer URL.
    // Supported flags:
    //   --workspace / --path   : workspace root (default: cwd)
    //   --indexer-url          : override MCP indexer URL (default env CTXCE_INDEXER_URL or http://localhost:8003/mcp)
    const args = argv.slice(1);
    let workspace = process.cwd();
    let indexerUrl = process.env.CTXCE_INDEXER_URL || "http://localhost:8003/mcp";
    let memoryUrl = process.env.CTXCE_MEMORY_URL || null;

    for (let i = 0; i < args.length; i += 1) {
      const a = args[i];
      if (a === "--workspace" || a === "--path") {
        if (i + 1 < args.length) {
          workspace = args[i + 1];
          i += 1;
          continue;
        }
      }
      if (a === "--indexer-url") {
        if (i + 1 < args.length) {
          indexerUrl = args[i + 1];
          i += 1;
          continue;
        }
      }
      if (a === "--memory-url") {
        if (i + 1 < args.length) {
          memoryUrl = args[i + 1];
          i += 1;
          continue;
        }
      }
    }

    // eslint-disable-next-line no-console
    console.error(
      `[ctxce] Starting MCP bridge: workspace=${workspace}, indexerUrl=${indexerUrl}, memoryUrl=${memoryUrl || "disabled"}`,
    );
    await runMcpServer({ workspace, indexerUrl, memoryUrl });
    return;
  }

  // Default help
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const binName = "ctxce";

  // eslint-disable-next-line no-console
  console.error(
    `Usage: ${binName} mcp-serve [--workspace <path>] [--indexer-url <url>] [--memory-url <url>]`,
  );
  process.exit(1);
}
