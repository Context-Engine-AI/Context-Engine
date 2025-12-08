// MCP stdio server implemented using the official MCP TypeScript SDK.
// Acts as a low-level proxy for tools, forwarding tools/list and tools/call
// to the remote qdrant-indexer MCP server while adding a local `ping` tool.

import process from "node:process";
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

export async function runMcpServer(options) {
  const workspace = options.workspace || process.cwd();
  const indexerUrl = options.indexerUrl;

  const config = loadConfig(workspace);
  const defaultCollection =
    config && typeof config.default_collection === "string"
      ? config.default_collection
      : null;

  // eslint-disable-next-line no-console
  console.error(
    `[ctxce] MCP low-level stdio bridge starting: workspace=${workspace}, indexerUrl=${indexerUrl}`,
  );

  if (defaultCollection) {
    // eslint-disable-next-line no-console
    console.error(
      `[ctxce] Using default collection from ctx_config.json: ${defaultCollection}`,
    );
  }

  // High-level MCP client for the remote HTTP /mcp indexer
  const clientTransport = new StreamableHTTPClientTransport(indexerUrl);
  const client = new Client(
    {
      name: "ctx-context-engine-bridge-http-client",
      version: "0.0.1",
    },
    {
      capabilities: {
        tools: {},
        resources: {},
        prompts: {},
      },
    },
  );

  try {
    await client.connect(clientTransport);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[ctxce] Failed to connect MCP HTTP client to indexer:", err);
    throw err;
  }

  const server = new Server(
    {
      name: "ctx-context-engine-bridge",
      version: "0.0.1",
    },
    {
      capabilities: {
        tools: {},
      },
    },
  );

  // tools/list → fetch tools from remote indexer and append local ping tool
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    let remote;
    try {
      remote = await client.listTools();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[ctxce] Error calling remote tools/list:", err);
      return { tools: [buildPingTool()] };
    }

    // eslint-disable-next-line no-console
    console.error("[ctxce] tools/list remote result:", JSON.stringify(remote));

    const tools = Array.isArray(remote?.tools) ? remote.tools.slice() : [];
    tools.push(buildPingTool());
    return { tools };
  });

  // tools/call → handle ping locally, everything else is proxied to indexer
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const params = request.params || {};
    const name = params.name;
    let args = params.arguments;

    if (name === "ping") {
      const branch = detectGitBranch(workspace);
      const text = args && typeof args.text === "string" ? args.text : "pong";
      const suffix = branch ? ` (branch=${branch})` : "";
      return {
        content: [
          {
            type: "text",
            text: `${text}${suffix}`,
          },
        ],
      };
    }

    // Inject default collection when not explicitly provided and arguments
    // are an object (indexer tools accept a collection parameter).
    if (
      defaultCollection &&
      (args === undefined || args === null || typeof args === "object")
    ) {
      const obj = args && typeof args === "object" ? { ...args } : {};
      if (
        !Object.prototype.hasOwnProperty.call(obj, "collection") ||
        obj.collection === undefined ||
        obj.collection === null ||
        obj.collection === ""
      ) {
        obj.collection = defaultCollection;
      }
      args = obj;
    }

    const result = await client.callTool({
      name,
      arguments: args,
    });
    return result;
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

function loadConfig(startDir) {
  try {
    let dir = startDir;
    for (let i = 0; i < 5; i += 1) {
      const cfgPath = path.join(dir, "ctx_config.json");
      if (fs.existsSync(cfgPath)) {
        try {
          const raw = fs.readFileSync(cfgPath, "utf8");
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") {
            return parsed;
          }
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error("[ctxce] Failed to parse ctx_config.json:", err);
          return null;
        }
      }
      const parent = path.dirname(dir);
      if (!parent || parent === dir) {
        break;
      }
      dir = parent;
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[ctxce] Error while loading ctx_config.json:", err);
  }
  return null;
}

function buildPingTool() {
  return {
    name: "ping",
    description: "Basic ping tool exposed by the ctx bridge",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Optional text to echo back.",
        },
      },
      required: [],
    },
  };
}

function detectGitBranch(workspace) {
  try {
    const out = execSync("git rev-parse --abbrev-ref HEAD", {
      cwd: workspace,
      stdio: ["ignore", "pipe", "ignore"],
    });
    const name = out.toString("utf8").trim();
    return name || null;
  } catch {
    return null;
  }
}

