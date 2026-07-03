#!/usr/bin/env node
import { runCli } from "../src/cli.js";

runCli().catch((err) => {
  console.error("[ctxce] Fatal error:", err && err.stack ? err.stack : err);
  process.exit(1);
});
