function createBridgeManager(deps) {
  const vscode = deps.vscode;
  const spawn = deps.spawn;
  const path = deps.path;
  const fs = deps.fs;
  const log = deps.log;

  const getEffectiveConfig = deps.getEffectiveConfig;
  const resolveBridgeWorkspacePath = deps.resolveBridgeWorkspacePath;
  const attachOutput = deps.attachOutput;
  const terminateProcess = deps.terminateProcess;
  const scheduleMcpConfigRefreshAfterBridge = deps.scheduleMcpConfigRefreshAfterBridge;

  let httpBridgeProcess;
  let httpBridgePort;
  let httpBridgeWorkspace;
  let stopInFlight;

  function normalizeBridgeUrl(url) {
    if (!url || typeof url !== 'string') {
      return '';
    }
    const trimmed = url.trim();
    if (!trimmed) {
      return '';
    }
    return trimmed;
  }

  function normalizeWorkspaceForBridge(workspacePath) {
    if (!workspacePath || typeof workspacePath !== 'string') {
      return '';
    }
    try {
      const resolved = path.resolve(workspacePath);
      if (process.platform === 'win32') {
        return resolved.replace(/\//g, '\\');
      }
      return resolved;
    } catch (_) {
      return workspacePath;
    }
  }

  function findLocalBridgeBin() {
    let localOnly = true;
    let configured = '';
    try {
      const settings = getEffectiveConfig();
      localOnly = settings.get('mcpBridgeLocalOnly', true);
      configured = (settings.get('mcpBridgeBinPath') || '').trim();
    } catch (_) {
      // ignore config lookup failures
    }
    // When local-only is disabled, skip local resolution and always fall back to npx
    if (localOnly === false) {
      return undefined;
    }
    if (configured && fs.existsSync(configured)) {
      return path.resolve(configured);
    }
    const envOverride = (process.env.CTXCE_BRIDGE_BIN || '').trim();
    if (envOverride && fs.existsSync(envOverride)) {
      return path.resolve(envOverride);
    }
    return undefined;
  }

  function resolveBridgeCliInvocation() {
    const binPath = findLocalBridgeBin();
    if (binPath) {
      return {
        command: 'node',
        args: [binPath],
        kind: 'local'
      };
    }
    const isWindows = process.platform === 'win32';
    if (isWindows) {
      return {
        command: 'cmd',
        args: ['/c', 'npx', '@context-engine-bridge/context-engine-mcp-bridge'],
        kind: 'npx'
      };
    }
    return {
      command: 'npx',
      args: ['@context-engine-bridge/context-engine-mcp-bridge'],
      kind: 'npx'
    };
  }


  function getState() {
    return {
      process: httpBridgeProcess,
      port: httpBridgePort,
      workspace: httpBridgeWorkspace,
    };
  }

  function isRunning() {
    return !!httpBridgeProcess;
  }

  function requiresHttpBridge(serverMode, transportMode) {
    return serverMode === 'bridge' && transportMode === 'http';
  }

  function resolveBridgeHttpUrl() {
    try {
      const settings = getEffectiveConfig();
      let port = Number(settings.get('mcpBridgePort') || 30810);
      if (!Number.isFinite(port) || port <= 0) {
        port = 30810;
      }
      const hostname = '127.0.0.1';
      return `http://${hostname}:${port}/mcp`;
    } catch (error) {
      log(`Failed to resolve bridge HTTP URL: ${error instanceof Error ? error.message : String(error)}`);
      return undefined;
    }
  }

  function resolveHttpBridgeOptions() {
    try {
      const settings = getEffectiveConfig();
      const serverModeRaw = settings.get('mcpServerMode') || 'bridge';
      const transportModeRaw = settings.get('mcpTransportMode') || 'sse-remote';
      const serverMode = typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge';
      const transportMode = typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote';
      if (serverMode !== 'bridge') {
        vscode.window.showWarningMessage('Context Engine Uploader: MCP server mode is not "bridge"; HTTP bridge will connect to raw endpoints.');
      }
      if (transportMode !== 'http') {
        log('Context Engine Uploader: MCP transport mode is not "http"; HTTP bridge will still start but downstream configs may expect SSE.');
      }
      const workspacePath = resolveBridgeWorkspacePath();
      if (!workspacePath) {
        vscode.window.showErrorMessage('Context Engine Uploader: open a workspace or set contextEngineUploader.targetPath before starting HTTP MCP bridge.');
        return undefined;
      }
      let indexerUrl = (settings.get('mcpIndexerUrl') || 'http://localhost:8003/mcp').trim();
      let memoryUrl = (settings.get('mcpMemoryUrl') || 'http://localhost:8002/mcp').trim();
      indexerUrl = normalizeBridgeUrl(indexerUrl);
      memoryUrl = normalizeBridgeUrl(memoryUrl);
      let port = Number(settings.get('mcpBridgePort') || 30810);
      if (!Number.isFinite(port) || port <= 0) {
        port = 30810;
      }
      return {
        workspacePath,
        indexerUrl,
        memoryUrl,
        port,
      };
    } catch (error) {
      log(`Failed to resolve HTTP bridge options: ${error instanceof Error ? error.message : String(error)}`);
      return undefined;
    }
  }

  async function start() {
    if (httpBridgeProcess) {
      vscode.window.showInformationMessage(`Context Engine HTTP MCP bridge already running on port ${httpBridgePort || 'unknown'}.`);
      return httpBridgePort;
    }
    const options = resolveHttpBridgeOptions();
    if (!options) {
      return undefined;
    }
    const invocation = resolveBridgeCliInvocation();
    if (!invocation) {
      vscode.window.showErrorMessage('Context Engine Uploader: unable to locate ctxce CLI for HTTP bridge.');
      return undefined;
    }
    const cliArgs = ['mcp-http-serve'];
    if (options.workspacePath) {
      cliArgs.push('--workspace', normalizeWorkspaceForBridge(options.workspacePath));
    }
    if (options.indexerUrl) {
      cliArgs.push('--indexer-url', options.indexerUrl);
    }
    if (options.memoryUrl) {
      cliArgs.push('--memory-url', options.memoryUrl);
    }
    if (options.port) {
      cliArgs.push('--port', String(options.port));
    }
    const finalArgs = [...invocation.args, ...cliArgs];
    log(`Starting HTTP MCP bridge via ${invocation.command} ${finalArgs.join(' ')}`);
    const child = spawn(invocation.command, finalArgs, {
      cwd: options.workspacePath,
      env: process.env,
    });
    httpBridgeProcess = child;
    httpBridgePort = options.port;
    httpBridgeWorkspace = options.workspacePath;
    attachOutput(child, 'mcp-http');
    child.on('exit', (code, signal) => {
      log(`HTTP MCP bridge exited with code ${code} signal ${signal || ''}`.trim());
      if (httpBridgeProcess === child) {
        httpBridgeProcess = undefined;
        httpBridgePort = undefined;
        httpBridgeWorkspace = undefined;
      }
    });
    child.on('error', error => {
      log(`HTTP MCP bridge process error: ${error instanceof Error ? error.message : String(error)}`);
      if (httpBridgeProcess === child) {
        httpBridgeProcess = undefined;
        httpBridgePort = undefined;
        httpBridgeWorkspace = undefined;
      }
    });
    vscode.window.showInformationMessage(`Context Engine HTTP MCP bridge listening on http://127.0.0.1:${options.port}/mcp`);
    if (typeof scheduleMcpConfigRefreshAfterBridge === 'function') {
      scheduleMcpConfigRefreshAfterBridge();
    }
    return options.port;
  }

  function stop() {
    if (!httpBridgeProcess) {
      return Promise.resolve();
    }
    if (stopInFlight) {
      return stopInFlight;
    }
    const proc = httpBridgeProcess;
    stopInFlight = terminateProcess(
      proc,
      'mcp-http',
      () => {
        if (httpBridgeProcess === proc) {
          httpBridgeProcess = undefined;
          httpBridgePort = undefined;
          httpBridgeWorkspace = undefined;
        }
      }
    ).finally(() => {
      stopInFlight = undefined;
    });
    return stopInFlight;
  }

  async function ensureReadyForConfigs() {
    try {
      if (httpBridgeProcess) {
        return true;
      }
      await start();
      return !!httpBridgeProcess;
    } catch (error) {
      log(`Failed to ensure HTTP bridge is ready: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async function handleSettingsChanged() {
    const config = getEffectiveConfig();
    const shouldRun = !!config.get('autoStartMcpBridge', false);
    const wasRunning = !!httpBridgeProcess;
    if (httpBridgeProcess) {
      await stop();
    }
    if (shouldRun || wasRunning) {
      const transportModeRaw = config.get('mcpTransportMode') || 'sse-remote';
      const serverModeRaw = config.get('mcpServerMode') || 'bridge';
      const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
      const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
      if (requiresHttpBridge(serverMode, transportMode)) {
        await start();
      } else {
        log('Context Engine Uploader: HTTP bridge settings changed, but current MCP wiring does not use the HTTP bridge; not restarting HTTP bridge.');
      }
    }
  }

  function dispose() {
    try {
      // Best-effort shutdown; ignore errors
      stop().catch(() => { });
    } catch (_) {
      // ignore
    }
  }

  return {
    getState,
    isRunning,
    requiresHttpBridge,
    resolveBridgeHttpUrl,
    ensureReadyForConfigs,
    start,
    stop,
    handleSettingsChanged,
    dispose,
    // Utility functions for mcpConfigManager
    normalizeBridgeUrl,
    normalizeWorkspaceForBridge,
    resolveBridgeCliInvocation,
  };
}

module.exports = {
  createBridgeManager,
};
