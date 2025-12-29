const vscode = require('vscode');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const { ensureAuthIfRequired, runAuthLoginFlow, runAuthLogoutFlow } = require('./auth_utils');
const profiles = require('./profiles');
const sidebar = require('./sidebar');
const { createBridgeManager } = require('./mcp_bridge');
const { createMcpConfigManager } = require('./mcp_config');
const { createCtxConfigManager } = require('./ctx_config');
const { createLogsTerminalManager } = require('./logs_terminal');
const { createPromptPlusManager } = require('./prompt_plus');
const { registerPromptPlusCommands } = require('./prompt_plus_commands');
const { createOnboardingManager } = require('./onboarding');
const { createPythonEnvManager } = require('./python_env');
const { createProcessManager } = require('./process_manager');
const { registerExtensionCommands } = require('./commands');
const { createConfigResolver } = require('./config_resolver');
let outputChannel;
let extensionRoot;
let statusBarItem;
let promptStatusBarItem;
let logsTerminalManager;
let statusMode = 'idle';
let globalStoragePath;
let pythonOverridePath;
let bridgeManager;
let mcpConfigManager;
let ctxConfigManager;

let promptPlusManager;
let onboardingManager;
let pythonEnvManager;
let processManager;
let configResolver;
let sidebarApi;
let pendingProfileRestartTimer;
const DEFAULT_CONTAINER_ROOT = '/work';
// const CLAUDE_HOOK_COMMAND = '/home/coder/project/Context-Engine/ctx-hook-simple.sh';

function getEffectiveConfig() {
  try {
    return profiles.getUploaderConfig();
  } catch (_) {
    return vscode.workspace.getConfiguration('contextEngineUploader');
  }
}

function getResolvedTargetPathForSidebar() {
  try {
    const config = getEffectiveConfig();
    const result = configResolver ? configResolver.resolveTargetPathFromConfig(config) : { path: (config.get('targetPath') || '').trim(), inspected: {}, inferred: false };
    let target = result && result.path ? result.path : undefined;
    let source = (result && result.inferred) ? 'inferred' : 'settings';
    const watched = processManager ? processManager.getWatchedTargetPath() : undefined;
    if (!target && watched) {
      target = watched;
      source = 'runtime';
    }
    return { path: target, source };
  } catch (_) {
    const watched = processManager ? processManager.getWatchedTargetPath() : undefined;
    return { path: watched, source: watched ? 'runtime' : undefined };
  }
}

function scheduleRestartAfterProfileChange() {
  try {
    if (pendingProfileRestartTimer) {
      clearTimeout(pendingProfileRestartTimer);
      pendingProfileRestartTimer = undefined;
    }
  } catch (_) {
  }
  try {
    pendingProfileRestartTimer = setTimeout(() => {
      pendingProfileRestartTimer = undefined;
      if (!processManager || !processManager.getState().watchProcess) {
        return;
      }
      runSequence('auto').catch(error => log(`Auto-restart after profile change failed: ${error instanceof Error ? error.message : String(error)}`));
    }, 250);
  } catch (_) {
  }
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel('Context Engine Upload');
  context.subscriptions.push(outputChannel);
  extensionRoot = context.extensionPath;
  globalStoragePath = context.globalStorageUri && context.globalStorageUri.fsPath ? context.globalStorageUri.fsPath : undefined;
  try {
    profiles.init({
      vscode,
      context,
      log,
      onProfileChanged: () => {
        try { if (configResolver) configResolver.ensureTargetPathConfigured(); } catch (_) { }
        try {
          if (processManager && processManager.getState().watchProcess) {
            scheduleRestartAfterProfileChange();
          }
        } catch (_) { }
      },
    });
  } catch (error) {
    log(`Profiles init failed: ${error instanceof Error ? error.message : String(error)}`);
  }



  try {
    logsTerminalManager = createLogsTerminalManager({
      vscode,
      fs,
      log,
      getEffectiveConfig,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
    });
  } catch (error) {
    logsTerminalManager = undefined;
    log(`Logs terminal manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    promptPlusManager = createPromptPlusManager({
      vscode,
      spawn,
      path,
      fs,
      log,
      extensionRoot,
      getEffectiveConfig,
      getTargetPath: (c) => configResolver ? configResolver.getTargetPath(c) : undefined,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      detectDefaultTargetPath: (p) => configResolver ? configResolver.detectDefaultTargetPath(p) : p,
      resolveBridgeHttpUrl: () => bridgeManager ? bridgeManager.resolveBridgeHttpUrl() : undefined,
      getPythonOverridePath: () => pythonOverridePath,
      appendOutput: (text) => {
        if (outputChannel) {
          outputChannel.append(text);
        }
      },
    });
  } catch (error) {
    promptPlusManager = undefined;
    log(`Prompt+ manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    onboardingManager = createOnboardingManager({
      vscode,
      context,
      log,
      appendOutput: text => {
        if (outputChannel) {
          outputChannel.append(text);
        }
      },
      showOutput: () => {
        if (outputChannel) {
          outputChannel.show(true);
        }
      },
    });
  } catch (error) {
    onboardingManager = undefined;
    log(`Onboarding manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  configResolver = createConfigResolver({
    vscode,
    path,
    fs,
    log,
    getEffectiveConfig,
    getPythonOverridePath: () => pythonOverridePath,
    getExtensionRoot: () => extensionRoot,
    getStatusBarItem: () => statusBarItem,
    DEFAULT_CONTAINER_ROOT
  });

  try {
    pythonEnvManager = createPythonEnvManager({
      vscode,
      spawn: spawn,
      path,
      fs,
      log,
      getEffectiveConfig,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      getExtensionRoot: () => extensionRoot,
      getGlobalStoragePath: () => globalStoragePath,
      getPythonOverridePath: () => pythonOverridePath,
      setPythonOverridePath: (p) => { pythonOverridePath = p; },
    });
  } catch (error) {
    pythonEnvManager = undefined;
    log(`Python env manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    processManager = createProcessManager({
      vscode,
      spawn: spawn,
      fs,
      path,
      log,
      getEffectiveConfig,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      getOutputChannel: () => outputChannel,
      setStatusBarState,
      getStatusMode: () => statusMode,
      getExtensionRoot: () => extensionRoot,
    });
  } catch (error) {
    processManager = undefined;
    log(`Process manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    bridgeManager = createBridgeManager({
      vscode,
      spawn: spawn,
      path,
      fs,
      log,
      getEffectiveConfig,
      resolveBridgeWorkspacePath: () => configResolver ? configResolver.resolveBridgeWorkspacePath() : undefined,
      attachOutput: (child, label) => processManager ? processManager.attachOutput(child, label) : undefined,
      terminateProcess: (proc, label, afterStop) => processManager ? processManager.terminateProcess(proc, label, afterStop) : Promise.resolve(),
      scheduleMcpConfigRefreshAfterBridge: (delay) => mcpConfigManager ? mcpConfigManager.scheduleMcpConfigRefreshAfterBridge(delay) : undefined,
    });
  } catch (error) {
    bridgeManager = undefined;
    log(`Bridge manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    ctxConfigManager = createCtxConfigManager({
      vscode,
      spawnSync: spawnSync,
      log,
      extensionRoot,
      getEffectiveConfig,
      resolveOptions: () => configResolver ? configResolver.resolveOptions() : undefined,
      ensurePythonDependencies: (pythonPath) => pythonEnvManager ? pythonEnvManager.ensurePythonDependencies(pythonPath) : Promise.resolve(false),
      buildChildEnv: (options) => processManager?.buildChildEnv?.(options) ?? {},
      resolveBridgeHttpUrl: () => bridgeManager ? bridgeManager.resolveBridgeHttpUrl() : undefined,
    });
  } catch (error) {
    ctxConfigManager = undefined;
    log(`CTX config manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    mcpConfigManager = createMcpConfigManager({
      vscode,
      log,
      extensionRoot,
      getEffectiveConfig,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      resolveBridgeWorkspacePath: () => configResolver ? configResolver.resolveBridgeWorkspacePath() : undefined,
      normalizeBridgeUrl: (url) => bridgeManager ? bridgeManager.normalizeBridgeUrl(url) : (url || '').trim(),
      normalizeWorkspaceForBridge: (p) => bridgeManager ? bridgeManager.normalizeWorkspaceForBridge(p) : p,
      resolveBridgeCliInvocation: () => bridgeManager ? bridgeManager.resolveBridgeCliInvocation() : undefined,
      resolveBridgeHttpUrl: () => bridgeManager ? bridgeManager.resolveBridgeHttpUrl() : undefined,
      requiresHttpBridge: (s, t) => bridgeManager ? bridgeManager.requiresHttpBridge(s, t) : (s === 'bridge' && t === 'http'),
      ensureHttpBridgeReadyForConfigs: () => bridgeManager ? bridgeManager.ensureReadyForConfigs() : Promise.resolve(false),
      getBridgeIsRunning: () => (bridgeManager && typeof bridgeManager.isRunning === 'function' ? bridgeManager.isRunning() : false),
      writeCtxConfig: () => ctxConfigManager ? ctxConfigManager.writeCtxConfig() : Promise.resolve(),
    });
  } catch (error) {
    mcpConfigManager = undefined;
    log(`MCP config manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    // Ensure manager resources are cleaned up when the extension deactivates.
    // All manager disposals are registered via context.subscriptions, so deactivate()
    // only needs to return a resolved promise - VS Code handles the cleanup automatically.
    const managerDisposable = {
      dispose: () => {
        try { if (processManager) { processManager.disposeIndexedWatcher(); processManager.dispose(); } } catch (_) { }
        try { if (mcpConfigManager && typeof mcpConfigManager.dispose === 'function') mcpConfigManager.dispose(); } catch (_) { }
        try { if (ctxConfigManager && typeof ctxConfigManager.dispose === 'function') ctxConfigManager.dispose(); } catch (_) { }
        try { if (bridgeManager && typeof bridgeManager.dispose === 'function') bridgeManager.dispose(); } catch (_) { }
        try { if (logsTerminalManager && typeof logsTerminalManager.dispose === 'function') logsTerminalManager.dispose(); } catch (_) { }
        try { if (promptPlusManager && typeof promptPlusManager.dispose === 'function') promptPlusManager.dispose(); } catch (_) { }
        try { if (onboardingManager && typeof onboardingManager.dispose === 'function') onboardingManager.dispose(); } catch (_) { }
      }
    };
    context.subscriptions.push(managerDisposable);
  } catch (_) {
    // ignore
  }
  try {
    const venvPy = pythonEnvManager ? pythonEnvManager.resolvePrivateVenvPython() : undefined;
    if (venvPy) {
      pythonOverridePath = venvPy;
      log(`Detected existing private venv interpreter: ${venvPy}`);
    }
  } catch (_) { }
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'contextEngineUploader.indexCodebase';
  context.subscriptions.push(statusBarItem);
  statusBarItem.show();
  setStatusBarState('idle');
  if (configResolver) configResolver.updateStatusBarTooltip();
  promptStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 90);
  promptStatusBarItem.command = 'contextEngineUploader.promptEnhance';
  promptStatusBarItem.text = '$(sparkle) Prompt+';
  promptStatusBarItem.tooltip = 'Enhance selection with Unicorn Mode via ctx.py';
  context.subscriptions.push(promptStatusBarItem);
  promptStatusBarItem.show();


  try {
    const disposables = profiles.registerCommands({
      resolveTargetPathFromConfig: (c) => configResolver ? configResolver.resolveTargetPathFromConfig(c) : { path: (c.get('targetPath') || '').trim(), inspected: {}, inferred: false },
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      detectDefaultTargetPath: (p) => configResolver ? configResolver.detectDefaultTargetPath(p) : p,
      normalizeWorkspaceForBridge: (p) => bridgeManager ? bridgeManager.normalizeWorkspaceForBridge(p) : p,
      runSequence,
      writeMcpConfig: (o) => mcpConfigManager ? mcpConfigManager.writeMcpConfig(o) : Promise.resolve(),
      writeCtxConfig: () => ctxConfigManager ? ctxConfigManager.writeCtxConfig() : Promise.resolve(),
      fetch: (typeof fetch === 'function' ? fetch : undefined),
    });
    if (Array.isArray(disposables)) {
      context.subscriptions.push(...disposables);
    }
  } catch (error) {
    log(`Profiles command registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    sidebarApi = sidebar.register(context, {
      profiles,
      getEffectiveConfig,
      getResolvedTargetPath: getResolvedTargetPathForSidebar,
      getState: () => ({
        statusMode,
        httpBridgeProcess: bridgeManager ? bridgeManager.getState().process : undefined,
        httpBridgePort: bridgeManager ? bridgeManager.getState().port : undefined,
      }),
      onboarding: onboardingManager,
      resolveBridgeCliInvocation: () => bridgeManager ? bridgeManager.resolveBridgeCliInvocation() : undefined,
      getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
      spawn,
      log,
    });
  } catch (error) {
    log(`Sidebar registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  // Register extension commands via the commands module
  try {
    const commandDisposables = registerExtensionCommands({
      vscode,
      log,
      getEffectiveConfig,
      getOutputChannel: () => outputChannel,
      runSequence,
      stopProcesses: () => processManager ? processManager.stopProcesses() : Promise.resolve(),
      writeMcpConfig: (options) => mcpConfigManager ? mcpConfigManager.writeMcpConfig(options) : Promise.resolve(),
      writeCtxConfig: () => ctxConfigManager ? ctxConfigManager.writeCtxConfig() : Promise.resolve(),
      startHttpBridgeProcess,
      stopHttpBridgeProcess,
      buildAuthDeps,
      runAuthLoginFlow,
      runAuthLogoutFlow,
      getOnboardingManager: () => onboardingManager,
      getLogsTerminalManager: () => logsTerminalManager,
    });
    if (Array.isArray(commandDisposables)) {
      context.subscriptions.push(...commandDisposables);
    }
  } catch (error) {
    log(`Command registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    const promptDisposables = registerPromptPlusCommands({
      vscode,
      fs,
      path,
      log,
      getEffectiveConfig,
      resolveTargetPathFromConfig: (c) => configResolver ? configResolver.resolveTargetPathFromConfig(c) : undefined,
      writeCtxConfig: () => ctxConfigManager ? ctxConfigManager.writeCtxConfig() : Promise.resolve(),
      getPromptPlusManager: () => promptPlusManager,
      getSidebarApi: () => sidebarApi,
    });
    if (Array.isArray(promptDisposables) && promptDisposables.length) {
      context.subscriptions.push(...promptDisposables);
    }
  } catch (error) {
    log(`Prompt+ command registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  const configDisposable = vscode.workspace.onDidChangeConfiguration(event => {
    if (event.affectsConfiguration('contextEngineUploader') && processManager && processManager.getState().watchProcess) {
      runSequence('auto').catch(error => log(`Auto-restart failed: ${error instanceof Error ? error.message : String(error)}`));
    }
    if (event.affectsConfiguration('contextEngineUploader.targetPath')) {
      if (configResolver) configResolver.updateStatusBarTooltip();
    }
    if (
      event.affectsConfiguration('contextEngineUploader.mcpIndexerUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpMemoryUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpClaudeEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpWindsurfEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpAugmentEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpTransportMode') ||
      event.affectsConfiguration('contextEngineUploader.windsurfMcpPath') ||
      event.affectsConfiguration('contextEngineUploader.augmentMcpPath') ||
      event.affectsConfiguration('contextEngineUploader.claudeHookEnabled') ||
      event.affectsConfiguration('contextEngineUploader.surfaceQdrantCollectionHint')
    ) {
      // Best-effort auto-update of MCP + hook configurations when settings change
      if (mcpConfigManager) mcpConfigManager.writeMcpConfig().catch(error => log(`Auto MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
    }
    if (
      event.affectsConfiguration('contextEngineUploader.autoStartMcpBridge') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgePort') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgeBinPath') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgeLocalOnly') ||
      event.affectsConfiguration('contextEngineUploader.mcpIndexerUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpMemoryUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpServerMode') ||
      event.affectsConfiguration('contextEngineUploader.mcpTransportMode')
    ) {
      if (bridgeManager) { bridgeManager.handleSettingsChanged().catch(error => log(`HTTP MCP bridge restart failed: ${error instanceof Error ? error.message : String(error)}`)); }
    }
  });
  const workspaceDisposable = vscode.workspace.onDidChangeWorkspaceFolders(() => {
    if (configResolver) configResolver.ensureTargetPathConfigured();
  });
  const terminalCloseDisposable = vscode.window.onDidCloseTerminal(term => {
    try {
      if (logsTerminalManager && typeof logsTerminalManager.handleDidCloseTerminal === 'function') {
        logsTerminalManager.handleDidCloseTerminal(term);
      }
    } catch (_) { }
  });
  context.subscriptions.push(
    configDisposable,
    workspaceDisposable,
    terminalCloseDisposable
  );
  const config = getEffectiveConfig();
  if (configResolver) configResolver.ensureTargetPathConfigured();

  if (onboardingManager && typeof onboardingManager.checkOnboarding === 'function') {
    onboardingManager.checkOnboarding(config, configResolver);
  }
  if (config.get('runOnStartup')) {
    runSequence('auto').catch(error => log(`Startup run failed: ${error instanceof Error ? error.message : String(error)}`));
  }

  // Optionally keep MCP + hook + ctx config in sync on activation
  if (config.get('autoWriteMcpConfigOnStartup')) {
    if (mcpConfigManager) mcpConfigManager.writeMcpConfig().catch(error => log(`MCP config auto-write on activation failed: ${error instanceof Error ? error.message : String(error)}`));
  } else if (config.get('scaffoldCtxConfig', true)) {
    // Legacy behavior: scaffold ctx_config.json/.env directly when MCP auto-write is disabled
    if (ctxConfigManager) ctxConfigManager.writeCtxConfig().catch(error => log(`CTX config auto-scaffold on activation failed: ${error instanceof Error ? error.message : String(error)}`));
  }
  if (config.get('autoStartMcpBridge', false)) {
    const transportModeRaw = config.get('mcpTransportMode') || 'sse-remote';
    const serverModeRaw = config.get('mcpServerMode') || 'bridge';
    const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
    const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
    if (bridgeManager && bridgeManager.requiresHttpBridge(serverMode, transportMode)) {
      startHttpBridgeProcess().catch(error => log(`Auto-start HTTP MCP bridge failed: ${error instanceof Error ? error.message : String(error)}`));
    } else {
      log('Context Engine Uploader: autoStartMcpBridge is enabled, but current MCP wiring does not use the HTTP bridge; skipping auto-start.');
    }
  }
}
function buildAuthDeps() {
  return {
    vscode,
    spawn,
    spawnSync,
    resolveBridgeCliInvocation: () => bridgeManager ? bridgeManager.resolveBridgeCliInvocation() : undefined,
    getWorkspaceFolderPath: () => configResolver ? configResolver.getWorkspaceFolderPath() : undefined,
    attachOutput: (child, label) => processManager ? processManager.attachOutput(child, label) : undefined,
    log,
    getEffectiveConfig,
    fetchGlobal: (typeof fetch === 'function' ? fetch : undefined),
  };
}
async function runSequence(mode = 'auto') {
  const options = configResolver ? configResolver.resolveOptions() : undefined;
  if (!options) {
    return;
  }

  try {
    await ensureAuthIfRequired(options.endpoint, buildAuthDeps());
  } catch (error) {
    log(`Auth preflight check failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  const depsSatisfied = pythonEnvManager
    ? await pythonEnvManager.ensurePythonDependencies(options.pythonPath)
    : false;
  if (!depsSatisfied) {
    setStatusBarState('idle');
    return;
  }
  // Re-resolve options in case ensurePythonDependencies switched to a better interpreter
  const reoptions = configResolver ? configResolver.resolveOptions() : undefined;
  if (reoptions) {
    Object.assign(options, reoptions);
  }
  if (processManager) {
    await processManager.stopProcesses();
  }
  const needsForce = mode === 'force' || mode === 'uploadGitHistory' || (configResolver ? configResolver.needsForceSync(options.targetPath) : true);
  if (needsForce) {
    setStatusBarState('indexing');
    if (outputChannel) { outputChannel.show(true); }
    const code = processManager ? await processManager.runOnce(options) : 1;
    if (code === 0) {
      setStatusBarState('indexed');
      if (processManager) { processManager.ensureIndexedWatcher(options.targetPath); }
      // Only start watching after a regular force sync, not after git history upload
      if (mode === 'force' && options.startWatchAfterForce && processManager) {
        processManager.startWatch(options);
      }
    } else {
      setStatusBarState('idle');
    }
    return;
  }
  if (processManager) {
    processManager.startWatch(options);
  }
}



async function startHttpBridgeProcess() {
  if (bridgeManager && typeof bridgeManager.start === 'function') {
    return bridgeManager.start();
  }
  return undefined;
}

function stopHttpBridgeProcess() {
  if (bridgeManager && typeof bridgeManager.stop === 'function') {
    return bridgeManager.stop();
  }
  return Promise.resolve();
}

function setStatusBarState(mode) {
  if (!statusBarItem) {
    return;
  }
  statusMode = mode;
  if (mode === 'indexing') {
    statusBarItem.text = '$(sync~spin) Indexing...';
    statusBarItem.color = undefined;
  } else if (mode === 'indexed') {
    statusBarItem.text = '$(check) Indexed';
    statusBarItem.color = new vscode.ThemeColor('charts.green');
  } else if (mode === 'watch') {
    statusBarItem.text = '$(sync) Watching (Click Force Index)';
    statusBarItem.color = new vscode.ThemeColor('charts.purple');
  } else {
    statusBarItem.text = '$(sync) Index Codebase';
    statusBarItem.color = undefined;
  }
}
function log(message) {
  if (!outputChannel) {
    return;
  }
  const timestamp = new Date().toISOString();
  outputChannel.appendLine(`[${timestamp}] ${message}`);
}

function deactivate() {
  // All manager disposals are handled via context.subscriptions.
  // VS Code automatically calls dispose() on all subscriptions when deactivating.
  return Promise.resolve();
}
module.exports = {
  activate,
  deactivate
};
