/**
 * Command registration for Context Engine extension.
 * Consolidates all vscode.commands.registerCommand calls.
 */
function registerExtensionCommands(deps) {
    if (!deps || typeof deps !== 'object') {
        throw new Error('registerExtensionCommands: deps object is required');
    }

    const vscode = deps.vscode;
    const log = deps.log;

    // Verify critical dependencies immediately
    if (!vscode) {
        throw new Error('Context Engine Uploader: vscode dependency is missing (extension failed to initialize).');
    }
    if (!log) {
        throw new Error('Context Engine Uploader: log dependency is missing (extension failed to initialize).');
    }

    const getEffectiveConfig = deps.getEffectiveConfig;
    const getOutputChannel = deps.getOutputChannel;
    const runSequence = deps.runSequence;
    const stopProcesses = deps.stopProcesses;
    const writeMcpConfig = deps.writeMcpConfig;
    const writeCtxConfig = deps.writeCtxConfig;
    const startHttpBridgeProcess = deps.startHttpBridgeProcess;
    const stopHttpBridgeProcess = deps.stopHttpBridgeProcess;
    const buildAuthDeps = deps.buildAuthDeps;
    const runAuthLoginFlow = deps.runAuthLoginFlow;
    const runAuthLogoutFlow = deps.runAuthLogoutFlow;
    const getOnboardingManager = deps.getOnboardingManager;
    const getLogsTerminalManager = deps.getLogsTerminalManager;

    const disposables = [];

    const handleCatch = (error, prefix) => {
        const msg = error instanceof Error ? error.message : String(error);
        log(`${prefix}: ${msg}`);
        vscode.window.showErrorMessage(`Context Engine Uploader: ${prefix}: ${msg}`);
    };

    const requireDep = (value, name) => {
        if (value === undefined || value === null) {
            throw new Error(`Context Engine Uploader: ${name} is unavailable (extension failed to initialize this component).`);
        }
        return value;
    };

    const resolveEndpointOrThrow = () => {
        const getEffectiveConfigFn = requireDep(getEffectiveConfig, 'getEffectiveConfig');
        if (typeof getEffectiveConfigFn !== 'function') {
            throw new Error('getEffectiveConfig is not available (not a function).');
        }
        const cfg = getEffectiveConfigFn();
        const endpoint = (cfg.get('endpoint') || '').trim();
        if (!endpoint) {
            throw new Error('backend endpoint is not configured (contextEngineUploader.endpoint).');
        }
        return endpoint;
    };

    // Start/Stop/Restart commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.start', () => {
        try {
            requireDep(runSequence, 'runSequence')('auto').catch(error => handleCatch(error, 'Start failed'));
        } catch (error) {
            handleCatch(error, 'Start failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.stop', () => {
        try {
            requireDep(stopProcesses, 'stopProcesses')().catch(error => handleCatch(error, 'Stop failed'));
        } catch (error) {
            handleCatch(error, 'Stop failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.restart', () => {
        try {
            requireDep(stopProcesses, 'stopProcesses')()
                .then(() => requireDep(runSequence, 'runSequence')('auto'))
                .catch(error => handleCatch(error, 'Restart failed'));
        } catch (error) {
            handleCatch(error, 'Restart failed');
        }
    }));

    // Index commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.indexCodebase', () => {
        vscode.window.showInformationMessage('Context Engine indexing started.');
        const outputChannel = getOutputChannel();
        if (outputChannel) { outputChannel.show(true); }
        try {
            requireDep(runSequence, 'runSequence')('force').catch(error => handleCatch(error, 'Index failed'));
        } catch (error) {
            handleCatch(error, 'Index failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.uploadGitHistory', () => {
        vscode.window.showInformationMessage('Context Engine git history upload (force sync bundle) started.');
        const outputChannel = getOutputChannel();
        if (outputChannel) { outputChannel.show(true); }
        try {
            requireDep(runSequence, 'runSequence')('uploadGitHistory').catch(error => handleCatch(error, 'Git history upload failed'));
        } catch (error) {
            handleCatch(error, 'Git history upload failed');
        }
    }));

    // Config commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeCtxConfig', () => {
        try {
            requireDep(writeCtxConfig, 'writeCtxConfig')().catch(error => handleCatch(error, 'Failed to write CTX config'));
        } catch (error) {
            handleCatch(error, 'Failed to write CTX config');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfig', () => {
        try {
            requireDep(writeMcpConfig, 'writeMcpConfig')().catch(error => handleCatch(error, 'Failed to write MCP config'));
        } catch (error) {
            handleCatch(error, 'Failed to write MCP config');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigSelect', async () => {
        try {
            const getEffectiveConfigFn = requireDep(getEffectiveConfig, 'getEffectiveConfig');
            if (typeof getEffectiveConfigFn !== 'function') {
                throw new Error('getEffectiveConfig is not available (not a function).');
            }
            const cfg = getEffectiveConfigFn();
            const claudeEnabled = !!cfg.get('mcpClaudeEnabled', true);
            const windsurfEnabled = !!cfg.get('mcpWindsurfEnabled', false);
            const augmentEnabled = !!cfg.get('mcpAugmentEnabled', false);
            const antigravityEnabled = !!cfg.get('mcpAntigravityEnabled', false);

            const items = [
                {
                    label: 'All enabled targets',
                    description: 'Writes MCP config for all enabled clients',
                    id: 'all',
                },
                {
                    label: 'Claude Code (.mcp.json)',
                    description: claudeEnabled ? 'Enabled' : 'Disabled in settings',
                    id: 'claude',
                },
                {
                    label: 'Windsurf (mcp_config.json)',
                    description: windsurfEnabled ? 'Enabled' : 'Disabled in settings',
                    id: 'windsurf',
                },
                {
                    label: 'Augment Code (~/.augment/settings.json)',
                    description: augmentEnabled ? 'Enabled' : 'Disabled in settings',
                    id: 'augment',
                },
                {
                    label: 'Antigravity (~/.gemini/antigravity/mcp_config.json)',
                    description: antigravityEnabled ? 'Enabled' : 'Disabled in settings',
                    id: 'antigravity',
                },
            ];

            const picked = await vscode.window.showQuickPick(items, { placeHolder: 'Select which MCP config to write' });
            if (!picked) {
                return;
            }

            if (picked.id === 'all') {
                await requireDep(writeMcpConfig, 'writeMcpConfig')();
            } else if (picked.id === 'claude') {
                await requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['claude'] });
            } else if (picked.id === 'windsurf') {
                await requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['windsurf'] });
            } else if (picked.id === 'augment') {
                await requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['augment'] });
            } else if (picked.id === 'antigravity') {
                await requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['antigravity'] });
            }
        } catch (error) {
            handleCatch(error, 'MCP config select failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigClaude', () => {
        try {
            requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['claude'] }).catch(error => handleCatch(error, 'Failed to write Claude MCP config'));
        } catch (error) {
            handleCatch(error, 'Failed to write Claude MCP config');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigWindsurf', () => {
        try {
            requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['windsurf'] }).catch(error => handleCatch(error, 'Failed to write Windsurf MCP config'));
        } catch (error) {
            handleCatch(error, 'Failed to write Windsurf MCP config');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigAugment', () => {
        try {
            requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['augment'] }).catch(error => handleCatch(error, 'Failed to write Augment MCP config'));
        } catch (error) {
            handleCatch(error, 'Failed to write Augment MCP config');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigAntigravity', () => {
        try {
            requireDep(writeMcpConfig, 'writeMcpConfig')({ targets: ['antigravity'] }).catch(error => handleCatch(error, 'Failed to write Antigravity MCP config'));
        } catch (error) {
            handleCatch(error, 'Failed to write Antigravity MCP config');
        }
    }));

    // Onboarding/Stack commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.cloneAndStartStack', async () => {
        try {
            const onboardingManager = requireDep(getOnboardingManager, 'getOnboardingManager')();
            if (!onboardingManager || typeof onboardingManager.cloneAndStartStack !== 'function') {
                throw new Error('Context Engine onboarding is unavailable in this session.');
            }
            await onboardingManager.cloneAndStartStack();
        } catch (error) {
            handleCatch(error, 'Clone and start stack failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.startSavedStack', async () => {
        try {
            const onboardingManager = requireDep(getOnboardingManager, 'getOnboardingManager')();
            if (!onboardingManager || typeof onboardingManager.startSavedStack !== 'function') {
                throw new Error('Context Engine onboarding is unavailable in this session.');
            }
            await onboardingManager.startSavedStack();
        } catch (error) {
            handleCatch(error, 'Start saved stack failed');
        }
    }));

    // Logs commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.showUploadServiceLogs', () => {
        try {
            const outputChannel = getOutputChannel();
            if (outputChannel) {
                outputChannel.show(true);
            } else {
                throw new Error('output channel is unavailable');
            }
        } catch (e) {
            handleCatch(e, 'Show logs failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.tailUploadServiceLogs', () => {
        try {
            const logsTerminalManager = requireDep(getLogsTerminalManager, 'getLogsTerminalManager')();
            if (logsTerminalManager && typeof logsTerminalManager.open === 'function') {
                logsTerminalManager.open();
            } else {
                throw new Error('log tailing is unavailable (extension failed to initialize logs terminal manager)');
            }
        } catch (e) {
            handleCatch(e, 'Tail logs failed');
        }
    }));

    // Bridge commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.startMcpHttpBridge', () => {
        try {
            requireDep(startHttpBridgeProcess, 'startHttpBridgeProcess')().catch(error => handleCatch(error, 'HTTP MCP bridge start failed'));
        } catch (error) {
            handleCatch(error, 'HTTP MCP bridge start failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.stopMcpHttpBridge', () => {
        try {
            requireDep(stopHttpBridgeProcess, 'stopHttpBridgeProcess')().catch(error => handleCatch(error, 'HTTP MCP bridge stop failed'));
        } catch (error) {
            handleCatch(error, 'HTTP MCP bridge stop failed');
        }
    }));

    // Auth commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.authLogin', async () => {
        try {
            const endpoint = resolveEndpointOrThrow();
            await requireDep(runAuthLoginFlow, 'runAuthLoginFlow')(endpoint, requireDep(buildAuthDeps, 'buildAuthDeps')());
        } catch (error) {
            handleCatch(error, 'Auth login failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.authLogout', async () => {
        try {
            const endpoint = resolveEndpointOrThrow();
            await requireDep(runAuthLogoutFlow, 'runAuthLogoutFlow')(endpoint, requireDep(buildAuthDeps, 'buildAuthDeps')());
        } catch (error) {
            handleCatch(error, 'Auth logout failed');
        }
    }));

    return disposables;
}

module.exports = {
    registerExtensionCommands,
};
