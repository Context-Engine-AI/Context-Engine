/**
 * Command registration for Context Engine extension.
 * Consolidates all vscode.commands.registerCommand calls.
 */
function registerExtensionCommands(deps) {
    if (!deps || typeof deps !== 'object') {
        throw new Error('registerExtensionCommands: deps object is required');
    }

    const requiredDeps = [
        'vscode', 'log', 'getEffectiveConfig', 'getOutputChannel', 'runSequence',
        'stopProcesses', 'writeMcpConfig', 'writeCtxConfig', 'startHttpBridgeProcess',
        'stopHttpBridgeProcess', 'buildAuthDeps', 'runAuthLoginFlow', 'runAuthLogoutFlow',
        'getOnboardingManager', 'getLogsTerminalManager'
    ];

    for (const key of requiredDeps) {
        if (deps[key] === undefined || deps[key] === null) {
            throw new Error(`registerExtensionCommands: dependency "${key}" is missing or null`);
        }
    }

    const vscode = deps.vscode;
    const log = deps.log;

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

    const resolveEndpointOrThrow = () => {
        const cfg = getEffectiveConfig();
        const endpoint = (cfg.get('endpoint') || '').trim();
        if (!endpoint) {
            throw new Error('backend endpoint is not configured (contextEngineUploader.endpoint).');
        }
        return endpoint;
    };

    // Start/Stop/Restart commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.start', () => {
        runSequence('auto').catch(error => handleCatch(error, 'Start failed'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.stop', () => {
        stopProcesses().catch(error => handleCatch(error, 'Stop failed'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.restart', () => {
        stopProcesses()
            .then(() => runSequence('auto'))
            .catch(error => handleCatch(error, 'Restart failed'));
    }));

    // Index commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.indexCodebase', () => {
        vscode.window.showInformationMessage('Context Engine indexing started.');
        const outputChannel = getOutputChannel();
        if (outputChannel) { outputChannel.show(true); }
        runSequence('force').catch(error => handleCatch(error, 'Index failed'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.uploadGitHistory', () => {
        vscode.window.showInformationMessage('Context Engine git history upload (force sync bundle) started.');
        const outputChannel = getOutputChannel();
        if (outputChannel) { outputChannel.show(true); }
        runSequence('uploadGitHistory').catch(error => handleCatch(error, 'Git history upload failed'));
    }));

    // Config commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeCtxConfig', () => {
        writeCtxConfig().catch(error => handleCatch(error, 'Failed to write CTX config'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfig', () => {
        writeMcpConfig().catch(error => handleCatch(error, 'Failed to write MCP config'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigSelect', async () => {
        try {
            const cfg = getEffectiveConfig();
            const claudeEnabled = !!cfg.get('mcpClaudeEnabled', true);
            const windsurfEnabled = !!cfg.get('mcpWindsurfEnabled', false);
            const augmentEnabled = !!cfg.get('mcpAugmentEnabled', false);

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
            ];

            const picked = await vscode.window.showQuickPick(items, { placeHolder: 'Select which MCP config to write' });
            if (!picked) {
                return;
            }

            if (picked.id === 'all') {
                await writeMcpConfig();
            } else if (picked.id === 'claude') {
                await writeMcpConfig({ targets: ['claude'] });
            } else if (picked.id === 'windsurf') {
                await writeMcpConfig({ targets: ['windsurf'] });
            } else if (picked.id === 'augment') {
                await writeMcpConfig({ targets: ['augment'] });
            }
        } catch (error) {
            handleCatch(error, 'MCP config select failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigClaude', () => {
        writeMcpConfig({ targets: ['claude'] }).catch(error => handleCatch(error, 'Failed to write Claude MCP config'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigWindsurf', () => {
        writeMcpConfig({ targets: ['windsurf'] }).catch(error => handleCatch(error, 'Failed to write Windsurf MCP config'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigAugment', () => {
        writeMcpConfig({ targets: ['augment'] }).catch(error => handleCatch(error, 'Failed to write Augment MCP config'));
    }));

    // Onboarding/Stack commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.cloneAndStartStack', async () => {
        try {
            const onboardingManager = getOnboardingManager();
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
            const onboardingManager = getOnboardingManager();
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
            const logsTerminalManager = getLogsTerminalManager();
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
        startHttpBridgeProcess().catch(error => handleCatch(error, 'HTTP MCP bridge start failed'));
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.stopMcpHttpBridge', () => {
        stopHttpBridgeProcess().catch(error => handleCatch(error, 'HTTP MCP bridge stop failed'));
    }));

    // Auth commands
    disposables.push(vscode.commands.registerCommand('contextEngineUploader.authLogin', async () => {
        try {
            const endpoint = resolveEndpointOrThrow();
            await runAuthLoginFlow(endpoint, buildAuthDeps());
        } catch (error) {
            handleCatch(error, 'Auth login failed');
        }
    }));

    disposables.push(vscode.commands.registerCommand('contextEngineUploader.authLogout', async () => {
        try {
            const endpoint = resolveEndpointOrThrow();
            await runAuthLogoutFlow(endpoint, buildAuthDeps());
        } catch (error) {
            handleCatch(error, 'Auth logout failed');
        }
    }));

    return disposables;
}

module.exports = {
    registerExtensionCommands,
};
