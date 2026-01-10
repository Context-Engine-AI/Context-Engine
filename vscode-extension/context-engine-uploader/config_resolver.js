/**
 * Configuration and path resolution for Context Engine extension.
 * Consolidates resolveOptions, workspace path logic, and status bar tooltip updates.
 * This module replaces workspace_paths.js and reduces logic in extension.js.
 */
function createConfigResolver(deps) {
    const vscode = deps.vscode;
    const path = deps.path;
    const fs = deps.fs;
    const log = deps.log;

    const getEffectiveConfig = deps.getEffectiveConfig;
    const getPythonOverridePath = deps.getPythonOverridePath;
    const getExtensionRoot = deps.getExtensionRoot;
    const getStatusBarItem = deps.getStatusBarItem;
    const DEFAULT_CONTAINER_ROOT = deps.DEFAULT_CONTAINER_ROOT || '/work';
    let _hasLoggedResolveInfo = false; // Only log script path and host mapping once per session

    let lastAutoDetectLogKey = '';
    let lastResolvedTargetLogKey = '';

    function getWorkspaceFolderPath() {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || !folders.length) {
            return undefined;
        }
        return folders[0].uri.fsPath;
    }

    function looksLikeRepoRoot(dirPath) {
        try {
            const codebaseStatePath = path.join(dirPath, '.codebase', 'state.json');
            const gitDir = path.join(dirPath, '.git');
            if (fs.existsSync(codebaseStatePath) || fs.existsSync(gitDir)) {
                return true;
            }
        } catch (error) {
            log(`Repo root detection failed for ${dirPath}: ${error instanceof Error ? error.message : String(error)}`);
        }
        return false;
    }

    function detectDefaultTargetPath(workspaceFolderPath) {
        try {
            const resolved = path.resolve(workspaceFolderPath);
            if (!fs.existsSync(resolved)) {
                return workspaceFolderPath;
            }
            const rootLooksLikeRepo = looksLikeRepoRoot(resolved);
            let entries;
            try {
                entries = fs.readdirSync(resolved);
            } catch (error) {
                log(`Auto targetPath discovery failed to read workspace folder: ${error instanceof Error ? error.message : String(error)}`);
                return resolved;
            }
            const candidates = [];
            for (const name of entries) {
                const fullPath = path.join(resolved, name);
                let stats;
                try {
                    stats = fs.statSync(fullPath);
                } catch (_) {
                    continue;
                }
                if (!stats.isDirectory()) {
                    continue;
                }
                if (looksLikeRepoRoot(fullPath)) {
                    candidates.push(path.resolve(fullPath));
                }
            }
            if (candidates.length === 1) {
                const detected = candidates[0];
                const key = `${resolved}::${detected}`;
                if (key !== lastAutoDetectLogKey) {
                    lastAutoDetectLogKey = key;
                    log(`Target path auto-detected as ${detected} (under workspace folder).`);
                }
                return detected;
            }
            if (rootLooksLikeRepo) {
                if (candidates.length > 1) {
                    const key = `${resolved}::multiple`;
                    if (key !== lastAutoDetectLogKey) {
                        lastAutoDetectLogKey = key;
                        log('Auto targetPath discovery found multiple candidate repos under workspace; using workspace folder instead.');
                    }
                }
                return resolved;
            }
            return resolved;
        } catch (error) {
            log(`Auto targetPath discovery failed: ${error instanceof Error ? error.message : String(error)}`);
            return workspaceFolderPath;
        }
    }

    function resolveTargetPathFromConfig(config) {
        let inspected;
        try {
            if (typeof config.inspect === 'function') {
                inspected = config.inspect('targetPath');
            }
        } catch (error) {
            inspected = undefined;
        }
        let targetPath = (config.get('targetPath') || '').trim();
        const metadata = inspected || {};
        if (targetPath) {
            return { path: targetPath, inspected: metadata };
        }
        const folderPath = getWorkspaceFolderPath();
        if (!folderPath) {
            return { path: undefined, inspected: metadata };
        }
        const autoTarget = detectDefaultTargetPath(folderPath);
        return { path: autoTarget, inspected: metadata, inferred: true };
    }

    function getTargetPath(config) {
        const result = resolveTargetPathFromConfig(config);
        let targetPath = result.path;
        const inspected = result.inspected;
        if (inspected && targetPath) {
            let sourceLabel = 'default';
            if (inspected.workspaceFolderValue !== undefined) {
                sourceLabel = 'workspaceFolder';
            } else if (inspected.workspaceValue !== undefined) {
                sourceLabel = 'workspace';
            } else if (inspected.globalValue !== undefined) {
                sourceLabel = 'user';
            }
            const key = `${sourceLabel}::${targetPath}`;
            if (key !== lastResolvedTargetLogKey) {
                lastResolvedTargetLogKey = key;
                log(`Target path resolved to ${targetPath} (source: ${sourceLabel} settings)`);
            }
        }
        if (targetPath) {
            const folderPath = getWorkspaceFolderPath();
            if (folderPath && !path.isAbsolute(targetPath)) {
                targetPath = path.resolve(folderPath, targetPath);
            }
            updateStatusBarTooltip(targetPath);
            return targetPath;
        }
        vscode.window.showErrorMessage('Context Engine Uploader: open a folder or set contextEngineUploader.targetPath.');
        updateStatusBarTooltip();
        return undefined;
    }

    function saveTargetPath(config, targetPath) {
        const hasWorkspace = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length;
        const target = hasWorkspace ? vscode.ConfigurationTarget.Workspace : vscode.ConfigurationTarget.Global;
        config.update('targetPath', targetPath, target).catch(error => {
            log(`Target path save failed: ${error instanceof Error ? error.message : String(error)}`);
        });
    }

    function updateStatusBarTooltip(targetPath) {
        const statusBarItem = getStatusBarItem();
        if (!statusBarItem) {
            return;
        }
        if (targetPath) {
            statusBarItem.tooltip = `Index Codebase (${targetPath})`;
        } else {
            statusBarItem.tooltip = 'Index Codebase';
        }
    }

    function ensureTargetPathConfigured() {
        const config = getEffectiveConfig();
        const current = (config.get('targetPath') || '').trim();
        if (current) {
            updateStatusBarTooltip(current);
            return;
        }
        const folderPath = getWorkspaceFolderPath();
        if (!folderPath) {
            updateStatusBarTooltip();
            return;
        }
        const autoTarget = detectDefaultTargetPath(folderPath);
        updateStatusBarTooltip(autoTarget);
    }

    function resolveOptions() {
        const config = getEffectiveConfig();
        const pythonOverridePath = getPythonOverridePath();
        const extensionRoot = getExtensionRoot();

        const configuredPython = (config.get('pythonPath') || '').trim();
        let pythonPath = configuredPython || 'python3';
        let pythonPathSource = configuredPython ? 'configured' : 'default';
        if (pythonOverridePath && fs.existsSync(pythonOverridePath)) {
            pythonPath = pythonOverridePath;
            pythonPathSource = 'override';
        }
        const endpoint = (config.get('endpoint') || '').trim();
        const targetPath = getTargetPath(config);
        const interval = config.get('intervalSeconds') || 5;
        const extraForceArgs = config.get('extraForceArgs') || [];
        const extraWatchArgs = config.get('extraWatchArgs') || [];
        const hostRootOverride = (config.get('hostRoot') || '').trim();
        const containerRoot = (config.get('containerRoot') || DEFAULT_CONTAINER_ROOT).trim() || DEFAULT_CONTAINER_ROOT;
        const startWatchAfterForce = config.get('startWatchAfterForce') ?? true;
        const configuredScriptDir = (config.get('scriptWorkingDirectory') || '').trim();

        const candidates = [];
        if (configuredScriptDir) {
            candidates.push(configuredScriptDir);
        }
        // Prefer packaged script; also try workspace ./scripts fallback for dev
        candidates.push(extensionRoot);
        const wsRoot = getWorkspaceFolderPath();
        if (wsRoot) {
            candidates.push(path.join(wsRoot, 'scripts'));
        }
        candidates.push(path.join(extensionRoot, '..', 'out'));

        let workingDirectory;
        let scriptPath;
        for (const candidate of candidates) {
            if (!candidate) {
                continue;
            }
            const resolved = path.resolve(candidate);
            const testPath = path.join(resolved, 'standalone_upload_client.py');
            if (fs.existsSync(testPath)) {
                workingDirectory = resolved;
                scriptPath = testPath;
                break;
            }
        }

        if (!workingDirectory || !scriptPath) {
            vscode.window.showErrorMessage('Context Engine Uploader: extension path unavailable.');
            return undefined;
        }

        const scriptSource = workingDirectory === extensionRoot ? 'packaged' : (path.basename(workingDirectory) === 'out' ? 'staged out' : 'custom');
        if (!endpoint) {
            vscode.window.showErrorMessage('Context Engine Uploader: set contextEngineUploader.endpoint.');
            return undefined;
        }
        if (!targetPath) {
            return undefined;
        }

        const resolvedTarget = path.resolve(targetPath);
        let derivedHostRoot = path.dirname(resolvedTarget);
        if (!derivedHostRoot || derivedHostRoot === resolvedTarget) {
            derivedHostRoot = resolvedTarget;
        }
        const hostRoot = hostRootOverride || derivedHostRoot;
        // Only log script path and host mapping on first call to reduce noise
        if (!_hasLoggedResolveInfo) {
            log(`Using ${scriptSource} standalone_upload_client.py at ${scriptPath}`);
            log(`Uploader path mapping hostRoot=${hostRoot || 'n/a'} -> containerRoot=${containerRoot}`);
            _hasLoggedResolveInfo = true;
        }

        return {
            pythonPath,
            pythonPathSource,
            workingDirectory,
            scriptPath,
            targetPath,
            endpoint,
            interval,
            extraForceArgs,
            extraWatchArgs,
            hostRoot,
            containerRoot,
            startWatchAfterForce
        };
    }

    function resolveBridgeWorkspacePath() {
        try {
            const settings = getEffectiveConfig();
            const target = getTargetPath(settings);
            if (target) {
                return path.resolve(target);
            }
        } catch (error) {
            log(`Context Engine Uploader: failed to resolve bridge workspace path via getTargetPath: ${error instanceof Error ? error.message : String(error)}`);
        }
        const fallbackFolder = getWorkspaceFolderPath();
        if (!fallbackFolder) {
            return undefined;
        }
        try {
            const autoTarget = detectDefaultTargetPath(fallbackFolder);
            return autoTarget ? path.resolve(autoTarget) : path.resolve(fallbackFolder);
        } catch (error) {
            log(`Context Engine Uploader: failed fallback bridge workspace path detection: ${error instanceof Error ? error.message : String(error)}`);
            return undefined;
        }
    }

    function needsForceSync(targetPath) {
        try {
            const cachePath = path.join(targetPath, '.context-engine', 'file_cache.json');
            if (!fs.existsSync(cachePath)) {
                return true;
            }
            const stats = fs.statSync(cachePath);
            return stats.size === 0;
        } catch (error) {
            log(`Force detection failed: ${error instanceof Error ? error.message : String(error)}`);
            return true;
        }
    }

    return {
        getWorkspaceFolderPath,
        getTargetPath,
        detectDefaultTargetPath,
        resolveTargetPathFromConfig,
        ensureTargetPathConfigured,
        updateStatusBarTooltip,
        resolveOptions,
        resolveBridgeWorkspacePath,
        needsForceSync,
        saveTargetPath,
        looksLikeRepoRoot,
    };
}

module.exports = {
    createConfigResolver,
};
