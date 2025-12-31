/**
 * Process and watcher management for Context Engine extension.
 * Handles spawning/terminating Python processes and file system watchers.
 */
function createProcessManager(deps) {
    const vscode = deps.vscode;
    const spawn = deps.spawn;
    const fs = deps.fs;
    const path = deps.path;
    const log = deps.log;

    const getEffectiveConfig = deps.getEffectiveConfig;
    const getWorkspaceFolderPath = deps.getWorkspaceFolderPath;
    const getOutputChannel = deps.getOutputChannel;
    const setStatusBarState = deps.setStatusBarState;
    const getStatusMode = deps.getStatusMode;
    const getExtensionRoot = deps.getExtensionRoot;

    let forceProcess;
    let watchProcess;
    let workspaceWatcher;
    let watchedTargetPath;
    let indexedWatchDisposables = [];
    let _hasLoggedDevRemoteMode = false; // Only log devRemoteMode once per session
    let _hasLoggedPythonPath = false; // Only log PYTHONPATH once per session

    function buildChildEnv(options) {
        const env = {
            ...process.env,
            WORKSPACE_PATH: options.targetPath,
            WATCH_ROOT: options.targetPath
        };
        try {
            const settings = getEffectiveConfig();
            const devRemoteMode = settings.get('devRemoteMode', false);
            if (devRemoteMode && !_hasLoggedDevRemoteMode) {
                // Enable dev-remote upload mode for the standalone upload client.
                // This causes standalone_upload_client.py to ignore any 'dev-workspace'
                // directories when scanning for files to upload.
                env.REMOTE_UPLOAD_MODE = 'development';
                env.DEV_REMOTE_MODE = '1';
                log('Context Engine Uploader: devRemoteMode enabled (REMOTE_UPLOAD_MODE=development, DEV_REMOTE_MODE=1).');
                _hasLoggedDevRemoteMode = true;
            } else if (devRemoteMode) {
                env.REMOTE_UPLOAD_MODE = 'development';
                env.DEV_REMOTE_MODE = '1';
            }
            const gitMaxCommits = settings.get('gitMaxCommits');
            if (typeof gitMaxCommits === 'number' && !Number.isNaN(gitMaxCommits)) {
                env.REMOTE_UPLOAD_GIT_MAX_COMMITS = String(gitMaxCommits);
            }
            const gitSinceRaw = settings.get('gitSince');
            const gitSince = typeof gitSinceRaw === 'string' ? gitSinceRaw.trim() : '';
            if (gitSince) {
                env.REMOTE_UPLOAD_GIT_SINCE = gitSince;
            }
        } catch (error) {
            log(`Failed to read devRemoteMode setting: ${error instanceof Error ? error.message : String(error)}`);
        }
        if (options.hostRoot) {
            env.HOST_ROOT = options.hostRoot;
        }
        if (options.containerRoot) {
            env.CONTAINER_ROOT = options.containerRoot;
        }
        try {
            const libsPath = path.join(options.workingDirectory, 'python_libs');
            if (fs.existsSync(libsPath)) {
                const existing = env.PYTHONPATH || '';
                env.PYTHONPATH = existing ? `${libsPath}${path.delimiter}${existing}` : libsPath;
                if (!_hasLoggedPythonPath) {
                    log(`Detected bundled python_libs at ${libsPath}; setting PYTHONPATH for child process.`);
                    _hasLoggedPythonPath = true;
                }
            }
        } catch (error) {
            log(`Failed to configure PYTHONPATH for bundled deps: ${error instanceof Error ? error.message : String(error)}`);
        }
        return env;
    }

    function buildArgs(options, mode) {
        const args = ['-u', options.scriptPath, '--path', options.targetPath, '--endpoint', options.endpoint];
        if (mode === 'force') {
            args.push('--force');
            if (options.extraForceArgs && options.extraForceArgs.length) {
                args.push(...options.extraForceArgs);
            }
        } else {
            args.push('--watch', '--interval', String(options.interval));
            if (options.extraWatchArgs && options.extraWatchArgs.length) {
                args.push(...options.extraWatchArgs);
            }
        }
        return args;
    }

    function attachOutput(child, label) {
        const outputChannel = getOutputChannel();
        if (!outputChannel) {
            return;
        }
        if (child.stdout) {
            child.stdout.on('data', data => {
                const chunk = data.toString();
                outputChannel.append(`[${label}] ${chunk}`);
            });
        }
        if (child.stderr) {
            child.stderr.on('data', data => {
                const chunk = data.toString();
                outputChannel.append(`[${label} err] ${chunk}`);
            });
        }
    }

    function terminateProcess(proc, label, afterStop) {
        if (!proc) {
            return Promise.resolve();
        }
        return new Promise(resolve => {
            let finished = false;
            let termTimer;
            let killTimer;
            const clearTimers = () => {
                if (termTimer) clearTimeout(termTimer);
                if (killTimer) clearTimeout(killTimer);
            };
            const finalize = (reason) => {
                if (finished) return;
                finished = true;
                clearTimers();
                if (typeof afterStop === 'function') {
                    afterStop();
                }
                if (proc === forceProcess) {
                    forceProcess = undefined;
                }
                if (proc === watchProcess) {
                    watchProcess = undefined;
                }
                log(`${label} process stopped${reason ? ` (${reason})` : ''}.`);
                resolve();
            };

            // Resolve only after the child actually exits (or after forced kill path)
            const onExit = (code, signal) => {
                finalize(`exit code=${code} signal=${signal || ''}`.trim());
            };
            proc.once('exit', onExit);
            proc.once('close', onExit);

            try {
                proc.kill(); // default SIGTERM
            } catch (error) {
                finalize('kill() threw');
                return;
            }

            const waitSigtermMs = 4000;
            const waitSigkillMs = 2000;

            // If process doesn't exit after SIGTERM, escalate to SIGKILL and then force-resolve
            termTimer = setTimeout(() => {
                try {
                    if (proc && !proc.killed) {
                        proc.kill('SIGKILL');
                        log(`${label} process did not exit after ${waitSigtermMs}ms; sent SIGKILL.`);
                    }
                } catch (_) {
                    // ignore
                }
                killTimer = setTimeout(() => {
                    finalize(`forced after ${waitSigtermMs + waitSigkillMs}ms`);
                }, waitSigkillMs);
            }, waitSigtermMs);
        });
    }

    async function stopProcesses() {
        await Promise.all([terminateProcess(forceProcess, 'force'), terminateProcess(watchProcess, 'watch')]);
        if (!forceProcess && !watchProcess && getStatusMode() !== 'indexing') {
            setStatusBarState('idle');
        }
    }

    function disposeIndexedWatcher() {
        try {
            for (const d of indexedWatchDisposables) {
                try { if (d && typeof d.dispose === 'function') d.dispose(); } catch (_) { }
            }
            indexedWatchDisposables = [];
            if (workspaceWatcher && typeof workspaceWatcher.dispose === 'function') {
                workspaceWatcher.dispose();
            }
            workspaceWatcher = undefined;
            watchedTargetPath = undefined;
        } catch (e) {
            // ignore
        }
    }

    function ensureIndexedWatcher(targetPath) {
        try {
            disposeIndexedWatcher();
            watchedTargetPath = targetPath;
            let pattern;
            if (targetPath && fs.existsSync(targetPath)) {
                pattern = new vscode.RelativePattern(targetPath, '**/*');
            } else {
                const folder = getWorkspaceFolderPath();
                if (folder && fs.existsSync(folder)) {
                    pattern = new vscode.RelativePattern(folder, '**/*');
                } else {
                    pattern = '**/*';
                }
            }
            workspaceWatcher = vscode.workspace.createFileSystemWatcher(pattern, false, false, false);
            const flipToIdle = () => {
                if (getStatusMode() === 'indexed') {
                    setStatusBarState('idle');
                }
            };
            indexedWatchDisposables.push(workspaceWatcher);
            indexedWatchDisposables.push(workspaceWatcher.onDidCreate(flipToIdle));
            indexedWatchDisposables.push(workspaceWatcher.onDidChange(flipToIdle));
            indexedWatchDisposables.push(workspaceWatcher.onDidDelete(flipToIdle));

            // Restrict listener to watched targetPath
            indexedWatchDisposables.push(vscode.workspace.onDidChangeTextDocument((event) => {
                if (watchedTargetPath) {
                    const relativePath = path.relative(watchedTargetPath, event.document.uri.fsPath);
                    // A file is inside the watched directory if the relative path doesn't start with '..' and is not absolute
                    const isInsideWatchedDir = !relativePath.startsWith('..') && !path.isAbsolute(relativePath);
                    if (isInsideWatchedDir) {
                        flipToIdle();
                    }
                }
            }));

            log('Indexed watcher armed; any file change will return status bar to "Index Codebase".');
        } catch (e) {
            log(`Failed to arm indexed watcher: ${e && e.message ? e.message : String(e)}`);
        }
    }

    async function runOnce(options) {
        if (forceProcess) {
            log('Force sync already in progress; terminating existing process.');
            await terminateProcess(forceProcess, 'force');
        }

        return new Promise(resolve => {
            try {
                const args = buildArgs(options, 'force');
                const baseEnv = buildChildEnv(options);
                const childEnv = { ...baseEnv, REMOTE_UPLOAD_GIT_FORCE: '1' };
                const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory, env: childEnv });
                forceProcess = child;
                attachOutput(child, 'force');
                let finished = false;
                const finish = code => {
                    if (finished) {
                        return;
                    }
                    finished = true;
                    log(`Force sync exited with code ${code}`);
                    if (forceProcess === child) {
                        forceProcess = undefined;
                    }
                    resolve(typeof code === 'number' ? code : 1);
                };
                child.on('close', finish);
                child.on('error', error => {
                    const msg = error instanceof Error ? error.message : String(error);
                    log(`Force sync failed: ${msg}`);
                    finish(1);
                });
            } catch (error) {
                const msg = error instanceof Error ? error.message : String(error);
                log(`Failed to spawn force sync process: ${msg}`);
                resolve(1);
            }
        });
    }

    async function startWatch(options) {
        if (watchProcess) {
            log('Watch process already running; terminating existing process.');
            await terminateProcess(watchProcess, 'watch');
        }

        disposeIndexedWatcher();

        try {
            const args = buildArgs(options, 'watch');
            const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory, env: buildChildEnv(options) });
            watchProcess = child;
            attachOutput(child, 'watch');
            const outputChannel = getOutputChannel();
            if (outputChannel) { outputChannel.show(true); }
            setStatusBarState('watch');

            const cleanupWatch = (reason) => {
                log(`Watch process stopped: ${reason}`);
                if (watchProcess === child) {
                    watchProcess = undefined;
                    if (getStatusMode() !== 'indexing') {
                        setStatusBarState('idle');
                    }
                }
            };

            child.on('close', code => {
                cleanupWatch(`exited with code ${code !== undefined ? code : 'unknown'}`);
            });
            child.on('error', error => {
                const msg = error instanceof Error ? error.message : String(error);
                cleanupWatch(`failed with error: ${msg}`);
            });
            vscode.window.showInformationMessage('Context Engine remote watch started.');
        } catch (error) {
            log(`Failed to spawn watch process: ${error.message}`);
        }
    }

    function getState() {
        return {
            forceProcess,
            watchProcess,
            watchedTargetPath,
        };
    }

    function getWatchedTargetPath() {
        return watchedTargetPath;
    }

    function dispose() {
        disposeIndexedWatcher();
        return stopProcesses();
    }

    return {
        buildChildEnv,
        buildArgs,
        attachOutput,
        terminateProcess,
        stopProcesses,
        disposeIndexedWatcher,
        ensureIndexedWatcher,
        runOnce,
        startWatch,
        getState,
        getWatchedTargetPath,
        dispose,
    };
}

module.exports = {
    createProcessManager,
};
