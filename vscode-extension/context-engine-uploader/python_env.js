/**
 * Python environment management for Context Engine extension.
 * Handles dependency checking, venv creation, and Python interpreter detection.
 */
function createPythonEnvManager(deps) {
    const vscode = deps.vscode;
    const spawn = deps.spawn;
    const path = deps.path;
    const fs = deps.fs;
    const log = deps.log;

    // State for deduplication and caching
    const _checkCache = new Map();
    const _checkInFlight = new Map();
    const CACHE_TTL_MS = 30_000;

    // Helper to spawn processes asynchronously with Promise wrapper
    function execAsync(command, args, options = {}) {
        return new Promise((resolve) => {
            // Diagnostic check for spawn injection
            if (typeof spawn !== 'function') {
                resolve({ code: -1, stdout: '', stderr: `createPythonEnvManager: spawn is ${typeof spawn}` });
                return;
            }

            const child = spawn(command, args, {
                ...options,
                env: options.env || process.env
            });

            let stdout = '';
            let stderr = '';

            if (child.stdout) {
                child.stdout.on('data', (data) => {
                    const str = data.toString();
                    stdout += str;
                    if (options.onStdout) options.onStdout(str);
                });
            }

            if (child.stderr) {
                child.stderr.on('data', (data) => {
                    const str = data.toString();
                    stderr += str;
                    if (options.onStderr) options.onStderr(str);
                });
            }

            let finished = false;

            child.on('error', (err) => {
                if (!finished) {
                    finished = true;
                    resolve({ code: -1, stdout, stderr: stderr || err.message });
                }
            });

            child.on('close', (code) => {
                if (!finished) {
                    finished = true;
                    resolve({ code: code === null ? -1 : code, stdout, stderr });
                }
            });

            // Handle cancellation if token provided
            if (options.token) {
                options.token.onCancellationRequested(() => {
                    if (!finished) {
                        finished = true;
                        try { child.kill(); } catch (_) { }
                        resolve({ code: -1, stdout, stderr: 'Cancelled' });
                    }
                });
            }

            // Safety timeout
            if (options.timeout) {
                setTimeout(() => {
                    try { child.kill(); } catch (_) { }
                    // Don't resolve here, let 'close' handle it
                }, options.timeout);
            }
        });
    }

    function getExtensionRoot() {
        if (deps.extensionRoot) return deps.extensionRoot;
        try {
            return vscode.extensions.getExtension('context-engine.context-engine-uploader').extensionPath;
        } catch (_) {
            return __dirname;
        }
    }

    function getPythonOverridePath() {
        return typeof deps.getPythonOverridePath === 'function' ? deps.getPythonOverridePath() : undefined;
    }

    function setPythonOverridePath(p) {
        if (typeof deps.setPythonOverridePath === 'function') {
            deps.setPythonOverridePath(p);
        }
    }

    const REQUIRED_PYTHON_MODULES = ['requests', 'urllib3', 'charset_normalizer', 'watchdog'];

    function venvRootDir() {
        // Prefer workspace storage; fallback to extension storage
        try {
            const ws = deps.getWorkspaceFolderPath();
            const globalStorage = deps.getGlobalStoragePath() || path.join(getExtensionRoot(), '.storage');
            const base = ws && fs.existsSync(ws) ? path.join(ws, '.vscode', '.context-engine-uploader')
                : globalStorage;
            if (!fs.existsSync(base)) fs.mkdirSync(base, { recursive: true });
            return base;
        } catch (e) {
            return getExtensionRoot();
        }
    }

    function privateVenvPath() {
        return path.join(venvRootDir(), 'py-venv');
    }

    function resolvePrivateVenvPython() {
        const venvPath = privateVenvPath();
        const bin = process.platform === 'win32' ? path.join(venvPath, 'Scripts', 'python.exe') : path.join(venvPath, 'bin', 'python');
        return fs.existsSync(bin) ? bin : undefined;
    }

    async function detectSystemPython() {
        // Try configured pythonPath, then common names
        const candidates = [];
        try {
            const cfg = vscode.workspace.getConfiguration('contextEngineUploader');
            const configured = (cfg.get('pythonPath') || '').trim();
            if (configured) candidates.push(configured);
        } catch { }
        if (process.platform === 'win32') {
            candidates.push('py', 'python3', 'python');
        } else {
            candidates.push('python3', 'python');
            // Add common Homebrew path on Apple Silicon
            candidates.push('/opt/homebrew/bin/python3');
        }

        for (const cmd of candidates) {
            try {
                // Version check: major >= 3 and print executable
                const res = await execAsync(cmd, ['-c', 'import sys; print(f"{sys.version_info[0]}|{sys.executable}")'], { timeout: 3000 });
                if (res.code === 0) {
                    const parts = res.stdout.trim().split('|');
                    if (parts.length === 2) {
                        const major = parseInt(parts[0], 10);
                        const executable = parts[1].trim();
                        if (major >= 3 && executable) return executable;
                    }
                }
            } catch (e) {
                // Skip candidate
            }
        }
        return undefined;
    }

    async function checkPythonDeps(pythonPath) {
        // 1. Check Success Cache (deduplicate subsequent calls if recent success)
        const cacheKey = (pythonPath || '').trim();
        const now = Date.now();
        const cached = _checkCache.get(cacheKey);
        if (cached && (now - cached.timestamp < CACHE_TTL_MS)) {
            // Return cached true result without logging
            return true;
        }

        // 2. Request Coalescing (deduplicate concurrent in-flight calls)
        if (_checkInFlight.has(cacheKey)) {
            return _checkInFlight.get(cacheKey);
        }

        const checkPromise = (async () => {
            const missing = [];
            const env = { ...process.env };
            let usedBundled = false;
            try {
                const extensionRoot = getExtensionRoot();
                const libsPath = path.join(extensionRoot, 'python_libs');
                if (fs.existsSync(libsPath)) {
                    const existing = env.PYTHONPATH || '';
                    env.PYTHONPATH = existing ? `${libsPath}${path.delimiter}${existing}` : libsPath;
                    usedBundled = true;
                }
            } catch (error) {
                log(`Failed to configure PYTHONPATH for dependency check: ${error instanceof Error ? error.message : String(error)}`);
            }

            // Only log about bundled libs if we haven't successfully cached it recently
            // or if we are about to fail. We defer logging success until end.

            for (const moduleName of REQUIRED_PYTHON_MODULES) {
                try {
                    const check = await execAsync(pythonPath, ['-c', `import ${moduleName}`], { env, timeout: 5000 });
                    if (check.code !== 0) {
                        missing.push(moduleName);
                    }
                } catch (error) {
                    log(`Dependency check failed for ${moduleName} on ${pythonPath}: ${error instanceof Error ? error.message : String(error)}`);
                    return false;
                }
            }

            if (missing.length) {
                if (usedBundled) {
                    log(`Using bundled python_libs for dependency check.`);
                }
                log(`Missing Python modules for ${pythonPath}: ${missing.join(', ')}`);
                return false;
            }

            // Success! Cache it.
            _checkCache.set(cacheKey, { timestamp: Date.now() });
            return true;
        })();

        _checkInFlight.set(cacheKey, checkPromise);
        try {
            return await checkPromise;
        } finally {
            _checkInFlight.delete(cacheKey);
        }
    }

    async function ensurePrivateVenv() {
        try {
            const python = resolvePrivateVenvPython();
            if (python) {
                log('Private venv already exists.');
                return true;
            }
            const venvPath = privateVenvPath();
            const basePy = await detectSystemPython();
            if (!basePy) {
                vscode.window.showErrorMessage('Context Engine Uploader: no Python 3 interpreter found to bootstrap venv.');
                return false;
            }

            // Verify venv module presence
            try {
                const venvCheck = await execAsync(basePy, ['-c', 'import venv'], { timeout: 5000 });
                if (venvCheck.code !== 0) {
                    log(`Python "venv" module missing in ${basePy}: ${venvCheck.stderr}`);
                    vscode.window.showErrorMessage(`Context Engine Uploader: Python "venv" module is missing in ${basePy}.`);
                    return false;
                }
            } catch (e) {
                log(`Failed to check for venv module: ${e.message}`);
                return false;
            }

            log(`Creating private venv at ${venvPath} using ${basePy}`);
            const res = await execAsync(basePy, ['-m', 'venv', venvPath], { timeout: 30000 });
            if (res.code !== 0) {
                log(`venv creation failed: ${res.stderr || res.stdout}`);
                vscode.window.showErrorMessage('Context Engine Uploader: failed to create private venv.');
                return false;
            }
            return true;
        } catch (e) {
            log(`ensurePrivateVenv error: ${e && e.message ? e.message : String(e)}`);
            return false;
        }
    }

    async function installDepsInto(pythonBin) {
        return vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Context Engine Uploader: Installing Python dependencies...",
            cancellable: true
        }, async (progress, token) => {
            try {
                log(`Installing Python deps into private venv via ${pythonBin}`);
                const args = ['-m', 'pip', 'install', ...REQUIRED_PYTHON_MODULES];

                const res = await execAsync(pythonBin, args, {
                    timeout: 60000,
                    token,
                    onStdout: (data) => {
                        progress.report({ message: data.split('\n').pop() });
                    },
                    onStderr: (data) => {
                        log(`pip install stderr: ${data}`);
                    }
                });

                if (res.code !== 0) {
                    log(`pip install failed: ${res.stderr || res.stdout}`);
                    vscode.window.showErrorMessage('Context Engine Uploader: pip install failed. See Output for details.');
                    return false;
                }
                return true;
            } catch (e) {
                const msg = e && e.message ? e.message : String(e);
                log(`installDepsInto error: ${msg}`);
                vscode.window.showErrorMessage(`Context Engine Uploader: ${msg}`);
                return false;
            }
        });
    }

    async function ensurePythonDependencies(pythonPath) {
        // Probe current interpreter with bundled python_libs first
        let ok = await checkPythonDeps(pythonPath);
        if (ok) {
            return true;
        }

        // If that fails, try to auto-detect a better system Python before falling back to a venv
        const autoPython = await detectSystemPython();
        if (autoPython && autoPython !== pythonPath) {
            log(`Falling back to auto-detected Python interpreter: ${autoPython}`);
            ok = await checkPythonDeps(autoPython);
            if (ok) {
                setPythonOverridePath(autoPython);
                return true;
            }
        }

        // As a last resort, offer to create a private venv and install deps via pip
        const choice = await vscode.window.showErrorMessage(
            'Context Engine Uploader: missing Python modules. Create isolated environment and auto-install?',
            'Auto-install to private venv',
            'Cancel'
        );
        if (choice !== 'Auto-install to private venv') {
            return false;
        }
        const created = await ensurePrivateVenv();
        if (!created) return false;
        const venvPython = resolvePrivateVenvPython();
        if (!venvPython) {
            vscode.window.showErrorMessage('Context Engine Uploader: failed to locate private venv python.');
            return false;
        }
        const installed = await installDepsInto(venvPython);
        if (!installed) return false;
        setPythonOverridePath(venvPython);
        log(`Using private venv interpreter: ${getPythonOverridePath()}`);
        return await checkPythonDeps(venvPython);
    }

    return {
        resolvePrivateVenvPython,
        detectSystemPython,
        checkPythonDeps,
        ensurePythonDependencies,
    };
}

module.exports = {
    createPythonEnvManager,
};
