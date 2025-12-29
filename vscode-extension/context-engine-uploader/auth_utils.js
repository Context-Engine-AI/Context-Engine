const process = require('process');

const _skippedAuthCombos = new Set();

function getFetch(deps) {
  if (deps && typeof deps.fetchGlobal === 'function') {
    return deps.fetchGlobal;
  }
  try {
    if (typeof fetch === 'function') {
      return fetch;
    }
  } catch (_) {
  }
  return null;
}

/**
 * Check auth status for the given endpoint using ctxce CLI.
 * Returns { state: 'ok' | 'missing' | 'expired' | 'error', userId?: string }
 */
function checkAuthStatus(endpoint, deps) {
  if (!deps || !deps.spawnSync || !deps.resolveBridgeCliInvocation || !deps.getWorkspaceFolderPath) {
    return { state: 'error' };
  }
  const { spawnSync, resolveBridgeCliInvocation, getWorkspaceFolderPath } = deps;
  const raw = (endpoint || '').trim();
  if (!raw) {
    return { state: 'error' };
  }
  let backendUrl = raw;
  try {
    const u = new URL(raw);
    backendUrl = `${u.protocol}//${u.host}`;
  } catch (_) {
    backendUrl = raw.replace(/\/+$/, '');
  }

  const invocation = resolveBridgeCliInvocation();
  if (!invocation) {
    return { state: 'error' };
  }

  const args = [...invocation.args, 'auth', 'status', '--json', '--backend-url', backendUrl];
  let result;
  try {
    result = spawnSync(invocation.command, args, {
      cwd: getWorkspaceFolderPath() || process.cwd(),
      env: {
        ...process.env,
        CTXCE_AUTH_BACKEND_URL: backendUrl,
      },
      encoding: 'utf8',
    });
  } catch (_) {
    return { state: 'error' };
  }

  const stdout = (result && result.stdout) || '';
  let parsed;
  try {
    parsed = stdout ? JSON.parse(stdout) : null;
  } catch (_) {
    parsed = null;
  }
  const state = parsed && typeof parsed.state === 'string' ? parsed.state : 'error';
  const userId = parsed && typeof parsed.userId === 'string' ? parsed.userId : undefined;
  return { state, userId };
}


async function ensureAuthIfRequired(endpoint, deps) {
  try {
    if (!deps || !deps.vscode || !deps.spawnSync || !deps.resolveBridgeCliInvocation || !deps.getWorkspaceFolderPath || !deps.log) {
      return;
    }
    const { vscode, spawnSync, resolveBridgeCliInvocation, getWorkspaceFolderPath, log } = deps;
    const fetchFn = getFetch(deps);
    const raw = (endpoint || '').trim();
    if (!raw) {
      return;
    }
    if (!fetchFn) {
      log('Auth status probe skipped: fetch is not available in this runtime.');
      return;
    }

    let baseUrl = raw;
    try {
      const u = new URL(raw);
      baseUrl = `${u.protocol}//${u.host}`;
    } catch (_) {
      baseUrl = raw.replace(/\/+$/, '');
    }
    const statusUrl = `${baseUrl.replace(/\/+$/, '')}/auth/status`;

    let res;
    try {
      res = await fetchFn(statusUrl, { method: 'GET' });
    } catch (error) {
      log(`Auth status probe failed: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }
    if (!res || !res.ok) {
      return;
    }

    let json;
    try {
      json = await res.json();
    } catch (_) {
      return;
    }
    if (!json || !json.enabled) {
      return;
    }

    const invocation = resolveBridgeCliInvocation();
    if (!invocation) {
      log('Context Engine Uploader: ctxce CLI not found; skipping auth status check.');
      return;
    }

    const backendUrl = baseUrl;
    const workspacePath = (typeof getWorkspaceFolderPath === 'function' && getWorkspaceFolderPath()) || '';
    const skipKey = `${backendUrl}::${workspacePath}`;
    if (_skippedAuthCombos.has(skipKey)) {
      return;
    }

    const args = [...invocation.args, 'auth', 'status', '--json', '--backend-url', backendUrl];
    let result;
    try {
      result = spawnSync(invocation.command, args, {
        cwd: getWorkspaceFolderPath() || process.cwd(),
        env: {
          ...process.env,
          CTXCE_AUTH_BACKEND_URL: backendUrl,
        },
        encoding: 'utf8',
      });
    } catch (error) {
      log(`Auth status check failed to run: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }

    const stdout = (result && result.stdout) || '';
    let parsed;
    try {
      parsed = stdout ? JSON.parse(stdout) : null;
    } catch (_) {
      parsed = null;
    }
    const state = parsed && typeof parsed.state === 'string' ? parsed.state : undefined;
    const exitCode = result && typeof result.status === 'number' ? result.status : undefined;
    log(`Context Engine Uploader: auth status JSON state=${state || '<unknown>'} exitCode=${exitCode !== undefined ? exitCode : '<none>'}`);
    if (state === 'ok' || result.status === 0) {
      return;
    }

    const choice = await vscode.window.showInformationMessage(
      'Context Engine: authentication is enabled on the backend but no valid session is available.',
      'Sign In',
      'Skip for now',
    );
    if (choice !== 'Sign In') {
      _skippedAuthCombos.add(skipKey);
      return;
    }

    await runAuthLoginFlow(backendUrl, deps);
  } catch (error) {
    if (deps && typeof deps.log === 'function') {
      deps.log(`ensureAuthIfRequired error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

async function runAuthLoginFlow(explicitBackendUrl, deps) {
  if (!deps || !deps.vscode || !deps.spawn || !deps.resolveBridgeCliInvocation || !deps.getWorkspaceFolderPath || !deps.attachOutput || !deps.log) {
    return;
  }
  const { vscode, spawn, resolveBridgeCliInvocation, getWorkspaceFolderPath, attachOutput, log } = deps;
  let endpoint = '';
  try {
    if (deps && typeof deps.getEffectiveConfig === 'function') {
      const cfg = deps.getEffectiveConfig();
      endpoint = (cfg.get('endpoint') || '').trim();
    }
  } catch (_) {
    endpoint = '';
  }
  if (!endpoint) {
    const settings = vscode.workspace.getConfiguration('contextEngineUploader');
    endpoint = (settings.get('endpoint') || '').trim();
  }
  let backendUrl = explicitBackendUrl || endpoint;
  if (!backendUrl) {
    vscode.window.showErrorMessage('Context Engine Uploader: backend endpoint is not configured (contextEngineUploader.endpoint).');
    return;
  }

  try {
    const u = new URL(backendUrl);
    backendUrl = `${u.protocol}//${u.host}`;
  } catch (_) {
    backendUrl = backendUrl.replace(/\/+$/, '');
  }

  const mode = await vscode.window.showQuickPick(
    ['Token (shared dev token)', 'Username / password'],
    { placeHolder: 'Select Context Engine auth method' },
  );
  if (!mode) {
    return;
  }

  const invocation = resolveBridgeCliInvocation();
  if (!invocation) {
    vscode.window.showErrorMessage('Context Engine Uploader: unable to locate ctxce CLI for auth.');
    return;
  }
  const cwd = getWorkspaceFolderPath() || process.cwd();

  if (mode.startsWith('Token')) {
    const token = await vscode.window.showInputBox({
      prompt: 'Enter Context Engine shared auth token',
      password: true,
      ignoreFocusOut: true,
    });
    if (!token) {
      return;
    }
    const args = [...invocation.args, 'auth', 'login'];
    const env = {
      ...process.env,
      CTXCE_AUTH_BACKEND_URL: backendUrl,
      CTXCE_AUTH_TOKEN: token,
    };
    await new Promise(resolve => {
      const child = spawn(invocation.command, args, { cwd, env });
      attachOutput(child, 'auth');
      child.on('error', error => {
        log(`ctxce auth login (token) failed to start: ${error instanceof Error ? error.message : String(error)}`);
        vscode.window.showErrorMessage('Context Engine Uploader: auth login failed to start. See output for details.');
        resolve();
      });
      child.on('close', code => {
        if (code === 0) {
          vscode.window.showInformationMessage('Context Engine Uploader: auth login successful.');
        } else {
          vscode.window.showErrorMessage(`Context Engine Uploader: auth login failed with exit code ${code}. See output for details.`);
        }
        resolve();
      });
    });
    return;
  }

  const username = await vscode.window.showInputBox({
    prompt: 'Enter Context Engine username',
    ignoreFocusOut: true,
  });
  if (!username) {
    return;
  }
  const password = await vscode.window.showInputBox({
    prompt: 'Enter Context Engine password',
    password: true,
    ignoreFocusOut: true,
  });
  if (!password) {
    return;
  }

  const args = [...invocation.args, 'auth', 'login'];
  const env = {
    ...process.env,
    CTXCE_AUTH_BACKEND_URL: backendUrl,
    CTXCE_AUTH_USERNAME: username,
    CTXCE_AUTH_PASSWORD: password,
  };
  await new Promise(resolve => {
    const child = spawn(invocation.command, args, { cwd, env });
    attachOutput(child, 'auth');
    child.on('error', error => {
      log(`ctxce auth login failed to start: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: auth login failed to start. See output for details.');
      resolve();
    });
    child.on('close', code => {
      if (code === 0) {
        vscode.window.showInformationMessage('Context Engine Uploader: auth login successful.');
      } else {
        vscode.window.showErrorMessage(`Context Engine Uploader: auth login failed with exit code ${code}. See output for details.`);
      }
      resolve();
    });
  });
}

async function runAuthLogoutFlow(explicitBackendUrl, deps) {
  if (!deps || !deps.vscode || !deps.spawn || !deps.resolveBridgeCliInvocation || !deps.getWorkspaceFolderPath || !deps.attachOutput || !deps.log) {
    return;
  }
  const { vscode, spawn, resolveBridgeCliInvocation, getWorkspaceFolderPath, attachOutput, log } = deps;
  let endpoint = '';
  if (explicitBackendUrl) {
    endpoint = explicitBackendUrl;
  } else if (deps.getEffectiveConfig) {
    try {
      const cfg = deps.getEffectiveConfig();
      endpoint = (cfg.get('endpoint') || '').trim();
    } catch (_) { }
  }
  if (!endpoint) {
    return;
  }
  let backendUrl = endpoint;
  try {
    const u = new URL(endpoint);
    backendUrl = `${u.protocol}//${u.host}`;
  } catch (_) {
    backendUrl = endpoint.replace(/\/+$/, '');
  }

  const invocation = resolveBridgeCliInvocation();
  if (!invocation) {
    vscode.window.showErrorMessage('Context Engine Uploader: unable to locate ctxce CLI for auth logout.');
    return;
  }
  const cwd = getWorkspaceFolderPath() || process.cwd();

  const args = [...invocation.args, 'auth', 'logout'];
  const env = {
    ...process.env,
    CTXCE_AUTH_BACKEND_URL: backendUrl,
  };
  await new Promise(resolve => {
    const child = spawn(invocation.command, args, { cwd, env });
    attachOutput(child, 'auth');
    child.on('error', error => {
      log(`ctxce auth logout failed to start: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: auth logout failed to start. See output for details.');
      resolve();
    });
    child.on('close', code => {
      if (code === 0) {
        vscode.window.showInformationMessage('Context Engine Uploader: signed out successfully.');
      } else {
        vscode.window.showErrorMessage(`Context Engine Uploader: auth logout failed with exit code ${code}. See output for details.`);
      }
      resolve();
    });
  });
}

module.exports = {
  checkAuthStatus,
  ensureAuthIfRequired,
  runAuthLoginFlow,
  runAuthLogoutFlow,
};
