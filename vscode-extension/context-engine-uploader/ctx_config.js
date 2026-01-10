const path = require('path');
const fs = require('fs');

function createCtxConfigManager(deps) {
  const vscode = deps.vscode;
  const spawnSync = deps.spawnSync;
  const log = deps.log;

  const extensionRoot = deps.extensionRoot;
  const getEffectiveConfig = deps.getEffectiveConfig;
  const resolveOptions = deps.resolveOptions;
  const ensurePythonDependencies = deps.ensurePythonDependencies;
  const buildChildEnv = deps.buildChildEnv;
  const resolveBridgeHttpUrl = deps.resolveBridgeHttpUrl;

  function dispose() {
    // Currently no timers/processes; kept for consistent manager lifecycle.
  }

  async function writeCtxConfig() {
    const settings = getEffectiveConfig();
    const enabled = settings.get('scaffoldCtxConfig', true);
    if (!enabled) {
      vscode.window.showInformationMessage('Context Engine Uploader: ctx_config/.env scaffolding is disabled (contextEngineUploader.scaffoldCtxConfig=false).');
      log('CTX config scaffolding skipped because scaffoldCtxConfig is false.');
      return;
    }
    let options = resolveOptions();
    if (!options) {
      return;
    }
    const depsOk = await ensurePythonDependencies(
      options.pythonPath,
      options.workingDirectory,
      options.pythonPathSource
    );
    if (!depsOk) {
      return;
    }
    options = resolveOptions() || options;
    const collectionName = inferCollectionFromUpload(options);
    if (!collectionName) {
      vscode.window.showErrorMessage('Context Engine Uploader: failed to infer collection name from upload client. Check the Output panel for details.');
      return;
    }
    await scaffoldCtxConfigFiles(options.targetPath, collectionName);
  }

  function inferCollectionFromUpload(options) {
    try {
      const args = ['-u', options.scriptPath, '--path', options.targetPath, '--endpoint', options.endpoint, '--show-mapping'];
      const result = spawnSync(options.pythonPath, args, {
        cwd: options.workingDirectory,
        env: buildChildEnv(options),
        encoding: 'utf8'
      });
      if (result.error) {
        log(`Failed to run standalone_upload_client for collection inference: ${result.error.message || String(result.error)}`);
        return undefined;
      }
      const stdout = result.stdout || '';
      const stderr = result.stderr || '';

      if (stdout) {
        log(`[ctx-config] upload client --show-mapping output:\n${stdout}`);
      }
      if (stderr) {
        log(`[ctx-config] upload client stderr:\n${stderr}`);
      }

      const combined = `${stdout}\n${stderr}`;
      if (combined.trim()) {
        const lines = combined.split(/\r?\n/);
        for (const line of lines) {
          const m = line.match(/collection_name:\s*(.+)$/);
          if (m && m[1]) {
            const name = m[1].trim();
            if (name) {
              return name;
            }
          }
        }
      }
    } catch (error) {
      log(`Error inferring collection from upload client: ${error instanceof Error ? error.message : String(error)}`);
    }
    return undefined;
  }

  async function scaffoldCtxConfigFiles(workspaceDir, collectionName) {
    try {
      const placeholders = new Set(['', 'default-collection', 'my-collection', 'codebase']);

      let uploaderSettings;
      try {
        uploaderSettings = getEffectiveConfig();
      } catch (error) {
        log(`Failed to read uploader settings: ${error instanceof Error ? error.message : String(error)}`);
        uploaderSettings = undefined;
      }

      let decoderRuntime = 'glm';
      let useGpuDecoderSetting = false;
      let glmApiKey = '';
      let glmApiBase = 'https://api.z.ai/api/coding/paas/v4/';
      let glmModel = 'glm-4.6';
      let gitMaxCommits = 500;
      let gitSince = '';
      if (uploaderSettings) {
        try {
          const runtimeSetting = String(uploaderSettings.get('decoderRuntime') ?? 'glm').trim().toLowerCase();
          if (runtimeSetting === 'llamacpp') {
            decoderRuntime = 'llamacpp';
          }
          useGpuDecoderSetting = !!uploaderSettings.get('useGpuDecoder', false);
          // TODO: glmApiKey is read from settings.json (plaintext). Consider migrating API keys to VS Code SecretStorage (context.secrets) with backwards-compatible fallback.
          const cfgKey = (uploaderSettings.get('glmApiKey') || '').trim();
          const cfgBase = (uploaderSettings.get('glmApiBase') || '').trim();
          const cfgModel = (uploaderSettings.get('glmModel') || '').trim();
          if (cfgKey) {
            glmApiKey = cfgKey;
          }
          if (cfgBase) {
            glmApiBase = cfgBase;
          }
          if (cfgModel) {
            glmModel = cfgModel;
          }
          const maxCommitsSetting = uploaderSettings.get('gitMaxCommits');
          if (typeof maxCommitsSetting === 'number' && !Number.isNaN(maxCommitsSetting)) {
            gitMaxCommits = maxCommitsSetting;
          }
          const sinceSetting = uploaderSettings.get('gitSince');
          if (typeof sinceSetting === 'string') {
            gitSince = sinceSetting.trim();
          }
        } catch (error) {
          log(`Failed to read decoder/GLM settings from configuration: ${error instanceof Error ? error.message : String(error)}`);
        }
      }

      const ctxConfigPath = path.join(workspaceDir, 'ctx_config.json');
      let ctxConfig = {};
      if (fs.existsSync(ctxConfigPath)) {
        try {
          const raw = fs.readFileSync(ctxConfigPath, 'utf8');
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            ctxConfig = parsed;
          }
        } catch (error) {
          log(`Failed to parse existing ctx_config.json at ${ctxConfigPath}; overwriting with minimal config. Error: ${error instanceof Error ? error.message : String(error)}`);
          ctxConfig = {};
        }
      }
      const currentDefault = typeof ctxConfig.default_collection === 'string' ? ctxConfig.default_collection.trim() : '';
      let ctxChanged = false;
      let notifiedDefault = false;
      if (!currentDefault || placeholders.has(currentDefault)) {
        ctxConfig.default_collection = collectionName;
        ctxChanged = true;
        notifiedDefault = true;
      }
      if (ctxConfig.default_mode === undefined) {
        ctxConfig.default_mode = 'default';
        ctxChanged = true;
      }
      if (ctxConfig.require_context === undefined) {
        ctxConfig.require_context = true;
        ctxChanged = true;
      }
      if (ctxConfig.surface_qdrant_collection_hint === undefined) {
        let surfaceHintSetting = true;
        if (uploaderSettings) {
          try {
            surfaceHintSetting = !!uploaderSettings.get('surfaceQdrantCollectionHint', true);
          } catch (error) {
            log(`Failed to read surfaceQdrantCollectionHint from configuration: ${error instanceof Error ? error.message : String(error)}`);
          }
        }
        ctxConfig.surface_qdrant_collection_hint = surfaceHintSetting;
        ctxChanged = true;
      }
      if (ctxConfig.refrag_runtime !== decoderRuntime) {
        ctxConfig.refrag_runtime = decoderRuntime;
        ctxChanged = true;
      }
      if (decoderRuntime === 'glm') {
        if (ctxConfig.glm_api_base === undefined) {
          ctxConfig.glm_api_base = glmApiBase;
          ctxChanged = true;
        }
        if (ctxConfig.glm_model === undefined) {
          ctxConfig.glm_model = glmModel;
          ctxChanged = true;
        }
        const existingGlmKey = typeof ctxConfig.glm_api_key === 'string' ? ctxConfig.glm_api_key.trim() : '';
        if (glmApiKey) {
          if (!existingGlmKey) {
            ctxConfig.glm_api_key = glmApiKey;
            ctxChanged = true;
          }
        } else if (ctxConfig.glm_api_key === undefined) {
          ctxConfig.glm_api_key = '';
          ctxChanged = true;
        }
      }
      if (decoderRuntime === 'llamacpp') {
        if (ctxConfig.llamacpp_model === undefined) {
          ctxConfig.llamacpp_model = 'llamacpp-4.6';
          ctxChanged = true;
        }
      }
      if (ctxChanged) {
        fs.writeFileSync(ctxConfigPath, JSON.stringify(ctxConfig, null, 2) + '\n', 'utf8');
        if (notifiedDefault) {
          vscode.window.showInformationMessage(`Context Engine Uploader: ctx_config.json updated with default_collection=${collectionName}.`);
        } else {
          vscode.window.showInformationMessage('Context Engine Uploader: ctx_config.json refreshed with required defaults.');
        }
        log(`Wrote ctx_config.json at ${ctxConfigPath}`);
      } else {
        log(`ctx_config.json at ${ctxConfigPath} already satisfied required values; not modified.`);
      }

      const envPath = path.join(workspaceDir, '.env');
      let envContent = '';

      const baseDir = extensionRoot || __dirname;
      const envExamplePath = path.join(baseDir, 'env.example');
      if (fs.existsSync(envPath)) {
        try {
          envContent = fs.readFileSync(envPath, 'utf8');
        } catch (error) {
          log(`Failed to read existing .env at ${envPath}; skipping .env update. Error: ${error instanceof Error ? error.message : String(error)}`);
          return;
        }
      } else if (fs.existsSync(envExamplePath)) {
        try {
          envContent = fs.readFileSync(envExamplePath, 'utf8');
          log(`Seeding new .env for ${workspaceDir} from bundled env.example.`);
        } catch (error) {
          log(`Failed to read bundled env.example at ${envExamplePath}; starting with minimal .env. Error: ${error instanceof Error ? error.message : String(error)}`);
          envContent = '';
        }
      }
      let envLines = envContent ? envContent.split(/\r?\n/) : [];
      let envChanged = false;
      let collectionUpdated = false;

      let idx = -1;
      for (let i = 0; i < envLines.length; i++) {
        if (envLines[i].trim().startsWith('COLLECTION_NAME=')) {
          idx = i;
          break;
        }
      }
      let currentEnvVal = '';
      if (idx >= 0) {
        const m = envLines[idx].match(/^COLLECTION_NAME=(.*)$/);
        if (m) {
          currentEnvVal = (m[1] || '').trim();
        }
      }
      if (idx === -1 || placeholders.has(currentEnvVal)) {
        const newLine = `COLLECTION_NAME=${collectionName}`;
        if (idx === -1) {
          if (envLines.length && envLines[envLines.length - 1].trim() !== '') {
            envLines.push('');
          }
          envLines.push(newLine);
        } else {
          envLines[idx] = newLine;
        }
        envChanged = true;
        collectionUpdated = true;
        vscode.window.showInformationMessage(`Context Engine Uploader: .env updated with COLLECTION_NAME=${collectionName}.`);
        log(`Updated .env at ${envPath}`);
      } else {
        log(`.env at ${envPath} already has non-placeholder COLLECTION_NAME; not modified.`);
      }

      function getEnvEntry(key) {
        for (let i = 0; i < envLines.length; i++) {
          const line = envLines[i];
          if (!line || line.trim().startsWith('#')) {
            continue;
          }
          const eqIndex = line.indexOf('=');
          if (eqIndex === -1) {
            continue;
          }
          const candidate = line.slice(0, eqIndex).trim();
          if (candidate === key) {
            return { index: i, value: line.slice(eqIndex + 1) };
          }
        }
        return { index: -1, value: undefined };
      }

      function upsertEnv(key, desiredValue, options = {}) {
        const {
          overwrite = false,
          treatEmptyAsUnset = false,
          placeholderValues = [],
          skipIfDesiredEmpty = false
        } = options;
        const desired = desiredValue ?? '';
        const desiredStr = String(desired);
        if (!desiredStr && skipIfDesiredEmpty) {
          return false;
        }
        const { index, value } = getEnvEntry(key);
        const current = typeof value === 'string' ? value.trim() : '';
        const normalizedDesired = desiredStr.trim();
        const placeholderSet = new Set((placeholderValues || []).map(val => (val || '').trim().toLowerCase()));
        let shouldUpdate = false;

        if (index === -1) {
          shouldUpdate = true;
        } else if (overwrite) {
          if (current !== normalizedDesired) {
            shouldUpdate = true;
          }
        } else if (treatEmptyAsUnset && !current) {
          shouldUpdate = true;
        } else if (placeholderSet.size && placeholderSet.has(current.toLowerCase())) {
          shouldUpdate = true;
        }

        if (!shouldUpdate) {
          return false;
        }

        const newLine = `${key}=${desiredStr}`;
        if (index === -1) {
          if (envLines.length && envLines[envLines.length - 1].trim() !== '') {
            envLines.push('');
          }
          envLines.push(newLine);
        } else {
          envLines[index] = newLine;
        }
        envChanged = true;
        return true;
      }

      upsertEnv('MULTI_REPO_MODE', '1', { overwrite: true });
      upsertEnv('REFRAG_MODE', '1', { overwrite: true });
      upsertEnv('REFRAG_DECODER', '1', { overwrite: true });
      upsertEnv('REFRAG_RUNTIME', decoderRuntime, { overwrite: true, placeholderValues: ['llamacpp', 'glm'] });
      upsertEnv('USE_GPU_DECODER', useGpuDecoderSetting ? '1' : '0', { overwrite: true });

      upsertEnv('REFRAG_ENCODER_MODEL', 'BAAI/bge-base-en-v1.5', { treatEmptyAsUnset: true });
      upsertEnv('REFRAG_PHI_PATH', '/work/models/refrag_phi_768_to_dmodel.bin', { treatEmptyAsUnset: true });
      upsertEnv('REFRAG_SENSE', 'heuristic', { treatEmptyAsUnset: true });

      if (decoderRuntime === 'glm') {
        const glmKeyPlaceholders = ['YOUR_GLM_API_KEY', '"YOUR_GLM_API_KEY"', "''", '""'];
        if (glmApiKey) {
          upsertEnv('GLM_API_KEY', glmApiKey, {
            treatEmptyAsUnset: true,
            placeholderValues: glmKeyPlaceholders
          });
        } else {
          upsertEnv('GLM_API_KEY', '', {});
        }
        upsertEnv('GLM_API_BASE', glmApiBase, { treatEmptyAsUnset: true });
        upsertEnv('GLM_MODEL', glmModel, { treatEmptyAsUnset: true });
      }

      if (uploaderSettings) {
        try {
          const transportModeRaw = uploaderSettings.get('mcpTransportMode') || 'sse-remote';
          const serverModeRaw = uploaderSettings.get('mcpServerMode') || 'bridge';
          const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
          const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
          let targetUrl = (uploaderSettings.get('ctxIndexerUrl') || 'http://localhost:8003/mcp').trim();
          if (serverMode === 'bridge' && transportMode === 'http') {
            const bridgeUrl = resolveBridgeHttpUrl();
            if (bridgeUrl) {
              targetUrl = bridgeUrl;
            }
          }
          if (targetUrl) {
            upsertEnv('MCP_INDEXER_URL', targetUrl, { treatEmptyAsUnset: true });
          }
        } catch (error) {
          log(`Failed to read ctxIndexerUrl setting for MCP_INDEXER_URL: ${error instanceof Error ? error.message : String(error)}`);
        }
      }

      if (typeof gitMaxCommits === 'number' && !Number.isNaN(gitMaxCommits)) {
        upsertEnv('REMOTE_UPLOAD_GIT_MAX_COMMITS', String(gitMaxCommits), { overwrite: true });
      }
      if (gitSince) {
        upsertEnv('REMOTE_UPLOAD_GIT_SINCE', gitSince, { overwrite: true, skipIfDesiredEmpty: true });
      }

      if (envChanged) {
        fs.writeFileSync(envPath, envLines.join('\n') + '\n', 'utf8');
        log(`Ensured decoder/GLM/MCP settings in .env at ${envPath}`);
      } else {
        log(`.env at ${envPath} already satisfied CTX defaults; not modified.`);
      }

      void collectionUpdated;
    } catch (error) {
      log(`Error scaffolding ctx_config/.env: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return {
    writeCtxConfig,
    inferCollectionFromUpload,
    scaffoldCtxConfigFiles,
    dispose,
  };
}

module.exports = {
  createCtxConfigManager,
};
