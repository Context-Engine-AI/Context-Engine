import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const CONFIG_DIR_NAME = ".ctxce";
const CONFIG_BASENAME = "auth.json";

function getConfigPath() {
  const home = os.homedir() || process.cwd();
  const dir = path.join(home, CONFIG_DIR_NAME);
  return path.join(dir, CONFIG_BASENAME);
}

export function readConfig() {
  try {
    const cfgPath = getConfigPath();
    const raw = fs.readFileSync(cfgPath, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch (err) {
  }
  return {};
}

function writeConfig(data) {
  try {
    const cfgPath = getConfigPath();
    const dir = path.dirname(cfgPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(cfgPath, JSON.stringify(data, null, 2), "utf8");
  } catch (err) {
  }
}

export function loadAuthEntry(backendUrl) {
  if (!backendUrl) {
    return null;
  }
  const all = readConfig();
  const key = String(backendUrl);
  const entry = all[key];
  if (!entry || typeof entry !== "object") {
    return null;
  }
  return entry;
}

export function saveAuthEntry(backendUrl, entry) {
  if (!backendUrl || !entry || typeof entry !== "object") {
    return;
  }
  const all = readConfig();
  const key = String(backendUrl);
  all[key] = entry;
  writeConfig(all);
}

export function deleteAuthEntry(backendUrl) {
  if (!backendUrl) {
    return;
  }
  const all = readConfig();
  const key = String(backendUrl);
  if (Object.prototype.hasOwnProperty.call(all, key)) {
    delete all[key];
    writeConfig(all);
  }
}

export function loadAnyAuthEntry() {
  const all = readConfig();
  const keys = Object.keys(all);
  for (const key of keys) {
    const entry = all[key];
    if (entry && typeof entry === "object") {
      return { backendUrl: key, entry };
    }
  }
  return null;
}
