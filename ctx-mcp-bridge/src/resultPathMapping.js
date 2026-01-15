import process from "node:process";
import fs from "node:fs";
import path from "node:path";

function envTruthy(value, defaultVal = false) {
  try {
    if (value === undefined || value === null) {
      return defaultVal;
    }
    const s = String(value).trim().toLowerCase();
    if (!s) {
      return defaultVal;
    }
    return s === "1" || s === "true" || s === "yes" || s === "on";
  } catch {
    return defaultVal;
  }
}

function _posixToNative(rel) {
  try {
    if (!rel) {
      return "";
    }
    return String(rel).split("/").join(path.sep);
  } catch {
    return rel;
  }
}

function _nativeToPosix(p) {
  try {
    if (!p) {
      return "";
    }
    return String(p).split(path.sep).join("/");
  } catch {
    return p;
  }
}

function _workPathToRepoRelPosix(p) {
  try {
    const s = typeof p === "string" ? p.trim() : "";
    if (!s || !s.startsWith("/work/")) {
      return null;
    }
    const rest = s.slice("/work/".length);
    const parts = rest.split("/").filter(Boolean);
    if (parts.length >= 2) {
      return parts.slice(1).join("/");
    }
    if (parts.length === 1) {
      return parts[0];
    }
    return "";
  } catch {
    return null;
  }
}

function normalizeToolArgPath(p, workspaceRoot) {
  try {
    const s = typeof p === "string" ? p.trim() : "";
    if (!s) {
      return p;
    }

    const root = typeof workspaceRoot === "string" ? workspaceRoot : "";
    const sPosix = s.replace(/\\/g, "/");

    const fromWork = _workPathToRepoRelPosix(sPosix);
    if (typeof fromWork === "string" && fromWork) {
      return fromWork;
    }
    if (fromWork === "") {
      return p;
    }

    if (root) {
      try {
        const sNorm = s.replace(/\\/g, path.sep);
        const rootNorm = root.replace(/\\/g, path.sep);
        if (sNorm === rootNorm || sNorm.startsWith(rootNorm + path.sep)) {
          const relNative = path.relative(rootNorm, sNorm);
          const relPosix = _nativeToPosix(relNative);
          if (relPosix && relPosix !== "." && relPosix !== ".." && !relPosix.startsWith("../")) {
            return relPosix;
          }
        }
      } catch {
        // ignore
      }
      try {
        const base = path.posix.basename(root.replace(/\\/g, "/"));
        if (base && sPosix.startsWith(base + "/")) {
          const rest = sPosix.slice((base + "/").length);
          if (rest && rest !== "." && rest !== ".." && !rest.startsWith("../")) {
            return rest;
          }
        }
      } catch {
        // ignore
      }
    }

    if (sPosix.startsWith("./")) {
      const rest = sPosix.slice(2);
      if (rest && rest !== "." && rest !== ".." && !rest.startsWith("../")) {
        return rest;
      }
    }
    if (sPosix === ".") {
      return "";
    }
    return p;
  } catch {
    return p;
  }
}

function normalizeToolArgGlob(p, workspaceRoot) {
  try {
    const s = typeof p === "string" ? p : "";
    if (!s) {
      return p;
    }
    // TODO(ctxce): If this becomes annoying, consider making glob normalization
    // more conservative (e.g. only strip a repo prefix when followed by "/",
    // and avoid collapsing "<repo>/**" into "**" which can broaden scope).
    if (s.startsWith("!")) {
      const rest = s.slice(1);
      const mapped = normalizeToolArgPath(rest, workspaceRoot);
      if (typeof mapped === "string") {
        return "!" + mapped;
      }
      return p;
    }
    return normalizeToolArgPath(s, workspaceRoot);
  } catch {
    return p;
  }
}

function applyPathMappingToArgs(value, workspaceRoot, keyHint = "") {
  try {
    if (value === null || value === undefined) {
      return value;
    }

    const key = typeof keyHint === "string" ? keyHint : "";
    const lowered = key.toLowerCase();
    const shouldMapString =
      lowered === "path" ||
      lowered === "under" ||
      lowered === "root" ||
      lowered === "subdir" ||
      lowered === "path_glob" ||
      lowered === "not_glob";

    if (typeof value === "string") {
      if (!shouldMapString) {
        return value;
      }
      if (lowered === "path_glob" || lowered === "not_glob") {
        return normalizeToolArgGlob(value, workspaceRoot);
      }
      return normalizeToolArgPath(value, workspaceRoot);
    }

    if (Array.isArray(value)) {
      return value.map((v) => applyPathMappingToArgs(v, workspaceRoot, keyHint));
    }

    if (typeof value === "object") {
      const out = { ...value };
      for (const [k, v] of Object.entries(out)) {
        out[k] = applyPathMappingToArgs(v, workspaceRoot, k);
      }
      return out;
    }

    return value;
  } catch {
    return value;
  }
}

function computeWorkspaceRelativePath(containerPath, hostPath) {
  try {
    const cont = typeof containerPath === "string" ? containerPath.trim() : "";
    const rel = _workPathToRepoRelPosix(cont);
    if (typeof rel === "string" && rel) {
      return rel;
    }
  } catch {
  }
  try {
    const hp = typeof hostPath === "string" ? hostPath.trim() : "";
    if (!hp) {
      return "";
    }
    // If we don't have a container path, at least try to return a basename.
    return path.posix.basename(hp.replace(/\\/g, "/"));
  } catch {
    return "";
  }
}

function remapRelatedPathToClient(p, workspaceRoot) {
  try {
    const s = typeof p === "string" ? p : "";
    const root = typeof workspaceRoot === "string" ? workspaceRoot : "";
    if (!s || !root) {
      return p;
    }

    const sNorm = s.replace(/\\/g, path.sep);
    if (sNorm.startsWith(root + path.sep) || sNorm === root) {
      return sNorm;
    }

    const rel = _workPathToRepoRelPosix(s);
    if (typeof rel === "string" && rel) {
      const relNative = _posixToNative(rel);
      return path.join(root, relNative);
    }

    // If it's already a relative path, join it to the workspace root.
    if (!s.startsWith("/") && !s.includes(":") && !s.includes("\\")) {
      const relPosix = s.trim();
      if (relPosix && relPosix !== "." && !relPosix.startsWith("../") && relPosix !== "..") {
        const relNative = _posixToNative(relPosix);
        const joined = path.join(root, relNative);
        const relCheck = path.relative(root, joined);
        if (relCheck && !relCheck.startsWith(`..${path.sep}`) && relCheck !== "..") {
          return joined;
        }
      }
    }

    return p;
  } catch {
    return p;
  }
}

function remapHitPaths(hit, workspaceRoot) {
  if (!hit || typeof hit !== "object") {
    return hit;
  }
  const rawPath = typeof hit.path === "string" ? hit.path : "";
  let hostPath = typeof hit.host_path === "string" ? hit.host_path : "";
  let containerPath = typeof hit.container_path === "string" ? hit.container_path : "";
  if (!hostPath && rawPath) {
    hostPath = rawPath;
  }
  if (!containerPath && rawPath) {
    containerPath = rawPath;
  }
  const relPath = computeWorkspaceRelativePath(containerPath, hostPath);
  const out = { ...hit };
  if (relPath) {
    out.rel_path = relPath;
  }
  // Remap related_paths nested under each hit (repo_search/hybrid_search emit this per result).
  try {
    if (Array.isArray(out.related_paths)) {
      out.related_paths = out.related_paths.map((p) => remapRelatedPathToClient(p, workspaceRoot));
    }
  } catch {
    // ignore
  }
  if (workspaceRoot && relPath) {
    try {
      const relNative = _posixToNative(relPath);
      const candidate = path.join(workspaceRoot, relNative);
      const diagnostics = envTruthy(process.env.CTXCE_BRIDGE_PATH_DIAGNOSTICS, false);
      const strictClientPath = envTruthy(process.env.CTXCE_BRIDGE_CLIENT_PATH_STRICT, false);
      if (strictClientPath) {
        out.client_path = candidate;
        if (diagnostics) {
          out.client_path_joined = candidate;
          out.client_path_source = "workspace_join";
        }
      } else {
        // Prefer a host_path that is within the current bridge workspace.
        // This keeps provenance (host_path) intact while providing a user-local
        // absolute path even when the bridge workspace is a parent directory.
        const hp = typeof hostPath === "string" ? hostPath : "";
        const hpNorm = hp ? hp.replace(/\\/g, path.sep) : "";
        if (
          hpNorm &&
          hpNorm.startsWith(workspaceRoot) &&
          (!fs.existsSync(candidate) || fs.existsSync(hpNorm))
        ) {
          out.client_path = hpNorm;
          if (diagnostics) {
            out.client_path_joined = candidate;
            out.client_path_source = "host_path";
          }
        } else {
          out.client_path = candidate;
          if (diagnostics) {
            out.client_path_joined = candidate;
            out.client_path_source = "workspace_join";
          }
        }
      }
    } catch {
      // ignore
    }
  }
  const overridePath = envTruthy(process.env.CTXCE_BRIDGE_OVERRIDE_PATH, true);
  if (overridePath) {
    if (typeof out.client_path === "string" && out.client_path) {
      out.path = out.client_path;
    } else if (relPath) {
      out.path = relPath;
    }
  }
  // Strip internal container_path before returning to client.
  delete out.container_path;
  return out;
}

function remapStringPath(p, workspaceRoot) {
  try {
    const s = typeof p === "string" ? p : "";
    if (!s) {
      return p;
    }
    // If this is already a path within the current client workspace, rewrite to a
    // workspace-relative string when override is enabled.
    try {
      const root = typeof workspaceRoot === "string" ? workspaceRoot : "";
      if (root) {
        const sNorm = s.replace(/\\/g, path.sep);
        if (sNorm.startsWith(root + path.sep) || sNorm === root) {
          const relNative = path.relative(root, sNorm);
          const relPosix = String(relNative).split(path.sep).join("/");
          if (relPosix && !relPosix.startsWith("../") && relPosix !== "..") {
            const override = envTruthy(process.env.CTXCE_BRIDGE_OVERRIDE_PATH, true);
            if (override) {
              return relPosix;
            }
          }
        }
      }
    } catch {
      // ignore
    }
    const rel = _workPathToRepoRelPosix(s);
    if (typeof rel === "string" && rel) {
      const override = envTruthy(process.env.CTXCE_BRIDGE_OVERRIDE_PATH, true);
      if (override) {
        return rel;
      }
      return p;
    }
    return p;
  } catch {
    return p;
  }
}

function maybeParseToolJson(result) {
  try {
    if (
      result &&
      typeof result === "object" &&
      result.structuredContent &&
      typeof result.structuredContent === "object"
    ) {
      return { mode: "structured", value: result.structuredContent };
    }
  } catch {
  }
  try {
    const content = result && result.content;
    if (!Array.isArray(content)) {
      return null;
    }
    const first = content.find(
      (c) => c && c.type === "text" && typeof c.text === "string",
    );
    if (!first) {
      return null;
    }
    const txt = String(first.text || "").trim();
    if (!txt || !(txt.startsWith("{") || txt.startsWith("["))) {
      return null;
    }
    return { mode: "text", value: JSON.parse(txt) };
  } catch {
    return null;
  }
}

function applyPathMappingToPayload(payload, workspaceRoot) {
  if (!payload || typeof payload !== "object") {
    return payload;
  }
  const out = Array.isArray(payload) ? payload.slice() : { ...payload };

  const mapHitsArray = (arr) => {
    if (!Array.isArray(arr)) {
      return arr;
    }
    return arr.map((h) => remapHitPaths(h, workspaceRoot));
  };

  // Common result shapes across tools
  if (Array.isArray(out.results)) {
    out.results = mapHitsArray(out.results);
  }
  if (Array.isArray(out.citations)) {
    out.citations = mapHitsArray(out.citations);
  }
  if (Array.isArray(out.related_paths)) {
    out.related_paths = out.related_paths.map((p) => remapRelatedPathToClient(p, workspaceRoot));
  }

  // Some tools nest under {result:{...}}
  if (out.result && typeof out.result === "object") {
    out.result = applyPathMappingToPayload(out.result, workspaceRoot);
  }

  return out;
}

export function maybeRemapToolResult(name, result, workspaceRoot) {
  try {
    if (!name || !result || !workspaceRoot) {
      return result;
    }
    const enabled = envTruthy(process.env.CTXCE_BRIDGE_MAP_PATHS, true);
    if (!enabled) {
      return result;
    }
    const lower = String(name).toLowerCase();
    const shouldMap = (
      lower === "repo_search" ||
      lower === "context_search" ||
      lower === "context_answer" ||
      lower.endsWith("search_tests_for") ||
      lower.endsWith("search_config_for") ||
      lower.endsWith("search_callers_for") ||
      lower.endsWith("search_importers_for")
    );
    if (!shouldMap) {
      return result;
    }

    const parsed = maybeParseToolJson(result);
    if (!parsed) {
      return result;
    }

    const mapped = applyPathMappingToPayload(parsed.value, workspaceRoot);
    let outResult = result;
    if (parsed.mode === "structured") {
      outResult = { ...result, structuredContent: mapped };
    }

    // Replace text payload for clients that only read `content[].text`
    try {
      const content = Array.isArray(outResult.content) ? outResult.content.slice() : [];
      const idx = content.findIndex(
        (c) => c && c.type === "text" && typeof c.text === "string",
      );
      if (idx >= 0) {
        content[idx] = { ...content[idx], text: JSON.stringify(mapped) };
        outResult = { ...outResult, content };
      }
    } catch {
      // ignore
    }
    return outResult;
  } catch {
    return result;
  }
}

export function maybeRemapToolArgs(name, args, workspaceRoot) {
  try {
    if (!name || !workspaceRoot) {
      return args;
    }
    const enabled = envTruthy(process.env.CTXCE_BRIDGE_MAP_ARGS, true);
    if (!enabled) {
      return args;
    }
    if (args === null || args === undefined || typeof args !== "object") {
      return args;
    }
    return applyPathMappingToArgs(args, workspaceRoot, "");
  } catch {
    return args;
  }
}
