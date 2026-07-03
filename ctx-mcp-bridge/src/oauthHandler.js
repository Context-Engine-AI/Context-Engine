// OAuth 2.0 Handler for HTTP MCP Server
// Implements RFC9728 Protected Resource Metadata and RFC7591 Dynamic Client Registration

import fs from "node:fs";
import { randomBytes } from "node:crypto";
import { loadAnyAuthEntry, saveAuthEntry } from "./authConfig.js";

// ============================================================================
// OAuth Storage (in-memory for bridge process)
// ============================================================================

// Maps bearer tokens to session IDs
const tokenStore = new Map();
// Maps authorization codes to session info
const pendingCodes = new Map();
// Maps client_id to client info
const registeredClients = new Map();

// ============================================================================
// OAuth Utilities
// ============================================================================

/**
 * Clean up expired tokens from tokenStore
 * Called periodically to prevent unbounded memory growth
 */
function cleanupExpiredTokens() {
  const now = Date.now();
  const expiryMs = 86400000; // 24 hours
  for (const [token, data] of tokenStore.entries()) {
    if (now - data.createdAt > expiryMs) {
      tokenStore.delete(token);
    }
  }
}

function generateToken() {
  return randomBytes(32).toString("hex");
}

function generateCode() {
  return randomBytes(16).toString("base64url");
}

function debugLog(message) {
  try {
    const text = typeof message === "string" ? message : String(message);
    console.error(text);
    const dest = process.env.CTXCE_DEBUG_LOG;
    if (dest) {
      fs.appendFileSync(dest, `${new Date().toISOString()} ${text}\n`, "utf8");
    }
  } catch {
    // ignore logging errors
  }
}

// ============================================================================
// OAuth 2.0 Metadata (RFC9728)
// ============================================================================

export function getOAuthMetadata(issuerUrl) {
  return {
    issuer: issuerUrl,
    authorization_endpoint: `${issuerUrl}/oauth/authorize`,
    token_endpoint: `${issuerUrl}/oauth/token`,
    registration_endpoint: `${issuerUrl}/oauth/register`, // RFC7591 Dynamic Client Registration
    response_types_supported: ["code"],
    grant_types_supported: ["authorization_code"],
    token_endpoint_auth_methods_supported: ["none"],
    code_challenge_methods_supported: ["S256"],
    scopes_supported: ["mcp"],
  };
}

// ============================================================================
// HTML Login Page
// ============================================================================

/**
 * Safely escape JSON for embedding in HTML script context
 * Escapes special characters that could break out of a script tag
 */
function escapeJsonForHtml(obj) {
  const json = JSON.stringify(obj);
  // Replace dangerous characters with HTML-safe equivalents
  // </script> can break out of script tag, so replace </ with \u003C/
  return json.replace(/</g, '\\u003C').replace(/>/g, '\\u003E');
}

export function getLoginPage(redirectUri, clientId, state, codeChallenge, codeChallengeMethod) {
  const params = new URLSearchParams({
    redirect_uri: redirectUri || "",
    client_id: clientId || "",
    state: state || "",
    code_challenge: codeChallenge || "",
    code_challenge_method: codeChallengeMethod || "",
  });

  return `<!DOCTYPE html>
<html>
<head>
  <title>Context Engine MCP - Login</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; }
    h1 { color: #333; }
    .form-group { margin-bottom: 15px; }
    label { display: block; margin-bottom: 5px; font-weight: 500; }
    input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
    button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    button:hover { background: #0056b3; }
    .info { background: #e7f3ff; padding: 10px; border-radius: 4px; margin-bottom: 20px; font-size: 14px; }
    .error { color: #dc3545; margin-top: 10px; }
    .success { color: #28a745; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>Context Engine MCP Bridge</h1>
  <div class="info">
    This MCP bridge requires authentication. Please log in to your Context Engine backend.
  </div>

  <div id="result"></div>

  <form id="loginForm">
    <div class="form-group">
      <label>Backend URL</label>
      <input type="url" id="backendUrl" placeholder="http://localhost:8004" required>
    </div>
    <div class="form-group">
      <label>Username (optional)</label>
      <input type="text" id="username" placeholder="Leave empty for token auth">
    </div>
    <div class="form-group">
      <label>Password (optional)</label>
      <input type="password" id="password" placeholder="Required if username provided">
    </div>
    <div class="form-group">
      <label>Auth Token (if no username)</label>
      <input type="text" id="token" placeholder="Your shared auth token">
    </div>
    <button type="submit">Login & Authorize</button>
  </form>

  <script>
  const params = ${escapeJsonForHtml(Object.fromEntries(params))};
  document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const result = document.getElementById('result');
    result.innerHTML = '<p style="color: #007bff;">Logging in...</p>';

    const backendUrl = document.getElementById('backendUrl').value;
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const token = document.getElementById('token').value;

    const usePassword = username && password;
    const body = usePassword
      ? { username, password, workspace: '/tmp/bridge-oauth' }
      : { client: 'ctxce', workspace: '/tmp/bridge-oauth', token: token || undefined };

    const target = backendUrl.replace(/\\/+$/, '') + (usePassword ? '/auth/login/password' : '/auth/login');

    try {
      const resp = await fetch(target, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!resp.ok) {
        throw new Error('Login failed: ' + resp.status);
      }

      const data = await resp.json();
      const sessionId = data.session_id || data.sessionId;
      if (!sessionId) {
        throw new Error('No session in response');
      }

      // Store the session and get authorization code
      const storeResp = await fetch('/oauth/store-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          backend_url: backendUrl,
          redirect_uri: params.redirect_uri,
          state: params.state,
          code_challenge: params.code_challenge,
          code_challenge_method: params.code_challenge_method,
          client_id: params.client_id
        })
      });

      if (!storeResp.ok) {
        throw new Error('Failed to store session');
      }

      const storeData = await storeResp.json();
      if (storeData.redirect) {
        window.location.href = storeData.redirect;
      } else {
        throw new Error('No redirect URL');
      }
    } catch (err) {
      result.innerHTML = '<p class="error">' + err.message + '</p>';
    }
  });
  </script>
</body>
</html>`;
}

// ============================================================================
// OAuth Endpoint Handlers
// ============================================================================

/**
 * Validate client_id and redirect_uri against registered clients
 * @param {string} clientId - OAuth client_id
 * @param {string} redirectUri - OAuth redirect_uri
 * @returns {boolean} - true if both client_id and redirect_uri are valid
 */
function validateClientAndRedirect(clientId, redirectUri) {
  if (!clientId || !redirectUri) {
    return false;
  }
  const client = registeredClients.get(clientId);
  if (!client) {
    return false;
  }
  // Check if redirect_uri exactly matches one of the registered URIs
  const redirectUris = client.redirectUris || [];
  return redirectUris.includes(redirectUri);
}

/**
 * Handle OAuth metadata endpoint (RFC9728)
 * GET /.well-known/oauth-authorization-server
 */
export function handleOAuthMetadata(_req, res, issuerUrl) {
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(getOAuthMetadata(issuerUrl)));
}

/**
 * Handle OAuth Dynamic Client Registration (RFC7591)
 * POST /oauth/register
 */
export function handleOAuthRegister(req, res) {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    try {
      const data = JSON.parse(body);

      // Validate required fields
      if (!data.redirect_uris || !Array.isArray(data.redirect_uris) || data.redirect_uris.length === 0) {
        res.statusCode = 400;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: "invalid_redirect_uri" }));
        return;
      }

      // Auto-approve any client registration for local bridge
      const clientId = generateToken().slice(0, 32);
      const client_id = `mcp_${clientId}`;

      registeredClients.set(client_id, {
        clientId: client_id,
        redirectUris: data.redirect_uris,
        grantTypes: data.grant_types || ["authorization_code"],
        createdAt: Date.now(),
      });

      res.setHeader("Content-Type", "application/json");
      res.statusCode = 201;
      res.end(JSON.stringify({
        client_id: client_id,
        client_id_issued_at: Math.floor(Date.now() / 1000),
        grant_types: ["authorization_code"],
        redirect_uris: data.redirect_uris,
        response_types: ["code"],
        token_endpoint_auth_method: "none",
      }));
    } catch (err) {
      debugLog("[ctxce] /oauth/register error: " + String(err));
      res.statusCode = 400;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "invalid_client_metadata", error_description: String(err) }));
    }
  });
}

/**
 * Handle OAuth authorize endpoint
 * GET /oauth/authorize
 */
export function handleOAuthAuthorize(_req, res, searchParams) {
  const redirectUri = searchParams.get("redirect_uri");
  const clientId = searchParams.get("client_id");
  const state = searchParams.get("state");
  const responseType = searchParams.get("response_type");
  const codeChallenge = searchParams.get("code_challenge");
  const codeChallengeMethod = searchParams.get("code_challenge_method") || "S256";

  // Validate response_type is "code" (authorization code flow)
  if (responseType !== "code") {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "unsupported_response_type", error_description: "Only response_type=code is supported" }));
    return;
  }

  // Validate client_id and redirect_uri against registered clients
  if (!validateClientAndRedirect(clientId, redirectUri)) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "invalid_client", error_description: "Unknown client_id or unauthorized redirect_uri" }));
    return;
  }

  // If already logged in (has valid session), auto-approve
  const existingAuth = loadAnyAuthEntry();
  if (existingAuth && existingAuth.entry && existingAuth.entry.sessionId) {
    // Auto-generate code and redirect
    const code = generateCode();
    pendingCodes.set(code, {
      clientId,
      sessionId: existingAuth.entry.sessionId,
      backendUrl: existingAuth.backendUrl,
      codeChallenge,
      codeChallengeMethod,
      redirectUri,
      createdAt: Date.now(),
    });

    const redirectUrl = new URL(redirectUri || "http://localhost/callback");
    redirectUrl.searchParams.set("code", code);
    if (state) redirectUrl.searchParams.set("state", state);
    res.setHeader("Location", redirectUrl.toString());
    res.statusCode = 302;
    res.end();
    return;
  }

  // Otherwise, show login page
  res.setHeader("Content-Type", "text/html");
  res.end(getLoginPage(redirectUri, clientId, state, codeChallenge, codeChallengeMethod));
}

/**
 * Handle OAuth store-session endpoint (helper for login page)
 * POST /oauth/store-session
 *
 * Security note: This endpoint is called from the browser after login.
 * Since the HTTP server binds to 127.0.0.1 only, this is only accessible from localhost.
 * For additional CSRF protection, we validate client_id and redirect_uri match a
 * previously registered client.
 */
export function handleOAuthStoreSession(req, res) {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    res.setHeader("Content-Type", "application/json");
    try {
      const data = JSON.parse(body);
      const { session_id, backend_url, redirect_uri, state, code_challenge, code_challenge_method, client_id } = data;

      if (!session_id || !backend_url) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "Missing session_id or backend_url" }));
        return;
      }

      // Validate backend_url is a valid URL string (prevent prototype pollution)
      // Only allow http/https schemes for backend API URLs
      try {
        const url = new URL(backend_url);
        if (url.protocol !== "http:" && url.protocol !== "https:") {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Invalid backend_url", error_description: "Only http/https URLs are allowed" }));
          return;
        }
      } catch {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "Invalid backend_url" }));
        return;
      }

      // Validate client_id and redirect_uri against registered clients
      // Note: client_id is passed from the login page which gets it from the initial auth request
      if (!validateClientAndRedirect(client_id, redirect_uri)) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "invalid_client", error_description: "Unknown client_id or unauthorized redirect_uri" }));
        return;
      }

      // Additional CSRF protection: verify request came from a local browser origin
      // Require Origin or Referer header to be present and from localhost
      const origin = req.headers["origin"] || req.headers["referer"];
      if (!origin) {
        res.statusCode = 403;
        res.end(JSON.stringify({ error: "forbidden", error_description: "Origin or Referer header required" }));
        return;
      }
      try {
        const originUrl = new URL(origin);
        const hostname = originUrl.hostname;
        // Only allow localhost or 127.0.0.1 origins
        if (hostname !== "localhost" && hostname !== "127.0.0.1") {
          res.statusCode = 403;
          res.end(JSON.stringify({ error: "forbidden", error_description: "Request must originate from localhost" }));
          return;
        }
      } catch {
        // If origin parsing fails, reject the request
        res.statusCode = 403;
        res.end(JSON.stringify({ error: "forbidden", error_description: "Invalid origin" }));
        return;
      }

      // Save the auth entry
      saveAuthEntry(backend_url, {
        sessionId: session_id,
        userId: "oauth-user",
        expiresAt: null,
      });

      // Generate auth code
      const code = generateCode();
      pendingCodes.set(code, {
        clientId: client_id,
        sessionId: session_id,
        backendUrl: backend_url,
        codeChallenge: code_challenge,
        codeChallengeMethod: code_challenge_method,
        redirectUri: redirect_uri,
        createdAt: Date.now(),
      });

      // Return redirect URL
      const redirectUrl = new URL(redirect_uri || "http://localhost/callback");
      redirectUrl.searchParams.set("code", code);
      if (state) redirectUrl.searchParams.set("state", state);

      res.end(JSON.stringify({ redirect: redirectUrl.toString() }));
    } catch (err) {
      debugLog("[ctxce] /oauth/store-session error: " + String(err));
      res.statusCode = 400;
      res.end(JSON.stringify({ error: String(err) }));
    }
  });
}

/**
 * Handle OAuth token endpoint
 * POST /oauth/token
 */
export function handleOAuthToken(req, res) {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    try {
      const data = new URLSearchParams(body);
      const code = data.get("code");
      const redirectUri = data.get("redirect_uri");
      const clientId = data.get("client_id");
      // PKCE code_verifier - extracted but not validated yet (local bridge, trusted)
      data.get("code_verifier");
      const grantType = data.get("grant_type");

      res.setHeader("Content-Type", "application/json");

      if (grantType !== "authorization_code") {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "unsupported_grant_type" }));
        return;
      }

      const pendingData = pendingCodes.get(code);
      if (!pendingData) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "invalid_grant", error_description: "Invalid or expired code" }));
        return;
      }

      // Check code age (10 minute expiry)
      if (Date.now() - pendingData.createdAt > 600000) {
        pendingCodes.delete(code);
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "invalid_grant", error_description: "Code expired" }));
        return;
      }

      // Validate client_id matches the one used in authorize request
      // This prevents code leakage from being used by a different client
      if (pendingData.clientId !== clientId) {
        pendingCodes.delete(code);
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "invalid_client", error_description: "client_id mismatch" }));
        return;
      }

      // Validate redirect_uri matches the one used in authorize request
      // This prevents code interception from being used on a different redirect URI
      if (pendingData.redirectUri !== redirectUri) {
        pendingCodes.delete(code);
        res.statusCode = 400;
        res.end(JSON.stringify({ error: "invalid_grant", error_description: "redirect_uri mismatch" }));
        return;
      }

      // TODO: Validate PKCE code_verifier against code_challenge
      // For now, skip validation (local bridge, trusted)

      // Clean up expired tokens periodically to prevent unbounded growth
      cleanupExpiredTokens();

      // Generate access token
      const accessToken = generateToken();
      tokenStore.set(accessToken, {
        sessionId: pendingData.sessionId,
        backendUrl: pendingData.backendUrl,
        createdAt: Date.now(),
      });

      // Clean up pending code
      pendingCodes.delete(code);

      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({
        access_token: accessToken,
        token_type: "Bearer",
        expires_in: 86400, // 24 hours
        scope: "mcp",
      }));
    } catch (err) {
      debugLog("[ctxce] /oauth/token error: " + String(err));
      res.statusCode = 400;
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "invalid_request" }));
    }
  });
}

/**
 * Validate Bearer token and return session info
 * @param {string} token - Bearer token
 * @returns {{sessionId: string, backendUrl: string} | null}
 */
export function validateBearerToken(token) {
  const tokenData = tokenStore.get(token);
  if (!tokenData) {
    return null;
  }

  // Check token age (24 hour expiry)
  const tokenAge = Date.now() - tokenData.createdAt;
  if (tokenAge > 86400000) {
    tokenStore.delete(token);
    return null;
  }

  return {
    sessionId: tokenData.sessionId,
    backendUrl: tokenData.backendUrl,
  };
}

/**
 * Check if a given pathname is an OAuth endpoint
 * @param {string} pathname - URL pathname
 * @returns {boolean}
 */
export function isOAuthEndpoint(pathname) {
  return (
    pathname === "/.well-known/oauth-authorization-server" ||
    pathname === "/oauth/register" ||
    pathname === "/oauth/authorize" ||
    pathname === "/oauth/store-session" ||
    pathname === "/oauth/token"
  );
}
