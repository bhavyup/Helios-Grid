import { API_BASE_URL } from "@/lib/api-base";

type AuthPayload = { access_token?: string };

export type AuthCredentials = {
  email: string;
  password: string;
};

type AuthResponse = {
  ok: boolean;
  status: number;
  payload: AuthPayload | Record<string, unknown>;
};

export const ACCESS_TOKEN_KEY = "helios.accessToken";
const DEV_AUTO_AUTH = process.env.NEXT_PUBLIC_DEV_AUTH_AUTO === "true";
const DEFAULT_EMAIL = process.env.NEXT_PUBLIC_DEV_AUTH_EMAIL ?? "dev@helios.local";
const DEFAULT_PASSWORD = process.env.NEXT_PUBLIC_DEV_AUTH_PASSWORD ?? "dev-pass-123";

let authPromise: Promise<string | null> | null = null;
let serverTokenPromise: Promise<string | null> | null = null;

const isBrowser = () => typeof window !== "undefined";

const getStoredToken = () => {
  if (!isBrowser()) {
    return null;
  }

  try {
    return window.localStorage.getItem(ACCESS_TOKEN_KEY);
  } catch {
    return null;
  }
};

const setStoredToken = (token: string) => {
  if (!isBrowser()) {
    return;
  }

  try {
    window.localStorage.setItem(ACCESS_TOKEN_KEY, token);
  } catch {
    // Ignore storage errors in restricted browser contexts.
  }
};

const clearStoredToken = () => {
  if (!isBrowser()) {
    return;
  }

  try {
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  } catch {
    // Ignore storage errors in restricted browser contexts.
  }
};

const postAuth = async (path: string, body: Record<string, unknown>): Promise<AuthResponse> => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const payload = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, payload };
};

const resolveTokenFromResponse = (response: AuthResponse) => {
  return (response.payload as AuthPayload).access_token ?? null;
};

export const getStoredAuthToken = () => getStoredToken();

export const setAuthToken = (token: string) => {
  setStoredToken(token);
};

export const clearAuthToken = () => {
  clearStoredToken();
};

export const loginWithCredentials = async (credentials: AuthCredentials) => {
  const login = await postAuth("/auth/login", credentials);
  if (!login.ok) {
    return null;
  }

  const token = resolveTokenFromResponse(login);
  if (token) {
    setStoredToken(token);
  }
  return token;
};

export const bootstrapDemoAuth = async () => {
  const credentials = { email: DEFAULT_EMAIL, password: DEFAULT_PASSWORD };
  const register = await postAuth("/auth/register", credentials);

  if (register.ok) {
    const token = resolveTokenFromResponse(register);
    if (token) {
      setStoredToken(token);
    }
    return token;
  }

  if (register.status === 409) {
    return loginWithCredentials(credentials);
  }

  return null;
};

const fetchAuthToken = async (): Promise<string | null> => {
  if (!DEV_AUTO_AUTH || !isBrowser()) {
    return null;
  }

  return bootstrapDemoAuth();
};

export const getAuthHeader = async (options?: { forceRefresh?: boolean }) => {
  // Client-side: use stored token or lazily run dev auto-auth flow.
  if (isBrowser()) {
    if (!options?.forceRefresh) {
      const existing = getStoredToken();
      if (existing) {
        return { Authorization: `Bearer ${existing}` };
      }
    }

    if (!DEV_AUTO_AUTH) {
      return {};
    }

    if (!authPromise) {
      authPromise = fetchAuthToken().finally(() => {
        authPromise = null;
      });
    }

    const token = await authPromise;
    if (!token) {
      return {};
    }

    return { Authorization: `Bearer ${token}` };
  }

  // Server-side (Node/SSR): perform a one-time server fetch to create/login dev user
  // and memoize the token so SSR requests include Authorization header.
  if (!DEV_AUTO_AUTH) {
    return {};
  }

  if (!serverTokenPromise) {
    serverTokenPromise = (async () => {
      try {
        const credentials = { email: DEFAULT_EMAIL, password: DEFAULT_PASSWORD };
        // Server-side: prefer calling the backend service directly when available
        // (HELIOS_BACKEND_URL is injected into the container). Fall back to
        // `API_BASE_URL` if not set.
        const serverApi = process.env.HELIOS_BACKEND_URL ?? API_BASE_URL;

        // Try register first, fall back to login on conflict.
        const reg = await fetch(`${serverApi}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(credentials),
        });

        if (reg.ok) {
          const payload = await reg.json().catch(() => ({}));
          return (payload as AuthPayload).access_token ?? null;
        }

        if (reg.status === 409) {
          const login = await fetch(`${serverApi}/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(credentials),
          });
          if (login.ok) {
            const payload = await login.json().catch(() => ({}));
            return (payload as AuthPayload).access_token ?? null;
          }
        }
      } catch {
        // ignore
      }
      return null;
    })();
  }

  const serverToken = await serverTokenPromise;
  if (!serverToken) return {};
  return { Authorization: `Bearer ${serverToken}` };
};
