import axios from 'axios';

const AUTH_TOKEN_KEY = 'thoughtlab_token';

// Create a separate axios instance for auth requests (no auth header needed)
const authClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

interface AuthStatus {
  enabled: boolean;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

/**
 * Get the stored auth token from localStorage
 */
export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Store the auth token in localStorage
 */
export function setToken(token: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

/**
 * Remove the auth token from localStorage
 */
export function removeToken(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

/**
 * Check if user is authenticated (has a token)
 */
export function isAuthenticated(): boolean {
  return !!getToken();
}

/**
 * Get the authorization header for API requests
 */
export function getAuthHeader(): { Authorization: string } | undefined {
  const token = getToken();
  if (!token) return undefined;
  return { Authorization: `Bearer ${token}` };
}

export const authApi = {
  /**
   * Login with password
   */
  async login(password: string): Promise<TokenResponse> {
    // Use JSON endpoint for easier frontend integration
    const response = await authClient.post<TokenResponse>('/auth/login/json', {
      password,
    });
    
    // Store the token
    if (response.data.access_token) {
      setToken(response.data.access_token);
    }
    
    return response.data;
  },

  /**
   * Check if authentication is enabled on the server
   */
  async getStatus(): Promise<AuthStatus> {
    const response = await authClient.get<AuthStatus>('/auth/status');
    return response.data;
  },

  /**
   * Verify the current token is valid
   */
  async verifyToken(): Promise<boolean> {
    const token = getToken();
    if (!token) return false;

    try {
      await authClient.get('/auth/verify', {
        headers: { Authorization: `Bearer ${token}` },
      });
      return true;
    } catch {
      return false;
    }
  },

  /**
   * Logout - remove the token
   */
  logout(): void {
    removeToken();
  },
};

/**
 * Initialize auth interceptor for the main API client
 * This should be called after creating the api instance
 */
export function initializeAuthInterceptor(api: typeof import('./api').default): void {
  // Add request interceptor to include auth token
  api.interceptors.request.use(
    (config) => {
      const token = getToken();
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  // Add response interceptor to handle 401 errors
  api.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Token expired or invalid - remove it and reload to trigger login
        removeToken();
        
        // Only reload if we're not already on the login page
        // (prevents infinite reload loop)
        if (window.location.pathname !== '/') {
          window.location.reload();
        }
      }
      return Promise.reject(error);
    }
  );
}
