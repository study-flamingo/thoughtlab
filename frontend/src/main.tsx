import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState, useEffect, useCallback } from 'react';
import App from './App';
import LoginPage from './components/LoginPage';
import './index.css';
import { registerCytoscapeExtensions } from './lib/cytoscape-extensions';
import { isAuthenticated, authApi } from './services/auth';

// Register Cytoscape extensions once at app startup
registerCytoscapeExtensions();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Auth wrapper component that handles authentication state
 */
function AuthWrapper() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  const checkAuth = useCallback(async () => {
    // First check if we have a token
    if (!isAuthenticated()) {
      // No token, need to login
      setIsChecking(false);
      return;
    }

    // Verify token is still valid
    try {
      const isValid = await authApi.verifyToken();
      if (isValid) {
        setIsLoggedIn(true);
      }
    } catch {
      // Token invalid, will stay on login page
    } finally {
      setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  const handleLogin = useCallback(() => {
    setIsLoggedIn(true);
  }, []);

  if (isChecking) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-600 dark:text-gray-400">Loading...</span>
        </div>
      </div>
    );
  }

  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return <App />;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <AuthWrapper />
  </QueryClientProvider>
);
