import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';
import { configure } from '@testing-library/react';

// Configure canvas for jsdom environment
// This enables canvas support for Cytoscape in tests
if (typeof window !== 'undefined') {
  const originalGetContext = window.HTMLCanvasElement.prototype.getContext;
  if (originalGetContext) {
    window.HTMLCanvasElement.prototype.getContext = function (type, options) {
      // Allow 2d context for canvas
      if (type === '2d') {
        return originalGetContext.call(this, type, options);
      }
      return originalGetContext.call(this, type, options);
    };
  }
}

// Configure testing library
configure({
  // Increase timeout for async operations
  asyncUtilTimeout: 5000,
});

// Cleanup after each test
afterEach(() => {
  cleanup();
});
