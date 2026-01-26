/**
 * Cytoscape Mock for Testing
 *
 * Since Cytoscape is canvas-based and complex to test in jsdom,
 * we provide a mock that simulates the essential API for component testing.
 *
 * This allows us to test component behavior without dealing with canvas limitations.
 */

export const createCytoscapeMock = () => {
  const mockCytoscape = {
    // Core methods
    on: vi.fn(),
    off: vi.fn(),
    destroy: vi.fn(),
    destroyed: vi.fn(() => false),
    container: vi.fn(() => document.createElement('div')),
    resize: vi.fn(),
    fit: vi.fn(),
    zoom: vi.fn(() => 1),
    pan: vi.fn(() => ({ x: 0, y: 0 })),

    // Layout methods
    layout: vi.fn(() => ({
      run: vi.fn(),
    })),

    // Style methods
    style: vi.fn(() => ({
      update: vi.fn(),
    })),

    // Element methods
    nodes: vi.fn(() => ({
      length: 0,
      forEach: vi.fn(),
    })),
    edges: vi.fn(() => ({
      length: 0,
      forEach: vi.fn(),
    })),
    elements: vi.fn(() => ({
      unselect: vi.fn(),
    })),
    getElementById: vi.fn(() => ({
      select: vi.fn(),
      unselect: vi.fn(),
      length: 1,
    })),

    // Configuration methods
    gridGuide: vi.fn(),
    navigator: vi.fn(),
    userPanningEnabled: vi.fn(),
    boxSelectionEnabled: vi.fn(),
    zoomingEnabled: vi.fn(),
    minZoom: vi.fn(),
    maxZoom: vi.fn(),
  };

  return mockCytoscape;
};

// Mock data for testing
export const mockCytoscapeInstance = createCytoscapeMock();

// Mock cytoscape module factory
export const createCytoscapeModule = () => {
  const mockModule = vi.fn(() => mockCytoscapeInstance);
  mockModule.use = vi.fn();
  return mockModule;
};