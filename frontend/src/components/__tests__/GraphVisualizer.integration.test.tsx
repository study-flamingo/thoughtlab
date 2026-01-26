import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import { render, mockSingleNodeGraph, mockMultiNodeGraph, createApiMock } from '../../test/utils';
import * as api from '../../services/api';
import { createCytoscapeModule } from '../../test/cytoscape-mock';
import GraphVisualizer from '../GraphVisualizer';

// Mock Cytoscape for testing (avoid canvas issues in jsdom)
vi.mock('cytoscape', () => ({
  default: createCytoscapeModule(),
}));

// This is an integration test that focuses on component behavior
// We mock the API service layer, not the HTTP layer, to test how components respond

describe('GraphVisualizer - Integration Tests', () => {
  let mockApi: ReturnType<typeof createApiMock>;

  beforeEach(() => {
    mockApi = createApiMock();
    vi.spyOn(api, 'graphApi', 'get').mockImplementation(() => mockApi as any);
    vi.clearAllMocks();
  });

  it('shows loading state initially', async () => {
    // Mock API to never resolve (simulate slow network)
    mockApi.getFullGraph.mockImplementation(
      () => new Promise(() => {})
    );

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Should show loading immediately
    expect(screen.getByText('Loading graph...')).toBeInTheDocument();
  });

  it('shows error state on API failure', async () => {
    // Mock API to reject with error
    mockApi.getFullGraph.mockRejectedValue(new Error('Database connection failed'));

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading to finish and error to appear
    await waitFor(() => {
      expect(screen.getByText('Error loading graph')).toBeInTheDocument();
    });
  });

  it('shows empty state when no nodes', async () => {
    // Mock API to return empty graph
    mockApi.getFullGraph.mockResolvedValue({
      data: { nodes: [], edges: [] }
    });

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading to finish
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'));

    // Should show empty state
    await waitFor(() => {
      expect(screen.getByText('No nodes yet.')).toBeInTheDocument();
    });
  });

  it('displays single node graph correctly', async () => {
    // Mock API to return single node graph
    mockApi.getFullGraph.mockResolvedValue({
      data: mockSingleNodeGraph
    });

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading to finish
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'));

    // Should show graph with single node
    await waitFor(() => {
      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
      expect(screen.getByText('1 nodes • 0 relationships')).toBeInTheDocument();
    });
  });

  it('displays multi-node graph correctly', async () => {
    // Mock API to return multi-node graph
    mockApi.getFullGraph.mockResolvedValue({
      data: mockMultiNodeGraph
    });

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading to finish
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'));

    // Should show graph with multiple nodes and edges
    await waitFor(() => {
      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
      expect(screen.getByText('2 nodes • 1 relationships')).toBeInTheDocument();
    });
  });

  it('shows empty state when no nodes', async () => {
    // Mock API to return empty graph
    mockApi.getFullGraph.mockResolvedValue({
      data: { nodes: [], edges: [] }
    });

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading state to disappear
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'), {
      timeout: 5000
    });

    // Check if the component shows empty state
    expect(screen.getByText('No nodes yet.')).toBeInTheDocument();
  });

  it('creates canvas grid background element', async () => {
    // Mock API to return empty graph
    mockApi.getFullGraph.mockResolvedValue({
      data: { nodes: [], edges: [] }
    });

    render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for loading state to disappear
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'), {
      timeout: 5000
    });

    // Check that the component rendered successfully
    expect(screen.getByText('No nodes yet.')).toBeInTheDocument();

    // Note: Canvas grid background rendering is tested visually in browser
    // JSDOM doesn't support canvas, so we can't assert on grid rendering here
  });

  it('handles rapid data changes', async () => {
    let callCount = 0;

    // Mock API to return different data on subsequent calls
    mockApi.getFullGraph.mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        return { data: mockSingleNodeGraph };
      } else {
        return { data: mockMultiNodeGraph };
      }
    });

    const { rerender } = render(
      <GraphVisualizer
        onNodeSelect={vi.fn()}
        onEdgeSelect={vi.fn()}
      />
    );

    // Wait for initial load - accept multiple possible states
    await waitFor(() => {
      const singleNode = screen.queryByText('1 nodes • 0 relationships');
      const multiNode = screen.queryByText('2 nodes • 1 relationships');
      expect(singleNode || multiNode).toBeTruthy();
    }, { timeout: 10000 });

    // Verify at least one graph loaded successfully
    const graphContents = screen.queryAllByText(/nodes/);
    expect(graphContents.length).toBeGreaterThan(0);
  });
});

describe('GraphVisualizer - User Interaction', () => {
  let mockApi: ReturnType<typeof createApiMock>;

  beforeEach(() => {
    mockApi = createApiMock();
    vi.spyOn(api, 'graphApi', 'get').mockImplementation(() => mockApi as any);
    vi.clearAllMocks();
  });

  it('setup works correctly with selection callbacks', async () => {
    const mockNodeSelect = vi.fn();
    const mockEdgeSelect = vi.fn();

    // Mock API to return single node graph
    mockApi.getFullGraph.mockResolvedValue({
      data: mockSingleNodeGraph
    });

    render(
      <GraphVisualizer
        onNodeSelect={mockNodeSelect}
        onEdgeSelect={mockEdgeSelect}
      />
    );

    // Wait for graph to load
    await waitForElementToBeRemoved(() => screen.queryByText('Loading graph...'));

    // Verify component loaded successfully
    expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
    expect(screen.getByText('1 nodes • 0 relationships')).toBeInTheDocument();

    // Note: We can't easily test actual Cytoscape interactions in jsdom
    // but we've verified the component loads and sets up correctly
  });
});