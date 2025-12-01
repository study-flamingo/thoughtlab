import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../test/utils';
import GraphVisualizer from '../GraphVisualizer';
import { graphApi } from '../../services/api';

// Mock the API
vi.mock('../../services/api', () => ({
  graphApi: {
    getFullGraph: vi.fn(),
    getSettings: vi.fn().mockResolvedValue({
      data: {
        id: 'app',
        theme: 'light',
        show_edge_labels: true,
        default_relation_confidence: 0.8,
        layout_name: 'cose',
        animate_layout: false,
        node_colors: {
          Observation: '#60A5FA',
          Hypothesis: '#34D399',
          Source: '#FBBF24',
          Concept: '#A78BFA',
          Entity: '#F87171',
        },
        relation_styles: {},
      },
    }),
  },
}));

// Create a mock cytoscape instance factory
const createMockCy = () => ({
  on: vi.fn(),
  off: vi.fn(),
  layout: vi.fn(() => ({ run: vi.fn() })),
  resize: vi.fn(),
  fit: vi.fn(),
  style: vi.fn(() => ({ update: vi.fn() })),
  nodes: vi.fn(() => ({ length: 0, forEach: vi.fn() })),
  edges: vi.fn(() => ({ length: 0, forEach: vi.fn() })),
  destroyed: vi.fn(() => false),
  destroy: vi.fn(),
  getElementById: vi.fn(() => ({
    select: vi.fn(),
    unselect: vi.fn(),
  })),
  elements: vi.fn(() => ({
    unselect: vi.fn(),
  })),
});

// Mock react-cytoscapejs to avoid canvas issues in jsdom
vi.mock('react-cytoscapejs', () => ({
  default: vi.fn(({ cy: cyCallback }) => {
    // Call cy callback with mock cytoscape instance if provided
    if (cyCallback) {
      cyCallback(createMockCy());
    }
    return <div data-testid="cytoscape-mock">Cytoscape Graph</div>;
  }),
}));

describe('GraphVisualizer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state initially', () => {
    (graphApi.getFullGraph as any).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<GraphVisualizer />);
    expect(screen.getByText('Loading graph...')).toBeInTheDocument();
  });

  it('shows error state on API failure', async () => {
    (graphApi.getFullGraph as any).mockRejectedValue(new Error('API Error'));

    render(<GraphVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('Error loading graph')).toBeInTheDocument();
    });
  });

  it('shows empty state when no nodes', async () => {
    (graphApi.getFullGraph as any).mockResolvedValue({
      data: { nodes: [], edges: [] },
    });

    render(<GraphVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('No nodes yet.')).toBeInTheDocument();
    });
  });

  it('displays graph header when data is available', async () => {
    const mockData = {
      nodes: [
        {
          id: '1',
          text: 'Test observation 1',
          confidence: 0.8,
          type: 'Observation',
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: '2',
          text: 'Test observation 2',
          confidence: 0.9,
          type: 'Observation',
          created_at: '2024-01-01T00:00:00Z',
        },
      ],
      edges: [
        {
          id: 'edge1',
          source: '1',
          target: '2',
          type: 'RELATES_TO',
        },
      ],
    };

    (graphApi.getFullGraph as any).mockResolvedValue({
      data: mockData,
    });

    render(<GraphVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('Knowledge Graph')).toBeInTheDocument();
    });
  });

  it('displays node and edge count', async () => {
    const mockData = {
      nodes: [
        { id: '1', text: 'Node 1', type: 'Observation', created_at: '2024-01-01T00:00:00Z' },
        { id: '2', text: 'Node 2', type: 'Observation', created_at: '2024-01-01T00:00:00Z' },
        { id: '3', text: 'Node 3', type: 'Hypothesis', created_at: '2024-01-01T00:00:00Z' },
      ],
      edges: [
        { id: 'e1', source: '1', target: '2', type: 'RELATES_TO' },
        { id: 'e2', source: '2', target: '3', type: 'SUPPORTS' },
      ],
    };

    (graphApi.getFullGraph as any).mockResolvedValue({
      data: mockData,
    });

    render(<GraphVisualizer />);

    await waitFor(() => {
      expect(screen.getByText('3 nodes • 2 relationships')).toBeInTheDocument();
    });
  });

  it('renders cytoscape component when data is loaded', async () => {
    (graphApi.getFullGraph as any).mockResolvedValue({
      data: {
        nodes: [{ id: '1', text: 'Test', type: 'Observation', created_at: '2024-01-01T00:00:00Z' }],
        edges: [],
      },
    });

    render(<GraphVisualizer />);

    await waitFor(() => {
      expect(screen.getByTestId('cytoscape-mock')).toBeInTheDocument();
    });
  });
});
