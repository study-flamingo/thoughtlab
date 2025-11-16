import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../test/utils';
import GraphVisualizer from '../GraphVisualizer';
import { graphApi } from '../../services/api';

// Mock the API
vi.mock('../../services/api', () => ({
  graphApi: {
    getFullGraph: vi.fn(),
  },
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

  it('displays graph when data is available', async () => {
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
      expect(screen.getByText('2 nodes • 1 relationships')).toBeInTheDocument();
    });
  });
});
