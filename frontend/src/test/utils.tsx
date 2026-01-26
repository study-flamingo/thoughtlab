import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi } from 'vitest';

// Default test data
export const mockGraphData = {
  nodes: [],
  edges: [],
};

export const mockSingleNodeGraph = {
  nodes: [
    {
      id: 'test-node-1',
      text: 'Test Observation',
      type: 'Observation' as const,
      confidence: 0.8,
      created_at: '2024-01-01T00:00:00Z',
    },
  ],
  edges: [],
};

export const mockMultiNodeGraph = {
  nodes: [
    {
      id: 'node-1',
      text: 'First Observation',
      type: 'Observation' as const,
      confidence: 0.9,
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'node-2',
      text: 'Test Hypothesis',
      type: 'Hypothesis' as const,
      confidence: 0.7,
      created_at: '2024-01-02T00:00:00Z',
    },
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      type: 'SUPPORTS' as const,
    },
  ],
};

// API mock helpers
export const createApiMock = () => ({
  getFullGraph: vi.fn(),
  createObservation: vi.fn(),
  createEntity: vi.fn(),
  createSource: vi.fn(),
  createHypothesis: vi.fn(),
  createConcept: vi.fn(),
  getNode: vi.fn(),
  deleteNode: vi.fn(),
  createRelationship: vi.fn(),
  getRelationship: vi.fn(),
  updateRelationship: vi.fn(),
  deleteRelationship: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
  getActivities: vi.fn(),
  getActivity: vi.fn(),
  getPendingSuggestions: vi.fn(),
  getProcessingStatus: vi.fn(),
  approveSuggestion: vi.fn(),
  rejectSuggestion: vi.fn(),
  findRelatedNodes: vi.fn(),
  summarizeNode: vi.fn(),
  summarizeNodeWithContext: vi.fn(),
  recalculateNodeConfidence: vi.fn(),
  reclassifyNode: vi.fn(),
  searchWebEvidence: vi.fn(),
  mergeNodes: vi.fn(),
  summarizeRelationship: vi.fn(),
  recalculateEdgeConfidence: vi.fn(),
  reclassifyRelationship: vi.fn(),
});

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  queryClient?: QueryClient;
}

export function renderWithProviders(
  ui: ReactElement,
  { queryClient, ...renderOptions }: CustomRenderOptions = {}
) {
  const testQueryClient = queryClient || createTestQueryClient();

  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={testQueryClient}>
        {children}
      </QueryClientProvider>
    );
  }

  return {
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
    queryClient: testQueryClient,
  };
}

export * from '@testing-library/react';
export { renderWithProviders as render };
