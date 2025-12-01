import { describe, it, expect, vi, beforeEach } from 'vitest';

// Use vi.hoisted to define mocks that will be available when vi.mock is hoisted
const { mockGet, mockPost, mockPut, mockDelete } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
  mockPut: vi.fn(),
  mockDelete: vi.fn(),
}));

// Mock axios
vi.mock('axios', () => ({
  default: {
    create: () => ({
      get: mockGet,
      post: mockPost,
      put: mockPut,
      delete: mockDelete,
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    }),
  },
}));

// Import after mock setup
import { graphApi } from '../api';

describe('graphApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGet.mockResolvedValue({ data: {} });
    mockPost.mockResolvedValue({ data: {} });
    mockPut.mockResolvedValue({ data: {} });
    mockDelete.mockResolvedValue({ data: {} });
  });

  describe('Graph endpoints', () => {
    it('getFullGraph calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: { nodes: [], edges: [] } });

      await graphApi.getFullGraph();
      expect(mockGet).toHaveBeenCalledWith('/graph/full');
    });
  });

  describe('Node endpoints', () => {
    it('createObservation calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { id: '123' } });

      const observationData = { text: 'Test', confidence: 0.8 };
      await graphApi.createObservation(observationData);

      expect(mockPost).toHaveBeenCalledWith('/nodes/observations', observationData);
    });

    it('createEntity calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { id: '123' } });

      const entityData = { name: 'Test Entity', entity_type: 'person' };
      await graphApi.createEntity(entityData);

      expect(mockPost).toHaveBeenCalledWith('/nodes/entities', entityData);
    });

    it('createSource calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { id: '123' } });

      const sourceData = { title: 'Test Source', source_type: 'paper' };
      await graphApi.createSource(sourceData);

      expect(mockPost).toHaveBeenCalledWith('/nodes/sources', sourceData);
    });

    it('createHypothesis calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { id: '123' } });

      const hypothesisData = { name: 'Test Hypothesis', claim: 'Test claim' };
      await graphApi.createHypothesis(hypothesisData);

      expect(mockPost).toHaveBeenCalledWith('/nodes/hypotheses', hypothesisData);
    });

    it('createConcept calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { id: '123' } });

      const conceptData = { name: 'Test Concept', domain: 'science' };
      await graphApi.createConcept(conceptData);

      expect(mockPost).toHaveBeenCalledWith('/nodes/concepts', conceptData);
    });

    it('getNode calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: { id: 'node-1' } });

      await graphApi.getNode('node-1');
      expect(mockGet).toHaveBeenCalledWith('/nodes/node-1');
    });

    it('deleteNode calls correct endpoint', async () => {
      mockDelete.mockResolvedValue({ data: { message: 'deleted' } });

      await graphApi.deleteNode('node-1');
      expect(mockDelete).toHaveBeenCalledWith('/nodes/node-1');
    });
  });

  describe('Relationship endpoints', () => {
    it('createRelationship calls correct endpoint with data', async () => {
      mockPost.mockResolvedValue({ data: { message: 'created' } });

      await graphApi.createRelationship('node-1', 'node-2', 'SUPPORTS', {
        confidence: 0.9,
        notes: 'Test notes',
      });

      expect(mockPost).toHaveBeenCalledWith('/nodes/relationships', {
        from_id: 'node-1',
        to_id: 'node-2',
        relationship_type: 'SUPPORTS',
        confidence: 0.9,
        notes: 'Test notes',
        inverse_relationship_type: undefined,
        inverse_confidence: undefined,
        inverse_notes: undefined,
      });
    });

    it('getRelationship calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: { id: 'rel-1' } });

      await graphApi.getRelationship('rel-1');
      expect(mockGet).toHaveBeenCalledWith('/nodes/relationships/rel-1');
    });

    it('updateRelationship calls correct endpoint', async () => {
      mockPut.mockResolvedValue({ data: { message: 'updated' } });

      await graphApi.updateRelationship('rel-1', { confidence: 0.95 });
      expect(mockPut).toHaveBeenCalledWith('/nodes/relationships/rel-1', { confidence: 0.95 });
    });

    it('deleteRelationship calls correct endpoint', async () => {
      mockDelete.mockResolvedValue({ data: { message: 'deleted' } });

      await graphApi.deleteRelationship('rel-1');
      expect(mockDelete).toHaveBeenCalledWith('/nodes/relationships/rel-1');
    });
  });

  describe('Settings endpoints', () => {
    it('getSettings calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: { theme: 'dark' } });

      await graphApi.getSettings();
      expect(mockGet).toHaveBeenCalledWith('/settings');
    });

    it('updateSettings calls correct endpoint', async () => {
      mockPut.mockResolvedValue({ data: { theme: 'light' } });

      await graphApi.updateSettings({ theme: 'light' });
      expect(mockPut).toHaveBeenCalledWith('/settings', { theme: 'light' });
    });
  });

  describe('Activity endpoints', () => {
    it('getActivities calls correct endpoint without filter', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities();
      expect(mockGet).toHaveBeenCalledWith('/activities', { params: {} });
    });

    it('getActivities calls with type filter', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({
        types: ['node_created', 'relationship_created'],
      });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: { types: 'node_created,relationship_created' },
      });
    });

    it('getActivities calls with status filter', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({ status: 'pending' });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: { status: 'pending' },
      });
    });

    it('getActivities calls with node_id filter', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({ node_id: 'node-1' });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: { node_id: 'node-1' },
      });
    });

    it('getActivities calls with limit', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({ limit: 20 });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: { limit: 20 },
      });
    });

    it('getActivities calls with include_dismissed', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({ include_dismissed: true });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: { include_dismissed: true },
      });
    });

    it('getActivities calls with combined filters', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getActivities({
        types: ['relationship_suggested'],
        status: 'pending',
        limit: 10,
      });

      expect(mockGet).toHaveBeenCalledWith('/activities', {
        params: {
          types: 'relationship_suggested',
          status: 'pending',
          limit: 10,
        },
      });
    });

    it('getActivity calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: { id: 'activity-1' } });

      await graphApi.getActivity('activity-1');
      expect(mockGet).toHaveBeenCalledWith('/activities/activity-1');
    });

    it('getPendingSuggestions calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getPendingSuggestions();
      expect(mockGet).toHaveBeenCalledWith('/activities/pending', { params: { limit: 20 } });
    });

    it('getPendingSuggestions respects custom limit', async () => {
      mockGet.mockResolvedValue({ data: [] });

      await graphApi.getPendingSuggestions(50);
      expect(mockGet).toHaveBeenCalledWith('/activities/pending', { params: { limit: 50 } });
    });

    it('getProcessingStatus calls correct endpoint', async () => {
      mockGet.mockResolvedValue({ data: null });

      await graphApi.getProcessingStatus('node-1');
      expect(mockGet).toHaveBeenCalledWith('/activities/processing/node-1');
    });

    it('approveSuggestion calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: { message: 'Approved', relationship_id: 'rel-1' },
      });

      await graphApi.approveSuggestion('activity-1');
      expect(mockPost).toHaveBeenCalledWith('/activities/activity-1/approve');
    });

    it('rejectSuggestion calls correct endpoint without feedback', async () => {
      mockPost.mockResolvedValue({
        data: { message: 'Rejected', feedback_stored: false },
      });

      await graphApi.rejectSuggestion('activity-1');
      expect(mockPost).toHaveBeenCalledWith('/activities/activity-1/reject', null, {
        params: undefined,
      });
    });

    it('rejectSuggestion calls correct endpoint with feedback', async () => {
      mockPost.mockResolvedValue({
        data: { message: 'Rejected', feedback_stored: true },
      });

      await graphApi.rejectSuggestion('activity-1', 'Not relevant');
      expect(mockPost).toHaveBeenCalledWith('/activities/activity-1/reject', null, {
        params: { feedback: 'Not relevant' },
      });
    });
  });
});
