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

  describe('AI Tools - Node Operations', () => {
    it('findRelatedNodes calls correct endpoint with defaults', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, related_nodes: [], message: 'Found 0 related nodes' },
      });

      await graphApi.findRelatedNodes('node-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/find-related', {});
    });

    it('findRelatedNodes calls with custom options', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, related_nodes: [], message: 'Found 0 related nodes' },
      });

      await graphApi.findRelatedNodes('node-1', { limit: 5, min_similarity: 0.7 });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/find-related', {
        limit: 5,
        min_similarity: 0.7,
      });
    });

    it('summarizeNode calls correct endpoint with defaults', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, summary: 'Test summary', key_points: [], word_count: 10 },
      });

      await graphApi.summarizeNode('node-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/summarize', {});
    });

    it('summarizeNode calls with custom options', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, summary: 'Test summary', key_points: [], word_count: 10 },
      });

      await graphApi.summarizeNode('node-1', { max_length: 100, style: 'bullet_points' });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/summarize', {
        max_length: 100,
        style: 'bullet_points',
      });
    });

    it('summarizeNodeWithContext calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          summary: 'Test summary',
          context: { supports: [], contradicts: [], related: [] },
          synthesis: 'Test synthesis',
          relationship_count: 0,
        },
      });

      await graphApi.summarizeNodeWithContext('node-1', { depth: 2 });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/summarize-with-context', {
        depth: 2,
      });
    });

    it('recalculateNodeConfidence calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          old_confidence: 0.7,
          new_confidence: 0.85,
          reasoning: 'Test reasoning',
          factors: [],
        },
      });

      await graphApi.recalculateNodeConfidence('node-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/recalculate-confidence', {});
    });

    it('recalculateNodeConfidence calls with options', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          old_confidence: 0.7,
          new_confidence: 0.85,
          reasoning: 'Test reasoning',
          factors: [],
        },
      });

      await graphApi.recalculateNodeConfidence('node-1', { factor_in_relationships: false });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/recalculate-confidence', {
        factor_in_relationships: false,
      });
    });

    it('reclassifyNode calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          old_type: 'Observation',
          new_type: 'Hypothesis',
          properties_preserved: ['text'],
          relationships_preserved: 2,
          message: 'Reclassified',
        },
      });

      await graphApi.reclassifyNode('node-1', { new_type: 'Hypothesis', preserve_relationships: true });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/reclassify', {
        new_type: 'Hypothesis',
        preserve_relationships: true,
      });
    });

    it('searchWebEvidence calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: false,
          results: [],
          message: 'Web search not configured',
          error: 'TAVILY_API_KEY not set',
        },
      });

      await graphApi.searchWebEvidence('node-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/search-web-evidence', {});
    });

    it('searchWebEvidence calls with options', async () => {
      mockPost.mockResolvedValue({
        data: { success: false, results: [], message: 'Web search not configured' },
      });

      await graphApi.searchWebEvidence('node-1', { evidence_type: 'supporting', max_results: 10 });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/node-1/search-web-evidence', {
        evidence_type: 'supporting',
        max_results: 10,
      });
    });

    it('mergeNodes calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          primary_node_id: 'node-1',
          secondary_node_id: 'node-2',
          merged_properties: ['text'],
          relationships_transferred: 3,
          message: 'Nodes merged',
        },
      });

      await graphApi.mergeNodes({
        primary_node_id: 'node-1',
        secondary_node_id: 'node-2',
        merge_strategy: 'combine',
      });
      expect(mockPost).toHaveBeenCalledWith('/tools/nodes/merge', {
        primary_node_id: 'node-1',
        secondary_node_id: 'node-2',
        merge_strategy: 'combine',
      });
    });
  });

  describe('AI Tools - Relationship Operations', () => {
    it('summarizeRelationship calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          edge_id: 'rel-1',
          from_node: { id: 'node-1', type: 'Observation', content: 'Test' },
          to_node: { id: 'node-2', type: 'Hypothesis', content: 'Test' },
          relationship_type: 'SUPPORTS',
          summary: 'Test summary',
          evidence: [],
          strength_assessment: 'strong',
        },
      });

      await graphApi.summarizeRelationship('rel-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/summarize', {});
    });

    it('summarizeRelationship calls with options', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, summary: 'Test summary' },
      });

      await graphApi.summarizeRelationship('rel-1', { include_evidence: false });
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/summarize', {
        include_evidence: false,
      });
    });

    it('recalculateEdgeConfidence calls correct endpoint', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          edge_id: 'rel-1',
          old_confidence: 0.6,
          new_confidence: 0.8,
          reasoning: 'Test reasoning',
          factors: [],
        },
      });

      await graphApi.recalculateEdgeConfidence('rel-1');
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/recalculate-confidence', {});
    });

    it('recalculateEdgeConfidence calls with options', async () => {
      mockPost.mockResolvedValue({
        data: { success: true, old_confidence: 0.6, new_confidence: 0.8 },
      });

      await graphApi.recalculateEdgeConfidence('rel-1', { consider_graph_structure: false });
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/recalculate-confidence', {
        consider_graph_structure: false,
      });
    });

    it('reclassifyRelationship calls correct endpoint with explicit type', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          edge_id: 'rel-1',
          old_type: 'RELATES_TO',
          new_type: 'SUPPORTS',
          suggested_by_ai: false,
          reasoning: 'Manual reclassification',
          notes_preserved: true,
        },
      });

      await graphApi.reclassifyRelationship('rel-1', { new_type: 'SUPPORTS' });
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/reclassify', {
        new_type: 'SUPPORTS',
      });
    });

    it('reclassifyRelationship calls correct endpoint for AI suggestion', async () => {
      mockPost.mockResolvedValue({
        data: {
          success: true,
          old_type: 'RELATES_TO',
          new_type: 'SUPPORTS',
          suggested_by_ai: true,
          reasoning: 'AI determined SUPPORTS is more appropriate',
        },
      });

      await graphApi.reclassifyRelationship('rel-1', { new_type: null });
      expect(mockPost).toHaveBeenCalledWith('/tools/relationships/rel-1/reclassify', {
        new_type: null,
      });
    });
  });
});
