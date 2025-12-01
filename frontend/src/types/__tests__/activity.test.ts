import { describe, it, expect } from 'vitest';
import {
  Activity,
  ActivityType,
  isInteractiveActivity,
  hasNavigation,
  getActivityIcon,
  getActivityColor,
  formatConfidence,
} from '../activity';

describe('Activity type helpers', () => {
  describe('isInteractiveActivity', () => {
    it('returns true for pending relationship suggestion', () => {
      const activity: Activity = {
        id: '1',
        type: 'relationship_suggested',
        message: 'Test',
        created_at: new Date().toISOString(),
        status: 'pending',
      };
      expect(isInteractiveActivity(activity)).toBe(true);
    });

    it('returns false for approved suggestion', () => {
      const activity: Activity = {
        id: '1',
        type: 'relationship_suggested',
        message: 'Test',
        created_at: new Date().toISOString(),
        status: 'approved',
      };
      expect(isInteractiveActivity(activity)).toBe(false);
    });

    it('returns false for rejected suggestion', () => {
      const activity: Activity = {
        id: '1',
        type: 'relationship_suggested',
        message: 'Test',
        created_at: new Date().toISOString(),
        status: 'rejected',
      };
      expect(isInteractiveActivity(activity)).toBe(false);
    });

    it('returns false for node_created (non-interactive type)', () => {
      const activity: Activity = {
        id: '1',
        type: 'node_created',
        message: 'Test',
        created_at: new Date().toISOString(),
      };
      expect(isInteractiveActivity(activity)).toBe(false);
    });

    it('returns false for processing activities', () => {
      const activity: Activity = {
        id: '1',
        type: 'processing_started',
        message: 'Test',
        created_at: new Date().toISOString(),
      };
      expect(isInteractiveActivity(activity)).toBe(false);
    });
  });

  describe('hasNavigation', () => {
    it('returns true when activity has node_id', () => {
      const activity: Activity = {
        id: '1',
        type: 'node_created',
        message: 'Test',
        created_at: new Date().toISOString(),
        node_id: 'node-1',
      };
      expect(hasNavigation(activity)).toBe(true);
    });

    it('returns true when activity has relationship_id', () => {
      const activity: Activity = {
        id: '1',
        type: 'relationship_created',
        message: 'Test',
        created_at: new Date().toISOString(),
        relationship_id: 'rel-1',
      };
      expect(hasNavigation(activity)).toBe(true);
    });

    it('returns true when activity has both node_id and relationship_id', () => {
      const activity: Activity = {
        id: '1',
        type: 'relationship_created',
        message: 'Test',
        created_at: new Date().toISOString(),
        node_id: 'node-1',
        relationship_id: 'rel-1',
      };
      expect(hasNavigation(activity)).toBe(true);
    });

    it('returns false when activity has neither node_id nor relationship_id', () => {
      const activity: Activity = {
        id: '1',
        type: 'info',
        message: 'Test',
        created_at: new Date().toISOString(),
      };
      expect(hasNavigation(activity)).toBe(false);
    });
  });

  describe('getActivityIcon', () => {
    it('returns correct icon for node_created', () => {
      expect(getActivityIcon('node_created')).toBe('➕');
    });

    it('returns correct icon for node_updated', () => {
      expect(getActivityIcon('node_updated')).toBe('✏️');
    });

    it('returns correct icon for node_deleted', () => {
      expect(getActivityIcon('node_deleted')).toBe('🗑️');
    });

    it('returns correct icon for relationship_created', () => {
      expect(getActivityIcon('relationship_created')).toBe('🔗');
    });

    it('returns correct icon for relationship_suggested', () => {
      expect(getActivityIcon('relationship_suggested')).toBe('💡');
    });

    it('returns correct icon for relationship_auto_created', () => {
      expect(getActivityIcon('relationship_auto_created')).toBe('🤖');
    });

    it('returns correct icons for processing stages', () => {
      expect(getActivityIcon('processing_started')).toBe('⏳');
      expect(getActivityIcon('processing_chunking')).toBe('📄');
      expect(getActivityIcon('processing_embedding')).toBe('🧮');
      expect(getActivityIcon('processing_analyzing')).toBe('🔍');
      expect(getActivityIcon('processing_completed')).toBe('✅');
      expect(getActivityIcon('processing_failed')).toBe('❌');
    });

    it('returns correct icons for system activities', () => {
      expect(getActivityIcon('error')).toBe('🚨');
      expect(getActivityIcon('warning')).toBe('⚠️');
      expect(getActivityIcon('info')).toBe('ℹ️');
    });

    it('returns default icon for unknown types', () => {
      // Cast to any to test fallback behavior
      expect(getActivityIcon('unknown_type' as ActivityType)).toBe('📌');
    });
  });

  describe('getActivityColor', () => {
    it('returns green for processing_completed', () => {
      expect(getActivityColor('processing_completed')).toBe('text-green-500');
    });

    it('returns red for processing_failed', () => {
      expect(getActivityColor('processing_failed')).toBe('text-red-500');
    });

    it('returns blue for other processing stages', () => {
      expect(getActivityColor('processing_started')).toBe('text-blue-500');
      expect(getActivityColor('processing_chunking')).toBe('text-blue-500');
      expect(getActivityColor('processing_embedding')).toBe('text-blue-500');
      expect(getActivityColor('processing_analyzing')).toBe('text-blue-500');
    });

    it('returns amber for relationship_suggested', () => {
      expect(getActivityColor('relationship_suggested')).toBe('text-amber-500');
    });

    it('returns purple for relationship_auto_created', () => {
      expect(getActivityColor('relationship_auto_created')).toBe('text-purple-500');
    });

    it('returns red for error', () => {
      expect(getActivityColor('error')).toBe('text-red-500');
    });

    it('returns yellow for warning', () => {
      expect(getActivityColor('warning')).toBe('text-yellow-500');
    });

    it('returns gray for regular activities', () => {
      expect(getActivityColor('node_created')).toBe('text-gray-500');
      expect(getActivityColor('relationship_created')).toBe('text-gray-500');
      expect(getActivityColor('info')).toBe('text-gray-500');
    });
  });

  describe('formatConfidence', () => {
    it('formats confidence as percentage', () => {
      expect(formatConfidence(0.75)).toBe('75%');
      expect(formatConfidence(0.5)).toBe('50%');
      expect(formatConfidence(1.0)).toBe('100%');
      expect(formatConfidence(0)).toBe('0%');
    });

    it('rounds confidence correctly', () => {
      expect(formatConfidence(0.756)).toBe('76%');
      expect(formatConfidence(0.754)).toBe('75%');
      expect(formatConfidence(0.999)).toBe('100%');
    });

    it('handles edge cases', () => {
      expect(formatConfidence(0.001)).toBe('0%');
      expect(formatConfidence(0.005)).toBe('1%');
    });
  });
});

describe('Activity interface', () => {
  it('accepts all required fields', () => {
    const activity: Activity = {
      id: 'test-id',
      type: 'node_created',
      message: 'Test message',
      created_at: '2024-01-01T00:00:00Z',
    };
    expect(activity.id).toBe('test-id');
    expect(activity.type).toBe('node_created');
  });

  it('accepts optional fields', () => {
    const activity: Activity = {
      id: 'test-id',
      type: 'relationship_suggested',
      message: 'Test message',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-02T00:00:00Z',
      node_id: 'node-1',
      node_type: 'Observation',
      relationship_id: 'rel-1',
      status: 'pending',
      created_by: 'system-llm',
      group_id: 'group-1',
      suggestion_data: {
        from_node_id: 'node-1',
        from_node_type: 'Observation',
        from_node_label: 'Node A',
        to_node_id: 'node-2',
        to_node_type: 'Hypothesis',
        to_node_label: 'Node B',
        relationship_type: 'SUPPORTS',
        confidence: 0.8,
        reasoning: 'Test reasoning',
      },
    };
    expect(activity.suggestion_data?.confidence).toBe(0.8);
  });

  it('accepts processing data', () => {
    const activity: Activity = {
      id: 'test-id',
      type: 'processing_embedding',
      message: 'Creating embeddings',
      created_at: '2024-01-01T00:00:00Z',
      processing_data: {
        node_id: 'source-1',
        node_type: 'Source',
        node_label: 'Research Paper',
        stage: 'embedding',
        progress: 0.5,
        chunks_created: 10,
        embeddings_created: 5,
      },
    };
    expect(activity.processing_data?.chunks_created).toBe(10);
  });
});

