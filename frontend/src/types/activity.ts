/**
 * Activity Feed Types
 * 
 * These types mirror the backend Activity models for type-safe API communication.
 */

export type ActivityType =
  // Node lifecycle
  | 'node_created'
  | 'node_updated'
  | 'node_deleted'
  // Relationship lifecycle
  | 'relationship_created'
  | 'relationship_updated'
  | 'relationship_deleted'
  // LLM suggestions
  | 'relationship_suggested'
  | 'relationship_auto_created'
  // Processing status
  | 'processing_started'
  | 'processing_chunking'
  | 'processing_embedding'
  | 'processing_analyzing'
  | 'processing_completed'
  | 'processing_failed'
  // System
  | 'error'
  | 'warning'
  | 'info';

export type ActivityStatus = 'pending' | 'approved' | 'rejected' | 'expired';

export interface SuggestionData {
  from_node_id: string;
  from_node_type: string;
  from_node_label: string;
  to_node_id: string;
  to_node_type: string;
  to_node_label: string;
  relationship_type: string;
  confidence: number;
  reasoning?: string;
}

export interface ProcessingData {
  node_id: string;
  node_type: string;
  node_label: string;
  stage: string;
  progress?: number;
  chunks_created?: number;
  embeddings_created?: number;
  suggestions_found?: number;
  error_message?: string;
}

export interface Activity {
  id: string;
  type: ActivityType;
  message: string;
  created_at: string;
  updated_at?: string;
  
  // Navigation references
  node_id?: string;
  node_type?: string;
  relationship_id?: string;
  
  // Structured data
  suggestion_data?: SuggestionData;
  processing_data?: ProcessingData;
  
  // Status
  status?: ActivityStatus;
  
  // Metadata
  created_by?: string;
  group_id?: string;
}

export interface ActivityFilter {
  types?: ActivityType[];
  status?: ActivityStatus;
  node_id?: string;
  group_id?: string;
  since?: string;
  limit?: number;
  include_dismissed?: boolean;
}

// Helper to check if activity is interactive (has pending actions)
export function isInteractiveActivity(activity: Activity): boolean {
  return (
    activity.type === 'relationship_suggested' &&
    activity.status === 'pending'
  );
}

// Helper to check if activity can navigate to a node
export function hasNavigation(activity: Activity): boolean {
  return !!activity.node_id || !!activity.relationship_id;
}

// Helper to get icon for activity type
export function getActivityIcon(type: ActivityType): string {
  const icons: Record<ActivityType, string> = {
    node_created: '➕',
    node_updated: '✏️',
    node_deleted: '🗑️',
    relationship_created: '🔗',
    relationship_updated: '🔗',
    relationship_deleted: '✂️',
    relationship_suggested: '💡',
    relationship_auto_created: '🤖',
    processing_started: '⏳',
    processing_chunking: '📄',
    processing_embedding: '🧮',
    processing_analyzing: '🔍',
    processing_completed: '✅',
    processing_failed: '❌',
    error: '🚨',
    warning: '⚠️',
    info: 'ℹ️',
  };
  return icons[type] || '📌';
}

// Helper to get color class for activity type
export function getActivityColor(type: ActivityType): string {
  if (type.startsWith('processing_')) {
    if (type === 'processing_completed') return 'text-green-500';
    if (type === 'processing_failed') return 'text-red-500';
    return 'text-blue-500';
  }
  if (type === 'relationship_suggested') return 'text-amber-500';
  if (type === 'relationship_auto_created') return 'text-purple-500';
  if (type === 'error') return 'text-red-500';
  if (type === 'warning') return 'text-yellow-500';
  return 'text-gray-500';
}

// Helper to format confidence as percentage
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

