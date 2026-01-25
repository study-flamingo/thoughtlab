export type NodeType = 'Observation' | 'Hypothesis' | 'Source' | 'Concept' | 'Entity';

export type RelationshipType =
  | 'SUPPORTS'
  | 'CONTRADICTS'
  | 'RELATES_TO'
  | 'OBSERVED_IN'
  | 'DISCUSSES'
  | 'CITES'
  | 'DERIVED_FROM'
  | 'INSPIRED_BY'
  | 'PRECEDES'
  | 'CAUSES'
  | 'PART_OF'
  | 'SIMILAR_TO'
  | 'HAS_CHUNK';

// Human-readable display names for relationship types
export const RELATIONSHIP_TYPE_DISPLAY: Record<RelationshipType, string> = {
  'SUPPORTS': 'Supports Evidence',
  'CONTRADICTS': 'Contradicts',
  'RELATES_TO': 'Relates To',
  'OBSERVED_IN': 'Observed In',
  'DISCUSSES': 'Discusses',
  'CITES': 'Cites Reference',
  'DERIVED_FROM': 'Derived From',
  'INSPIRED_BY': 'Inspired By',
  'PRECEDES': 'Precedes',
  'CAUSES': 'Causes',
  'PART_OF': 'Part Of',
  'SIMILAR_TO': 'Similar To',
  'HAS_CHUNK': 'Has Chunk',
};

// Status colors for hypothesis nodes
export type HypothesisStatus = 'proposed' | 'tested' | 'confirmed' | 'rejected';

export const STATUS_COLORS: Record<HypothesisStatus, string> = {
  'proposed': '#3B82F6',  // blue-500
  'tested': '#F59E0B',    // amber-500
  'confirmed': '#10B981', // emerald-500
  'rejected': '#EF4444',  // red-500
};

export interface LinkItem {
  url: string;
  label?: string;
}

export interface GraphNode {
  id: string;
  type: NodeType;
  text?: string;
  title?: string;
  name?: string; // For Entity nodes
  description?: string; // For Entity nodes
  confidence?: number;
  concept_names?: string[];
  links?: LinkItem[]; // Clickable links for any node type
  created_at: string;
  updated_at?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: RelationshipType;
  confidence?: number;
  notes?: string;
  // Inverse relationship metadata for asymmetrical relationships
  inverse_relationship_type?: RelationshipType;
  inverse_confidence?: number;
  inverse_notes?: string;
}

export interface RelationshipResponse {
  id: string;
  from_id: string;
  to_id: string;
  type: RelationshipType;
  confidence?: number;
  notes?: string;
  inverse_relationship_type?: RelationshipType;
  inverse_confidence?: number;
  inverse_notes?: string;
  created_at?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface CreateObservationData {
  text: string;
  confidence?: number;
  subject_ids?: string[];
  concept_names?: string[];
}

export interface ConnectionSuggestion {
  id: string;
  from_id: string;
  to_id: string;
  relationship_type: RelationshipType;
  confidence: number;
  reasoning: string;
  status: 'pending' | 'approved' | 'rejected';
}

export interface ActivityEvent {
  id: string;
  type: 'analysis_started' | 'connection_approved' | 'needs_review' | 'analysis_complete';
  data: any;
  timestamp: string;
}
