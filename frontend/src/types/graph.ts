export type NodeType = 'Observation' | 'Hypothesis' | 'Source' | 'Concept' | 'Entity';

export type RelationshipType = 
  | 'SUPPORTS' 
  | 'CONTRADICTS' 
  | 'RELATES_TO' 
  | 'OBSERVED_IN' 
  | 'DISCUSSES';

export interface GraphNode {
  id: string;
  type: NodeType;
  text?: string;
  title?: string;
  name?: string; // For Entity nodes
  description?: string; // For Entity nodes
  confidence?: number;
  concept_names?: string[];
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
