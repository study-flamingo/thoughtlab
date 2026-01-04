// === Request Types ===

export interface FindRelatedNodesRequest {
  limit?: number;
  min_similarity?: number;
  node_types?: string[];
  auto_link?: boolean;
}

export interface SummarizeNodeRequest {
  max_length?: number;
  style?: 'concise' | 'detailed' | 'bullet_points';
}

export interface SummarizeNodeWithContextRequest {
  depth?: number;
  relationship_types?: string[];
  max_length?: number;
}

export interface RecalculateConfidenceRequest {
  factor_in_relationships?: boolean;
}

export interface ReclassifyNodeRequest {
  new_type: string;
  preserve_relationships?: boolean;
}

export interface SearchWebEvidenceRequest {
  evidence_type?: 'supporting' | 'contradicting' | 'all';
  max_results?: number;
  auto_create_sources?: boolean;
}

export interface SummarizeRelationshipRequest {
  include_evidence?: boolean;
}

export interface RecalculateEdgeConfidenceRequest {
  consider_graph_structure?: boolean;
}

export interface ReclassifyRelationshipRequest {
  new_type?: string | null; // null = LLM suggests
  preserve_notes?: boolean;
}

export interface MergeNodesRequest {
  primary_node_id: string;
  secondary_node_id: string;
  merge_strategy?: 'combine' | 'prefer_primary' | 'prefer_secondary';
}

// === Response Types ===

export interface RelatedNodeResult {
  id: string;
  type: string;
  content: string;
  similarity_score: number;
  suggested_relationship: string;
  reasoning: string;
}

export interface FindRelatedNodesResponse {
  success: boolean;
  node_id: string;
  related_nodes: RelatedNodeResult[];
  links_created: number;
  message: string;
  error?: string;
}

export interface SummarizeNodeResponse {
  success: boolean;
  node_id: string;
  summary: string;
  key_points: string[];
  word_count: number;
  error?: string;
}

export interface NodeContextSummary {
  supports: string[];
  contradicts: string[];
  related: string[];
}

export interface SummarizeNodeWithContextResponse {
  success: boolean;
  node_id: string;
  summary: string;
  context: NodeContextSummary;
  synthesis: string;
  relationship_count: number;
  error?: string;
}

export interface ConfidenceFactor {
  factor: string;
  impact: string;
}

export interface RecalculateConfidenceResponse {
  success: boolean;
  node_id: string;
  old_confidence: number;
  new_confidence: number;
  reasoning: string;
  factors: ConfidenceFactor[];
  error?: string;
}

export interface ReclassifyNodeResponse {
  success: boolean;
  node_id: string;
  old_type: string;
  new_type: string;
  message: string;
  warning?: string;
  relationships_preserved: number;
  error?: string;
}

export interface WebEvidenceResult {
  url: string;
  title: string;
  snippet: string;
  relevance_score: number;
  evidence_type: 'supporting' | 'contradicting' | 'neutral';
  reasoning: string;
}

export interface SearchWebEvidenceResponse {
  success: boolean;
  node_id: string;
  results: WebEvidenceResult[];
  sources_created: number;
  message: string;
  error?: string;
}

export interface NodeInfo {
  id: string;
  type: string;
  content: string;
}

export interface SummarizeRelationshipResponse {
  success: boolean;
  edge_id: string;
  from_node: NodeInfo;
  to_node: NodeInfo;
  relationship_type: string;
  summary: string;
  evidence: string[];
  strength_assessment: 'strong' | 'moderate' | 'weak';
  error?: string;
}

export interface RecalculateEdgeConfidenceResponse {
  success: boolean;
  edge_id: string;
  from_node_id: string;
  to_node_id: string;
  old_confidence: number;
  new_confidence: number;
  reasoning: string;
  error?: string;
}

export interface ReclassifyRelationshipResponse {
  success: boolean;
  edge_id: string;
  old_type: string;
  new_type: string;
  confidence: number;
  reasoning: string;
  notes_preserved: boolean;
  error?: string;
}

export interface MergeNodesResponse {
  success: boolean;
  primary_node_id: string;
  secondary_node_id: string;
  merged_content: string;
  relationships_transferred: number;
  message: string;
  error?: string;
}
