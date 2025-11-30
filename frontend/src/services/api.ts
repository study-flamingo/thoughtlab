import axios from 'axios';
import type {
  GraphData,
  CreateObservationData,
  GraphNode,
  RelationshipResponse,
  LinkItem,
} from '../types/graph';
import type { AppSettings, AppSettingsUpdate } from '../types/settings';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth (when implemented)
api.interceptors.request.use((config) => {
  // const token = localStorage.getItem('token');
  // if (token) {
  //   config.headers.Authorization = `Bearer ${token}`;
  // }
  return config;
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.error('Unauthorized access');
    }
    return Promise.reject(error);
  }
);

export const graphApi = {
  getFullGraph: () => api.get<GraphData>('/graph/full'),

  createObservation: (data: CreateObservationData) =>
    api.post<{ id: string; message: string }>('/nodes/observations', data),

  createEntity: (data: { name: string; entity_type?: string; description?: string; links?: LinkItem[] }) =>
    api.post<{ id: string; message: string }>('/nodes/entities', data),

  createSource: (data: { title: string; url?: string; source_type?: string; content?: string; published_date?: string; links?: LinkItem[] }) =>
    api.post<{ id: string; message: string }>('/nodes/sources', data),

  createHypothesis: (data: { name: string; claim: string; status?: string; links?: LinkItem[] }) =>
    api.post<{ id: string; message: string }>('/nodes/hypotheses', data),

  createConcept: (data: { name: string; description?: string; domain?: string; links?: LinkItem[] }) =>
    api.post<{ id: string; message: string }>('/nodes/concepts', data),

  getNode: (id: string) => api.get<GraphNode>(`/nodes/${id}`),

  updateObservation: (id: string, data: { text?: string; confidence?: number; concept_names?: string[]; links?: LinkItem[] }) =>
    api.put<{ id: string; message: string }>(`/nodes/observations/${id}`, data),

  updateEntity: (id: string, data: { name?: string; entity_type?: string; description?: string; links?: LinkItem[] }) =>
    api.put<{ id: string; message: string }>(`/nodes/entities/${id}`, data),

  updateHypothesis: (id: string, data: { name?: string; claim?: string; status?: string; links?: LinkItem[] }) =>
    api.put<{ id: string; message: string }>(`/nodes/hypotheses/${id}`, data),

  updateConcept: (id: string, data: { name?: string; description?: string; domain?: string; links?: LinkItem[] }) =>
    api.put<{ id: string; message: string }>(`/nodes/concepts/${id}`, data),

  updateSource: (id: string, data: { title?: string; url?: string; source_type?: string; content?: string; links?: LinkItem[] }) =>
    api.put<{ id: string; message: string }>(`/nodes/sources/${id}`, data),

  getObservation: (id: string) => api.get<GraphNode>(`/nodes/observations/${id}`),

  getAllObservations: (limit?: number) =>
    api.get<{ nodes: GraphNode[] }>('/nodes/observations', {
      params: { limit },
    }),

  getConnections: (id: string, maxDepth = 2) =>
    api.get(`/nodes/${id}/connections`, { params: { max_depth: maxDepth } }),

  createRelationship: (
    fromId: string,
    toId: string,
    type: string,
    options?: {
      confidence?: number;
      notes?: string;
      inverse_relationship_type?: string;
      inverse_confidence?: number;
      inverse_notes?: string;
    }
  ) =>
    api.post('/nodes/relationships', {
      from_id: fromId,
      to_id: toId,
      relationship_type: type,
      confidence: options?.confidence,
      notes: options?.notes,
      inverse_relationship_type: options?.inverse_relationship_type,
      inverse_confidence: options?.inverse_confidence,
      inverse_notes: options?.inverse_notes,
    }),

  getRelationship: (relationshipId: string) =>
    api.get<RelationshipResponse>(`/nodes/relationships/${relationshipId}`),

  updateRelationship: (
    relationshipId: string,
    data: {
      relationship_type?: string;
      confidence?: number;
      notes?: string;
      inverse_relationship_type?: string;
      inverse_confidence?: number;
      inverse_notes?: string;
    }
  ) =>
    api.put<{ id: string; message: string }>(`/nodes/relationships/${relationshipId}`, data),
  
  deleteRelationship: (relationshipId: string) =>
    api.delete<{ id: string; message: string }>(`/nodes/relationships/${relationshipId}`),

  deleteNode: (nodeId: string) =>
    api.delete<{ id: string; message: string }>(`/nodes/${nodeId}`),

  // Settings
  getSettings: () => api.get<AppSettings>('/settings'),
  updateSettings: (data: AppSettingsUpdate) => api.put<AppSettings>('/settings', data),

  approveSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/approve`),

  rejectSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/reject`),
};

export default api;
