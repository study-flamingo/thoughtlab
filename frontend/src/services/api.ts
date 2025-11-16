import axios from 'axios';
import type {
  GraphData,
  CreateObservationData,
  GraphNode,
  ConnectionSuggestion,
} from '../types/graph';

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

  getObservation: (id: string) => api.get<GraphNode>(`/nodes/observations/${id}`),

  getAllObservations: (limit?: number) =>
    api.get<{ nodes: GraphNode[] }>('/nodes/observations', {
      params: { limit },
    }),

  getConnections: (id: string, maxDepth = 2) =>
    api.get(`/nodes/${id}/connections`, { params: { max_depth: maxDepth } }),

  createRelationship: (fromId: string, toId: string, type: string, confidence?: number) =>
    api.post('/nodes/relationships', {
      from_id: fromId,
      to_id: toId,
      relationship_type: type,
      confidence,
    }),

  approveSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/approve`),

  rejectSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/reject`),
};

export default api;
