import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';
import type { GraphNode } from '../types/graph';

interface Props {
  nodeId: string | null;
  onClose: () => void;
}

export default function NodeInspector({ nodeId, onClose }: Props) {
  const queryClient = useQueryClient();
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<any>({});

  const { data: node, isLoading } = useQuery({
    queryKey: ['node', nodeId],
    queryFn: async () => (await graphApi.getNode(nodeId!)).data,
    enabled: !!nodeId,
  });

  // Initialize form data when node loads
  useEffect(() => {
    if (node) {
      setFormData({
        // Observation fields
        text: node.text || '',
        confidence: node.confidence ?? 0.8,
        // Source fields
        title: node.title || '',
        url: (node as any).url || '',
        source_type: (node as any).source_type || 'paper',
        // Entity fields
        name: node.name || '',
        description: node.description || '',
        entity_type: (node as any).entity_type || 'generic',
        // Hypothesis fields
        claim: (node as any).claim || '',
        status: (node as any).status || 'proposed',
        // Concept fields
        domain: (node as any).domain || 'general',
      });
    }
  }, [node]);

  const updateObservationMutation = useMutation({
    mutationFn: (data: { text?: string; confidence?: number }) =>
      graphApi.updateObservation(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateEntityMutation = useMutation({
    mutationFn: (data: { name?: string; description?: string; entity_type?: string }) =>
      graphApi.updateEntity(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateHypothesisMutation = useMutation({
    mutationFn: (data: { claim?: string; status?: string }) =>
      graphApi.updateHypothesis(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateSourceMutation = useMutation({
    mutationFn: (data: { title?: string; url?: string; source_type?: string }) =>
      graphApi.updateSource(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateConceptMutation = useMutation({
    mutationFn: (data: { name?: string; description?: string; domain?: string }) =>
      graphApi.updateConcept(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const deleteNodeMutation = useMutation({
    mutationFn: (id: string) => graphApi.deleteNode(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const handleSave = () => {
    if (!node) return;

    switch (node.type) {
      case 'Observation':
        updateObservationMutation.mutate({
          text: formData.text,
          confidence: formData.confidence,
        });
        break;
      case 'Entity':
        updateEntityMutation.mutate({
          name: formData.name,
          description: formData.description,
          entity_type: formData.entity_type,
        });
        break;
      case 'Hypothesis':
        updateHypothesisMutation.mutate({
          claim: formData.claim,
          status: formData.status,
        });
        break;
      case 'Source':
        updateSourceMutation.mutate({
          title: formData.title,
          url: formData.url || undefined,
          source_type: formData.source_type,
        });
        break;
      case 'Concept':
        updateConceptMutation.mutate({
          name: formData.name,
          description: formData.description || undefined,
          domain: formData.domain,
        });
        break;
      default:
        console.warn('Update not implemented for node type:', node.type);
    }
  };

  if (!nodeId) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Node Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-gray-500">Select a node to view details</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Node Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (!node) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Node Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-red-500">Node not found</p>
        </div>
      </div>
    );
  }

  const isSaving =
    updateObservationMutation.isPending ||
    updateEntityMutation.isPending ||
    updateHypothesisMutation.isPending ||
    updateSourceMutation.isPending ||
    updateConceptMutation.isPending;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="font-semibold text-gray-800">Node Inspector</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 text-xl leading-none"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Node Type Badge */}
        <div>
          <span className="inline-block px-2 py-1 text-xs font-medium rounded bg-blue-100 text-blue-800">
            {node.type}
          </span>
        </div>

        {/* Node ID */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">ID</label>
          <p className="text-sm font-mono text-gray-700 break-all">{node.id}</p>
        </div>

        {/* Editable Fields Based on Node Type */}
        {node.type === 'Observation' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Text
              </label>
              {isEditing ? (
                <textarea
                  value={formData.text}
                  onChange={(e) => setFormData({ ...formData, text: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{node.text || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Confidence: {isEditing ? `${(formData.confidence * 100).toFixed(0)}%` : `${((node.confidence || 0) * 100).toFixed(0)}%`}
              </label>
              {isEditing ? (
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={formData.confidence}
                  onChange={(e) => setFormData({ ...formData, confidence: parseFloat(e.target.value) })}
                  className="w-full"
                />
              ) : (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${((node.confidence || 0) * 100)}%` }}
                  ></div>
                </div>
              )}
            </div>
          </>
        )}

        {node.type === 'Entity' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700">{node.name || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Description
              </label>
              {isEditing ? (
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{node.description || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Entity Type
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.entity_type}
                  onChange={(e) => setFormData({ ...formData, entity_type: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="generic, person, organization, etc."
                />
              ) : (
                <p className="text-sm text-gray-700">{(node as any).entity_type || 'generic'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Hypothesis' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Claim
              </label>
              {isEditing ? (
                <textarea
                  value={formData.claim}
                  onChange={(e) => setFormData({ ...formData, claim: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{(node as any).claim || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Status
              </label>
              {isEditing ? (
                <select
                  value={formData.status}
                  onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="proposed">Proposed</option>
                  <option value="tested">Tested</option>
                  <option value="confirmed">Confirmed</option>
                  <option value="rejected">Rejected</option>
                </select>
              ) : (
                <p className="text-sm text-gray-700">{(node as any).status || 'proposed'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Source' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Title
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700">{node.title || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                URL
              </label>
              {isEditing ? (
                <input
                  type="url"
                  value={formData.url}
                  onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="https://..."
                />
              ) : (
                (node as any).url ? (
                  <a 
                    href={(node as any).url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 hover:underline break-all"
                  >
                    {(node as any).url}
                  </a>
                ) : (
                  <p className="text-sm text-gray-700">-</p>
                )
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Source Type
              </label>
              {isEditing ? (
                <select
                  value={formData.source_type}
                  onChange={(e) => setFormData({ ...formData, source_type: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="paper">Paper</option>
                  <option value="article">Article</option>
                  <option value="book">Book</option>
                  <option value="website">Website</option>
                  <option value="forum">Forum</option>
                  <option value="video">Video</option>
                  <option value="other">Other</option>
                </select>
              ) : (
                <p className="text-sm text-gray-700">{(node as any).source_type || 'paper'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Concept' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700">{node.name || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Description
              </label>
              {isEditing ? (
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 whitespace-pre-wrap">{node.description || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Domain
              </label>
              {isEditing ? (
                <select
                  value={formData.domain}
                  onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                  className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="general">General</option>
                  <option value="science">Science</option>
                  <option value="technology">Technology</option>
                  <option value="philosophy">Philosophy</option>
                  <option value="mathematics">Mathematics</option>
                  <option value="social">Social Sciences</option>
                  <option value="humanities">Humanities</option>
                  <option value="other">Other</option>
                </select>
              ) : (
                <p className="text-sm text-gray-700">{(node as any).domain || 'general'}</p>
              )}
            </div>
          </>
        )}

        {/* Metadata */}
        <div className="pt-4 border-t space-y-2">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">Created</label>
            <p className="text-xs text-gray-600">
              {node.created_at ? new Date(node.created_at).toLocaleString() : '-'}
            </p>
          </div>
          {node.updated_at && (
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Updated</label>
              <p className="text-xs text-gray-600">
                {new Date(node.updated_at).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t bg-gray-50 flex gap-2">
        {isEditing ? (
          <>
            <button
              onClick={() => {
                setIsEditing(false);
                // Reset form data
                if (node) {
                  setFormData({
                    text: node.text || '',
                    confidence: node.confidence ?? 0.8,
                    title: node.title || '',
                    url: (node as any).url || '',
                    source_type: (node as any).source_type || 'paper',
                    name: node.name || '',
                    description: node.description || '',
                    entity_type: (node as any).entity_type || 'generic',
                    claim: (node as any).claim || '',
                    status: (node as any).status || 'proposed',
                    domain: (node as any).domain || 'general',
                  });
                }
              }}
              className="flex-1 px-4 py-2 text-sm text-gray-700 bg-white border rounded-md hover:bg-gray-50 transition-colors"
              disabled={isSaving}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex-1 px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => setIsEditing(true)}
              className="flex-1 px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Edit
            </button>
            <button
              onClick={() => {
                if (confirm('Delete this node and all connected relationships? This cannot be undone.')) {
                  if (nodeId) {
                    deleteNodeMutation.mutate(nodeId);
                  }
                }
              }}
              className="flex-1 px-4 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50"
              disabled={deleteNodeMutation.isPending}
            >
              {deleteNodeMutation.isPending ? 'Deleting...' : 'Delete'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
