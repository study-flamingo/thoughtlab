import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';
import type { GraphNode, LinkItem } from '../types/graph';

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
        hypothesisName: node.name || '',
        claim: (node as any).claim || '',
        status: (node as any).status || 'proposed',
        // Concept fields
        domain: (node as any).domain || 'general',
        // Links (all node types)
        links: node.links || [],
      });
    }
  }, [node]);

  const updateObservationMutation = useMutation({
    mutationFn: (data: { text?: string; confidence?: number; links?: LinkItem[] }) =>
      graphApi.updateObservation(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateEntityMutation = useMutation({
    mutationFn: (data: { name?: string; description?: string; entity_type?: string; links?: LinkItem[] }) =>
      graphApi.updateEntity(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateHypothesisMutation = useMutation({
    mutationFn: (data: { name?: string; claim?: string; status?: string; links?: LinkItem[] }) =>
      graphApi.updateHypothesis(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateSourceMutation = useMutation({
    mutationFn: (data: { title?: string; url?: string; source_type?: string; links?: LinkItem[] }) =>
      graphApi.updateSource(nodeId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['node', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const updateConceptMutation = useMutation({
    mutationFn: (data: { name?: string; description?: string; domain?: string; links?: LinkItem[] }) =>
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

    // Filter out empty links - always pass the array (even if empty) to allow deletion
    const validLinks = (formData.links || []).filter((link: LinkItem) => link.url?.trim());

    switch (node.type) {
      case 'Observation':
        updateObservationMutation.mutate({
          text: formData.text,
          confidence: formData.confidence,
          links: validLinks,
        });
        break;
      case 'Entity':
        updateEntityMutation.mutate({
          name: formData.name,
          description: formData.description,
          entity_type: formData.entity_type,
          links: validLinks,
        });
        break;
      case 'Hypothesis':
        updateHypothesisMutation.mutate({
          name: formData.hypothesisName,
          claim: formData.claim,
          status: formData.status,
          links: validLinks,
        });
        break;
      case 'Source':
        updateSourceMutation.mutate({
          title: formData.title,
          url: formData.url || undefined,
          source_type: formData.source_type,
          links: validLinks,
        });
        break;
      case 'Concept':
        updateConceptMutation.mutate({
          name: formData.name,
          description: formData.description || undefined,
          domain: formData.domain,
          links: validLinks,
        });
        break;
      default:
        console.warn('Update not implemented for node type:', node.type);
    }
  };

  if (!nodeId) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b dark:border-gray-700">
          <h2 className="font-semibold text-gray-800 dark:text-gray-100">Node Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">Select a node to view details</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b dark:border-gray-700">
          <h2 className="font-semibold text-gray-800 dark:text-gray-100">Node Inspector</h2>
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
        <div className="p-4 border-b dark:border-gray-700">
          <h2 className="font-semibold text-gray-800 dark:text-gray-100">Node Inspector</h2>
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
      <div className="p-4 border-b dark:border-gray-700 flex justify-between items-center">
        <h2 className="font-semibold text-gray-800 dark:text-gray-100">Node Inspector</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Node Type Badge */}
        <div>
          <span className="inline-block px-2 py-1 text-xs font-medium rounded bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
            {node.type}
          </span>
        </div>

        {/* Node ID */}
        <div>
          <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">ID</label>
          <p className="text-sm font-mono text-gray-700 dark:text-gray-300 break-all">{node.id}</p>
        </div>

        {/* Editable Fields Based on Node Type */}
        {node.type === 'Observation' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Text
              </label>
              {isEditing ? (
                <textarea
                  value={formData.text}
                  onChange={(e) => setFormData({ ...formData, text: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{node.text || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
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
                  className="w-full accent-blue-600"
                />
              ) : (
                <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
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
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{node.name || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description
              </label>
              {isEditing ? (
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{node.description || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Entity Type
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.entity_type}
                  onChange={(e) => setFormData({ ...formData, entity_type: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="generic, person, organization, etc."
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{(node as any).entity_type || 'generic'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Hypothesis' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.hypothesisName}
                  onChange={(e) => setFormData({ ...formData, hypothesisName: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{node.name || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Claim
              </label>
              {isEditing ? (
                <textarea
                  value={formData.claim}
                  onChange={(e) => setFormData({ ...formData, claim: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{(node as any).claim || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Status
              </label>
              {isEditing ? (
                <select
                  value={formData.status}
                  onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="proposed">Proposed</option>
                  <option value="tested">Tested</option>
                  <option value="confirmed">Confirmed</option>
                  <option value="rejected">Rejected</option>
                </select>
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{(node as any).status || 'proposed'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Source' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Title
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{node.title || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                URL
              </label>
              {isEditing ? (
                <input
                  type="url"
                  value={formData.url}
                  onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="https://..."
                />
              ) : (
                (node as any).url ? (
                  <a 
                    href={(node as any).url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 dark:text-blue-400 hover:underline break-all"
                  >
                    {(node as any).url}
                  </a>
                ) : (
                  <p className="text-sm text-gray-700 dark:text-gray-300">-</p>
                )
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Source Type
              </label>
              {isEditing ? (
                <>
                  <input
                    type="text"
                    list="source-type-suggestions"
                    value={formData.source_type}
                    onChange={(e) => setFormData({ ...formData, source_type: e.target.value })}
                    className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., paper, article, video..."
                  />
                  <datalist id="source-type-suggestions">
                    <option value="paper" />
                    <option value="article" />
                    <option value="book" />
                    <option value="website" />
                    <option value="forum" />
                    <option value="video" />
                    <option value="podcast" />
                    <option value="social media" />
                    <option value="documentation" />
                    <option value="report" />
                    <option value="other" />
                  </datalist>
                </>
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{(node as any).source_type || 'paper'}</p>
              )}
            </div>
          </>
        )}

        {node.type === 'Concept' && (
          <>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300">{node.name || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description
              </label>
              {isEditing ? (
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={4}
                />
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{node.description || '-'}</p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Domain
              </label>
              {isEditing ? (
                <select
                  value={formData.domain}
                  onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                <p className="text-sm text-gray-700 dark:text-gray-300">{(node as any).domain || 'general'}</p>
              )}
            </div>
          </>
        )}

        {/* Links Section (all node types) */}
        <div className="pt-4 border-t dark:border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
              Links
            </label>
            {isEditing && (
              <button
                type="button"
                onClick={() => {
                  setFormData({
                    ...formData,
                    links: [...(formData.links || []), { url: '', label: '' }]
                  });
                }}
                className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
              >
                + Add Link
              </button>
            )}
          </div>
          
          {isEditing ? (
            <div className="space-y-2">
              {(formData.links || []).map((link: LinkItem, index: number) => (
                <div key={index} className="flex gap-2 items-start">
                  <div className="flex-1 space-y-1">
                    <input
                      type="url"
                      value={link.url}
                      onChange={(e) => {
                        const newLinks = [...formData.links];
                        newLinks[index] = { ...newLinks[index], url: e.target.value };
                        setFormData({ ...formData, links: newLinks });
                      }}
                      className="w-full border dark:border-gray-600 rounded-md px-2 py-1 text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="https://..."
                    />
                    <input
                      type="text"
                      value={link.label || ''}
                      onChange={(e) => {
                        const newLinks = [...formData.links];
                        newLinks[index] = { ...newLinks[index], label: e.target.value };
                        setFormData({ ...formData, links: newLinks });
                      }}
                      className="w-full border dark:border-gray-600 rounded-md px-2 py-1 text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="Label (optional)"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      const newLinks = formData.links.filter((_: LinkItem, i: number) => i !== index);
                      setFormData({ ...formData, links: newLinks });
                    }}
                    className="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 text-lg leading-none mt-1"
                    title="Remove link"
                  >
                    ×
                  </button>
                </div>
              ))}
              {(!formData.links || formData.links.length === 0) && (
                <p className="text-xs text-gray-400 dark:text-gray-500 italic">No links added</p>
              )}
            </div>
          ) : (
            <div className="space-y-1">
              {node.links && node.links.length > 0 ? (
                node.links.map((link, index) => (
                  <a
                    key={index}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 hover:underline break-all"
                  >
                    <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    <span>{link.label || link.url}</span>
                  </a>
                ))
              ) : (
                <p className="text-xs text-gray-400 dark:text-gray-500 italic">No links</p>
              )}
            </div>
          )}
        </div>

        {/* Metadata */}
        <div className="pt-4 border-t dark:border-gray-700 space-y-2">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Created</label>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {node.created_at ? new Date(node.created_at).toLocaleString() : '-'}
            </p>
          </div>
          {node.updated_at && (
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Updated</label>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {new Date(node.updated_at).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t dark:border-gray-700 bg-gray-50 dark:bg-gray-900 flex gap-2">
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
                    hypothesisName: node.name || '',
                    claim: (node as any).claim || '',
                    status: (node as any).status || 'proposed',
                    domain: (node as any).domain || 'general',
                    links: node.links || [],
                  });
                }
              }}
              className="flex-1 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
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
