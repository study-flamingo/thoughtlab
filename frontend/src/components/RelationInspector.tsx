import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';
import type { RelationshipResponse, RelationshipType } from '../types/graph';

interface Props {
  relationshipId: string | null;
  onClose: () => void;
}

const RELATIONSHIP_TYPES: RelationshipType[] = [
  'SUPPORTS',
  'CONTRADICTS',
  'RELATES_TO',
  'OBSERVED_IN',
  'DISCUSSES',
];

export default function RelationInspector({ relationshipId, onClose }: Props) {
  const queryClient = useQueryClient();
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<any>({});

  // Debug logging
  useEffect(() => {
    console.log('RelationInspector: relationshipId prop:', relationshipId);
  }, [relationshipId]);

  const { data: relationship, isLoading, error } = useQuery({
    queryKey: ['relationship', relationshipId],
    queryFn: async () => {
      console.log('RelationInspector: Fetching relationship with ID:', relationshipId);
      const response = await graphApi.getRelationship(relationshipId!);
      console.log('RelationInspector: Received relationship data:', response.data);
      return response.data;
    },
    enabled: !!relationshipId,
  });

  // Debug logging for query state
  useEffect(() => {
    console.log('RelationInspector: isLoading:', isLoading, 'error:', error, 'relationship:', relationship);
  }, [isLoading, error, relationship]);

  // Fetch source and target nodes for display
  const { data: sourceNode } = useQuery({
    queryKey: ['node', relationship?.from_id],
    queryFn: async () => (await graphApi.getNode(relationship!.from_id)).data,
    enabled: !!relationship?.from_id,
  });

  const { data: targetNode } = useQuery({
    queryKey: ['node', relationship?.to_id],
    queryFn: async () => (await graphApi.getNode(relationship!.to_id)).data,
    enabled: !!relationship?.to_id,
  });

  // Initialize form data when relationship loads
  useEffect(() => {
    if (relationship) {
      setFormData({
        relationship_type: relationship.type || '',
        confidence: relationship.confidence ?? 0.8,
        notes: relationship.notes || '',
        inverse_relationship_type: relationship.inverse_relationship_type || '',
        inverse_confidence: relationship.inverse_confidence ?? 0.8,
        inverse_notes: relationship.inverse_notes || '',
      });
    }
  }, [relationship]);

  const updateRelationshipMutation = useMutation({
    mutationFn: (data: {
      confidence?: number;
      notes?: string;
      inverse_relationship_type?: string;
      inverse_confidence?: number;
      inverse_notes?: string;
      relationship_type?: string;
    }) => graphApi.updateRelationship(relationshipId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['relationship', relationshipId] });
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      setIsEditing(false);
    },
  });

  const deleteRelationshipMutation = useMutation({
    mutationFn: (id: string) => graphApi.deleteRelationship(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const handleSave = () => {
    if (!relationship) return;

    updateRelationshipMutation.mutate({
      relationship_type: formData.relationship_type || undefined,
      confidence: formData.confidence,
      notes: formData.notes || undefined,
      inverse_relationship_type: formData.inverse_relationship_type || undefined,
      inverse_confidence: formData.inverse_confidence,
      inverse_notes: formData.inverse_notes || undefined,
    });
  };

  if (!relationshipId) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Relation Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-gray-500">Select a relation to view details</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Relation Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (!relationship) {
    return (
      <div className="h-full flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-gray-800">Relation Inspector</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-red-500">Relation not found</p>
        </div>
      </div>
    );
  }

  const isSaving = updateRelationshipMutation.isPending;

  // Helper to get node display text
  const getNodeDisplayText = (node: any) => {
    if (node?.text) return node.text.substring(0, 50) + (node.text.length > 50 ? '...' : '');
    if (node?.title) return node.title.substring(0, 50) + (node.title.length > 50 ? '...' : '');
    if (node?.name) return node.name.substring(0, 50) + (node.name.length > 50 ? '...' : '');
    if (node?.claim) return node.claim.substring(0, 50) + (node.claim.length > 50 ? '...' : '');
    return node?.id?.substring(0, 8) || 'Unknown';
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="font-semibold text-gray-800">Relation Inspector</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 text-xl leading-none"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Relationship ID */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">ID</label>
          <p className="text-sm font-mono text-gray-700 break-all">{relationship.id}</p>
        </div>

        {/* Direction: Forward */}
        <div className="pt-2 border-t">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Forward Direction</h3>
          
          {/* Source Node */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-500 mb-1">From</label>
            <div className="text-sm text-gray-700">
              <span className="inline-block px-2 py-1 text-xs font-medium rounded bg-gray-100 text-gray-800 mr-2">
                {sourceNode?.type || 'Unknown'}
              </span>
              <span className="font-mono text-xs text-gray-500">{relationship.from_id.substring(0, 8)}...</span>
              <p className="mt-1 text-gray-600">{getNodeDisplayText(sourceNode)}</p>
            </div>
          </div>

          {/* Relationship Type */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Type</label>
            {isEditing ? (
              <select
                value={formData.relationship_type}
                onChange={(e) => setFormData({ ...formData, relationship_type: e.target.value })}
                className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {RELATIONSHIP_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-sm text-gray-700 font-medium">{relationship.type}</p>
            )}
          </div>

          {/* Confidence */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Confidence: {isEditing ? `${(formData.confidence * 100).toFixed(0)}%` : `${((relationship.confidence || 0) * 100).toFixed(0)}%`}
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
                  style={{ width: `${((relationship.confidence || 0) * 100)}%` }}
                ></div>
              </div>
            )}
          </div>

          {/* Notes */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Notes</label>
            {isEditing ? (
              <textarea
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={3}
                placeholder="Add notes about this relationship..."
              />
            ) : (
              <p className="text-sm text-gray-700 whitespace-pre-wrap">
                {relationship.notes || '-'}
              </p>
            )}
          </div>

          {/* Target Node */}
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">To</label>
            <div className="text-sm text-gray-700">
              <span className="inline-block px-2 py-1 text-xs font-medium rounded bg-gray-100 text-gray-800 mr-2">
                {targetNode?.type || 'Unknown'}
              </span>
              <span className="font-mono text-xs text-gray-500">{relationship.to_id.substring(0, 8)}...</span>
              <p className="mt-1 text-gray-600">{getNodeDisplayText(targetNode)}</p>
            </div>
          </div>
        </div>

        {/* Direction: Inverse */}
        <div className="pt-4 border-t">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Inverse Direction</h3>
          
          {/* Inverse Relationship Type */}
          <div className="mb-3">
            <label className="block text-xs font-medium text-gray-700 mb-1">Inverse Type</label>
            {isEditing ? (
              <select
                value={formData.inverse_relationship_type}
                onChange={(e) => setFormData({ ...formData, inverse_relationship_type: e.target.value })}
                className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">None</option>
                {RELATIONSHIP_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-sm text-gray-700">
                {relationship.inverse_relationship_type || '-'}
              </p>
            )}
          </div>

          {/* Inverse Confidence */}
          {(relationship.inverse_relationship_type || (isEditing && formData.inverse_relationship_type)) && (
            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Inverse Confidence: {isEditing ? `${(formData.inverse_confidence * 100).toFixed(0)}%` : `${((relationship.inverse_confidence || 0) * 100).toFixed(0)}%`}
              </label>
              {isEditing ? (
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={formData.inverse_confidence}
                  onChange={(e) => setFormData({ ...formData, inverse_confidence: parseFloat(e.target.value) })}
                  className="w-full"
                />
              ) : (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${((relationship.inverse_confidence || 0) * 100)}%` }}
                  ></div>
                </div>
              )}
            </div>
          )}

          {/* Inverse Notes */}
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Inverse Notes</label>
            {isEditing ? (
              <textarea
                value={formData.inverse_notes}
                onChange={(e) => setFormData({ ...formData, inverse_notes: e.target.value })}
                className="w-full border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={3}
                placeholder="Add notes about the inverse relationship..."
              />
            ) : (
              <p className="text-sm text-gray-700 whitespace-pre-wrap">
                {relationship.inverse_notes || '-'}
              </p>
            )}
          </div>
        </div>

        {/* Metadata */}
        {relationship.created_at && (
          <div className="pt-4 border-t">
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1">Created</label>
              <p className="text-xs text-gray-600">
                {new Date(relationship.created_at).toLocaleString()}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t bg-gray-50 flex gap-2">
        {isEditing ? (
          <>
            <button
              onClick={() => {
                setIsEditing(false);
                // Reset form data
                if (relationship) {
                  setFormData({
                    confidence: relationship.confidence ?? 0.8,
                    notes: relationship.notes || '',
                    inverse_relationship_type: relationship.inverse_relationship_type || '',
                    inverse_confidence: relationship.inverse_confidence ?? 0.8,
                    inverse_notes: relationship.inverse_notes || '',
                    relationship_type: relationship.type || '',
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
                if (confirm('Delete this relationship? This cannot be undone.')) {
                  if (relationshipId) {
                    deleteRelationshipMutation.mutate(relationshipId);
                  }
                }
              }}
              className="flex-1 px-4 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50"
              disabled={deleteRelationshipMutation.isPending}
            >
              {deleteRelationshipMutation.isPending ? 'Deleting...' : 'Delete'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
