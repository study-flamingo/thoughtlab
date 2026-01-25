import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';
import type { GraphNode, RelationshipType } from '../types/graph';
import { RELATIONSHIP_TYPE_DISPLAY } from '../types/graph';

interface Props {
  onClose: () => void;
}

const RELATIONSHIP_TYPES: RelationshipType[] = [
  'SUPPORTS',
  'CONTRADICTS',
  'RELATES_TO',
  'OBSERVED_IN',
  'DISCUSSES',
  'CITES',
  'DERIVED_FROM',
  'INSPIRED_BY',
  'PRECEDES',
  'CAUSES',
  'PART_OF',
  'SIMILAR_TO',
  'HAS_CHUNK',
];

export default function CreateRelationModal({ onClose }: Props) {
  const queryClient = useQueryClient();

  const { data: graphData } = useQuery({
    queryKey: ['graph', 'full', 'for-relation-modal'],
    queryFn: async () => (await graphApi.getFullGraph()).data,
  });

  const nodes: GraphNode[] = graphData?.nodes ?? [];

  const [fromId, setFromId] = useState<string>('');
  const [toId, setToId] = useState<string>('');
  const [relType, setRelType] = useState<RelationshipType>('RELATES_TO');
  const [confidence, setConfidence] = useState<number>(0.8);
  const [notes, setNotes] = useState<string>('');
  const [inverseType, setInverseType] = useState<RelationshipType | ''>('');
  const [inverseConfidence, setInverseConfidence] = useState<number | ''>('');
  const [inverseNotes, setInverseNotes] = useState<string>('');

  const createRelationshipMutation = useMutation({
    mutationFn: async () =>
      graphApi.createRelationship(fromId, toId, relType, {
        confidence,
        notes: notes || undefined,
        inverse_relationship_type: inverseType || undefined,
        inverse_confidence:
          typeof inverseConfidence === 'number' ? inverseConfidence : undefined,
        inverse_notes: inverseNotes || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!fromId || !toId || fromId === toId) return;
    createRelationshipMutation.mutate();
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-xl mx-4 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 py-4 border-b dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Add Relation</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-xl leading-none"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Origin Node
              </label>
              <select
                value={fromId}
                onChange={(e) => setFromId(e.target.value)}
                className="w-full border dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              >
                <option value="" disabled>
                  Select origin...
                </option>
                {nodes.map((n) => {
                  const label = n.text || n.title || n.name || n.id;
                  return (
                    <option key={n.id} value={n.id}>
                      [{n.type}] {label}
                    </option>
                  );
                })}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Destination Node
              </label>
              <select
                value={toId}
                onChange={(e) => setToId(e.target.value)}
                className="w-full border dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              >
                <option value="" disabled>
                  Select destination...
                </option>
                {nodes
                  .filter((n) => n.id !== fromId)
                  .map((n) => {
                    const label = n.text || n.title || n.name || n.id;
                    return (
                      <option key={n.id} value={n.id}>
                        [{n.type}] {label}
                      </option>
                    );
                  })}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Relationship Type
              </label>
              <select
                value={relType}
                onChange={(e) => setRelType(e.target.value as RelationshipType)}
                className="w-full border dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {RELATIONSHIP_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {RELATIONSHIP_TYPE_DISPLAY[t]}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Confidence: {(confidence * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-blue-600"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Notes (optional)
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full border dark:border-gray-600 rounded-md px-3 py-2 h-24 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Describe the nature of this relationship..."
            />
          </div>

          <div className="pt-2 border-t dark:border-gray-700">
            <div className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
              Inverse Relationship (when reversed)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Inverse Type (optional)
                </label>
                <select
                  value={inverseType}
                  onChange={(e) =>
                    setInverseType((e.target.value as RelationshipType) || '')
                  }
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Same as type / not specified</option>
                  {RELATIONSHIP_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {RELATIONSHIP_TYPE_DISPLAY[t]}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Inverse Confidence (optional)
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={inverseConfidence === '' ? 0 : inverseConfidence}
                  onChange={(e) =>
                    setInverseConfidence(parseFloat(e.target.value))
                  }
                  className="w-full accent-blue-600"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Inverse Notes (optional)
                </label>
                <input
                  type="text"
                  value={inverseNotes}
                  onChange={(e) => setInverseNotes(e.target.value)}
                  className="w-full border dark:border-gray-600 rounded-md px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., CONTRADICTS when reversed"
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t dark:border-gray-700 mt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createRelationshipMutation.isPending || !fromId || !toId || fromId === toId}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {createRelationshipMutation.isPending ? 'Creating...' : 'Create Relation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
