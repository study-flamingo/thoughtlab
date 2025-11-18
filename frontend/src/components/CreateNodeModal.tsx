import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';

interface Props {
  onClose: () => void;
}

export default function CreateNodeModal({ onClose }: Props) {
  const [nodeType, setNodeType] = useState('observation');
  const [text, setText] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  const [title, setTitle] = useState('');
  const [url, setUrl] = useState('');
  const [entityName, setEntityName] = useState('');
  const [entityDescription, setEntityDescription] = useState('');

  const queryClient = useQueryClient();

  const createObservationMutation = useMutation({
    mutationFn: graphApi.createObservation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const createEntityMutation = useMutation({
    mutationFn: graphApi.createEntity,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const createSourceMutation = useMutation({
    mutationFn: (payload: { title: string; url?: string }) =>
      graphApi.createSource({ title: payload.title, url: payload.url }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (nodeType === 'observation') {
      createObservationMutation.mutate({ text, confidence });
    } else if (nodeType === 'entity') {
      createEntityMutation.mutate({ 
        name: entityName, 
        description: entityDescription || undefined 
      });
    } else if (nodeType === 'source') {
      createSourceMutation.mutate({
        title,
        url: url || undefined,
      });
    }
    // TODO: Handle other node types (hypothesis, source, concept)
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 py-4 border-b flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-800">Create New Node</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-xl leading-none"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Node Type
            </label>
            <select
              value={nodeType}
              onChange={(e) => setNodeType(e.target.value)}
              className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="observation">Observation</option>
              <option value="hypothesis">Hypothesis</option>
              <option value="source">Source</option>
              <option value="entity">Entity</option>
              <option value="concept">Concept</option>
            </select>
          </div>

          {nodeType === 'observation' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Observation Text
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 h-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Describe what you observed..."
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </>
          )}

          {nodeType === 'source' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Source title..."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  URL (optional)
                </label>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="https://..."
                />
              </div>
            </>
          )}

          {nodeType === 'entity' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Entity Name
                </label>
                <input
                  type="text"
                  value={entityName}
                  onChange={(e) => setEntityName(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter entity name..."
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description (optional)
                </label>
                <textarea
                  value={entityDescription}
                  onChange={(e) => setEntityDescription(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 h-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Describe the entity..."
                />
              </div>
            </>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createObservationMutation.isPending || createEntityMutation.isPending || createSourceMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {(createObservationMutation.isPending || createEntityMutation.isPending || createSourceMutation.isPending) ? 'Creating...' : 'Create Node'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
