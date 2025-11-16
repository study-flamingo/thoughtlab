import { useRef, useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape, { type Core } from 'cytoscape';
import { graphApi } from '../services/api';
import type { GraphNode, GraphEdge } from '../types/graph';

// Cytoscape.js stylesheet configuration
const cytoscapeStylesheet = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'background-color': '#666',
      color: '#fff',
      'font-size': '10px',
      width: '40px',
      height: '40px',
      'text-wrap': 'wrap',
      'text-max-width': '80px',
      'text-overflow-wrap': 'whitespace',
    },
  },
  {
    selector: 'node[type="Observation"]',
    style: {
      'background-color': '#3B82F6', // blue
      shape: 'ellipse',
    },
  },
  {
    selector: 'node[type="Hypothesis"]',
    style: {
      'background-color': '#10B981', // green
      shape: 'diamond',
    },
  },
  {
    selector: 'node[type="Source"]',
    style: {
      'background-color': '#F59E0B', // yellow/orange
      shape: 'rectangle',
    },
  },
  {
    selector: 'node[type="Concept"]',
    style: {
      'background-color': '#8B5CF6', // purple
      shape: 'hexagon',
    },
  },
  {
    selector: 'node[type="Entity"]',
    style: {
      'background-color': '#EF4444', // red
      shape: 'round-rectangle',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 2,
      'line-color': '#ccc',
      'target-arrow-color': '#ccc',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      label: 'data(type)',
      'font-size': '8px',
      'text-rotation': 'autorotate',
      'text-margin-y': -10,
    },
  },
  {
    selector: 'edge[type="SUPPORTS"]',
    style: {
      'line-color': '#10B981',
      'target-arrow-color': '#10B981',
      width: 3,
    },
  },
  {
    selector: 'edge[type="CONTRADICTS"]',
    style: {
      'line-color': '#EF4444',
      'target-arrow-color': '#EF4444',
      width: 3,
      'line-style': 'dashed',
    },
  },
  {
    selector: 'edge[type="RELATES_TO"]',
    style: {
      'line-color': '#6B7280',
      'target-arrow-color': '#6B7280',
    },
  },
];

export default function GraphVisualizer() {
  const cyRef = useRef<Core | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['graph', 'full'],
    queryFn: async () => {
      const response = await graphApi.getFullGraph(500);
      return response.data;
    },
  });

  // Convert graph data to Cytoscape elements format
  const elements = data
    ? [
        // Nodes
        ...data.nodes.map((node: GraphNode) => ({
          data: {
            id: node.id,
            label: node.text
              ? node.text.substring(0, 30) + (node.text.length > 30 ? '...' : '')
              : node.title
              ? node.title.substring(0, 30) + (node.title.length > 30 ? '...' : '')
              : node.id.substring(0, 8),
            type: node.type,
            ...node,
          },
        })),
        // Edges
        ...data.edges.map((edge: GraphEdge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            ...edge,
          },
        })),
      ]
    : [];

  // Handle node click
  useEffect(() => {
    if (cyRef.current) {
      const cy = cyRef.current;

      const handleTap = (event: cytoscape.EventObject) => {
        const target = event.target;
        if (target.isNode()) {
          const nodeId = target.id();
          setSelectedNodeId(nodeId);
          // Highlight selected node and its neighbors
          cy.elements().removeClass('highlighted');
          target.addClass('highlighted');
          target.neighborhood().addClass('highlighted');
        } else {
          setSelectedNodeId(null);
          cy.elements().removeClass('highlighted');
        }
      };

      cy.on('tap', handleTap);

      return () => {
        cy.off('tap', handleTap);
      };
    }
  }, [data]);

  // Add highlight styles
  useEffect(() => {
    if (cyRef.current) {
      cyRef.current.style().append([
        {
          selector: '.highlighted',
          style: {
            'border-width': 3,
            'border-color': '#F59E0B',
            'background-color': (ele: cytoscape.NodeSingular) => {
              const type = ele.data('type');
              const colors: Record<string, string> = {
                Observation: '#60A5FA',
                Hypothesis: '#34D399',
                Source: '#FBBF24',
                Concept: '#A78BFA',
                Entity: '#F87171',
              };
              return colors[type] || '#666';
            },
          },
        },
      ]);
    }
  }, []);

  if (isLoading) {
    return (
      <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 mb-4">Error loading graph</p>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data || (data.nodes.length === 0 && data.edges.length === 0)) {
    return (
      <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-500 mb-2">No nodes yet.</p>
          <p className="text-sm text-gray-400">Create one to get started!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white rounded-lg shadow-sm border">
      {/* Graph Controls */}
      <div className="p-3 border-b flex items-center justify-between bg-gray-50">
        <div className="flex items-center gap-4">
          <h3 className="font-semibold text-gray-800">Knowledge Graph</h3>
          <div className="text-xs text-gray-500">
            {data.nodes.length} nodes • {data.edges.length} relationships
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (cyRef.current) {
                cyRef.current.fit();
              }
            }}
            className="px-3 py-1 text-xs bg-white border rounded hover:bg-gray-50"
            title="Fit to view"
          >
            Fit
          </button>
          <button
            onClick={() => {
              if (cyRef.current) {
                cyRef.current.reset();
              }
            }}
            className="px-3 py-1 text-xs bg-white border rounded hover:bg-gray-50"
            title="Reset view"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="px-3 py-2 border-b bg-gray-50 flex items-center gap-4 text-xs">
        <span className="text-gray-600 font-medium">Legend:</span>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span>Observation</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 transform rotate-45"></div>
          <span>Hypothesis</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-500"></div>
          <span>Source</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-purple-500 transform rotate-30"></div>
          <span>Concept</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500"></div>
          <span>Entity</span>
        </div>
      </div>

      {/* Cytoscape Graph */}
      <div className="flex-1 relative">
        <CytoscapeComponent
          elements={elements}
          stylesheet={cytoscapeStylesheet}
          layout={{
            name: 'cose',
            animate: true,
            animationDuration: 1000,
            idealEdgeLength: 100,
            nodeRepulsion: 4500,
            nestingFactor: 0.1,
            gravity: 0.25,
            numIter: 2500,
            tile: true,
            tilingPaddingVertical: 10,
            tilingPaddingHorizontal: 10,
            gravityRangeCompound: 1.5,
            gravityCompound: 1.0,
            gravityRange: 3.8,
            initialEnergyOnIncremental: 0.3,
          }}
          style={{ width: '100%', height: '100%' }}
          cy={(cy) => {
            cyRef.current = cy;
            // Enable pan and zoom
            cy.userPanningEnabled(true);
            cy.boxSelectionEnabled(true);
            cy.zoomingEnabled(true);
            cy.minZoom(0.1);
            cy.maxZoom(2);
          }}
        />

        {/* Node Details Panel */}
        {selectedNodeId && (
          <div className="absolute top-4 right-4 bg-white border rounded-lg shadow-lg p-4 max-w-xs z-10">
            <div className="flex justify-between items-start mb-2">
              <h4 className="font-semibold text-sm">Node Details</h4>
              <button
                onClick={() => setSelectedNodeId(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
            {data.nodes.find((n: GraphNode) => n.id === selectedNodeId) && (
              <div className="text-xs space-y-1">
                <div>
                  <span className="text-gray-500">ID:</span>{' '}
                  <span className="font-mono">{selectedNodeId.substring(0, 12)}...</span>
                </div>
                <div>
                  <span className="text-gray-500">Type:</span>{' '}
                  {data.nodes.find((n: GraphNode) => n.id === selectedNodeId)?.type}
                </div>
                {data.nodes.find((n: GraphNode) => n.id === selectedNodeId)?.text && (
                  <div className="mt-2 pt-2 border-t">
                    <div className="text-gray-500 mb-1">Content:</div>
                    <div className="text-gray-700">
                      {data.nodes.find((n: GraphNode) => n.id === selectedNodeId)?.text}
                    </div>
                  </div>
                )}
                {data.nodes.find((n: GraphNode) => n.id === selectedNodeId)?.title && (
                  <div className="mt-2 pt-2 border-t">
                    <div className="text-gray-500 mb-1">Title:</div>
                    <div className="text-gray-700">
                      {data.nodes.find((n: GraphNode) => n.id === selectedNodeId)?.title}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
