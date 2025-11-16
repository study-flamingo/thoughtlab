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

interface Props {
  onNodeSelect?: (nodeId: string | null) => void;
  selectedNodeId?: string | null;
}

export default function GraphVisualizer({ onNodeSelect, selectedNodeId: externalSelectedNodeId }: Props) {
  const cyRef = useRef<Core | null>(null);
  const [isReady, setIsReady] = useState<boolean>(false);
  const setupDoneRef = useRef<boolean>(false);
  const onNodeSelectRef = useRef(onNodeSelect);
  const [internalSelectedNodeId, setInternalSelectedNodeId] = useState<string | null>(null);
  
  // Keep the ref updated with the latest callback
  useEffect(() => {
    onNodeSelectRef.current = onNodeSelect;
  }, [onNodeSelect]);
  
  // Use external selectedNodeId if provided, otherwise use internal state
  const selectedNodeId = externalSelectedNodeId !== undefined ? externalSelectedNodeId : internalSelectedNodeId;
  
  const setSelectedNodeId = (id: string | null) => {
    if (onNodeSelectRef.current) {
      onNodeSelectRef.current(id);
    } else {
      setInternalSelectedNodeId(id);
    }
  };

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
        ...data.nodes.map((node: GraphNode) => {
          // Determine label based on node type
          let label = node.id.substring(0, 8);
          if (node.text) {
            label = node.text.substring(0, 30) + (node.text.length > 30 ? '...' : '');
          } else if (node.title) {
            label = node.title.substring(0, 30) + (node.title.length > 30 ? '...' : '');
          } else if (node.name) {
            label = node.name.substring(0, 30) + (node.name.length > 30 ? '...' : '');
          }
          
          return {
            data: {
              id: node.id,
              label,
              type: node.type,
              ...node,
            },
          };
        }),
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

  // Set up tap handler when Cytoscape instance and data are ready
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || cy.destroyed() || !data || !isReady) {
      console.log('Tap handler setup skipped:', { cy: !!cy, destroyed: cy?.destroyed(), data: !!data, isReady }); // Debug log
      return;
    }

    console.log('Setting up tap handler'); // Debug log

    const handleTap = (event: cytoscape.EventObject) => {
      try {
        const target = event.target;
        console.log('Tap event fired, target:', target); // Debug log
        if (target.isNode()) {
          const nodeId = target.id();
          console.log('Node tapped:', nodeId); // Debug log
          setSelectedNodeId(nodeId);
          // Highlight selected node and its neighbors
          if (!cy.destroyed()) {
            cy.elements().removeClass('highlighted');
            target.addClass('highlighted');
            target.neighborhood().addClass('highlighted');
          }
        } else {
          console.log('Background tapped, deselecting'); // Debug log
          setSelectedNodeId(null);
          if (!cy.destroyed()) {
            cy.elements().removeClass('highlighted');
          }
        }
      } catch (error) {
        console.warn('Error handling tap event:', error);
      }
    };

    // Remove any existing tap handlers first to avoid duplicates
    cy.off('tap');
    cy.on('tap', handleTap);
    console.log('Tap handler attached'); // Debug log

    return () => {
      if (cy && !cy.destroyed()) {
        try {
          cy.off('tap', handleTap);
          console.log('Tap handler removed'); // Debug log
        } catch (error) {
          console.warn('Error removing tap handler:', error);
        }
      }
    };
  }, [data, isReady]); // setSelectedNodeId uses ref, so no need in deps

  // Add highlight styles
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !isReady || cy.destroyed()) return;

    try {
      cy.style().append([
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
    } catch (error) {
      console.warn('Failed to append highlight styles:', error);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const cy = cyRef.current;
      if (cy && !cy.destroyed()) {
        try {
          cy.destroy();
        } catch (error) {
          console.warn('Error destroying Cytoscape instance:', error);
        }
      }
      setIsReady(false);
      setupDoneRef.current = false;
    };
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
              const cy = cyRef.current;
              if (cy && !cy.destroyed() && isReady) {
                try {
                  cy.fit();
                } catch (error) {
                  console.warn('Failed to fit graph:', error);
                }
              }
            }}
            className="px-3 py-1 text-xs bg-white border rounded hover:bg-gray-50"
            title="Fit to view"
          >
            Fit
          </button>
          <button
            onClick={() => {
              const cy = cyRef.current;
              if (cy && !cy.destroyed() && isReady) {
                try {
                  cy.reset();
                } catch (error) {
                  console.warn('Failed to reset graph:', error);
                }
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
            animate: false, // Disable animation to prevent race conditions
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
            if (!cy) {
              setIsReady(false);
              setupDoneRef.current = false;
              return;
            }
            
            // Guard against destroyed instances
            if (cy.destroyed()) {
              setIsReady(false);
              setupDoneRef.current = false;
              return;
            }

            // Prevent multiple setups if already done
            if (setupDoneRef.current && cyRef.current === cy) {
              console.log('Already set up, skipping'); // Debug log
              // Don't reset isReady if already set up
              return;
            }

            cyRef.current = cy;
            // Only reset isReady if we're doing a fresh setup
            if (!setupDoneRef.current) {
              setIsReady(false);
              console.log('Resetting isReady to false for new setup'); // Debug log
            }

            // Set up event listeners and configuration
            const setupCytoscape = () => {
              try {
                console.log('setupCytoscape called', { destroyed: cy.destroyed(), setupDone: setupDoneRef.current }); // Debug log
                if (cy.destroyed() || setupDoneRef.current) {
                  console.log('setupCytoscape skipped - already done or destroyed'); // Debug log
                  return;
                }

                // Enable pan and zoom
                cy.userPanningEnabled(true);
                cy.boxSelectionEnabled(true);
                cy.zoomingEnabled(true);
                cy.minZoom(0.1);
                cy.maxZoom(2);

                // Note: Tap handler is set up in useEffect when data is ready

                setupDoneRef.current = true;
                console.log('setupDoneRef set to true'); // Debug log

                // Mark as ready immediately since we've set up the instance
                // The ready event will also fire, but we can mark ready now
                setIsReady(true);
                console.log('Cytoscape instance marked as ready immediately'); // Debug log
              } catch (error) {
                console.warn('Error setting up Cytoscape:', error);
                setIsReady(false);
                setupDoneRef.current = false;
              }
            };

            // Listen for ready event (only once)
            if (!setupDoneRef.current) {
              console.log('Setting up ready handler'); // Debug log
              const readyHandler = () => {
                console.log('Ready event fired'); // Debug log
                // Ensure ready state is set when ready event fires
                if (!cy.destroyed() && cy === cyRef.current) {
                  setIsReady(true);
                  console.log('Ready event: marking instance as ready'); // Debug log
                }
              };

              cy.on('ready', readyHandler);
              
              // Also set up immediately if already ready
              try {
                if (cy.container()) {
                  console.log('Container exists, calling setupCytoscape immediately'); // Debug log
                  setupCytoscape();
                } else {
                  console.log('No container yet, scheduling setupCytoscape'); // Debug log
                  // Fallback: set up after a short delay
                  setTimeout(() => {
                    if (!cy.destroyed() && cy === cyRef.current) {
                      setupCytoscape();
                    }
                  }, 150);
                }
              } catch (error) {
                console.log('Container check failed, using timeout fallback', error); // Debug log
                // If container check fails, use timeout fallback
                setTimeout(() => {
                  if (!cy.destroyed() && cy === cyRef.current) {
                    setupCytoscape();
                  }
                }, 150);
              }
            } else {
              console.log('setupDoneRef already true, skipping setup'); // Debug log
            }
          }}
        />

      </div>
    </div>
  );
}
