import React, { useEffect, useRef, useState } from 'react';
import Cytoscape from 'cytoscape';
import { getExtendedCytoscape } from '../lib/cytoscape-extensions';
import { graphApi } from '../services/api';
import type { GraphData, NodeType } from '../types/graph';

interface GraphVisualizerProps {
  onNodeSelect: (id: string | null) => void;
  onEdgeSelect: (id: string | null) => void;
  selectedNodeId?: string | null;
  selectedEdgeId?: string | null;
}

const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  onNodeSelect,
  onEdgeSelect,
  selectedNodeId,
  selectedEdgeId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Cytoscape.Core | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<GraphData | null>(null);

  // Node type colors
  const nodeColors: Record<NodeType, string> = {
    Observation: '#60A5FA',
    Hypothesis: '#34D399',
    Source: '#FBBF24',
    Concept: '#A78BFA',
    Entity: '#F87171',
  };

  // Load graph data
  useEffect(() => {
    let mounted = true;

    const loadGraph = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await graphApi.getFullGraph();
        if (mounted) {
          setData(response.data);
          setLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to load graph');
          setLoading(false);
        }
      }
    };

    loadGraph();

    return () => {
      mounted = false;
    };
  }, []);

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current || !data || loading) return;

    // Register extensions (ensures they're available)
    getExtendedCytoscape();

    // Initialize Cytoscape instance
    const cy = Cytoscape({
      container: containerRef.current,
      elements: [
        ...data.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.text,
            type: node.type,
            confidence: node.confidence,
          },
        })),
        ...data.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.type,
          },
        })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) =>
              nodeColors[ele.data('type') as NodeType] || '#6B7280',
            'label': 'data(label)',
            'color': '#fff',
            'text-outline-width': 2,
            'text-outline-color': '#000',
            'width': 40,
            'height': 40,
            'font-size': 12,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#9CA3AF',
            'target-arrow-color': '#9CA3AF',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': 10,
            'text-rotation': 'autorotate',
            'text-background-opacity': 1,
            'text-background-color': '#1F2937',
            'text-background-padding': '2px',
          },
        },
        {
          selector: ':selected',
          style: {
            'border-width': 3,
            'border-color': '#3B82F6',
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: true,
        animationDuration: 500,
        padding: 50,
      },
      wheelSensitivity: 0.2,
    });

    // Store reference
    cyRef.current = cy;

    // Initialize infinite dot grid using modular CSS arithmetic
    try {
      const gridSpacing = 25;
      const dotSize = 2.5;
      const dotColor = '#D1D5DB';

      const container = cy.container();
      if (container) {
        // Clean up any existing grid
        const existingGrid = container.querySelector('.css-dot-grid-bg');
        if (existingGrid) existingGrid.remove();

        // Create grid element
        const gridDiv = document.createElement('div');
        gridDiv.className = 'css-dot-grid-bg';
        gridDiv.style.cssText = `
          position: absolute; top: 0; left: 0; width: 100%; height: 100%;
          pointer-events: none; z-index: 0; opacity: 0.7;
          background-image: radial-gradient(circle, ${dotColor} ${dotSize}px, transparent ${dotSize + 1}px);
          background-size: ${gridSpacing}px ${gridSpacing}px;
        `;

        container.style.position = 'relative';
        container.insertBefore(gridDiv, container.firstChild);

        // Make Cytoscape canvas transparent
        const cyCanvas = container.querySelector('canvas');
        if (cyCanvas) cyCanvas.style.backgroundColor = 'transparent';

        // Modular arithmetic for infinite grid effect
        const updateGrid = () => {
          if (cy?.destroyed()) return;
          const { x, y } = cy.pan();
          const zoom = cy.zoom();
          const scaled = gridSpacing * zoom;
          const offset = (p: number) => ((p % scaled) + scaled) % scaled;
          gridDiv.style.backgroundPosition = `${offset(x)}px ${offset(y)}px`;
          gridDiv.style.backgroundSize = `${scaled}px ${scaled}px`;
        };

        cy.on('zoom pan resize', () => requestAnimationFrame(updateGrid));
        (cy as any)._gridEventHandler = updateGrid;
        updateGrid();
      }
    } catch (error) {
      console.error('Grid init failed:', error);
    }

    // Initialize gridGuide for snapping (without drawing grid lines)
    try {
      if (typeof cy.gridGuide === 'function') {
        cy.gridGuide({
          // Don't draw the line grid (we're using CSS dot grid)
          drawGrid: false,

          // Snapping behavior
          snapToGridOnRelease: true,
          snapToGridDuringDrag: false, // Smoother dragging, snap on release

          // Grid spacing must match our CSS grid
          gridSpacing: 25,
        });
      }
    } catch (error) {
      // Grid guide snapping is optional, fail silently
    }

    // Event handlers
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      onNodeSelect(node.id());
    });

    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      onEdgeSelect(edge.id());
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        onNodeSelect(null);
        onEdgeSelect(null);
      }
    });

    // Handle resize
    const handleResize = () => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit();
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (cyRef.current) {
        // Clean up grid event handler if it exists
        if ((cyRef.current as any)._gridEventHandler) {
          cyRef.current.off('zoom pan resize', (cyRef.current as any)._gridEventHandler);
        }
        cyRef.current.destroy();
        cyRef.current = null;
      }
      // Clean up CSS grid background
      if (containerRef.current) {
        const existingGrid = containerRef.current.querySelector('.css-dot-grid-bg');
        if (existingGrid) {
          existingGrid.remove();
        }
      }
    };
  }, [data, loading, onNodeSelect, onEdgeSelect]);

  // Handle selection changes
  useEffect(() => {
    if (!cyRef.current) return;

    // Clear previous selection
    cyRef.current.elements().unselect();

    // Select node if provided
    if (selectedNodeId) {
      const node = cyRef.current.getElementById(selectedNodeId);
      if (node.length > 0) {
        node.select();
      }
    }

    // Select edge if provided
    if (selectedEdgeId) {
      const edge = cyRef.current.getElementById(selectedEdgeId);
      if (edge.length > 0) {
        edge.select();
      }
    }
  }, [selectedNodeId, selectedEdgeId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center w-full h-full bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading graph...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center w-full h-full bg-gray-50 dark:bg-gray-900">
        <div className="text-center text-red-600 dark:text-red-400">
          <p className="font-semibold">Error loading graph</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  if (!data || (data.nodes.length === 0 && data.edges.length === 0)) {
    return (
      <div className="flex items-center justify-center w-full h-full bg-gray-50 dark:bg-gray-900">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <p className="font-semibold">No nodes yet.</p>
          <p className="text-sm mt-2">Create your first node to get started.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-white dark:bg-gray-950">
      {/* Header overlay */}
      <div className="absolute top-4 left-4 z-10 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg px-4 py-2 shadow-md">
        <div className="flex items-center gap-4">
          <h2 className="font-semibold text-gray-900 dark:text-gray-100">Knowledge Graph</h2>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {data.nodes.length} nodes • {data.edges.length} relationships
          </span>
        </div>
      </div>

      {/* Grid help indicator */}
      <div className="absolute bottom-4 left-4 z-10 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg px-3 py-2 shadow-md text-xs text-gray-600 dark:text-gray-400">
        Grid snapping enabled • Drag nodes to snap
      </div>

      {/* Cytoscape container */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default React.memo(GraphVisualizer);
