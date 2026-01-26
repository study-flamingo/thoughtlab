import React, { useEffect, useRef } from 'react';
import Cytoscape from 'cytoscape';
import { useQuery } from '@tanstack/react-query';
import { getExtendedCytoscape } from '../lib/cytoscape-extensions';
import { graphApi } from '../services/api';
import type { NodeType } from '../types/graph';

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
  const [zoomLevel, setZoomLevel] = React.useState(1);

  // Fetch graph data using React Query - auto-refreshes when other components invalidate ['graph']
  const {
    data,
    isLoading: loading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['graph', 'full'],
    queryFn: async () => {
      const response = await graphApi.getFullGraph();
      return response.data;
    },
  });

  // Light theme node colors (more saturated, visible on white)
  const nodeColors: Record<NodeType, string> = {
    Observation: '#3B82F6', // blue-500
    Hypothesis: '#10B981', // emerald-500
    Source: '#F59E0B', // amber-500
    Concept: '#8B5CF6', // violet-500
    Entity: '#EF4444', // red-500
  };

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
            'color': '#374151', // gray-700 for light theme
            'text-rotation': 'autorotate',
            'text-background-opacity': 1,
            'text-background-color': '#F9FAFB', // gray-50 for light theme
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
        {
          selector: 'node.highlighted',
          style: {
            'border-width': 4,
            'border-color': '#F59E0B',
            'border-opacity': 1,
          },
        },
        {
          selector: 'edge.highlighted',
          style: {
            'line-color': '#F59E0B',
            'target-arrow-color': '#F59E0B',
            'width': 3,
            'z-index': 10,
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: true,
        animationDuration: 500,
        padding: 50,
      },
      wheelSensitivity: 0.1, // Reduced from 0.2 for finer zoom control
    });

    // Store reference
    cyRef.current = cy;

    // Set better default zoom after layout completes
    cy.one('layoutstop', () => {
      // Fit the graph with some padding, then zoom out a bit for better overview
      cy.fit(50); // Fit with 50px padding
      const currentZoom = cy.zoom();
      const betterZoom = currentZoom * 0.7; // Zoom out to 70% of fitted zoom
      cy.zoom(betterZoom);
      cy.center();
      setZoomLevel(betterZoom);
    });

    // Initialize infinite dot grid using Canvas with fixed 1px dots
    try {
      const gridSpacing = 25;
      const dotColor = '#D1D5DB';

      const container = cy.container();
      if (container) {
        // Clean up any existing grid
        const existingGrid = container.querySelector('.grid-canvas-bg');
        if (existingGrid) existingGrid.remove();

        // Create canvas element
        const gridCanvas = document.createElement('canvas');
        gridCanvas.className = 'grid-canvas-bg';
        gridCanvas.style.cssText = `
          position: absolute; top: 0; left: 0;
          pointer-events: none; z-index: 0;
        `;

        // Size canvas to container
        const rect = container.getBoundingClientRect();
        gridCanvas.width = rect.width;
        gridCanvas.height = rect.height;

        const ctx = gridCanvas.getContext('2d');
        if (!ctx) throw new Error('Canvas context unavailable');

        container.style.position = 'relative';
        container.insertBefore(gridCanvas, container.firstChild);

        // Make Cytoscape canvas transparent
        const cyCanvas = container.querySelector('canvas');
        if (cyCanvas) cyCanvas.style.backgroundColor = 'transparent';

        // Render grid with fixed 1px dots
        const updateGrid = () => {
          if (cy?.destroyed() || !ctx) return;

          const zoom = cy.zoom();
          const { x, y } = cy.pan();

          // Clear canvas
          ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

          // Calculate visible grid range
          const scaledSpacing = gridSpacing * zoom;
          const startX = Math.floor(-x / scaledSpacing);
          const endX = Math.ceil((gridCanvas.width - x) / scaledSpacing);
          const startY = Math.floor(-y / scaledSpacing);
          const endY = Math.ceil((gridCanvas.height - y) / scaledSpacing);

          // Draw 1px dots at screen coordinates
          ctx.fillStyle = dotColor;
          for (let i = startX; i <= endX; i++) {
            for (let j = startY; j <= endY; j++) {
              const screenX = (i * scaledSpacing) + x;
              const screenY = (j * scaledSpacing) + y;

              // Draw exactly 1px dot (0.5 offset for center alignment)
              ctx.fillRect(screenX - 0.5, screenY - 0.5, 1, 1);
            }
          }
        };

        // Handle canvas resize
        const handleCanvasResize = () => {
          if (container) {
            const rect = container.getBoundingClientRect();
            gridCanvas.width = rect.width;
            gridCanvas.height = rect.height;
            updateGrid();
          }
        };

        cy.on('zoom pan resize', () => requestAnimationFrame(updateGrid));
        (cy as any)._gridEventHandler = updateGrid;
        (cy as any)._gridResizeHandler = handleCanvasResize;
        updateGrid();
      }
    } catch (error) {
      console.error('Grid init failed:', error);
    }

    // Note: Grid snapping removed - using manual canvas grid without snapping extension

    // Event handlers with neighborhood highlighting
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      cy.elements().removeClass('highlighted');
      node.addClass('highlighted');
      node.neighborhood().addClass('highlighted');
      onNodeSelect(node.id());
    });

    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      cy.elements().removeClass('highlighted');
      edge.addClass('highlighted');
      edge.source().addClass('highlighted');
      edge.target().addClass('highlighted');
      onEdgeSelect(edge.id());
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted');
        onNodeSelect(null);
        onEdgeSelect(null);
      }
    });

    // Sync zoom level state with Cytoscape zoom changes
    cy.on('zoom', () => {
      setZoomLevel(cy.zoom());
    });

    // Handle resize
    const handleResize = () => {
      if (cyRef.current) {
        // Trigger resize handler for canvas grid if it exists
        if ((cyRef.current as any)._gridResizeHandler) {
          (cyRef.current as any)._gridResizeHandler();
        }
        cyRef.current.resize();
        cyRef.current.fit();
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (cyRef.current) {
        // Clean up grid event handlers if they exist
        if ((cyRef.current as any)._gridEventHandler) {
          cyRef.current.off('zoom pan resize', (cyRef.current as any)._gridEventHandler);
        }
        cyRef.current.destroy();
        cyRef.current = null;
      }
      // Clean up canvas grid background
      if (containerRef.current) {
        const existingGrid = containerRef.current.querySelector('.grid-canvas-bg');
        if (existingGrid) {
          existingGrid.remove();
        }
      }
    };
  }, [data, loading, onNodeSelect, onEdgeSelect]);

  // Handle selection and highlighting changes from external sources
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || cy.destroyed()) return;

    // Clear previous selection and highlighting
    cy.elements().unselect();
    cy.elements().removeClass('highlighted');

    // Select and highlight node if provided
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId);
      if (node.length > 0) {
        node.select();
        node.addClass('highlighted');
        node.neighborhood().addClass('highlighted');
      }
    }

    // Select and highlight edge if provided
    if (selectedEdgeId) {
      const edge = cy.getElementById(selectedEdgeId);
      if (edge.length > 0) {
        edge.select();
        edge.addClass('highlighted');
        edge.source().addClass('highlighted');
        edge.target().addClass('highlighted');
      }
    }
  }, [selectedNodeId, selectedEdgeId]);

  // Handle zoom slider changes
  const handleZoomChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newZoom = parseFloat(event.target.value);
    setZoomLevel(newZoom);
    if (cyRef.current && !cyRef.current.destroyed()) {
      cyRef.current.zoom(newZoom);
    }
  };

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
          <p className="text-sm mt-2">{error instanceof Error ? error.message : 'Failed to load graph'}</p>
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

          {/* Zoom slider */}
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
            <input
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={zoomLevel}
              onChange={handleZoomChange}
              className="w-24 h-1.5 bg-gray-300 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
              title={`Zoom: ${Math.round(zoomLevel * 100)}%`}
            />
            <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
            </svg>
            <span className="text-xs text-gray-500 dark:text-gray-400 w-10 text-right">{Math.round(zoomLevel * 100)}%</span>
          </div>

          <button
            onClick={() => refetch()}
            className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400"
            title="Refresh graph"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Cytoscape container */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default React.memo(GraphVisualizer);
