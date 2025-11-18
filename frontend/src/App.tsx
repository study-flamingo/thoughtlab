import { useState, useEffect, useCallback } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import ActivityFeed from './components/ActivityFeed';
import NodeInspector from './components/NodeInspector';
import RelationInspector from './components/RelationInspector';
import CreateNodeModal from './components/CreateNodeModal';
import CreateRelationModal from './components/CreateRelationModal';

function App() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isCreateRelationOpen, setIsCreateRelationOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Debug: Track state changes
  useEffect(() => {
    console.log('useEffect: selectedEdgeId changed to:', selectedEdgeId);
  }, [selectedEdgeId]);

  // Callback for edge selection - wrap in useCallback to ensure stability
  const handleEdgeSelect = useCallback((id: string | null) => {
    console.log('handleEdgeSelect CALLED with id:', id);
    // Use functional update to ensure React processes it
    setSelectedEdgeId(() => {
      console.log('Inside setSelectedEdgeId updater, returning:', id);
      return id;
    });
    setSelectedNodeId(() => null);
  }, []);

  // Callback for node selection
  const handleNodeSelect = (id: string | null) => {
    setSelectedNodeId(id);
    // Only clear edge selection when a node is actually selected.
    // Do NOT clear when the intent is just to clear node selection (id === null),
    // e.g. when an edge is selected and we programmatically clear node selection.
    if (id !== null) {
      setSelectedEdgeId(null);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-semibold text-gray-800">
          Research Connection Graph
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsCreateRelationOpen(true)}
            className="bg-white text-gray-800 px-4 py-2 rounded-md border hover:bg-gray-50 transition-colors"
          >
            Add Relation
          </button>
          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center gap-2 transition-colors"
          >
            <span className="text-lg">+</span>
            Add Node
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph Visualizer (main area) */}
        <main className="flex-1 p-4">
          <GraphVisualizer 
            onNodeSelect={handleNodeSelect}
            selectedNodeId={selectedNodeId}
            onEdgeSelect={handleEdgeSelect}
            selectedEdgeId={selectedEdgeId}
          />
        </main>

        {/* Sidebar - Show RelationInspector if edge selected, NodeInspector if node selected, otherwise ActivityFeed */}
        <aside className="w-80 bg-white border-l shadow-sm overflow-hidden flex flex-col">
          {(() => {
            console.log('Sidebar render check - selectedEdgeId:', selectedEdgeId, 'selectedNodeId:', selectedNodeId);
            if (selectedEdgeId !== null && selectedEdgeId !== undefined) {
              console.log('Rendering RelationInspector');
              return <RelationInspector relationshipId={selectedEdgeId} onClose={() => setSelectedEdgeId(null)} />;
            } else if (selectedNodeId !== null && selectedNodeId !== undefined) {
              return <NodeInspector nodeId={selectedNodeId} onClose={() => setSelectedNodeId(null)} />;
            } else {
              return <ActivityFeed />;
            }
          })()}
        </aside>
      </div>

      {/* Create Node Modal */}
      {isCreateModalOpen && (
        <CreateNodeModal onClose={() => setIsCreateModalOpen(false)} />
      )}

      {/* Create Relation Modal */}
      {isCreateRelationOpen && (
        <CreateRelationModal onClose={() => setIsCreateRelationOpen(false)} />
      )}
    </div>
  );
}

export default App;
