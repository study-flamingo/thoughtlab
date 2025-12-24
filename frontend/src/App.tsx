import { useState, useEffect, useCallback } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import ActivityFeed from './components/ActivityFeed';
import NodeInspector from './components/NodeInspector';
import RelationInspector from './components/RelationInspector';
import CreateNodeModal from './components/CreateNodeModal';
import CreateRelationModal from './components/CreateRelationModal';
import SettingsModal from './components/SettingsModal';

function App() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isCreateRelationOpen, setIsCreateRelationOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Callback for edge selection - wrap in useCallback to ensure stability
  const handleEdgeSelect = useCallback((id: string | null) => {
    setSelectedEdgeId(id);
    // Only clear node selection when an edge is actually selected,
    // not when clearing edge selection (id === null)
    if (id !== null) {
      setSelectedNodeId(null);
    }
  }, []);

  // Callback for node selection - wrap in useCallback to ensure stability
  const handleNodeSelect = useCallback((id: string | null) => {
    setSelectedNodeId(id);
    // Only clear edge selection when a node is actually selected,
    // not when clearing node selection (id === null)
    if (id !== null) {
      setSelectedEdgeId(null);
    }
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b px-6 py-4 flex justify-between items-center dark:bg-gray-800 dark:border-gray-700">
        <h1 className="text-xl font-semibold font-[Geo] text-4xl text-gray-800 dark:text-gray-100">
          toughtlab.ai alpha 0.2.0
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="bg-white text-gray-800 px-3 py-2 rounded-md border hover:bg-gray-50 transition-colors dark:bg-gray-800 dark:text-gray-100 dark:border-gray-700 dark:hover:bg-gray-700"
            title="Open Settings"
          >
            Settings
          </button>
          <button
            onClick={() => setIsCreateRelationOpen(true)}
            className="bg-white text-gray-800 px-4 py-2 rounded-md border hover:bg-gray-50 transition-colors dark:bg-gray-800 dark:text-gray-100 dark:border-gray-700 dark:hover:bg-gray-700"
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
        <aside className="w-80 bg-white border-l shadow-sm overflow-hidden flex flex-col dark:bg-gray-800 dark:border-gray-700">
          {selectedEdgeId != null ? (
            <RelationInspector relationshipId={selectedEdgeId} onClose={() => setSelectedEdgeId(null)} />
          ) : selectedNodeId != null ? (
            <NodeInspector nodeId={selectedNodeId} onClose={() => setSelectedNodeId(null)} />
          ) : (
            <ActivityFeed onSelectNode={handleNodeSelect} />
          )}
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
      
      {/* Settings Modal */}
      {isSettingsOpen && (
        <SettingsModal onClose={() => setIsSettingsOpen(false)} />
      )}
    </div>
  );
}

export default App;
