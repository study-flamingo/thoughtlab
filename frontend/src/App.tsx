import { useState } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import ActivityFeed from './components/ActivityFeed';
import CreateNodeModal from './components/CreateNodeModal';

function App() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-semibold text-gray-800">
          Research Connection Graph
        </h1>
        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center gap-2 transition-colors"
        >
          <span className="text-lg">+</span>
          Add Node
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph Visualizer (main area) */}
        <main className="flex-1 p-4">
          <GraphVisualizer />
        </main>

        {/* Activity Feed (sidebar) */}
        <aside className="w-80 bg-white border-l shadow-sm overflow-y-auto">
          <ActivityFeed />
        </aside>
      </div>

      {/* Create Node Modal */}
      {isCreateModalOpen && (
        <CreateNodeModal onClose={() => setIsCreateModalOpen(false)} />
      )}
    </div>
  );
}

export default App;
