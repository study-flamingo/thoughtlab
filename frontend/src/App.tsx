import { useState, useCallback, useEffect, useMemo } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import ActivityFeed from './components/ActivityFeed';
import NodeInspector from './components/NodeInspector';
import RelationInspector from './components/RelationInspector';
import CreateNodeModal from './components/CreateNodeModal';
import CreateRelationModal from './components/CreateRelationModal';
import SettingsModal from './components/SettingsModal';
import { ToastProvider } from './components/Toast';

// New layout components
import MinimalTopBar from './components/layout/MinimalTopBar';
import FloatingPanel from './components/layout/FloatingPanel';
import SlideDrawer, { DrawerMenuItemComponent, DrawerDivider, DrawerSection } from './components/layout/SlideDrawer';
import AIChatBubble from './components/ai/AIChatBubble';
import AIChatPanel from './components/ai/AIChatPanel';
import { Z_INDEX } from './types/layout';
import type { PanelState } from './types/layout';

// Memoize GraphVisualizer to prevent re-renders when panels change
import React from 'react';
const MemoizedGraphVisualizer = React.memo(GraphVisualizer);

function App() {
  // Modal state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isCreateRelationOpen, setIsCreateRelationOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Selection state
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Panel visibility state (separate from selection)
  const [panels, setPanels] = useState<PanelState>({
    nodeInspector: false,
    relationInspector: false,
    activityFeed: false,
    aiChat: false,
  });

  // Drawer state
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Auto-open inspector when node is selected
  useEffect(() => {
    if (selectedNodeId) {
      setPanels((p) => ({
        ...p,
        nodeInspector: true,
        relationInspector: false,
      }));
    }
  }, [selectedNodeId]);

  // Auto-open inspector when edge is selected
  useEffect(() => {
    if (selectedEdgeId) {
      setPanels((p) => ({
        ...p,
        relationInspector: true,
        nodeInspector: false,
      }));
    }
  }, [selectedEdgeId]);

  // Callback for edge selection
  const handleEdgeSelect = useCallback((id: string | null) => {
    setSelectedEdgeId(id);
    if (id !== null) {
      setSelectedNodeId(null);
    }
  }, []);

  // Callback for node selection
  const handleNodeSelect = useCallback((id: string | null) => {
    setSelectedNodeId(id);
    if (id !== null) {
      setSelectedEdgeId(null);
    }
  }, []);

  // Panel close handlers
  const handleCloseNodeInspector = useCallback(() => {
    setPanels((p) => ({ ...p, nodeInspector: false }));
  }, []);

  const handleCloseRelationInspector = useCallback(() => {
    setPanels((p) => ({ ...p, relationInspector: false }));
  }, []);

  const handleCloseActivityFeed = useCallback(() => {
    setPanels((p) => ({ ...p, activityFeed: false }));
  }, []);

  const handleCloseAIChat = useCallback(() => {
    setPanels((p) => ({ ...p, aiChat: false }));
  }, []);

  // Toggle handlers
  const toggleActivityFeed = useCallback(() => {
    setPanels((p) => ({ ...p, activityFeed: !p.activityFeed }));
  }, []);

  const toggleAIChat = useCallback(() => {
    setPanels((p) => ({ ...p, aiChat: !p.aiChat }));
  }, []);

  // Drawer menu items
  const drawerMenuItems = useMemo(
    () => [
      {
        id: 'settings',
        label: 'Settings',
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        ),
        onClick: () => {
          setIsSettingsOpen(true);
          setIsDrawerOpen(false);
        },
      },
      {
        id: 'activity',
        label: 'View Activity Feed',
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
        ),
        onClick: () => {
          setPanels((p) => ({ ...p, activityFeed: true }));
          setIsDrawerOpen(false);
        },
      },
      {
        id: 'ai-chat',
        label: 'Open AI Chat',
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        ),
        onClick: () => {
          setPanels((p) => ({ ...p, aiChat: true }));
          setIsDrawerOpen(false);
        },
      },
    ],
    []
  );

  return (
    <ToastProvider>
      <div className="h-screen w-screen overflow-hidden bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
        {/* Minimal Top Bar */}
        <MinimalTopBar
          onMenuClick={() => setIsDrawerOpen(true)}
          onAddNode={() => setIsCreateModalOpen(true)}
          onAddRelation={() => setIsCreateRelationOpen(true)}
          onNotificationClick={toggleActivityFeed}
          hasUnreadNotifications={false}
        />

        {/* Full-Screen Graph */}
        <main className="absolute inset-0 pt-14">
          <MemoizedGraphVisualizer
            onNodeSelect={handleNodeSelect}
            selectedNodeId={selectedNodeId}
            onEdgeSelect={handleEdgeSelect}
            selectedEdgeId={selectedEdgeId}
          />
        </main>

        {/* Slide Drawer (Menu) */}
        <SlideDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)}>
          <DrawerSection title="Navigation" />
          {drawerMenuItems.map((item) => (
            <DrawerMenuItemComponent key={item.id} {...item} />
          ))}
          <DrawerDivider />
          <DrawerSection title="Quick Actions" />
          <DrawerMenuItemComponent
            id="add-node"
            label="Add Node"
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            }
            onClick={() => {
              setIsCreateModalOpen(true);
              setIsDrawerOpen(false);
            }}
          />
          <DrawerMenuItemComponent
            id="add-relation"
            label="Add Relation"
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            }
            onClick={() => {
              setIsCreateRelationOpen(true);
              setIsDrawerOpen(false);
            }}
          />
        </SlideDrawer>

        {/* Node Inspector Panel */}
        <FloatingPanel
          isOpen={selectedNodeId != null && panels.nodeInspector}
          onClose={handleCloseNodeInspector}
          position="right"
          size="full-height"
          title="Node Inspector"
          zIndex={Z_INDEX.inspector}
        >
          {selectedNodeId && (
            <NodeInspector
              nodeId={selectedNodeId}
              onClose={() => setSelectedNodeId(null)}
            />
          )}
        </FloatingPanel>

        {/* Relation Inspector Panel */}
        <FloatingPanel
          isOpen={selectedEdgeId != null && panels.relationInspector}
          onClose={handleCloseRelationInspector}
          position="right"
          size="full-height"
          title="Relation Inspector"
          zIndex={Z_INDEX.inspector}
        >
          {selectedEdgeId && (
            <RelationInspector
              relationshipId={selectedEdgeId}
              onClose={() => setSelectedEdgeId(null)}
            />
          )}
        </FloatingPanel>

        {/* Activity Feed Panel */}
        <FloatingPanel
          isOpen={panels.activityFeed}
          onClose={handleCloseActivityFeed}
          position="top-right"
          size="md"
          title="Activity Feed"
          zIndex={Z_INDEX.activityFeed}
          className="h-[500px] max-h-[70vh]"
        >
          <ActivityFeed onSelectNode={handleNodeSelect} />
        </FloatingPanel>

        {/* AI Chat Bubble */}
        <AIChatBubble onClick={toggleAIChat} isExpanded={panels.aiChat} />

        {/* AI Chat Panel */}
        <AIChatPanel isOpen={panels.aiChat} onClose={handleCloseAIChat} />

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
    </ToastProvider>
  );
}

export default App;
