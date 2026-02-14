# Graph Toolbar Implementation

This implementation adds a Photoshop-style toolbar to ThoughtLab with selection tools, keyboard shortcuts, and modal-aware focus management.

## Files Created

1. **`src/hooks/useGraphTools.ts`** - Main hook for tool state, selection operations, and keyboard shortcuts
2. **`src/components/GraphToolbar.tsx`** - The toolbar UI component
3. **`src/store/modalStore.ts`** - Zustand store for tracking modal state (focus management)

## Features

### Tools
- **Selector (S)** - Default click-to-select tool
- **Lasso (M)** - Box/lasso selection for multi-select

### Selection Operations
- **Select All (Ctrl+A)** - Select all nodes
- **Deselect All (Ctrl+Shift+A)** - Clear selection
- **Select Inverse (Ctrl+Shift+I)** - Invert current selection

### Clipboard
- **Cut (Ctrl+X)** - Cut selected nodes
- **Copy (Ctrl+C)** - Copy selected nodes
- **Paste (Ctrl+V)** - Paste nodes from clipboard

### Smart Focus Management
- Shortcuts only work when no modals are open
- Modal backdrop click closes modal and refocuses graph
- Visual indicator when graph has focus (toolbar opacity changes)

## Integration Steps

### 1. Install Dependencies

```bash
npm install lucide-react
```

### 2. Add to GraphStore

Update your existing `graphStore.ts` to include the new fields:

```typescript
interface GraphState {
  // ... existing fields
  
  // Tool state
  activeTool: 'selector' | 'lasso';
  setActiveTool: (tool: 'selector' | 'lasso') => void;
  
  // Clipboard
  clipboard: any[];
  cutNodes: (nodeIds: string[]) => void;
  copyNodes: (nodeIds: string[]) => void;
  pasteNodes: () => void;
}
```

### 3. Wire Up Modals

Make sure your existing modals use the `modalStore`:

```typescript
// In your NodeInspector component
import { useModalStore } from '../store/modalStore';

const NodeInspector = ({ isOpen, node }) => {
  const { openModal, closeModal } = useModalStore();
  
  useEffect(() => {
    if (isOpen) {
      openModal('nodeInspector', node);
    } else {
      closeModal();
    }
    
    return () => closeModal();
  }, [isOpen, node]);
  
  // ... rest of component
};
```

### 4. Add Toolbar to Graph View

```typescript
import { GraphToolbar } from './components/GraphToolbar';

const GraphView = () => {
  return (
    <div className="relative w-full h-full">
      {/* Your existing cytoscape container */}
      <div ref={cyRef} className="w-full h-full" />
      
      {/* Add the toolbar */}
      <GraphToolbar accentColor="#9900EB" />
      
      {/* Your existing modals */}
      <NodeInspector />
      <ChatModal />
    </div>
  );
};
```

### 5. Implement Lasso Selection

Add this to your Cytoscape initialization to support the lasso tool:

```typescript
useEffect(() => {
  if (!cy) return;
  
  let isDrawing = false;
  let startPos: { x: number; y: number } | null = null;
  let selectionBox: HTMLDivElement | null = null;
  
  const handleMouseDown = (e: MouseEvent) => {
    if (activeTool !== 'lasso') return;
    if (e.target !== cy.container()) return; // Only on canvas, not nodes
    
    isDrawing = true;
    startPos = { x: e.clientX, y: e.clientY };
    
    // Create visual selection box
    selectionBox = document.createElement('div');
    selectionBox.className = 'absolute border-2 border-dashed border-purple-500 bg-purple-500/20 pointer-events-none z-40';
    selectionBox.style.left = `${e.clientX}px`;
    selectionBox.style.top = `${e.clientY}px`;
    cy.container().appendChild(selectionBox);
  };
  
  const handleMouseMove = (e: MouseEvent) => {
    if (!isDrawing || !startPos || !selectionBox) return;
    
    const width = Math.abs(e.clientX - startPos.x);
    const height = Math.abs(e.clientY - startPos.y);
    const left = Math.min(e.clientX, startPos.x);
    const top = Math.min(e.clientY, startPos.y);
    
    selectionBox.style.width = `${width}px`;
    selectionBox.style.height = `${height}px`;
    selectionBox.style.left = `${left}px`;
    selectionBox.style.top = `${top}px`;
  };
  
  const handleMouseUp = (e: MouseEvent) => {
    if (!isDrawing || !startPos || !selectionBox) return;
    
    // Calculate selection box in graph coordinates
    const box = selectionBox.getBoundingClientRect();
    const containerRect = cy.container().getBoundingClientRect();
    
    const x1 = box.left - containerRect.left;
    const y1 = box.top - containerRect.top;
    const x2 = x1 + box.width;
    const y2 = y1 + box.height;
    
    // Find nodes inside box
    const selectedNodes = cy.nodes().filter((node: any) => {
      const pos = node.renderedPosition();
      return pos.x >= x1 && pos.x <= x2 && pos.y >= y1 && pos.y <= y2;
    });
    
    // Update selection
    const selectedIds = selectedNodes.map((n: any) => n.id());
    setSelection(selectedIds);
    selectedNodes.select();
    
    // Cleanup
    selectionBox.remove();
    selectionBox = null;
    isDrawing = false;
    startPos = null;
  };
  
  const container = cy.container();
  container.addEventListener('mousedown', handleMouseDown);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);
  
  return () => {
    container.removeEventListener('mousedown', handleMouseDown);
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
  };
}, [cy, activeTool]);
```

### 6. Implement Clipboard Operations

Add to your graphStore:

```typescript
cutNodes: (nodeIds) => {
  const nodes = get().cy?.nodes().filter((n: any) => nodeIds.includes(n.id()));
  if (!nodes) return;
  
  const nodeData = nodes.map((n: any) => ({
    ...n.data(),
    position: n.position()
  }));
  
  set({ clipboard: nodeData });
  nodes.remove(); // Remove from graph
  set({ selection: [] });
},

copyNodes: (nodeIds) => {
  const nodes = get().cy?.nodes().filter((n: any) => nodeIds.includes(n.id()));
  if (!nodes) return;
  
  const nodeData = nodes.map((n: any) => ({
    ...n.data(),
    position: n.position()
  }));
  
  set({ clipboard: nodeData });
},

pasteNodes: () => {
  const { clipboard, cy } = get();
  if (!cy || clipboard.length === 0) return;
  
  const newIds: string[] = [];
  
  clipboard.forEach((nodeData, index) => {
    const newId = `${nodeData.id}_copy_${Date.now()}_${index}`;
    newIds.push(newId);
    
    cy.add({
      group: 'nodes',
      data: { ...nodeData, id: newId, name: `${nodeData.name} (copy)` },
      position: {
        x: nodeData.position.x + 50 + (index * 20),
        y: nodeData.position.y + 50 + (index * 20)
      }
    });
  });
  
  set({ selection: newIds });
  cy.getElementById(newIds[0]).select();
},
```

## Next Steps

### Automove Integration (Future)
When you're ready to add the selective reorganization feature:

```typescript
// Add to toolbar
<button onClick={() => reorganizeWithAnchors(selection)}>
  Reorganize Others
</button>

// Implementation
const reorganizeWithAnchors = (anchoredNodeIds: string[]) => {
  // Lock selected nodes with automove
  anchoredNodeIds.forEach(id => {
    cy.getElementById(id).automove({
      mode: 'lock'
    });
  });
  
  // Run layout on everything else
  cy.layout({
    name: 'cola',
    fit: false,
    infinite: true,
    handleDisconnected: true
  }).run();
  
  // Release locks after layout settles
  setTimeout(() => {
    anchoredNodeIds.forEach(id => {
      cy.getElementById(id).automove('destroy');
    });
  }, 1000);
};
```

### Paste Special Modal
For handling different paste types (URL, text, JSON):

```typescript
// Detect content type
const detectPasteType = (content: string): 'nodes' | 'url' | 'text' | 'json' => {
  try {
    const parsed = JSON.parse(content);
    if (parsed.nodes) return 'nodes';
    return 'json';
  } catch {
    if (content.startsWith('http')) return 'url';
    if (content.length > 0) return 'text';
    return 'nodes';
  }
};

// Show modal for non-node content
if (pasteType !== 'nodes') {
  openModal('pasteSpecial', { type: pasteType, content });
}
```

## Customization

### Accent Color
Pass your preferred accent color to the toolbar:

```typescript
<GraphToolbar accentColor="#FF6B35" />  // Orange
<GraphToolbar accentColor="#00D9FF" />  // Cyan
<GraphToolbar accentColor="#9900EB" />  // Purple (default)
```

### Positioning
The toolbar is positioned absolute in the UR corner. Adjust in `GraphToolbar.tsx`:

```typescript
// Change from:
className="absolute top-4 right-4 ..."

// To:
className="absolute top-4 left-4 ..."  // UL corner
className="absolute bottom-4 right-4 ..."  // LR corner
```

## Notes

- The toolbar uses `lucide-react` for icons — make sure it's installed
- All keyboard shortcuts respect modal state (no shortcuts when modals open)
- Selection counter appears when nodes are selected
- Tool buttons show their shortcut keys below the icon
