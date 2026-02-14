import { useEffect, useCallback } from 'react';
import { useGraphStore } from '../store/graphStore';
import { useModalStore } from '../store/modalStore';

export type ToolType = 'selector' | 'lasso';

interface UseGraphToolsReturn {
  activeTool: ToolType;
  setActiveTool: (tool: ToolType) => void;
  selection: string[];
  selectAll: () => void;
  deselectAll: () => void;
  selectInverse: () => void;
  cutSelection: () => void;
  copySelection: () => void;
  pasteSelection: () => void;
  reorganizeOthers: () => void;
  reorganizeSelection: () => void;
  isGraphFocused: boolean;
}

export const useGraphTools = (): UseGraphToolsReturn => {
  const { 
    cy,
    selection, 
    setSelection,
    cutNodes,
    copyNodes,
    pasteNodes,
    clipboard
  } = useGraphStore();
  
  const { isAnyModalOpen } = useModalStore();
  const activeTool = useGraphStore((state) => state.activeTool);
  const setActiveTool = useGraphStore((state) => state.setActiveTool);

  const isGraphFocused = !isAnyModalOpen;

  // Selection operations
  const selectAll = useCallback(() => {
    if (!cy || !isGraphFocused) return;
    const allNodes = cy.nodes().map((n: any) => n.id());
    setSelection(allNodes);
    cy.nodes().select();
  }, [cy, isGraphFocused, setSelection]);

  const deselectAll = useCallback(() => {
    if (!cy || !isGraphFocused) return;
    setSelection([]);
    cy.nodes().unselect();
  }, [cy, isGraphFocused, setSelection]);

  const selectInverse = useCallback(() => {
    if (!cy || !isGraphFocused) return;
    const allNodeIds = cy.nodes().map((n: any) => n.id());
    const inverseIds = allNodeIds.filter((id: string) => !selection.includes(id));
    setSelection(inverseIds);
    cy.nodes().unselect();
    inverseIds.forEach((id: string) => {
      cy.getElementById(id).select();
    });
  }, [cy, isGraphFocused, selection, setSelection]);

  // Clipboard operations (placeholders - implement based on your needs)
  const cutSelection = useCallback(() => {
    if (!cy || !isGraphFocused || selection.length === 0) return;
    cutNodes(selection);
  }, [cy, isGraphFocused, selection, cutNodes]);

  const copySelection = useCallback(() => {
    if (!cy || !isGraphFocused || selection.length === 0) return;
    copyNodes(selection);
  }, [cy, isGraphFocused, selection, copyNodes]);

  const pasteSelection = useCallback(() => {
    if (!cy || !isGraphFocused || clipboard.length === 0) return;
    pasteNodes();
  }, [cy, isGraphFocused, clipboard.length, pasteNodes]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isGraphFocused) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Tool switching (single keys, no modifiers)
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        switch (e.key.toLowerCase()) {
          case 's':
            e.preventDefault();
            setActiveTool('selector');
            return;
          case 'm':
            e.preventDefault();
            setActiveTool('lasso');
            return;
        }
      }

      // Selection operations (with modifiers)
      const isCtrlOrCmd = e.ctrlKey || e.metaKey;
      
      if (isCtrlOrCmd) {
        // Ctrl/Cmd + A: Select All
        if (e.key.toLowerCase() === 'a' && !e.shiftKey) {
          e.preventDefault();
          selectAll();
          return;
        }

        // Ctrl/Cmd + Shift + A: Deselect All
        if (e.key.toLowerCase() === 'a' && e.shiftKey) {
          e.preventDefault();
          deselectAll();
          return;
        }

        // Ctrl/Cmd + Shift + I: Select Inverse
        if (e.key.toLowerCase() === 'i' && e.shiftKey) {
          e.preventDefault();
          selectInverse();
          return;
        }

        // Ctrl/Cmd + X: Cut
        if (e.key.toLowerCase() === 'x' && !e.shiftKey) {
          e.preventDefault();
          cutSelection();
          return;
        }

        // Ctrl/Cmd + C: Copy
        if (e.key.toLowerCase() === 'c' && !e.shiftKey) {
          e.preventDefault();
          copySelection();
          return;
        }

        // Ctrl/Cmd + V: Paste
        if (e.key.toLowerCase() === 'v' && !e.shiftKey) {
          e.preventDefault();
          pasteSelection();
          return;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    isGraphFocused, 
    setActiveTool, 
    selectAll, 
    deselectAll, 
    selectInverse,
    cutSelection,
    copySelection,
    pasteSelection
  ]);

  return {
    activeTool,
    setActiveTool,
    selection,
    selectAll,
    deselectAll,
    selectInverse,
    cutSelection,
    copySelection,
    pasteSelection,
    isGraphFocused
  };
};
