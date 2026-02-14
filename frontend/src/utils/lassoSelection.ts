/**
 * Utility functions for lasso/box selection in Cytoscape
 */

interface Point {
  x: number;
  y: number;
}

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * Check if a point is inside a box
 */
export const isPointInBox = (point: Point, box: Box): boolean => {
  return (
    point.x >= Math.min(box.x1, box.x2) &&
    point.x <= Math.max(box.x1, box.x2) &&
    point.y >= Math.min(box.y1, box.y2) &&
    point.y <= Math.max(box.y1, box.y2)
  );
};

/**
 * Create a selection box div for visual feedback
 */
export const createSelectionBox = (
  container: HTMLElement,
  startX: number,
  startY: number
): HTMLDivElement => {
  const box = document.createElement('div');
  box.className = 
    'absolute border-2 border-dashed border-purple-500 ' +
    'bg-purple-500/20 pointer-events-none z-40';
  box.style.left = `${startX}px`;
  box.style.top = `${startY}px`;
  box.style.width = '0px';
  box.style.height = '0px';
  container.appendChild(box);
  return box;
};

/**
 * Update selection box dimensions
 */
export const updateSelectionBox = (
  box: HTMLDivElement,
  startX: number,
  startY: number,
  currentX: number,
  currentY: number
): void => {
  const left = Math.min(startX, currentX);
  const top = Math.min(startY, currentY);
  const width = Math.abs(currentX - startX);
  const height = Math.abs(currentY - startY);

  box.style.left = `${left}px`;
  box.style.top = `${top}px`;
  box.style.width = `${width}px`;
  box.style.height = `${height}px`;
};

/**
 * Get the final box coordinates in container-relative space
 */
export const getBoxCoordinates = (
  box: HTMLDivElement,
  container: HTMLElement
): Box => {
  const boxRect = box.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();

  return {
    x1: boxRect.left - containerRect.left,
    y1: boxRect.top - containerRect.top,
    x2: boxRect.right - containerRect.left,
    y2: boxRect.bottom - containerRect.top
  };
};

/**
 * Find all Cytoscape nodes inside a selection box
 */
export const findNodesInBox = (cy: any, box: Box): string[] => {
  const selectedIds: string[] = [];

  cy.nodes().forEach((node: any) => {
    const pos = node.renderedPosition();
    if (isPointInBox(pos, box)) {
      selectedIds.push(node.id());
    }
  });

  return selectedIds;
};

/**
 * Setup lasso selection handlers for Cytoscape
 * Returns cleanup function
 */
export const setupLassoSelection = (
  cy: any,
  onSelection: (nodeIds: string[]) => void,
  isLassoActive: () => boolean
): (() => void) => {
  const container = cy.container();
  
  let isDrawing = false;
  let startPos: Point | null = null;
  let selectionBox: HTMLDivElement | null = null;

  const handleMouseDown = (e: MouseEvent) => {
    if (!isLassoActive()) return;
    if (e.target !== container && !container.contains(e.target as Node)) return;
    if ((e.target as HTMLElement).closest('.node')) return; // Don't start on nodes

    e.preventDefault();
    isDrawing = true;
    startPos = { x: e.clientX, y: e.clientY };
    selectionBox = createSelectionBox(container, e.clientX, e.clientY);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDrawing || !startPos || !selectionBox) return;
    updateSelectionBox(selectionBox, startPos.x, startPos.y, e.clientX, e.clientY);
  };

  const handleMouseUp = () => {
    if (!isDrawing || !selectionBox || !startPos) return;

    // Get selected nodes
    const box = getBoxCoordinates(selectionBox, container);
    const selectedIds = findNodesInBox(cy, box);

    // Notify parent
    onSelection(selectedIds);

    // Select in cytoscape
    cy.nodes().unselect();
    selectedIds.forEach(id => {
      cy.getElementById(id).select();
    });

    // Cleanup
    selectionBox.remove();
    selectionBox = null;
    isDrawing = false;
    startPos = null;
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    // Cancel lasso on Escape
    if (e.key === 'Escape' && isDrawing) {
      if (selectionBox) {
        selectionBox.remove();
        selectionBox = null;
      }
      isDrawing = false;
      startPos = null;
    }
  };

  // Attach listeners
  container.addEventListener('mousedown', handleMouseDown);
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);
  window.addEventListener('keydown', handleKeyDown);

  // Return cleanup function
  return () => {
    container.removeEventListener('mousedown', handleMouseDown);
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
    window.removeEventListener('keydown', handleKeyDown);
    
    if (selectionBox) {
      selectionBox.remove();
    }
  };
};

export default {
  isPointInBox,
  createSelectionBox,
  updateSelectionBox,
  getBoxCoordinates,
  findNodesInBox,
  setupLassoSelection
};
