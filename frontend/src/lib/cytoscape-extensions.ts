import Cytoscape from 'cytoscape';
// Import extensions - type definitions come from ../types/*.d.ts files
import gridGuide from 'cytoscape-grid-guide';
import navigator from 'cytoscape-navigator';

// Register extensions once with guards against double registration
// This is critical for Vite HMR (Hot Module Replacement)

export function registerCytoscapeExtensions(): void {
  // Check if gridGuide is already registered
  try {
    const hasGridGuide = typeof Cytoscape('core', 'gridGuide') === 'function';
    if (!hasGridGuide) {
      // Cast to any because Cytoscape.use exists at runtime but may not be in type definitions
      (Cytoscape as any).use(gridGuide);
      console.log('[Cytoscape] gridGuide extension registered');
    }
  } catch (error) {
    console.warn('[Cytoscape] Failed to register gridGuide:', error);
  }

  // Check if navigator is already registered
  try {
    const hasNavigator = typeof Cytoscape('core', 'navigator') === 'function';
    if (!hasNavigator) {
      (Cytoscape as any).use(navigator);
      console.log('[Cytoscape] navigator extension registered');
    }
  } catch (error) {
    console.warn('[Cytoscape] Failed to register navigator:', error);
  }
}

// Create a configured Cytoscape instance with extensions
export function getExtendedCytoscape(): typeof Cytoscape {
  registerCytoscapeExtensions();
  return Cytoscape;
}

export default Cytoscape;
