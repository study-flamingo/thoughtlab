import Cytoscape from 'cytoscape';

// Extension registration removed - using manual grid implementation
// No external Cytoscape extensions required

export function registerCytoscapeExtensions(): void {
  // No extensions to register
}

// Create a configured Cytoscape instance
export function getExtendedCytoscape(): typeof Cytoscape {
  return Cytoscape;
}

export default Cytoscape;
