import { Core } from 'cytoscape';

export interface GridGuideOptions {
  drawGrid?: boolean;
  gridSpacing?: number;
  gridColor?: string;
  lineWidth?: number;
  snapToGridOnRelease?: boolean;
  snapToGridDuringDrag?: boolean;
  panGrid?: boolean;
  gridStackOrder?: number;
  lineDash?: number[];
  strokeStyle?: string;
  zoomDash?: boolean;
  maxZoom?: number;
  minZoom?: number;
}

declare module 'cytoscape-grid-guide' {
  const gridGuide: (options?: GridGuideOptions) => Core;
  export = gridGuide;
}

declare module 'cytoscape' {
  interface Core {
    gridGuide(options?: GridGuideOptions): this;
  }
}