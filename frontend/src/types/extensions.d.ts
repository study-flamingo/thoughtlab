// Type declarations for cytoscape extensions
// These are loaded in addition to the specific .d.ts files

declare module 'cytoscape-grid-guide' {
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

  const gridGuide: (options?: GridGuideOptions) => Core;
  export = gridGuide;
}

declare module 'cytoscape-navigator' {
  import { Core } from 'cytoscape';

  export interface NavigatorOptions {
    container?: HTMLElement | string;
    viewLiveFramerate?: number;
    thumbnailEventFramerate?: number;
    thumbnailLiveFramerate?: number;
    dblClickDelay?: number;
    removeCustomContainer?: boolean;
  }

  const navigator: (options?: NavigatorOptions) => Core;
  export = navigator;
}