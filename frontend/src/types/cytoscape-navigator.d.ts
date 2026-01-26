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

declare module 'cytoscape' {
  interface Core {
    navigator(options?: import('cytoscape-navigator').NavigatorOptions): this;
  }
}