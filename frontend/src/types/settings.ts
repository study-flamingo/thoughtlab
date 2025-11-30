export type Theme = 'light' | 'dark';
export type LayoutName = 'cose' | 'grid' | 'circle';

export type LineStyle = 'solid' | 'dashed' | 'dotted';
export type ArrowShape = 'triangle' | 'tee' | 'none';

export interface RelationStyle {
  line_color: string;
  target_arrow_color?: string;
  width?: number;
  line_style?: LineStyle;
  target_arrow_shape?: ArrowShape;
}

export interface AppSettings {
  id: string;
  theme: Theme;
  show_edge_labels: boolean;
  default_relation_confidence: number;
  layout_name: LayoutName;
  animate_layout: boolean;
  node_colors: Record<string, string>;
  relation_styles: Record<string, RelationStyle>;
}

export interface AppSettingsUpdate {
  theme?: Theme;
  show_edge_labels?: boolean;
  default_relation_confidence?: number;
  layout_name?: LayoutName;
  animate_layout?: boolean;
  node_colors?: Record<string, string>;
  relation_styles?: Record<string, RelationStyle>;
}


