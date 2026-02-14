/**
 * Layout component types for ThoughtLab UI
 */

// FloatingPanel position options
export type PanelPosition = 'right' | 'top-right' | 'bottom-right';

// FloatingPanel size options
export type PanelSize = 'sm' | 'md' | 'lg' | 'full-height';

// FloatingPanel props
export interface FloatingPanelProps {
  /** Whether the panel is visible */
  isOpen: boolean;
  /** Callback when panel is closed */
  onClose: () => void;
  /** Position of the panel */
  position?: PanelPosition;
  /** Size of the panel */
  size?: PanelSize;
  /** Title displayed in the panel header */
  title?: React.ReactNode;
  /** Z-index layer (higher = closer to front) */
  zIndex?: number;
  /** Panel contents */
  children: React.ReactNode;
  /** Whether to show the close button in header */
  showCloseButton?: boolean;
  /** Optional custom class names */
  className?: string;
}

// SlideDrawer props
export interface SlideDrawerProps {
  /** Whether the drawer is open */
  isOpen: boolean;
  /** Callback when drawer is closed */
  onClose: () => void;
  /** Menu items to display */
  children: React.ReactNode;
}

// SlideDrawer menu item
export interface DrawerMenuItem {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Icon (emoji or component) */
  icon?: React.ReactNode;
  /** Click handler */
  onClick: () => void;
  /** Whether item is destructive/danger */
  variant?: 'default' | 'danger';
}

// MinimalTopBar props
export interface MinimalTopBarProps {
  /** Hamburger menu click handler */
  onMenuClick: () => void;
  /** Add Node button click handler */
  onAddNode: () => void;
  /** Add Relation button click handler */
  onAddRelation: () => void;
  /** Notification bell click handler */
  onNotificationClick: () => void;
  /** Whether there are unread notifications */
  hasUnreadNotifications?: boolean;
}

// NotificationBell props
export interface NotificationBellProps {
  /** Click handler */
  onClick: () => void;
  /** Whether there are unread notifications */
  hasUnread?: boolean;
}

// AIChatBubble props
export interface AIChatBubbleProps {
  /** Click handler to expand chat */
  onClick: () => void;
  /** Whether chat is currently expanded */
  isExpanded?: boolean;
}

// AIChatPanel props
export interface AIChatPanelProps {
  /** Whether the chat panel is visible */
  isOpen: boolean;
  /** Callback when panel is closed */
  onClose: () => void;
}

// Panel state management
export interface PanelState {
  nodeInspector: boolean;
  relationInspector: boolean;
  activityFeed: boolean;
  aiChat: boolean;
}

// Z-index hierarchy constants
export const Z_INDEX = {
  graph: 0,
  activityFeed: 20,
  inspector: 30,
  topBar: 40,
  drawer: 50,
  chat: 50,
  modal: 60,
  toast: 70,
} as const;
