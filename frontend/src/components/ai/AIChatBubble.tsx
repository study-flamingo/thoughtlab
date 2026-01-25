import type { AIChatBubbleProps } from '../../types/layout';
import { Z_INDEX } from '../../types/layout';

/**
 * AIChatBubble - Floating chat button in bottom-right corner
 *
 * Features:
 * - Fixed position at bottom-right
 * - Blue gradient background
 * - Hover/active states with animations
 * - Dark mode support
 * - Expandable to AIChatPanel
 */
export default function AIChatBubble({ onClick, isExpanded = false }: AIChatBubbleProps) {
  // Hide the bubble when chat is expanded
  if (isExpanded) return null;

  return (
    <button
      onClick={onClick}
      className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg hover:shadow-xl hover:scale-105 active:scale-95 transition-all duration-200 flex items-center justify-center group"
      style={{ zIndex: Z_INDEX.chat }}
      aria-label="Open AI chat"
    >
      {/* Chat Icon */}
      <svg
        className="w-6 h-6 group-hover:scale-110 transition-transform"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
        />
      </svg>

      {/* Sparkle indicator (AI) */}
      <span className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-400 rounded-full flex items-center justify-center text-[10px] shadow-sm">
        <svg className="w-2.5 h-2.5 text-yellow-900" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2L9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2z" />
        </svg>
      </span>
    </button>
  );
}
