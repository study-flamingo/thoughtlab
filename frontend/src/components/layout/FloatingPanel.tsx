import { useEffect } from 'react';
import type { FloatingPanelProps, PanelPosition, PanelSize } from '../../types/layout';
import { Z_INDEX } from '../../types/layout';
import { useIsMobile } from '../../hooks/useMediaQuery';
import { useEscapeKey } from '../../hooks/useFocusTrap';

// Position classes for different panel placements
const positionClasses: Record<PanelPosition, string> = {
  right: 'top-16 right-4 bottom-4',
  'top-right': 'top-16 right-4',
  'bottom-right': 'bottom-20 right-4',
};

// Size classes for different panel sizes
const sizeClasses: Record<PanelSize, string> = {
  sm: 'w-80',
  md: 'w-96',
  lg: 'w-[480px]',
  'full-height': 'w-96',
};

// Mobile position classes (convert to bottom sheets)
const mobilePositionClasses: Record<PanelPosition, string> = {
  right: 'bottom-0 left-0 right-0 top-auto',
  'top-right': 'bottom-0 left-0 right-0 top-auto',
  'bottom-right': 'bottom-0 left-0 right-0 top-auto',
};

/**
 * FloatingPanel - Reusable floating panel component with backdrop blur
 *
 * Features:
 * - Slide-in animations
 * - Backdrop blur (frosted glass effect)
 * - Multiple positions: right, top-right, bottom-right
 * - Multiple sizes: sm (320px), md (384px), lg (480px), full-height
 * - Dark mode support
 * - Mobile responsive (converts to bottom sheet)
 * - Escape key to close
 */
export default function FloatingPanel({
  isOpen,
  onClose,
  position = 'right',
  size = 'md',
  title,
  zIndex = Z_INDEX.inspector,
  children,
  showCloseButton = true,
  className = '',
}: FloatingPanelProps) {
  const isMobile = useIsMobile();

  // Handle Escape key to close panel
  useEscapeKey(isOpen, onClose);

  // Prevent body scroll when panel is open on mobile
  useEffect(() => {
    if (isMobile && isOpen) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [isMobile, isOpen]);

  if (!isOpen) return null;

  // Determine classes based on mobile vs desktop
  const positionClass = isMobile
    ? mobilePositionClasses[position]
    : positionClasses[position];

  const sizeClass = isMobile
    ? 'w-full max-h-[80vh]'
    : sizeClasses[size];

  const roundedClass = isMobile
    ? 'rounded-t-2xl rounded-b-none'
    : 'rounded-lg';

  const animationClass = isMobile
    ? 'animate-slide-in-up'
    : position === 'right' || position.includes('right')
      ? 'animate-slide-in-right'
      : 'animate-slide-in-right';

  const heightClass = !isMobile && size === 'full-height'
    ? 'max-h-[calc(100vh-5rem)]'
    : isMobile
      ? ''
      : 'max-h-[70vh]';

  return (
    <>
      {/* Mobile backdrop overlay */}
      {isMobile && (
        <div
          className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Panel */}
      <div
        className={`
          fixed ${positionClass}
          ${sizeClass}
          ${heightClass}
          bg-white/95 dark:bg-gray-800/95
          backdrop-blur-md
          ${roundedClass}
          shadow-2xl
          border border-gray-200 dark:border-gray-700
          overflow-hidden
          flex flex-col
          ${animationClass}
          ${className}
        `}
        style={{ zIndex }}
        role="dialog"
        aria-modal="true"
        aria-label={title || 'Panel'}
      >
        {/* Header */}
        {(title || showCloseButton) && (
          <div className="flex-shrink-0 px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm flex items-center justify-between">
            {title && (
              <h2 className="font-semibold text-gray-800 dark:text-gray-100">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none p-1 -mr-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                aria-label="Close panel"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          {children}
        </div>
      </div>
    </>
  );
}
