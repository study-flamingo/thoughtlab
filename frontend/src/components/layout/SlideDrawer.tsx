import { useEffect } from 'react';
import type { SlideDrawerProps, DrawerMenuItem } from '../../types/layout';
import { Z_INDEX } from '../../types/layout';
import { useFocusTrap } from '../../hooks/useFocusTrap';

/**
 * SlideDrawer - Left-side sliding menu drawer
 *
 * Features:
 * - Slides from left with animation
 * - Backdrop overlay with click-to-close
 * - Focus trap for accessibility
 * - Escape key to close
 * - Dark mode support
 */
export default function SlideDrawer({ isOpen, onClose, children }: SlideDrawerProps) {
  const containerRef = useFocusTrap<HTMLDivElement>(isOpen, onClose);

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-sm animate-fade-in"
        style={{ zIndex: Z_INDEX.drawer - 1 }}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <div
        ref={containerRef}
        className="fixed top-0 left-0 bottom-0 w-80 max-w-[85vw] bg-white dark:bg-gray-800 shadow-2xl animate-slide-in-left flex flex-col"
        style={{ zIndex: Z_INDEX.drawer }}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold font-[Geo] text-gray-800 dark:text-gray-100">
            thoughtlab.ai
          </h2>
          <button
            onClick={onClose}
            className="p-2 -mr-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Close menu"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Menu Content */}
        <nav className="flex-1 overflow-y-auto py-2">
          {children}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          <p>ThoughtLab v0.2.0</p>
        </div>
      </div>
    </>
  );
}

/**
 * DrawerMenuItem - Individual menu item for the drawer
 */
export function DrawerMenuItemComponent({
  icon,
  label,
  onClick,
  variant = 'default',
}: DrawerMenuItem) {
  const baseClasses = 'w-full flex items-center gap-3 px-4 py-3 text-sm font-medium transition-colors';
  const variantClasses = variant === 'danger'
    ? 'text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20'
    : 'text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700';

  return (
    <button
      onClick={onClick}
      className={`${baseClasses} ${variantClasses}`}
    >
      {icon && <span className="text-lg">{icon}</span>}
      <span>{label}</span>
    </button>
  );
}

/**
 * DrawerDivider - Divider line for grouping menu items
 */
export function DrawerDivider() {
  return <div className="my-2 border-t border-gray-200 dark:border-gray-700" />;
}

/**
 * DrawerSection - Section header for grouping menu items
 */
export function DrawerSection({ title }: { title: string }) {
  return (
    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
      {title}
    </div>
  );
}
