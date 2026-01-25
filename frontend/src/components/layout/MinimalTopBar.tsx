import type { MinimalTopBarProps } from '../../types/layout';
import { Z_INDEX } from '../../types/layout';
import NotificationBell from './NotificationBell';

/**
 * MinimalTopBar - Streamlined top navigation bar
 *
 * Features:
 * - Hamburger menu (left)
 * - Logo/title (center-left)
 * - Quick actions (right): Add Node, Add Relation, Notification Bell
 * - Fixed position with backdrop blur
 * - Dark mode support
 */
export default function MinimalTopBar({
  onMenuClick,
  onAddNode,
  onAddRelation,
  onNotificationClick,
  hasUnreadNotifications = false,
}: MinimalTopBarProps) {
  return (
    <header
      className="fixed top-0 left-0 right-0 h-14 bg-white/80 dark:bg-gray-800/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 px-4 flex items-center justify-between"
      style={{ zIndex: Z_INDEX.topBar }}
    >
      {/* Left section: Hamburger + Logo */}
      <div className="flex items-center gap-3">
        {/* Hamburger Menu Button */}
        <button
          onClick={onMenuClick}
          className="p-2 -ml-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white transition-colors"
          aria-label="Open menu"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Logo/Title */}
        <h1 className="text-lg font-semibold font-[Geo] text-gray-800 dark:text-gray-100 hidden sm:block">
          thoughtlab.ai
        </h1>
        {/* Mobile: shortened title */}
        <h1 className="text-lg font-semibold font-[Geo] text-gray-800 dark:text-gray-100 sm:hidden">
          TL
        </h1>
      </div>

      {/* Right section: Quick Actions */}
      <div className="flex items-center gap-1 sm:gap-2">
        {/* Add Relation Button */}
        <button
          onClick={onAddRelation}
          className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
          aria-label="Add relation"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          <span>Relation</span>
        </button>

        {/* Add Relation - Mobile Icon Only */}
        <button
          onClick={onAddRelation}
          className="sm:hidden p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Add relation"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
        </button>

        {/* Add Node Button */}
        <button
          onClick={onAddNode}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span className="hidden sm:inline">Add Node</span>
        </button>

        {/* Notification Bell */}
        <NotificationBell
          onClick={onNotificationClick}
          hasUnread={hasUnreadNotifications}
        />
      </div>
    </header>
  );
}
