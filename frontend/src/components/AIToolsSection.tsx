import { useState } from 'react';

interface AIToolsSectionProps {
  children: React.ReactNode;
  defaultExpanded?: boolean;
}

export function AIToolsSection({ children, defaultExpanded = false }: AIToolsSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border-t dark:border-gray-700 pt-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left px-1 py-1 rounded hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
      >
        <span className="text-xs font-semibold text-purple-600 dark:text-purple-400 flex items-center gap-2">
          <span>✨</span>
          AI Tools
        </span>
        <span className="text-gray-400 text-xs">{isExpanded ? '▼' : '▶'}</span>
      </button>

      {isExpanded && (
        <div className="mt-3 space-y-2">
          {children}
        </div>
      )}
    </div>
  );
}

interface AIToolButtonProps {
  label: string;
  icon?: string;
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  variant?: 'default' | 'danger';
}

export function AIToolButton({
  label,
  icon,
  onClick,
  isLoading = false,
  disabled = false,
  variant = 'default'
}: AIToolButtonProps) {
  const baseStyles = "w-full px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed";

  const variantStyles = variant === 'danger'
    ? "bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30"
    : "bg-purple-50 text-purple-700 hover:bg-purple-100 dark:bg-purple-900/20 dark:text-purple-400 dark:hover:bg-purple-900/30";

  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`${baseStyles} ${variantStyles}`}
    >
      {isLoading ? (
        <span className="animate-spin">⏳</span>
      ) : icon ? (
        <span>{icon}</span>
      ) : null}
      <span className="flex-1 text-left">{isLoading ? 'Processing...' : label}</span>
    </button>
  );
}
