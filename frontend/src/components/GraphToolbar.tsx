import React from 'react';
import { 
  MousePointer2, 
  SquareDashed, 
  Scissors, 
  Copy, 
  ClipboardPaste,
  Grid3X3,
  Grid2X2,
  FlipHorizontal,
  Anchor,
  LayoutGrid
} from 'lucide-react';
import { useGraphTools, ToolType } from '../hooks/useGraphTools';

interface ToolButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  accentColor?: string;
}

const ToolButton: React.FC<ToolButtonProps> = ({ 
  active, 
  onClick, 
  icon, 
  label, 
  shortcut,
  accentColor = '#9900EB'
}) => {
  return (
    <button
      onClick={onClick}
      className={`
        relative flex flex-col items-center justify-center
        w-12 h-12 rounded-lg
        transition-all duration-150 ease-out
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900
        ${active 
          ? 'bg-gray-800 shadow-inner transform translate-y-0.5' 
          : 'bg-gray-700 hover:bg-gray-600 shadow-md hover:shadow-lg transform hover:-translate-y-0.5'
        }
      `}
      style={{
        backgroundColor: active ? accentColor : undefined,
        boxShadow: active 
          ? 'inset 0 2px 4px rgba(0,0,0,0.3)' 
          : '0 2px 4px rgba(0,0,0,0.2)'
      }}
      title={`${label}${shortcut ? ` (${shortcut})` : ''}`}
    >
      <span className={`
        transition-colors duration-150
        ${active ? 'text-white' : 'text-gray-300'}
      `}>
        {icon}
      </span>
      {shortcut && (
        <span className={`
          absolute -bottom-4 text-[10px] font-mono
          ${active ? 'text-white' : 'text-gray-500'}
        `}>
          {shortcut}
        </span>
      )}
    </button>
  );
};

interface ActionButtonProps {
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  shortcut?: string;
  disabled?: boolean;
}

const ActionButton: React.FC<ActionButtonProps> = ({ 
  onClick, 
  icon, 
  label, 
  shortcut,
  disabled = false
}) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        relative flex flex-col items-center justify-center
        w-10 h-10 rounded-lg
        transition-all duration-150 ease-out
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900
        ${disabled
          ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
          : 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white shadow-md hover:shadow-lg'
        }
      `}
      title={`${label}${shortcut ? ` (${shortcut})` : ''}`}
    >
      {icon}
    </button>
  );
};

interface GraphToolbarProps {
  accentColor?: string;
}

export const GraphToolbar: React.FC<GraphToolbarProps> = ({ 
  accentColor = '#9900EB' 
}) => {
  const {
    activeTool,
    setActiveTool,
    selection,
    selectAll,
    deselectAll,
    selectInverse,
    cutSelection,
    copySelection,
    pasteSelection,
    isGraphFocused
  } = useGraphTools();

  const hasSelection = selection.length > 0;

  return (
    <div 
      className={`
        absolute top-4 right-4
        flex flex-col gap-3
        p-3 rounded-xl
        bg-gray-900/95 backdrop-blur-sm
        border border-gray-700/50
        shadow-2xl
        z-50
        transition-opacity duration-200
        ${isGraphFocused ? 'opacity-100' : 'opacity-60'}
      `}
    >
      {/* Tool Section */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold px-1">
          Tools
        </span>
        <div className="flex gap-2">
          <ToolButton
            active={activeTool === 'selector'}
            onClick={() => setActiveTool('selector')}
            icon={<MousePointer2 size={18} />}
            label="Selector"
            shortcut="S"
            accentColor={accentColor}
          />
          <ToolButton
            active={activeTool === 'lasso'}
            onClick={() => setActiveTool('lasso')}
            icon={<SquareDashed size={18} />}
            label="Lasso"
            shortcut="M"
            accentColor={accentColor}
          />
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gray-700/50" />

      {/* Selection Operations */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold px-1">
          Select
        </span>
        <div className="flex gap-1">
          <ActionButton
            onClick={selectAll}
            icon={<Grid3X3 size={16} />}
            label="Select All"
            shortcut="Ctrl+A"
          />
          <ActionButton
            onClick={deselectAll}
            icon={<Grid2X2 size={16} />}
            label="Deselect All"
            shortcut="Ctrl+Shift+A"
          />
          <ActionButton
            onClick={selectInverse}
            icon={<FlipHorizontal size={16} />}
            label="Select Inverse"
            shortcut="Ctrl+Shift+I"
          />
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gray-700/50" />

      {/* Clipboard Operations */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold px-1">
          Edit
        </span>
        <div className="flex gap-1">
          <ActionButton
            onClick={cutSelection}
            icon={<Scissors size={16} />}
            label="Cut"
            shortcut="Ctrl+X"
            disabled={!hasSelection}
          />
          <ActionButton
            onClick={copySelection}
            icon={<Copy size={16} />}
            label="Copy"
            shortcut="Ctrl+C"
            disabled={!hasSelection}
          />
          <ActionButton
            onClick={pasteSelection}
            icon={<ClipboardPaste size={16} />}
            label="Paste"
            shortcut="Ctrl+V"
          />
        </div>
      </div>

      {/* Selection Counter */}
      {hasSelection && (
        <>
          <div className="h-px bg-gray-700/50" />
          <div className="text-center">
            <span className="text-xs text-gray-400">
              {selection.length} node{selection.length !== 1 ? 's' : ''} selected
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default GraphToolbar;
