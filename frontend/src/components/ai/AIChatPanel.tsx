import { useState } from 'react';
import type { AIChatPanelProps } from '../../types/layout';
import FloatingPanel from '../layout/FloatingPanel';
import { Z_INDEX } from '../../types/layout';

// Placeholder messages for prototype UI
const placeholderMessages = [
  {
    id: '1',
    role: 'assistant' as const,
    content: 'Hello! I\'m your AI research assistant. I can help you explore your knowledge graph, find connections between ideas, and generate insights. What would you like to explore?',
    timestamp: new Date(Date.now() - 60000).toISOString(),
  },
  {
    id: '2',
    role: 'user' as const,
    content: 'Can you help me find connections between my recent observations?',
    timestamp: new Date(Date.now() - 30000).toISOString(),
  },
  {
    id: '3',
    role: 'assistant' as const,
    content: 'I\'d be happy to help! I can analyze your knowledge graph to find patterns and relationships. Try selecting a node and using the "Find Related Nodes" tool, or ask me about specific topics you\'re researching.',
    timestamp: new Date().toISOString(),
  },
];

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

/**
 * AIChatPanel - Chat UI prototype (no backend integration)
 *
 * Features:
 * - Floating panel with chat messages
 * - Input field + send button (non-functional for now)
 * - Placeholder messages to demonstrate UI
 * - Dark mode support
 */
export default function AIChatPanel({ isOpen, onClose }: AIChatPanelProps) {
  const [inputValue, setInputValue] = useState('');
  const [messages] = useState<Message[]>(placeholderMessages);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement actual chat functionality
    // For now, just clear the input
    if (inputValue.trim()) {
      setInputValue('');
    }
  };

  return (
    <FloatingPanel
      isOpen={isOpen}
      onClose={onClose}
      position="bottom-right"
      size="lg"
      title="AI Assistant"
      zIndex={Z_INDEX.chat}
      className="!bottom-6 !right-6 h-[500px] max-h-[70vh]"
    >
      <div className="flex flex-col h-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>

        {/* Prototype Notice */}
        <div className="px-4 py-2 bg-amber-50 dark:bg-amber-900/20 border-t border-amber-200 dark:border-amber-800">
          <p className="text-xs text-amber-700 dark:text-amber-300 text-center">
            This is a prototype UI. Full AI chat coming soon!
          </p>
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about your knowledge graph..."
              className="flex-1 px-4 py-2 text-sm bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={!inputValue.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </FloatingPanel>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-md'
        }`}
      >
        {/* AI Icon for assistant messages */}
        {!isUser && (
          <div className="flex items-center gap-2 mb-1">
            <span className="w-5 h-5 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2L9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2z" />
              </svg>
            </span>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">AI Assistant</span>
          </div>
        )}

        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>

        <p className={`text-xs mt-1 ${isUser ? 'text-blue-200' : 'text-gray-400 dark:text-gray-500'}`}>
          {formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}

function formatTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
