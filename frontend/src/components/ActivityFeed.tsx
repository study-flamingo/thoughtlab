import { useState, useEffect, useCallback } from 'react';
import { graphApi } from '../services/api';
import type {
  Activity,
  ActivityType,
  SuggestionData,
  ProcessingData,
} from '../types/activity';
import {
  isInteractiveActivity,
  hasNavigation,
  getActivityIcon,
  getActivityColor,
  formatConfidence,
} from '../types/activity';

interface ActivityFeedProps {
  onSelectNode?: (nodeId: string) => void;
  refreshInterval?: number; // ms
}

export default function ActivityFeed({
  onSelectNode,
  refreshInterval = 5000,
}: ActivityFeedProps) {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processingActions, setProcessingActions] = useState<Set<string>>(new Set());

  const fetchActivities = useCallback(async () => {
    try {
      const response = await graphApi.getActivities({ limit: 50 });
      setActivities(response.data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch activities:', err);
      setError('Failed to load activities');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchActivities();
    
    const interval = setInterval(fetchActivities, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchActivities, refreshInterval]);

  const handleApprove = async (activity: Activity) => {
    if (!activity.id || processingActions.has(activity.id)) return;
    
    setProcessingActions(prev => new Set(prev).add(activity.id));
    
    try {
      await graphApi.approveSuggestion(activity.id);
      // Refresh to get updated status
      await fetchActivities();
    } catch (err) {
      console.error('Failed to approve suggestion:', err);
      setError('Failed to approve suggestion');
    } finally {
      setProcessingActions(prev => {
        const next = new Set(prev);
        next.delete(activity.id);
        return next;
      });
    }
  };

  const handleReject = async (activity: Activity, feedback?: string) => {
    if (!activity.id || processingActions.has(activity.id)) return;
    
    setProcessingActions(prev => new Set(prev).add(activity.id));
    
    try {
      await graphApi.rejectSuggestion(activity.id, feedback);
      await fetchActivities();
    } catch (err) {
      console.error('Failed to reject suggestion:', err);
      setError('Failed to reject suggestion');
    } finally {
      setProcessingActions(prev => {
        const next = new Set(prev);
        next.delete(activity.id);
        return next;
      });
    }
  };

  const handleNavigate = (activity: Activity) => {
    const nodeId = activity.node_id || activity.suggestion_data?.from_node_id;
    if (nodeId && onSelectNode) {
      onSelectNode(nodeId);
    }
  };

  if (loading) {
    return (
      <div className="h-full flex flex-col">
        <ActivityHeader />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-pulse text-gray-500">Loading activities...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <ActivityHeader activityCount={activities.length} />
      
      {error && (
        <div className="px-4 py-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm">
          {error}
        </div>
      )}
      
      <div className="flex-1 overflow-y-auto">
        {activities.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-gray-800">
            {activities.map((activity) => (
              <ActivityItem
                key={activity.id}
                activity={activity}
                isProcessing={processingActions.has(activity.id)}
                onApprove={() => handleApprove(activity)}
                onReject={(feedback) => handleReject(activity, feedback)}
                onNavigate={() => handleNavigate(activity)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityHeader({ activityCount = 0 }: { activityCount?: number }) {
  return (
    <div className="p-4 border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-gray-800 dark:text-gray-100">
          Activities Feed
        </h2>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Live {activityCount > 0 && `(${activityCount})`}
          </span>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="p-4 text-center">
      <div className="text-gray-400 dark:text-gray-500 mb-2">
        <svg className="w-12 h-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      <p className="text-sm text-gray-500 dark:text-gray-400">No activity yet</p>
      <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
        Activities will appear here when you create nodes or when the AI finds connections.
      </p>
    </div>
  );
}

interface ActivityItemProps {
  activity: Activity;
  isProcessing: boolean;
  onApprove: () => void;
  onReject: (feedback?: string) => void;
  onNavigate: () => void;
}

function ActivityItem({
  activity,
  isProcessing,
  onApprove,
  onReject,
  onNavigate,
}: ActivityItemProps) {
  const icon = getActivityIcon(activity.type);
  const colorClass = getActivityColor(activity.type);
  const isInteractive = isInteractiveActivity(activity);
  const canNavigate = hasNavigation(activity);
  
  return (
    <div className="p-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
      <div className="flex gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 text-lg ${colorClass}`}>
          {icon}
        </div>
        
        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Message */}
          <p className="text-sm text-gray-700 dark:text-gray-300 leading-snug">
            {activity.message}
          </p>
          
          {/* Suggestion details */}
          {activity.suggestion_data && (
            <SuggestionDetails suggestion={activity.suggestion_data} />
          )}
          
          {/* Processing details */}
          {activity.processing_data && (
            <ProcessingDetails processing={activity.processing_data} />
          )}
          
          {/* Timestamp */}
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            {formatTimestamp(activity.created_at)}
          </p>
          
          {/* Actions */}
          <div className="flex gap-2 mt-2">
            {canNavigate && (
              <button
                onClick={onNavigate}
                className="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 
                         text-gray-600 dark:text-gray-300 hover:bg-gray-200 
                         dark:hover:bg-gray-600 transition-colors"
              >
                View
              </button>
            )}
            
            {isInteractive && (
              <>
                <button
                  onClick={onApprove}
                  disabled={isProcessing}
                  className="text-xs px-2 py-1 rounded bg-green-100 dark:bg-green-900/30 
                           text-green-700 dark:text-green-400 hover:bg-green-200 
                           dark:hover:bg-green-900/50 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? '...' : 'Approve'}
                </button>
                <button
                  onClick={() => onReject()}
                  disabled={isProcessing}
                  className="text-xs px-2 py-1 rounded bg-red-100 dark:bg-red-900/30 
                           text-red-700 dark:text-red-400 hover:bg-red-200 
                           dark:hover:bg-red-900/50 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? '...' : 'Reject'}
                </button>
              </>
            )}
          </div>
        </div>
        
        {/* Status badge */}
        {activity.status && activity.status !== 'pending' && (
          <div className="flex-shrink-0">
            <StatusBadge status={activity.status} />
          </div>
        )}
      </div>
    </div>
  );
}

function SuggestionDetails({ suggestion }: { suggestion: SuggestionData }) {
  return (
    <div className="mt-2 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs">
      <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200">
        <span className="font-medium truncate" title={suggestion.from_node_label}>
          {suggestion.from_node_label}
        </span>
        <span className="text-amber-600 dark:text-amber-400">
          {suggestion.relationship_type}
        </span>
        <span className="font-medium truncate" title={suggestion.to_node_label}>
          {suggestion.to_node_label}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-1 text-amber-600 dark:text-amber-400">
        <span>Confidence: {formatConfidence(suggestion.confidence)}</span>
        {suggestion.reasoning && (
          <span className="truncate" title={suggestion.reasoning}>
            — {suggestion.reasoning}
          </span>
        )}
      </div>
    </div>
  );
}

function ProcessingDetails({ processing }: { processing: ProcessingData }) {
  const stages = ['started', 'chunking', 'embedding', 'analyzing', 'completed'];
  const currentIndex = stages.indexOf(processing.stage);
  
  return (
    <div className="mt-2 space-y-2">
      {/* Progress bar */}
      {processing.stage !== 'failed' && processing.stage !== 'completed' && (
        <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-500"
            style={{ width: `${((currentIndex + 1) / stages.length) * 100}%` }}
          />
        </div>
      )}
      
      {/* Stats */}
      <div className="flex gap-3 text-xs text-gray-500 dark:text-gray-400">
        {processing.chunks_created !== undefined && processing.chunks_created > 0 && (
          <span>📄 {processing.chunks_created} chunks</span>
        )}
        {processing.embeddings_created !== undefined && processing.embeddings_created > 0 && (
          <span>🧮 {processing.embeddings_created} embeddings</span>
        )}
        {processing.suggestions_found !== undefined && processing.suggestions_found > 0 && (
          <span>💡 {processing.suggestions_found} suggestions</span>
        )}
      </div>
      
      {/* Error message */}
      {processing.error_message && (
        <p className="text-xs text-red-600 dark:text-red-400">
          {processing.error_message}
        </p>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    approved: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    rejected: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    expired: 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400',
  };
  
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full ${styles[status] || ''}`}>
      {status}
    </span>
  );
}

function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return date.toLocaleDateString();
}
