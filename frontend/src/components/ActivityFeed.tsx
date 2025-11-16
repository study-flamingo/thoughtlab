export default function ActivityFeed() {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-semibold text-gray-800 mb-2">Activity Feed</h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">● Live</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="text-sm text-gray-500">No activity yet</div>
        <div className="mt-4 text-xs text-gray-400">
          Activity feed will show real-time updates when connection analysis is running.
        </div>
      </div>
    </div>
  );
}
