import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '../../test/utils';
import userEvent from '@testing-library/user-event';
import ActivityFeed from '../ActivityFeed';
import { graphApi } from '../../services/api';
import type { Activity } from '../../types/activity';

// Mock the API
vi.mock('../../services/api', () => ({
  graphApi: {
    getActivities: vi.fn(),
    approveSuggestion: vi.fn(),
    rejectSuggestion: vi.fn(),
  },
}));

const mockActivity: Activity = {
  id: 'activity-1',
  type: 'node_created',
  message: 'Observation created',
  created_at: new Date().toISOString(),
  node_id: 'node-1',
  node_type: 'Observation',
};

const mockSuggestionActivity: Activity = {
  id: 'suggestion-1',
  type: 'relationship_suggested',
  message: 'Suggested: Node A SUPPORTS Node B (75% confidence)',
  created_at: new Date().toISOString(),
  status: 'pending',
  node_id: 'node-a',
  suggestion_data: {
    from_node_id: 'node-a',
    from_node_type: 'Observation',
    from_node_label: 'Node A',
    to_node_id: 'node-b',
    to_node_type: 'Hypothesis',
    to_node_label: 'Node B',
    relationship_type: 'SUPPORTS',
    confidence: 0.75,
    reasoning: 'These nodes discuss similar topics',
  },
};

const mockProcessingActivity: Activity = {
  id: 'processing-1',
  type: 'processing_embedding',
  message: 'Creating embeddings for Research Paper',
  created_at: new Date().toISOString(),
  node_id: 'source-1',
  processing_data: {
    node_id: 'source-1',
    node_type: 'Source',
    node_label: 'Research Paper',
    stage: 'embedding',
    progress: 0.5,
    chunks_created: 10,
    embeddings_created: 5,
  },
};

describe('ActivityFeed', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders loading state initially', () => {
    (graphApi.getActivities as any).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<ActivityFeed />);
    expect(screen.getByText('Loading activities...')).toBeInTheDocument();
  });

  it('renders activity feed content area', async () => {
    (graphApi.getActivities as any).mockResolvedValue({ data: [] });

    render(<ActivityFeed />);

    // The feed should render with live indicator (header now handled by FloatingPanel wrapper)
    await waitFor(() => {
      expect(screen.getByText(/Live/)).toBeInTheDocument();
    });
  });

  it('shows live indicator', async () => {
    (graphApi.getActivities as any).mockResolvedValue({ data: [] });

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText(/Live/)).toBeInTheDocument();
    });
  });

  it('shows empty state when no activities', async () => {
    (graphApi.getActivities as any).mockResolvedValue({ data: [] });

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText('No activity yet')).toBeInTheDocument();
    });
  });

  it('shows error state on API failure', async () => {
    (graphApi.getActivities as any).mockRejectedValue(new Error('API Error'));

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load activities')).toBeInTheDocument();
    });
  });

  it('displays activities when data is available', async () => {
    (graphApi.getActivities as any).mockResolvedValue({
      data: [mockActivity],
    });

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText('Observation created')).toBeInTheDocument();
    });
  });

  it('displays activity count in header', async () => {
    (graphApi.getActivities as any).mockResolvedValue({
      data: [mockActivity, { ...mockActivity, id: 'activity-2' }],
    });

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText(/Live \(2\)/)).toBeInTheDocument();
    });
  });

  it('shows View button for activities with navigation', async () => {
    (graphApi.getActivities as any).mockResolvedValue({
      data: [mockActivity],
    });

    render(<ActivityFeed />);

    await waitFor(() => {
      expect(screen.getByText('View')).toBeInTheDocument();
    });
  });

  it('calls onSelectNode when View button is clicked', async () => {
    (graphApi.getActivities as any).mockResolvedValue({
      data: [mockActivity],
    });

    const onSelectNode = vi.fn();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<ActivityFeed onSelectNode={onSelectNode} />);

    await waitFor(() => {
      expect(screen.getByText('View')).toBeInTheDocument();
    });

    await user.click(screen.getByText('View'));
    expect(onSelectNode).toHaveBeenCalledWith('node-1');
  });

  describe('Suggestion activities', () => {
    it('shows Approve and Reject buttons for pending suggestions', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Approve')).toBeInTheDocument();
        expect(screen.getByText('Reject')).toBeInTheDocument();
      });
    });

    it('shows suggestion details', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Node A')).toBeInTheDocument();
        expect(screen.getByText('SUPPORTS')).toBeInTheDocument();
        expect(screen.getByText('Node B')).toBeInTheDocument();
        expect(screen.getByText(/Confidence: 75%/)).toBeInTheDocument();
      });
    });

    it('calls approveSuggestion when Approve is clicked', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });
      (graphApi.approveSuggestion as any).mockResolvedValue({
        data: { message: 'Approved', relationship_id: 'rel-1' },
      });

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Approve')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Approve'));

      await waitFor(() => {
        expect(graphApi.approveSuggestion).toHaveBeenCalledWith('suggestion-1');
      });
    });

    it('calls rejectSuggestion when Reject is clicked', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });
      (graphApi.rejectSuggestion as any).mockResolvedValue({
        data: { message: 'Rejected' },
      });

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Reject')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Reject'));

      await waitFor(() => {
        expect(graphApi.rejectSuggestion).toHaveBeenCalledWith('suggestion-1', undefined);
      });
    });

    it('disables buttons while processing approval', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });
      // Make approve take time
      (graphApi.approveSuggestion as any).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ data: {} }), 1000))
      );

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Approve')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Approve'));

      // Buttons should show loading state - both buttons show "..."
      await waitFor(() => {
        const loadingButtons = screen.getAllByText('...');
        expect(loadingButtons.length).toBeGreaterThan(0);
        // All loading buttons should be disabled
        loadingButtons.forEach(btn => expect(btn).toBeDisabled());
      });
    });

    it('shows status badge for approved suggestions', async () => {
      const approvedActivity: Activity = {
        ...mockSuggestionActivity,
        status: 'approved',
      };
      (graphApi.getActivities as any).mockResolvedValue({
        data: [approvedActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('approved')).toBeInTheDocument();
      });

      // Should not show Approve/Reject buttons
      expect(screen.queryByText('Approve')).not.toBeInTheDocument();
      expect(screen.queryByText('Reject')).not.toBeInTheDocument();
    });

    it('shows status badge for rejected suggestions', async () => {
      const rejectedActivity: Activity = {
        ...mockSuggestionActivity,
        status: 'rejected',
      };
      (graphApi.getActivities as any).mockResolvedValue({
        data: [rejectedActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('rejected')).toBeInTheDocument();
      });
    });
  });

  describe('Processing activities', () => {
    it('shows progress bar for processing activities', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockProcessingActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Creating embeddings for Research Paper')).toBeInTheDocument();
      });
    });

    it('shows chunk and embedding counts', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockProcessingActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText(/10 chunks/)).toBeInTheDocument();
        expect(screen.getByText(/5 embeddings/)).toBeInTheDocument();
      });
    });

    it('shows error message for failed processing', async () => {
      const failedActivity: Activity = {
        id: 'processing-failed-1',
        type: 'processing_failed',
        message: 'Processing failed',
        created_at: new Date().toISOString(),
        processing_data: {
          node_id: 'source-1',
          node_type: 'Source',
          node_label: 'Research Paper',
          stage: 'failed',
          error_message: 'API rate limit exceeded',
        },
      };
      (graphApi.getActivities as any).mockResolvedValue({
        data: [failedActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('API rate limit exceeded')).toBeInTheDocument();
      });
    });
  });

  describe('Polling', () => {
    it('polls for updates at specified interval', async () => {
      (graphApi.getActivities as any).mockResolvedValue({ data: [] });

      render(<ActivityFeed refreshInterval={1000} />);

      // Initial call
      await waitFor(() => {
        expect(graphApi.getActivities).toHaveBeenCalledTimes(1);
      });

      // Advance time by refresh interval
      vi.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(graphApi.getActivities).toHaveBeenCalledTimes(2);
      });

      // Another interval
      vi.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(graphApi.getActivities).toHaveBeenCalledTimes(3);
      });
    });

    it('uses default 5 second refresh interval', async () => {
      (graphApi.getActivities as any).mockResolvedValue({ data: [] });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(graphApi.getActivities).toHaveBeenCalledTimes(1);
      });

      // Advance less than default interval
      vi.advanceTimersByTime(4000);

      expect(graphApi.getActivities).toHaveBeenCalledTimes(1);

      // Advance to complete the default interval
      vi.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(graphApi.getActivities).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Activity icons and colors', () => {
    it('shows correct icon for node_created', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('➕')).toBeInTheDocument();
      });
    });

    it('shows correct icon for relationship_suggested', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('💡')).toBeInTheDocument();
      });
    });

    it('shows correct icon for processing activities', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockProcessingActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('🧮')).toBeInTheDocument();
      });
    });
  });

  describe('Timestamp formatting', () => {
    it('shows "Just now" for very recent activities', async () => {
      const recentActivity: Activity = {
        ...mockActivity,
        created_at: new Date().toISOString(),
      };
      (graphApi.getActivities as any).mockResolvedValue({
        data: [recentActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Just now')).toBeInTheDocument();
      });
    });

    it('shows minutes ago for activities within an hour', async () => {
      const fifteenMinutesAgo = new Date(Date.now() - 15 * 60 * 1000);
      const pastActivity: Activity = {
        ...mockActivity,
        created_at: fifteenMinutesAgo.toISOString(),
      };
      (graphApi.getActivities as any).mockResolvedValue({
        data: [pastActivity],
      });

      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('15m ago')).toBeInTheDocument();
      });
    });
  });

  describe('Error handling', () => {
    it('shows error when approve fails', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });
      (graphApi.approveSuggestion as any).mockRejectedValue(new Error('Server error'));

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Approve')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Approve'));

      await waitFor(() => {
        expect(screen.getByText('Failed to approve suggestion')).toBeInTheDocument();
      });
    });

    it('shows error when reject fails', async () => {
      (graphApi.getActivities as any).mockResolvedValue({
        data: [mockSuggestionActivity],
      });
      (graphApi.rejectSuggestion as any).mockRejectedValue(new Error('Server error'));

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<ActivityFeed />);

      await waitFor(() => {
        expect(screen.getByText('Reject')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Reject'));

      await waitFor(() => {
        expect(screen.getByText('Failed to reject suggestion')).toBeInTheDocument();
      });
    });
  });
});
