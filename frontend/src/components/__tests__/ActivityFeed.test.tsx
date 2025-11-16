import { describe, it, expect } from 'vitest';
import { render, screen } from '../../test/utils';
import ActivityFeed from '../ActivityFeed';

describe('ActivityFeed', () => {
  it('renders activity feed header', () => {
    render(<ActivityFeed />);
    expect(screen.getByText('Activity Feed')).toBeInTheDocument();
  });

  it('shows live indicator', () => {
    render(<ActivityFeed />);
    expect(screen.getByText('● Live')).toBeInTheDocument();
  });

  it('shows empty state message', () => {
    render(<ActivityFeed />);
    expect(screen.getByText('No activity yet')).toBeInTheDocument();
  });
});
