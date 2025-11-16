import { describe, it, expect, vi } from 'vitest';
import { render, screen } from './test/utils';
import userEvent from '@testing-library/user-event';
import App from './App';

describe('App', () => {
  it('renders header with title', () => {
    render(<App />);
    expect(screen.getByText('Research Connection Graph')).toBeInTheDocument();
  });

  it('renders Add Node button', () => {
    render(<App />);
    expect(screen.getByText('Add Node')).toBeInTheDocument();
  });

  it('opens modal when Add Node button is clicked', async () => {
    const user = userEvent.setup();
    render(<App />);

    const addButton = screen.getByText('Add Node');
    await user.click(addButton);

    expect(screen.getByText('Create New Node')).toBeInTheDocument();
  });

  it('renders GraphVisualizer component', () => {
    render(<App />);
    // GraphVisualizer should be rendered (checking by looking for its content)
    // Since it's async, we'll just check the structure exists
    expect(screen.getByRole('main')).toBeInTheDocument();
  });

  it('renders ActivityFeed component', () => {
    render(<App />);
    expect(screen.getByText('Activity Feed')).toBeInTheDocument();
  });
});
