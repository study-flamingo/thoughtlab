import { describe, it, expect, vi } from 'vitest';
import { render, screen } from './test/utils';
import userEvent from '@testing-library/user-event';
import App from './App';

// Mock the API to prevent actual calls
vi.mock('./services/api', () => ({
  graphApi: {
    getFullGraph: vi.fn().mockResolvedValue({ data: { nodes: [], edges: [] } }),
    getActivities: vi.fn().mockResolvedValue({ data: [] }),
    getSettings: vi.fn().mockResolvedValue({
      data: {
        id: 'app',
        theme: 'light',
        show_edge_labels: true,
        default_relation_confidence: 0.8,
        layout_name: 'cose',
        animate_layout: false,
        node_colors: {},
        relation_styles: {},
      },
    }),
  },
}));

describe('App', () => {
  it('renders header with title', () => {
    render(<App />);
    expect(screen.getByText(/thoughtlab\.ai/)).toBeInTheDocument();
  });

  it('renders Add Node button', () => {
    render(<App />);
    // The button may show just "Add Node" or icon only on mobile
    expect(screen.getByRole('button', { name: /add node/i })).toBeInTheDocument();
  });

  it('opens modal when Add Node button is clicked', async () => {
    const user = userEvent.setup();
    render(<App />);

    const addButton = screen.getByRole('button', { name: /add node/i });
    await user.click(addButton);

    expect(screen.getByText('Create New Node')).toBeInTheDocument();
  });

  it('renders GraphVisualizer component', () => {
    render(<App />);
    // GraphVisualizer should be rendered (checking by looking for its content)
    // Since it's async, we'll just check the structure exists
    expect(screen.getByRole('main')).toBeInTheDocument();
  });

  it('renders hamburger menu button to access settings', () => {
    render(<App />);
    // Settings is now in the drawer, accessed via hamburger menu
    expect(screen.getByRole('button', { name: /open menu/i })).toBeInTheDocument();
  });

  it('renders Add Relation button', () => {
    render(<App />);
    // There are two buttons (desktop and mobile variants), find at least one
    const buttons = screen.getAllByRole('button', { name: /relation/i });
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('renders notification bell to toggle activity feed', () => {
    render(<App />);
    // Activity feed is hidden by default, accessed via notification bell
    expect(screen.getByRole('button', { name: /notification/i })).toBeInTheDocument();
  });
});
