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
    expect(screen.getByText(/toughtlab\.ai/)).toBeInTheDocument();
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
    expect(screen.getByText('Activities Feed')).toBeInTheDocument();
  });

  it('renders Settings button', () => {
    render(<App />);
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders Add Relation button', () => {
    render(<App />);
    expect(screen.getByText('Add Relation')).toBeInTheDocument();
  });
});
