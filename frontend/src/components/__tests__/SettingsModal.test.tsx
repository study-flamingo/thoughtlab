import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '../../test/utils';
import userEvent from '@testing-library/user-event';
import SettingsModal from '../SettingsModal';
import { graphApi } from '../../services/api';

// Mock the API
vi.mock('../../services/api', () => ({
  graphApi: {
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
  },
}));

const mockSettings = {
  id: 'app',
  theme: 'light',
  show_edge_labels: true,
  default_relation_confidence: 0.8,
  layout_name: 'cose',
  animate_layout: false,
  node_colors: {
    Observation: '#60A5FA',
    Hypothesis: '#34D399',
    Source: '#FBBF24',
    Concept: '#A78BFA',
    Entity: '#F87171',
  },
  relation_styles: {
    SUPPORTS: { line_color: '#10B981', target_arrow_color: '#10B981', width: 3, target_arrow_shape: 'triangle' },
  },
};

describe('SettingsModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (graphApi.getSettings as any).mockResolvedValue({ data: mockSettings });
    (graphApi.updateSettings as any).mockResolvedValue({ data: mockSettings });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders settings modal with title', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      expect(screen.getByText('User Settings')).toBeInTheDocument();
    });
  });

  it('shows edge labels checkbox', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      expect(screen.getByText('Show edge labels')).toBeInTheDocument();
    });
  });

  it('shows node colors section', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      expect(screen.getByText('Node Colors')).toBeInTheDocument();
    });
  });

  it('displays all node type color pickers', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      expect(screen.getByText('Observation')).toBeInTheDocument();
      expect(screen.getByText('Hypothesis')).toBeInTheDocument();
      expect(screen.getByText('Source')).toBeInTheDocument();
      expect(screen.getByText('Concept')).toBeInTheDocument();
      expect(screen.getByText('Entity')).toBeInTheDocument();
    });
  });

  it('closes modal when close button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    
    render(<SettingsModal onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText('User Settings')).toBeInTheDocument();
    });

    await user.click(screen.getByText('×'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes modal when Cancel button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    
    render(<SettingsModal onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls updateSettings when Save Settings button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    
    render(<SettingsModal onClose={onClose} />);
    
    await waitFor(() => {
      expect(screen.getByText('Save Settings')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Save Settings'));
    
    await waitFor(() => {
      expect(graphApi.updateSettings).toHaveBeenCalled();
    });
  });

  it('has relation styles section', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      expect(screen.getByText('Relation Styles')).toBeInTheDocument();
    });
  });

  it('checkbox starts with correct initial value from settings', async () => {
    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });
  });

  it('checkbox reflects false value when settings say false', async () => {
    (graphApi.getSettings as any).mockResolvedValue({
      data: { ...mockSettings, show_edge_labels: false },
    });

    render(<SettingsModal onClose={() => {}} />);
    
    await waitFor(() => {
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();
    });
  });
});
