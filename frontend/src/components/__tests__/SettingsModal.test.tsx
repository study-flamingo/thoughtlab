import { vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '../../test/utils';
import SettingsModal from '../SettingsModal';
import { graphApi } from '../../services/api';

describe('SettingsModal', () => {
  beforeEach(() => {
    vi.spyOn(graphApi, 'getSettings').mockResolvedValue({
      data: {
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
      },
    } as any);
    vi.spyOn(graphApi, 'updateSettings').mockResolvedValue({
      data: {},
    } as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders and allows toggling edge labels', async () => {
    render(<SettingsModal onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByText('User Settings')).toBeInTheDocument();
    });
    const checkbox = screen.getByLabelText(/Show edge labels/i) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(false);
  });
});


