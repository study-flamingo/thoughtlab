import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../test/utils';
import userEvent from '@testing-library/user-event';
import CreateNodeModal from '../CreateNodeModal';

// Mock the API
vi.mock('../../services/api', () => ({
  graphApi: {
    createObservation: vi.fn().mockResolvedValue({ data: { id: '123' } }),
    createEntity: vi.fn().mockResolvedValue({ data: { id: '124' } }),
    createSource: vi.fn().mockResolvedValue({ data: { id: '125' } }),
    createHypothesis: vi.fn().mockResolvedValue({ data: { id: '126' } }),
    createConcept: vi.fn().mockResolvedValue({ data: { id: '127' } }),
  },
}));

describe('CreateNodeModal', () => {
  it('renders modal when open', () => {
    render(<CreateNodeModal onClose={vi.fn()} />);
    expect(screen.getByText('Create New Node')).toBeInTheDocument();
  });

  it('closes modal when cancel button is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={onClose} />);

    const cancelButton = screen.getByText('Cancel');
    await user.click(cancelButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes modal when close button (×) is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={onClose} />);

    const closeButton = screen.getByText('×');
    await user.click(closeButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows observation form fields when observation is selected', () => {
    render(<CreateNodeModal onClose={vi.fn()} />);
    // Check for observation-specific placeholder text
    expect(screen.getByPlaceholderText('Describe what you observed...')).toBeInTheDocument();
    // Check for confidence display
    expect(screen.getByText(/Confidence:/)).toBeInTheDocument();
  });

  it('allows typing in observation text field', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const textarea = screen.getByPlaceholderText('Describe what you observed...');
    await user.type(textarea, 'Test observation');

    expect(textarea).toHaveValue('Test observation');
  });

  it('displays confidence percentage', () => {
    render(<CreateNodeModal onClose={vi.fn()} />);
    // Default confidence is 0.8 = 80%
    expect(screen.getByText('Confidence: 80%')).toBeInTheDocument();
  });

  it('shows node type selector with all options', () => {
    render(<CreateNodeModal onClose={vi.fn()} />);
    
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getByText('Observation')).toBeInTheDocument();
    expect(screen.getByText('Hypothesis')).toBeInTheDocument();
    expect(screen.getByText('Source')).toBeInTheDocument();
    expect(screen.getByText('Entity')).toBeInTheDocument();
    expect(screen.getByText('Concept')).toBeInTheDocument();
  });

  it('switches to source form when source is selected', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'source');

    expect(screen.getByPlaceholderText('Source title...')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('https://...')).toBeInTheDocument();
  });

  it('switches to entity form when entity is selected', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'entity');

    expect(screen.getByPlaceholderText('Enter entity name...')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Describe the entity...')).toBeInTheDocument();
  });

  it('switches to hypothesis form when hypothesis is selected', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'hypothesis');

    expect(screen.getByPlaceholderText('Short name for display...')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Full hypothesis statement...')).toBeInTheDocument();
  });

  it('switches to concept form when concept is selected', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'concept');

    expect(screen.getByPlaceholderText('Enter concept name...')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Describe the concept...')).toBeInTheDocument();
  });

  it('has a Create Node submit button', () => {
    render(<CreateNodeModal onClose={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Create Node' })).toBeInTheDocument();
  });
});
