import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '../../test/utils';
import userEvent from '@testing-library/user-event';
import CreateNodeModal from '../CreateNodeModal';

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
    expect(screen.getByLabelText(/Observation Text/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Confidence/i)).toBeInTheDocument();
  });

  it('allows typing in observation text field', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const textarea = screen.getByLabelText(/Observation Text/i);
    await user.type(textarea, 'Test observation');

    expect(textarea).toHaveValue('Test observation');
  });

  it('updates confidence slider', async () => {
    const user = userEvent.setup();
    render(<CreateNodeModal onClose={vi.fn()} />);

    const slider = screen.getByLabelText(/Confidence/i);
    expect(slider).toHaveValue('0.8');

    await user.clear(slider);
    await user.type(slider, '0.5');

    expect(slider).toHaveValue('0.5');
  });
});
