import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AIToolsSection, AIToolButton } from '../AIToolsSection';

describe('AIToolsSection', () => {
  it('renders collapsed by default', () => {
    render(
      <AIToolsSection>
        <div data-testid="child-content">Test content</div>
      </AIToolsSection>
    );

    // Header should be visible
    expect(screen.getByText('AI Tools')).toBeInTheDocument();
    // Collapse indicator should show collapsed state
    expect(screen.getByText('▶')).toBeInTheDocument();
    // Children should not be visible
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
  });

  it('renders expanded when defaultExpanded is true', () => {
    render(
      <AIToolsSection defaultExpanded={true}>
        <div data-testid="child-content">Test content</div>
      </AIToolsSection>
    );

    // Collapse indicator should show expanded state
    expect(screen.getByText('▼')).toBeInTheDocument();
    // Children should be visible
    expect(screen.getByTestId('child-content')).toBeInTheDocument();
  });

  it('expands on click when collapsed', () => {
    render(
      <AIToolsSection>
        <div data-testid="child-content">Test content</div>
      </AIToolsSection>
    );

    // Initially collapsed
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(screen.getByRole('button'));

    // Should now be expanded
    expect(screen.getByTestId('child-content')).toBeInTheDocument();
    expect(screen.getByText('▼')).toBeInTheDocument();
  });

  it('collapses on click when expanded', () => {
    render(
      <AIToolsSection defaultExpanded={true}>
        <div data-testid="child-content">Test content</div>
      </AIToolsSection>
    );

    // Initially expanded
    expect(screen.getByTestId('child-content')).toBeInTheDocument();

    // Click to collapse
    fireEvent.click(screen.getByRole('button'));

    // Should now be collapsed
    expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
    expect(screen.getByText('▶')).toBeInTheDocument();
  });

  it('renders the sparkle emoji', () => {
    render(
      <AIToolsSection>
        <div>Content</div>
      </AIToolsSection>
    );

    expect(screen.getByText('✨')).toBeInTheDocument();
  });
});

describe('AIToolButton', () => {
  it('renders with label and icon', () => {
    render(
      <AIToolButton
        label="Find Related"
        icon="🔍"
        onClick={() => {}}
      />
    );

    expect(screen.getByText('Find Related')).toBeInTheDocument();
    expect(screen.getByText('🔍')).toBeInTheDocument();
  });

  it('renders without icon when not provided', () => {
    render(
      <AIToolButton
        label="Test Button"
        onClick={() => {}}
      />
    );

    expect(screen.getByText('Test Button')).toBeInTheDocument();
    // Only the label text should be in the button
    const button = screen.getByRole('button');
    expect(button).toHaveTextContent('Test Button');
  });

  it('calls onClick handler when clicked', () => {
    const handleClick = vi.fn();
    render(
      <AIToolButton
        label="Click Me"
        onClick={handleClick}
      />
    );

    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('shows loading state when isLoading is true', () => {
    render(
      <AIToolButton
        label="Find Related"
        icon="🔍"
        onClick={() => {}}
        isLoading={true}
      />
    );

    // Should show loading spinner and text
    expect(screen.getByText('⏳')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    // Original icon and label should not be shown
    expect(screen.queryByText('🔍')).not.toBeInTheDocument();
    expect(screen.queryByText('Find Related')).not.toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    const handleClick = vi.fn();
    render(
      <AIToolButton
        label="Disabled Button"
        onClick={handleClick}
        disabled={true}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();

    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('is disabled when isLoading is true', () => {
    const handleClick = vi.fn();
    render(
      <AIToolButton
        label="Loading Button"
        onClick={handleClick}
        isLoading={true}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();

    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('applies danger variant styles', () => {
    render(
      <AIToolButton
        label="Delete"
        onClick={() => {}}
        variant="danger"
      />
    );

    const button = screen.getByRole('button');
    // Check for danger variant class (red styling)
    expect(button.className).toContain('bg-red-50');
    expect(button.className).toContain('text-red-700');
  });

  it('applies default variant styles by default', () => {
    render(
      <AIToolButton
        label="Default"
        onClick={() => {}}
      />
    );

    const button = screen.getByRole('button');
    // Check for default variant class (purple styling)
    expect(button.className).toContain('bg-purple-50');
    expect(button.className).toContain('text-purple-700');
  });
});
