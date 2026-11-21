import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ErrorBoundary from '../components/common/ErrorBoundary';

function ProblemChild({ shouldThrow }) {
  if (shouldThrow) {
    throw new Error('Test rendering crash');
  }
  return <div>Normal Content Rendered</div>;
}

describe('ErrorBoundary Component', () => {
  it('renders children without error under standard execution', () => {
    render(
      <ErrorBoundary>
        <ProblemChild shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByText('Normal Content Rendered')).toBeInTheDocument();
  });

  it('safely catches render exceptions and presents the recovery safeguard UI', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <ProblemChild shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/FloraGuard Resilience Safeguard/i)).toBeInTheDocument();
    expect(screen.getByText(/Return to AI Scanner/i)).toBeInTheDocument();
    expect(screen.getByText(/Reload View/i)).toBeInTheDocument();

    consoleSpy.mockRestore();
  });
});
