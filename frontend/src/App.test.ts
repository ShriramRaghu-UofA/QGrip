import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import App from './App.svelte';

vi.mock('uplot', () => ({
  default: class {
    destroy(): void {}
  },
}));

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: 'synthetic.json',
        device: 'synthetic',
        gestures: ['rest', 'open', 'close'],
        models: ['transformer', 'dense'],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes('/api/v1/artifacts')) body = { artifacts: [], calibration_ready: false };
      if (path.includes('/api/v1/doctor'))
        body = {
          ready: true,
          kind: 'synthetic',
          sample_rate_hz: 200,
          channels: 8,
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
});

afterEach(cleanup);

test('all workflow stages are keyboard-accessible', async () => {
  const user = userEvent.setup();
  render(App);
  for (const stage of ['Setup', 'Collect', 'Train', 'Validate', 'Handi']) {
    await user.click(screen.getByRole('button', { name: stage }));
    expect(screen.getByRole('heading', { name: stage, level: 2 })).toBeInTheDocument();
  }
});

test('theme selection persists', async () => {
  const user = userEvent.setup();
  render(App);
  await user.selectOptions(screen.getByRole('combobox', { name: 'Theme' }), 'nord');
  expect(localStorage.getItem('qgrip-theme')).toBe('nord');
});

test('proportional collection makes calibration the required next step', async () => {
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole('button', { name: 'Connect device' }));
  const continueButton = screen.getByRole('button', {
    name: 'Continue to collection',
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  expect(screen.getByText('Calibrate activation')).toBeInTheDocument();
  expect(screen.getByText('Locked')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Start required calibration' })).toBeEnabled();
  expect(screen.queryByRole('button', { name: 'Start training data collection' })).not.toBeInTheDocument();
});

test('discrete collection skips calibration', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: 'synthetic.json',
        device: 'synthetic',
        gestures: ['rest', 'open'],
        models: ['dense'],
        proportional: false,
        activation_tolerance: 0.1,
      };
      if (path.includes('/api/v1/artifacts')) body = { artifacts: [], calibration_ready: false };
      if (path.includes('/api/v1/doctor'))
        body = {
          ready: true,
          kind: 'synthetic',
          sample_rate_hz: 200,
          channels: 8,
        };
      if (path.includes('/api/v1/sgt/start')) body = { state: 'running', kind: 'sgt' };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole('button', { name: 'Connect device' }));
  const continueButton = screen.getByRole('button', {
    name: 'Continue to collection',
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  expect(screen.getByText('No calibration needed')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Start training data collection' }));
  const startCall = vi.mocked(fetch).mock.calls.find(([input]) => String(input).includes('/api/v1/sgt/start'));
  expect(startCall).toBeDefined();
  expect(JSON.parse(String(startCall?.[1]?.body))).toMatchObject({
    discrete: true,
  });
});

test('SGT activation guidance remains visible during preparation', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: 'synthetic.json',
        device: 'synthetic',
        gestures: ['rest', 'open'],
        models: ['dense'],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes('/api/v1/artifacts')) body = { artifacts: [], calibration_ready: true };
      if (path.includes('/api/v1/doctor'))
        body = {
          ready: true,
          kind: 'synthetic',
          sample_rate_hz: 200,
          channels: 8,
        };
      if (path.includes('/api/v1/sgt/start'))
        body = {
          state: 'running',
          kind: 'sgt',
          stage: 'preparation',
          gesture: 'open',
          activation: 0.75,
          measured_activation: 0.42,
          duration_seconds: 2,
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole('button', { name: 'Connect device' }));
  const continueButton = screen.getByRole('button', {
    name: 'Continue to collection',
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  await user.click(screen.getByRole('button', { name: 'Start training data collection' }));
  await waitFor(() => expect(screen.getByText('Next target')).toBeInTheDocument());
  expect(screen.getByText('75%')).toBeInTheDocument();
  expect(screen.getByText('42%')).toBeInTheDocument();
});

test('live inference renders backend predictions', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: 'synthetic.json',
        device: 'synthetic',
        gestures: ['rest', 'open', 'close'],
        models: ['dense'],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes('/api/v1/artifacts')) body = { artifacts: ['C:/data/model.pt'], calibration_ready: false };
      if (path.includes('/api/v1/inference/start')) body = { state: 'running', kind: 'inference' };
      if (path.includes('/api/v1/inference/status'))
        body = {
          state: 'running',
          kind: 'inference',
          prediction: {
            gesture: 'open',
            confidence: 0.91,
            activation: 0.64,
            latency_ms: 3.2,
          },
          health: {
            severity: 'warning',
            warnings: ['device samples lost'],
            missing_values: 2,
            lost_samples: 3,
            malformed_packets: 0,
            misaligned_packets: 0,
            consumer_overruns: 1,
          },
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole('button', { name: 'Validate' }));
  const start = screen.getByRole('button', { name: 'Start live inference' });
  await waitFor(() => expect(start).toBeEnabled());
  await user.click(start);
  await waitFor(() => expect(screen.getByText('91%')).toBeInTheDocument(), {
    timeout: 2000,
  });
  expect(screen.getByText('open')).toBeInTheDocument();
  expect(screen.getByText('64%')).toBeInTheDocument();
  expect(screen.getByText(/Device loss: 3/)).toBeInTheDocument();
  expect(screen.getByText(/consumer overruns: 1/)).toBeInTheDocument();
});
