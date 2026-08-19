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
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    const body = path.includes('/api/v1/artifacts')
      ? { artifacts: [] }
      : { api_version: 1, profile: 'synthetic.json', device: 'synthetic', gestures: ['rest', 'open', 'close'], models: ['transformer', 'dense'] };
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }));
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

test('live inference renders backend predictions', async () => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    let body: object = { api_version: 1, profile: 'synthetic.json', device: 'synthetic', gestures: ['rest', 'open', 'close'], models: ['dense'] };
    if (path.includes('/api/v1/artifacts')) body = { artifacts: ['C:/data/model.pt'] };
    if (path.includes('/api/v1/inference/start')) body = { state: 'running', kind: 'inference' };
    if (path.includes('/api/v1/inference/status')) body = { state: 'running', kind: 'inference', prediction: { gesture: 'open', confidence: 0.91, activation: 0.64, latency_ms: 3.2 } };
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }));
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole('button', { name: 'Validate' }));
  const start = screen.getByRole('button', { name: 'Start live inference' });
  await waitFor(() => expect(start).toBeEnabled());
  await user.click(start);
  await waitFor(() => expect(screen.getByText('91%')).toBeInTheDocument(), { timeout: 2000 });
  expect(screen.getByText('open')).toBeInTheDocument();
  expect(screen.getByText('64%')).toBeInTheDocument();
});
