import { cleanup, render, screen } from '@testing-library/svelte';
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
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ api_version: 1, profile: 'synthetic.json', device: 'synthetic', gestures: ['rest', 'open', 'close'], models: ['transformer', 'dense'], token: 'test' }), { status: 200, headers: { 'Content-Type': 'application/json' } })));
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
