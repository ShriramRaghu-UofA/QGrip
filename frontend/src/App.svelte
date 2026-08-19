<script lang="ts">
  import { onMount } from 'svelte';
  import type { Bootstrap, JobStatus } from './api';
  import { QGripApi } from './api';
  import MetricPlot from './MetricPlot.svelte';
  import StagePanel from './StagePanel.svelte';

  const stages = ['Setup', 'Collect', 'Train', 'Validate', 'Handi'] as const;
  type Stage = (typeof stages)[number];
  type Theme = 'dracula' | 'nord' | 'light';

  let stage: Stage = $state('Setup');
  let subject = $state('demo');
  let model = $state('transformer');
  let theme: Theme = $state('dracula');
  let bootstrap = $state.raw<Bootstrap | null>(null);
  let status = $state.raw<JobStatus>({ state: 'idle' });
  let error = $state('');
  let online = $state(true);
  let polling: number | undefined;

  const stageIndex = $derived(stages.indexOf(stage));
  const progress = $derived(Math.round((status.progress ?? 0) * 100));
  const token = new URLSearchParams(location.search).get('token') ?? '';
  const api = new QGripApi(token);

  onMount(() => {
    theme = (localStorage.getItem('qgrip-theme') as Theme | null) ?? 'dracula';
    document.documentElement.dataset.theme = theme;
    void loadBootstrap();
    return () => window.clearInterval(polling);
  });

  function chooseTheme(value: Theme): void {
    theme = value;
    document.documentElement.dataset.theme = value;
    localStorage.setItem('qgrip-theme', value);
  }

  async function loadBootstrap(): Promise<void> {
    try {
      bootstrap = await api.request<Bootstrap>('/api/v1/bootstrap');
      model = bootstrap.models[0] ?? 'transformer';
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function start(path: string, body: object): Promise<void> {
    try {
      status = await api.request<JobStatus>(path, { method: 'POST', body: JSON.stringify(body) });
      error = '';
      window.clearInterval(polling);
      polling = window.setInterval(() => void poll(), 500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function poll(): Promise<void> {
    try {
      status = await api.request<JobStatus>('/api/v1/training/status');
      if (status.state !== 'running') window.clearInterval(polling);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function handleKeys(event: KeyboardEvent): void {
    if (event.altKey && event.key === 'ArrowRight') stage = stages[Math.min(stages.length - 1, stageIndex + 1)];
    if (event.altKey && event.key === 'ArrowLeft') stage = stages[Math.max(0, stageIndex - 1)];
  }
</script>

<svelte:window bind:online onkeydown={handleKeys} />
<svelte:head><title>{stage} · QGrip Dashboard</title></svelte:head>

<div class="min-h-screen bg-base-100 text-base-content">
  <header class="navbar sticky top-0 z-20 border-b border-base-300 bg-base-100/95 px-4 backdrop-blur">
    <div class="flex-1 gap-3">
      <div class="grid size-10 place-items-center rounded-xl bg-primary font-black text-primary-content">Q</div>
      <div><h1 class="text-xl font-bold">QGrip</h1><p class="text-xs opacity-60">EMG workflow console</p></div>
    </div>
    <div class="flex-none gap-2">
      <span class={['badge', online ? 'badge-success' : 'badge-error']}>{online ? 'Online' : 'Offline'}</span>
      <select class="select select-sm" aria-label="Theme" value={theme} onchange={(event) => chooseTheme(event.currentTarget.value as Theme)}>
        <option value="dracula">Dracula</option><option value="nord">Nord</option><option value="light">Light</option>
      </select>
    </div>
  </header>

  <main class="mx-auto grid max-w-7xl gap-6 p-4 lg:grid-cols-[16rem_1fr] lg:p-8">
    <aside>
      <ul class="steps steps-vertical w-full" aria-label="Workflow progress">
        {#each stages as item (item)}
          <li class={['step', stages.indexOf(item) <= stageIndex && 'step-primary']}>
            <button class="btn btn-ghost w-full justify-start" aria-current={item === stage ? 'step' : undefined} onclick={() => (stage = item)}>{item}</button>
          </li>
        {/each}
      </ul>
      <div class="alert mt-6 text-sm"><span>Tip: Alt + ←/→ changes stages.</span></div>
    </aside>

    <div class="space-y-6">
      {#if error}<div class="alert alert-error" role="alert"><span>{error}</span><button class="btn btn-sm" onclick={() => (error = '')}>Dismiss</button></div>{/if}
      <div class="sr-only" aria-live="polite">{status.state}: {status.message ?? ''}</div>

      {#if stage === 'Setup'}
        <StagePanel title="Setup" description="Choose a subject, validate the profile, and confirm device readiness." active>
          <fieldset class="fieldset"><legend class="fieldset-legend">Subject</legend><input class="input w-full" bind:value={subject} autocomplete="off" /></fieldset>
          <div class="stats stats-vertical bg-base-300 sm:stats-horizontal"><div class="stat"><div class="stat-title">Device</div><div class="stat-value text-xl">{bootstrap?.device ?? 'Loading…'}</div></div><div class="stat"><div class="stat-title">Profile</div><div class="stat-desc max-w-72 truncate">{bootstrap?.profile ?? '—'}</div></div></div>
          <div class="card-actions justify-end"><button class="btn btn-primary" onclick={() => void api.request('/api/v1/doctor').catch((cause) => (error = String(cause)))}>Check readiness</button><button class="btn" onclick={() => (stage = 'Collect')}>Continue</button></div>
        </StagePanel>
      {:else if stage === 'Collect'}
        <StagePanel title="Collect" description="Backend timing and markers drive calibration, practice, and gesture trials." active>
          <div class="text-center"><div class="text-6xl font-black">{status.message || 'Ready'}</div><progress class="progress progress-primary mt-5 w-full" value={progress} max="100"></progress><p>{progress}% complete</p></div>
          <div class="card-actions justify-end"><button class="btn btn-error btn-outline" onclick={() => void api.request('/api/v1/sgt/command?command=abort', { method: 'POST' })}>Abort</button><button class="btn btn-primary" onclick={() => void start('/api/v1/sgt/start', { subject, discrete: false })}>Start collection</button></div>
        </StagePanel>
      {:else if stage === 'Train'}
        <StagePanel title="Train" description="Use the latest compatible session by default or explicitly combine sessions." active>
          <select class="select w-full" bind:value={model} aria-label="Model preset">{#each bootstrap?.models ?? [] as item (item)}<option value={item}>{item}</option>{/each}</select>
          <progress class="progress progress-secondary w-full" value={progress} max="100"></progress>
          <div class="card-actions justify-end"><button class="btn" onclick={() => void api.request('/api/v1/training/cancel', { method: 'POST' })}>Cancel</button><button class="btn btn-secondary" onclick={() => void start('/api/v1/training/start', { subject, model, inputs: [], discrete: false })}>Train model</button></div>
        </StagePanel>
      {:else if stage === 'Validate'}
        <StagePanel title="Validate" description="Inspect class, confidence, activation, signal health, and end-to-end latency." active>
          <div class="stats stats-vertical bg-base-300 sm:stats-horizontal"><div class="stat"><div class="stat-title">Class</div><div class="stat-value">rest</div></div><div class="stat"><div class="stat-title">Confidence</div><div class="stat-value text-success">—</div></div><div class="stat"><div class="stat-title">Latency</div><div class="stat-value">— ms</div></div></div>
          <MetricPlot />
          <div class="alert"><span>Start inference when a compatible checkpoint is selected.</span></div>
        </StagePanel>
      {:else}
        <StagePanel title="Handi" description="Observe the remote standalone controller and perform bounded calibration." active>
          <div class="alert alert-warning"><span>Stopping software commands does not disable servo torque or replace a physical emergency stop.</span></div>
          <div class="stats bg-base-300"><div class="stat"><div class="stat-title">UNO Q</div><div class="stat-value text-xl">Not connected</div><div class="stat-desc">Dashboard uses a server-side proxy</div></div></div>
          <div class="card-actions justify-end"><button class="btn btn-primary" onclick={() => void api.request('/api/v1/handi/status').catch((cause) => (error = String(cause)))}>Refresh status</button></div>
        </StagePanel>
      {/if}
    </div>
  </main>
</div>
