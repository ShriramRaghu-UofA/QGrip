<script lang="ts">
  import { onMount } from 'svelte';
  import type {
    ArtifactList,
    BenchmarkResult,
    BenchmarkSuite,
    Bootstrap,
    DoctorReport,
    JobStatus,
    ModelSummary,
    Notification,
    Prediction,
  } from './api';
  import { QGripApi } from './api';
  import BenchmarkPlot from './BenchmarkPlot.svelte';
  import MetricPlot from './MetricPlot.svelte';
  import ModelSummaryCard from './ModelSummaryCard.svelte';
  import StagePanel from './StagePanel.svelte';

  /** Ordered dashboard workflow stages used for navigation and progress context. */
  const stages = ['Setup', 'Collect', 'Train', 'Validate'] as const;
  type Stage = (typeof stages)[number];
  type Theme = 'dracula' | 'nord' | 'light';

  let stage: Stage = $state('Setup');
  let subject = $state('demo');
  let model = $state('transformer');
  let theme: Theme = $state('dracula');
  let bootstrap = $state.raw<Bootstrap | null>(null);
  let status = $state.raw<JobStatus>({ state: 'idle' });
  let artifacts = $state.raw<string[]>([]);
  let calibrationReady = $state(false);
  let capturePath = $state('');
  let trainingInput = $state('');
  let modelPath = $state('');
  let predictionHistory = $state.raw<Prediction[]>([]);
  let benchmarkResults = $state.raw<BenchmarkResult[]>([]);
  let benchmarkIterations = $state(1000);
  let benchmarkWarmup = $state(20);
  let benchmarking = $state(false);
  let trainingModelSummary = $state.raw<ModelSummary | null>(null);
  let checkpointModelSummary = $state.raw<ModelSummary | null>(null);
  let trainingModelSummaryLoading = $state(false);
  let checkpointModelSummaryLoading = $state(false);
  let error = $state('');
  let online = $state(true);

  // Ephemeral, ignorable notifications surfaced as daisyUI toasts. Missing one
  // (e.g. while the tab is hidden) is acceptable — authoritative state lives on
  // the `status` channel, so we simply drop them when the tab is not visible.
  let toasts = $state.raw<{ id: number; level: Notification['level']; message: string }[]>([]);
  let toastSeq = 0;

  // Device readiness (Setup) — surfaced instead of a silent doctor call.
  let doctor = $state.raw<DoctorReport | null>(null);
  let checkingDevice = $state(false);

  // Screen Guided Training state machine controls.
  const activationDisplayTimeConstantMs = 300;
  let autoMode = $state(true);
  let stimulusFailed = $state(false);
  let displayedActivation = $state(0);
  let activationDisplayUpdatedAt: number | undefined;

  // Transport: prefer server push (SSE), fall back to polling when unavailable.
  let sseActive = $state(false);
  let disposeStream: (() => void) | null = null;
  let polling: number | undefined;
  let statusPath = '/api/v1/training/status';

  // Frontend-owned smooth countdown so pacing never depends on network jitter.
  let localElapsed = $state(0);
  let countdownTimer: number | undefined;
  let phaseKey = '';
  let trainingSummaryRequest = 0;
  let checkpointSummaryRequest = 0;

  const stageIndex = $derived(stages.indexOf(stage));
  const progress = $derived(Math.round((status.progress ?? 0) * 100));
  const sgtRunning = $derived(status.kind === 'sgt' && status.state === 'running');
  const calibrationRunning = $derived(status.kind === 'calibration' && status.state === 'running');
  const proportional = $derived(bootstrap?.proportional ?? true);
  const deviceReady = $derived(doctor?.ready ?? false);
  const inferenceRunning = $derived(status.kind === 'inference' && status.state === 'running');
  const awaiting = $derived(!!status.awaiting_command);
  const stimulusUrl = $derived(
    status.stimulus_image ? `/stimuli/${encodeURIComponent(status.stimulus_image)}` : ''
  );
  const gestureLabel = $derived((status.gesture ?? '').replace(/_/g, ' '));
  const duration = $derived(status.duration_seconds ?? 0);
  const timedStage = $derived(
    (sgtRunning || calibrationRunning) &&
      !awaiting &&
      duration > 0 &&
      ['calibration', 'preparation', 'presentation', 'practice'].includes(status.stage ?? '')
  );
  const preparing = $derived(status.stage === 'preparation');
  const countdownPercent = $derived(
    duration > 0 ? Math.min(100, Math.round((100 * localElapsed) / duration)) : 0
  );
  const countdownRemaining = $derived(Math.max(0, duration - localElapsed));
  const targetPercent = $derived(Math.round((status.activation ?? 0) * 100));
  const measuredPercent = $derived(Math.round(displayedActivation * 100));
  const displayInTolerance = $derived(
    Math.abs(displayedActivation - (status.activation ?? 0)) <=
      (bootstrap?.activation_tolerance ?? 0.1)
  );

  // Training telemetry surfaced on the Train stage.
  const latestMetric = $derived(status.metrics?.at(-1));
  const trainingSummary = $derived(status.training_summary);
  const percent = (value: number | undefined): string => `${Math.round((value ?? 0) * 100)}%`;

  const token = new URLSearchParams(location.search).get('token') ?? '';
  const api = new QGripApi(token);

  onMount(() => {
    theme = (localStorage.getItem('qgrip-theme') as Theme | null) ?? 'dracula';
    document.documentElement.dataset.theme = theme;
    void loadBootstrap();
    disposeStream = api.subscribe({
      onStatus: applyStatus,
      onNotification: notify,
      onError: () => (sseActive = false),
      onOpen: () => {
        // The stream (re)connected: stop any fallback polling and prefer push.
        sseActive = true;
        window.clearInterval(polling);
      },
    });
    sseActive = disposeStream !== null;
    return () => {
      disposeStream?.();
      window.clearInterval(polling);
      stopCountdown();
    };
  });

  // Drive the countdown locally whenever a new timed stimulus begins.
  $effect(() => {
    const key = `${status.kind}|${status.stage}|${status.gesture}|${status.trial}|${status.activation}|${status.duration_seconds}`;
    if (timedStage) {
      if (key !== phaseKey) {
        phaseKey = key;
        startCountdown(duration);
      }
    } else {
      phaseKey = '';
      stopCountdown();
    }
  });

  // Reset the broken-image fallback whenever the stimulus changes.
  $effect(() => {
    void stimulusUrl;
    stimulusFailed = false;
  });

  /** Start a frontend-only smooth timer for the currently backend-timed SGT phase. */
  function startCountdown(seconds: number): void {
    stopCountdown();
    localElapsed = 0;
    const started = performance.now();
    countdownTimer = window.setInterval(() => {
      localElapsed = (performance.now() - started) / 1000;
      if (localElapsed >= seconds) {
        localElapsed = seconds;
        stopCountdown();
      }
    }, 50);
  }

  /** Stop and clear the local countdown interval, if one is active. */
  function stopCountdown(): void {
    if (countdownTimer !== undefined) {
      window.clearInterval(countdownTimer);
      countdownTimer = undefined;
    }
  }

  /** Display a best-effort terminal notification while the tab is visible. */
  function notify(note: Notification): void {
    // Ignorable by design: skip the toast entirely when the tab is not visible.
    if (typeof document !== 'undefined' && document.hidden) return;
    const id = ++toastSeq;
    toasts = [...toasts, { id, level: note.level, message: note.message }];
    window.setTimeout(() => {
      toasts = toasts.filter((toast) => toast.id !== id);
    }, 5000);
  }

  /** Map a backend notification severity to its daisyUI alert class. */
  function toastAlert(level: Notification['level']): string {
    if (level === 'success') return 'alert-success';
    if (level === 'error') return 'alert-error';
    if (level === 'warning') return 'alert-warning';
    return 'alert-info';
  }

  /** Apply and persist the selected dashboard theme. */
  function chooseTheme(value: Theme): void {
    theme = value;
    document.documentElement.dataset.theme = value;
    localStorage.setItem('qgrip-theme', value);
  }

  /** Load initial profile UI choices, then refresh the current subject's artifacts. */
  async function loadBootstrap(): Promise<void> {
    try {
      bootstrap = await api.request<Bootstrap>('/api/v1/bootstrap');
      model = bootstrap.models[0] ?? 'transformer';
      await Promise.all([loadArtifacts(), loadTrainingModelSummary(model)]);
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  /** Refresh artifacts and select sensible first Parquet/checkpoint defaults. */
  async function loadArtifacts(): Promise<void> {
    const response = await api.request<ArtifactList>(
      `/api/v1/artifacts?subject=${encodeURIComponent(subject)}`
    );
    artifacts = response.artifacts ?? [];
    calibrationReady = response.calibration_ready ?? false;
    trainingInput ||= artifacts.find((path) => path.endsWith('.parquet')) ?? '';
    if (!modelPath) {
      modelPath = artifacts.find((path) => path.endsWith('.pt')) ?? '';
      if (modelPath) await loadCheckpointModelSummary(modelPath);
    }
  }

  /** Load a profile-shaped preview for the selected training preset. */
  async function loadTrainingModelSummary(selected: string): Promise<void> {
    const request = ++trainingSummaryRequest;
    trainingModelSummaryLoading = true;
    try {
      const summary = await api.request<ModelSummary>(
        `/api/v1/models/${encodeURIComponent(selected)}/summary?proportional=${proportional}`
      );
      if (request === trainingSummaryRequest) trainingModelSummary = summary;
    } catch (cause) {
      if (request === trainingSummaryRequest) {
        trainingModelSummary = null;
        error = cause instanceof Error ? cause.message : String(cause);
      }
    } finally {
      if (request === trainingSummaryRequest) trainingModelSummaryLoading = false;
    }
  }

  /** Change the training preset and refresh its effective architecture preview. */
  function chooseTrainingModel(value: string): void {
    model = value;
    void loadTrainingModelSummary(value);
  }

  /** Load strict facts from a selected checkpoint, optionally promoting them to Train. */
  async function loadCheckpointModelSummary(
    path: string,
    promoteToTraining = false
  ): Promise<void> {
    const request = ++checkpointSummaryRequest;
    if (!path) {
      checkpointModelSummary = null;
      checkpointModelSummaryLoading = false;
      return;
    }
    checkpointModelSummaryLoading = true;
    try {
      const summary = await api.request<ModelSummary>(
        `/api/v1/checkpoints/summary?model=${encodeURIComponent(path)}`
      );
      if (request === checkpointSummaryRequest) {
        checkpointModelSummary = summary;
        if (promoteToTraining) trainingModelSummary = summary;
      }
    } catch (cause) {
      if (request === checkpointSummaryRequest) {
        checkpointModelSummary = null;
        error = cause instanceof Error ? cause.message : String(cause);
      }
    } finally {
      if (request === checkpointSummaryRequest) checkpointModelSummaryLoading = false;
    }
  }

  /** Run the server-side device probe and surface either its result or error. */
  async function checkDevice(): Promise<void> {
    checkingDevice = true;
    doctor = null;
    try {
      doctor = await api.request<DoctorReport>('/api/v1/doctor');
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      checkingDevice = false;
    }
  }

  /** Benchmark CPU and each available GPU provider for the selected checkpoint. */
  async function runBenchmark(): Promise<void> {
    benchmarking = true;
    benchmarkResults = [];
    try {
      const suite = await api.request<BenchmarkSuite>('/api/v1/benchmark', {
        method: 'POST',
        body: JSON.stringify({
          model: modelPath,
          iterations: benchmarkIterations,
          warmup: benchmarkWarmup,
        }),
      });
      benchmarkResults = suite.results;
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      benchmarking = false;
    }
  }

  /** Select a checkpoint and discard benchmark results for the previous model. */
  function chooseCheckpoint(value: string): void {
    modelPath = value;
    benchmarkResults = [];
    void loadCheckpointModelSummary(value);
  }

  /** Clear subject-scoped artifact and checkpoint state while the subject changes. */
  function resetSubjectArtifacts(): void {
    calibrationReady = false;
    capturePath = '';
    trainingInput = '';
    modelPath = '';
    checkpointModelSummary = null;
    checkpointModelSummaryLoading = false;
    benchmarkResults = [];
    checkpointSummaryRequest += 1;
  }

  /** Replace local state with an authoritative status snapshot and derive UI updates. */
  function applyStatus(next: JobStatus): void {
    const wasRunning = status.state === 'running';
    const wasSgtRunning = status.kind === 'sgt' && wasRunning;
    status = next;
    updateDisplayedActivation(next, wasSgtRunning);
    if (next.prediction) {
      predictionHistory = [...predictionHistory.slice(-79), next.prediction];
    }
    if (next.state !== 'running') {
      if (!sseActive) window.clearInterval(polling);
      if (wasRunning) {
        if (next.kind === 'calibration' && next.state === 'completed' && next.result)
          calibrationReady = true;
        if (next.kind === 'sgt' && next.result) capturePath = next.result;
        if (next.kind === 'export' && next.result) trainingInput = next.result;
        if (next.kind === 'training' && next.result) {
          modelPath = next.result;
          void loadCheckpointModelSummary(next.result, true);
        }
        void loadArtifacts();
      }
    }
  }

  /** Smooth noisy live activation for display without changing captured or trained data. */
  function updateDisplayedActivation(next: JobStatus, wasSgtRunning: boolean): void {
    if (next.kind !== 'sgt' || next.state !== 'running') {
      displayedActivation = 0;
      activationDisplayUpdatedAt = undefined;
      return;
    }
    const raw = Math.max(0, Math.min(1, next.measured_activation ?? 0));
    const now = performance.now();
    if (!wasSgtRunning || activationDisplayUpdatedAt === undefined) {
      displayedActivation = raw;
    } else {
      const elapsed = Math.max(0, now - activationDisplayUpdatedAt);
      const alpha = 1 - Math.exp(-elapsed / activationDisplayTimeConstantMs);
      displayedActivation += alpha * (raw - displayedActivation);
    }
    activationDisplayUpdatedAt = now;
  }

  /** Start one backend workflow and activate polling only when SSE is unavailable. */
  async function start(path: string, body: object, nextStatusPath: string): Promise<void> {
    try {
      const initial = await api.request<JobStatus>(path, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      applyStatus(initial);
      statusPath = nextStatusPath;
      error = '';
      if (!sseActive) {
        window.clearInterval(polling);
        polling = window.setInterval(() => void poll(), 500);
      }
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  /** Fetch and apply the current workflow snapshot for the polling fallback. */
  async function poll(): Promise<void> {
    try {
      applyStatus(await api.request<JobStatus>(statusPath));
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  /** Reset live telemetry and start the selected automatic or manual SGT collection. */
  function startCollection(): void {
    predictionHistory = [];
    void start(
      '/api/v1/sgt/start',
      { subject, discrete: !proportional, auto: autoMode },
      '/api/v1/sgt/status'
    );
  }

  /** Start the canonical subject activation calibration. */
  function startCalibration(): void {
    void start('/api/v1/sgt/calibration/start', { subject }, '/api/v1/sgt/calibration/status');
  }

  /** Refresh subject-scoped prerequisites before opening the collection stage. */
  async function continueToCollection(): Promise<void> {
    calibrationReady = false;
    try {
      await loadArtifacts();
      stage = 'Collect';
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  /** Move between workflow panels with Alt+left and Alt+right keyboard shortcuts. */
  function handleKeys(event: KeyboardEvent): void {
    if (event.altKey && event.key === 'ArrowRight')
      stage = stages[Math.min(stages.length - 1, stageIndex + 1)];
    if (event.altKey && event.key === 'ArrowLeft')
      stage = stages[Math.max(0, stageIndex - 1)];
  }
</script>

<svelte:window bind:online onkeydown={handleKeys} />
<svelte:head><title>{stage} · QGrip Dashboard</title></svelte:head>

<div class="min-h-screen bg-base-100 text-base-content">
  {#if toasts.length}
    <div class="toast toast-end z-50" aria-live="polite">
      {#each toasts as toast (toast.id)}
        <div class={['alert', toastAlert(toast.level)]}><span>{toast.message}</span></div>
      {/each}
    </div>
  {/if}
  <header class="navbar sticky top-0 z-20 border-b border-base-300 bg-base-100/95 px-4 backdrop-blur">
    <div class="flex-1 gap-3">
      <div class="grid size-10 place-items-center rounded-xl bg-primary font-black text-primary-content">Q</div>
      <div><h1 class="text-xl font-bold">QGrip</h1><p class="text-xs opacity-60">EMG workflow console</p></div>
    </div>
    <div class="flex-none items-center gap-2">
      {#if doctor}
        <span class="badge badge-success gap-1">Device ready</span>
      {:else}
        <span class="badge badge-ghost">Device unverified</span>
      {/if}
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
        <StagePanel title="Setup" description="Review the profile, connect the device, and confirm readiness before collecting." active>
          <fieldset class="fieldset"><legend class="fieldset-legend">Subject</legend><input class="input w-full" bind:value={subject} autocomplete="off" oninput={resetSubjectArtifacts} /></fieldset>
          <div class="stats stats-vertical bg-base-300 sm:stats-horizontal">
            <div class="stat"><div class="stat-title">Device</div><div class="stat-value text-xl">{bootstrap?.device ?? 'Loading…'}</div></div>
            <div class="stat"><div class="stat-title">Profile</div><div class="stat-desc max-w-72 truncate">{bootstrap?.profile ?? '—'}</div></div>
            <div class="stat"><div class="stat-title">Gestures</div><div class="stat-value text-xl">{bootstrap?.gestures.length ?? 0}</div></div>
          </div>
          {#if bootstrap?.gestures.length}
            <div class="flex flex-wrap gap-2">{#each bootstrap.gestures as item (item)}<span class="badge badge-outline">{item.replace(/_/g, ' ')}</span>{/each}</div>
          {/if}
          <div class="card border border-base-300 bg-base-100">
            <div class="card-body gap-3">
              <div class="flex items-center justify-between">
                <h3 class="font-semibold">Device readiness</h3>
                <button class="btn btn-primary btn-sm" disabled={checkingDevice} onclick={() => void checkDevice()}>
                  {#if checkingDevice}<span class="loading loading-spinner loading-xs"></span> Checking…{:else}Connect device{/if}
                </button>
              </div>
              {#if doctor}
                <div class="stats bg-base-300">
                  <div class="stat"><div class="stat-title">Status</div><div class="stat-value text-success text-xl">Ready</div><div class="stat-desc">{doctor.kind}</div></div>
                  <div class="stat"><div class="stat-title">Sample rate</div><div class="stat-value text-xl">{doctor.sample_rate_hz} Hz</div></div>
                  <div class="stat"><div class="stat-title">Channels</div><div class="stat-value text-xl">{doctor.channels}</div></div>
                </div>
              {:else}
                <p class="text-sm text-base-content/70">Connect to verify the EMG stream through the same worker path used during capture.</p>
              {/if}
            </div>
          </div>
          <div class="card-actions justify-end"><button class="btn btn-primary" disabled={!deviceReady} onclick={() => void continueToCollection()}>Continue to collection</button></div>
        </StagePanel>
      {:else if stage === 'Collect'}
        <StagePanel title="Collect" description={proportional ? 'Complete subject calibration before collecting proportional training data.' : 'Discrete collection does not require subject activation calibration.'} active>
          {#if !deviceReady}
            <div class="alert alert-warning"><span>Connect and verify the device in Setup before starting a session.</span></div>
          {/if}

          <div class="grid gap-3 md:grid-cols-2">
            {#if proportional}
              <div class={['card border', calibrationReady ? 'border-success bg-success/10' : 'border-primary bg-primary/10']}>
                <div class="card-body gap-2 p-4">
                  <div class="flex items-center justify-between gap-2">
                    <span class="badge badge-primary badge-outline">Step 1</span>
                    <span class={['badge', calibrationReady ? 'badge-success' : 'badge-warning']}>{calibrationReady ? 'Complete' : 'Required'}</span>
                  </div>
                  <h3 class="card-title text-lg">Calibrate activation</h3>
                  <p class="text-sm text-base-content/70">Record rest and maximum effort for this subject so proportional targets can be measured.</p>
                </div>
              </div>
            {:else}
              <div class="card border border-base-300 bg-base-100">
                <div class="card-body gap-2 p-4">
                  <div class="flex items-center justify-between gap-2"><span class="badge badge-outline">Calibration</span><span class="badge badge-ghost">Skipped</span></div>
                  <h3 class="card-title text-lg">No calibration needed</h3>
                  <p class="text-sm text-base-content/70">This profile uses discrete labels, so collection can begin immediately.</p>
                </div>
              </div>
            {/if}
            <div class={['card border', proportional && !calibrationReady ? 'border-base-300 bg-base-100 opacity-60' : 'border-primary bg-primary/10']}>
              <div class="card-body gap-2 p-4">
                <div class="flex items-center justify-between gap-2">
                  <span class="badge badge-primary badge-outline">Step {proportional ? 2 : 1}</span>
                  <span class={['badge', proportional && !calibrationReady ? 'badge-ghost' : 'badge-info']}>{proportional && !calibrationReady ? 'Locked' : 'Next'}</span>
                </div>
                <h3 class="card-title text-lg">Collect training data</h3>
                <p class="text-sm text-base-content/70">Follow the gesture prompts to create the authoritative training capture.</p>
              </div>
            </div>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-3">
            <label class="label cursor-pointer gap-3">
              <span class={['label-text', !autoMode && 'font-semibold']}>Manual</span>
              <input type="checkbox" class="toggle toggle-primary" bind:checked={autoMode} disabled={sgtRunning || calibrationRunning} aria-label="Auto advance" />
              <span class={['label-text', autoMode && 'font-semibold']}>Auto</span>
            </label>
            <span class="badge badge-outline">{proportional ? 'Proportional' : 'Discrete'} collection</span>
            <span class="badge badge-primary badge-outline">{awaiting ? 'Waiting for you' : (status.stage ?? ((sgtRunning || calibrationRunning) ? 'Running' : 'Ready'))}</span>
          </div>

          <div class="rounded-box border border-base-300 bg-base-200 p-6 text-center">
            <h2 class="mb-4 text-2xl font-bold">{(status.instruction ?? status.message) || 'Ready to begin collection.'}</h2>
            <div class="grid min-h-64 place-items-center">
              {#if awaiting}
                <div class="space-y-1"><div class="text-5xl">✓</div><p class="text-base-content/70">Stimulus recorded — repeat it or proceed to the next.</p></div>
              {:else if stimulusUrl && !stimulusFailed}
                <img class="max-h-80 rounded-box object-contain" src={stimulusUrl} alt={`Gesture: ${gestureLabel}`} onerror={() => (stimulusFailed = true)} />
              {:else if status.gesture}
                <div class="rounded-box border border-base-300 px-12 py-10 text-5xl font-black capitalize">{gestureLabel}</div>
              {:else}
                <p class="text-base-content/60">Press start to begin the guided session.</p>
              {/if}
            </div>

            {#if (sgtRunning || calibrationRunning) && !awaiting && duration > 0}
              <div class="mt-6 space-y-1 text-left">
                <div class="flex justify-between text-sm"><span>{preparing ? 'Get ready' : 'Hold the gesture'}</span><span>{countdownRemaining.toFixed(1)} s</span></div>
                <progress class={['progress w-full', preparing ? 'progress-warning' : 'progress-accent']} value={countdownPercent} max="100"></progress>
              </div>
            {/if}
          </div>

          <div class="space-y-1 text-left">
            <div class="flex justify-between text-sm"><span>Session progress</span><span>{progress}%</span></div>
            <progress class="progress progress-primary w-full" value={progress} max="100"></progress>
            {#if sgtRunning}
              <div class="mt-2 space-y-1">
                <div class="flex justify-between text-sm text-base-content/70"><span>{preparing ? 'Next target' : 'Target'}</span><span>{targetPercent}%</span></div>
                <progress class="progress progress-secondary w-full" value={targetPercent} max="100"></progress>
                <div class="flex justify-between text-sm text-base-content/70"><span>Measured EMG</span><span>{measuredPercent}%</span></div>
                <progress class="progress progress-accent w-full" value={measuredPercent} max="100"></progress>
                <div class={['text-xs', displayInTolerance ? 'text-success' : 'text-warning']}>
                  {displayInTolerance ? 'Within target band' : 'Outside target band'}
                </div>
              </div>
            {/if}
          </div>

          {#if capturePath}<div class="alert alert-success"><span>Capture saved: {capturePath}</span></div>{/if}

          <div class="card-actions justify-between">
            <div class="flex gap-2">
              {#if capturePath}<button class="btn btn-secondary" onclick={() => void start('/api/v1/export/start', { capture: capturePath }, '/api/v1/export/status')}>Export Parquet</button>{/if}
            </div>
            <div class="flex gap-2">
              {#if calibrationRunning}
                <button class="btn btn-error btn-outline" onclick={() => void api.sgtCommand('abort')}>Abort calibration</button>
              {:else if sgtRunning}
                {#if awaiting}
                  <button class="btn btn-warning" onclick={() => void api.sgtCommand('repeat')}>Repeat</button>
                  <button class="btn btn-primary" onclick={() => void api.sgtCommand('resume')}>Proceed</button>
                {:else}
                  {#if autoMode}<button class="btn" onclick={() => void api.sgtCommand('pause')}>Pause</button>{/if}
                  <button class="btn btn-error btn-outline" onclick={() => void api.sgtCommand('abort')}>Abort</button>
                {/if}
              {:else}
                {#if proportional && !calibrationReady}
                  <button class="btn btn-primary" disabled={!deviceReady} onclick={startCalibration}>Start required calibration</button>
                {:else}
                  {#if proportional}<button class="btn btn-ghost" disabled={!deviceReady} onclick={startCalibration}>Recalibrate</button>{/if}
                  <button class="btn btn-primary" disabled={!deviceReady} onclick={startCollection}>Start training data collection</button>
                {/if}
              {/if}
            </div>
          </div>
        </StagePanel>
      {:else if stage === 'Train'}
        <StagePanel title="Train" description="Use the latest compatible session by default or explicitly combine sessions." active>
          <select class="select w-full" value={model} onchange={(event) => chooseTrainingModel(event.currentTarget.value)} aria-label="Model preset">{#each bootstrap?.models ?? [] as item (item)}<option value={item}>{item}</option>{/each}</select>
          <select class="select w-full" bind:value={trainingInput} aria-label="Training session"><option value="">Latest compatible session</option>{#each artifacts.filter((path) => path.endsWith('.parquet')) as path (path)}<option value={path}>{path}</option>{/each}</select>
          <ModelSummaryCard summary={trainingModelSummary} loading={trainingModelSummaryLoading} />
          <div class="space-y-1">
            <div class="flex justify-between text-sm"><span>{latestMetric ? `Epoch ${latestMetric.epoch}` : 'Training progress'}</span><span>{progress}%</span></div>
            <progress class="progress progress-secondary w-full" value={progress} max="100"></progress>
          </div>

          {#if trainingSummary}
            <div class="card border border-base-300 bg-base-100">
              <div class="card-body gap-3">
                <div class="flex items-center justify-between">
                  <h3 class="font-semibold">Dataset</h3>
                  <span class="text-xs opacity-60">{trainingSummary.window_size}-sample windows</span>
                </div>
                <div class="stats bg-base-300">
                  <div class="stat"><div class="stat-title">Total windows</div><div class="stat-value text-xl">{trainingSummary.training_samples + trainingSummary.validation_samples}</div></div>
                  <div class="stat"><div class="stat-title">Training</div><div class="stat-value text-xl">{trainingSummary.training_samples}</div></div>
                  <div class="stat"><div class="stat-title">Validation</div><div class="stat-value text-xl">{trainingSummary.validation_samples}</div></div>
                </div>
                <div class="overflow-x-auto">
                  <table class="table table-sm">
                    <thead><tr><th>Class</th><th class="text-right">Train</th><th class="text-right">Validation</th><th class="text-right">Total</th></tr></thead>
                    <tbody>
                      {#each trainingSummary.classes as entry (entry.label)}
                        <tr><td class="capitalize">{entry.label.replace(/_/g, ' ')}</td><td class="text-right">{entry.training}</td><td class="text-right">{entry.validation}</td><td class="text-right">{entry.training + entry.validation}</td></tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          {/if}

          {#if latestMetric}
            <div class="stats stats-vertical bg-base-300 sm:stats-horizontal">
              <div class="stat"><div class="stat-title">Training accuracy</div><div class="stat-value text-xl">{percent(latestMetric.training_accuracy)}</div><div class="stat-desc">Loss {latestMetric.training_loss.toFixed(4)}</div></div>
              <div class="stat"><div class="stat-title">Validation accuracy</div><div class="stat-value text-xl">{percent(latestMetric.accuracy)}</div><div class="stat-desc">Loss {latestMetric.loss.toFixed(4)}</div></div>
            </div>
          {/if}

          {#if status.metrics?.length}
            <details class="collapse collapse-arrow border border-base-300 bg-base-100">
              <summary class="collapse-title font-semibold">Per-epoch history</summary>
              <div class="collapse-content overflow-x-auto">
                <table class="table table-sm table-pin-rows">
                  <thead><tr><th>Epoch</th><th class="text-right">Train loss</th><th class="text-right">Train acc</th><th class="text-right">Val loss</th><th class="text-right">Val acc</th></tr></thead>
                  <tbody>
                    {#each status.metrics as row (row.epoch)}
                      <tr><td>{row.epoch}</td><td class="text-right">{row.training_loss.toFixed(4)}</td><td class="text-right">{percent(row.training_accuracy)}</td><td class="text-right">{row.loss.toFixed(4)}</td><td class="text-right">{percent(row.accuracy)}</td></tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            </details>
          {/if}
          {#if modelPath}<div class="alert alert-success"><span>Checkpoint ready: {modelPath}</span></div>{/if}
          <div class="card-actions justify-end"><button class="btn" onclick={() => void api.request('/api/v1/training/cancel', { method: 'POST' })}>Cancel</button><button class="btn btn-secondary" onclick={() => void start('/api/v1/training/start', { subject, model, inputs: trainingInput ? [trainingInput] : [], discrete: !proportional }, '/api/v1/training/status')}>Train model</button></div>
        </StagePanel>
      {:else}
        <StagePanel title="Validate" description="Inspect class, confidence, activation, signal health, and end-to-end latency." active>
          <select class="select w-full" value={modelPath} onchange={(event) => chooseCheckpoint(event.currentTarget.value)} aria-label="Inference checkpoint"><option value="">Select a checkpoint</option>{#each artifacts.filter((path) => path.endsWith('.pt')) as path (path)}<option value={path}>{path}</option>{/each}</select>
          <ModelSummaryCard summary={checkpointModelSummary} loading={checkpointModelSummaryLoading} />
          <div class="card border border-base-300 bg-base-100">
            <div class="card-body gap-4">
              <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 class="card-title text-lg">Inference benchmark</h3>
                  <p class="text-sm text-base-content/70">Compare ONNX Runtime and Torch on CPU, plus CUDA providers when this computer has them.</p>
                  <span class="badge badge-outline mt-2">Live preference: {(bootstrap?.device_preference ?? 'gpu').toUpperCase()}</span>
                </div>
                <button class="btn btn-secondary" disabled={!modelPath || benchmarking || status.state === 'running'} onclick={() => void runBenchmark()}>
                  {#if benchmarking}<span class="loading loading-spinner loading-sm"></span> Benchmarking…{:else}Run benchmark{/if}
                </button>
              </div>
              <div class="grid gap-3 sm:grid-cols-2">
                <label class="fieldset">
                  <span class="fieldset-legend">Timed inference windows</span>
                  <input class="input w-full" aria-label="Timed inference windows" type="number" min="1" max="10000" step="1" bind:value={benchmarkIterations} disabled={benchmarking} />
                  <span class="label">One window is one batch-1 prediction per runtime.</span>
                </label>
                <label class="fieldset">
                  <span class="fieldset-legend">Warmup windows</span>
                  <input class="input w-full" aria-label="Warmup windows" type="number" min="0" max="1000" step="1" bind:value={benchmarkWarmup} disabled={benchmarking} />
                  <span class="label">Run before timing for every runtime.</span>
                </label>
              </div>
              {#if benchmarkResults.length}
                <BenchmarkPlot results={benchmarkResults} />
                <details class="collapse collapse-arrow border border-base-300 bg-base-200">
                  <summary class="collapse-title font-semibold">Raw benchmark results</summary>
                  <div class="collapse-content overflow-x-auto">
                    <table class="table table-sm">
                      <thead><tr><th>Runtime</th><th>Device</th><th class="text-right">Median</th><th class="text-right">p95</th><th class="text-right">p99</th><th class="text-right">Throughput</th></tr></thead>
                      <tbody>
                        {#each benchmarkResults as result (`${result.backend}-${result.device}`)}
                          <tr><td class="uppercase">{result.backend}</td><td class="uppercase"><span class={['badge', result.device === 'gpu' ? 'badge-accent' : 'badge-ghost']}>{result.device}</span></td><td class="text-right">{result.median_ms.toFixed(3)} ms</td><td class="text-right">{result.p95_ms.toFixed(3)} ms</td><td class="text-right">{result.p99_ms.toFixed(3)} ms</td><td class="text-right">{result.throughput_hz.toFixed(1)}/s</td></tr>
                        {/each}
                      </tbody>
                    </table>
                  </div>
                </details>
              {:else if !benchmarking}
                <p class="text-sm text-base-content/60">Select a checkpoint and run a hardware-local benchmark. No EMG device is required.</p>
              {/if}
            </div>
          </div>
          <div class="stats stats-vertical bg-base-300 sm:stats-horizontal"><div class="stat"><div class="stat-title">Class</div><div class="stat-value">{status.prediction?.gesture ?? '—'}</div></div><div class="stat"><div class="stat-title">Confidence</div><div class="stat-value text-success">{status.prediction ? `${Math.round(status.prediction.confidence * 100)}%` : '—'}</div></div><div class="stat"><div class="stat-title">Activation</div><div class="stat-value">{status.prediction ? `${Math.round(status.prediction.activation * 100)}%` : '—'}</div></div><div class="stat"><div class="stat-title">Latency</div><div class="stat-value">{status.prediction ? status.prediction.latency_ms.toFixed(1) : '—'} ms</div></div></div>
          {#if status.health}
            <div class={[
              'alert',
              status.health.severity === 'healthy'
                ? 'alert-success'
                : ['critical', 'fatal'].includes(status.health.severity)
                  ? 'alert-error'
                  : 'alert-warning'
            ]}>
              <span>
                Signal health: {status.health.severity}. Device loss: {status.health.lost_samples}; missing values: {status.health.missing_values}; malformed packets: {status.health.malformed_packets}; consumer overruns: {status.health.consumer_overruns}.
                {status.health.warnings.join(' ')}
              </span>
            </div>
          {/if}
          <MetricPlot history={predictionHistory} />
          <div class="card-actions justify-end">{#if inferenceRunning}<button class="btn btn-error" onclick={() => void api.request('/api/v1/inference/stop', { method: 'POST' })}>Stop live inference</button>{:else}<button class="btn btn-primary" disabled={!modelPath} onclick={() => void start('/api/v1/inference/start', { model: modelPath }, '/api/v1/inference/status')}>Start live inference</button>{/if}</div>
        </StagePanel>
      {/if}
    </div>
  </main>
</div>
