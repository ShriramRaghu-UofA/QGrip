/** Lifecycle states exposed by the backend's single-workflow coordinator. */
export type JobState = 'idle' | 'running' | 'completed' | 'cancelled' | 'failed';

/** Profile-derived values needed to initialize dashboard controls. */
export interface Bootstrap {
  api_version: number;
  profile: string;
  device: string;
  gestures: string[];
  models: string[];
  proportional: boolean;
  activation_tolerance: number;
}

/** Authoritative, replace-not-patch snapshot broadcast on the status SSE channel. */
export interface JobStatus {
  state: JobState;
  kind?: string;
  progress?: number;
  message?: string;
  gesture?: string;
  trial?: number;
  stage?: string;
  instruction?: string;
  stimulus_image?: string;
  elapsed_seconds?: number;
  duration_seconds?: number;
  activation?: number;
  measured_activation?: number;
  in_tolerance?: boolean;
  result?: string;
  prediction?: Prediction;
  metrics?: EpochMetric[];
  training_summary?: TrainingSummary;
  health?: LiveSignalHealth;
  awaiting_command?: boolean;
}

/** Manual Screen Guided Training command accepted while capture awaits input. */
export type SGTCommand = 'abort' | 'pause' | 'resume' | 'repeat';

/** Device-probe result used to determine whether collection can start. */
export interface DoctorReport {
  ready: boolean;
  kind: string;
  sample_rate_hz: number;
  channels: number;
}

/** One classified gesture with confidence, effort estimate, and model latency. */
export interface Prediction {
  gesture: string;
  confidence: number;
  activation: number;
  latency_ms: number;
}

/** Metrics recorded after a complete training epoch. */
export interface EpochMetric {
  epoch: number;
  loss: number;
  accuracy: number;
  training_loss: number;
  training_accuracy: number;
}

/** Number of window samples from one label in each data-split partition. */
export interface ClassSampleCount {
  label: string;
  training: number;
  validation: number;
}

/** Dataset window construction result sent before/while training proceeds. */
export interface TrainingSummary {
  training_samples: number;
  validation_samples: number;
  window_size: number;
  classes: ClassSampleCount[];
}

/** Streamer and QGrip consumer continuity assessment for live inference. */
export interface LiveSignalHealth {
  severity: string;
  warnings: string[];
  missing_values: number;
  lost_samples: number;
  malformed_packets: number;
  misaligned_packets: number;
  consumer_overruns: number;
}

/** Artifact-discovery response body. */
export interface ArtifactList {
  artifacts: string[];
  calibration_ready: boolean;
}

/**
 * A discrete, ephemeral notification pushed on the `notification` SSE channel.
 * It is best-effort: the dashboard may drop it (e.g. when the tab is hidden)
 * without losing correctness, since authoritative state lives on `status`.
 */
export interface Notification {
  kind: string;
  level: 'success' | 'info' | 'warning' | 'error';
  message: string;
}

/** Handlers for the named channels multiplexed over the SSE connection. */
export interface StreamHandlers {
  onStatus: (status: JobStatus) => void;
  onNotification?: (notification: Notification) => void;
  onError?: () => void;
  onOpen?: () => void;
}

export class QGripApi {
  /** Wrap authenticated dashboard HTTP and Server-Sent Event operations. */
  constructor(private token: string) {}

  /** Send an authenticated JSON API request and decode its successful response. */
  async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(path, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        'X-QGrip-Token': this.token,
        ...init?.headers,
      },
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { error?: { message?: string } };
      throw new Error(body.error?.message ?? `Request failed (${response.status})`);
    }
    return (await response.json()) as T;
  }

  /** Send an interactive Screen Guided Training control command. */
  sgtCommand(command: SGTCommand): Promise<{ accepted: boolean; command: SGTCommand }> {
    return this.request(`/api/v1/sgt/command?command=${command}`, { method: 'POST' });
  }

  /**
   * Subscribe to the server's named SSE channels, returning a disposer.
   * Falls back to `null` when the runtime lacks EventSource (e.g. jsdom tests),
   * letting callers poll instead.
   *
   * Two channels share one authenticated connection:
   *   - `status`: the authoritative snapshot the UI must always apply.
   *   - `notification`: discrete, ignorable toasts the caller may drop.
   *
   * `onOpen` fires whenever the connection is (re)established. Because
   * EventSource reconnects automatically after a drop, callers use it to clear
   * any fallback polling that `onError` may have started.
   */
  subscribe(handlers: StreamHandlers): (() => void) | null {
    if (typeof EventSource === 'undefined') return null;
    const source = new EventSource(`/api/v1/stream?token=${encodeURIComponent(this.token)}`);
    source.onopen = () => handlers.onOpen?.();
    source.addEventListener('status', (event) => {
      try {
        handlers.onStatus(JSON.parse((event as MessageEvent).data) as JobStatus);
      } catch {
        /* ignore malformed frames */
      }
    });
    source.addEventListener('notification', (event) => {
      try {
        handlers.onNotification?.(JSON.parse((event as MessageEvent).data) as Notification);
      } catch {
        /* ignore malformed frames */
      }
    });
    source.onerror = () => handlers.onError?.();
    return () => source.close();
  }
}
