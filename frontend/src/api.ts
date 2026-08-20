export type JobState = 'idle' | 'running' | 'completed' | 'cancelled' | 'failed';

export interface Bootstrap {
  api_version: number;
  profile: string;
  device: string;
  gestures: string[];
  models: string[];
}

export interface JobStatus {
  state: JobState;
  kind?: string;
  progress?: number;
  message?: string;
  gesture?: string;
  stage?: string;
  instruction?: string;
  stimulus_image?: string;
  elapsed_seconds?: number;
  duration_seconds?: number;
  activation?: number;
  result?: string;
  prediction?: Prediction;
  metrics?: EpochMetric[];
  health?: LiveSignalHealth;
  awaiting_command?: boolean;
}

export type SGTCommand = 'abort' | 'pause' | 'resume' | 'repeat';

export interface DoctorReport {
  ready: boolean;
  kind: string;
  sample_rate_hz: number;
  channels: number;
}

export interface Prediction {
  gesture: string;
  confidence: number;
  activation: number;
  latency_ms: number;
}

export interface EpochMetric {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface LiveSignalHealth {
  severity: string;
  warnings: string[];
  missing_values: number;
  lost_samples: number;
  malformed_packets: number;
  misaligned_packets: number;
  consumer_overruns: number;
}

export interface ArtifactList {
  artifacts: string[];
}

export class QGripApi {
  constructor(private token: string) {}

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
   * Subscribe to server-pushed job status over SSE, returning a disposer.
   * Falls back to `null` when the runtime lacks EventSource (e.g. jsdom tests),
   * letting callers poll instead.
   *
   * `onOpen` fires whenever the connection is (re)established. Because
   * EventSource reconnects automatically after a drop, callers use it to clear
   * any fallback polling that `onError` may have started.
   */
  subscribe(
    onStatus: (status: JobStatus) => void,
    onError?: () => void,
    onOpen?: () => void
  ): (() => void) | null {
    if (typeof EventSource === 'undefined') return null;
    const source = new EventSource(`/api/v1/stream?token=${encodeURIComponent(this.token)}`);
    source.onopen = () => onOpen?.();
    source.onmessage = (event) => {
      try {
        onStatus(JSON.parse(event.data) as JobStatus);
      } catch {
        /* ignore malformed frames */
      }
    };
    source.onerror = () => onError?.();
    return () => source.close();
  }
}
