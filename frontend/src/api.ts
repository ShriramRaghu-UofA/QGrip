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
}
