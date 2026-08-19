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
  result?: string;
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
