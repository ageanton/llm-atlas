import type {
  AblationConfig,
  EngineInitResult,
  EngineTraceResult,
} from './types';

interface PendingCall {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
}

interface WorkerResponse {
  id: number;
  ok: boolean;
  data?: unknown;
  error?: string;
}

export class TransformerEngineClient {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<number, PendingCall>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent) => {
      const res = event.data as WorkerResponse;
      const call = this.pending.get(res.id);
      if (!call) return;
      this.pending.delete(res.id);
      if (res.ok) {
        call.resolve(res.data);
      } else {
        call.reject(new Error(res.error ?? 'Unknown worker error'));
      }
    };
    this.worker.onerror = (event: ErrorEvent) => {
      const err = new Error(event.message || 'Worker crashed');
      for (const call of this.pending.values()) {
        call.reject(err);
      }
      this.pending.clear();
    };
  }

  private send<T>(payload: Record<string, unknown>): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.worker.postMessage({ id, ...payload });
    });
  }

  init(modelBaseUrl: string): Promise<EngineInitResult> {
    return this.send<EngineInitResult>({ type: 'init', modelBaseUrl });
  }

  trace(prompt: string, ablation?: AblationConfig): Promise<EngineTraceResult> {
    return this.send<EngineTraceResult>({ type: 'trace', prompt, ablation });
  }

  getTensor(name: string): Promise<Float32Array> {
    return this.send<Float32Array>({ type: 'getTensor', name });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}
