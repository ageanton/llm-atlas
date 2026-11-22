import { TinyBPETokenizer } from './tokenizer';
import { loadWeights, runForwardTrace } from './transformer';
import type {
  AblationConfig,
  ModelManifest,
  TokenizerData,
  WeightStore,
} from './types';

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent) => void) | null;
}

type WorkerRequest =
  | { id: number; type: 'init'; modelBaseUrl: string }
  | { id: number; type: 'trace'; prompt: string; ablation?: AblationConfig }
  | { id: number; type: 'getTensor'; name: string };

const workerSelf = self as unknown as WorkerScope;

let store: WeightStore | null = null;
let tokenizer: TinyBPETokenizer | null = null;

workerSelf.onmessage = async (event: MessageEvent) => {
  const req = event.data as WorkerRequest;
  try {
    if (req.type === 'init') {
      const t0 = performance.now();
      const base = req.modelBaseUrl.endsWith('/')
        ? req.modelBaseUrl
        : `${req.modelBaseUrl}/`;

      const [manifestRes, tokRes, weightsRes] = await Promise.all([
        fetch(`${base}manifest.json`),
        fetch(`${base}tokenizer.json`),
        fetch(`${base}weights.fp16.bin`),
      ]);

      if (!manifestRes.ok) {
        throw new Error(`Failed to fetch manifest.json (${manifestRes.status})`);
      }
      if (!tokRes.ok) {
        throw new Error(`Failed to fetch tokenizer.json (${tokRes.status})`);
      }
      if (!weightsRes.ok) {
        throw new Error(
          `Failed to fetch weights.fp16.bin (${weightsRes.status})`
        );
      }

      const manifest = (await manifestRes.json()) as ModelManifest;
      const tokData = (await tokRes.json()) as TokenizerData;
      const weightsBuf = await weightsRes.arrayBuffer();

      if (weightsBuf.byteLength !== manifest.total_bytes) {
        throw new Error(
          `Weight byte size mismatch: expected ${manifest.total_bytes}, got ${weightsBuf.byteLength}`
        );
      }

      store = loadWeights(manifest, weightsBuf);
      tokenizer = new TinyBPETokenizer(tokData);

      workerSelf.postMessage({
        id: req.id,
        ok: true,
        data: {
          manifest,
          tokenizer: tokData,
          loadMs: Math.round(performance.now() - t0),
        },
      });
      return;
    }

    if (!store || !tokenizer) {
      throw new Error('Engine not initialized. Call init first.');
    }

    if (req.type === 'trace') {
      const t0 = performance.now();
      const tokens = tokenizer
        .encodeWithSpans(req.prompt)
        .slice(0, store.manifest.config.max_seq_len);
      if (tokens.length === 0) {
        throw new Error('Enter at least one character to trace.');
      }
      const trace = runForwardTrace(
        store,
        tokens.map((t) => t.id),
        req.ablation
      );
      workerSelf.postMessage({
        id: req.id,
        ok: true,
        data: {
          prompt: req.prompt,
          tokens,
          trace,
          elapsedMs: Number((performance.now() - t0).toFixed(2)),
        },
      });
      return;
    }

    if (req.type === 'getTensor') {
      const tensor = store.tensors.get(req.name);
      if (!tensor) {
        throw new Error(`Unknown tensor: ${req.name}`);
      }
      const copy = new Float32Array(tensor);
      workerSelf.postMessage(
        { id: req.id, ok: true, data: copy },
        [copy.buffer]
      );
      return;
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    workerSelf.postMessage({ id: req.id, ok: false, error: message });
  }
};
