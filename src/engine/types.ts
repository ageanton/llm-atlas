export interface ModelConfig {
  vocab_size: number;
  max_seq_len: number;
  n_layers: number;
  d_model: number;
  n_heads: number;
  d_head: number;
  d_mlp: number;
  ln_eps: number;
  gelu: string;
  tied_embeddings: boolean;
}

export interface TensorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface TensorHistogram {
  counts: number[];
  edges: number[];
}

export interface TensorMeta {
  name: string;
  shape: number[];
  count: number;
  dtype: 'float16';
  offset_bytes: number;
  byte_length: number;
  role: string;
  stats: TensorStats;
  histogram: TensorHistogram;
}

export interface ModelManifest {
  version: number;
  weights_file: string;
  weights_sha256: string;
  total_bytes: number;
  total_params: number;
  config: ModelConfig;
  tensors: TensorMeta[];
}

export interface TokenizerData {
  version: number;
  vocab_size: number;
  eos_id: number;
  unk_id: number;
  pretokenize_regex: string;
  vocab: string[];
  merges: [string, string][];
}

export interface EncodedToken {
  id: number;
  text: string;
  start: number;
  end: number;
}

export interface AblationConfig {
  disabledHeads?: [number, number][];
  disabledAttnBlocks?: number[];
  disabledMlpBlocks?: number[];
  disabledResiduals?: ['attn' | 'mlp', number][];
  disabledLayerNorms?: ['ln_1' | 'ln_2' | 'ln_f', number][];
  disabledLayers?: number[];
}

export interface AttentionTrace {
  q: number[][][]; // [heads, seq, d_head]
  k: number[][][];
  v: number[][][];
  rawScores: number[][][]; // [heads, seq, seq]
  maskedScores: number[][][];
  attnWeights: number[][][];
  weightedV: number[][][]; // [heads, seq, d_head]
  output: number[][]; // [seq, d_model]
}

export interface MLPTrace {
  hidden: number[][]; // [seq, d_mlp]
  output: number[][]; // [seq, d_model]
}

export interface SkippedLayerTrace {
  skipped: true;
  residualOut: number[][];
}

export interface ActiveLayerTrace {
  skipped: false;
  residualBeforeAttn: number[][];
  ln1Out: number[][];
  attn: AttentionTrace;
  residualAfterAttn: number[][];
  ln2Out: number[][];
  mlp: MLPTrace;
  residualAfterMlp: number[][];
}

export type LayerTrace = SkippedLayerTrace | ActiveLayerTrace;

export interface TopTokenProb {
  id: number;
  prob: number;
}

export interface LogitLensEntry {
  afterLayer: number;
  topTokens: TopTokenProb[];
}

export interface Trace {
  tokenIds: number[];
  tokEmb: number[][];
  posEmb: number[][];
  layers: LayerTrace[];
  logitLens: LogitLensEntry[];
  lnFOut: number[][];
  finalLogits: number[];
  finalProbs: number[];
  greedyNextTokenId: number;
  inputLoss: number | null;
}

export interface WeightStore {
  manifest: ModelManifest;
  tensors: Map<string, Float32Array>;
}

export interface EngineInitResult {
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  loadMs: number;
}

export interface EngineTraceResult {
  prompt: string;
  tokens: EncodedToken[];
  trace: Trace;
  elapsedMs: number;
}
