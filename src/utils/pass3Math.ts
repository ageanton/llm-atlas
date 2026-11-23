export interface ReplayCheckpoint {
  step: number;
  train_loss: number;
  val_loss: number;
  scalars: Record<string, number>;
  prompts: {
    prompt: string;
    completion: string;
    top5: { id: number; token: string; prob: number }[];
  }[];
}

export interface TrainingReplayData {
  version: number;
  tracked_scalars_meta: {
    key: string;
    param: string;
    coord: number[];
    description: string;
  }[];
  loss_curve: { step: number; train_loss: number; val_loss: number }[];
  checkpoints: ReplayCheckpoint[];
}

export interface ScaleStop {
  label: string;
  params: number;
  layers: number;
  dModel: number;
  heads: number;
  context: number;
  note: string;
}

export const SCALE_STOPS: ScaleStop[] = [
  {
    label: 'This Atlas Model (TinyStories)',
    params: 862_464,
    layers: 4,
    dModel: 128,
    heads: 4,
    context: 64,
    note: '1.65 MB FP16. Exact Transformer math; writes simple story clauses, no world encyclopedia.',
  },
  {
    label: 'GPT-2 Small (2019)',
    params: 124_000_000,
    layers: 12,
    dModel: 768,
    heads: 12,
    context: 1024,
    note: '144× larger. Holds coherent multi-paragraph topics and basic factual patterns.',
  },
  {
    label: '8B Open-Weight Class',
    params: 8_000_000_000,
    layers: 32,
    dModel: 4096,
    heads: 32,
    context: 8192,
    note: '9,275× larger. Uses RoPE, GQA, RMSNorm, SwiGLU; capable of coding and broad recall.',
  },
  {
    label: '70B Class',
    params: 70_000_000_000,
    layers: 80,
    dModel: 8192,
    heads: 64,
    context: 128000,
    note: '81,163× larger. Reliable multi-step reasoning and nuanced instruction following emerge.',
  },
  {
    label: 'Frontier Scale (~1T+)',
    params: 1_000_000_000_000,
    layers: 128,
    dModel: 16384,
    heads: 128,
    context: 1000000,
    note: 'Over 1.15M× larger. Adds sparse Mixture-of-Experts routing atop the same residual loop.',
  },
];

export const FRONTIER_DIFFS: { part: string; here: string; frontier: string }[] = [
  { part: 'Position', here: 'wpe [64×128]', frontier: 'RoPE rotations' },
  { part: 'Norm', here: 'Pre-LayerNorm', frontier: 'Pre-RMSNorm' },
  { part: 'Attention', here: '4-head MHA', frontier: 'GQA / MLA + KV Cache' },
  { part: 'FFN / MLP', here: 'Dense GELU 4×', frontier: 'SwiGLU + Sparse MoE' },
  { part: 'Alignment', here: 'Pretraining only', frontier: 'SFT + RLHF + Reasoning RL' },
];

const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

export function geluTanh(x: number): number {
  return 0.5 * x * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
}

export function countActiveNeurons(vec: number[], threshold = 0.05): number {
  let count = 0;
  for (let i = 0; i < vec.length; i++) {
    if (vec[i] > threshold) count++;
  }
  return count;
}

export interface SampledToken {
  id: number;
  prob: number;
  logit: number;
}

export function computeSampledDistribution(
  logits: number[],
  temperature: number,
  topK: number,
  topP: number
): SampledToken[] {
  const temp = Math.max(0.05, temperature);
  const scaled = logits.map((v) => v / temp);
  let maxVal = -Infinity;
  for (let i = 0; i < scaled.length; i++) {
    if (scaled[i] > maxVal) maxVal = scaled[i];
  }
  const indexed = scaled.map((v, id) => ({
    id,
    logit: logits[id],
    exp: Math.exp(v - maxVal),
  }));
  indexed.sort((a, b) => b.exp - a.exp || a.id - b.id);

  const kClamped = Math.max(1, Math.min(indexed.length, Math.round(topK)));
  const afterK = indexed.slice(0, kClamped);
  let kSum = 0;
  for (const item of afterK) kSum += item.exp;

  const afterP: SampledToken[] = [];
  let cumulative = 0;
  for (const item of afterK) {
    const p = item.exp / kSum;
    afterP.push({ id: item.id, prob: p, logit: item.logit });
    cumulative += p;
    if (cumulative >= topP && afterP.length >= 1) break;
  }
  let finalSum = 0;
  for (const item of afterP) finalSum += item.prob;
  return afterP.map((item) => ({
    id: item.id,
    logit: item.logit,
    prob: finalSum > 0 ? item.prob / finalSum : 0,
  }));
}

export function sampleFromDistribution(dist: SampledToken[]): number {
  const r = Math.random();
  let acc = 0;
  for (const item of dist) {
    acc += item.prob;
    if (r <= acc) return item.id;
  }
  return dist[dist.length - 1]?.id ?? 0;
}

function crossEntropyAndTargetProb(
  logits: number[],
  targetId: number
): { loss: number; targetProb: number } {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    sumExp += Math.exp(logits[i] - maxVal);
  }
  const logSumExp = maxVal + Math.log(sumExp);
  const loss = logSumExp - logits[targetId];
  return { loss, targetProb: Math.exp(-loss) };
}

export interface TokenLossEntry {
  pos: number;
  targetId: number;
  loss: number;
  targetProb: number;
}

/**
 * Computes exact per-position teacher-forced logits & cross-entropy loss directly
 * from the live trace's lnFOut [T, 128] and real wte.weight [512, 128].
 */
export function computePerTokenLoss(
  lnFOut: number[][],
  tokenIds: number[],
  wte: Float32Array,
  vocabSize: number
): { entries: TokenLossEntry[]; allLogits: number[][] } {
  const T = tokenIds.length;
  const C = lnFOut[0]?.length ?? 128;
  const allLogits: number[][] = [];
  const entries: TokenLossEntry[] = [];

  for (let t = 0; t < T - 1; t++) {
    const row = lnFOut[t];
    const logits = new Array<number>(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      const base = v * C;
      let sum = 0;
      for (let d = 0; d < C; d++) sum += row[d] * wte[base + d];
      logits[v] = sum;
    }
    allLogits.push(logits);
    const targetId = tokenIds[t + 1];
    const { loss, targetProb } = crossEntropyAndTargetProb(logits, targetId);
    entries.push({ pos: t + 1, targetId, loss, targetProb });
  }

  return { entries, allLogits };
}

export interface ManualBackpropLandscape {
  targetTokenId: number;
  dim: number;
  trainedWeight: number;
  currentWeight: number;
  baseLoss: number;
  currentLoss: number;
  gradient: number;
  curve: { delta: number; loss: number }[];
}

/**
 * Exact live single-weight perturbation on lm_head (wte.weight[targetId, dim]).
 * Evaluates true cross-entropy loss and analytical dL/dw across delta in [-2, +2].
 */
export function computeManualWeightLandscape(
  lnFOut: number[][],
  tokenIds: number[],
  wte: Float32Array,
  vocabSize: number,
  userDelta: number
): ManualBackpropLandscape | null {
  const T = tokenIds.length;
  if (T < 2) return null;
  const C = lnFOut[0]?.length ?? 128;
  const lastPos = T - 2;
  const targetTokenId = tokenIds[T - 1];

  // Pick the channel with strongest activation so nudging this single weight visibly moves loss
  let dim = 0;
  let bestAct = -1;
  for (let d = 0; d < C; d++) {
    const a = Math.abs(lnFOut[lastPos][d]);
    if (a > bestAct) {
      bestAct = a;
      dim = d;
    }
  }

  const { allLogits } = computePerTokenLoss(lnFOut, tokenIds, wte, vocabSize);
  const evalAtDelta = (delta: number): { loss: number; grad: number } => {
    let totalLoss = 0;
    let totalGrad = 0;
    const count = T - 1;
    for (let t = 0; t < count; t++) {
      const base = allLogits[t];
      const act = lnFOut[t][dim];
      const shift = delta * act;
      const orig = base[targetTokenId];
      base[targetTokenId] = orig + shift;
      const { loss } = crossEntropyAndTargetProb(base, tokenIds[t + 1]);
      // Softmax probability of targetTokenId at position t
      let maxV = -Infinity;
      for (let v = 0; v < vocabSize; v++) if (base[v] > maxV) maxV = base[v];
      let sumE = 0;
      for (let v = 0; v < vocabSize; v++) sumE += Math.exp(base[v] - maxV);
      const pTarget = Math.exp(base[targetTokenId] - maxV) / sumE;
      const indicator = tokenIds[t + 1] === targetTokenId ? 1 : 0;
      totalGrad += (pTarget - indicator) * act;
      totalLoss += loss;
      base[targetTokenId] = orig;
    }
    return { loss: totalLoss / count, grad: totalGrad / count };
  };

  const baseEval = evalAtDelta(0);
  const curEval = evalAtDelta(userDelta);
  const curve: { delta: number; loss: number }[] = [];
  for (let i = 0; i <= 24; i++) {
    const d = -2.0 + (i / 24) * 4.0;
    curve.push({ delta: d, loss: evalAtDelta(d).loss });
  }

  const trainedWeight = wte[targetTokenId * C + dim] ?? 0;
  return {
    targetTokenId,
    dim,
    trainedWeight,
    currentWeight: trainedWeight + userDelta,
    baseLoss: baseEval.loss,
    currentLoss: curEval.loss,
    gradient: curEval.grad,
    curve,
  };
}
