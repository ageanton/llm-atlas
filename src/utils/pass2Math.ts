import { normalizeText } from '../engine/tokenizer';
import type { TokenizerData } from '../engine/types';

const PRETOKEN_RE = / ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
const UNK_TOKEN = '<|unk|>';

export interface BpeStepSnapshot {
  step: number;
  pieces: string[];
  mergedPair: [string, string] | null;
  mergedResult: string | null;
  mergeRank: number | null;
}

/**
 * Replays our model's real rank-ordered BPE merges one merge rule at a time
 * so students can scrub from raw characters to final subword tokens.
 */
export function traceBpeMerges(
  rawText: string,
  tokData: TokenizerData
): BpeStepSnapshot[] {
  const text = normalizeText(rawText);
  const vocabSet = new Set(tokData.vocab);
  const mergeRanks = new Map<string, number>();
  for (let r = 0; r < tokData.merges.length; r++) {
    const [a, b] = tokData.merges[r];
    mergeRanks.set(`${a}\u0000${b}`, r);
  }

  const groups: string[][] = [];
  for (const match of text.matchAll(PRETOKEN_RE)) {
    const piece = match[0];
    const chars: string[] = [];
    for (let i = 0; i < piece.length; i++) {
      const ch = piece[i];
      chars.push(vocabSet.has(ch) ? ch : UNK_TOKEN);
    }
    groups.push(chars);
  }

  const flatten = (): string[] => groups.flat();
  const snapshots: BpeStepSnapshot[] = [
    {
      step: 0,
      pieces: flatten(),
      mergedPair: null,
      mergedResult: null,
      mergeRank: null,
    },
  ];

  let step = 0;
  while (step < 80) {
    let bestRank: number | null = null;
    let bestA = '';
    let bestB = '';

    for (const grp of groups) {
      for (let i = 0; i < grp.length - 1; i++) {
        const rank = mergeRanks.get(`${grp[i]}\u0000${grp[i + 1]}`);
        if (rank !== undefined && (bestRank === null || rank < bestRank)) {
          bestRank = rank;
          bestA = grp[i];
          bestB = grp[i + 1];
        }
      }
    }

    if (bestRank === null) break;

    for (let g = 0; g < groups.length; g++) {
      const grp = groups[g];
      const next: string[] = [];
      let i = 0;
      while (i < grp.length) {
        if (i < grp.length - 1 && grp[i] === bestA && grp[i + 1] === bestB) {
          next.push(bestA + bestB);
          i += 2;
        } else {
          next.push(grp[i]);
          i += 1;
        }
      }
      groups[g] = next;
    }

    step += 1;
    snapshots.push({
      step,
      pieces: flatten(),
      mergedPair: [bestA, bestB],
      mergedResult: bestA + bestB,
      mergeRank: bestRank,
    });
  }

  return snapshots;
}

export interface PcaPoint {
  id: number;
  text: string;
  x: number;
  y: number;
  z: number;
  norm: number;
}

export interface PcaCloudData {
  points: PcaPoint[];
  explainedVariance: [number, number, number];
}

/**
 * Exact 3D Principal Component Analysis (centered covariance + deterministic
 * power iteration with Hotelling deflation) on our real trained wte [512, 128].
 */
export function computeWtePca3D(
  wte: Float32Array,
  vocab: string[],
  dModel: number
): PcaCloudData {
  const N = Math.min(vocab.length, Math.floor(wte.length / dModel));
  const mean = new Float64Array(dModel);
  for (let i = 0; i < N; i++) {
    const base = i * dModel;
    for (let d = 0; d < dModel; d++) mean[d] += wte[base + d];
  }
  for (let d = 0; d < dModel; d++) mean[d] /= Math.max(1, N);

  const centered = new Float64Array(N * dModel);
  const norms = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const base = i * dModel;
    let sq = 0;
    for (let d = 0; d < dModel; d++) {
      const raw = wte[base + d];
      sq += raw * raw;
      centered[base + d] = raw - mean[d];
    }
    norms[i] = Math.sqrt(sq);
  }

  // Covariance C = (1/N) X^T X [dModel x dModel]
  const cov = new Float64Array(dModel * dModel);
  let totalVar = 0;
  for (let r = 0; r < dModel; r++) {
    for (let c = r; c < dModel; c++) {
      let sum = 0;
      for (let i = 0; i < N; i++) {
        const base = i * dModel;
        sum += centered[base + r] * centered[base + c];
      }
      const val = sum / Math.max(1, N);
      cov[r * dModel + c] = val;
      cov[c * dModel + r] = val;
      if (r === c) totalVar += val;
    }
  }

  const axes: Float64Array[] = [];
  const eigVals: number[] = [];

  for (let comp = 0; comp < 3; comp++) {
    const v = new Float64Array(dModel);
    for (let d = 0; d < dModel; d++) {
      v[d] = Math.sin((d + 1) * (comp + 1.37)) + 0.25 * Math.cos(d * 0.7);
    }
    let vNorm = 0;
    for (let d = 0; d < dModel; d++) vNorm += v[d] * v[d];
    vNorm = Math.sqrt(Math.max(1e-12, vNorm));
    for (let d = 0; d < dModel; d++) v[d] /= vNorm;

    let eig = 0;
    for (let iter = 0; iter < 36; iter++) {
      const next = new Float64Array(dModel);
      for (let r = 0; r < dModel; r++) {
        const rowOff = r * dModel;
        let s = 0;
        for (let c = 0; c < dModel; c++) s += cov[rowOff + c] * v[c];
        next[r] = s;
      }
      let nNorm = 0;
      for (let d = 0; d < dModel; d++) nNorm += next[d] * next[d];
      eig = Math.sqrt(Math.max(1e-12, nNorm));
      for (let d = 0; d < dModel; d++) v[d] = next[d] / eig;
    }

    axes.push(v);
    eigVals.push(eig);

    // Hotelling deflation: C <- C - eig * v v^T
    for (let r = 0; r < dModel; r++) {
      const rowOff = r * dModel;
      const vr = eig * v[r];
      for (let c = 0; c < dModel; c++) {
        cov[rowOff + c] -= vr * v[c];
      }
    }
  }

  const rawPts: { x: number; y: number; z: number }[] = new Array(N);
  let maxCoord = 1e-6;
  for (let i = 0; i < N; i++) {
    const base = i * dModel;
    let x = 0;
    let y = 0;
    let z = 0;
    for (let d = 0; d < dModel; d++) {
      const val = centered[base + d];
      x += val * axes[0][d];
      y += val * axes[1][d];
      z += val * axes[2][d];
    }
    rawPts[i] = { x, y, z };
    maxCoord = Math.max(maxCoord, Math.abs(x), Math.abs(y), Math.abs(z));
  }

  const scale = 3.2 / maxCoord;
  const points: PcaPoint[] = rawPts.map((p, id) => ({
    id,
    text: vocab[id] ?? '',
    x: p.x * scale,
    y: p.y * scale,
    z: p.z * scale,
    norm: norms[id],
  }));

  const denom = Math.max(1e-12, totalVar);
  return {
    points,
    explainedVariance: [
      eigVals[0] / denom,
      eigVals[1] / denom,
      eigVals[2] / denom,
    ],
  };
}

export interface CosineNeighbor {
  id: number;
  text: string;
  cosine: number;
}

export function findCosineNeighbors(
  wte: Float32Array,
  vocab: string[],
  dModel: number,
  targetId: number,
  k = 5
): CosineNeighbor[] {
  const N = Math.min(vocab.length, Math.floor(wte.length / dModel));
  const tBase = targetId * dModel;
  let tSq = 0;
  for (let d = 0; d < dModel; d++) {
    const v = wte[tBase + d];
    tSq += v * v;
  }
  const tNorm = Math.sqrt(Math.max(1e-12, tSq));

  const scored: CosineNeighbor[] = [];
  for (let id = 0; id < N; id++) {
    if (id === targetId || id <= 1) continue;
    const base = id * dModel;
    let dot = 0;
    let sq = 0;
    for (let d = 0; d < dModel; d++) {
      const v = wte[base + d];
      dot += wte[tBase + d] * v;
      sq += v * v;
    }
    const denom = tNorm * Math.sqrt(Math.max(1e-12, sq));
    scored.push({ id, text: vocab[id] ?? '', cosine: dot / denom });
  }

  scored.sort((a, b) => b.cosine - a.cosine);
  return scored.slice(0, k);
}

export interface VectorStats {
  mean: number;
  std: number;
  rms: number;
  min: number;
  max: number;
  l2Norm: number;
}

export function computeVectorStats(vec: number[]): VectorStats {
  const n = Math.max(1, vec.length);
  let sum = 0;
  let sqSum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    sum += v;
    sqSum += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < vec.length; i++) {
    const diff = vec[i] - mean;
    varSum += diff * diff;
  }
  return {
    mean,
    std: Math.sqrt(varSum / n),
    rms: Math.sqrt(sqSum / n),
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
    l2Norm: Math.sqrt(sqSum),
  };
}

export function computeRmsNorm(vec: number[], eps = 1e-5): number[] {
  const n = Math.max(1, vec.length);
  let sq = 0;
  for (let i = 0; i < vec.length; i++) sq += vec[i] * vec[i];
  const invRms = 1 / Math.sqrt(sq / n + eps);
  return vec.map((v) => v * invRms);
}
