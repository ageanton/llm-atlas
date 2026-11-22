import { decodeFloat16Buffer } from './fp16';
import type {
  AblationConfig,
  LayerTrace,
  LogitLensEntry,
  ModelManifest,
  TopTokenProb,
  Trace,
  WeightStore,
} from './types';

const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

export function loadWeights(
  manifest: ModelManifest,
  weightsBuffer: ArrayBuffer
): WeightStore {
  const tensors = new Map<string, Float32Array>();
  for (const meta of manifest.tensors) {
    const arr = decodeFloat16Buffer(
      weightsBuffer,
      meta.offset_bytes,
      meta.byte_length
    );
    tensors.set(meta.name, arr);
  }
  return { manifest, tensors };
}

function getTensor(store: WeightStore, name: string): Float32Array {
  const t = store.tensors.get(name);
  if (!t) {
    throw new Error(`Missing required weight tensor: ${name}`);
  }
  return t;
}

function clone2D(mat: number[][]): number[][] {
  return mat.map((row) => row.slice());
}

function zeros2D(rows: number, cols: number): number[][] {
  const out: number[][] = new Array(rows);
  for (let r = 0; r < rows; r++) {
    out[r] = new Array<number>(cols).fill(0);
  }
  return out;
}

function add2D(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0].length;
  const out: number[][] = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) {
      row[c] = a[r][c] + b[r][c];
    }
    out[r] = row;
  }
  return out;
}

function layerNorm(
  x: number[][],
  weight: Float32Array,
  bias: Float32Array,
  eps: number
): number[][] {
  const T = x.length;
  const C = x[0].length;
  const out: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const row = x[t];
    let mean = 0;
    for (let c = 0; c < C; c++) mean += row[c];
    mean /= C;

    let variance = 0;
    for (let c = 0; c < C; c++) {
      const diff = row[c] - mean;
      variance += diff * diff;
    }
    variance /= C;

    const invStd = 1.0 / Math.sqrt(variance + eps);
    const outRow = new Array<number>(C);
    for (let c = 0; c < C; c++) {
      outRow[c] = (row[c] - mean) * invStd * weight[c] + bias[c];
    }
    out[t] = outRow;
  }
  return out;
}

/**
 * Compute x @ W^T where x is [T, inDim] and W is [outDim, inDim] row-major.
 */
function linear(
  x: number[][],
  weight: Float32Array,
  inDim: number,
  outDim: number
): number[][] {
  const T = x.length;
  const out: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const inRow = x[t];
    const outRow = new Array<number>(outDim);
    for (let o = 0; o < outDim; o++) {
      const base = o * inDim;
      let sum = 0;
      for (let i = 0; i < inDim; i++) {
        sum += inRow[i] * weight[base + i];
      }
      outRow[o] = sum;
    }
    out[t] = outRow;
  }
  return out;
}

function geluTanh(v: number): number {
  return (
    0.5 * v * (1.0 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))
  );
}

export function softmax(values: number[]): number[] {
  let maxVal = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > maxVal) maxVal = values[i];
  }
  const exps = new Array<number>(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const e = Math.exp(values[i] - maxVal);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < values.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function topKTokens(probs: number[], k: number): TopTokenProb[] {
  const indexed = probs.map((prob, id) => ({ id, prob }));
  indexed.sort((a, b) => b.prob - a.prob || a.id - b.id);
  return indexed.slice(0, k);
}

function crossEntropyRow(logits: number[], targetId: number): number {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    sumExp += Math.exp(logits[i] - maxVal);
  }
  const logSumExp = maxVal + Math.log(sumExp);
  return logSumExp - logits[targetId];
}

export function runForwardTrace(
  store: WeightStore,
  rawTokenIds: number[],
  ablation?: AblationConfig
): Trace {
  const cfg = store.manifest.config;
  if (rawTokenIds.length === 0) {
    throw new Error('Cannot trace an empty token sequence.');
  }

  const tokenIds = rawTokenIds.slice(0, cfg.max_seq_len);
  const T = tokenIds.length;
  const C = cfg.d_model;
  const H = cfg.n_heads;
  const D = cfg.d_head;
  const dMlp = cfg.d_mlp;
  const eps = cfg.ln_eps;
  const scale = 1.0 / Math.sqrt(D);

  const disabledHeads = new Set(
    (ablation?.disabledHeads ?? []).map(([l, h]) => `${l}:${h}`)
  );
  const disabledAttn = new Set(ablation?.disabledAttnBlocks ?? []);
  const disabledMlp = new Set(ablation?.disabledMlpBlocks ?? []);
  const disabledRes = new Set(
    (ablation?.disabledResiduals ?? []).map(([kind, l]) => `${kind}:${l}`)
  );
  const disabledLn = new Set(
    (ablation?.disabledLayerNorms ?? []).map(([kind, l]) => `${kind}:${l}`)
  );
  const disabledLayers = new Set(ablation?.disabledLayers ?? []);

  const wte = getTensor(store, 'wte.weight');
  const wpe = getTensor(store, 'wpe.weight');
  const lnFW = getTensor(store, 'ln_f.weight');
  const lnFB = getTensor(store, 'ln_f.bias');

  const tokEmb: number[][] = new Array(T);
  const posEmb: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const tid = tokenIds[t];
    const tRow = new Array<number>(C);
    const pRow = new Array<number>(C);
    const tBase = tid * C;
    const pBase = t * C;
    for (let c = 0; c < C; c++) {
      tRow[c] = wte[tBase + c];
      pRow[c] = wpe[pBase + c];
    }
    tokEmb[t] = tRow;
    posEmb[t] = pRow;
  }

  let x = add2D(tokEmb, posEmb);
  const layers: LayerTrace[] = [];
  const logitLens: LogitLensEntry[] = [];

  for (let l = 0; l < cfg.n_layers; l++) {
    if (disabledLayers.has(l)) {
      layers.push({ skipped: true, residualOut: clone2D(x) });
    } else {
      const resBeforeAttn = clone2D(x);
      const ln1Out = disabledLn.has(`ln_1:${l}`)
        ? clone2D(x)
        : layerNorm(
            x,
            getTensor(store, `blocks.${l}.ln_1.weight`),
            getTensor(store, `blocks.${l}.ln_1.bias`),
            eps
          );

      const qFlat = linear(
        ln1Out,
        getTensor(store, `blocks.${l}.attn.W_q.weight`),
        C,
        C
      );
      const kFlat = linear(
        ln1Out,
        getTensor(store, `blocks.${l}.attn.W_k.weight`),
        C,
        C
      );
      const vFlat = linear(
        ln1Out,
        getTensor(store, `blocks.${l}.attn.W_v.weight`),
        C,
        C
      );

      const q: number[][][] = new Array(H);
      const k: number[][][] = new Array(H);
      const v: number[][][] = new Array(H);
      const rawScores: number[][][] = new Array(H);
      const maskedScores: number[][][] = new Array(H);
      const attnWeights: number[][][] = new Array(H);
      const weightedV: number[][][] = new Array(H);

      for (let h = 0; h < H; h++) {
        const hOffset = h * D;
        const qH: number[][] = new Array(T);
        const kH: number[][] = new Array(T);
        const vH: number[][] = new Array(T);
        for (let t = 0; t < T; t++) {
          qH[t] = qFlat[t].slice(hOffset, hOffset + D);
          kH[t] = kFlat[t].slice(hOffset, hOffset + D);
          vH[t] = vFlat[t].slice(hOffset, hOffset + D);
        }
        q[h] = qH;
        k[h] = kH;
        v[h] = vH;

        const rawH: number[][] = new Array(T);
        const maskedH: number[][] = new Array(T);
        const weightsH: number[][] = new Array(T);
        for (let t1 = 0; t1 < T; t1++) {
          const rawRow = new Array<number>(T);
          const maskedRow = new Array<number>(T);
          for (let t2 = 0; t2 < T; t2++) {
            let dot = 0;
            for (let d = 0; d < D; d++) {
              dot += qH[t1][d] * kH[t2][d];
            }
            const s = dot * scale;
            rawRow[t2] = s;
            maskedRow[t2] = t2 > t1 ? -1e9 : s;
          }
          rawH[t1] = rawRow;
          maskedH[t1] = maskedRow;
          weightsH[t1] = softmax(maskedRow);
        }
        rawScores[h] = rawH;
        maskedScores[h] = maskedH;
        attnWeights[h] = weightsH;

        const wvH: number[][] = new Array(T);
        const isHeadDisabled = disabledHeads.has(`${l}:${h}`);
        for (let t1 = 0; t1 < T; t1++) {
          const wvRow = new Array<number>(D).fill(0);
          if (!isHeadDisabled) {
            for (let t2 = 0; t2 < T; t2++) {
              const w = weightsH[t1][t2];
              const vRow = vH[t2];
              for (let d = 0; d < D; d++) {
                wvRow[d] += w * vRow[d];
              }
            }
          }
          wvH[t1] = wvRow;
        }
        weightedV[h] = wvH;
      }

      const concat: number[][] = new Array(T);
      for (let t = 0; t < T; t++) {
        const row = new Array<number>(C);
        for (let h = 0; h < H; h++) {
          const hOffset = h * D;
          for (let d = 0; d < D; d++) {
            row[hOffset + d] = weightedV[h][t][d];
          }
        }
        concat[t] = row;
      }

      const attnOut = disabledAttn.has(l)
        ? zeros2D(T, C)
        : linear(
            concat,
            getTensor(store, `blocks.${l}.attn.W_o.weight`),
            C,
            C
          );

      const resAfterAttn = disabledRes.has(`attn:${l}`)
        ? clone2D(attnOut)
        : add2D(resBeforeAttn, attnOut);

      const ln2Out = disabledLn.has(`ln_2:${l}`)
        ? clone2D(resAfterAttn)
        : layerNorm(
            resAfterAttn,
            getTensor(store, `blocks.${l}.ln_2.weight`),
            getTensor(store, `blocks.${l}.ln_2.bias`),
            eps
          );

      const preAct = linear(
        ln2Out,
        getTensor(store, `blocks.${l}.mlp.W_up.weight`),
        C,
        dMlp
      );
      const hidden: number[][] = new Array(T);
      for (let t = 0; t < T; t++) {
        const hRow = new Array<number>(dMlp);
        for (let m = 0; m < dMlp; m++) {
          hRow[m] = geluTanh(preAct[t][m]);
        }
        hidden[t] = hRow;
      }

      const mlpOut = disabledMlp.has(l)
        ? zeros2D(T, C)
        : linear(
            hidden,
            getTensor(store, `blocks.${l}.mlp.W_down.weight`),
            dMlp,
            C
          );

      const resAfterMlp = disabledRes.has(`mlp:${l}`)
        ? clone2D(mlpOut)
        : add2D(resAfterAttn, mlpOut);

      layers.push({
        skipped: false,
        residualBeforeAttn: resBeforeAttn,
        ln1Out,
        attn: {
          q,
          k,
          v,
          rawScores,
          maskedScores,
          attnWeights,
          weightedV,
          output: attnOut,
        },
        residualAfterAttn: resAfterAttn,
        ln2Out,
        mlp: {
          hidden,
          output: mlpOut,
        },
        residualAfterMlp: resAfterMlp,
      });

      x = resAfterMlp;
    }

    // Logit Lens peek at final token after this block
    const lensNorm = layerNorm([x[T - 1]], lnFW, lnFB, eps);
    const lensLogits = linear(lensNorm, wte, C, cfg.vocab_size)[0];
    const lensProbs = softmax(lensLogits);
    logitLens.push({
      afterLayer: l,
      topTokens: topKTokens(lensProbs, 5),
    });
  }

  const lnFOut = disabledLn.has('ln_f:-1')
    ? clone2D(x)
    : layerNorm(x, lnFW, lnFB, eps);

  const allLogits = linear(lnFOut, wte, C, cfg.vocab_size);
  const finalLogits = allLogits[T - 1];
  const finalProbs = softmax(finalLogits);

  let greedyNextTokenId = 0;
  let bestProb = -1;
  for (let i = 0; i < finalProbs.length; i++) {
    if (finalProbs[i] > bestProb) {
      bestProb = finalProbs[i];
      greedyNextTokenId = i;
    }
  }

  let inputLoss: number | null = null;
  if (T > 1) {
    let totalLoss = 0;
    for (let t = 0; t < T - 1; t++) {
      totalLoss += crossEntropyRow(allLogits[t], tokenIds[t + 1]);
    }
    inputLoss = totalLoss / (T - 1);
  }

  return {
    tokenIds,
    tokEmb,
    posEmb,
    layers,
    logitLens,
    lnFOut,
    finalLogits,
    finalProbs,
    greedyNextTokenId,
    inputLoss,
  };
}
