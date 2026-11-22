/**
 * Perceptually uniform, colorblind-safe sequential ramp for attention weights [0..1].
 * Maps 0 -> pale cool slate (#f1f5f9), 0.5 -> vivid cobalt (#2563eb), 1.0 -> deep indigo (#172554).
 */
export function sequentialColor(tRaw: number): string {
  const t = Math.max(0, Math.min(1, tRaw));
  const stops: [number, [number, number, number]][] = [
    [0.0, [241, 245, 249]],
    [0.25, [191, 219, 254]],
    [0.55, [59, 130, 246]],
    [0.8, [29, 78, 216]],
    [1.0, [23, 37, 84]],
  ];
  return interpolateStops(t, stops);
}

/**
 * Colorblind-safe diverging ramp for signed weights & activations [-1..1].
 * Negative -> cobalt blue (#1d4ed8), zero -> crisp off-white (#f8fafc), positive -> burnt orange (#c2410c).
 */
export function divergingColor(normRaw: number): string {
  const v = Math.max(-1, Math.min(1, normRaw));
  const t = (v + 1) / 2;
  const stops: [number, [number, number, number]][] = [
    [0.0, [30, 58, 138]],
    [0.25, [96, 165, 250]],
    [0.5, [248, 250, 252]],
    [0.75, [251, 146, 60]],
    [1.0, [154, 52, 18]],
  ];
  return interpolateStops(t, stops);
}

function interpolateStops(
  t: number,
  stops: [number, [number, number, number]][]
): string {
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const span = t1 - t0;
      const ratio = span > 0 ? (t - t0) / span : 0;
      const r = Math.round(c0[0] + ratio * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + ratio * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + ratio * (c1[2] - c0[2]));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  const last = stops[stops.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

export function formatToken(raw: string): string {
  if (raw === '\n') return '\\n';
  if (raw === ' ') return '·';
  return raw.replace(/^ /, '·');
}

export function formatNum(n: number, digits = 3): string {
  if (!Number.isFinite(n)) return '0';
  return n.toFixed(digits);
}

export function vectorL2Norm(vec: number[]): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  return Math.sqrt(sum);
}

export function topIndices(arr: number[] | Float32Array, k = 5): number[] {
  const pairs: { idx: number; val: number }[] = [];
  for (let i = 0; i < arr.length; i++) {
    pairs.push({ idx: i, val: arr[i] });
  }
  pairs.sort((a, b) => b.val - a.val);
  return pairs.slice(0, k).map((p) => p.idx);
}

export interface SampledToken {
  id: number;
  prob: number;
  cumProb: number;
}

export function computeSampledDistribution(
  logits: number[],
  temperature: number,
  topK: number,
  topP: number
): SampledToken[] {
  const t = Math.max(0.05, temperature);
  const scaled = logits.map((l) => l / t);
  const maxScaled = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - maxScaled));
  const sumExp = exps.reduce((acc, v) => acc + v, 0);
  const probs = exps.map((e) => e / (sumExp || 1e-6));

  const sorted = probs
    .map((prob, id) => ({ id, prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, Math.max(1, topK));

  let cum = 0;
  const filtered: { id: number; prob: number }[] = [];
  for (const item of sorted) {
    filtered.push(item);
    cum += item.prob;
    if (cum >= topP && filtered.length >= 1) break;
  }

  const normSum = filtered.reduce((acc, it) => acc + it.prob, 0) || 1e-6;
  let runningCum = 0;
  return filtered.map((it) => {
    const p = it.prob / normSum;
    runningCum += p;
    return {
      id: it.id,
      prob: p,
      cumProb: runningCum,
    };
  });
}

export function sampleFromDistribution(dist: SampledToken[]): number {
  if (dist.length === 0) return 0;
  const r = Math.random();
  for (const item of dist) {
    if (r <= item.cumProb) return item.id;
  }
  return dist[dist.length - 1].id;
}
