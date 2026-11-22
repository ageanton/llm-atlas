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
