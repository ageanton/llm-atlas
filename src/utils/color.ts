export function sequentialColor(tRaw: number): string {
  const t = Math.max(0, Math.min(1, tRaw));
  const stops: [number, [number, number, number]][] = [
    [0.0, [242, 245, 250]],
    [0.25, [191, 219, 254]],
    [0.55, [59, 130, 246]],
    [0.82, [29, 78, 216]],
    [1.0, [23, 37, 84]],
  ];
  return interpolateStops(t, stops);
}

export function divergingColor(normRaw: number): string {
  const v = Math.max(-1, Math.min(1, normRaw));
  const t = (v + 1) / 2;
  const stops: [number, [number, number, number]][] = [
    [0.0, [30, 58, 138]],
    [0.25, [96, 165, 250]],
    [0.5, [248, 250, 252]],
    [0.75, [251, 146, 60]],
    [1.0, [180, 83, 9]],
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
      const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  const last = stops[stops.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

export function formatToken(raw: string): string {
  if (raw === '\n') return '↵';
  if (raw === ' ') return '␣';
  return raw.replace(/^ /, '');
}

export function formatTokenExplicit(raw: string): string {
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
