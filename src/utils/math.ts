import * as THREE from 'three';

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

export function createTensorTexture(
  matrix: number[][],
  mode: 'diverging' | 'sequential' = 'diverging',
  transpose = false
): THREE.CanvasTexture {
  const srcRows = matrix.length;
  const srcCols = matrix[0]?.length ?? 1;
  const rows = transpose ? srcCols : srcRows;
  const cols = transpose ? srcRows : srcCols;

  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.min(cols, 512));
  canvas.height = Math.max(1, Math.min(rows, 512));
  const ctx = canvas.getContext('2d')!;

  let maxAbs = 1e-6;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let r = 0; r < srcRows; r++) {
    const row = matrix[r];
    for (let c = 0; c < srcCols; c++) {
      const v = row[c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
    }
  }
  if (!Number.isFinite(minVal)) minVal = 0;
  if (!Number.isFinite(maxVal)) maxVal = 1;

  const w = canvas.width;
  const h = canvas.height;
  for (let py = 0; py < h; py++) {
    const r = Math.min(rows - 1, Math.floor((py / h) * rows));
    for (let px = 0; px < w; px++) {
      const c = Math.min(cols - 1, Math.floor((px / w) * cols));
      const val = transpose ? matrix[c][r] : matrix[r][c];
      if (mode === 'sequential') {
        const norm =
          maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0;
        ctx.fillStyle = sequentialColor(norm);
      } else {
        ctx.fillStyle = divergingColor(val / maxAbs);
      }
      ctx.fillRect(px, py, 1, 1);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  return tex;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
): void {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

export function createPlaqueTexture(
  title: string,
  subtitle?: string,
  accent = '#1d4ed8',
  muted = false
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 420;
  canvas.height = subtitle ? 124 : 84;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  roundRect(ctx, 8, 8, canvas.width - 16, canvas.height - 16, 18);
  ctx.fillStyle = muted
    ? 'rgba(241, 245, 249, 0.94)'
    : 'rgba(255, 255, 255, 0.95)';
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = muted ? '#cbd5e1' : accent;
  ctx.stroke();

  ctx.fillStyle = muted ? '#64748b' : '#141923';
  ctx.font =
    '700 28px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(title.slice(0, 24), canvas.width / 2, subtitle ? 44 : 42);

  if (subtitle) {
    ctx.fillStyle = muted ? '#b45309' : accent;
    ctx.font = '600 21px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(subtitle.slice(0, 30), canvas.width / 2, 86);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
