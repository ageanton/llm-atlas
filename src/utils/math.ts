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

export interface DotProductBreakdown {
  dim: number;
  products: number[];
  rawDot: number;
  scaleDivisor: number;
  scaledScore: number;
  positiveVotes: number;
  negativeVotes: number;
}

/**
 * Compute exact element-by-element Q · K / sqrt(d_head) breakdown for Beat 2.
 */
export function computeDotProductBreakdown(
  qVec: number[],
  kVec: number[]
): DotProductBreakdown {
  const dim = Math.min(qVec.length, kVec.length);
  const products = new Array<number>(dim);
  let rawDot = 0;
  let positiveVotes = 0;
  let negativeVotes = 0;

  for (let i = 0; i < dim; i++) {
    const p = qVec[i] * kVec[i];
    products[i] = p;
    rawDot += p;
    if (p >= 0) positiveVotes += p;
    else negativeVotes += p;
  }

  const scaleDivisor = Math.sqrt(Math.max(1, dim));
  return {
    dim,
    products,
    rawDot,
    scaleDivisor,
    scaledScore: rawDot / scaleDivisor,
    positiveVotes,
    negativeVotes,
  };
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
        const norm = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0;
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

function drawRoundRect(
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

export function createLabelSpriteTexture(
  primary: string,
  secondary?: string,
  accent = '#1d4ed8',
  muted = false
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 440;
  canvas.height = secondary ? 128 : 88;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawRoundRect(ctx, 8, 8, canvas.width - 16, canvas.height - 16, 20);
  ctx.fillStyle = muted
    ? 'rgba(241, 245, 249, 0.94)'
    : 'rgba(255, 255, 255, 0.96)';
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = muted ? '#cbd5e1' : accent;
  ctx.stroke();

  ctx.fillStyle = muted ? '#64748b' : '#111827';
  ctx.font =
    '700 30px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(primary.slice(0, 26), canvas.width / 2, secondary ? 46 : 44);

  if (secondary) {
    ctx.fillStyle = muted ? '#b45309' : accent;
    ctx.font = '600 22px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(secondary.slice(0, 32), canvas.width / 2, 90);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
