import * as THREE from 'three';

export function sequentialColor(tRaw: number): string {
  const t = Math.max(0, Math.min(1, tRaw));
  const stops: [number, [number, number, number]][] = [
    [0.0, [254, 250, 243]],
    [0.2, [253, 230, 138]],
    [0.45, [245, 158, 11]],
    [0.75, [217, 70, 239]],
    [1.0, [67, 24, 255]],
  ];
  return interpolateStops(t, stops);
}

export function divergingColor(val: number, maxAbs: number): string {
  if (maxAbs <= 1e-9) return 'rgb(240, 243, 246)';
  const t = Math.max(-1, Math.min(1, val / maxAbs));
  if (t < 0) {
    const k = -t;
    const r = Math.round(247 - 40 * k);
    const g = Math.round(248 - 140 * k);
    const b = Math.round(250 - 200 * k);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const k = t;
  const r = Math.round(247 + 0 * k);
  const g = Math.round(248 - 150 * k);
  const b = Math.round(250 - 150 * k);
  return `rgb(${r}, ${g}, ${b})`;
}

export function threeSequentialColor(tRaw: number): THREE.Color {
  const css = sequentialColor(tRaw);
  return new THREE.Color(css);
}

export function threeDivergingColor(val: number, maxAbs: number): THREE.Color {
  const css = divergingColor(val, maxAbs);
  return new THREE.Color(css);
}

function interpolateStops(
  t: number,
  stops: [number, [number, number, number]][]
): string {
  if (t <= stops[0][0]) {
    const c = stops[0][1];
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (t >= p0 && t <= p1) {
      const u = (t - p0) / Math.max(1e-6, p1 - p0);
      const r = Math.round(c0[0] + u * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + u * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + u * (c1[2] - c0[2]));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  const last = stops[stops.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

export function formatToken(tok: string): string {
  return tok
    .replace(/\n/g, '↵')
    .replace(/\t/g, '⇥')
    .replace(/ /g, '·');
}

export function formatTokenExplicit(tok: string): string {
  if (tok === ' ') return '␣ (space)';
  if (tok === '\n') return '↵ (newline)';
  if (tok.startsWith(' ')) return `·${tok.slice(1)}`;
  return tok;
}

export function formatNum(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  if (Math.abs(n) < 1e-4 && n !== 0) return n.toExponential(2);
  return n.toFixed(digits);
}

export function computeDotProductBreakdown(
  q: number[],
  k: number[]
): {
  products: number[];
  sum: number;
  scaled: number;
} {
  const d = Math.min(q.length, k.length);
  const products: number[] = new Array(d);
  let sum = 0;
  for (let i = 0; i < d; i++) {
    const p = (q[i] ?? 0) * (k[i] ?? 0);
    products[i] = p;
    sum += p;
  }
  const scaled = sum / Math.sqrt(d || 1);
  return { products, sum, scaled };
}

export function makeTextSprite(
  text: string,
  options?: {
    fontSize?: number;
    fontFamily?: string;
    fontColor?: string;
    bgColor?: string;
    borderColor?: string;
    borderWidth?: number;
    padding?: number;
    scale?: number;
  }
): THREE.Sprite {
  const fontSize = options?.fontSize ?? 26;
  const fontFamily =
    options?.fontFamily ??
    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  const fontColor = options?.fontColor ?? '#1a2233';
  const bgColor = options?.bgColor ?? 'rgba(255, 255, 255, 0.94)';
  const borderColor = options?.borderColor ?? 'rgba(209, 217, 226, 0.9)';
  const borderWidth = options?.borderWidth ?? 2;
  const padding = options?.padding ?? 14;
  const scale = options?.scale ?? 1.0;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Sprite();

  const font = `600 ${fontSize}px ${fontFamily}`;
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const textWidth = Math.ceil(metrics.width);
  const textHeight = fontSize * 1.25;

  canvas.width = Math.max(32, textWidth + padding * 2);
  canvas.height = Math.max(32, Math.ceil(textHeight + padding * 2));

  ctx.font = font;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';

  if (bgColor && bgColor !== 'transparent') {
    ctx.fillStyle = bgColor;
    ctx.beginPath();
    const r = 8;
    const x = borderWidth / 2;
    const y = borderWidth / 2;
    const w = canvas.width - borderWidth;
    const h = canvas.height - borderWidth;
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y + x, x, y, r);
    ctx.closePath();
    ctx.fill();

    if (borderWidth > 0 && borderColor) {
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = borderWidth;
      ctx.stroke();
    }
  }

  ctx.fillStyle = fontColor;
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });

  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  const baseH = (canvas.height / 100) * scale;
  sprite.scale.set(baseH * aspect, baseH, 1);
  return sprite;
}

export function makeMatrixCanvasTexture(
  matrix: number[][],
  options?: {
    isPreSoftmax?: boolean;
    cellSize?: number;
    maxAbs?: number;
  }
): THREE.CanvasTexture {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const cellSize = options?.cellSize ?? 16;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, cols * cellSize);
  canvas.height = Math.max(1, rows * cellSize);
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.CanvasTexture(canvas);

  let maxAbs = options?.maxAbs ?? 1e-6;
  if (!options?.maxAbs && options?.isPreSoftmax) {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = matrix[i][j];
        if (Number.isFinite(v) && Math.abs(v) > maxAbs) {
          maxAbs = Math.abs(v);
        }
      }
    }
  }

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = matrix[i][j];
      const x = j * cellSize;
      const y = i * cellSize;

      if (!Number.isFinite(v) || v <= -9000) {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(x, y, cellSize, cellSize);
        continue;
      }

      ctx.fillStyle = options?.isPreSoftmax
        ? divergingColor(v, maxAbs)
        : sequentialColor(v);
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
