import * as THREE from 'three';

export function formatToken(tok: string): string {
  return tok
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/ /g, '·');
}

export function sequentialColor(tRaw: number): string {
  const t = Math.max(0, Math.min(1, tRaw));
  const r = Math.round(245 - 200 * t);
  const g = Math.round(247 - 140 * t);
  const b = Math.round(250 - 40 * t);
  return `rgb(${r}, ${g}, ${b})`;
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
  const t = Math.max(0, Math.min(1, tRaw));
  const r = (245 - 200 * t) / 255;
  const g = (247 - 140 * t) / 255;
  const b = (250 - 40 * t) / 255;
  return new THREE.Color(r, g, b);
}

export function threeDivergingColor(val: number, maxAbs: number): THREE.Color {
  if (maxAbs <= 1e-9) return new THREE.Color(0xf0f3f6);
  const t = Math.max(-1, Math.min(1, val / maxAbs));
  if (t < 0) {
    const k = -t;
    return new THREE.Color(
      (247 - 40 * k) / 255,
      (248 - 140 * k) / 255,
      (250 - 200 * k) / 255
    );
  }
  const k = t;
  return new THREE.Color(
    (247 + 0 * k) / 255,
    (248 - 150 * k) / 255,
    (250 - 150 * k) / 255
  );
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
  const bgColor = options?.bgColor ?? 'rgba(255, 255, 255, 0.92)';
  const borderColor = options?.borderColor ?? 'rgba(209, 217, 226, 0.8)';
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
    ctx.arcTo(x, y, x + w, y, r);
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

export function drawAttentionHeatmap(
  canvas: HTMLCanvasElement,
  scores: number[][],
  options?: {
    labels?: string[];
    queryHighlight?: number | null;
    keyHighlight?: number | null;
    isPreSoftmax?: boolean;
    cellSize?: number;
  }
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const N = scores.length;
  if (N === 0) return;

  const cellSize = options?.cellSize ?? 22;
  const marginL = 50;
  const marginT = 50;
  const marginR = 16;
  const marginB = 16;

  const w = marginL + N * cellSize + marginR;
  const h = marginT + N * cellSize + marginB;

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  let maxAbs = 1e-6;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      const v = scores[i][j];
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }

  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = marginL + j * cellSize;
      const y = marginT + i * cellSize;

      if (j > i) {
        ctx.fillStyle = '#f0f3f6';
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
        ctx.fillStyle = '#c5d0dc';
        ctx.fillText('—', x + cellSize / 2, y + cellSize / 2);
        continue;
      }

      const val = scores[i][j];
      ctx.fillStyle = options?.isPreSoftmax
        ? divergingColor(val, maxAbs)
        : sequentialColor(val);
      ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

      if (
        (options?.queryHighlight === i && options?.keyHighlight === j) ||
        options?.queryHighlight === i
      ) {
        ctx.strokeStyle = '#1d4ed8';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 2, cellSize - 2);
      }
    }
  }

  ctx.fillStyle = '#334155';
  ctx.font = '10px ui-monospace, monospace';
  const labels = options?.labels ?? [];
  for (let k = 0; k < N; k++) {
    const raw = labels[k] ?? `${k}`;
    const clean = formatToken(raw).slice(0, 4);

    ctx.save();
    ctx.translate(marginL + k * cellSize + cellSize / 2, marginT - 10);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'left';
    ctx.fillText(clean, 0, 0);
    ctx.restore();

    ctx.textAlign = 'right';
    ctx.fillText(clean, marginL - 6, marginT + k * cellSize + cellSize / 2);
  }
}
