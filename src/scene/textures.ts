import * as THREE from 'three';

function drawRoundedRect(
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

export function createPorcelainTileTexture(
  label: string,
  sublabel: string,
  highlighted: boolean
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 160;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = highlighted ? '#eff6ff' : '#fcfbf8';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawRoundedRect(ctx, 8, 8, 240, 144, 24);
  ctx.fillStyle = highlighted ? '#e0edff' : '#ffffff';
  ctx.fill();
  ctx.lineWidth = highlighted ? 6 : 3;
  ctx.strokeStyle = highlighted ? '#2563eb' : '#d6d3cb';
  ctx.stroke();

  ctx.fillStyle = highlighted ? '#1d4ed8' : '#1c2434';
  const fontSize = label.length > 6 ? 44 : 58;
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label.slice(0, 9), 128, 72);

  if (sublabel) {
    ctx.fillStyle = '#64748b';
    ctx.font = '500 22px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(sublabel, 128, 124);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export function createThoughtBubbleTexture(
  title: string,
  tokenText: string,
  pctText: string,
  accentHex = '#2563eb'
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 170;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawRoundedRect(ctx, 14, 12, 292, 126, 32);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
  ctx.stroke();

  // Little speech pointer dot
  ctx.beginPath();
  ctx.arc(160, 152, 8, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#64748b';
  ctx.font = '600 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(title, 160, 42);

  ctx.fillStyle = '#141923';
  ctx.font = '700 44px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.fillText(`“${tokenText.slice(0, 8)}”`, 160, 88);

  ctx.fillStyle = accentHex;
  ctx.font = '700 24px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(pctText, 160, 122);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createPillLabelTexture(
  primary: string,
  secondary?: string,
  accentHex = '#1d4ed8',
  dimmed = false
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 112;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawRoundedRect(ctx, 10, 10, 300, 92, 46);
  ctx.fillStyle = dimmed
    ? 'rgba(241, 245, 249, 0.85)'
    : 'rgba(255, 255, 255, 0.94)';
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = dimmed ? 'rgba(180, 83, 9, 0.55)' : accentHex;
  ctx.stroke();

  ctx.fillStyle = dimmed ? '#9a3412' : '#141923';
  ctx.font = '700 32px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(primary.slice(0, 14), 160, secondary ? 44 : 56);

  if (secondary) {
    ctx.fillStyle = dimmed ? '#b45309' : accentHex;
    ctx.font = '600 22px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(secondary.slice(0, 18), 160, 78);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
