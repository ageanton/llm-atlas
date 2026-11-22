import { useEffect, useMemo, useRef, useState } from 'react';
import { divergingColor, formatNum, sequentialColor } from '../utils/color';

export interface CanvasHeatmapProps {
  rows: number;
  cols: number;
  getValue: (r: number, c: number) => number;
  mode?: 'diverging' | 'sequential';
  title?: string;
  selectedCell?: { row: number; col: number } | null;
  onCellClick?: (row: number, col: number) => void;
  height?: number;
}

export function CanvasHeatmap({
  rows,
  cols,
  getValue,
  mode = 'sequential',
  title,
  selectedCell,
  onCellClick,
  height = 240,
}: CanvasHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    row: number;
    col: number;
    val: number;
  } | null>(null);

  const { minVal, maxVal, maxAbs } = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = getValue(r, c);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    return {
      minVal: mn === Infinity ? 0 : mn,
      maxVal: mx === -Infinity ? 1 : mx,
      maxAbs: Math.max(Math.abs(mn), Math.abs(mx), 1e-6),
    };
  }, [rows, cols, getValue]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, width, h);

    const cellW = width / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = getValue(r, c);
        if (mode === 'sequential') {
          const t = maxVal === minVal ? 0 : (val - minVal) / (maxVal - minVal);
          ctx.fillStyle = sequentialColor(t);
        } else {
          const norm = val / maxAbs;
          ctx.fillStyle = divergingColor(norm);
        }
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    if (selectedCell && selectedCell.row < rows && selectedCell.col < cols) {
      ctx.strokeStyle = '#c2410c';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        selectedCell.col * cellW,
        selectedCell.row * cellH,
        cellW,
        cellH
      );
    }
  }, [rows, cols, getValue, mode, minVal, maxVal, maxAbs, selectedCell]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor((x / rect.width) * cols);
    const row = Math.floor((y / rect.height) * rows);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      setHoverInfo({ row, col, val: getValue(row, col) });
    }
  };

  const handleMouseLeave = () => setHoverInfo(null);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCellClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor((x / rect.width) * cols);
    const row = Math.floor((y / rect.height) * rows);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      onCellClick(row, col);
    }
  };

  return (
    <div className="canvas-heatmap-wrap">
      {title && <div className="heatmap-title mono">{title}</div>}
      <canvas
        ref={canvasRef}
        width={cols > 120 ? cols : 480}
        height={height}
        className="canvas-heatmap"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      <div className="heatmap-footer mono">
        <span>
          shape [{rows} × {cols}]
        </span>
        {hoverInfo ? (
          <span>
            [{hoverInfo.row}, {hoverInfo.col}] = {formatNum(hoverInfo.val, 4)}
          </span>
        ) : (
          <span>
            range [{formatNum(minVal, 3)} .. {formatNum(maxVal, 3)}]
          </span>
        )}
      </div>
    </div>
  );
}
