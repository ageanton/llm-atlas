import { useMemo, useState } from 'react';
import type { ActiveLayerTrace, EncodedToken, Trace } from '../engine/types';
import {
  divergingColor,
  formatNum,
  formatTokenExplicit,
  sequentialColor,
} from '../utils/color';

interface NumbersOverlayProps {
  tokens: EncodedToken[];
  trace: Trace;
  layer: number;
  head: number;
  hoveredTokenIdx: number | null;
  onHoverToken: (idx: number | null) => void;
  onClose: () => void;
}

export function NumbersOverlay({
  tokens,
  trace,
  layer,
  head,
  hoveredTokenIdx,
  onHoverToken,
  onClose,
}: NumbersOverlayProps) {
  const [showRaw, setShowRaw] = useState(false);
  const layerTrace = trace.layers[layer];

  const { maxAbsScore } = useMemo(() => {
    if (layerTrace.skipped) return { maxAbsScore: 1 };
    const scores = layerTrace.attn.rawScores[head];
    let mx = 1e-6;
    for (let r = 0; r < tokens.length; r++) {
      for (let c = 0; c <= r; c++) {
        mx = Math.max(mx, Math.abs(scores[r][c]));
      }
    }
    return { maxAbsScore: mx };
  }, [layerTrace, head, tokens.length]);

  if (layerTrace.skipped) return null;
  const active: ActiveLayerTrace = layerTrace;
  const matrix = showRaw
    ? active.attn.rawScores[head]
    : active.attn.attnWeights[head];

  return (
    <div className="numbers-drawer glass-panel" role="dialog" aria-label="Exact attention numbers">
      <div className="numbers-head">
        <div>
          <strong>
            Block {layer} · Head {head} exact matrix
          </strong>
          <span className="mono muted-inline">
            {' '}
            [{tokens.length} × {tokens.length}]
          </span>
        </div>
        <div className="numbers-actions">
          <button
            type="button"
            className={`pill-btn ${showRaw ? 'is-active' : ''}`}
            onClick={() => setShowRaw((v) => !v)}
          >
            {showRaw ? 'Pre-softmax QKᵀ/√32' : 'Post-softmax weights'}
          </button>
          <button type="button" className="pill-btn" onClick={onClose}>
            Fold back to ribbons
          </button>
        </div>
      </div>

      <div className="numbers-scroll">
        <table className="numbers-table mono">
          <thead>
            <tr>
              <th>Q \ K</th>
              {tokens.map((tok, c) => (
                <th
                  key={`c-${c}`}
                  className={hoveredTokenIdx === c ? 'is-cross' : ''}
                  onMouseEnter={() => onHoverToken(c)}
                  onMouseLeave={() => onHoverToken(null)}
                >
                  {formatTokenExplicit(tok.text)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((rowTok, r) => (
              <tr key={`r-${r}`}>
                <th
                  className={hoveredTokenIdx === r ? 'is-cross' : ''}
                  onMouseEnter={() => onHoverToken(r)}
                  onMouseLeave={() => onHoverToken(null)}
                >
                  {formatTokenExplicit(rowTok.text)}
                </th>
                {tokens.map((_, c) => {
                  const masked = c > r;
                  const val = matrix[r][c];
                  const bg = masked
                    ? '#f1f5f9'
                    : showRaw
                      ? divergingColor(val / maxAbsScore)
                      : sequentialColor(val);
                  const dark = !masked && !showRaw && val > 0.45;
                  return (
                    <td
                      key={`cell-${r}-${c}`}
                      style={{
                        backgroundColor: bg,
                        color: dark ? '#ffffff' : '#141923',
                      }}
                      onMouseEnter={() => !masked && onHoverToken(c)}
                      onMouseLeave={() => onHoverToken(null)}
                    >
                      {masked
                        ? ''
                        : showRaw
                          ? formatNum(val, 2)
                          : (val * 100).toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
