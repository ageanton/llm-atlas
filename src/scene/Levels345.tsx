import { useEffect, useMemo, useState } from 'react';
import type { TransformerEngineClient } from '../engine/client';
import type { ModelManifest, TrainingReplayData, TransformerTrace } from '../engine/types';
import type { Register, SelectionTarget } from '../state/urlState';
import { buildParamSelection } from '../state/urlState';
import { divergingColor, formatNum, formatToken, sequentialColor } from '../utils/color';
import { CanvasHeatmap } from './CanvasHeatmap';

export interface Level345Props {
  tokens: string[];
  trace: TransformerTrace;
  target: SelectionTarget;
  register?: Register;
  onSelect: (sel: string, zoomLevel5?: boolean) => void;
  hoveredTokenIdx?: number | null;
  onHoverToken?: (idx: number | null) => void;
  manifest?: ModelManifest;
  client?: TransformerEngineClient;
  replay?: TrainingReplayData | null;
  onOpenReplay?: () => void;
}

export function Level3View({
  tokens,
  trace,
  target,
  onSelect,
  hoveredTokenIdx,
  onHoverToken,
}: Level345Props) {
  const layer = target.layer;
  const head = target.head;
  const T = tokens.length;

  const rawHeadAttn = trace.blocks[layer].attn_weights[head];
  const headAttn = useMemo(() => {
    return rawHeadAttn.slice(0, T).map((row) => row.slice(0, T));
  }, [rawHeadAttn, T]);

  const activeFocusRow = hoveredTokenIdx !== null && hoveredTokenIdx !== undefined && hoveredTokenIdx < T
    ? hoveredTokenIdx
    : T - 1;

  const focusScores = headAttn[activeFocusRow] || [];

  return (
    <div className="scene-view level-3-view">
      <div className="l3-header">
        <div>
          <h2 className="l3-title mono">
            Block {layer} · Head {head} Self-Attention Matrix
          </h2>
          <span className="badge-chip mono">A = softmax(Q·K^T / √32 + CausalMask)</span>
        </div>

        <div className="head-switcher-row mono">
          {[0, 1, 2, 3].map((h) => (
            <button
              key={h}
              type="button"
              className={`seg-btn ${h === head ? 'is-active' : ''}`}
              onClick={() => onSelect(`block${layer}.attn.head${h}.K`)}
            >
              Head {h}
            </button>
          ))}
        </div>
      </div>

      <div className="l3-workspace-grid">
        <div className="l3-matrix-container">
          <div className="matrix-top-tokens">
            <span className="matrix-corner-label mono">Key →</span>
            {tokens.map((tok, c) => (
              <span
                key={c}
                className={`matrix-col-tok mono ${hoveredTokenIdx === c ? 'is-hovered' : ''}`}
                onMouseEnter={() => onHoverToken?.(c)}
                onMouseLeave={() => onHoverToken?.(null)}
              >
                {formatToken(tok)}
              </span>
            ))}
          </div>

          <div className="matrix-body-wrap">
            <div className="matrix-left-tokens">
              {tokens.map((tok, r) => (
                <div
                  key={r}
                  className={`matrix-row-tok mono ${r === activeFocusRow ? 'is-active' : ''}`}
                  onMouseEnter={() => onHoverToken?.(r)}
                  onMouseLeave={() => onHoverToken?.(null)}
                >
                  <span className="r-idx">{r}:</span>
                  {formatToken(tok)}
                </div>
              ))}
            </div>

            <div className="matrix-cells-grid">
              {headAttn.map((row, r) => (
                <div key={r} className="attn-matrix-row">
                  {row.map((val, c) => {
                    const isMasked = c > r;
                    const bg = isMasked ? '#f1f5f9' : sequentialColor(val);
                    const isFocus = r === activeFocusRow;
                    return (
                      <div
                        key={c}
                        className={`attn-matrix-cell mono ${isMasked ? 'is-masked' : ''} ${isFocus ? 'is-focus-row' : ''}`}
                        style={{ backgroundColor: bg }}
                        title={`Row ${r} ("${tokens[r]}") attending to Col ${c} ("${tokens[c]}") = ${formatNum(val, 3)}`}
                        onClick={() => {
                          onSelect(
                            buildParamSelection(
                              `blocks.${layer}.attn.c_attn.weight`,
                              r,
                              c
                            ),
                            true
                          );
                        }}
                      >
                        {!isMasked && T <= 16 && (
                          <span className="cell-num">{formatNum(val, 2)}</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="l3-readout-card">
          <h3 className="mono">
            Token Focus: “{formatToken(tokens[activeFocusRow] ?? '')}” (#{activeFocusRow})
          </h3>
          <p className="card-subtext">
            Attention distribution over earlier words for this token:
          </p>
          <div className="focus-distribution-list">
            {focusScores.map((score, c) => {
              if (c > activeFocusRow) return null;
              const pct = Math.round(score * 100);
              return (
                <div key={c} className="focus-bar-row">
                  <div className="focus-bar-tok mono">
                    <span className="c-idx">c{c}:</span>
                    <span>“{formatToken(tokens[c])}”</span>
                  </div>
                  <div className="focus-bar-track">
                    <div
                      className="focus-bar-fill"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: sequentialColor(score),
                      }}
                    />
                  </div>
                  <span className="focus-bar-pct mono">{pct}%</span>
                </div>
              );
            })}
          </div>

          <div className="l3-qk-routing-box">
            <span className="mono badge-chip">Vector Projections</span>
            <p className="card-subtext">
              Inspect how Key and Query weights transform tokens into relevance scores:
            </p>
            <div className="btn-group-column">
              <button
                type="button"
                className="seg-btn"
                onClick={() =>
                  onSelect(`param:blocks.${layer}.attn.c_attn.weight@0,0`, true)
                }
              >
                Inspect Q/K/V Projection Matrix →
              </button>
              <button
                type="button"
                className="seg-btn"
                onClick={() =>
                  onSelect(`param:blocks.${layer}.attn.c_proj.weight@0,0`, true)
                }
              >
                Inspect Attention Output Projection →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Level4View({
  manifest,
  client,
  target,
  onSelect,
  tokens,
  trace,
}: Level345Props) {
  const [sourceKind, setSourceKind] = useState<'weight' | 'activation'>('weight');
  const [tensorData, setTensorData] = useState<Float32Array | null>(null);
  const [showGrid, setShowGrid] = useState(false);

  const paramMeta = useMemo(() => {
    if (!manifest) return null;
    return (
      manifest.tensors.find((t) => t.name === target.matrix) ||
      manifest.tensors[0]
    );
  }, [manifest, target.matrix]);

  useEffect(() => {
    if (!client || !paramMeta || sourceKind !== 'weight') return;
    client.getTensor(paramMeta.name).then((buf) => {
      setTensorData(buf);
    });
  }, [client, paramMeta, sourceKind]);

  if (!paramMeta || !manifest) {
    return <div className="scene-view level-4-view">Loading tensor metadata...</div>;
  }

  const rows = paramMeta.shape[0];
  const cols = paramMeta.shape[1] || 1;

  const getWeightVal = (r: number, c: number) => {
    if (!tensorData) return 0;
    return tensorData[r * cols + c] || 0;
  };

  const actMatrix = useMemo(() => {
    const layer = target.layer;
    return trace.blocks[layer].resid_post;
  }, [trace, target.layer]);

  const getActVal = (r: number, c: number) => {
    return actMatrix[r]?.[c] || 0;
  };

  const gridRows = Math.min(rows, 10);
  const gridCols = Math.min(cols, 10);

  return (
    <div className="scene-view level-4-view">
      <div className="l4-header">
        <div>
          <h2 className="l4-title mono">{paramMeta.name}</h2>
          <span className="badge-chip mono">
            {paramMeta.shape.join(' × ')} · {paramMeta.count.toLocaleString()} parameters · float16
          </span>
        </div>

        <div className="l4-mode-controls mono">
          <button
            type="button"
            className={`seg-btn ${sourceKind === 'weight' ? 'is-active' : ''}`}
            onClick={() => setSourceKind('weight')}
          >
            Weights Matrix
          </button>
          <button
            type="button"
            className={`seg-btn ${sourceKind === 'activation' ? 'is-active' : ''}`}
            onClick={() => setSourceKind('activation')}
          >
            Runtime Activation Matrix
          </button>
        </div>
      </div>

      <div className="l4-selector-strip mono">
        <span>Select Parameter:</span>
        {sourceKind === 'weight' && (
          <select
            className="select-input"
            value={paramMeta.name}
            onChange={(e) => onSelect(buildParamSelection(e.target.value))}
          >
            {manifest.tensors.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name} [{t.shape.join('×')}]
              </option>
            ))}
          </select>
        )}

        <button
          type="button"
          className={`seg-btn ${showGrid ? 'is-active' : ''}`}
          onClick={() => setShowGrid((v) => !v)}
        >
          {showGrid ? 'Hide numeric grid' : 'Show numeric grid'}
        </button>
      </div>

      {sourceKind === 'weight' ? (
        <>
          <p className="param-role">{paramMeta.role}</p>
          {tensorData ? (
            <CanvasHeatmap
              title={paramMeta.name}
              rows={rows}
              cols={cols}
              getValue={getWeightVal}
              mode="diverging"
              selectedCell={
                target.matrix === paramMeta.name
                  ? { row: target.row, col: target.col }
                  : null
              }
              onCellClick={(r, c) =>
                onSelect(buildParamSelection(paramMeta.name, r, c), true)
              }
              height={260}
            />
          ) : (
            <div className="empty-notice">Loading weight tensor from worker...</div>
          )}
        </>
      ) : (
        <CanvasHeatmap
          title={`Block ${target.layer} Head ${target.head} Key Activations (${tokens.length} tokens × 32)`}
          rows={actMatrix.length}
          cols={actMatrix[0].length}
          getValue={getActVal}
          mode="diverging"
          height={220}
        />
      )}

      {showGrid && (
        <div className="numeric-grid-wrap">
          <div className="numeric-grid-caption mono">
            Showing top-left {gridRows}×{gridCols} real entries (click any weight entry to open Level 5)
          </div>
          <table className="numeric-table mono">
            <thead>
              <tr>
                <th />
                {Array.from({ length: gridCols }, (_, c) => (
                  <th key={c}>c{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: gridRows }, (_, r) => (
                <tr key={r}>
                  <th>r{r}</th>
                  {Array.from({ length: gridCols }, (_, c) => {
                    const val = getWeightVal(r, c);
                    return (
                      <td
                        key={c}
                        onClick={() =>
                          onSelect(buildParamSelection(paramMeta.name, r, c), true)
                        }
                        style={{ backgroundColor: divergingColor(val / 1.5) }}
                      >
                        {formatNum(val, 3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function Level5View({
  target,
  manifest,
  client,
  replay,
  onSelect,
  onOpenReplay,
}: Level345Props) {
  const [scalarVal, setScalarVal] = useState<number | null>(null);

  const paramMeta = useMemo(() => {
    if (!manifest) return null;
    return (
      manifest.tensors.find((t) => t.name === target.matrix) ||
      manifest.tensors[0]
    );
  }, [manifest, target.matrix]);

  useEffect(() => {
    if (!client || !paramMeta) return;
    client.getTensor(paramMeta.name).then((buf) => {
      const cols = paramMeta.shape[1] || 1;
      const idx = target.row * cols + target.col;
      setScalarVal(buf[idx] ?? 0);
    });
  }, [client, paramMeta, target.row, target.col]);

  if (!paramMeta) {
    return <div className="scene-view level-5-view">Loading scalar coordinate...</div>;
  }

  const cols = paramMeta.shape[1] || 1;
  const flatIdx = target.row * cols + target.col;

  const trajectoryKey = `${paramMeta.name}@${target.row},${target.col}`;
  const trajectory = replay?.trajectories?.[trajectoryKey] ?? null;

  return (
    <div className="scene-view level-5-view">
      <div className="l5-header">
        <div>
          <span className="badge-chip mono">Single Scalar Coordinate</span>
          <h2 className="l5-title mono">
            {paramMeta.name}[{target.row}, {target.col}]
          </h2>
          <span className="param-role">{paramMeta.role}</span>
        </div>

        <div className="scalar-hero-badge mono">
          <span className="scalar-label">Real Float16 Value</span>
          <span className="scalar-number">
            {scalarVal !== null ? formatNum(scalarVal, 5) : '...'}
          </span>
        </div>
      </div>

      <div className="l5-cards-grid">
        <div className="l5-card">
          <h3>Coordinate metadata</h3>
          <dl className="mono metadata-dl">
            <dt>Tensor:</dt>
            <dd>{paramMeta.name}</dd>
            <dt>Row (Input Dim):</dt>
            <dd>{target.row} / {paramMeta.shape[0]}</dd>
            <dt>Col (Output Dim):</dt>
            <dd>{target.col} / {cols}</dd>
            <dt>Flat Buffer Offset:</dt>
            <dd>{flatIdx.toLocaleString()} (byte offset {flatIdx * 2})</dd>
            <dt>Storage Type:</dt>
            <dd>IEEE 754 float16 (2 bytes)</dd>
          </dl>
          <div className="hist-axis mono">
            <span>min {formatNum(paramMeta.stats.min, 3)}</span>
            <span>mean {formatNum(paramMeta.stats.mean, 3)}</span>
            <span>max {formatNum(paramMeta.stats.max, 3)}</span>
          </div>
        </div>

        <div className="l5-card">
          <h3>Recorded training history</h3>
          {trajectory ? (
            <ScalarTrajectoryChart points={trajectory} />
          ) : (
            <div className="untracked-note">
              <p>
                This coordinate was trained identically by gradient descent, though only 12 coordinates were logged across every checkpoint to keep files small.
              </p>
              <button
                type="button"
                className="btn-inline"
                onClick={() =>
                  onSelect(buildParamSelection('blocks.0.attn.c_attn.weight', 0, 0), true)
                }
              >
                Jump to logged coordinate (Block 0 c_attn [0,0]) →
              </button>
            </div>
          )}

          {onOpenReplay && (
            <button
              type="button"
              className="seg-btn full-width"
              onClick={onOpenReplay}
            >
              Open Full Training Replay (1,500 Steps) →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ScalarTrajectoryChart({ points }: { points: number[] }) {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1e-5;

  const w = 400;
  const h = 120;
  const pad = 12;

  const path = points
    .map((pt, i) => {
      const x = pad + (i / (points.length - 1)) * (w - pad * 2);
      const y = h - pad - ((pt - min) / range) * (h - pad * 2);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <div className="trajectory-chart-wrap">
      <svg viewBox={`0 0 ${w} ${h}`} className="trajectory-svg">
        <line
          x1={pad}
          y1={h / 2}
          x2={w - pad}
          y2={h / 2}
          stroke="#e2e8f0"
          strokeDasharray="4 4"
        />
        <path d={path} fill="none" stroke="#2563eb" strokeWidth="2.5" />
      </svg>
      <div className="trajectory-caption mono">
        <span>Step 0: {formatNum(points[0], 4)}</span>
        <span>Final Step 1,500: {formatNum(points[points.length - 1], 4)}</span>
      </div>
    </div>
  );
}
