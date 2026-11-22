import { useEffect, useMemo, useState } from 'react';
import type { TransformerEngineClient } from '../engine/client';
import type {
  ActiveLayerTrace,
  EncodedToken,
  ModelManifest,
  Trace,
} from '../engine/types';
import type { TrainingReplayData } from '../panels/TeachingModes';
import {
  buildParamSelection,
  type Register,
  type SelectionTarget,
} from '../state/urlState';
import {
  divergingColor,
  formatNum,
  formatToken,
  sequentialColor,
} from '../utils/color';
import { CanvasHeatmap } from './CanvasHeatmap';

interface Level3Props {
  tokens: EncodedToken[];
  trace: Trace;
  register: Register;
  target: SelectionTarget;
  hoveredTokenIdx: number | null;
  onHoverToken: (idx: number | null) => void;
  onSelect: (sel: string, zoomIn?: boolean) => void;
}

export function Level3View({
  tokens,
  trace,
  register,
  target,
  hoveredTokenIdx,
  onHoverToken,
  onSelect,
}: Level3Props) {
  const [showPreSoftmax, setShowPreSoftmax] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{
    q: number;
    k: number;
  } | null>(null);

  const layerIdx = target.layer;
  const headIdx = target.head;
  const layerTrace = trace.layers[layerIdx];

  if (layerTrace.skipped) {
    return (
      <div className="scene-stage">
        <div className="empty-notice">
          Block {layerIdx} is disabled in Ablation Mode. Switch to another block or re-enable Block {layerIdx} to inspect its attention weights.
        </div>
      </div>
    );
  }

  const activeLayer = layerTrace as ActiveLayerTrace;
  const seqLen = tokens.length;
  const weights = activeLayer.attn.attnWeights[headIdx] ?? [];
  const rawScores = activeLayer.attn.rawScores[headIdx] ?? [];

  const focusRow = hoveredTokenIdx !== null && hoveredTokenIdx < seqLen
    ? hoveredTokenIdx
    : seqLen - 1;

  const rowWeights = weights[focusRow] ?? [];

  return (
    <div className="scene-stage level3-stage">
      <div className="l3-header">
        <div>
          <h2>
            Block {layerIdx} · Head {headIdx} Attention Weights
          </h2>
          <span className="mono l3-meta">
            A = softmax(Q·Kᵀ / √32 + CausalMask) · causal lower-triangular matrix
          </span>
        </div>

        <div className="l3-actions">
          <button
            type="button"
            className={`seg-btn mono ${showPreSoftmax ? 'is-active' : ''}`}
            onClick={() => setShowPreSoftmax((v) => !v)}
          >
            {showPreSoftmax ? 'Showing Raw Q·Kᵀ scores' : 'Showing Softmax probabilities [0..1]'}
          </button>
        </div>
      </div>

      <div className="l3-matrix-layout">
        <div className="l3-matrix-card">
          <div className="matrix-col-headers" style={{ marginLeft: 96 }}>
            {tokens.map((tok, c) => (
              <span
                key={`${tok.start}-${c}`}
                className={`matrix-tok-col mono ${hoveredTokenIdx === c ? 'is-active' : ''}`}
                onMouseEnter={() => onHoverToken(c)}
                onMouseLeave={() => onHoverToken(null)}
              >
                {formatToken(tok.text)}
              </span>
            ))}
          </div>

          <div className="matrix-rows-wrap">
            <div className="matrix-row-headers">
              {tokens.map((tok, r) => (
                <span
                  key={`${tok.start}-${r}`}
                  className={`matrix-tok-row mono ${focusRow === r ? 'is-active' : ''}`}
                  onMouseEnter={() => onHoverToken(r)}
                  onMouseLeave={() => onHoverToken(null)}
                >
                  <span className="r-idx">{r}:</span> {formatToken(tok.text)}
                </span>
              ))}
            </div>

            <div className="matrix-cells-body">
              {tokens.map((_, r) => (
                <div key={r} className="heatmap-row">
                  {tokens.map((_, c) => {
                    const isMasked = c > r;
                    const val = showPreSoftmax
                      ? rawScores[r]?.[c] ?? 0
                      : weights[r]?.[c] ?? 0;
                    const bg = isMasked
                      ? '#f1f5f9'
                      : showPreSoftmax
                        ? divergingColor(val / 3.0)
                        : sequentialColor(val);

                    return (
                      <button
                        key={c}
                        type="button"
                        className={`heatmap-cell ${isMasked ? 'is-masked' : ''} ${focusRow === r ? 'is-focus-row' : ''}`}
                        style={{ backgroundColor: bg }}
                        onMouseEnter={() => setHoveredCell({ q: r, k: c })}
                        onMouseLeave={() => setHoveredCell(null)}
                        onClick={() => {
                          onSelect(
                            buildParamSelection(
                              `blocks.${layerIdx}.attn.c_attn.weight`,
                              r,
                              c
                            ),
                            true
                          );
                        }}
                        aria-label={`Attention from token ${r} to token ${c}: ${val.toFixed(3)}`}
                      >
                        {!isMasked && seqLen <= 14 && (
                          <span className="cell-text mono">{val.toFixed(2)}</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="matrix-caption mono">
            {hoveredCell ? (
              <span>
                Row {hoveredCell.q} (“{tokens[hoveredCell.q]?.text}”) attending to Col {hoveredCell.k} (“{tokens[hoveredCell.k]?.text}”) ={' '}
                {weights[hoveredCell.q]?.[hoveredCell.k]?.toFixed(4)}
              </span>
            ) : (
              <span>Hover any cell to read precise attention weight. Click to inspect underlying weight coordinate.</span>
            )}
          </div>
        </div>

        <div className="l3-focus-card">
          <div className="focus-header">
            <h3>Attention Distribution for “{tokens[focusRow]?.text}”</h3>
            <span className="mono focus-sub">Row {focusRow} distribution</span>
          </div>

          <div className="focus-bars">
            {tokens.slice(0, focusRow + 1).map((tok, c) => {
              const weight = rowWeights[c] ?? 0;
              const pct = (weight * 100).toFixed(1);
              return (
                <div key={c} className="focus-bar-row">
                  <span className="bar-tok mono">
                    {c}: “{formatToken(tok.text)}”
                  </span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: sequentialColor(weight),
                      }}
                    />
                  </div>
                  <span className="bar-pct mono">{pct}%</span>
                </div>
              );
            })}
          </div>

          <div className="focus-math-box mono">
            <code>
              {register === 'complete'
                ? `A[${focusRow}, j] = exp(q_${focusRow} · k_j / √32) / Σ exp(q_${focusRow} · k_m / √32)`
                : `Higher attention weights copy more value vector from earlier tokens into "${tokens[focusRow]?.text}".`}
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}

interface Level4Props {
  tokens: EncodedToken[];
  trace: Trace;
  manifest: ModelManifest;
  client: TransformerEngineClient;
  target: SelectionTarget;
  onSelect: (sel: string, zoomIn?: boolean) => void;
}

export function Level4View({
  tokens,
  trace,
  manifest,
  client,
  target,
  onSelect,
}: Level4Props) {
  const [source, setSource] = useState<'weight' | 'activation'>('weight');
  const [tensorData, setTensorData] = useState<Float32Array | null>(null);
  const [showTable, setShowTable] = useState(false);

  const meta = useMemo(
    () => manifest.tensors.find((t) => t.name === target.matrix) ?? manifest.tensors[0],
    [manifest, target.matrix]
  );

  useEffect(() => {
    if (source !== 'weight') return;
    let alive = true;
    void client.getTensor(meta.name).then((buf) => {
      if (alive) setTensorData(buf);
    });
    return () => {
      alive = false;
    };
  }, [client, meta.name, source]);

  const rows = source === 'weight' ? meta.shape[0] : tokens.length;
  const cols = source === 'weight' ? meta.shape[1] ?? 1 : manifest.config.d_model;

  const getWeightVal = (r: number, c: number): number => {
    if (!tensorData) return 0;
    return tensorData[r * cols + c] ?? 0;
  };

  const layerTrace = trace.layers[target.layer];
  const actMatrix = !layerTrace.skipped
    ? (layerTrace as ActiveLayerTrace).residualAfterMlp
    : trace.tokEmb;

  const getActVal = (r: number, c: number): number => {
    return actMatrix[r]?.[c] ?? 0;
  };

  return (
    <div className="scene-stage level4-stage">
      <div className="l4-header">
        <div>
          <h2>Tensor Inspector: {meta.name}</h2>
          <span className="mono l4-meta">
            Shape [{meta.shape.join(' × ')}] · {meta.count.toLocaleString()} parameters · {meta.dtype}
          </span>
        </div>

        <div className="l4-controls">
          <button
            type="button"
            className={`seg-btn mono ${source === 'weight' ? 'is-active' : ''}`}
            onClick={() => setSource('weight')}
          >
            Weights Matrix
          </button>
          <button
            type="button"
            className={`seg-btn mono ${source === 'activation' ? 'is-active' : ''}`}
            onClick={() => setSource('activation')}
          >
            Runtime Activation Matrix
          </button>
          <button
            type="button"
            className={`seg-btn mono ${showTable ? 'is-active' : ''}`}
            onClick={() => setShowTable((v) => !v)}
          >
            {showTable ? 'Hide numeric grid' : 'Show numeric grid'}
          </button>
        </div>
      </div>

      <div className="l4-body">
        <CanvasHeatmap
          rows={rows}
          cols={cols}
          getValue={source === 'weight' ? getWeightVal : getActVal}
          mode="diverging"
          title={`${meta.name} — ${source === 'weight' ? 'Learned Weights' : 'Live Activations'}`}
          selectedCell={
            source === 'weight' && target.matrix === meta.name
              ? { row: target.row, col: target.col }
              : null
          }
          onCellClick={(r, c) => onSelect(buildParamSelection(meta.name, r, c), true)}
          height={280}
        />

        {showTable && (
          <div className="l4-table-card">
            <div className="table-caption mono">
              First 8×8 slice of {meta.name}:
            </div>
            <table className="l4-num-table mono">
              <thead>
                <tr>
                  <th>r\c</th>
                  {Array.from({ length: Math.min(8, cols) }, (_, c) => (
                    <th key={c}>c{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: Math.min(8, rows) }, (_, r) => (
                  <tr key={r}>
                    <th>r{r}</th>
                    {Array.from({ length: Math.min(8, cols) }, (_, c) => {
                      const v = source === 'weight' ? getWeightVal(r, c) : getActVal(r, c);
                      return (
                        <td
                          key={c}
                          onClick={() => onSelect(buildParamSelection(meta.name, r, c), true)}
                          style={{ backgroundColor: divergingColor(v / 1.5) }}
                        >
                          {v.toFixed(3)}
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
    </div>
  );
}

interface Level5Props {
  client: TransformerEngineClient;
  manifest: ModelManifest;
  replay: TrainingReplayData | null;
  target: SelectionTarget;
  onSelect: (sel: string, zoomIn?: boolean) => void;
  onOpenReplay: () => void;
}

export function Level5View({
  client,
  manifest,
  replay,
  target,
  onSelect,
  onOpenReplay,
}: Level5Props) {
  const [scalarVal, setScalarVal] = useState<number | null>(null);

  const paramMeta = useMemo(
    () => manifest.tensors.find((t) => t.name === target.matrix) ?? manifest.tensors[0],
    [manifest, target.matrix]
  );

  const cols = paramMeta.shape[1] ?? 1;
  const flatIdx = target.row * cols + target.col;

  useEffect(() => {
    let alive = true;
    void client.getTensor(paramMeta.name).then((buf) => {
      if (alive) {
        setScalarVal(buf[flatIdx] ?? 0);
      }
    });
    return () => {
      alive = false;
    };
  }, [client, paramMeta.name, flatIdx]);

  const trajectoryKey = `${paramMeta.name}@${target.row},${target.col}`;
  const trajectory = replay?.trajectories?.[trajectoryKey] ?? null;

  return (
    <div className="scene-stage level5-stage">
      <div className="l5-hero">
        <div className="hero-left">
          <span className="badge-chip mono">Single Scalar Coordinate</span>
          <h2>
            {paramMeta.name}[{target.row}, {target.col}]
          </h2>
          <p className="hero-role">{paramMeta.role}</p>
        </div>

        <div className="hero-value-card mono">
          <span className="card-label">Real Float16 Value</span>
          <span className="card-value">
            {scalarVal !== null ? formatNum(scalarVal, 5) : '...'}
          </span>
        </div>
      </div>

      <div className="l5-grid">
        <div className="l5-meta-card">
          <h3>Coordinate Details</h3>
          <dl className="mono meta-dl">
            <dt>Tensor:</dt>
            <dd>{paramMeta.name}</dd>
            <dt>Row (Input Dim):</dt>
            <dd>{target.row} / {paramMeta.shape[0]}</dd>
            <dt>Col (Output Dim):</dt>
            <dd>{target.col} / {cols}</dd>
            <dt>Flat Buffer Offset:</dt>
            <dd>{flatIdx.toLocaleString()} (byte {flatIdx * 2})</dd>
            <dt>Storage Type:</dt>
            <dd>IEEE 754 float16 (2 bytes)</dd>
          </dl>

          <div className="hist-axis mono">
            <span>min {formatNum(paramMeta.stats.min, 3)}</span>
            <span>mean {formatNum(paramMeta.stats.mean, 3)}</span>
            <span>max {formatNum(paramMeta.stats.max, 3)}</span>
          </div>
        </div>

        <div className="l5-trajectory-card">
          <h3>Recorded Training Trajectory</h3>
          {trajectory ? (
            <ScalarTrajectoryChart points={trajectory} />
          ) : (
            <div className="untracked-note">
              <p>
                This coordinate was trained identically by stochastic gradient descent with AdamW, though only select coordinates were logged across every checkpoint to preserve bandwidth.
              </p>
              <button
                type="button"
                className="btn-inline"
                onClick={() => onSelect('blocks.0.attn.c_attn.weight@0,0', true)}
              >
                Jump to logged coordinate (Block 0 c_attn [0,0]) →
              </button>
            </div>
          )}

          <button type="button" className="seg-btn full-width" onClick={onOpenReplay}>
            Open Full Training Replay (1,500 Steps) →
          </button>
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
  const pad = 14;

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
