import { useEffect, useMemo, useRef, useState } from 'react';
import { FRONTIER_BRIDGES, SCALE_STOPS } from '../content/curriculum';
import type { TransformerEngineClient } from '../engine/client';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  ModelManifest,
  TokenizerData,
} from '../engine/types';
import { hasActiveAblation, type DrawerId } from '../state/atlasState';
import {
  divergingColor,
  formatNum,
  formatTokenExplicit,
  sequentialColor,
} from '../utils/math';

export interface ReplayCheckpoint {
  step: number;
  train_loss: number;
  val_loss: number;
  scalars: Record<string, number>;
  prompts: {
    prompt: string;
    completion: string;
    top5: { id: number; token: string; prob: number }[];
  }[];
}

export interface TrainingReplayData {
  version: number;
  tracked_scalars_meta: {
    key: string;
    param: string;
    coord: number[];
    description: string;
  }[];
  loss_curve: { step: number; train_loss: number; val_loss: number }[];
  checkpoints: ReplayCheckpoint[];
}

interface InspectorDrawersProps {
  drawer: DrawerId;
  client: TransformerEngineClient;
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  baseline: EngineTraceResult;
  ablated: EngineTraceResult | null;
  ablation: AblationConfig;
  layer: number;
  head: number;
  showPreSoftmax: boolean;
  hoveredTokenIdx: number | null;
  replay: TrainingReplayData | null;
  paramName: string;
  paramRow: number;
  paramCol: number;
  onChangeAblation: (next: AblationConfig) => void;
  onSelectParam: (name: string, row: number, col: number) => void;
  onTogglePreSoftmax: () => void;
  onHoverToken: (idx: number | null) => void;
  onClose: () => void;
}

export function InspectorDrawers(props: InspectorDrawersProps) {
  if (props.drawer === 'none') return null;

  return (
    <section className="lab-drawer glass-panel" role="dialog" aria-label="Educational Lab">
      <div className="lab-drawer-head">
        <h2>
          {props.drawer === 'ablate' && 'Surgical Ablation Lab (Counterfactuals)'}
          {props.drawer === 'weights' && 'Parameter Matrices & Single-Scalar Origin'}
          {props.drawer === 'training' && 'Training Replay & Frontier Architecture Bridge'}
          {props.drawer === 'matrix' &&
            `Exact Attention Table · Block ${props.layer} Head ${props.head}`}
        </h2>
        <button type="button" className="pill-btn" onClick={props.onClose}>
          Close
        </button>
      </div>

      <div className="lab-drawer-body">
        {props.drawer === 'ablate' && (
          <AblationLab
            baseline={props.baseline}
            ablated={props.ablated}
            ablation={props.ablation}
            tokenizer={props.tokenizer}
            onChange={props.onChangeAblation}
          />
        )}
        {props.drawer === 'weights' && (
          <WeightsLab
            client={props.client}
            manifest={props.manifest}
            replay={props.replay}
            paramName={props.paramName}
            paramRow={props.paramRow}
            paramCol={props.paramCol}
            onSelectParam={props.onSelectParam}
          />
        )}
        {props.drawer === 'training' && (
          <TrainingAndFrontierLab
            manifest={props.manifest}
            replay={props.replay}
            onJumpScalar={(name, r, c) => props.onSelectParam(name, r, c)}
          />
        )}
        {props.drawer === 'matrix' && (
          <ExactAttentionMatrix
            res={props.ablated ?? props.baseline}
            layer={props.layer}
            head={props.head}
            showPreSoftmax={props.showPreSoftmax}
            hoveredTokenIdx={props.hoveredTokenIdx}
            onTogglePreSoftmax={props.onTogglePreSoftmax}
            onHoverToken={props.onHoverToken}
          />
        )}
      </div>
    </section>
  );
}

function AblationLab({
  baseline,
  ablated,
  ablation,
  tokenizer,
  onChange,
}: {
  baseline: EngineTraceResult;
  ablated: EngineTraceResult | null;
  ablation: AblationConfig;
  tokenizer: TokenizerData;
  onChange: (next: AblationConfig) => void;
}) {
  const active = hasActiveAblation(ablation);

  const toggleHead = (l: number, h: number) => {
    const cur = ablation.disabledHeads ?? [];
    const exists = cur.some(([a, b]) => a === l && b === h);
    onChange({
      ...ablation,
      disabledHeads: exists
        ? cur.filter(([a, b]) => !(a === l && b === h))
        : [...cur, [l, h]],
    });
  };

  const toggleList = (
    key: 'disabledAttnBlocks' | 'disabledMlpBlocks' | 'disabledLayers',
    l: number
  ) => {
    const cur = ablation[key] ?? [];
    onChange({
      ...ablation,
      [key]: cur.includes(l) ? cur.filter((x) => x !== l) : [...cur, l],
    });
  };

  const toggleResidual = (kind: 'attn' | 'mlp', l: number) => {
    const cur = ablation.disabledResiduals ?? [];
    const exists = cur.some(([k, i]) => k === kind && i === l);
    onChange({
      ...ablation,
      disabledResiduals: exists
        ? cur.filter(([k, i]) => !(k === kind && i === l))
        : [...cur, [kind, l]],
    });
  };

  const toggleLn = (kind: 'ln_1' | 'ln_2' | 'ln_f', l: number) => {
    const cur = ablation.disabledLayerNorms ?? [];
    const exists = cur.some(([k, i]) => k === kind && i === l);
    onChange({
      ...ablation,
      disabledLayerNorms: exists
        ? cur.filter(([k, i]) => !(k === kind && i === l))
        : [...cur, [kind, l]],
    });
  };

  const rows = useMemo(() => {
    const ids = new Set<number>();
    baseline.trace.finalProbs
      .map((p, id) => ({ id, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 6)
      .forEach((x) => ids.add(x.id));
    if (ablated) {
      ablated.trace.finalProbs
        .map((p, id) => ({ id, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 4)
        .forEach((x) => ids.add(x.id));
    }
    return Array.from(ids)
      .map((id) => ({
        id,
        text: tokenizer.vocab[id],
        baseP: baseline.trace.finalProbs[id],
        ablP: ablated ? ablated.trace.finalProbs[id] : baseline.trace.finalProbs[id],
      }))
      .sort((a, b) => Math.max(b.baseP, b.ablP) - Math.max(a.baseP, a.ablP))
      .slice(0, 7);
  }, [baseline, ablated, tokenizer]);

  const baseTop = baseline.trace.greedyNextTokenId;
  const ablTop = ablated ? ablated.trace.greedyNextTokenId : baseTop;

  return (
    <div className="two-col">
      <div>
        <div className="row-between" style={{ marginBottom: 8 }}>
          <span className="muted-sm">Disable any component to test if it is load-bearing</span>
          {active && (
            <button type="button" className="pill-btn is-warm" onClick={() => onChange({})}>
              Reset all
            </button>
          )}
        </div>
        {[0, 1, 2, 3].map((l) => (
          <div key={l} className="ablate-row">
            <span className="mono ablate-block-lbl">Block {l}</span>
            <div className="chip-wrap">
              {[0, 1, 2, 3].map((h) => {
                const off = (ablation.disabledHeads ?? []).some(
                  ([a, b]) => a === l && b === h
                );
                return (
                  <button
                    key={h}
                    type="button"
                    className={`chip mono ${off ? 'is-off' : ''}`}
                    onClick={() => toggleHead(l, h)}
                  >
                    H{h}
                  </button>
                );
              })}
              <button
                type="button"
                className={`chip ${(ablation.disabledAttnBlocks ?? []).includes(l) ? 'is-off' : ''}`}
                onClick={() => toggleList('disabledAttnBlocks', l)}
              >
                Attn
              </button>
              <button
                type="button"
                className={`chip ${(ablation.disabledMlpBlocks ?? []).includes(l) ? 'is-off' : ''}`}
                onClick={() => toggleList('disabledMlpBlocks', l)}
              >
                MLP
              </button>
              <button
                type="button"
                className={`chip ${
                  (ablation.disabledResiduals ?? []).some(([k, i]) => k === 'attn' && i === l)
                    ? 'is-off'
                    : ''
                }`}
                onClick={() => toggleResidual('attn', l)}
              >
                Res-A
              </button>
              <button
                type="button"
                className={`chip ${
                  (ablation.disabledLayerNorms ?? []).some(([k, i]) => k === 'ln_1' && i === l)
                    ? 'is-off'
                    : ''
                }`}
                onClick={() => toggleLn('ln_1', l)}
              >
                LN1
              </button>
              <button
                type="button"
                className={`chip ${(ablation.disabledLayers ?? []).includes(l) ? 'is-off' : ''}`}
                onClick={() => toggleList('disabledLayers', l)}
              >
                Skip
              </button>
            </div>
          </div>
        ))}
        <div className="ablate-row">
          <span className="mono ablate-block-lbl">Final</span>
          <div className="chip-wrap">
            <button
              type="button"
              className={`chip ${
                (ablation.disabledLayerNorms ?? []).some(([k]) => k === 'ln_f')
                  ? 'is-off'
                  : ''
              }`}
              onClick={() => toggleLn('ln_f', -1)}
            >
              Disable Final LN_f
            </button>
          </div>
        </div>
      </div>

      <div>
        <div className="callout-box">
          {active && ablated ? (
            <>
              Top prediction moved from{' '}
              <strong className="mono">
                “{formatTokenExplicit(tokenizer.vocab[baseTop])}” (
                {(baseline.trace.finalProbs[baseTop] * 100).toFixed(1)}%)
              </strong>{' '}
              to{' '}
              <strong className="mono">
                “{formatTokenExplicit(tokenizer.vocab[ablTop])}” (
                {(ablated.trace.finalProbs[ablTop] * 100).toFixed(1)}%)
              </strong>
              .
              {baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null && (
                <span>
                  {' '}
                  Cross-entropy loss Δ:{' '}
                  <strong className="mono">
                    {ablated.trace.inputLoss - baseline.trace.inputLoss >= 0 ? '+' : ''}
                    {(ablated.trace.inputLoss - baseline.trace.inputLoss).toFixed(3)}
                  </strong>
                  .
                </span>
              )}
            </>
          ) : (
            'Toggle any switch on the left. Blue bars show the original distribution; orange bars show the ablated counterfactual.'
          )}
        </div>
        {rows.map((item) => {
          const delta = (item.ablP - item.baseP) * 100;
          return (
            <div key={item.id} className="diff-row">
              <span className="mono">{formatTokenExplicit(item.text)}</span>
              <div className="dual-bars">
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Math.min(100, item.baseP * 100)}%` }} />
                </div>
                {active && (
                  <div className="bar-track">
                    <div
                      className="bar-fill-warm"
                      style={{ width: `${Math.min(100, item.ablP * 100)}%` }}
                    />
                  </div>
                )}
              </div>
              <span className="mono diff-val">
                {(item.baseP * 100).toFixed(1)}%
                {active && (
                  <span className={delta >= 0 ? 'txt-up' : 'txt-down'}>
                    {' '}
                    ({delta >= 0 ? '+' : ''}
                    {delta.toFixed(1)})
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function WeightsLab({
  client,
  manifest,
  replay,
  paramName,
  paramRow,
  paramCol,
  onSelectParam,
}: {
  client: TransformerEngineClient;
  manifest: ModelManifest;
  replay: TrainingReplayData | null;
  paramName: string;
  paramRow: number;
  paramCol: number;
  onSelectParam: (name: string, row: number, col: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [data, setData] = useState<Float32Array | null>(null);
  const [hover, setHover] = useState<{ r: number; c: number; v: number } | null>(null);

  const meta =
    manifest.tensors.find((t) => t.name === paramName) ?? manifest.tensors[0];
  const is1D = meta.shape.length === 1;
  const rows = meta.shape[0];
  const cols = is1D ? 1 : (meta.shape[1] ?? 1);
  const clampedR = Math.min(rows - 1, Math.max(0, paramRow));
  const clampedC = is1D ? 0 : Math.min(cols - 1, Math.max(0, paramCol));

  useEffect(() => {
    let alive = true;
    client.getTensor(meta.name).then((arr) => {
      if (alive) setData(arr);
    });
    return () => {
      alive = false;
    };
  }, [client, meta.name]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxAbs = Math.max(Math.abs(meta.stats.min), Math.abs(meta.stats.max), 1e-6);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = divergingColor((data[r * cols + c] ?? 0) / maxAbs);
        ctx.fillRect(c, r, 1, 1);
      }
    }
    ctx.fillStyle = '#141923';
    ctx.fillRect(clampedC, clampedR, 1, 1);
  }, [data, rows, cols, meta.stats.min, meta.stats.max, clampedR, clampedC]);

  const scalarVal = data ? (data[clampedR * cols + clampedC] ?? 0) : null;
  const coordKey = is1D
    ? `${meta.name}[${clampedR}]`
    : `${meta.name}[${clampedR}, ${clampedC}]`;

  const trajectory = useMemo(() => {
    if (!replay) return null;
    const pts = replay.checkpoints
      .map((ck) => ({ step: ck.step, val: ck.scalars[coordKey] }))
      .filter((p): p is { step: number; val: number } => typeof p.val === 'number');
    return pts.length > 1 ? pts : null;
  }, [replay, coordKey]);

  const maxHist = Math.max(...meta.histogram.counts, 1);

  const pickCell = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return null;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width - 0.001, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height - 0.001, e.clientY - rect.top));
    const c = Math.min(cols - 1, Math.floor((x / rect.width) * cols));
    const r = Math.min(rows - 1, Math.floor((y / rect.height) * rows));
    return { r, c, v: data[r * cols + c] ?? 0 };
  };

  return (
    <div className="two-col">
      <div>
        <div className="row-between" style={{ marginBottom: 8 }}>
          <select
            className="select-pill mono"
            value={meta.name}
            onChange={(e) => onSelectParam(e.target.value, 0, 0)}
            aria-label="Choose weight matrix"
          >
            {manifest.tensors.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name} [{t.shape.join('×')}] ({t.count.toLocaleString()})
              </option>
            ))}
          </select>
        </div>
        <p className="muted-sm" style={{ margin: '0 0 8px' }}>
          {meta.role}
        </p>
        <canvas
          ref={canvasRef}
          className="weight-canvas"
          onMouseMove={(e) => setHover(pickCell(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const p = pickCell(e);
            if (p) onSelectParam(meta.name, p.r, p.c);
          }}
        />
        <div className="row-between mono muted-sm" style={{ marginTop: 6 }}>
          <span>
            {hover
              ? `(${hover.r}, ${hover.c}) = ${hover.v.toFixed(5)}`
              : 'Hover to read · Click any pixel to isolate scalar'}
          </span>
          <span>
            [{formatNum(meta.stats.min, 3)}, {formatNum(meta.stats.max, 3)}]
          </span>
        </div>
      </div>

      <div>
        <div className="scalar-box">
          <div className="muted-sm mono">{coordKey}</div>
          <div className="scalar-big mono">
            {scalarVal !== null ? scalarVal.toFixed(6) : '...'}
          </div>
          <div className="muted-sm">
            Real float16 weight decoded from <span className="mono">weights.fp16.bin</span>.
          </div>
        </div>

        <div className="hist-wrap">
          {meta.histogram.counts.map((cnt, i) => {
            const e0 = meta.histogram.edges[i];
            const e1 = meta.histogram.edges[i + 1];
            const hit = scalarVal !== null && scalarVal >= e0 && scalarVal <= e1;
            return (
              <div
                key={i}
                className={`hist-col ${hit ? 'is-hit' : ''}`}
                style={{ height: `${Math.max(5, (cnt / maxHist) * 100)}%` }}
                title={`[${e0.toFixed(3)}, ${e1.toFixed(3)}]: ${cnt}`}
              />
            );
          })}
        </div>

        {trajectory ? (
          <MiniCurve label="Recorded AdamW training trajectory (Step 0 → Final)" points={trajectory} />
        ) : (
          <p className="muted-sm" style={{ marginTop: 10 }}>
            Pick one of the 12 checkpoint-tracked weights below to plot its exact Step 0 → Final training curve:
          </p>
        )}

        {replay && (
          <div className="chip-wrap" style={{ marginTop: 8 }}>
            {replay.tracked_scalars_meta.slice(0, 6).map((m) => (
              <button
                key={m.key}
                type="button"
                className={`chip mono ${m.key === coordKey ? 'is-active' : ''}`}
                onClick={() => onSelectParam(m.param, m.coord[0] ?? 0, m.coord[1] ?? 0)}
              >
                {m.key}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MiniCurve({
  points,
  label,
}: {
  points: { step: number; val: number }[];
  label: string;
}) {
  const w = 340;
  const h = 96;
  const pad = 16;
  const maxS = Math.max(...points.map((p) => p.step), 1);
  const minV = Math.min(...points.map((p) => p.val));
  const maxV = Math.max(...points.map((p) => p.val));
  const span = Math.max(maxV - minV, 1e-4);
  const d = points
    .map((p, i) => {
      const x = pad + (p.step / maxS) * (w - pad * 2);
      const y = h - pad - ((p.val - minV) / span) * (h - pad * 2);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <div style={{ marginTop: 10 }}>
      <div className="muted-sm" style={{ marginBottom: 4 }}>
        {label}
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="mini-svg">
        <path d={d} fill="none" stroke="#1d4ed8" strokeWidth="2.2" />
      </svg>
    </div>
  );
}

function TrainingAndFrontierLab({
  manifest,
  replay,
  onJumpScalar,
}: {
  manifest: ModelManifest;
  replay: TrainingReplayData | null;
  onJumpScalar: (name: string, r: number, c: number) => void;
}) {
  const [ckptIdx, setCkptIdx] = useState(0);
  const [scaleIdx, setScaleIdx] = useState(0);
  const stop = SCALE_STOPS[scaleIdx];
  const ckpt = replay
    ? replay.checkpoints[Math.min(ckptIdx, replay.checkpoints.length - 1)]
    : null;

  return (
    <div className="two-col">
      <div>
        <h3 className="sub-h3">Real Training Replay (Backpropagation & AdamW)</h3>
        {replay && ckpt ? (
          <>
            <div className="slider-box">
              <label htmlFor="tr-scrub" className="mono">
                Step {ckpt.step} / {replay.checkpoints[replay.checkpoints.length - 1].step} · val loss{' '}
                {ckpt.val_loss.toFixed(3)}
              </label>
              <input
                id="tr-scrub"
                type="range"
                min={0}
                max={replay.checkpoints.length - 1}
                value={ckptIdx}
                onChange={(e) => setCkptIdx(Number(e.target.value))}
              />
            </div>
            <MiniCurve
              points={replay.loss_curve.map((p) => ({ step: p.step, val: p.val_loss }))}
              label={`Validation cross-entropy loss (${replay.checkpoints[0].val_loss.toFixed(2)} → ${replay.checkpoints[replay.checkpoints.length - 1].val_loss.toFixed(2)})`}
            />
            <div style={{ marginTop: 10 }}>
              {ckpt.prompts.slice(0, 2).map((item, i) => (
                <div key={i} className="sample-item">
                  <div className="muted-sm">{item.prompt}</div>
                  <div className="mono" style={{ color: '#1d4ed8' }}>
                    → {item.completion}
                  </div>
                </div>
              ))}
            </div>
            <div className="chip-wrap" style={{ marginTop: 8 }}>
              {replay.tracked_scalars_meta.slice(0, 4).map((m) => (
                <button
                  key={m.key}
                  type="button"
                  className="chip mono"
                  onClick={() => onJumpScalar(m.param, m.coord[0] ?? 0, m.coord[1] ?? 0)}
                >
                  {m.key}: {(ckpt.scalars[m.key] ?? 0).toFixed(4)}
                </button>
              ))}
            </div>
          </>
        ) : (
          <p className="muted-sm">Loading training_replay.json...</p>
        )}
      </div>

      <div>
        <h3 className="sub-h3">Scale Ladder & Frontier Architecture Bridge</h3>
        <div className="slider-box">
          <label htmlFor="sc-scrub">
            <strong>{stop.label}</strong>{' '}
            <span className="mono">
              ({stop.params.toLocaleString()} params ·{' '}
              {Math.max(1, Math.round(stop.params / manifest.total_params)).toLocaleString()}×)
            </span>
          </label>
          <input
            id="sc-scrub"
            type="range"
            min={0}
            max={SCALE_STOPS.length - 1}
            value={scaleIdx}
            onChange={(e) => setScaleIdx(Number(e.target.value))}
          />
        </div>
        <div className="muted-sm mono" style={{ marginBottom: 6 }}>
          {stop.layers} layers · d_model {stop.dModel.toLocaleString()} · {stop.heads} heads · ctx{' '}
          {stop.context.toLocaleString()}
        </div>
        <p className="muted-sm" style={{ margin: '0 0 10px' }}>
          {stop.note}
        </p>

        <table className="bridge-table">
          <thead>
            <tr>
              <th>Component</th>
              <th>This Miniature Model</th>
              <th>Modern Frontier LLMs</th>
            </tr>
          </thead>
          <tbody>
            {FRONTIER_BRIDGES.map((b) => (
              <tr key={b.concept}>
                <td>
                  <strong>{b.concept}</strong>
                </td>
                <td>{b.here}</td>
                <td>{b.frontier}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ExactAttentionMatrix({
  res,
  layer,
  head,
  showPreSoftmax,
  hoveredTokenIdx,
  onTogglePreSoftmax,
  onHoverToken,
}: {
  res: EngineTraceResult;
  layer: number;
  head: number;
  showPreSoftmax: boolean;
  hoveredTokenIdx: number | null;
  onTogglePreSoftmax: () => void;
  onHoverToken: (idx: number | null) => void;
}) {
  const lt = res.trace.layers[layer];
  if (lt.skipped) {
    return <p className="muted-sm">Block {layer} is disabled in Ablation Lab.</p>;
  }
  const active: ActiveLayerTrace = lt;
  const mat = showPreSoftmax
    ? active.attn.rawScores[head]
    : active.attn.attnWeights[head];
  const tokens = res.tokens;
  let maxAbs = 1e-6;
  for (let r = 0; r < tokens.length; r++) {
    for (let c = 0; c <= r; c++) {
      maxAbs = Math.max(maxAbs, Math.abs(active.attn.rawScores[head][r][c]));
    }
  }

  return (
    <div>
      <div className="row-between" style={{ marginBottom: 8 }}>
        <span className="muted-sm">
          Rows = Query tokens · Columns = Key tokens · Upper-right triangle causally masked (-∞)
        </span>
        <button
          type="button"
          className={`pill-btn ${showPreSoftmax ? 'is-active' : ''}`}
          onClick={onTogglePreSoftmax}
        >
          {showPreSoftmax ? 'Showing: Pre-Softmax QKᵀ/√32' : 'Showing: Post-Softmax Weights'}
        </button>
      </div>
      <div className="matrix-scroll">
        <table className="exact-table mono">
          <thead>
            <tr>
              <th>Q \ K</th>
              {tokens.map((t, c) => (
                <th
                  key={`c-${c}`}
                  className={hoveredTokenIdx === c ? 'is-cross' : ''}
                  onMouseEnter={() => onHoverToken(c)}
                  onMouseLeave={() => onHoverToken(null)}
                >
                  {formatTokenExplicit(t.text)}
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
                  const v = mat[r][c];
                  const bg = masked
                    ? '#f1f5f9'
                    : showPreSoftmax
                      ? divergingColor(v / maxAbs)
                      : sequentialColor(v);
                  const dark = !masked && !showPreSoftmax && v > 0.45;
                  return (
                    <td
                      key={`td-${r}-${c}`}
                      style={{ backgroundColor: bg, color: dark ? '#fff' : '#141923' }}
                      onMouseEnter={() => !masked && onHoverToken(r)}
                      onMouseLeave={() => onHoverToken(null)}
                    >
                      {masked ? '' : showPreSoftmax ? formatNum(v, 2) : (v * 100).toFixed(0)}
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
