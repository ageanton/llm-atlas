import { useEffect, useMemo, useRef, useState } from 'react';
import {
  resolveParamName,
  SCALE_STOPS,
  type AnatomyNode,
} from '../content/llmAnatomy';
import type { TransformerEngineClient } from '../engine/client';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  ModelManifest,
  TokenizerData,
} from '../engine/types';
import { hasActiveAblation } from '../state/atlasState';
import {
  computeSampledDistribution,
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

export interface NodeLabProps {
  node: AnatomyNode;
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
  temperature: number;
  topK: number;
  topP: number;
  busy: boolean;
  onChangeAblation: (next: AblationConfig) => void;
  onInspectScalar: (paramName: string, row: number, col: number) => void;
  onTogglePreSoftmax: () => void;
  onHoverToken: (idx: number | null) => void;
  onChangeSampling: (patch: {
    temperature?: number;
    topK?: number;
    topP?: number;
  }) => void;
  onStepAutoregressive: () => void;
}

export function NodeLab(props: NodeLabProps) {
  const activeRes = props.ablated ?? props.baseline;

  if (props.node.labKind === 'none') return null;

  if (props.node.labKind === 'tokenizer') {
    return (
      <div className="lab-card">
        <div className="lab-card-title">
          Live Tokenization ({activeRes.tokens.length} / {props.manifest.config.max_seq_len} slots)
        </div>
        <div className="tok-grid">
          {activeRes.tokens.map((tok, i) => (
            <button
              key={`${tok.start}-${i}`}
              type="button"
              className={`tok-badge mono ${props.hoveredTokenIdx === i ? 'is-hover' : ''}`}
              onMouseEnter={() => props.onHoverToken(i)}
              onMouseLeave={() => props.onHoverToken(null)}
            >
              <span>{formatTokenExplicit(tok.text)}</span>
              <span className="tok-badge-id">#{tok.id}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (props.node.labKind === 'logit-lens') {
    return (
      <div className="lab-card">
        <div className="lab-card-title">Live Logit Lens Across All 4 Blocks</div>
        {activeRes.trace.logitLens.map((entry) => {
          const top = entry.topTokens[0];
          const tok = top ? formatTokenExplicit(props.tokenizer.vocab[top.id] ?? '') : '-';
          const pct = top ? top.prob * 100 : 0;
          return (
            <div key={entry.afterLayer} className="lens-row mono">
              <span>Block {entry.afterLayer}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, pct)}%` }} />
              </div>
              <span>
                {tok} ({pct.toFixed(0)}%)
              </span>
            </div>
          );
        })}
      </div>
    );
  }

  if (props.node.labKind === 'attention-matrix') {
    return (
      <InlineAttentionMatrix
        res={activeRes}
        layer={props.layer}
        head={props.head}
        showPreSoftmax={props.showPreSoftmax}
        hoveredTokenIdx={props.hoveredTokenIdx}
        onTogglePreSoftmax={props.onTogglePreSoftmax}
        onHoverToken={props.onHoverToken}
      />
    );
  }

  if (props.node.labKind === 'weight-matrix') {
    const resolvedParam = resolveParamName(
      props.node,
      props.layer,
      props.paramName
    );
    return (
      <InlineWeightMatrix
        client={props.client}
        manifest={props.manifest}
        paramName={resolvedParam}
        onPickScalar={props.onInspectScalar}
      />
    );
  }

  if (props.node.labKind === 'scalar') {
    return (
      <InlineScalarView
        client={props.client}
        manifest={props.manifest}
        replay={props.replay}
        paramName={props.paramName}
        paramRow={props.paramRow}
        paramCol={props.paramCol}
        onPickScalar={props.onInspectScalar}
      />
    );
  }

  if (props.node.labKind === 'sampling') {
    return (
      <InlineSamplingLab
        res={activeRes}
        tokenizer={props.tokenizer}
        temperature={props.temperature}
        topK={props.topK}
        topP={props.topP}
        busy={props.busy}
        onChange={props.onChangeSampling}
        onStep={props.onStepAutoregressive}
      />
    );
  }

  if (props.node.labKind === 'replay') {
    return (
      <InlineReplayLab
        replay={props.replay}
        onPickScalar={props.onInspectScalar}
      />
    );
  }

  if (props.node.labKind === 'ablation') {
    return (
      <InlineAblationLab
        baseline={props.baseline}
        ablated={props.ablated}
        ablation={props.ablation}
        tokenizer={props.tokenizer}
        onChange={props.onChangeAblation}
      />
    );
  }

  if (props.node.labKind === 'scale') {
    return <InlineScaleLab manifest={props.manifest} />;
  }

  return null;
}

function InlineAttentionMatrix({
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
    return (
      <div className="lab-card muted-sm">
        Block {layer} is currently skipped by Ablation Lab.
      </div>
    );
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

  const focusRow = hoveredTokenIdx ?? tokens.length - 1;
  const rowSum = active.attn.attnWeights[head][focusRow].reduce((a, b) => a + b, 0);

  return (
    <div className="lab-card">
      <div className="row-between" style={{ marginBottom: 8 }}>
        <span className="lab-card-title" style={{ margin: 0 }}>
          Block {layer} · Head {head} [{tokens.length}×{tokens.length}]
        </span>
        <button
          type="button"
          className={`mini-btn ${showPreSoftmax ? 'is-active' : ''}`}
          onClick={onTogglePreSoftmax}
        >
          {showPreSoftmax ? 'Raw QKᵀ/√32' : 'Softmax A'}
        </button>
      </div>
      <div className="matrix-scroll">
        <table className="exact-table mono">
          <thead>
            <tr>
              <th>Q\K</th>
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
                      {masked ? '' : showPreSoftmax ? formatNum(v, 1) : (v * 100).toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="muted-xs mono" style={{ marginTop: 6 }}>
        Row “{formatTokenExplicit(tokens[focusRow].text)}” post-softmax sum = {rowSum.toFixed(6)}
      </div>
    </div>
  );
}

function InlineWeightMatrix({
  client,
  manifest,
  paramName,
  onPickScalar,
}: {
  client: TransformerEngineClient;
  manifest: ModelManifest;
  paramName: string;
  onPickScalar: (param: string, row: number, col: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [data, setData] = useState<Float32Array | null>(null);
  const [hover, setHover] = useState<{ r: number; c: number; v: number } | null>(null);

  const meta =
    manifest.tensors.find((t) => t.name === paramName) ?? manifest.tensors[0];
  const is1D = meta.shape.length === 1;
  const rows = meta.shape[0];
  const cols = is1D ? 1 : (meta.shape[1] ?? 1);

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
    canvas.width = is1D ? rows : cols;
    canvas.height = is1D ? 1 : rows;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxAbs = Math.max(Math.abs(meta.stats.min), Math.abs(meta.stats.max), 1e-6);
    if (is1D) {
      for (let i = 0; i < rows; i++) {
        ctx.fillStyle = divergingColor((data[i] ?? 0) / maxAbs);
        ctx.fillRect(i, 0, 1, 1);
      }
    } else {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          ctx.fillStyle = divergingColor((data[r * cols + c] ?? 0) / maxAbs);
          ctx.fillRect(c, r, 1, 1);
        }
      }
    }
  }, [data, rows, cols, is1D, meta.stats.min, meta.stats.max]);

  const pick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return null;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width - 0.001, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height - 0.001, e.clientY - rect.top));
    if (is1D) {
      const r = Math.min(rows - 1, Math.floor((x / rect.width) * rows));
      return { r, c: 0, v: data[r] ?? 0 };
    }
    const c = Math.min(cols - 1, Math.floor((x / rect.width) * cols));
    const r = Math.min(rows - 1, Math.floor((y / rect.height) * rows));
    return { r, c, v: data[r * cols + c] ?? 0 };
  };

  return (
    <div className="lab-card">
      <div className="row-between" style={{ marginBottom: 6 }}>
        <span className="lab-card-title mono" style={{ margin: 0 }}>
          {meta.name} [{meta.shape.join('×')}]
        </span>
        <span className="muted-xs mono">{meta.count.toLocaleString()} params</span>
      </div>
      <canvas
        ref={canvasRef}
        className="weight-canvas"
        style={{ height: is1D ? 44 : 185 }}
        onMouseMove={(e) => setHover(pick(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const p = pick(e);
          if (p) onPickScalar(meta.name, p.r, p.c);
        }}
      />
      <div className="row-between muted-xs mono" style={{ marginTop: 6 }}>
        <span>
          {hover
            ? `(${hover.r}${is1D ? '' : `, ${hover.c}`}) = ${hover.v.toFixed(5)}`
            : 'Click any cell to zoom into that Single Scalar'}
        </span>
        <span>
          [{formatNum(meta.stats.min, 3)}, {formatNum(meta.stats.max, 3)}]
        </span>
      </div>
    </div>
  );
}

function InlineScalarView({
  client,
  manifest,
  replay,
  paramName,
  paramRow,
  paramCol,
  onPickScalar,
}: {
  client: TransformerEngineClient;
  manifest: ModelManifest;
  replay: TrainingReplayData | null;
  paramName: string;
  paramRow: number;
  paramCol: number;
  onPickScalar: (param: string, row: number, col: number) => void;
}) {
  const [val, setVal] = useState<number | null>(null);
  const meta =
    manifest.tensors.find((t) => t.name === paramName) ?? manifest.tensors[0];
  const is1D = meta.shape.length === 1;
  const r = Math.min(meta.shape[0] - 1, Math.max(0, paramRow));
  const c = is1D ? 0 : Math.min((meta.shape[1] ?? 1) - 1, Math.max(0, paramCol));

  useEffect(() => {
    let alive = true;
    client.getTensor(meta.name).then((arr) => {
      if (!alive) return;
      const idx = is1D ? r : r * (meta.shape[1] ?? 1) + c;
      setVal(arr[idx] ?? 0);
    });
    return () => {
      alive = false;
    };
  }, [client, meta, r, c, is1D]);

  const coordKey = is1D ? `${meta.name}[${r}]` : `${meta.name}[${r}, ${c}]`;
  const trajectory = useMemo(() => {
    if (!replay) return null;
    const pts = replay.checkpoints
      .map((ck) => ({ step: ck.step, val: ck.scalars[coordKey] }))
      .filter((p): p is { step: number; val: number } => typeof p.val === 'number');
    return pts.length > 1 ? pts : null;
  }, [replay, coordKey]);

  const maxHist = Math.max(...meta.histogram.counts, 1);

  return (
    <div className="lab-card">
      <div className="muted-xs mono">{coordKey}</div>
      <div className="scalar-readout mono">
        {val !== null ? val.toFixed(6) : '...'}
      </div>
      <div className="muted-xs" style={{ marginBottom: 8 }}>
        {meta.role}
      </div>

      <div className="hist-wrap" aria-label="Weight distribution histogram">
        {meta.histogram.counts.map((cnt, i) => {
          const e0 = meta.histogram.edges[i];
          const e1 = meta.histogram.edges[i + 1];
          const hit = val !== null && val >= e0 && val <= e1;
          return (
            <div
              key={i}
              className={`hist-col ${hit ? 'is-hit' : ''}`}
              style={{ height: `${Math.max(5, (cnt / maxHist) * 100)}%` }}
            />
          );
        })}
      </div>

      {trajectory ? (
        <MiniLineChart label="Recorded Step 0 → Final AdamW trajectory" points={trajectory} />
      ) : (
        <div className="muted-xs" style={{ marginTop: 8 }}>
          Jump to one of the 12 checkpoint-tracked weights to plot its training trajectory:
        </div>
      )}

      {replay && (
        <div className="chip-wrap" style={{ marginTop: 8 }}>
          {replay.tracked_scalars_meta.slice(0, 6).map((m) => (
            <button
              key={m.key}
              type="button"
              className={`mini-btn mono ${m.key === coordKey ? 'is-active' : ''}`}
              onClick={() => onPickScalar(m.param, m.coord[0] ?? 0, m.coord[1] ?? 0)}
            >
              {m.key}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function MiniLineChart({
  points,
  label,
}: {
  points: { step: number; val: number }[];
  label: string;
}) {
  const w = 340;
  const h = 86;
  const pad = 14;
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
      <div className="muted-xs" style={{ marginBottom: 4 }}>
        {label}
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="mini-svg">
        <path d={d} fill="none" stroke="#1d4ed8" strokeWidth="2.2" />
      </svg>
    </div>
  );
}

function InlineSamplingLab({
  res,
  tokenizer,
  temperature,
  topK,
  topP,
  busy,
  onChange,
  onStep,
}: {
  res: EngineTraceResult;
  tokenizer: TokenizerData;
  temperature: number;
  topK: number;
  topP: number;
  busy: boolean;
  onChange: (patch: { temperature?: number; topK?: number; topP?: number }) => void;
  onStep: () => void;
}) {
  const dist = useMemo(
    () =>
      computeSampledDistribution(
        res.trace.finalLogits,
        temperature,
        topK,
        topP
      ),
    [res.trace.finalLogits, temperature, topK, topP]
  );

  return (
    <div className="lab-card">
      <div className="lab-card-title">Live Sampling & Autoregressive Step</div>
      <div className="slider-row">
        <div className="row-between">
          <span>Temperature (τ)</span>
          <span className="mono">{temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={2.0}
          step={0.05}
          value={temperature}
          onChange={(e) => onChange({ temperature: Number(e.target.value) })}
        />
      </div>
      <div className="slider-row">
        <div className="row-between">
          <span>Top-k</span>
          <span className="mono">{topK}</span>
        </div>
        <input
          type="range"
          min={1}
          max={40}
          step={1}
          value={topK}
          onChange={(e) => onChange({ topK: Number(e.target.value) })}
        />
      </div>
      <div className="slider-row">
        <div className="row-between">
          <span>Top-p (Nucleus)</span>
          <span className="mono">{topP.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={topP}
          onChange={(e) => onChange({ topP: Number(e.target.value) })}
        />
      </div>

      <button
        type="button"
        className="primary-btn"
        style={{ width: '100%', margin: '6px 0 10px' }}
        disabled={busy}
        onClick={onStep}
      >
        Step +1 Token (Sample & Loop)
      </button>

      {dist.slice(0, 5).map((item) => (
        <div key={item.id} className="lens-row mono">
          <span>{formatTokenExplicit(tokenizer.vocab[item.id] ?? '')}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.min(100, item.prob * 100)}%` }} />
          </div>
          <span>{(item.prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function InlineReplayLab({
  replay,
  onPickScalar,
}: {
  replay: TrainingReplayData | null;
  onPickScalar: (param: string, row: number, col: number) => void;
}) {
  const [idx, setIdx] = useState(0);
  if (!replay) {
    return <div className="lab-card muted-sm">Loading training_replay.json...</div>;
  }
  const ckpt = replay.checkpoints[Math.min(idx, replay.checkpoints.length - 1)];

  return (
    <div className="lab-card">
      <div className="lab-card-title">Real Training Checkpoints</div>
      <div className="slider-row">
        <div className="row-between mono">
          <span>Step {ckpt.step} / {replay.checkpoints[replay.checkpoints.length - 1].step}</span>
          <span>val loss {ckpt.val_loss.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={replay.checkpoints.length - 1}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
        />
      </div>

      <MiniLineChart
        points={replay.loss_curve.map((p) => ({ step: p.step, val: p.val_loss }))}
        label={`Validation loss (${replay.checkpoints[0].val_loss.toFixed(2)} → ${replay.checkpoints[replay.checkpoints.length - 1].val_loss.toFixed(2)})`}
      />

      <div style={{ marginTop: 10 }}>
        {ckpt.prompts.slice(0, 2).map((item, i) => (
          <div key={i} className="sample-block">
            <div className="muted-xs">{item.prompt}</div>
            <div className="mono sample-out">→ {item.completion}</div>
          </div>
        ))}
      </div>

      <div className="chip-wrap" style={{ marginTop: 8 }}>
        {replay.tracked_scalars_meta.slice(0, 4).map((m) => (
          <button
            key={m.key}
            type="button"
            className="mini-btn mono"
            onClick={() => onPickScalar(m.param, m.coord[0] ?? 0, m.coord[1] ?? 0)}
          >
            {m.key}: {(ckpt.scalars[m.key] ?? 0).toFixed(4)}
          </button>
        ))}
      </div>
    </div>
  );
}

function InlineAblationLab({
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
      .slice(0, 5)
      .forEach((x) => ids.add(x.id));
    if (ablated) {
      ablated.trace.finalProbs
        .map((p, id) => ({ id, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3)
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
      .slice(0, 6);
  }, [baseline, ablated, tokenizer]);

  return (
    <div className="lab-card">
      <div className="row-between" style={{ marginBottom: 8 }}>
        <span className="lab-card-title" style={{ margin: 0 }}>
          Disable Live Components
        </span>
        {active && (
          <button type="button" className="mini-btn is-warm" onClick={() => onChange({})}>
            Reset all
          </button>
        )}
      </div>

      {[0, 1, 2, 3].map((l) => (
        <div key={l} className="ablate-line">
          <span className="mono ablate-tag">B{l}</span>
          <div className="chip-wrap">
            {[0, 1, 2, 3].map((h) => {
              const off = (ablation.disabledHeads ?? []).some(
                ([a, b]) => a === l && b === h
              );
              return (
                <button
                  key={h}
                  type="button"
                  className={`mini-btn mono ${off ? 'is-warm' : ''}`}
                  onClick={() => toggleHead(l, h)}
                >
                  H{h}
                </button>
              );
            })}
            <button
              type="button"
              className={`mini-btn ${(ablation.disabledAttnBlocks ?? []).includes(l) ? 'is-warm' : ''}`}
              onClick={() => toggleList('disabledAttnBlocks', l)}
            >
              Attn
            </button>
            <button
              type="button"
              className={`mini-btn ${(ablation.disabledMlpBlocks ?? []).includes(l) ? 'is-warm' : ''}`}
              onClick={() => toggleList('disabledMlpBlocks', l)}
            >
              MLP
            </button>
            <button
              type="button"
              className={`mini-btn ${
                (ablation.disabledResiduals ?? []).some(([k, i]) => k === 'attn' && i === l)
                  ? 'is-warm'
                  : ''
              }`}
              onClick={() => toggleResidual('attn', l)}
            >
              Res
            </button>
            <button
              type="button"
              className={`mini-btn ${
                (ablation.disabledLayerNorms ?? []).some(([k, i]) => k === 'ln_1' && i === l)
                  ? 'is-warm'
                  : ''
              }`}
              onClick={() => toggleLn('ln_1', l)}
            >
              LN1
            </button>
            <button
              type="button"
              className={`mini-btn ${(ablation.disabledLayers ?? []).includes(l) ? 'is-warm' : ''}`}
              onClick={() => toggleList('disabledLayers', l)}
            >
              Skip
            </button>
          </div>
        </div>
      ))}

      <div style={{ marginTop: 10 }}>
        {active && ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null && (
          <div className="muted-xs mono" style={{ marginBottom: 6 }}>
            Loss shift: {baseline.trace.inputLoss.toFixed(3)} → {ablated.trace.inputLoss.toFixed(3)} (
            {ablated.trace.inputLoss - baseline.trace.inputLoss >= 0 ? '+' : ''}
            {(ablated.trace.inputLoss - baseline.trace.inputLoss).toFixed(3)})
          </div>
        )}
        {rows.map((item) => (
          <div key={item.id} className="lens-row mono">
            <span>{formatTokenExplicit(item.text)}</span>
            <div className="dual-bars">
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, item.baseP * 100)}%` }} />
              </div>
              {active && (
                <div className="bar-track">
                  <div className="bar-fill-warm" style={{ width: `${Math.min(100, item.ablP * 100)}%` }} />
                </div>
              )}
            </div>
            <span>
              {(item.baseP * 100).toFixed(0)}%{active ? `→${(item.ablP * 100).toFixed(0)}%` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function InlineScaleLab({ manifest }: { manifest: ModelManifest }) {
  const [idx, setIdx] = useState(0);
  const stop = SCALE_STOPS[idx];
  const ratio = Math.max(1, Math.round(stop.params / manifest.total_params));

  return (
    <div className="lab-card">
      <div className="lab-card-title">Logarithmic Scale Comparison</div>
      <div className="slider-row">
        <div className="row-between">
          <strong>{stop.label}</strong>
          <span className="mono">{ratio === 1 ? '1× (this)' : `${ratio.toLocaleString()}×`}</span>
        </div>
        <input
          type="range"
          min={0}
          max={SCALE_STOPS.length - 1}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
        />
      </div>
      <div className="muted-xs mono" style={{ marginBottom: 6 }}>
        {stop.params.toLocaleString()} params · {stop.layers} layers · d_model {stop.dModel.toLocaleString()} · {stop.heads} heads · ctx {stop.context.toLocaleString()}
      </div>
      <div className="muted-sm">{stop.note}</div>
    </div>
  );
}
