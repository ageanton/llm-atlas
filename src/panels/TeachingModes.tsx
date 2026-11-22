import { useMemo, useState } from 'react';
import type { TransformerEngineClient } from '../engine/client';
import type {
  AblationConfig,
  EngineTraceResult,
  ModelManifest,
  TokenizerData,
  TrainingReplayData,
} from '../engine/types';
import type { TeachingMode } from '../state/urlState';
import { buildParamSelection, hasActiveAblation } from '../state/urlState';
import { formatNum, formatToken, topIndices } from '../utils/color';
import { CanvasHeatmap } from '../scene/CanvasHeatmap';

export interface TeachingModePanelProps {
  mode: TeachingMode;
  onClose: () => void;
  ablation: AblationConfig;
  onChangeAblation: (next: AblationConfig) => void;
  baseline: EngineTraceResult | null;
  ablated: EngineTraceResult | null;
  replay: TrainingReplayData | null;
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  client: TransformerEngineClient | null;
  onInspectScalar: (sel: string) => void;
}

export function TeachingModePanel({
  mode,
  onClose,
  ablation,
  onChangeAblation,
  baseline,
  ablated,
  replay,
  manifest,
  tokenizer,
  client,
  onInspectScalar,
}: TeachingModePanelProps) {
  if (mode === 'explore') return null;

  return (
    <section className="teaching-mode-modal" role="dialog" aria-modal="true">
      <div className="mode-modal-header">
        <h2 className="mono">
          {mode === 'ablate' && 'Ablation / Counterfactual Mode'}
          {mode === 'replay' && 'Training Replay (Real Checkpoints)'}
          {mode === 'sampling' && 'Sampling Playground'}
          {mode === 'inspector' && 'Parameter Inspector'}
          {mode === 'scale' && 'Scale Ladder'}
        </h2>
        <button type="button" className="seg-btn" onClick={onClose}>
          Return to scene
        </button>
      </div>

      {mode === 'ablate' && (
        <AblationView
          baseline={baseline!}
          ablated={ablated}
          ablation={ablation}
          tokenizer={tokenizer}
          onChange={onChangeAblation}
        />
      )}

      {mode === 'replay' && (
        <TrainingReplayView
          replay={replay}
          onInspectScalar={onInspectScalar}
        />
      )}

      {mode === 'sampling' && (
        <SamplingView
          baseline={baseline!}
          tokenizer={tokenizer}
        />
      )}

      {mode === 'inspector' && (
        <ParameterInspectorView
          client={client!}
          manifest={manifest}
          onInspectScalar={onInspectScalar}
        />
      )}

      {mode === 'scale' && <ScaleLadderView manifest={manifest} />}
    </section>
  );
}

function AblationView({
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
  const current = ablated || baseline;

  const baseTop = useMemo(() => {
    const last = baseline.trace.logits[baseline.trace.logits.length - 1];
    return topIndices(last, 5).map((id) => ({
      id,
      token: tokenizer.id_to_token[id] ?? `<${id}>`,
      logit: last[id],
    }));
  }, [baseline, tokenizer]);

  const ablTop = useMemo(() => {
    const last = current.trace.logits[current.trace.logits.length - 1];
    return topIndices(last, 5).map((id) => ({
      id,
      token: tokenizer.id_to_token[id] ?? `<${id}>`,
      logit: last[id],
    }));
  }, [current, tokenizer]);

  const toggleBlock = (b: number) => {
    const curr = ablation.zeroBlocks || [];
    const next = curr.includes(b) ? curr.filter((x) => x !== b) : [...curr, b];
    onChange({ ...ablation, zeroBlocks: next });
  };

  const toggleMlp = (b: number) => {
    const curr = ablation.zeroMlpLayers || [];
    const next = curr.includes(b) ? curr.filter((x) => x !== b) : [...curr, b];
    onChange({ ...ablation, zeroMlpLayers: next });
  };

  const toggleHead = (layer: number, head: number) => {
    const curr = ablation.zeroHeads || [];
    const exists = curr.some(([l, h]) => l === layer && h === head);
    const next = exists
      ? curr.filter(([l, h]) => !(l === layer && h === head))
      : [...curr, [layer, head] as [number, number]];
    onChange({ ...ablation, zeroHeads: next });
  };

  const reset = () => onChange({});

  return (
    <div className="mode-content-body ablation-layout">
      <div className="ablation-controls-panel">
        <div className="ablation-panel-title">
          <h3>Knockout Toggles</h3>
          {active && (
            <button type="button" className="btn-inline" onClick={reset}>
              Reset all ablations
            </button>
          )}
        </div>
        <p className="card-subtext">
          Zero out specific layers or attention heads at runtime to test their causal contribution.
        </p>

        <div className="ablation-group">
          <span className="mono ab-group-label">Zero entire Transformer Block:</span>
          <div className="toggle-btn-row">
            {[0, 1, 2, 3].map((b) => {
              const on = (ablation.zeroBlocks || []).includes(b);
              return (
                <button
                  key={b}
                  type="button"
                  className={`toggle-btn mono ${on ? 'is-ablated' : ''}`}
                  onClick={() => toggleBlock(b)}
                >
                  Block {b} {on ? '(ZEROED)' : ''}
                </button>
              );
            })}
          </div>
        </div>

        <div className="ablation-group">
          <span className="mono ab-group-label">Zero MLP sublayer:</span>
          <div className="toggle-btn-row">
            {[0, 1, 2, 3].map((b) => {
              const on = (ablation.zeroMlpLayers || []).includes(b);
              return (
                <button
                  key={b}
                  type="button"
                  className={`toggle-btn mono ${on ? 'is-ablated' : ''}`}
                  onClick={() => toggleMlp(b)}
                >
                  MLP {b} {on ? '(ZEROED)' : ''}
                </button>
              );
            })}
          </div>
        </div>

        <div className="ablation-group">
          <span className="mono ab-group-label">Zero individual Attention Heads:</span>
          <div className="heads-ablation-grid">
            {[0, 1, 2, 3].map((l) => (
              <div key={l} className="head-row-wrap">
                <span className="mono head-layer-label">L{l}:</span>
                {[0, 1, 2, 3].map((h) => {
                  const on = (ablation.zeroHeads || []).some(
                    ([bl, bh]) => bl === l && bh === h
                  );
                  return (
                    <button
                      key={h}
                      type="button"
                      className={`head-toggle-btn mono ${on ? 'is-ablated' : ''}`}
                      onClick={() => toggleHead(l, h)}
                    >
                      H{h}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="ablation-diff-panel">
        <div className="diff-columns-grid">
          <div className="diff-col">
            <span className="badge-chip mono">Baseline Prediction</span>
            <div className="preds-list-box">
              {baseTop.map((p, idx) => (
                <div key={p.id} className="pred-row">
                  <span className="pred-rank mono">#{idx + 1}</span>
                  <span className="pred-tok mono">“{formatToken(p.token)}”</span>
                  <span className="pred-val mono">{formatNum(p.logit, 2)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="diff-col">
            <span className={`badge-chip mono ${active ? 'delta' : ''}`}>
              {active ? 'Ablated Prediction (Live)' : 'Ablated (None active)'}
            </span>
            <div className="preds-list-box">
              {ablTop.map((p, idx) => {
                const baseLogit = baseTop.find((b) => b.id === p.id)?.logit ?? 0;
                const diff = p.logit - baseLogit;
                return (
                  <div key={p.id} className="pred-row">
                    <span className="pred-rank mono">#{idx + 1}</span>
                    <span className="pred-tok mono">“{formatToken(p.token)}”</span>
                    <span className="pred-val mono">{formatNum(p.logit, 2)}</span>
                    {active && Math.abs(diff) > 0.01 && (
                      <span className={`diff-tag mono ${diff > 0 ? 'pos' : 'neg'}`}>
                        {diff > 0 ? `+${formatNum(diff, 2)}` : formatNum(diff, 2)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="ablation-summary-note">
          <p>
            Ablation directly zeroes the output tensors inside the forward pass before residual
            addition. Notice how zeroing Block 2 alters pronoun tracking and sentence grammar.
          </p>
        </div>
      </div>
    </div>
  );
}

function TrainingReplayView({
  replay,
  onInspectScalar,
}: {
  replay: TrainingReplayData | null;
  onInspectScalar: (sel: string) => void;
}) {
  const [stepIdx, setStepIdx] = useState(0);

  if (!replay || !replay.checkpoints || replay.checkpoints.length === 0) {
    return (
      <div className="mode-content-body">
        <div className="empty-notice">
          Training replay data is loading or was not found in public/model/training_replay.json.
        </div>
      </div>
    );
  }

  const checkpoints = replay.checkpoints;
  const current = checkpoints[Math.min(stepIdx, checkpoints.length - 1)];

  return (
    <div className="mode-content-body replay-layout">
      <div className="replay-timeline-card">
        <div className="timeline-header">
          <span className="badge-chip mono">Checkpoint {stepIdx + 1} / {checkpoints.length}</span>
          <span className="mono step-tag">Training Step: {current.step}</span>
          <span className="mono loss-tag">Loss: {formatNum(current.loss, 3)}</span>
        </div>

        <input
          type="range"
          min={0}
          max={checkpoints.length - 1}
          value={stepIdx}
          onChange={(e) => setStepIdx(Number.parseInt(e.target.value, 10))}
          className="timeline-slider"
        />

        <div className="slider-ticks mono">
          <span>Step 0 (Random babble)</span>
          <span>Step 500 (Punctuation)</span>
          <span>Step 1,000 (Local syntax)</span>
          <span>Step 1,500 (Coherent stories)</span>
        </div>
      </div>

      <div className="replay-details-grid">
        <div className="replay-sample-box">
          <h4 className="mono">Generated text at step {current.step}:</h4>
          <blockquote className="generated-story-text">
            “{current.sample || 'No sample recorded for this checkpoint'}”
          </blockquote>
        </div>

        <div className="replay-coords-box">
          <h4 className="mono">Logged Coordinates at step {current.step}:</h4>
          <div className="coords-table-wrap">
            <table className="numeric-table mono">
              <thead>
                <tr>
                  <th>Tensor Coordinate</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(current.coords || {}).map(([key, val]) => (
                  <tr key={key} onClick={() => onInspectScalar(`param:${key}`)}>
                    <td className="coord-key">{key}</td>
                    <td className="coord-val">{formatNum(val, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function SamplingView({
  baseline,
  tokenizer,
}: {
  baseline: EngineTraceResult;
  tokenizer: TokenizerData;
}) {
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(10);
  const [topP, setTopP] = useState(0.9);

  const rawLogits = baseline.trace.logits[baseline.trace.logits.length - 1];

  const processedProbs = useMemo(() => {
    const scaled = Array.from(rawLogits).map((l) => l / Math.max(0.05, temperature));
    let maxScaled = -Infinity;
    for (const v of scaled) if (v > maxScaled) maxScaled = v;

    const exps = scaled.map((v) => Math.exp(v - maxScaled));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const softmax = exps.map((e) => e / sumExp);

    const indexed = softmax.map((p, idx) => ({ id: idx, p }));
    indexed.sort((a, b) => b.p - a.p);

    const truncatedK = indexed.slice(0, Math.max(1, topK));

    let cum = 0;
    const truncatedP: { id: number; p: number }[] = [];
    for (const item of truncatedK) {
      truncatedP.push(item);
      cum += item.p;
      if (cum >= topP && truncatedP.length >= 1) break;
    }

    const renormSum = truncatedP.reduce((s, it) => s + it.p, 0);
    return truncatedP.map((it) => ({
      id: it.id,
      token: tokenizer.id_to_token[it.id] ?? `<${it.id}>`,
      prob: it.p / renormSum,
    }));
  }, [rawLogits, temperature, topK, topP, tokenizer]);

  return (
    <div className="mode-content-body sampling-layout">
      <div className="sampling-controls-card">
        <h3>Decoding Parameters</h3>

        <div className="control-slider-group">
          <div className="slider-label-row mono">
            <span>Temperature: {temperature.toFixed(2)}</span>
            <span className="control-desc">
              {temperature < 0.2 ? 'Greedy / Deterministic' : temperature > 1.2 ? 'High entropy / Creative' : 'Balanced'}
            </span>
          </div>
          <input
            type="range"
            min={0.05}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number.parseFloat(e.target.value))}
          />
        </div>

        <div className="control-slider-group">
          <div className="slider-label-row mono">
            <span>Top-k: {topK}</span>
            <span className="control-desc">Truncate to k highest logits</span>
          </div>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number.parseInt(e.target.value, 10))}
          />
        </div>

        <div className="control-slider-group">
          <div className="slider-label-row mono">
            <span>Top-p (Nucleus): {topP.toFixed(2)}</span>
            <span className="control-desc">Smallest set whose cumulative mass ≥ p</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={topP}
            onChange={(e) => setTopP(Number.parseFloat(e.target.value))}
          />
        </div>
      </div>

      <div className="sampling-distribution-card">
        <h3 className="mono">Effective Token Sampling Probabilities</h3>
        <div className="sampling-bars-list">
          {processedProbs.map((item, idx) => {
            const pct = (item.prob * 100).toFixed(1);
            return (
              <div key={item.id} className="sampling-bar-row">
                <span className="mono sample-rank">#{idx + 1}</span>
                <span className="mono sample-token">“{formatToken(item.token)}”</span>
                <div className="sampling-track">
                  <div className="sampling-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="mono sample-pct">{pct}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ParameterInspectorView({
  client,
  manifest,
  onInspectScalar,
}: {
  client: TransformerEngineClient;
  manifest: ModelManifest;
  onInspectScalar: (sel: string) => void;
}) {
  const [selectedName, setSelectedName] = useState(manifest.tensors[0]?.name || '');
  const [data, setData] = useState<Float32Array | null>(null);

  const meta = useMemo(() => {
    return manifest.tensors.find((t) => t.name === selectedName) || manifest.tensors[0];
  }, [manifest, selectedName]);

  useState(() => {
    if (!client || !meta) return;
    client.getTensor(meta.name).then((buf) => setData(buf));
  });

  const rows = meta.shape[0];
  const cols = meta.shape[1] || 1;

  const getValue = (r: number, c: number) => {
    if (!data) return 0;
    return data[r * cols + c] || 0;
  };

  return (
    <div className="mode-content-body inspector-layout">
      <div className="inspector-sidebar">
        {manifest.tensors.map((t) => (
          <button
            key={t.name}
            type="button"
            className={`inspector-item-btn mono ${t.name === meta.name ? 'is-active' : ''}`}
            onClick={() => {
              setSelectedName(t.name);
              client.getTensor(t.name).then((buf) => setData(buf));
            }}
          >
            <div className="inspector-row-top">
              <span className="inspector-name">{t.name}</span>
              <span className="inspector-shape">
                [{t.shape.join('×')}] ({t.count.toLocaleString()})
              </span>
            </div>
            <div className="inspector-row-role">{t.role}</div>
          </button>
        ))}
      </div>

      <div className="inspector-detail">
        <p className="param-role">{meta.role}</p>
        {data && (
          <CanvasHeatmap
            title={meta.name}
            rows={rows}
            cols={cols}
            getValue={getValue}
            mode="diverging"
            onCellClick={(r, c) =>
              onInspectScalar(buildParamSelection(meta.name, r, c))
            }
            height={230}
          />
        )}
        <div className="hist-axis mono">
          <span>min {formatNum(meta.stats.min, 4)}</span>
          <span>mean {formatNum(meta.stats.mean, 4)}</span>
          <span>std {formatNum(meta.stats.std, 4)}</span>
          <span>max {formatNum(meta.stats.max, 4)}</span>
        </div>
      </div>
    </div>
  );
}

function ScaleLadderView({ manifest }: { manifest: ModelManifest }) {
  const models = [
    {
      name: 'This Explainer Model',
      params: '866K',
      layers: 4,
      dim: 128,
      heads: 4,
      vocab: 2048,
      size: '1.65 MB',
      note: 'Runs in pure TypeScript browser worker with exact math.',
    },
    {
      name: 'GPT-2 Small (2019)',
      params: '124M',
      layers: 12,
      dim: 768,
      heads: 12,
      vocab: 50257,
      size: '500 MB',
      note: '143× larger. Identical Pre-LN Transformer architecture.',
    },
    {
      name: 'Llama 3 8B (2024)',
      params: '8.03B',
      layers: 32,
      dim: 4096,
      heads: 32,
      vocab: 128256,
      size: '16 GB',
      note: '9,268× larger. Uses RoPE, SwiGLU, and Grouped-Query Attention.',
    },
    {
      name: 'Llama 3 70B (2024)',
      params: '70.6B',
      layers: 80,
      dim: 8192,
      heads: 64,
      vocab: 128256,
      size: '140 GB',
      note: '81,495× larger. Multi-node GPU cluster required.',
    },
  ];

  return (
    <div className="mode-content-body scale-layout">
      <div className="scale-intro-card">
        <h3>Architectural Scaling Ladder</h3>
        <p>
          The 4-layer miniature transformer in your browser has the same fundamental building blocks
          (Token Embedding, Multi-Head Attention, Residual Stream, and MLP) as production LLMs.
          The difference is scale and parameter volume.
        </p>
      </div>

      <div className="scale-cards-grid">
        {models.map((m, i) => (
          <div key={m.name} className={`scale-model-card ${i === 0 ? 'is-self' : ''}`}>
            <div className="scale-card-header">
              <span className="scale-name mono">{m.name}</span>
              <span className="scale-params mono">{m.params}</span>
            </div>
            <dl className="mono scale-dl">
              <dt>Layers:</dt>
              <dd>{m.layers}</dd>
              <dt>Embedding Dim:</dt>
              <dd>{m.dim}</dd>
              <dt>Attention Heads:</dt>
              <dd>{m.heads}</dd>
              <dt>Vocabulary:</dt>
              <dd>{m.vocab.toLocaleString()}</dd>
              <dt>FP16 Size:</dt>
              <dd>{m.size}</dd>
            </dl>
            <p className="scale-note">{m.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
