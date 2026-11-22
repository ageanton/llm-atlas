import { useEffect, useMemo, useState } from 'react';
import { SCALE_LADDER_STOPS } from '../content/atlasContent';
import type { TransformerEngineClient } from '../engine/client';
import type {
  AblationConfig,
  EngineTraceResult,
  ModelManifest,
  TokenizerData,
} from '../engine/types';
import {
  buildParamSelection,
  hasActiveAblation,
  type TeachingMode,
} from '../state/urlState';
import {
  computeSampledDistribution,
  formatNum,
  formatToken,
  sampleFromDistribution,
} from '../utils/color';
import { CanvasHeatmap } from '../scene/CanvasHeatmap';

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
  trajectories?: Record<string, number[]>;
}

export interface TeachingModePanelProps {
  mode: TeachingMode;
  onClose: () => void;
  ablation: AblationConfig;
  onChangeAblation: (next: AblationConfig) => void;
  baseline: EngineTraceResult;
  ablated: EngineTraceResult | null;
  replay: TrainingReplayData | null;
  manifest: ModelManifest;
  tokenizer: TokenizerData;
  client: TransformerEngineClient;
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
          baseline={baseline}
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
          baseline={baseline}
          tokenizer={tokenizer}
        />
      )}

      {mode === 'inspector' && (
        <ParameterInspectorView
          client={client}
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

  const toggleHead = (l: number, h: number) => {
    const cur = ablation.disabledHeads ?? [];
    const exists = cur.some(([a, b]) => a === l && b === h);
    const next = exists
      ? cur.filter(([a, b]) => !(a === l && b === h))
      : [...cur, [l, h] as [number, number]];
    onChange({ ...ablation, disabledHeads: next });
  };

  const toggleNumList = (
    key: 'disabledAttnBlocks' | 'disabledMlpBlocks' | 'disabledLayers',
    l: number
  ) => {
    const cur = ablation[key] ?? [];
    const next = cur.includes(l) ? cur.filter((x) => x !== l) : [...cur, l];
    onChange({ ...ablation, [key]: next });
  };

  const toggleRes = (kind: 'attn' | 'mlp', l: number) => {
    const cur = ablation.disabledResiduals ?? [];
    const exists = cur.some(([k, idx]) => k === kind && idx === l);
    const next = exists
      ? cur.filter(([k, idx]) => !(k === kind && idx === l))
      : [...cur, [kind, l] as ['attn' | 'mlp', number]];
    onChange({ ...ablation, disabledResiduals: next });
  };

  const toggleLn = (kind: 'ln_1' | 'ln_2' | 'ln_f', l: number) => {
    const cur = ablation.disabledLayerNorms ?? [];
    const exists = cur.some(([k, idx]) => k === kind && idx === l);
    const next = exists
      ? cur.filter(([k, idx]) => !(k === kind && idx === l))
      : [...cur, [kind, l] as ['ln_1' | 'ln_2' | 'ln_f', number]];
    onChange({ ...ablation, disabledLayerNorms: next });
  };

  const compareIds = useMemo(() => {
    const idSet = new Set<number>();
    const topBase = baseline.trace.finalProbs
      .map((p, id) => ({ id, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 6);
    topBase.forEach((x) => idSet.add(x.id));
    if (ablated) {
      ablated.trace.finalProbs
        .map((p, id) => ({ id, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 4)
        .forEach((x) => idSet.add(x.id));
    }
    return Array.from(idSet)
      .map((id) => ({
        id,
        text: tokenizer.vocab[id] ?? `<${id}>`,
        baseProb: baseline.trace.finalProbs[id] ?? 0,
        ablProb: ablated ? ablated.trace.finalProbs[id] ?? 0 : null,
      }))
      .sort((a, b) => (b.ablProb ?? b.baseProb) - (a.ablProb ?? a.baseProb));
  }, [baseline, ablated, tokenizer]);

  return (
    <div className="mode-body ablation-grid">
      <div className="ablation-controls">
        <h3>Intervention Toggles</h3>
        <p className="card-subtext">
          Zero specific components during inference to observe causal effects on prediction:
        </p>

        <div className="toggle-group">
          <span className="mono group-title">Attention Heads (disable Q·Kᵀ):</span>
          <div className="head-toggles-grid">
            {[0, 1, 2, 3].map((l) => (
              <div key={l} className="head-row">
                <span className="mono l-tag">L{l}:</span>
                {[0, 1, 2, 3].map((h) => {
                  const on = (ablation.disabledHeads ?? []).some(
                    ([bl, bh]) => bl === l && bh === h
                  );
                  return (
                    <button
                      key={h}
                      type="button"
                      className={`head-btn mono ${on ? 'is-ablated' : ''}`}
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

        <div className="toggle-group">
          <span className="mono group-title">Sublayers:</span>
          <div className="sublayer-toggles">
            {[0, 1, 2, 3].map((l) => (
              <div key={l} className="sublayer-row">
                <span className="mono l-tag">L{l}:</span>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledAttnBlocks ?? []).includes(l) ? 'is-active' : ''}`}
                  onClick={() => toggleNumList('disabledAttnBlocks', l)}
                >
                  Attn {l}
                </button>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledMlpBlocks ?? []).includes(l) ? 'is-active' : ''}`}
                  onClick={() => toggleNumList('disabledMlpBlocks', l)}
                >
                  MLP {l}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="toggle-group">
          <span className="mono group-title">Residual Skips & LayerNorms:</span>
          <div className="res-ln-toggles">
            {[0, 1, 2, 3].map((l) => (
              <div key={l} className="sublayer-row">
                <span className="mono l-tag">L{l}:</span>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledResiduals ?? []).some(([k, idx]) => k === 'attn' && idx === l) ? 'is-active' : ''}`}
                  onClick={() => toggleRes('attn', l)}
                >
                  Skip Attn
                </button>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledResiduals ?? []).some(([k, idx]) => k === 'mlp' && idx === l) ? 'is-active' : ''}`}
                  onClick={() => toggleRes('mlp', l)}
                >
                  Skip MLP
                </button>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledLayerNorms ?? []).some(([k, idx]) => k === 'ln_1' && idx === l) ? 'is-active' : ''}`}
                  onClick={() => toggleLn('ln_1', l)}
                >
                  LN1
                </button>
                <button
                  type="button"
                  className={`seg-btn mono ${(ablation.disabledLayerNorms ?? []).some(([k, idx]) => k === 'ln_2' && idx === l) ? 'is-active' : ''}`}
                  onClick={() => toggleLn('ln_2', l)}
                >
                  LN2
                </button>
              </div>
            ))}
          </div>
        </div>

        {active && (
          <button
            type="button"
            className="seg-btn"
            style={{ marginTop: 12 }}
            onClick={() => onChange({})}
          >
            Reset all ablations
          </button>
        )}
      </div>

      <div className="ablation-results">
        <h3>Live Output Probability Comparison</h3>
        <div className="comparison-list">
          {compareIds.map((item) => {
            const basePct = (item.baseProb * 100).toFixed(1);
            const ablPct = item.ablProb !== null ? (item.ablProb * 100).toFixed(1) : null;
            return (
              <div key={item.id} className="comp-row">
                <span className="comp-tok mono">“{formatToken(item.text)}”</span>
                <div className="comp-tracks">
                  <div className="track-row">
                    <span className="track-label mono">Base:</span>
                    <div className="bar-track">
                      <div className="bar-fill base" style={{ width: `${basePct}%` }} />
                    </div>
                    <span className="track-pct mono">{basePct}%</span>
                  </div>

                  {ablPct !== null && (
                    <div className="track-row">
                      <span className="track-label mono">Ablated:</span>
                      <div className="bar-track">
                        <div className="bar-fill ablated" style={{ width: `${ablPct}%` }} />
                      </div>
                      <span className="track-pct mono">{ablPct}%</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
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
      <div className="mode-body">
        <div className="empty-notice">
          Training replay data loading or not found in public/model/training_replay.json.
        </div>
      </div>
    );
  }

  const checkpoints = replay.checkpoints;
  const current = checkpoints[Math.min(stepIdx, checkpoints.length - 1)];

  return (
    <div className="mode-body replay-layout">
      <div className="replay-timeline">
        <div className="timeline-header">
          <span className="badge-chip mono">Checkpoint {stepIdx + 1} / {checkpoints.length}</span>
          <span className="mono step-tag">Step: {current.step}</span>
          <span className="mono loss-tag">
            Train Loss: {current.train_loss?.toFixed(3) ?? '—'} · Val Loss: {current.val_loss?.toFixed(3) ?? '—'}
          </span>
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

      <div className="replay-grid">
        <div className="replay-prompts-card">
          <h4>Sample Generations at Step {current.step}:</h4>
          <div className="prompts-list">
            {(current.prompts ?? []).map((p, idx) => (
              <div key={idx} className="prompt-sample-item">
                <div className="prompt-prompt mono">“{p.prompt}”</div>
                <div className="prompt-completion">→ “{p.completion}”</div>
              </div>
            ))}
          </div>
        </div>

        <div className="replay-scalars-card">
          <h4>Tracked Weight Coordinates at Step {current.step}:</h4>
          <div className="coords-table-wrap">
            <table className="coords-table mono">
              <thead>
                <tr>
                  <th>Coordinate</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(current.scalars ?? {}).map(([key, val]) => (
                  <tr key={key} onClick={() => onInspectScalar(`param:${key}`)}>
                    <td className="coord-key">{key}</td>
                    <td className="coord-val">{val.toFixed(4)}</td>
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
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(10);
  const [topP, setTopP] = useState(0.9);
  const [sampledWord, setSampledWord] = useState<string | null>(null);

  const dist = useMemo(() => {
    return computeSampledDistribution(
      baseline.trace.finalLogits,
      temperature,
      topK,
      topP
    );
  }, [baseline.trace.finalLogits, temperature, topK, topP]);

  const handleRoll = () => {
    if (dist.length > 0) {
      const id = sampleFromDistribution(dist);
      setSampledWord(tokenizer.vocab[id] ?? `<${id}>`);
    }
  };

  return (
    <div className="mode-body sampling-layout">
      <div className="sampling-controls">
        <h3>Decoding Parameters</h3>

        <div className="slider-group">
          <div className="slider-header mono">
            <span>Temperature: {temperature.toFixed(2)}</span>
            <span className="slider-sub">
              {temperature < 0.3 ? 'Deterministic' : temperature > 1.2 ? 'Creative/Noisy' : 'Balanced'}
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

        <div className="slider-group">
          <div className="slider-header mono">
            <span>Top-K: {topK}</span>
            <span className="slider-sub">Truncate to k highest logits</span>
          </div>
          <input
            type="range"
            min={1}
            max={40}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number.parseInt(e.target.value, 10))}
          />
        </div>

        <div className="slider-group">
          <div className="slider-header mono">
            <span>Top-P (Nucleus): {topP.toFixed(2)}</span>
            <span className="slider-sub">Smallest set with cum prob ≥ p</span>
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

        <div className="roll-action-box">
          <button type="button" className="seg-btn is-active" onClick={handleRoll}>
            🎲 Roll Random Token
          </button>
          {sampledWord && (
            <div className="roll-result mono">
              Sampled: <b>“{formatToken(sampledWord)}”</b>
            </div>
          )}
        </div>
      </div>

      <div className="sampling-dist">
        <h3>Effective Sampling Distribution ({dist.length} candidate tokens)</h3>
        <div className="dist-list">
          {dist.map((item, idx) => {
            const word = tokenizer.vocab[item.id] ?? `<${item.id}>`;
            const pct = (item.prob * 100).toFixed(1);
            return (
              <div key={item.id} className="dist-row">
                <span className="dist-rank mono">#{idx + 1}</span>
                <span className="dist-word mono">“{formatToken(word)}”</span>
                <div className="dist-track">
                  <div className="dist-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="dist-pct mono">{pct}%</span>
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
  const [selectedName, setSelectedName] = useState(manifest.tensors[0]?.name ?? '');
  const [data, setData] = useState<Float32Array | null>(null);

  const meta = useMemo(
    () => manifest.tensors.find((t) => t.name === selectedName) ?? manifest.tensors[0],
    [manifest, selectedName]
  );

  useEffect(() => {
    let alive = true;
    void client.getTensor(meta.name).then((buf) => {
      if (alive) setData(buf);
    });
    return () => {
      alive = false;
    };
  }, [client, meta.name]);

  const rows = meta.shape[0];
  const cols = meta.shape[1] ?? 1;

  const getValue = (r: number, c: number) => {
    if (!data) return 0;
    return data[r * cols + c] ?? 0;
  };

  return (
    <div className="mode-body inspector-layout">
      <div className="inspector-sidebar">
        {manifest.tensors.map((t) => (
          <button
            key={t.name}
            type="button"
            className={`inspector-item mono ${t.name === meta.name ? 'is-active' : ''}`}
            onClick={() => setSelectedName(t.name)}
          >
            <div className="item-name">{t.name}</div>
            <div className="item-shape">[{t.shape.join(' × ')}]</div>
          </button>
        ))}
      </div>

      <div className="inspector-main">
        <p className="param-role">{meta.role}</p>
        {data && (
          <CanvasHeatmap
            title={meta.name}
            rows={rows}
            cols={cols}
            getValue={getValue}
            mode="diverging"
            onCellClick={(r, c) => onInspectScalar(buildParamSelection(meta.name, r, c))}
            height={260}
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
  const [idx, setIdx] = useState(0);
  const stop = SCALE_LADDER_STOPS[idx];

  return (
    <div className="mode-body scale-layout">
      <div className="scale-intro">
        <h3>Architectural Scaling Ladder</h3>
        <p>
          Compare this {manifest.total_params.toLocaleString()}-parameter educational model against GPT-2, Llama 3, and Frontier LLMs. The foundational math (Q, K, V attention, LayerNorm, residual highway, and feed-forward MLP) is identical across 6 orders of magnitude.
        </p>
      </div>

      <div className="scale-slider-card">
        <div className="scale-header mono">
          <span className="scale-title">{stop.name}</span>
          <span className="scale-params">{stop.params} parameters</span>
        </div>

        <input
          type="range"
          min={0}
          max={SCALE_LADDER_STOPS.length - 1}
          value={idx}
          onChange={(e) => setIdx(Number.parseInt(e.target.value, 10))}
          className="scale-slider"
        />

        <div className="scale-ticks mono">
          {SCALE_LADDER_STOPS.map((s, i) => (
            <span
              key={s.name}
              className={i === idx ? 'is-active' : ''}
              onClick={() => setIdx(i)}
            >
              {s.params}
            </span>
          ))}
        </div>

        <dl className="mono scale-dl">
          <dt>Layers:</dt>
          <dd>{stop.layers}</dd>
          <dt>Hidden Dim (d_model):</dt>
          <dd>{stop.dModel}</dd>
          <dt>Attention Heads:</dt>
          <dd>{stop.heads}</dd>
          <dt>Context Window:</dt>
          <dd>{stop.context.toLocaleString()} tokens</dd>
        </dl>

        <p className="scale-note">{stop.note}</p>
      </div>
    </div>
  );
}
