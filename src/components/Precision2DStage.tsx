import { useEffect, useMemo, useState } from 'react';
import type { SceneBeatId } from '../content/bookData';
import type { ActiveLayerTrace, EngineTraceResult } from '../engine/types';
import {
  divergingColor,
  formatNum,
  formatToken,
  formatTokenExplicit,
} from '../utils/math';
import type { BpeStepSnapshot } from '../utils/pass2Math';
import { countActiveNeurons, geluTanh } from '../utils/pass3Math';

const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

function geluDerivative(x: number): number {
  const u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
  const t = Math.tanh(u);
  const du = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

export interface Precision2DStageProps {
  beat: SceneBeatId;
  layer: number;
  queryIdx: number;
  traceRes: EngineTraceResult;
  bpeSteps: BpeStepSnapshot[];
  bpeStepIdx: number;
  bpePlaying: boolean;
  onSetBpeStep: (step: number) => void;
  onToggleBpePlay: () => void;
  geluProbeX: number;
  onChangeGeluX: (x: number) => void;
  onSelectToken: (idx: number) => void;
}

export function Precision2DStage(props: Precision2DStageProps) {
  return (
    <div className="p2d-overlay" role="region" aria-label="Precision Interactive Lab">
      <div className="p2d-inner">
        {props.beat === 'ch1-2-bpe-merges' && (
          <BpeWelderLab
            steps={props.bpeSteps}
            stepIdx={props.bpeStepIdx}
            playing={props.bpePlaying}
            onSetStep={props.onSetBpeStep}
            onTogglePlay={props.onToggleBpePlay}
          />
        )}
        {props.beat === 'ch5-3-gelu' && (
          <GeluNeuronLab
            layer={props.layer}
            queryIdx={props.queryIdx}
            traceRes={props.traceRes}
            probeX={props.geluProbeX}
            onChangeX={props.onChangeGeluX}
            onSelectToken={props.onSelectToken}
          />
        )}
        {props.beat === 'ch7-3-manual-backprop' && <ChainRuleBackpropLab />}
        {props.beat === 'ch8-4-post-training' && <PostTrainingSimulator />}
      </div>
    </div>
  );
}

/* ============================================================================
 * 1. CHAPTER 1 BEAT 2: PRECISION 2D BPE TILE WELDER
 * ========================================================================== */
function BpeWelderLab({
  steps,
  stepIdx,
  playing,
  onSetStep,
  onTogglePlay,
}: {
  steps: BpeStepSnapshot[];
  stepIdx: number;
  playing: boolean;
  onSetStep: (s: number) => void;
  onTogglePlay: () => void;
}) {
  const maxStep = Math.max(0, steps.length - 1);
  const cur = steps[Math.min(stepIdx, maxStep)];
  const initialCount = steps[0]?.pieces.length ?? 1;
  if (!cur) return null;

  const compression = initialCount / Math.max(1, cur.pieces.length);
  const pctSaved = Math.round((1 - cur.pieces.length / Math.max(1, initialCount)) * 100);

  return (
    <div className="p2d-card">
      <div className="p2d-header">
        <div>
          <div className="p2d-kicker">Live Byte-Pair Encoding Replay</div>
          <h2 className="p2d-title">
            Step {cur.step} of {maxStep} — {cur.pieces.length} tokens ({pctSaved}% compressed, {compression.toFixed(2)}×)
          </h2>
        </div>
        <div className="p2d-controls">
          <button
            type="button"
            className={`mini-btn ${playing ? 'is-active' : ''}`}
            onClick={onTogglePlay}
          >
            {playing ? 'Pause' : 'Play Merges'}
          </button>
          <button
            type="button"
            className="mini-btn"
            disabled={stepIdx <= 0}
            onClick={() => onSetStep(Math.max(0, stepIdx - 1))}
          >
            −1
          </button>
          <button
            type="button"
            className="mini-btn"
            disabled={stepIdx >= maxStep}
            onClick={() => onSetStep(Math.min(maxStep, stepIdx + 1))}
          >
            +1
          </button>
        </div>
      </div>

      <div className="bpe-rule-banner mono">
        {cur.mergedPair ? (
          <>
            <span className="bpe-rank-pill">Rule #{cur.mergeRank}</span>
            <span>
              Welded <strong>“{formatTokenExplicit(cur.mergedPair[0])}”</strong> +{' '}
              <strong>“{formatTokenExplicit(cur.mergedPair[1])}”</strong> →{' '}
              <strong className="bpe-result-hl">“{formatTokenExplicit(cur.mergedResult ?? '')}”</strong>
            </span>
          </>
        ) : (
          <span>Step 0: Starting from raw individual characters ({initialCount} base symbols). Press Play Merges or scrub below.</span>
        )}
      </div>

      <div className="bpe-scrubber-row">
        <input
          type="range"
          min={0}
          max={maxStep}
          value={Math.min(stepIdx, maxStep)}
          onChange={(e) => onSetStep(Number(e.target.value))}
          aria-label="BPE merge step"
        />
      </div>

      <div className="bpe-tile-field" aria-live="polite">
        {cur.pieces.map((piece, idx) => {
          const isFresh = cur.mergedResult !== null && piece === cur.mergedResult;
          const isMerged = piece.length > 1;
          return (
            <div
              key={`${idx}-${piece}`}
              className={`bpe-tile mono ${isFresh ? 'is-fresh' : isMerged ? 'is-merged' : ''}`}
            >
              <span className="bpe-tile-text">{formatTokenExplicit(piece)}</span>
              <span className="bpe-tile-len">{piece.length}c</span>
            </div>
          );
        })}
      </div>

      <div className="p2d-footnote">
        <span>Leading dots (<code>·</code>) mark spaces glued onto tokens.</span>
        <span>Warm amber tiles were welded on this exact step.</span>
      </div>
    </div>
  );
}

/* ============================================================================
 * 2. CHAPTER 5 BEAT 3: GELU TANGENT PLOT + CLICKABLE 512-NEURON INSPECTOR
 * ========================================================================== */
function GeluNeuronLab({
  layer,
  queryIdx,
  traceRes,
  probeX,
  onChangeX,
  onSelectToken,
}: {
  layer: number;
  queryIdx: number;
  traceRes: EngineTraceResult;
  probeX: number;
  onChangeX: (x: number) => void;
  onSelectToken: (idx: number) => void;
}) {
  const [showRefs, setShowRefs] = useState(true);
  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(null);

  const lt = traceRes.trace.layers[layer];
  const active: ActiveLayerTrace | null = lt.skipped ? null : lt;
  const hiddenRow = active ? active.mlp.hidden[queryIdx] : null;

  // Pick the strongest firing neuron on the current token as smart default
  const defaultNeuron = useMemo(() => {
    if (!hiddenRow) return 0;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < hiddenRow.length; i++) {
      if (hiddenRow[i] > bestVal) {
        bestVal = hiddenRow[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }, [hiddenRow]);

  const activeNeuronIdx = selectedNeuron ?? defaultNeuron;

  // SVG coordinates for x in [-3.5, 3.5], y in [-1.2, 3.6]
  const W = 400;
  const H = 250;
  const pad = 28;
  const toSvgX = (x: number) => pad + ((x + 3.5) / 7.0) * (W - pad * 2);
  const toSvgY = (y: number) => H - pad - ((y + 1.2) / 4.8) * (H - pad * 2);

  const geluPath = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i <= 80; i++) {
      const x = -3.5 + (i / 80) * 7.0;
      const y = geluTanh(x);
      pts.push(`${i === 0 ? 'M' : 'L'} ${toSvgX(x).toFixed(1)} ${toSvgY(y).toFixed(1)}`);
    }
    return pts.join(' ');
  }, []);

  const probeY = geluTanh(probeX);
  const slope = geluDerivative(probeX);
  const tanDx = 1.15;
  const tx1 = probeX - tanDx;
  const ty1 = probeY - slope * tanDx;
  const tx2 = probeX + tanDx;
  const ty2 = probeY + slope * tanDx;

  const maxRowAbs = useMemo(() => {
    if (!hiddenRow) return 1;
    let m = 1e-5;
    for (const v of hiddenRow) m = Math.max(m, Math.abs(v));
    return m;
  }, [hiddenRow]);

  // Cross-token activation profile for the clicked neuron across all T tokens
  const neuronAcrossTokens = useMemo(() => {
    if (!active) return [];
    return traceRes.tokens.map((tok, t) => ({
      t,
      text: formatToken(tok.text),
      val: active.mlp.hidden[t][activeNeuronIdx] ?? 0,
    }));
  }, [active, traceRes.tokens, activeNeuronIdx]);

  const maxTokenNeuronAbs = useMemo(() => {
    let m = 0.25;
    for (const item of neuronAcrossTokens) m = Math.max(m, Math.abs(item.val));
    return m;
  }, [neuronAcrossTokens]);

  const firingCount = hiddenRow ? countActiveNeurons(hiddenRow, 0.05) : 0;

  return (
    <div className="p2d-split-2">
      {/* Left: Exact 2D GELU Curve & Live Derivative Tangent */}
      <div className="p2d-card">
        <div className="p2d-header">
          <div>
            <div className="p2d-kicker">Non-Linearity & Local Gradient</div>
            <h2 className="p2d-title">GELU(x) & Derivative Tangent</h2>
          </div>
          <button
            type="button"
            className={`mini-btn ${showRefs ? 'is-active' : ''}`}
            onClick={() => setShowRefs((v) => !v)}
          >
            {showRefs ? 'Hide ReLU / Linear' : 'Compare ReLU / Linear'}
          </button>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="gelu-svg" role="img" aria-label="Interactive GELU curve with derivative tangent">
          {/* Grid axes */}
          <line x1={pad} y1={toSvgY(0)} x2={W - pad} y2={toSvgY(0)} stroke="#cbd5e1" strokeWidth="1.5" />
          <line x1={toSvgX(0)} y1={pad} x2={toSvgX(0)} y2={H - pad} stroke="#cbd5e1" strokeWidth="1.5" />

          {/* Optional Linear & ReLU comparisons */}
          {showRefs && (
            <>
              <line
                x1={toSvgX(-1.2)}
                y1={toSvgY(-1.2)}
                x2={toSvgX(3.5)}
                y2={toSvgY(3.5)}
                stroke="#94a3b8"
                strokeWidth="1.2"
                strokeDasharray="4 4"
              />
              <polyline
                points={`${toSvgX(-3.5)},${toSvgY(0)} ${toSvgX(0)},${toSvgY(0)} ${toSvgX(3.5)},${toSvgY(3.5)}`}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="1.4"
                strokeDasharray="3 3"
                opacity="0.7"
              />
            </>
          )}

          {/* GELU curve */}
          <path d={geluPath} fill="none" stroke="#1d4ed8" strokeWidth="3" />

          {/* Tangent line showing GELU'(x) */}
          <line
            x1={toSvgX(tx1)}
            y1={toSvgY(ty1)}
            x2={toSvgX(tx2)}
            y2={toSvgY(ty2)}
            stroke="#0f766e"
            strokeWidth="2.2"
          />

          {/* Probe dot */}
          <circle
            cx={toSvgX(probeX)}
            cy={toSvgY(probeY)}
            r="6.5"
            fill="#d97706"
            stroke="#ffffff"
            strokeWidth="2"
          />

          <text x={W - pad - 4} y={toSvgY(0) - 6} textAnchor="end" fontSize="10" fill="#64748b">x</text>
          <text x={toSvgX(0) + 6} y={pad + 8} fontSize="10" fill="#64748b">GELU(x)</text>
        </svg>

        <div className="bpe-scrubber-row">
          <div className="p2d-metric-row mono">
            <span>x = <strong>{probeX.toFixed(2)}</strong></span>
            <span>GELU(x) = <strong>{probeY.toFixed(3)}</strong></span>
            <span style={{ color: '#0f766e' }}>Slope GELU'(x) = <strong>{slope.toFixed(3)}</strong></span>
          </div>
          <input
            type="range"
            min={-3.2}
            max={3.2}
            step={0.05}
            value={probeX}
            onChange={(e) => onChangeX(Number(e.target.value))}
            aria-label="GELU input x"
          />
        </div>
        <div className="p2d-footnote">
          <span>Green line = exact derivative backprop multiplies by. Notice the slight negative dip near x = −0.75!</span>
        </div>
      </div>

      {/* Right: All 512 Real Neurons + Cross-Token Firing Profile */}
      <div className="p2d-card">
        <div className="p2d-header">
          <div>
            <div className="p2d-kicker">Block {layer} Real MLP Activations</div>
            <h2 className="p2d-title">
              512 Neurons ({firingCount} firing &gt; 0.05)
            </h2>
          </div>
          <span className="bpe-rank-pill mono">Neuron #{activeNeuronIdx}</span>
        </div>

        {hiddenRow ? (
          <>
            <div className="neuron-grid-512" role="grid" aria-label="512 GELU neurons">
              {hiddenRow.map((val, idx) => (
                <button
                  key={idx}
                  type="button"
                  className={`neuron-cell ${idx === activeNeuronIdx ? 'is-picked' : ''}`}
                  style={{ backgroundColor: divergingColor(val / maxRowAbs) }}
                  title={`Neuron #${idx}: ${val.toFixed(4)}`}
                  onClick={() => setSelectedNeuron(idx)}
                />
              ))}
            </div>

            <div className="cross-token-box">
              <div className="p2d-kicker" style={{ marginBottom: 6 }}>
                Where Neuron #{activeNeuronIdx} Fires Across Your Sentence (click any bar)
              </div>
              <div className="cross-token-bars">
                {neuronAcrossTokens.map((item) => {
                  const pct = Math.max(4, (Math.abs(item.val) / maxTokenNeuronAbs) * 100);
                  const isPos = item.val >= 0;
                  return (
                    <button
                      key={item.t}
                      type="button"
                      className={`ct-col ${item.t === queryIdx ? 'is-current' : ''}`}
                      onClick={() => onSelectToken(item.t)}
                      title={`Slot ${item.t} "${item.text}": ${item.val.toFixed(4)}`}
                    >
                      <div className="ct-track">
                        <div
                          className={`ct-fill ${isPos ? 'is-pos' : 'is-neg'}`}
                          style={{ height: `${pct}%` }}
                        />
                      </div>
                      <span className="ct-lbl mono">{item.text.slice(0, 4)}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </>
        ) : (
          <div className="p2d-footnote">Block {layer} is currently disabled by Ablation.</div>
        )}
      </div>
    </div>
  );
}

/* ============================================================================
 * 3. CHAPTER 7 BEAT 3: 2-LAYER CHAIN-RULE BACKPROPAGATION MINI-CIRCUIT
 * ========================================================================== */
function ChainRuleBackpropLab() {
  const x = 1.5;
  const targetY = 1.0;
  const [w1, setW1] = useState(0.35);
  const [w2, setW2] = useState(0.6);
  const [pulseStage, setPulseStage] = useState<number>(0); // 0=idle, 1=Loss, 2=w2/h, 3=GELU', 4=w1
  const [lossHistory, setLossHistory] = useState<number[]>([]);

  useEffect(() => {
    if (pulseStage === 0) return;
    const t = window.setTimeout(() => {
      setPulseStage((s) => (s >= 4 ? 0 : s + 1));
    }, 420);
    return () => window.clearTimeout(t);
  }, [pulseStage]);

  // Exact forward pass
  const z = w1 * x;
  const h = geluTanh(z);
  const yHat = w2 * h;
  const loss = 0.5 * (yHat - targetY) * (yHat - targetY);

  // Exact backward pass (chain rule)
  const dL_dyHat = yHat - targetY;
  const dL_dw2 = dL_dyHat * h;
  const dL_dh = dL_dyHat * w2;
  const gPrime = geluDerivative(z);
  const dL_dz = dL_dh * gPrime;
  const dL_dw1 = dL_dz * x;

  const takeGradientStep = () => {
    const lr = 0.2;
    setLossHistory((prev) => [...prev.slice(-14), loss]);
    setW1((v) => Math.max(-3, Math.min(3, v - lr * dL_dw1)));
    setW2((v) => Math.max(-3, Math.min(3, v - lr * dL_dw2)));
  };

  return (
    <div className="p2d-card">
      <div className="p2d-header">
        <div>
          <div className="p2d-kicker">How Error Reaches Earlier Layers</div>
          <h2 className="p2d-title">2-Layer Chain-Rule Backpropagation Circuit</h2>
        </div>
        <div className="p2d-controls">
          <button
            type="button"
            className={`mini-btn ${pulseStage > 0 ? 'is-warm' : ''}`}
            onClick={() => setPulseStage(1)}
          >
            {pulseStage > 0 ? 'Pulsing ←' : 'Pulse Backprop ←'}
          </button>
          <button type="button" className="primary-btn" onClick={takeGradientStep}>
            Take 1 Descent Step
          </button>
          <button
            type="button"
            className="mini-btn"
            onClick={() => {
              setW1(-2.2);
              setW2(0.8);
              setLossHistory([]);
              setPulseStage(1);
            }}
          >
            Test Vanishing Grad
          </button>
          <button
            type="button"
            className="mini-btn"
            onClick={() => {
              setW1(0.35);
              setW2(0.6);
              setLossHistory([]);
              setPulseStage(0);
            }}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Crisp SVG Computational Graph showing Forward (blue) and Backward Chain Rule (crimson) */}
      <svg viewBox="0 0 760 195" className="circuit-svg" role="img" aria-label="2-layer backpropagation chain rule diagram">
        {/* Forward wires */}
        <line x1="95" y1="78" x2="215" y2="78" stroke="#93c5fd" strokeWidth="3" />
        <line x1="305" y1="78" x2="395" y2="78" stroke="#93c5fd" strokeWidth="3" />
        <line x1="485" y1="78" x2="575" y2="78" stroke="#93c5fd" strokeWidth="3" />
        <line x1="665" y1="78" x2="705" y2="78" stroke="#93c5fd" strokeWidth="3" />

        {/* Backward gradient wires */}
        <line x1="705" y1="120" x2="575" y2="120" stroke={pulseStage >= 1 ? '#dc2626' : '#fecaca'} strokeWidth={pulseStage === 1 ? 4.5 : 2.5} strokeDasharray="5 3" />
        <line x1="575" y1="120" x2="395" y2="120" stroke={pulseStage >= 2 ? '#dc2626' : '#fecaca'} strokeWidth={pulseStage === 2 ? 4.5 : 2.5} strokeDasharray="5 3" />
        <line x1="395" y1="120" x2="215" y2="120" stroke={pulseStage >= 3 ? '#dc2626' : '#fecaca'} strokeWidth={pulseStage === 3 ? 4.5 : 2.5} strokeDasharray="5 3" />
        <line x1="215" y1="120" x2="95" y2="120" stroke={pulseStage >= 4 ? '#dc2626' : '#fecaca'} strokeWidth={pulseStage === 4 ? 4.5 : 2.5} strokeDasharray="5 3" />

        {/* Node 1: Input x */}
        <rect x="16" y="48" width="84" height="60" rx="10" fill="#eff6ff" stroke="#1d4ed8" strokeWidth="2" />
        <text x="58" y="72" textAnchor="middle" fontSize="11" fontWeight="700" fill="#1e3a8a">Input x</text>
        <text x="58" y="92" textAnchor="middle" fontSize="13" fontFamily="monospace" fill="#1d4ed8">{x.toFixed(1)}</text>

        {/* Weight w1 pill */}
        <rect x="120" y="56" width="74" height="44" rx="8" fill={pulseStage === 4 ? '#fef2f2' : '#ffffff'} stroke={pulseStage === 4 ? '#dc2626' : '#cbd5e1'} strokeWidth="2" />
        <text x="157" y="74" textAnchor="middle" fontSize="11" fontWeight="700" fill="#111827">× w₁</text>
        <text x="157" y="90" textAnchor="middle" fontSize="12" fontFamily="monospace" fill="#1d4ed8">{w1.toFixed(2)}</text>

        {/* Node 2: GELU(z) */}
        <rect x="215" y="48" width="110" height="60" rx="10" fill={pulseStage === 3 ? '#fff7ed' : '#f8fafc'} stroke={pulseStage === 3 ? '#d97706' : '#64748b'} strokeWidth="2" />
        <text x="270" y="70" textAnchor="middle" fontSize="11" fontWeight="700" fill="#111827">GELU(z={z.toFixed(2)})</text>
        <text x="270" y="90" textAnchor="middle" fontSize="12" fontFamily="monospace" fill="#0f766e">h = {h.toFixed(3)}</text>

        {/* Weight w2 pill */}
        <rect x="355" y="56" width="74" height="44" rx="8" fill={pulseStage === 2 ? '#fef2f2' : '#ffffff'} stroke={pulseStage === 2 ? '#dc2626' : '#cbd5e1'} strokeWidth="2" />
        <text x="392" y="74" textAnchor="middle" fontSize="11" fontWeight="700" fill="#111827">× w₂</text>
        <text x="392" y="90" textAnchor="middle" fontSize="12" fontFamily="monospace" fill="#1d4ed8">{w2.toFixed(2)}</text>

        {/* Node 3: Output yHat */}
        <rect x="460" y="48" width="104" height="60" rx="10" fill="#f0fdfa" stroke="#0f766e" strokeWidth="2" />
        <text x="512" y="70" textAnchor="middle" fontSize="11" fontWeight="700" fill="#134e4a">Pred ŷ (y=1)</text>
        <text x="512" y="90" textAnchor="middle" fontSize="13" fontFamily="monospace" fill="#0f766e">{yHat.toFixed(3)}</text>

        {/* Node 4: Loss */}
        <rect x="615" y="48" width="128" height="60" rx="10" fill={pulseStage === 1 ? '#fef2f2' : '#fff7ed'} stroke={pulseStage === 1 ? '#dc2626' : '#d97706'} strokeWidth="2" />
        <text x="679" y="70" textAnchor="middle" fontSize="11" fontWeight="700" fill="#9a3412">Loss ½(ŷ−1)²</text>
        <text x="679" y="91" textAnchor="middle" fontSize="14" fontFamily="monospace" fontWeight="700" fill="#b45309">{loss.toFixed(4)}</text>

        {/* Backward gradient labels along the bottom crimson path */}
        <text x="612" y="144" textAnchor="middle" fontSize="11" fontFamily="monospace" fill="#b91c1c">∂L/∂ŷ = {formatNum(dL_dyHat, 3)}</text>
        <text x="430" y="144" textAnchor="middle" fontSize="11" fontFamily="monospace" fill="#b91c1c">∂L/∂w₂ = {formatNum(dL_dw2, 3)}</text>
        <text x="270" y="144" textAnchor="middle" fontSize="11" fontFamily="monospace" fill="#b91c1c">× GELU'({gPrime.toFixed(2)})</text>
        <text x="128" y="144" textAnchor="middle" fontSize="11" fontFamily="monospace" fontWeight="700" fill="#b91c1c">∂L/∂w₁ = {formatNum(dL_dw1, 3)}</text>

        <text x="380" y="178" textAnchor="middle" fontSize="11" fontFamily="monospace" fill="#475569">
          Chain Rule to Layer 1: ∂L/∂w₁ = ({formatNum(dL_dyHat, 2)}) × ({w2.toFixed(2)}) × ({gPrime.toFixed(2)}) × ({x.toFixed(1)}) = {formatNum(dL_dw1, 4)}
        </text>
      </svg>

      <div className="circuit-sliders">
        <label className="circuit-slider-item mono">
          <span>Layer 1 weight w₁ = <strong>{w1.toFixed(2)}</strong> (∂L/∂w₁ = {formatNum(dL_dw1, 3)})</span>
          <input type="range" min={-2.5} max={2.5} step={0.05} value={w1} onChange={(e) => setW1(Number(e.target.value))} />
        </label>
        <label className="circuit-slider-item mono">
          <span>Layer 2 weight w₂ = <strong>{w2.toFixed(2)}</strong> (∂L/∂w₂ = {formatNum(dL_dw2, 3)})</span>
          <input type="range" min={-2.5} max={2.5} step={0.05} value={w2} onChange={(e) => setW2(Number(e.target.value))} />
        </label>
      </div>

      {lossHistory.length > 0 && (
        <div className="p2d-footnote mono">
          <span>
            Recent steps ({lossHistory.length}): {lossHistory[0].toFixed(4)} → {loss.toFixed(4)}
          </span>
          {Math.abs(gPrime) < 0.05 && (
            <span style={{ color: '#dc2626', fontWeight: 700 }}>
              Vanishing gradient! GELU'(z) ≈ 0 is choking the backward signal to w₁.
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================================================================
 * 4. CHAPTER 8 BEAT 4: POST-TRAINING BEHAVIORAL COMPARATOR
 * ========================================================================== */
interface PostTrainCase {
  id: string;
  prompt: string;
  base: string;
  sft: string;
  rlhf: string;
  reasoningThink: string;
  reasoningFinal: string;
}

const POST_TRAIN_CASES: PostTrainCase[] = [
  {
    id: 'gravity',
    prompt: 'Explain gravity to a 10-year-old.',
    base: 'Explain gravity to a 10-year-old. 2. Explain magnetism to a 10-year-old. 3. Explain electricity to a 10-year-old. Worksheet #4 — Middle School Science Archive...',
    sft: '<|user|> Explain gravity to a 10-year-old. <|assistant|> Gravity is an invisible pulling force between anything that has mass. Earth is huge, so it pulls you, your shoes, and even the Moon toward it.',
    rlhf: 'Imagine putting a heavy bowling ball in the middle of a trampoline—it makes a dip, and marbles roll toward it. Earth does the same thing to space around it, which is why dropped balls fall down and planets stay in orbit.',
    reasoningThink: 'Audience is age 10 → avoid equations like F=Gm₁m₂/r². Use a concrete physical image (trampoline dip or invisible pull), connect to everyday experience (jumping, Moon), keep under 3 sentences.',
    reasoningFinal: 'Gravity is an invisible pull that every object with mass makes—the bigger the object, the stronger the pull. Because Earth is gigantic, it pulls you back down when you jump and holds the Moon in orbit like a cosmic tether.',
  },
  {
    id: 'strawberry',
    prompt: 'How many letter R’s are in "strawberry"?',
    base: 'How many letter R’s are in "strawberry"? Answer: 2. Click here for more fun brainteasers and riddles for kids!',
    sft: '<|user|> How many letter R’s are in "strawberry"? <|assistant|> There are 2 R’s in the word "strawberry".',
    rlhf: 'There are 3 R’s in "strawberry" (st-r-awbe-r-r-y). It is easy to miss the double R at the end!',
    reasoningThink: 'Warning: subword tokenization ("straw" + "berry") hides letters. Spell character-by-character: s-t-r(1)-a-w-b-e-r(2)-r(3)-y. Total count = 3.',
    reasoningFinal: 'There are **3** R’s in "strawberry": st**r**awbe**rr**y.',
  },
  {
    id: 'python',
    prompt: 'Write a 1-line Python function to check if a word is a palindrome.',
    base: 'def is_palindrome(s): # TODO: implement in lab 3\n# Posted by user4821 on StackForum (12 upvotes)...',
    sft: '<|user|> Write a 1-line Python function... <|assistant|> ```python\ndef is_palindrome(w): return w == w[::-1]\n```',
    rlhf: 'Here is a concise one-liner that also ignores uppercase/lowercase differences:\n`is_palindrome = lambda w: (s := w.lower()) == s[::-1]`',
    reasoningThink: 'Check edge cases: capitalization ("Racecar"), spaces? User asked for a single word and 1 line. Lowercasing handles "Racecar" cleanly without breaking 1-line readability.',
    reasoningFinal: '`def is_palindrome(w: str) -> bool: return w.lower() == w.lower()[::-1]`\n*(Normalizes case so `"Racecar"` returns `True`.)*',
  },
];

function PostTrainingSimulator() {
  const [caseIdx, setCaseIdx] = useState(0);
  const [showThink, setShowThink] = useState(true);
  const item = POST_TRAIN_CASES[caseIdx];

  return (
    <div className="p2d-card">
      <div className="p2d-header">
        <div>
          <div className="p2d-kicker">Same Transformer Weights · Different Training Policy π_θ(y|x)</div>
          <h2 className="p2d-title">How Post-Training Reshapes Behavior</h2>
        </div>
        <div className="p2d-controls">
          {POST_TRAIN_CASES.map((c, i) => (
            <button
              key={c.id}
              type="button"
              className={`mini-btn ${i === caseIdx ? 'is-active' : ''}`}
              onClick={() => setCaseIdx(i)}
            >
              {c.id === 'gravity' ? 'Gravity' : c.id === 'strawberry' ? 'Strawberry' : 'Python'}
            </button>
          ))}
        </div>
      </div>

      <div className="bpe-rule-banner">
        <strong>Prompt:</strong> “{item.prompt}”
      </div>

      <div className="post-grid">
        <div className="post-tile">
          <div className="post-tile-head">
            <span className="post-badge is-base">1. Base Pretrained</span>
            <span className="mono">Next-token NLL only</span>
          </div>
          <p className="post-body">{item.base}</p>
          <div className="post-why">Mimics web documents—continues the pattern instead of answering.</div>
        </div>

        <div className="post-tile">
          <div className="post-tile-head">
            <span className="post-badge is-sft">2. After SFT</span>
            <span className="mono">Chat Templates + Demos</span>
          </div>
          <p className="post-body mono" style={{ fontSize: '0.76rem' }}>{item.sft}</p>
          <div className="post-why">Learns assistant turn-taking, but may still slip on tricky traps.</div>
        </div>

        <div className="post-tile">
          <div className="post-tile-head">
            <span className="post-badge is-rlhf">3. After RLHF / DPO</span>
            <span className="mono">Preference Alignment</span>
          </div>
          <p className="post-body">{item.rlhf}</p>
          <div className="post-why">Optimized for clarity, helpful analogies, calibration, and safety.</div>
        </div>

        <div className="post-tile">
          <div className="post-tile-head">
            <span className="post-badge is-rl">4. Reasoning RL</span>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setShowThink((v) => !v)}
            >
              {showThink ? 'Hide <think>' : 'Show <think>'}
            </button>
          </div>
          {showThink && (
            <div className="think-box mono">
              &lt;think&gt; {item.reasoningThink} &lt;/think&gt;
            </div>
          )}
          <p className="post-body">{item.reasoningFinal}</p>
          <div className="post-why">Spends extra autoregressive tokens verifying steps before answering.</div>
        </div>
      </div>

      <div className="p2d-footnote">
        <span>Illustrative behavioral comparison: our 862k browser specimen is pretrained-only (Stage 1) on TinyStories.</span>
      </div>
    </div>
  );
}
