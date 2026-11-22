import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  InspectorDrawers,
  type TrainingReplayData,
} from './components/InspectorDrawers';
import {
  CURRICULUM,
  PRESET_PROMPTS,
  registerLabel,
} from './content/curriculum';
import { TransformerEngineClient } from './engine/client';
import type {
  ActiveLayerTrace,
  EngineInitResult,
  EngineTraceResult,
} from './engine/types';
import { ArchitecturalWorld } from './scene/ArchitecturalWorld';
import {
  hasActiveAblation,
  isHeadDisabled,
  STAGES,
  stateFromUrl,
  stateToSearch,
  type AtlasState,
  type DrawerId,
  type Register,
  type StageId,
} from './state/atlasState';
import {
  computeSampledDistribution,
  countActiveNeurons,
  formatNum,
  formatToken,
  formatTokenExplicit,
  sampleFromDistribution,
  vectorL2Norm,
} from './utils/math';

const REGISTERS: Register[] = ['intuitive', 'technical', 'math'];

export function App() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const worldRef = useRef<ArchitecturalWorld | null>(null);
  const clientRef = useRef<TransformerEngineClient | null>(null);

  const [state, setState] = useState<AtlasState>(() =>
    stateFromUrl(window.location.search)
  );
  const [draftPrompt, setDraftPrompt] = useState<string>(state.prompt);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [replay, setReplay] = useState<TrainingReplayData | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(12);
  const [topP, setTopP] = useState<number>(0.9);
  const [explainOpen, setExplainOpen] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Sync deep-linkable URL state
  useEffect(() => {
    const nextSearch = stateToSearch(state);
    if (window.location.search !== nextSearch) {
      window.history.replaceState(
        null,
        '',
        `${window.location.pathname}${nextSearch}`
      );
    }
  }, [state]);

  // Mount Three.js Architectural World once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const world = new ArchitecturalWorld(el, {
      onSelectStage: (stage, layer) => {
        setState((prev) => ({
          ...prev,
          stage,
          layer: typeof layer === 'number' ? layer : prev.layer,
        }));
      },
      onSelectHead: (head) => {
        setState((prev) => ({ ...prev, head }));
      },
      onHoverToken: (idx) => {
        setHoveredTokenIdx(idx);
      },
    });
    worldRef.current = world;
    return () => {
      world.dispose();
      worldRef.current = null;
    };
  }, []);

  // Boot verified TypeScript engine + load real weights & training replay
  useEffect(() => {
    const client = new TransformerEngineClient();
    clientRef.current = client;
    const modelBaseUrl = new URL(
      `${import.meta.env.BASE_URL}model/`,
      window.location.origin
    ).href;

    client
      .init(modelBaseUrl)
      .then((info) => setInitInfo(info))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err))
      );

    fetch(`${modelBaseUrl}training_replay.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: TrainingReplayData | null) => {
        if (data) setReplay(data);
      })
      .catch(() => undefined);

    return () => {
      client.dispose();
      clientRef.current = null;
    };
  }, []);

  // Recompute real forward pass & ablation counterfactual on change
  useEffect(() => {
    const client = clientRef.current;
    if (!client || !initInfo) return;
    let alive = true;
    setBusy(true);
    setError(null);

    const run = async () => {
      try {
        const baseRes = await client.trace(state.prompt);
        if (!alive) return;
        setBaseline(baseRes);
        if (hasActiveAblation(state.ablation)) {
          const ablRes = await client.trace(state.prompt, state.ablation);
          if (!alive) return;
          setAblated(ablRes);
        } else {
          setAblated(null);
        }
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (alive) setBusy(false);
      }
    };

    void run();
    return () => {
      alive = false;
    };
  }, [initInfo, state.prompt, state.ablation]);

  // Push live trace & parameters into the 3D scene
  useEffect(() => {
    if (!worldRef.current || !initInfo || !baseline) return;
    worldRef.current.update({
      stage: state.stage,
      layer: state.layer,
      head: state.head,
      showPreSoftmax: state.showPreSoftmax,
      ablation: state.ablation,
      baseline,
      ablated,
      tokenizer: initInfo.tokenizer,
      hoveredTokenIdx,
      temperature,
      topK,
      topP,
    });
  }, [
    initInfo,
    baseline,
    ablated,
    state.stage,
    state.layer,
    state.head,
    state.showPreSoftmax,
    state.ablation,
    hoveredTokenIdx,
    temperature,
    topK,
    topP,
  ]);

  const setStage = useCallback((stage: StageId) => {
    setState((prev) => ({ ...prev, stage, drawer: 'none' }));
  }, []);

  const toggleDrawer = useCallback((drawer: DrawerId) => {
    setState((prev) => ({
      ...prev,
      drawer: prev.drawer === drawer ? 'none' : drawer,
    }));
  }, []);

  // Autoregressive +1 token step directly from the real sampled distribution
  const handleStepAutoregressive = () => {
    const active = ablated ?? baseline;
    if (!active || !initInfo) return;
    const dist = computeSampledDistribution(
      active.trace.finalLogits,
      temperature,
      topK,
      topP
    );
    const nextId = sampleFromDistribution(dist);
    const piece = initInfo.tokenizer.vocab[nextId] ?? '';
    if (!piece || piece === '<|eos|>') return;
    const nextPrompt = `${state.prompt}${piece}`;
    setDraftPrompt(nextPrompt);
    setState((prev) => ({ ...prev, prompt: nextPrompt }));
  };

  const activeTrace = ablated ?? baseline;
  const spec = CURRICULUM[state.stage];

  // Live numeric telemetry lines specific to the current stage and real trace
  const liveTelemetry = useMemo(() => {
    if (!activeTrace || !initInfo) return [];
    const { trace, tokens } = activeTrace;
    const T = tokens.length;
    const last = T - 1;
    const topId = trace.greedyNextTokenId;
    const topTok = formatTokenExplicit(initInfo.tokenizer.vocab[topId] ?? '');
    const topProb = (trace.finalProbs[topId] * 100).toFixed(1);

    if (state.stage === 'overview') {
      return [
        `Sequence: ${T} / ${initInfo.manifest.config.max_seq_len} tokens · ${initInfo.manifest.total_params.toLocaleString()} params · computed in ${activeTrace.elapsedMs}ms`,
        `Top prediction: "${topTok}" (${topProb}%)${trace.inputLoss !== null ? ` · Prompt loss: ${trace.inputLoss.toFixed(3)}` : ''}`,
      ];
    }
    if (state.stage === 'embed') {
      const embNorm = vectorL2Norm(trace.tokEmb[last]);
      const posNorm = vectorL2Norm(trace.posEmb[last]);
      return [
        `Token #${last} ("${formatTokenExplicit(tokens[last].text)}", ID ${tokens[last].id})`,
        `||wte|| = ${formatNum(embNorm, 3)} · ||wpe[${last}]|| = ${formatNum(posNorm, 3)} · output slab [${T} × 128]`,
      ];
    }
    const lt = trace.layers[state.layer];
    if (lt.skipped) {
      return [`Block ${state.layer} is skipped by Ablation Lab (identity pass-through).`];
    }
    const active: ActiveLayerTrace = lt;
    if (state.stage === 'block') {
      return [
        `Block ${state.layer} · ||h_in|| = ${formatNum(vectorL2Norm(active.residualBeforeAttn[last]), 2)} → +Attn (${formatNum(vectorL2Norm(active.attn.output[last]), 2)}) → +MLP (${formatNum(vectorL2Norm(active.mlp.output[last]), 2)})`,
        `||h_out|| = ${formatNum(vectorL2Norm(active.residualAfterMlp[last]), 2)} · shape [${T} × 128]`,
      ];
    }
    if (state.stage === 'attention') {
      const focusIdx = hoveredTokenIdx ?? last;
      const row = active.attn.attnWeights[state.head][focusIdx];
      let bestCol = 0;
      for (let c = 1; c <= focusIdx; c++) {
        if (row[c] > row[bestCol]) bestCol = c;
      }
      const sum = row.reduce((a, b) => a + b, 0);
      const headOff = isHeadDisabled(state.ablation, state.layer, state.head);
      return [
        `Block ${state.layer} Head ${state.head}${headOff ? ' (DISABLED)' : ''} · Query "${formatTokenExplicit(tokens[focusIdx].text)}" attends most to "${formatTokenExplicit(tokens[bestCol].text)}" (${(row[bestCol] * 100).toFixed(1)}%)`,
        `Row softmax sum = ${sum.toFixed(6)} · Q, K, V shape [${T} × 32]`,
      ];
    }
    if (state.stage === 'mlp') {
      const activeNeurons = countActiveNeurons(active.mlp.hidden[last], 0.05);
      return [
        `Block ${state.layer} MLP · Token "${formatTokenExplicit(tokens[last].text)}": ${activeNeurons} / 512 GELU neurons active (> 0.05)`,
        `Expansion: 128 → 512 → 128 · ||Δmlp|| = ${formatNum(vectorL2Norm(active.mlp.output[last]), 2)}`,
      ];
    }
    const pool = computeSampledDistribution(
      trace.finalLogits,
      temperature,
      topK,
      topP
    );
    return [
      `Top logit: "${topTok}" z = ${formatNum(trace.finalLogits[topId], 2)} → raw P = ${topProb}%`,
      `Active sampling pool: ${pool.length} tokens (τ=${temperature.toFixed(2)}, k=${topK}, p=${topP.toFixed(2)})`,
    ];
  }, [
    activeTrace,
    initInfo,
    state.stage,
    state.layer,
    state.head,
    state.ablation,
    hoveredTokenIdx,
    temperature,
    topK,
    topP,
  ]);

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="studio-viewport">
      <div
        ref={mountRef}
        className="canvas-mount"
        aria-label="Interactive 3D architectural miniature transformer"
      />

      {!initInfo && !error && (
        <div className="loading-pill glass-panel">
          Loading 862,464 real trained parameters...
        </div>
      )}
      {error && <div className="error-toast">{error}</div>}

      {/* Top Architectural Navigation + Labs */}
      <header className="hud-top">
        <nav className="hud-bar glass-panel" aria-label="Transformer stages">
          <span className="brand-mark">LLM Atlas</span>
          {STAGES.map((id) => (
            <button
              key={id}
              type="button"
              className={`stage-btn ${state.stage === id ? 'is-current' : ''}`}
              onClick={() => setStage(id)}
            >
              {CURRICULUM[id].navLabel}
            </button>
          ))}
        </nav>

        <div className="hud-bar glass-panel">
          <div className="lens-group" role="group" aria-label="Explanation register">
            {REGISTERS.map((r) => (
              <button
                key={r}
                type="button"
                className={`lens-btn ${state.register === r ? 'is-active' : ''}`}
                onClick={() => setState((prev) => ({ ...prev, register: r }))}
              >
                {registerLabel(r)}
              </button>
            ))}
          </div>
          <button
            type="button"
            className={`pill-btn ${
              state.drawer === 'ablate'
                ? 'is-active'
                : hasActiveAblation(state.ablation)
                  ? 'is-warm'
                  : ''
            }`}
            onClick={() => toggleDrawer('ablate')}
          >
            {hasActiveAblation(state.ablation) ? 'Ablation (on)' : 'Ablate'}
          </button>
          <button
            type="button"
            className={`pill-btn ${state.drawer === 'weights' ? 'is-active' : ''}`}
            onClick={() => toggleDrawer('weights')}
          >
            Weights
          </button>
          <button
            type="button"
            className={`pill-btn ${state.drawer === 'training' ? 'is-active' : ''}`}
            onClick={() => toggleDrawer('training')}
          >
            Training & Scale
          </button>
          <button type="button" className="pill-btn" onClick={() => void copyLink()}>
            {copied ? 'Copied' : 'Share'}
          </button>
        </div>
      </header>

      {/* Bottom-Left Floating Curriculum + Live Numeric Telemetry */}
      <aside className="explain-card glass-panel" aria-live="polite">
        <div className="explain-head">
          <h1 className="explain-title">{spec.title}</h1>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span
              className={`honesty-tag ${
                spec.honestyKind === 'literal' ? 'is-literal' : 'is-structural'
              }`}
            >
              {spec.honestyLabel}
            </span>
            <button
              type="button"
              className="pill-btn"
              onClick={() => setExplainOpen((v) => !v)}
            >
              {explainOpen ? 'Hide' : 'Read'}
            </button>
          </div>
        </div>

        {explainOpen && (
          <>
            <p className="explain-prose">{spec.copy[state.register]}</p>

            {liveTelemetry.length > 0 && (
              <div className="telemetry-box mono">
                {liveTelemetry.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}

            {spec.nextStage && spec.nextQuestion && (
              <div className="explain-next">
                <span className="next-q">{spec.nextQuestion}</span>
                <button
                  type="button"
                  className="cta-btn"
                  onClick={() => setStage(spec.nextStage!)}
                >
                  {CURRICULUM[spec.nextStage].navLabel} →
                </button>
              </div>
            )}
          </>
        )}
      </aside>

      {/* Bottom-Right Contextual Architectural Controls */}
      <div className="stage-dock">
        {(state.stage === 'block' ||
          state.stage === 'attention' ||
          state.stage === 'mlp') && (
          <div className="dock-strip glass-panel">
            <span className="dock-label">Block</span>
            {[0, 1, 2, 3].map((l) => (
              <button
                key={l}
                type="button"
                className={`pill-btn mono ${state.layer === l ? 'is-active' : ''}`}
                onClick={() => setState((prev) => ({ ...prev, layer: l }))}
              >
                {l}
              </button>
            ))}
          </div>
        )}

        {state.stage === 'attention' && (
          <div className="dock-strip glass-panel">
            <span className="dock-label">Head</span>
            {[0, 1, 2, 3].map((h) => {
              const off = isHeadDisabled(state.ablation, state.layer, h);
              return (
                <button
                  key={h}
                  type="button"
                  className={`pill-btn mono ${
                    off ? 'is-warm' : state.head === h ? 'is-active' : ''
                  }`}
                  onClick={() => setState((prev) => ({ ...prev, head: h }))}
                >
                  H{h}
                </button>
              );
            })}
            <button
              type="button"
              className={`pill-btn ${state.showPreSoftmax ? 'is-active' : ''}`}
              onClick={() =>
                setState((prev) => ({
                  ...prev,
                  showPreSoftmax: !prev.showPreSoftmax,
                }))
              }
            >
              {state.showPreSoftmax ? 'Raw QKᵀ/√32' : 'Softmax A'}
            </button>
            <button
              type="button"
              className={`pill-btn ${state.drawer === 'matrix' ? 'is-active' : ''}`}
              onClick={() => toggleDrawer('matrix')}
            >
              Exact Table
            </button>
          </div>
        )}

        {state.stage === 'output' && (
          <div className="dock-strip sampling-dock glass-panel">
            <div className="mini-slider">
              <div className="row-between">
                <span>Temperature</span>
                <span className="mono">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={2.0}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>
            <div className="mini-slider">
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
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <div className="mini-slider">
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
                onChange={(e) => setTopP(Number(e.target.value))}
              />
            </div>
            <button
              type="button"
              className="cta-btn"
              disabled={busy || !activeTrace}
              onClick={handleStepAutoregressive}
            >
              Step +1 Token (Autoregressive)
            </button>
          </div>
        )}
      </div>

      {/* Educational Labs Drawer */}
      {initInfo && baseline && clientRef.current && (
        <InspectorDrawers
          drawer={state.drawer}
          client={clientRef.current}
          manifest={initInfo.manifest}
          tokenizer={initInfo.tokenizer}
          baseline={baseline}
          ablated={ablated}
          ablation={state.ablation}
          layer={state.layer}
          head={state.head}
          showPreSoftmax={state.showPreSoftmax}
          hoveredTokenIdx={hoveredTokenIdx}
          replay={replay}
          paramName={state.paramName}
          paramRow={state.paramRow}
          paramCol={state.paramCol}
          onChangeAblation={(ablation) =>
            setState((prev) => ({ ...prev, ablation }))
          }
          onSelectParam={(paramName, paramRow, paramCol) =>
            setState((prev) => ({
              ...prev,
              drawer: 'weights',
              paramName,
              paramRow,
              paramCol,
            }))
          }
          onTogglePreSoftmax={() =>
            setState((prev) => ({
              ...prev,
              showPreSoftmax: !prev.showPreSoftmax,
            }))
          }
          onHoverToken={setHoveredTokenIdx}
          onClose={() => setState((prev) => ({ ...prev, drawer: 'none' }))}
        />
      )}

      {/* Bottom Sentence Input & Interactive BPE Token Strip */}
      <div className="hud-bottom">
        {activeTrace && (
          <div className="token-ribbon" aria-label="BPE Tokens">
            {activeTrace.tokens.map((tok, idx) => (
              <button
                key={`${tok.start}-${idx}`}
                type="button"
                className={`tok-chip mono ${
                  hoveredTokenIdx === idx ? 'is-hover' : ''
                }`}
                onMouseEnter={() => setHoveredTokenIdx(idx)}
                onMouseLeave={() => setHoveredTokenIdx(null)}
                title={`Slot ${idx} · Token ID #${tok.id} · chars [${tok.start}, ${tok.end})`}
              >
                {formatToken(tok.text)}
              </button>
            ))}
          </div>
        )}

        <form
          className="sentence-bar glass-panel"
          onSubmit={(e) => {
            e.preventDefault();
            if (!draftPrompt.trim()) return;
            setState((prev) => ({ ...prev, prompt: draftPrompt }));
          }}
        >
          <input
            className="sentence-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type any sentence to run a live forward pass..."
            aria-label="Prompt sentence"
          />
          <select
            className="select-pill"
            value=""
            aria-label="Choose an example prompt"
            onChange={(e) => {
              const p = PRESET_PROMPTS.find((x) => x.id === e.target.value);
              if (!p) return;
              setDraftPrompt(p.text);
              setState((prev) => ({ ...prev, prompt: p.text }));
            }}
          >
            <option value="">Examples...</option>
            {PRESET_PROMPTS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <button type="submit" className="cta-btn" disabled={busy || !initInfo}>
            {busy ? 'Tracing...' : 'Run'}
          </button>
        </form>
      </div>
    </div>
  );
}
