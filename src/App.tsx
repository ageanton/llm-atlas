import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnatomyInspector } from './components/AnatomyInspector';
import type { TrainingReplayData } from './components/NodeLab';
import { getAnatomyNode, PRESET_PROMPTS } from './content/llmAnatomy';
import { TransformerEngineClient } from './engine/client';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineInitResult,
  EngineTraceResult,
} from './engine/types';
import { ExplodedWorld } from './scene/ExplodedWorld';
import {
  hasActiveAblation,
  isHeadDisabled,
  stateFromUrl,
  stateToSearch,
  type AtlasState,
  type Register,
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

export function App() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const worldRef = useRef<ExplodedWorld | null>(null);
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
  const [copied, setCopied] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Persist exact exploration state in URL
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

  // Mount Three.js Exploded World once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const world = new ExplodedWorld(el, {
      onSelectNode: (nodeId, layer) => {
        setState((prev) => ({
          ...prev,
          nodeId,
          layer: typeof layer === 'number' ? layer : prev.layer,
        }));
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

  // Initialize verified Web Worker + load real weights & training replay
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

  // Recompute real forward pass & ablation counterfactual when prompt or ablation changes
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

  // Keep 3D exploded scene synced with active node & live trace
  useEffect(() => {
    if (!worldRef.current || !initInfo || !baseline) return;
    worldRef.current.update({
      nodeId: state.nodeId,
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
    state.nodeId,
    state.layer,
    state.head,
    state.showPreSoftmax,
    state.ablation,
    hoveredTokenIdx,
    temperature,
    topK,
    topP,
  ]);

  // Keyboard shortcuts: Esc zooms up one parent folder, 1/2/3 switches register
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Escape') {
        setState((prev) => {
          const cur = getAnatomyNode(prev.nodeId);
          return cur.parentId ? { ...prev, nodeId: cur.parentId } : prev;
        });
      } else if (e.key === '1') {
        setState((prev) => ({ ...prev, register: 'simple' }));
      } else if (e.key === '2') {
        setState((prev) => ({ ...prev, register: 'sufficient' }));
      } else if (e.key === '3') {
        setState((prev) => ({ ...prev, register: 'complete' }));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const handleSelectNode = useCallback((nodeId: string) => {
    setState((prev) => ({ ...prev, nodeId }));
  }, []);

  const handleChangeRegister = useCallback((register: Register) => {
    setState((prev) => ({ ...prev, register }));
  }, []);

  const handleChangeAblation = useCallback((ablation: AblationConfig) => {
    setState((prev) => ({ ...prev, ablation }));
  }, []);

  const handleInspectScalar = useCallback(
    (paramName: string, paramRow: number, paramCol: number) => {
      setState((prev) => ({
        ...prev,
        nodeId: 'scalar',
        paramName,
        paramRow,
        paramCol,
      }));
    },
    []
  );

  const handleTogglePreSoftmax = useCallback(() => {
    setState((prev) => ({
      ...prev,
      showPreSoftmax: !prev.showPreSoftmax,
    }));
  }, []);

  const handleChangeSampling = useCallback(
    (patch: { temperature?: number; topK?: number; topP?: number }) => {
      if (patch.temperature !== undefined) setTemperature(patch.temperature);
      if (patch.topK !== undefined) setTopK(patch.topK);
      if (patch.topP !== undefined) setTopP(patch.topP);
    },
    []
  );

  const handleStepAutoregressive = useCallback(() => {
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
  }, [ablated, baseline, initInfo, temperature, topK, topP, state.prompt]);

  const activeNode = getAnatomyNode(state.nodeId);
  const activeTrace = ablated ?? baseline;
  const ablationOn = hasActiveAblation(state.ablation);

  const telemetryLines = useMemo(() => {
    if (!activeTrace || !initInfo) return [];
    const { trace, tokens } = activeTrace;
    const T = tokens.length;
    const last = T - 1;
    const topId = trace.greedyNextTokenId;
    const topTok = formatTokenExplicit(initInfo.tokenizer.vocab[topId] ?? '');
    const topProb = (trace.finalProbs[topId] * 100).toFixed(1);

    if (activeNode.sceneView === 'embed') {
      return [
        `Tokens: ${T} / ${initInfo.manifest.config.max_seq_len} · Last token: "${formatTokenExplicit(tokens[last].text)}" (#${tokens[last].id})`,
        `||wte|| = ${formatNum(vectorL2Norm(trace.tokEmb[last]), 3)} · ||wpe[${last}]|| = ${formatNum(vectorL2Norm(trace.posEmb[last]), 3)}`,
      ];
    }

    const lt = trace.layers[state.layer];
    if (activeNode.sceneView === 'block') {
      if (lt.skipped) {
        return [`Block ${state.layer} is skipped by Ablation Lab (identity pass-through).`];
      }
      const active: ActiveLayerTrace = lt;
      return [
        `Block ${state.layer} · ||h_in||=${formatNum(vectorL2Norm(active.residualBeforeAttn[last]), 2)} → +Attn(${formatNum(vectorL2Norm(active.attn.output[last]), 2)}) → +MLP(${formatNum(vectorL2Norm(active.mlp.output[last]), 2)})`,
        `||h_out|| = ${formatNum(vectorL2Norm(active.residualAfterMlp[last]), 2)} · shape [${T} × 128]`,
      ];
    }

    if (activeNode.sceneView === 'attention') {
      if (lt.skipped) {
        return [`Block ${state.layer} is skipped by Ablation Lab.`];
      }
      const active: ActiveLayerTrace = lt;
      const focusIdx = hoveredTokenIdx ?? last;
      const row = active.attn.attnWeights[state.head][focusIdx];
      let bestCol = 0;
      for (let c = 1; c <= focusIdx; c++) {
        if (row[c] > row[bestCol]) bestCol = c;
      }
      const sum = row.reduce((a, b) => a + b, 0);
      const off = isHeadDisabled(state.ablation, state.layer, state.head);
      return [
        `Block ${state.layer} Head ${state.head}${off ? ' (OFF)' : ''} · "${formatTokenExplicit(tokens[focusIdx].text)}" attends most to "${formatTokenExplicit(tokens[bestCol].text)}" (${(row[bestCol] * 100).toFixed(1)}%)`,
        `Row softmax sum = ${sum.toFixed(6)} · Q, K, V shape [${T} × 32]`,
      ];
    }

    if (activeNode.sceneView === 'mlp') {
      if (lt.skipped) {
        return [`Block ${state.layer} is skipped by Ablation Lab.`];
      }
      const active: ActiveLayerTrace = lt;
      const firing = countActiveNeurons(active.mlp.hidden[last], 0.05);
      return [
        `Block ${state.layer} MLP · Token "${formatTokenExplicit(tokens[last].text)}": ${firing} / 512 GELU neurons active (>0.05)`,
        `||Δmlp|| = ${formatNum(vectorL2Norm(active.mlp.output[last]), 2)} · expansion 128 → 512 → 128`,
      ];
    }

    if (activeNode.sceneView === 'output') {
      const pool = computeSampledDistribution(
        trace.finalLogits,
        temperature,
        topK,
        topP
      );
      return [
        `Top logit: "${topTok}" z=${formatNum(trace.finalLogits[topId], 2)} → P=${topProb}%`,
        `Sampling nucleus: ${pool.length} tokens (τ=${temperature.toFixed(2)}, k=${topK}, p=${topP.toFixed(2)})`,
      ];
    }

    return [
      `${initInfo.manifest.total_params.toLocaleString()} real params · ${T} tokens · traced in ${activeTrace.elapsedMs}ms`,
      `Next prediction: "${topTok}" (${topProb}%)${trace.inputLoss !== null ? ` · Prompt loss: ${trace.inputLoss.toFixed(3)}` : ''}`,
    ];
  }, [
    activeTrace,
    initInfo,
    activeNode.sceneView,
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

  const showBlockSelector =
    activeNode.sceneView === 'block' ||
    activeNode.sceneView === 'attention' ||
    activeNode.sceneView === 'mlp';

  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          type="button"
          className="brand-btn"
          onClick={() => handleSelectNode('root')}
          title="Zoom out to the complete LLM"
        >
          LLM Atlas
        </button>

        <form
          className="prompt-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!draftPrompt.trim()) return;
            setState((prev) => ({ ...prev, prompt: draftPrompt }));
          }}
        >
          <input
            className="prompt-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type any sentence to trace live..."
            aria-label="Prompt sentence"
          />
          <select
            className="preset-select"
            value=""
            aria-label="Example prompts"
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
          <button type="submit" className="primary-btn" disabled={busy || !initInfo}>
            {busy ? 'Tracing...' : 'Run'}
          </button>
        </form>

        <div className="topbar-actions">
          <button
            type="button"
            className={`mini-btn ${ablationOn ? 'is-warm' : state.nodeId === 'ablation' ? 'is-active' : ''}`}
            onClick={() => handleSelectNode('ablation')}
          >
            {ablationOn ? 'Ablation (on)' : 'Ablate'}
          </button>
          <button
            type="button"
            className="mini-btn"
            onClick={() => void copyLink()}
          >
            {copied ? 'Copied' : 'Share'}
          </button>
        </div>
      </header>

      <div className="split-workspace">
        <section className="stage-pane" aria-label="Interactive 3D exploded LLM">
          <div ref={mountRef} className="canvas-mount" />

          {!initInfo && !error && (
            <div className="loading-overlay">
              Loading 862,464 real trained parameters...
            </div>
          )}
          {error && <div className="error-banner">{error}</div>}

          <div className="stage-hud-top">
            <div className="stage-pill-group">
              {state.nodeId !== 'root' && (
                <button
                  type="button"
                  className="mini-btn"
                  onClick={() => handleSelectNode('root')}
                >
                  Zoom Out (Full LLM)
                </button>
              )}

              {showBlockSelector && (
                <>
                  <span className="stage-pill-label">Block</span>
                  {[0, 1, 2, 3].map((l) => (
                    <button
                      key={l}
                      type="button"
                      className={`mini-btn mono ${state.layer === l ? 'is-active' : ''}`}
                      onClick={() => setState((prev) => ({ ...prev, layer: l }))}
                    >
                      {l}
                    </button>
                  ))}
                </>
              )}

              {activeNode.sceneView === 'attention' && (
                <>
                  <span className="stage-pill-label">Head</span>
                  {[0, 1, 2, 3].map((h) => {
                    const off = isHeadDisabled(state.ablation, state.layer, h);
                    return (
                      <button
                        key={h}
                        type="button"
                        className={`mini-btn mono ${
                          off ? 'is-warm' : state.head === h ? 'is-active' : ''
                        }`}
                        onClick={() => setState((prev) => ({ ...prev, head: h }))}
                      >
                        H{h}
                      </button>
                    );
                  })}
                </>
              )}
            </div>

            {ablationOn && (
              <div className="stage-pill-group">
                <span className="stage-pill-label">Ablation active</span>
                <button
                  type="button"
                  className="mini-btn is-warm"
                  onClick={() => handleChangeAblation({})}
                >
                  Reset
                </button>
              </div>
            )}
          </div>

          {activeTrace && (
            <div className="stage-hud-bottom">
              <div className="token-strip-float" aria-label="Prompt tokens">
                {activeTrace.tokens.map((tok, idx) => (
                  <button
                    key={`${tok.start}-${idx}`}
                    type="button"
                    className={`tok-badge mono ${
                      hoveredTokenIdx === idx ? 'is-hover' : ''
                    }`}
                    onMouseEnter={() => setHoveredTokenIdx(idx)}
                    onMouseLeave={() => setHoveredTokenIdx(null)}
                    title={`Slot ${idx} · Token #${tok.id}`}
                  >
                    {formatToken(tok.text)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {initInfo && baseline && clientRef.current ? (
          <AnatomyInspector
            nodeId={state.nodeId}
            register={state.register}
            telemetryLines={telemetryLines}
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
            temperature={temperature}
            topK={topK}
            topP={topP}
            busy={busy}
            onSelectNode={handleSelectNode}
            onChangeRegister={handleChangeRegister}
            onChangeAblation={handleChangeAblation}
            onInspectScalar={handleInspectScalar}
            onTogglePreSoftmax={handleTogglePreSoftmax}
            onHoverToken={setHoveredTokenIdx}
            onChangeSampling={handleChangeSampling}
            onStepAutoregressive={handleStepAutoregressive}
          />
        ) : (
          <aside className="inspector-rail">
            <div className="inspector-scroll muted-sm">
              Initializing verified miniature transformer engine...
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
