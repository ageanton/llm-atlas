import { useEffect, useMemo, useRef, useState } from 'react';
import { BookReader } from './components/BookReader';
import {
  PRESET_PROMPTS,
  TEXTBOOK_CHAPTERS,
  type SceneBeatId,
} from './content/bookData';
import { TransformerEngineClient } from './engine/client';
import type {
  ActiveLayerTrace,
  EngineInitResult,
  EngineTraceResult,
} from './engine/types';
import { TextbookStage } from './scene/TextbookStage';
import {
  clampInt,
  hasActiveAblation,
  isHeadDisabled,
  stateFromUrl,
  stateToSearch,
  toggleHeadDisabled,
  type BookState,
} from './state/bookState';
import {
  computeDotProductBreakdown,
  formatNum,
  formatToken,
  formatTokenExplicit,
} from './utils/math';

export function App() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<TextbookStage | null>(null);
  const clientRef = useRef<TransformerEngineClient | null>(null);

  const [state, setState] = useState<BookState>(() =>
    stateFromUrl(window.location.search)
  );
  const [draftPrompt, setDraftPrompt] = useState<string>(state.prompt);
  const [showContents, setShowContents] = useState<boolean>(false);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const chapter =
    TEXTBOOK_CHAPTERS.find((c) => c.id === state.chapterId) ??
    TEXTBOOK_CHAPTERS[3];
  const activeBeat =
    chapter.beats[clampInt(state.beatIndex, 0, Math.max(0, chapter.beats.length - 1))] ??
    null;
  const activeSceneBeat: SceneBeatId = activeBeat?.sceneBeat ?? 'attn-1-problem';

  // Sync deep-link URL state
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

  // Mount Three.js Stage once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const stage = new TextbookStage(el, {
      onPickToken: (idx) => {
        setState((prev) => {
          // On Beat 2 (One Comparison), clicking a token picks the Key token being compared
          if (prev.chapterId === 4 && prev.beatIndex === 1) {
            return { ...prev, keyIdx: idx };
          }
          return { ...prev, queryIdx: idx };
        });
      },
      onPickHead: (head) => {
        setState((prev) => ({ ...prev, head }));
      },
    });
    stageRef.current = stage;

    return () => {
      stage.dispose();
      stageRef.current = null;
    };
  }, []);

  // Boot verified TypeScript engine and load real weights
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

    return () => {
      client.dispose();
      clientRef.current = null;
    };
  }, []);

  // Re-run real forward pass & ablation counterfactual on prompt or ablation change
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

  const activeTrace = ablated ?? baseline;

  // Resolve sensible default Query (last token) & Key (strongest earlier token or "Max")
  const { resolvedQueryIdx, resolvedKeyIdx } = useMemo(() => {
    if (!activeTrace) return { resolvedQueryIdx: 0, resolvedKeyIdx: 0 };
    const T = activeTrace.tokens.length;
    const q = clampInt(state.queryIdx ?? T - 1, 0, Math.max(0, T - 1));

    if (state.keyIdx !== null && Number.isFinite(state.keyIdx)) {
      return { resolvedQueryIdx: q, resolvedKeyIdx: clampInt(state.keyIdx, 0, Math.max(0, T - 1)) };
    }

    // Default Key: pick the earlier token with highest attention weight from q, or slot 6 ("Max")
    const lt = activeTrace.trace.layers[state.layer];
    let bestK = Math.max(0, q - 1);
    if (!lt.skipped) {
      const active: ActiveLayerTrace = lt;
      const weights = active.attn.attnWeights[state.head][q] ?? [];
      let maxW = -1;
      for (let k = 0; k < q; k++) {
        if (weights[k] > maxW) {
          maxW = weights[k];
          bestK = k;
        }
      }
    }
    return { resolvedQueryIdx: q, resolvedKeyIdx: bestK };
  }, [activeTrace, state.queryIdx, state.keyIdx, state.layer, state.head]);

  // Push live trace & beat choreography to 3D stage
  useEffect(() => {
    if (!stageRef.current || !activeTrace) return;
    stageRef.current.update({
      beat: activeSceneBeat,
      layer: state.layer,
      head: state.head,
      queryIdx: resolvedQueryIdx,
      keyIdx: resolvedKeyIdx,
      showPreSoftmax: state.showPreSoftmax,
      ablation: state.ablation,
      traceRes: activeTrace,
    });
  }, [
    activeTrace,
    activeSceneBeat,
    state.layer,
    state.head,
    resolvedQueryIdx,
    resolvedKeyIdx,
    state.showPreSoftmax,
    state.ablation,
  ]);

  // Keyboard shortcuts: Left/Right arrows step beats, 1/2/3 switch editions
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'ArrowRight') {
        setState((prev) => ({
          ...prev,
          beatIndex: clampInt(prev.beatIndex + 1, 0, 5),
        }));
      } else if (e.key === 'ArrowLeft') {
        setState((prev) => ({
          ...prev,
          beatIndex: clampInt(prev.beatIndex - 1, 0, 5),
        }));
      } else if (e.key === '1') {
        setState((prev) => ({ ...prev, edition: 'foundations' }));
      } else if (e.key === '2') {
        setState((prev) => ({ ...prev, edition: 'practitioner' }));
      } else if (e.key === '3') {
        setState((prev) => ({ ...prev, edition: 'formal' }));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Build real, beat-specific numeric proof lines from the live trace
  const liveCardLines = useMemo(() => {
    if (!activeTrace || !baseline || !initInfo) return [];
    const { tokens, trace } = activeTrace;
    const lt = trace.layers[state.layer];
    if (lt.skipped) return [`Block ${state.layer} is disabled by Ablation.`];

    const active: ActiveLayerTrace = lt;
    const qTok = formatTokenExplicit(tokens[resolvedQueryIdx].text);
    const kTok = formatTokenExplicit(tokens[resolvedKeyIdx].text);
    const rowW = active.attn.attnWeights[state.head][resolvedQueryIdx];
    const rowS = active.attn.rawScores[state.head][resolvedQueryIdx];

    if (activeSceneBeat === 'attn-1-problem') {
      const topId = trace.greedyNextTokenId;
      return [
        `Sentence length: ${tokens.length} BPE tokens · Highlighted Query: "${qTok}" (slot ${resolvedQueryIdx})`,
        `Current model top guess after all 4 blocks: "${formatTokenExplicit(initInfo.tokenizer.vocab[topId] ?? '')}" (${(trace.finalProbs[topId] * 100).toFixed(1)}%)`,
      ];
    }

    if (activeSceneBeat === 'attn-2-dot-product') {
      const bd = computeDotProductBreakdown(
        active.attn.q[state.head][resolvedQueryIdx],
        active.attn.k[state.head][resolvedKeyIdx]
      );
      return [
        `Comparing Q("${qTok}") · K("${kTok}") in Block ${state.layer} Head ${state.head} (32 dims):`,
        `Positive votes: +${formatNum(bd.positiveVotes, 2)} · Negative votes: ${formatNum(bd.negativeVotes, 2)} → Raw dot = ${formatNum(bd.rawDot, 3)}`,
        `Scaled score = ${formatNum(bd.rawDot, 3)} ÷ √32 (${formatNum(bd.scaleDivisor, 3)}) = ${formatNum(bd.scaledScore, 3)}`,
      ];
    }

    if (activeSceneBeat === 'attn-3-scan-row') {
      let maxC = 0;
      let minC = 0;
      for (let c = 1; c < tokens.length; c++) {
        if (rowS[c] > rowS[maxC]) maxC = c;
        if (rowS[c] < rowS[minC]) minC = c;
      }
      return [
        `Query "${qTok}" scanned ${tokens.length} Keys in Block ${state.layer} Head ${state.head}:`,
        `Highest raw score: "${formatTokenExplicit(tokens[maxC].text)}" (${formatNum(rowS[maxC], 2)}) · Lowest: "${formatTokenExplicit(tokens[minC].text)}" (${formatNum(rowS[minC], 2)})`,
      ];
    }

    if (activeSceneBeat === 'attn-4-mask-softmax') {
      const rowSum = rowW.reduce((a, b) => a + b, 0);
      const maskedCount = tokens.length - 1 - resolvedQueryIdx;
      let bestC = 0;
      for (let c = 1; c <= resolvedQueryIdx; c++) {
        if (rowW[c] > rowW[bestC]) bestC = c;
      }
      return [
        `Future tokens sealed by Causal Mask: ${maskedCount} · Surviving tokens: ${resolvedQueryIdx + 1}`,
        `Post-Softmax winner: "${formatTokenExplicit(tokens[bestC].text)}" (${(rowW[bestC] * 100).toFixed(1)}%) · Exact row sum = ${rowSum.toFixed(6)}`,
      ];
    }

    if (activeSceneBeat === 'attn-5-value-mixer') {
      const top3 = rowW
        .slice(0, resolvedQueryIdx + 1)
        .map((w, idx) => ({ idx, w }))
        .sort((a, b) => b.w - a.w)
        .slice(0, 3);
      const blendText = top3
        .map((x) => `${(x.w * 100).toFixed(0)}% V("${formatTokenExplicit(tokens[x.idx].text)}")`)
        .join(' + ');
      return [
        `Value blend poured into "${qTok}" (Block ${state.layer} Head ${state.head}):`,
        `${blendText} + ...`,
      ];
    }

    // Beat 6: Full matrix + live ablation diff
    const baseTop = baseline.trace.greedyNextTokenId;
    const curTop = trace.greedyNextTokenId;
    const off = isHeadDisabled(state.ablation, state.layer, state.head);
    const lines = [
      `Block ${state.layer} Head ${state.head}: ${off ? 'DISABLED (Z_h zeroed before W_o)' : 'Active'} · Matrix [${tokens.length} × ${tokens.length}]`,
      `Prediction: "${formatTokenExplicit(initInfo.tokenizer.vocab[baseTop] ?? '')}" (${(baseline.trace.finalProbs[baseTop] * 100).toFixed(1)}%)${
        ablated
          ? ` → "${formatTokenExplicit(initInfo.tokenizer.vocab[curTop] ?? '')}" (${(trace.finalProbs[curTop] * 100).toFixed(1)}%)`
          : ''
      }`,
    ];
    if (off) {
      lines.push('Observe how zeroing Head 3 shifts next-token probabilities!');
    }
    return lines;
  }, [
    activeTrace,
    baseline,
    ablated,
    initInfo,
    state.layer,
    state.head,
    resolvedQueryIdx,
    resolvedKeyIdx,
    activeSceneBeat,
    state.ablation,
  ]);

  const activeTokens = activeTrace?.tokens ?? [];
  const headOff = isHeadDisabled(state.ablation, state.layer, state.head);

  return (
    <div className={`app-root ${state.lectureMode ? 'is-lecture' : ''}`}>
      {/* Top Application Bar */}
      <header className="top-navbar">
        <div className="nav-left">
          <span className="brand-badge">LLM Atlas</span>
          <span className="brand-pill">Interactive Textbook</span>
          <span className="parity-badge">100% PyTorch Parity</span>
        </div>

        <div className="nav-center">
          <form
            className="prompt-form"
            onSubmit={(e) => {
              e.preventDefault();
              if (draftPrompt.trim()) {
                setState((prev) => ({
                  ...prev,
                  prompt: draftPrompt.trim(),
                  queryIdx: null,
                  keyIdx: null,
                }));
              }
            }}
          >
            <input
              type="text"
              className="prompt-input mono"
              value={draftPrompt}
              onChange={(e) => setDraftPrompt(e.target.value)}
              placeholder="Type any sentence to see real tensors..."
            />
            <button
              type="submit"
              className="primary-btn"
              disabled={busy || !initInfo}
            >
              {busy ? 'Running...' : 'Run Trace'}
            </button>
          </form>
        </div>

        <div className="nav-right">
          <select
            className="preset-select"
            aria-label="Preset teaching prompts"
            value=""
            onChange={(e) => {
              const p = PRESET_PROMPTS.find((x) => x.id === e.target.value);
              if (p) {
                setDraftPrompt(p.text);
                setState((prev) => ({
                  ...prev,
                  prompt: p.text,
                  queryIdx: null,
                  keyIdx: null,
                }));
              }
            }}
          >
            <option value="" disabled>
              Select Teaching Example…
            </option>
            {PRESET_PROMPTS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>

          <button
            type="button"
            className="outline-btn"
            onClick={() => {
              navigator.clipboard.writeText(window.location.href);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
          >
            {copied ? 'Copied Link!' : 'Share'}
          </button>

          <button
            type="button"
            className={`outline-btn ${state.lectureMode ? 'is-active' : ''}`}
            onClick={() =>
              setState((prev) => ({
                ...prev,
                lectureMode: !prev.lectureMode,
              }))
            }
          >
            {state.lectureMode ? 'Reader + 3D' : 'Lecture 3D'}
          </button>
        </div>
      </header>

      {/* Main Split Screen */}
      <div className="main-workspace">
        {/* Left Column: Interactive Textbook Reader */}
        {!state.lectureMode && (
          <BookReader
            chapter={chapter}
            beat={activeBeat}
            beatIndex={state.beatIndex}
            edition={state.edition}
            showContents={showContents}
            liveCardLines={liveCardLines}
            onToggleContents={() => setShowContents((v) => !v)}
            onSelectChapter={(id) => {
              setState((prev) => ({
                ...prev,
                chapterId: id,
                beatIndex: 0,
                queryIdx: null,
                keyIdx: null,
              }));
              setShowContents(false);
            }}
            onSelectBeat={(idx) => setState((prev) => ({ ...prev, beatIndex: idx }))}
            onChangeEdition={(ed) => setState((prev) => ({ ...prev, edition: ed }))}
          />
        )}

        {/* Right Column: 3D Sunlit Architectural Stage */}
        <section className="stage-pane" aria-label="Interactive 3D Stage">
          <div ref={mountRef} className="three-container" />

          {/* Top HUD Controls */}
          <div className="stage-top-hud">
            <div className="hud-pill-group">
              <span className="hud-title">Block:</span>
              {[0, 1, 2, 3].map((l) => (
                <button
                  key={l}
                  type="button"
                  className={`mini-pill mono ${state.layer === l ? 'is-active' : ''}`}
                  onClick={() => setState((prev) => ({ ...prev, layer: l }))}
                >
                  L{l}
                </button>
              ))}

              <span className="hud-title" style={{ marginLeft: 6 }}>
                Head:
              </span>
              {[0, 1, 2, 3].map((h) => {
                const off = isHeadDisabled(state.ablation, state.layer, h);
                return (
                  <button
                    key={h}
                    type="button"
                    className={`mini-pill mono ${
                      off ? 'is-warn' : state.head === h ? 'is-active' : ''
                    }`}
                    onClick={() => setState((prev) => ({ ...prev, head: h }))}
                  >
                    H{h}
                  </button>
                );
              })}

              {activeSceneBeat === 'attn-4-mask-softmax' && (
                <button
                  type="button"
                  className={`mini-pill ${state.showPreSoftmax ? 'is-active' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      showPreSoftmax: !prev.showPreSoftmax,
                    }))
                  }
                >
                  {state.showPreSoftmax ? 'Raw Logits' : 'Softmax %'}
                </button>
              )}

              {(activeSceneBeat === 'attn-5-value-mixer' ||
                activeSceneBeat === 'attn-6-multihead-full') && (
                <button
                  type="button"
                  className={`mini-pill ${headOff ? 'is-warn' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleHeadDisabled(
                        prev.ablation,
                        prev.layer,
                        prev.head
                      ),
                    }))
                  }
                >
                  {headOff ? `Re-enable H${state.head}` : `Ablate H${state.head}`}
                </button>
              )}
            </div>

            <div className="hud-pill-group">
              {hasActiveAblation(state.ablation) && (
                <button
                  type="button"
                  className="mini-pill is-warn"
                  onClick={() => setState((prev) => ({ ...prev, ablation: {} }))}
                >
                  Reset Ablation
                </button>
              )}
              <button
                type="button"
                className="mini-pill"
                onClick={() => stageRef.current?.resetCamera()}
              >
                Reset Camera
              </button>
            </div>
          </div>

          {/* Bottom HUD: Sequence Token Bar */}
          {activeTokens.length > 0 && (
            <div className="stage-bottom-hud">
              <div className="token-strip" aria-label="Select active token">
                <span className="hud-title">
                  {activeSceneBeat === 'attn-2-dot-product'
                    ? 'Comparing Key:'
                    : 'Query Token:'}
                </span>
                {activeTokens.map((tok, idx) => {
                  const isQ = idx === resolvedQueryIdx;
                  const isK =
                    activeSceneBeat === 'attn-2-dot-product' && idx === resolvedKeyIdx;
                  return (
                    <button
                      key={`${tok.start}-${idx}`}
                      type="button"
                      className={`token-btn mono ${isQ ? 'is-query' : ''} ${
                        isK ? 'is-key' : ''
                      }`}
                      onClick={() =>
                        setState((prev) =>
                          activeSceneBeat === 'attn-2-dot-product'
                            ? { ...prev, keyIdx: idx }
                            : { ...prev, queryIdx: idx }
                        )
                      }
                      title={`Token ${idx}: "${formatTokenExplicit(tok.text)}"`}
                    >
                      {formatToken(tok.text)}
                    </button>
                  );
                })}
              </div>
              <div className="camera-guide">
                Drag to orbit · Right-drag or 2-finger drag to pan · Scroll to zoom · Click any token to inspect
              </div>
            </div>
          )}

          {error && <div className="stage-error mono">{error}</div>}
        </section>
      </div>
    </div>
  );
}
