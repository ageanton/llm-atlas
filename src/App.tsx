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
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (alive) setBusy(false);
      }
    };

    run();
    return () => {
      alive = false;
    };
  }, [initInfo, state.prompt, state.ablation]);

  // Keep Three.js Stage updated with live trace and active beat
  useEffect(() => {
    if (!stageRef.current || !baseline) return;

    const numTokens = baseline.tokens.length;
    const qIdx = clampInt(
      state.queryIdx ?? numTokens - 1,
      0,
      Math.max(0, numTokens - 1)
    );
    const kIdx = clampInt(
      state.keyIdx ?? Math.max(0, qIdx - 1),
      0,
      Math.max(0, numTokens - 1)
    );

    stageRef.current.render({
      beat: activeSceneBeat,
      layer: state.layer,
      head: state.head,
      queryIdx: qIdx,
      keyIdx: kIdx,
      showPreSoftmax: state.showPreSoftmax,
      ablation: state.ablation,
      traceRes: ablated ?? baseline,
    });
  }, [
    baseline,
    ablated,
    activeSceneBeat,
    state.layer,
    state.head,
    state.queryIdx,
    state.keyIdx,
    state.showPreSoftmax,
    state.ablation,
  ]);

  // Handle window resizing
  useEffect(() => {
    const onResize = () => {
      if (!mountRef.current || !stageRef.current) return;
      stageRef.current.resize(
        mountRef.current.clientWidth,
        mountRef.current.clientHeight
      );
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Compute live proof numbers from real forward pass to feed the BookReader card
  const liveCardLines = useMemo(() => {
    if (!baseline) return [];
    const layer: ActiveLayerTrace | undefined = baseline.layers[state.layer];
    if (!layer) return [];

    const numTokens = baseline.tokens.length;
    const qIdx = clampInt(
      state.queryIdx ?? numTokens - 1,
      0,
      Math.max(0, numTokens - 1)
    );
    const kIdx = clampInt(
      state.keyIdx ?? Math.max(0, qIdx - 1),
      0,
      Math.max(0, numTokens - 1)
    );

    const qTok = formatToken(baseline.tokens[qIdx]?.text ?? '');
    const kTok = formatToken(baseline.tokens[kIdx]?.text ?? '');

    const qVec = layer.attn.q[state.head][qIdx] ?? [];
    const kVec = layer.attn.k[state.head][kIdx] ?? [];
    const breakdown = computeDotProductBreakdown(qVec, kVec);

    const probs = layer.attn.probs[state.head][qIdx] ?? [];
    const pVal = probs[kIdx] ?? 0;

    switch (activeSceneBeat) {
      case 'attn-1-problem':
        return [
          `Active Token [${qIdx}]: "${qTok}" · Block ${state.layer} · Head ${state.head}`,
          `||Q||_2 = ${formatNum(Math.hypot(...qVec))} · ||K||_2 = ${formatNum(
            Math.hypot(...kVec)
          )} · ||V||_2 = ${formatNum(
            Math.hypot(...(layer.attn.v[state.head][qIdx] ?? []))
          )}`,
          `Residual dim 64 projected to 3 × 16 per head via real verified weights.`,
        ];
      case 'attn-2-dot-product':
        return [
          `Q("${qTok}") · K("${kTok}") elementwise dot product across 16 coords:`,
          `Positive coord products: +${formatNum(breakdown.positiveVotes)}`,
          `Negative coord products: ${formatNum(breakdown.negativeVotes)}`,
          `Sum = ${formatNum(breakdown.rawDot)} · Scaled (÷4) = ${formatNum(
            breakdown.scaledScore
          )}`,
        ];
      case 'attn-3-scan-row':
        return [
          `Query "${qTok}" scanning all ${qIdx + 1} past positions simultaneously:`,
          `Scale Divisor: √16 = 4.000`,
          `Max raw dot: ${formatNum(
            Math.max(...(layer.attn.preSoftmax[state.head][qIdx] ?? []).map((x) => x * 4))
          )} → Max scaled logit: ${formatNum(
            Math.max(...(layer.attn.preSoftmax[state.head][qIdx] ?? []))
          )}`,
        ];
      case 'attn-4-mask-softmax':
        return [
          `Row ${qIdx} ("${qTok}") Attention Budget Distribution:`,
          `Attention to "${kTok}" [pos ${kIdx}]: ${(pVal * 100).toFixed(2)}%`,
          `Future positions (${qIdx + 1}..${numTokens - 1}): Forced to -∞ (0.00% attention)`,
          `Row Sum: ${(probs.reduce((a, b) => a + b, 0) * 100).toFixed(2)}% (exact probability simplex)`,
        ];
      case 'attn-5-value-mixer':
        return [
          `Value transport into "${qTok}" context vector:`,
          `Weight on "${kTok}": ${(pVal * 100).toFixed(1)}% of V("${kTok}")`,
          `Resulting HeadOut [16]: [${(layer.attn.out[state.head][qIdx] ?? [])
            .slice(0, 4)
            .map((x) => formatNum(x, 2))
            .join(', ')}, ...]`,
        ];
      case 'attn-6-multihead-full': {
        const topTok = baseline.logits
          ? Object.entries(baseline.logits)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([tok, logit]) => `"${tok}": ${formatNum(logit, 1)}`)
              .join(', ')
          : '';
        const isOff = isHeadDisabled(state.ablation, state.layer, state.head);
        return [
          `Block ${state.layer} 4-Head Output (4 × 16 = 64 dimensions):`,
          `Head ${state.head} status: ${isOff ? 'ABLATED (zeroed out)' : 'Active'}`,
          topTok ? `Next Token Logits: ${topTok}` : 'Inference ready.',
        ];
      }
    }
  }, [baseline, state.layer, state.head, state.queryIdx, state.keyIdx, activeSceneBeat, state.ablation]);

  const tokens = baseline?.tokens ?? [];
  const numTokens = tokens.length;
  const resolvedQ = clampInt(
    state.queryIdx ?? numTokens - 1,
    0,
    Math.max(0, numTokens - 1)
  );
  const resolvedK = clampInt(
    state.keyIdx ?? Math.max(0, resolvedQ - 1),
    0,
    Math.max(0, numTokens - 1)
  );

  const headAblated = isHeadDisabled(state.ablation, state.layer, state.head);

  return (
    <div className="app-root">
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
      <div className={`main-workspace ${state.lectureMode ? 'is-lecture' : ''}`}>
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
                  className={`mini-pill ${headAblated ? 'is-warn' : ''}`}
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
                  {headAblated ? `Re-enable H${state.head}` : `Ablate H${state.head}`}
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
          {tokens.length > 0 && (
            <div className="stage-bottom-hud">
              <div className="token-strip" aria-label="Select active token">
                <span className="hud-title">
                  {activeSceneBeat === 'attn-2-dot-product'
                    ? 'Comparing Key:'
                    : 'Query Token:'}
                </span>
                {tokens.map((tok, idx) => {
                  const isQ = idx === resolvedQ;
                  const isK =
                    activeSceneBeat === 'attn-2-dot-product' && idx === resolvedK;
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
