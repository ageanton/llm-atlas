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
  const [initError, setInitError] = useState<string | null>(null);
  const [trace, setTrace] = useState<EngineTraceResult | null>(null);
  const [inferring, setInferring] = useState<boolean>(false);
  const [inferError, setInferError] = useState<string | null>(null);

  // Sync state to URL without reloading
  useEffect(() => {
    const search = stateToSearch(state);
    const target = `${window.location.pathname}${search}${window.location.hash}`;
    window.history.replaceState(null, '', target);
  }, [state]);

  // Keep draftPrompt aligned when state.prompt changes externally
  useEffect(() => {
    setDraftPrompt(state.prompt);
  }, [state.prompt]);

  // Initialize Web Worker engine
  useEffect(() => {
    let unmounted = false;
    const client = new TransformerEngineClient();
    clientRef.current = client;

    client
      .init()
      .then((info) => {
        if (!unmounted) setInitInfo(info);
      })
      .catch((err) => {
        if (!unmounted) {
          setInitError(err instanceof Error ? err.message : String(err));
        }
      });

    return () => {
      unmounted = true;
      client.destroy();
      clientRef.current = null;
    };
  }, []);

  // Run forward pass whenever prompt or ablation changes
  useEffect(() => {
    const client = clientRef.current;
    if (!client || !initInfo) return;

    let cancelled = false;
    setInferring(true);
    setInferError(null);

    client
      .runTrace(state.prompt, {
        ablation: state.ablation,
        temperature: 0.0,
      })
      .then((res) => {
        if (!cancelled) {
          setTrace(res);
          setInferring(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setInferError(err instanceof Error ? err.message : String(err));
          setInferring(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [initInfo, state.prompt, state.ablation]);

  // Initialize Three.js Stage
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const stage = new TextbookStage(container);
    stageRef.current = stage;

    stage.setSelectTokenCallback((qIdx, kIdx) => {
      setState((prev) => ({
        ...prev,
        queryIdx: qIdx,
        keyIdx: kIdx,
      }));
    });

    const ro = new ResizeObserver(() => {
      if (container) {
        stage.resize(container.clientWidth, container.clientHeight);
      }
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      stage.destroy();
      stageRef.current = null;
    };
  }, []);

  // Active Chapter & Beat resolution
  const currentChapter = useMemo(
    () =>
      TEXTBOOK_CHAPTERS.find((c) => c.id === state.chapterId) ??
      TEXTBOOK_CHAPTERS[3],
    [state.chapterId]
  );

  const currentBeat = useMemo(
    () => currentChapter.beats[state.beatIndex] ?? currentChapter.beats[0],
    [currentChapter, state.beatIndex]
  );

  const activeBeatId: SceneBeatId = currentBeat?.sceneBeat ?? 'attn-1-problem';

  // Push updates to Three.js stage
  useEffect(() => {
    if (!stageRef.current) return;
    stageRef.current.update({
      state,
      trace,
      activeBeatId,
    });
  }, [state, trace, activeBeatId]);

  // Keyboard navigation: Left/Right arrow moves between beats
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      ) {
        return;
      }

      if (e.key === 'ArrowRight') {
        setState((prev) => {
          const ch =
            TEXTBOOK_CHAPTERS.find((c) => c.id === prev.chapterId) ??
            TEXTBOOK_CHAPTERS[3];
          if (prev.beatIndex < ch.beats.length - 1) {
            return { ...prev, beatIndex: prev.beatIndex + 1 };
          }
          return prev;
        });
      } else if (e.key === 'ArrowLeft') {
        setState((prev) => {
          if (prev.beatIndex > 0) {
            return { ...prev, beatIndex: prev.beatIndex - 1 };
          }
          return prev;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!draftPrompt.trim()) return;
    setState((prev) => ({
      ...prev,
      prompt: draftPrompt.trim(),
      queryIdx: null,
      keyIdx: null,
    }));
  };

  const handleSelectChapter = (id: number) => {
    setState((prev) => ({
      ...prev,
      chapterId: id,
      beatIndex: 0,
      queryIdx: null,
      keyIdx: null,
    }));
    setShowContents(false);
  };

  const tokens = trace?.tokens ?? [];
  const numTokens = tokens.length;
  const resolvedQ = Math.max(
    0,
    Math.min(numTokens - 1, state.queryIdx ?? numTokens - 1)
  );
  const resolvedK = Math.max(
    0,
    Math.min(numTokens - 1, state.keyIdx ?? resolvedQ)
  );

  const currentLayerTrace: ActiveLayerTrace | undefined =
    trace?.layers[state.layer];
  const qVec = currentLayerTrace?.attn.q[state.head][resolvedQ] ?? [];
  const kVec = currentLayerTrace?.attn.k[state.head][resolvedK] ?? [];
  const breakdown = computeDotProductBreakdown(qVec, kVec);

  const headAblated = isHeadDisabled(state.ablation, state.layer, state.head);

  return (
    <div className="app-root">
      {/* Top Navigation */}
      <header className="top-nav">
        <div className="brand-section">
          <span className="brand-title">LLM Atlas</span>
          <span className="brand-badge">Interactive Textbook</span>
        </div>

        <form className="top-prompt-area" onSubmit={handlePromptSubmit}>
          <input
            type="text"
            className="prompt-input mono"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type any prompt..."
            aria-label="Input prompt"
          />
          <button
            type="submit"
            className="btn-run"
            disabled={!initInfo || inferring}
          >
            {inferring ? 'Computing...' : 'Run'}
          </button>
        </form>

        <div className="top-actions">
          <button
            type="button"
            className={`btn-mode ${state.lectureMode ? 'is-active' : ''}`}
            onClick={() =>
              setState((prev) => ({
                ...prev,
                lectureMode: !prev.lectureMode,
              }))
            }
          >
            {state.lectureMode ? 'Exit Lecture' : 'Lecture Mode'}
          </button>

          <button
            type="button"
            className="btn-mode"
            onClick={() => setShowContents(true)}
          >
            Contents
          </button>

          <a
            href="https://github.com/ageanton/llm-atlas"
            target="_blank"
            rel="noreferrer"
            className="btn-mode"
          >
            GitHub
          </a>
        </div>
      </header>

      {/* Table of Contents Modal */}
      {showContents && (
        <div className="modal-overlay" onClick={() => setShowContents(false)}>
          <div
            className="contents-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <header className="modal-header">
              <h2 className="modal-title">Table of Contents</h2>
              <button
                type="button"
                className="btn-close-modal"
                onClick={() => setShowContents(false)}
              >
                ✕
              </button>
            </header>
            <div className="modal-list">
              {TEXTBOOK_CHAPTERS.map((ch) => {
                const isActive = ch.id === state.chapterId;
                const isReady = ch.status === 'published';
                return (
                  <button
                    key={ch.id}
                    type="button"
                    disabled={!isReady}
                    className={`chapter-card-btn ${isActive ? 'is-active' : ''}`}
                    onClick={() => handleSelectChapter(ch.id)}
                  >
                    <div className="chapter-card-header">
                      <span className="chapter-num">Chapter {ch.id}</span>
                      <span
                        className={`chapter-status ${
                          isReady ? 'is-ready' : 'is-soon'
                        }`}
                      >
                        {isReady ? 'Available Now' : 'Coming Soon'}
                      </span>
                    </div>
                    <div className="chapter-card-title">{ch.title}</div>
                    <div className="chapter-card-teaser">{ch.teaser}</div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Main Workspace */}
      <div
        className={`main-workspace ${state.lectureMode ? 'is-lecture' : ''}`}
      >
        {/* Left Pane: Book Reader */}
        {!state.lectureMode && (
          <BookReader
            state={state}
            trace={trace}
            onUpdateState={setState}
            onOpenContents={() => setShowContents(true)}
          />
        )}

        {/* Right Pane: 3D Stage */}
        <section className="stage-pane" aria-label="3D Interactive Stage">
          <div ref={mountRef} className="three-stage-canvas-container" />

          {/* Top Stage Controls Overlay */}
          <div className="stage-hud-top">
            <div className="hud-group">
              <span className="hud-label">Layer</span>
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

              <span className="hud-label">Head</span>
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

              {activeBeatId === 'attn-4-mask-softmax' && (
                <button
                  type="button"
                  className={`mini-btn ${state.showPreSoftmax ? 'is-active' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      showPreSoftmax: !prev.showPreSoftmax,
                    }))
                  }
                >
                  {state.showPreSoftmax
                    ? 'Showing: Raw Scores'
                    : 'Showing: Softmax %'}
                </button>
              )}

              {(activeBeatId === 'attn-5-value-mixer' ||
                activeBeatId === 'attn-6-multihead-full') && (
                <button
                  type="button"
                  className={`mini-btn ${headAblated ? 'is-warm' : ''}`}
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
                  {headAblated
                    ? `Re-enable H${state.head}`
                    : `Disable H${state.head}`}
                </button>
              )}
            </div>

            <div className="hud-group">
              {hasActiveAblation(state.ablation) && (
                <button
                  type="button"
                  className="mini-btn is-warm"
                  onClick={() => setState((prev) => ({ ...prev, ablation: {} }))}
                >
                  Reset Ablation
                </button>
              )}
              <button
                type="button"
                className="mini-btn"
                onClick={() => stageRef.current?.resetCamera()}
              >
                Reset View
              </button>
            </div>
          </div>

          {/* Bottom Stage Token Selector */}
          {trace && numTokens > 0 && (
            <div className="stage-hud-bottom">
              <div className="token-bar" aria-label="Select Query or Key token">
                <span className="hud-label">
                  {activeBeatId === 'attn-2-dot-product'
                    ? 'Compare Key:'
                    : 'Query Token:'}
                </span>
                {tokens.map((tok, idx) => {
                  const isQ = idx === resolvedQ;
                  const isK =
                    activeBeatId === 'attn-2-dot-product' && idx === resolvedK;
                  const disabledKey =
                    activeBeatId === 'attn-2-dot-product' && idx > resolvedQ;
                  return (
                    <button
                      key={`${tok.start}-${idx}`}
                      type="button"
                      disabled={disabledKey}
                      className={`tok-btn mono ${isQ ? 'is-query' : ''} ${
                        isK ? 'is-key' : ''
                      }`}
                      onClick={() =>
                        setState((prev) =>
                          activeBeatId === 'attn-2-dot-product'
                            ? { ...prev, keyIdx: idx }
                            : { ...prev, queryIdx: idx, keyIdx: null }
                        )
                      }
                    >
                      {formatToken(tok.text)}
                    </button>
                  );
                })}
              </div>
              <div className="camera-hint">
                Drag to orbit · Right-drag or two-finger drag to pan · Scroll to
                zoom · Left/Right keys step beats
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
