import { useEffect, useMemo, useRef, useState } from 'react';
import {
  type BookState,
  DEFAULT_PROMPT,
  isHeadDisabled,
  hasActiveAblation,
  stateFromUrl,
  stateToUrl,
  toggleHeadDisabled,
} from './state/bookState';
import { ALL_CHAPTERS, type Chapter } from './content/bookData';
import { BookReader } from './components/BookReader';
import { TextbookStage } from './scene/TextbookStage';
import { TransformerEngineClient } from './engine/client';
import type { ModelTrace } from './engine/types';
import { formatToken } from './utils/math';

export function App() {
  const [state, setState] = useState<BookState>(() =>
    stateFromUrl(window.location.search)
  );

  const [inputVal, setInputVal] = useState<string>(state.prompt);
  const [activeTrace, setActiveTrace] = useState<ModelTrace | null>(null);
  const [engineReady, setEngineReady] = useState<boolean>(false);
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const clientRef = useRef<TransformerEngineClient | null>(null);
  const stageRef = useRef<TextbookStage | null>(null);
  const stageContainerRef = useRef<HTMLDivElement | null>(null);

  // Initialize Web Worker Client & Load Weights
  useEffect(() => {
    let unmounted = false;
    const client = new TransformerEngineClient();
    clientRef.current = client;

    const base = import.meta.env.BASE_URL.replace(/\/$/, '');
    const weightsUrl = `${base}/model/weights.fp16.bin`;
    const manifestUrl = `${base}/model/manifest.json`;
    const tokenizerUrl = `${base}/model/tokenizer.json`;

    async function load() {
      try {
        await client.loadModel(weightsUrl, manifestUrl, tokenizerUrl);
        if (!unmounted) {
          setEngineReady(true);
        }
      } catch (err: unknown) {
        if (!unmounted) {
          setErrorMessage(
            err instanceof Error ? err.message : 'Failed to load model weights'
          );
        }
      }
    }

    load();

    return () => {
      unmounted = true;
      client.terminate();
    };
  }, []);

  // Sync state to URL hash/query
  useEffect(() => {
    const url = stateToUrl(state);
    window.history.replaceState(null, '', url);
  }, [state]);

  // Run Inference when prompt, layer, or ablation changes
  useEffect(() => {
    if (!engineReady || !clientRef.current) return;

    let cancelled = false;
    setIsInferring(true);
    setErrorMessage(null);

    clientRef.current
      .run(state.prompt, {
        ablation: state.ablation,
        temperature: 0.0,
      })
      .then((trace) => {
        if (!cancelled) {
          setActiveTrace(trace);
          setIsInferring(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setIsInferring(false);
          setErrorMessage(
            err instanceof Error ? err.message : 'Inference failed'
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [engineReady, state.prompt, state.ablation]);

  // Initialize Three.js Stage
  useEffect(() => {
    if (!stageContainerRef.current) return;
    const stage = new TextbookStage(stageContainerRef.current);
    stageRef.current = stage;

    stage.setSelectTokenCallback((qIdx, kIdx) => {
      setState((prev) => ({
        ...prev,
        queryIdx: qIdx,
        keyIdx: kIdx,
      }));
    });

    const handleResize = () => {
      if (!stageContainerRef.current) return;
      stage.resize(
        stageContainerRef.current.clientWidth,
        stageContainerRef.current.clientHeight
      );
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      stage.destroy();
      stageRef.current = null;
    };
  }, []);

  // Update Three.js Stage Content
  const activeChapter: Chapter = useMemo(
    () => ALL_CHAPTERS.find((c) => c.id === state.chapterId) ?? ALL_CHAPTERS[3],
    [state.chapterId]
  );

  const activeBeat = useMemo(
    () => activeChapter.beats[state.beatIndex] ?? activeChapter.beats[0],
    [activeChapter, state.beatIndex]
  );

  const activeSceneBeat = activeBeat?.stageFocus ?? 'attn-1-qkv-projection';

  useEffect(() => {
    if (!stageRef.current) return;
    stageRef.current.update({
      state,
      trace: activeTrace,
      activeSceneBeat,
    });
  }, [state, activeTrace, activeSceneBeat]);

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
          const ch = ALL_CHAPTERS.find((c) => c.id === prev.chapterId);
          if (!ch) return prev;
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
    if (!inputVal.trim()) return;
    setState((prev) => ({
      ...prev,
      prompt: inputVal.trim(),
      queryIdx: null,
      keyIdx: null,
    }));
  };

  const handleResetPrompt = () => {
    setInputVal(DEFAULT_PROMPT);
    setState((prev) => ({
      ...prev,
      prompt: DEFAULT_PROMPT,
      queryIdx: null,
      keyIdx: null,
    }));
  };

  const headOff = isHeadDisabled(state.ablation, state.layer, state.head);

  const resolvedQueryIdx = Math.max(
    0,
    Math.min(
      (activeTrace?.tokens.length ?? 1) - 1,
      state.queryIdx ?? (activeTrace?.tokens.length ?? 1) - 1
    )
  );

  const resolvedKeyIdx = Math.max(
    0,
    Math.min((activeTrace?.tokens.length ?? 1) - 1, state.keyIdx ?? resolvedQueryIdx)
  );

  return (
    <div className="app-root">
      {/* Top Header Bar */}
      <header className="top-nav">
        <div className="brand-section">
          <span className="brand-title">LLM Atlas</span>
          <span className="brand-badge">Interactive Textbook</span>
        </div>

        <form className="top-prompt-area" onSubmit={handlePromptSubmit}>
          <input
            type="text"
            className="prompt-input mono"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            placeholder="Type any prompt..."
          />
          <button
            type="submit"
            className="btn-run"
            disabled={!engineReady || isInferring}
          >
            {isInferring ? 'Thinking...' : 'Run'}
          </button>
          {inputVal !== DEFAULT_PROMPT && (
            <button
              type="button"
              className="btn-mode"
              onClick={handleResetPrompt}
            >
              Reset Prompt
            </button>
          )}
        </form>

        <div className="top-actions">
          <button
            type="button"
            className={`btn-mode ${state.lectureMode ? 'is-active' : ''}`}
            onClick={() =>
              setState((prev) => ({ ...prev, lectureMode: !prev.lectureMode }))
            }
          >
            {state.lectureMode ? 'Exit Lecture Mode' : 'Lecture Mode'}
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

      {/* Main Workspace */}
      <div className={`main-workspace ${state.lectureMode ? 'is-lecture' : ''}`}>
        {/* Left Pane: Book Reader */}
        {!state.lectureMode && (
          <aside className="reader-pane">
            <BookReader
              state={state}
              trace={activeTrace}
              onUpdateState={setState}
            />
          </aside>
        )}

        {/* Right Pane: 3D Textbook Stage */}
        <section className="stage-pane" aria-label="3D Interactive Stage">
          <div ref={stageContainerRef} className="three-stage-canvas-container" />

          {/* Stage HUD Top */}
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

              {activeSceneBeat === 'attn-4-mask-softmax' && (\n                <button\n                  type=\"button\"\n                  className={`mini-btn ${state.showPreSoftmax ? 'is-active' : ''}`}\n                  onClick={() =>\n                    setState((prev) => ({\n                      ...prev,\n                      showPreSoftmax: !prev.showPreSoftmax,\n                    }))\n                  }\n                >\n                  {state.showPreSoftmax ? 'Showing: Raw Scores' : 'Showing: Softmax %'}\n                </button>\n              )}

              {(activeSceneBeat === 'attn-5-value-mixer' ||
                activeSceneBeat === 'attn-6-multihead-full') && (
                <button
                  type="button"
                  className={`mini-btn ${headOff ? 'is-warm' : ''}`}
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
                  {headOff ? `Re-enable H${state.head}` : `Disable H${state.head}`}
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

          {activeTrace && (
            <div className="stage-hud-bottom">
              <div className="token-bar" aria-label="Select Query or Key token">
                <span className="hud-label">
                  {activeSceneBeat === 'attn-2-dot-product'
                    ? 'Compare Key:'
                    : 'Query Token:'}
                </span>
                {activeTrace.tokens.map((tok, idx) => {
                  const isQ = idx === resolvedQueryIdx;
                  const isK =
                    activeSceneBeat === 'attn-2-dot-product' &&
                    idx === resolvedKeyIdx;
                  const disabledKey =
                    activeSceneBeat === 'attn-2-dot-product' &&
                    idx > resolvedQueryIdx;
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
                          activeSceneBeat === 'attn-2-dot-product'
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
                Drag to orbit · Right-drag or two-finger drag to pan · Scroll to zoom · Left/Right keys step beats
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
