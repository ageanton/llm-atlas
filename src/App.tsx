import { useEffect, useMemo, useRef, useState } from 'react';
import { GUIDED_TOURS } from './content/tours';
import { PRESET_PROMPTS } from './content/atlasContent';
import { createEngineClient, type TransformerEngineClient } from './engine/client';
import type {
  AblationConfig,
  EngineInitResult,
  EngineTraceResult,
  TrainingReplayData,
} from './engine/types';
import { ExplanationPanel } from './panels/ExplanationPanel';
import { TeachingModePanel } from './panels/TeachingModes';
import { Level0View, Level1View, Level2View } from './scene/Levels012';
import { Level3View, Level4View, Level5View } from './scene/Levels345';
import {
  clampLevel,
  DEFAULT_PROMPT,
  DEFAULT_SELECTION,
  parseSelection,
  stateFromUrl,
  stateToSearch,
  type AtlasState,
  type Register,
  type TeachingMode,
  type ZoomLevel,
} from './state/urlState';
import { formatToken } from './utils/color';

const LEVELS: ZoomLevel[] = [0, 1, 2, 3, 4, 5];
const REGISTERS: Register[] = ['simple', 'sufficient', 'complete'];

export function App() {
  const clientRef = useRef<TransformerEngineClient | null>(null);
  const wheelLockRef = useRef<number>(0);

  const [state, setState] = useState<AtlasState>(() =>
    stateFromUrl(window.location.search)
  );
  const [draftPrompt, setDraftPrompt] = useState<string>(state.prompt);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [replay, setReplay] = useState<TrainingReplayData | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sync state to URL cleanly
  useEffect(() => {
    const nextSearch = stateToSearch(state);
    if (window.location.search !== nextSearch) {
      window.history.replaceState(null, '', `${window.location.pathname}${nextSearch}`);
    }
  }, [state]);

  // Handle browser back/forward buttons
  useEffect(() => {
    const onPopState = () => {
      setState(stateFromUrl(window.location.search));
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  // Initialize Web Worker and load real model
  useEffect(() => {
    let client: TransformerEngineClient;
    try {
      client = createEngineClient();
      clientRef.current = client;
    } catch (err) {
      setError(`Failed to create worker: ${(err as Error).message}`);
      return;
    }

    client
      .init()
      .then((info) => {
        setInitInfo(info);
      })
      .catch((err) => {
        setError(`Failed to load real model weights: ${(err as Error).message}`);
      });

    client
      .getReplay()
      .then((data: TrainingReplayData | null) => {
        setReplay(data);
      })
      .catch(() => undefined);

    return () => {
      client.terminate();
    };
  }, []);

  // Run forward pass whenever prompt changes
  useEffect(() => {
    const client = clientRef.current;
    if (!client || !initInfo || !state.prompt) return;

    let cancelled = false;
    setBusy(true);

    client
      .run(state.prompt)
      .then((res) => {
        if (!cancelled) {
          setBaseline(res);
          setBusy(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(`Inference failed: ${(err as Error).message}`);
          setBusy(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [initInfo, state.prompt]);

  // Run ablated forward pass when ablation changes
  useEffect(() => {
    const client = clientRef.current;
    if (!client || !initInfo || !state.prompt) return;

    const hasAb =
      Boolean(state.ablation.zeroBlocks?.length) ||
      Boolean(state.ablation.zeroMlpLayers?.length) ||
      Boolean(state.ablation.zeroHeads?.length) ||
      state.ablation.freezePosition !== undefined;

    if (!hasAb) {
      setAblated(null);
      return;
    }

    let cancelled = false;
    client
      .run(state.prompt, state.ablation)
      .then((res) => {
        if (!cancelled) setAblated(res);
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, [initInfo, state.prompt, state.ablation]);

  // Keyboard navigation
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault();
        setLevel(clampLevel(state.level - 1));
      } else if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault();
        setLevel(clampLevel(state.level + 1));
      } else if (['0', '1', '2', '3', '4', '5'].includes(e.key)) {
        e.preventDefault();
        setLevel(Number.parseInt(e.key, 10) as ZoomLevel);
      } else if (e.key === 'Escape') {
        if (state.mode !== 'explore') {
          e.preventDefault();
          setMode('explore');
        } else if (state.tourId) {
          e.preventDefault();
          endTour();
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [state.level, state.mode, state.tourId]);

  // Semantic Zoom on wheel (locked with debounce)
  const handleWheel = (e: React.WheelEvent) => {
    if (Math.abs(e.deltaY) < 40) return;
    const now = Date.now();
    if (now - wheelLockRef.current < 260) return;
    wheelLockRef.current = now;

    if (e.deltaY > 0 && state.level < 5) {
      setLevel((state.level + 1) as ZoomLevel);
    } else if (e.deltaY < 0 && state.level > 0) {
      setLevel((state.level - 1) as ZoomLevel);
    }
  };

  const setLevel = (lv: ZoomLevel) => {
    setState((prev) => ({ ...prev, level: lv }));
  };

  const setRegister = (r: Register) => {
    setState((prev) => ({ ...prev, register: r }));
  };

  const setMode = (m: TeachingMode) => {
    setState((prev) => ({ ...prev, mode: m }));
  };

  const handleSelect = (sel: string, zoomLevel5 = false) => {
    setState((prev) => ({
      ...prev,
      sel,
      level: zoomLevel5 ? 5 : prev.level,
    }));
  };

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (draftPrompt.trim()) {
      setState((prev) => ({ ...prev, prompt: draftPrompt.trim() }));
    }
  };

  const selectPreset = (p: string) => {
    setDraftPrompt(p);
    setState((prev) => ({ ...prev, prompt: p }));
  };

  const handleAblationChange = (next: AblationConfig) => {
    setState((prev) => ({ ...prev, ablation: next }));
  };

  const copyShareLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    } catch {
      // Fallback
    }
  };

  const activeTour = useMemo(() => {
    if (!state.tourId) return null;
    return GUIDED_TOURS.find((t) => t.id === state.tourId) || null;
  }, [state.tourId]);

  const startTour = (tourId: string) => {
    const tour = GUIDED_TOURS.find((t) => t.id === tourId);
    if (!tour || tour.steps.length === 0) return;
    const first = tour.steps[0];
    setState((prev) => ({
      ...prev,
      tourId,
      tourStep: 0,
      level: first.level,
      sel: first.sel,
      register: first.register,
      mode: first.mode,
      prompt: first.prompt || prev.prompt,
    }));
    if (first.prompt) setDraftPrompt(first.prompt);
  };

  const nextTourStep = () => {
    if (!activeTour) return;
    const nextIdx = state.tourStep + 1;
    if (nextIdx >= activeTour.steps.length) {
      endTour();
      return;
    }
    const step = activeTour.steps[nextIdx];
    setState((prev) => ({
      ...prev,
      tourStep: nextIdx,
      level: step.level,
      sel: step.sel,
      register: step.register,
      mode: step.mode,
      prompt: step.prompt || prev.prompt,
    }));
    if (step.prompt) setDraftPrompt(step.prompt);
  };

  const prevTourStep = () => {
    if (!activeTour || state.tourStep <= 0) return;
    const prevIdx = state.tourStep - 1;
    const step = activeTour.steps[prevIdx];
    setState((prev) => ({
      ...prev,
      tourStep: prevIdx,
      level: step.level,
      sel: step.sel,
      register: step.register,
      mode: step.mode,
      prompt: step.prompt || prev.prompt,
    }));
    if (step.prompt) setDraftPrompt(step.prompt);
  };

  const endTour = () => {
    setState((prev) => ({ ...prev, tourId: null, tourStep: 0 }));
  };

  const target = useMemo(() => parseSelection(state.sel), [state.sel]);
  const activeTrace = ablated || baseline;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-brand">
          <span className="app-title">LLM Atlas</span>
          <span className="param-badge mono">866K Parameters · Real Weights</span>
        </div>

        {activeTour ? (
          <div className="header-tour-controls">
            <div className="tour-nav-box">
              <span className="badge-chip mono">Tour: {activeTour.title}</span>
              <span className="mono">
                Step {state.tourStep + 1} of {activeTour.steps.length}
              </span>
              <button
                type="button"
                className="seg-btn"
                onClick={prevTourStep}
                disabled={state.tourStep === 0}
              >
                ← Back
              </button>
              <button
                type="button"
                className="seg-btn is-active"
                onClick={nextTourStep}
              >
                {state.tourStep === activeTour.steps.length - 1 ? 'Finish' : 'Next →'}
              </button>
              <button type="button" className="seg-btn" onClick={endTour}>
                Exit Tour
              </button>
            </div>
            <p className="tour-step-msg">
              {activeTour.steps[state.tourStep]?.message}
            </p>
          </div>
        ) : (
          <div className="header-tour-controls">
            <select
              className="select-input"
              value=""
              onChange={(e) => {
                if (e.target.value) startTour(e.target.value);
              }}
            >
              <option value="">Start Guided Tour...</option>
              {GUIDED_TOURS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.title} ({t.steps.length} steps)
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="header-actions">
          <button
            type="button"
            className="seg-btn"
            onClick={copyShareLink}
            title="Copy deep-linked URL to clipboard"
          >
            {copiedLink ? 'Link Copied!' : 'Share URL'}
          </button>
          <a
            href="https://github.com/ageanton/llm-atlas"
            target="_blank"
            rel="noreferrer"
            className="seg-btn"
          >
            GitHub
          </a>
        </div>
      </header>

      <nav className="app-toolbar" aria-label="Explorer navigation">
        <div className="zoom-level-nav">
          <span className="nav-label mono">Zoom Level:</span>
          {LEVELS.map((lv) => (
            <button
              key={lv}
              type="button"
              className={`seg-btn mono ${state.level === lv ? 'is-active' : ''}`}
              onClick={() => setLevel(lv)}
              title={`Zoom to Level ${lv}`}
            >
              L{lv}
            </button>
          ))}
        </div>

        <div className="teaching-mode-nav">
          <span className="nav-label mono">Mode:</span>
          {(['explore', 'ablate', 'replay', 'sampling', 'inspector', 'scale'] as TeachingMode[]).map(
            (m) => (
              <button
                key={m}
                type="button"
                className={`seg-btn mono ${state.mode === m ? 'is-active' : ''}`}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            )
          )}
        </div>

        <div className="register-nav">
          <span className="nav-label mono">Register:</span>
          {REGISTERS.map((r) => (
            <button
              key={r}
              type="button"
              className={`seg-btn mono ${state.register === r ? 'is-active' : ''}`}
              onClick={() => setRegister(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </nav>

      <section className="prompt-bar" aria-label="Input prompt">
        <span className="prompt-bar-label mono">Presets:</span>
        <div className="prompt-chips-row">
          {PRESET_PROMPTS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-chip ${state.prompt === p.text ? 'is-active' : ''}`}
              onClick={() => selectPreset(p.text)}
              title={p.note}
            >
              {p.label}
            </button>
          ))}
        </div>

        <form onSubmit={handlePromptSubmit} className="prompt-custom-form">
          <input
            type="text"
            className="prompt-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type your own story prompt..."
          />
          <button type="submit" className="prompt-submit-btn" disabled={busy}>
            {busy ? 'Running...' : 'Run'}
          </button>
        </form>
      </section>

      {error && (
        <div className="error-banner" role="alert">
          <b>Error:</b> {error}
        </div>
      )}

      <div className="app-main-layout">
        <main
          className="scene-viewport"
          onWheel={handleWheel}
          aria-label="Interactive Transformer Scene"
        >
          {!initInfo || !activeTrace ? (
            <div className="empty-notice">
              Loading real miniature transformer weights (1.65 MB)...
            </div>
          ) : (
            <>
              {state.mode !== 'explore' && clientRef.current && (
                <TeachingModePanel
                  mode={state.mode}
                  ablation={state.ablation}
                  onChangeAblation={handleAblationChange}
                  baseline={baseline}
                  ablated={ablated}
                  replay={replay}
                  manifest={initInfo.manifest}
                  tokenizer={initInfo.tokenizer}
                  client={clientRef.current}
                  onInspectScalar={(sel) => {
                    setState((prev) => ({
                      ...prev,
                      mode: 'explore',
                      level: 5,
                      sel,
                    }));
                  }}
                  onClose={() => setMode('explore')}
                />
              )}

              {state.level === 0 && (
                <Level0View
                  tokens={activeTrace.tokens}
                  trace={activeTrace.trace}
                  manifest={initInfo.manifest}
                  tokenizer={initInfo.tokenizer}
                  register={state.register}
                  prompt={state.prompt}
                  onZoomIn={() => setLevel(1)}
                />
              )}

              {state.level === 1 && (
                <Level1View
                  tokens={activeTrace.tokens}
                  trace={activeTrace.trace}
                  manifest={initInfo.manifest}
                  tokenizer={initInfo.tokenizer}
                  target={target}
                  register={state.register}
                  onSelect={handleSelect}
                />
              )}

              {state.level === 2 && (
                <Level2View
                  tokens={activeTrace.tokens}
                  trace={activeTrace.trace}
                  manifest={initInfo.manifest}
                  target={target}
                  register={state.register}
                  onSelect={handleSelect}
                />
              )}

              {state.level === 3 && (
                <Level3View
                  tokens={activeTrace.tokens}
                  trace={activeTrace.trace}
                  target={target}
                  register={state.register}
                  hoveredTokenIdx={hoveredTokenIdx}
                  onHoverToken={setHoveredTokenIdx}
                  onSelect={handleSelect}
                />
              )}

              {state.level === 4 && clientRef.current && (
                <Level4View
                  tokens={activeTrace.tokens}
                  trace={activeTrace.trace}
                  manifest={initInfo.manifest}
                  client={clientRef.current}
                  target={target}
                  onSelect={handleSelect}
                />
              )}

              {state.level === 5 && clientRef.current && (
                <Level5View
                  target={target}
                  manifest={initInfo.manifest}
                  client={clientRef.current}
                  replay={replay}
                  onSelect={handleSelect}
                  onOpenReplay={() => setMode('replay')}
                />
              )}
            </>
          )}
        </main>

        <ExplanationPanel
          level={state.level}
          register={state.register}
          mobileOpen={mobileSheetOpen}
          onToggleMobile={() => setMobileSheetOpen((v) => !v)}
          onJumpLevel={setLevel}
        />
      </div>
    </div>
  );
}
