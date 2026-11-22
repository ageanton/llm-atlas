import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  formatRegisterName,
  LEVEL_SPECS,
  PRESET_PROMPTS,
} from './content/atlasContent';
import { GUIDED_TOURS } from './content/tours';
import { TransformerEngineClient } from './engine/client';
import type {
  AblationConfig,
  EngineInitResult,
  EngineTraceResult,
} from './engine/types';
import { ExplanationPanel } from './panels/ExplanationPanel';
import {
  TeachingModePanel,
  type TrainingReplayData,
} from './panels/TeachingModes';
import { Level0View, Level1View, Level2View } from './scene/Levels012';
import { Level3View, Level4View, Level5View } from './scene/Levels345';
import {
  clampLevel,
  hasActiveAblation,
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

  // Initialize worker + load real weights & training replay
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

  // Re-run real forward pass whenever prompt or ablation changes
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

  const setLevel = useCallback((lv: ZoomLevel) => {
    setState((prev) => ({ ...prev, level: lv }));
  }, []);

  const setRegister = useCallback((register: Register) => {
    setState((prev) => ({ ...prev, register }));
  }, []);

  const setMode = useCallback((mode: TeachingMode) => {
    setState((prev) => ({ ...prev, mode }));
  }, []);

  const handleSelect = useCallback((sel: string, zoomIn = false) => {
    setState((prev) => ({
      ...prev,
      sel,
      level: zoomIn ? clampLevel(prev.level + 1) : prev.level,
    }));
  }, []);

  const handleAblationChange = useCallback((ablation: AblationConfig) => {
    setState((prev) => ({ ...prev, ablation }));
  }, []);

  // Keyboard navigation across the semantic zoom tree
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      if (
        activeEl instanceof HTMLInputElement ||
        activeEl instanceof HTMLTextAreaElement ||
        activeEl instanceof HTMLSelectElement
      ) {
        return;
      }

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
          setState((prev) => ({ ...prev, tourId: null, tourStep: 0 }));
        }
      }
    };

    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [state.level, state.mode, state.tourId, setLevel, setMode]);

  // Semantic zoom via mouse wheel with rate-limiting lock
  const handleSceneWheel = (e: React.WheelEvent) => {
    if (Math.abs(e.deltaY) < 35) return;
    const now = Date.now();
    if (now - wheelLockRef.current < 280) return;
    wheelLockRef.current = now;

    if (e.deltaY > 0 && state.level < 5) {
      setLevel(clampLevel(state.level + 1));
    } else if (e.deltaY < 0 && state.level > 0) {
      setLevel(clampLevel(state.level - 1));
    }
  };

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (draftPrompt.trim()) {
      setState((prev) => ({ ...prev, prompt: draftPrompt.trim() }));
    }
  };

  const applyPreset = (text: string) => {
    setDraftPrompt(text);
    setState((prev) => ({ ...prev, prompt: text }));
  };

  const copyShareUrl = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    } catch {
      // Ignore clipboard write failures
    }
  };

  const activeTour = useMemo(() => {
    if (!state.tourId) return null;
    return GUIDED_TOURS.find((t) => t.id === state.tourId) ?? null;
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
      prompt: first.prompt ?? prev.prompt,
    }));
    if (first.prompt) setDraftPrompt(first.prompt);
  };

  const advanceTour = (delta: number) => {
    if (!activeTour) return;
    const nextIdx = state.tourStep + delta;
    if (nextIdx < 0) return;
    if (nextIdx >= activeTour.steps.length) {
      setState((prev) => ({ ...prev, tourId: null, tourStep: 0 }));
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
      prompt: step.prompt ?? prev.prompt,
    }));
    if (step.prompt) setDraftPrompt(step.prompt);
  };

  const target = useMemo(() => parseSelection(state.sel), [state.sel]);
  const activeTrace = ablated || baseline;

  return (
    <div className="atlas-app">
      <header className="atlas-topbar">
        <div className="topbar-left">
          <span className="atlas-brand">LLM Atlas</span>
          <span className="truth-badge mono">
            {initInfo
              ? `${initInfo.manifest.total_params.toLocaleString()} params · float16 truth`
              : '866K params · real weights'}
          </span>
        </div>

        {activeTour ? (
          <div className="topbar-tour">
            <span className="tour-badge mono">
              Tour: {activeTour.title} ({state.tourStep + 1}/{activeTour.steps.length})
            </span>
            <button
              type="button"
              className="seg-btn"
              disabled={state.tourStep === 0}
              onClick={() => advanceTour(-1)}
            >
              ← Prev
            </button>
            <button
              type="button"
              className="seg-btn is-active"
              onClick={() => advanceTour(1)}
            >
              {state.tourStep === activeTour.steps.length - 1 ? 'Finish' : 'Next →'}
            </button>
            <button
              type="button"
              className="seg-btn"
              onClick={() => setState((prev) => ({ ...prev, tourId: null, tourStep: 0 }))}
            >
              Exit
            </button>
          </div>
        ) : (
          <div className="topbar-tour">
            <select
              className="tour-select"
              value=""
              onChange={(e) => {
                if (e.target.value) startTour(e.target.value);
              }}
            >
              <option value="">Start a guided tour...</option>
              {GUIDED_TOURS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.title}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="topbar-right">
          <button
            type="button"
            className="seg-btn"
            onClick={copyShareUrl}
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

      <nav className="atlas-subbar" aria-label="Explorer navigation">
        <div className="nav-group zoom-group">
          <span className="group-label mono">Level:</span>
          {LEVELS.map((lv) => (
            <button
              key={lv}
              type="button"
              className={`seg-btn mono ${state.level === lv ? 'is-active' : ''}`}
              onClick={() => setLevel(lv)}
            >
              L{lv}
            </button>
          ))}
        </div>

        <div className="nav-group mode-group">
          <span className="group-label mono">Mode:</span>
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

        <div className="nav-group register-group">
          <span className="group-label mono">Register:</span>
          {REGISTERS.map((r) => (
            <button
              key={r}
              type="button"
              className={`seg-btn mono ${state.register === r ? 'is-active' : ''}`}
              onClick={() => setRegister(r)}
            >
              {formatRegisterName(r)}
            </button>
          ))}
        </div>
      </nav>

      <div className="prompt-toolbar">
        <div className="prompt-presets">
          <span className="preset-label mono">Presets:</span>
          {PRESET_PROMPTS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-pill ${state.prompt === p.text ? 'is-active' : ''}`}
              onClick={() => applyPreset(p.text)}
              title={p.note}
            >
              {p.label}
            </button>
          ))}
        </div>

        <form onSubmit={handlePromptSubmit} className="prompt-form">
          <input
            className="prompt-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type your own story prompt..."
            aria-label="Prompt text"
          />
          <button type="submit" className="seg-btn is-active" disabled={busy}>
            {busy ? 'Running...' : 'Run'}
          </button>
        </form>

        {activeTrace && (
          <div className="live-token-ribbon">
            <span className="ribbon-label mono">Input:</span>
            {activeTrace.tokens.map((tok, idx) => (
              <span
                key={`${tok.start}-${idx}`}
                className={`ribbon-token mono ${hoveredTokenIdx === idx ? 'is-hovered' : ''}`}
                onMouseEnter={() => setHoveredTokenIdx(idx)}
                onMouseLeave={() => setHoveredTokenIdx(null)}
                title={`Token #${tok.id}, characters ${tok.start}..${tok.end}`}
              >
                {formatToken(tok.text)}
              </span>
            ))}
            <span className="l0-meta mono">
              {activeTrace.elapsedMs}ms ·{' '}
              {hasActiveAblation(state.ablation) ? 'ablated trace' : 'real forward pass'}
            </span>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="atlas-workspace">
        <nav className="level-rail" aria-label="Semantic zoom levels">
          {LEVELS.map((lv) => (
            <button
              key={lv}
              type="button"
              className={`rail-btn ${state.level === lv ? 'is-current' : ''}`}
              onClick={() => setLevel(lv)}
            >
              {LEVEL_SPECS[lv].shortName}
            </button>
          ))}
        </nav>

        <main className="scene-column" onWheel={handleSceneWheel}>
          {!initInfo || !activeTrace || !baseline ? (
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
