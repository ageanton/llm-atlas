import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { NumbersOverlay } from './components/NumbersOverlay';
import {
  LEVEL_CONTENT,
  PRESET_PROMPTS,
  registerLabel,
} from './content/atlasContent';
import { TransformerEngineClient } from './engine/client';
import type { EngineInitResult, EngineTraceResult } from './engine/types';
import { studioSound } from './scene/sound';
import { TransformerWorld } from './scene/TransformerWorld';
import {
  clampLevel,
  hasActiveAblation,
  isHeadDisabled,
  stateFromUrl,
  stateToSearch,
  toggleHeadAblation,
  type AtlasState,
  type Register,
  type ZoomLevel,
} from './state/urlState';

const REGISTERS: Register[] = ['kid', 'curious', 'deep'];
const LEVELS: ZoomLevel[] = [0, 1, 2, 3];

export function App() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const worldRef = useRef<TransformerWorld | null>(null);
  const clientRef = useRef<TransformerEngineClient | null>(null);

  const [state, setState] = useState<AtlasState>(() =>
    stateFromUrl(window.location.search)
  );
  const [draftPrompt, setDraftPrompt] = useState<string>(state.prompt);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [captionOpen, setCaptionOpen] = useState<boolean>(true);
  const [muted, setMuted] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Keep shareable URL in sync
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

  // Mount Three.js sunlit studio once
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const world = new TransformerWorld(el, {
      onSelectLevel: (level) => {
        setState((prev) => ({ ...prev, level }));
      },
      onSelectChamber: (layer, zoomIn) => {
        setState((prev) => ({
          ...prev,
          layer,
          level: zoomIn ? 2 : prev.level,
        }));
      },
      onSelectHead: (head) => {
        setState((prev) => ({
          ...prev,
          head,
          level: prev.level === 2 ? 3 : prev.level,
        }));
      },
      onToggleHeadAblation: (layer, head) => {
        setState((prev) => ({
          ...prev,
          ablation: toggleHeadAblation(prev.ablation, layer, head),
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

  // Boot the verified Milestone 2 Web Worker and load real weights
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

  // Recompute real forward pass & real ablation diff whenever prompt or ablation changes
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

  // Push real data snapshot into the 3D world
  useEffect(() => {
    if (!worldRef.current || !initInfo || !baseline) return;
    worldRef.current.update({
      level: state.level,
      layer: state.layer,
      head: state.head,
      register: state.register,
      ablation: state.ablation,
      baseline,
      ablated,
      tokenizer: initInfo.tokenizer,
      hoveredTokenIdx,
    });
  }, [
    initInfo,
    baseline,
    ablated,
    state.level,
    state.layer,
    state.head,
    state.register,
    state.ablation,
    hoveredTokenIdx,
  ]);

  useEffect(() => {
    worldRef.current?.setHoveredToken(hoveredTokenIdx);
  }, [hoveredTokenIdx]);

  // Keyboard shortcuts: Enter/+ zoom in, Esc/- zoom out, 1/2/3 lens
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'Enter' || e.key === '=' || e.key === '+') {
        setState((prev) => ({ ...prev, level: clampLevel(prev.level + 1) }));
      } else if (e.key === 'Escape' || e.key === '-') {
        setState((prev) => ({
          ...prev,
          showNumbers: false,
          level: clampLevel(prev.level - 1),
        }));
      } else if (e.key === '1') {
        setState((prev) => ({ ...prev, register: 'kid' }));
      } else if (e.key === '2') {
        setState((prev) => ({ ...prev, register: 'curious' }));
      } else if (e.key === '3') {
        setState((prev) => ({ ...prev, register: 'deep' }));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const setLevel = useCallback((level: ZoomLevel) => {
    setState((prev) => ({ ...prev, level, showNumbers: false }));
  }, []);

  const content = LEVEL_CONTENT[state.level];
  const activeTrace = ablated ?? baseline;
  const ablationOn = hasActiveAblation(state.ablation);

  const breadcrumbs = useMemo(() => {
    const items: { label: string; level: ZoomLevel }[] = [
      { label: 'Machine', level: 0 },
    ];
    if (state.level >= 1) items.push({ label: 'Factory', level: 1 });
    if (state.level >= 2)
      items.push({ label: `Chamber ${state.layer}`, level: 2 });
    if (state.level >= 3)
      items.push({ label: `Head ${state.head}`, level: 3 });
    return items;
  }, [state.level, state.layer, state.head]);

  const toggleSound = () => {
    const next = !muted;
    setMuted(next);
    studioSound.setMuted(next);
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="studio-viewport">
      <div
        ref={mountRef}
        className="canvas-mount"
        aria-label="Interactive 3D miniature transformer studio"
      />

      {!initInfo && !error && (
        <div className="loading-pill glass-panel">
          Waking up the miniature transformer (862,464 real parameters)...
        </div>
      )}

      {error && <div className="error-toast">{error}</div>}

      {/* Top Floating Chrome */}
      <header className="hud-top">
        <nav className="hud-pill-row glass-panel" aria-label="Breadcrumb">
          <span className="brand-title">LLM Atlas</span>
          {breadcrumbs.map((crumb, idx) => (
            <span key={crumb.level} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              {idx > 0 && <span className="crumb-sep">›</span>}
              <button
                type="button"
                className={`crumb-btn ${crumb.level === state.level ? 'is-current' : ''}`}
                onClick={() => setLevel(crumb.level)}
              >
                {crumb.label}
              </button>
            </span>
          ))}
          <button
            type="button"
            className={`honesty-dot ${
              content.honesty.kind === 'literal' ? 'is-literal' : 'is-abstraction'
            }`}
            title={content.honesty.detail}
            onClick={() => {
              if (content.honesty.literalLevel !== undefined) {
                setLevel(content.honesty.literalLevel);
              }
            }}
          >
            {content.honesty.label}
          </button>
        </nav>

        <div className="hud-pill-row glass-panel">
          <div className="lens-group" role="group" aria-label="Explanation depth">
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
          <button type="button" className="pill-btn" onClick={toggleSound}>
            {muted ? 'Sound off' : 'Sound on'}
          </button>
          <button type="button" className="pill-btn" onClick={() => void copyLink()}>
            {copied ? 'Copied' : 'Share'}
          </button>
        </div>
      </header>

      {/* Quiet Floating Explanation Card */}
      <aside className="caption-card glass-panel" aria-live="polite">
        <div className="caption-top">
          <h1 className="caption-title">{content.title}</h1>
          <button
            type="button"
            className="pill-btn"
            onClick={() => setCaptionOpen((v) => !v)}
          >
            {captionOpen ? 'Hide' : 'Read'}
          </button>
        </div>

        {captionOpen && (
          <>
            <p className="caption-body">{content.copy[state.register]}</p>
            {state.register === 'kid' && content.simplificationNote && (
              <p className="caption-note">{content.simplificationNote}</p>
            )}
            {content.createsQuestion && content.nextCta && state.level < 3 && (
              <div className="caption-next">
                <span className="caption-question">{content.createsQuestion}</span>
                <button
                  type="button"
                  className="cta-btn"
                  onClick={() => setLevel(clampLevel(state.level + 1))}
                >
                  {content.nextCta}
                </button>
              </div>
            )}
          </>
        )}
      </aside>

      {/* Contextual Tactile Controls for Chambers & Head Orbs */}
      <div className="play-dock">
        {state.level >= 2 && (
          <div className="play-strip glass-panel">
            <span className="strip-label">Chamber</span>
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

        {state.level >= 2 && (
          <div className="play-strip glass-panel">
            <span className="strip-label">Orbs</span>
            {[0, 1, 2, 3].map((h) => {
              const off = isHeadDisabled(state.ablation, state.layer, h);
              return (
                <button
                  key={h}
                  type="button"
                  className={`pill-btn mono ${
                    off ? 'is-warm' : state.head === h ? 'is-active' : ''
                  }`}
                  onClick={() => {
                    studioSound.playTileTap(h);
                    setState((prev) => ({ ...prev, head: h }));
                  }}
                >
                  H{h} {off ? '(off)' : ''}
                </button>
              );
            })}
            <button
              type="button"
              className={`pill-btn ${
                isHeadDisabled(state.ablation, state.layer, state.head)
                  ? 'is-warm'
                  : ''
              }`}
              onClick={() => {
                const currentlyOff = isHeadDisabled(
                  state.ablation,
                  state.layer,
                  state.head
                );
                studioSound.playPop(!currentlyOff);
                setState((prev) => ({
                  ...prev,
                  ablation: toggleHeadAblation(
                    prev.ablation,
                    prev.layer,
                    prev.head
                  ),
                }));
              }}
            >
              {isHeadDisabled(state.ablation, state.layer, state.head)
                ? `Plug H${state.head} back in`
                : `Pop out H${state.head}`}
            </button>
          </div>
        )}

        {state.level === 3 && (
          <div className="play-strip glass-panel">
            <button
              type="button"
              className={`pill-btn ${state.showNumbers ? 'is-active' : ''}`}
              onClick={() =>
                setState((prev) => ({ ...prev, showNumbers: !prev.showNumbers }))
              }
            >
              {state.showNumbers ? 'Fold to ribbons' : 'Unfold numbers'}
            </button>
            {ablationOn && (
              <button
                type="button"
                className="pill-btn is-warm"
                onClick={() => {
                  studioSound.playPop(false);
                  setState((prev) => ({ ...prev, ablation: {} }));
                }}
              >
                Restore all orbs
              </button>
            )}
          </div>
        )}
      </div>

      {/* Exact Attention Matrix Overlay (only when user clicks "Unfold numbers") */}
      {state.level === 3 && state.showNumbers && activeTrace && (
        <NumbersOverlay
          tokens={activeTrace.tokens}
          trace={activeTrace.trace}
          layer={state.layer}
          head={state.head}
          hoveredTokenIdx={hoveredTokenIdx}
          onHoverToken={setHoveredTokenIdx}
          onClose={() => setState((prev) => ({ ...prev, showNumbers: false }))}
        />
      )}

      {/* Bottom Floating Sentence Pill */}
      <div className="hud-bottom">
        <div className="preset-row">
          {PRESET_PROMPTS.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`pill-btn ${state.prompt === p.text ? 'is-active' : ''}`}
              onClick={() => {
                studioSound.playTileTap(1);
                setDraftPrompt(p.text);
                setState((prev) => ({ ...prev, prompt: p.text }));
                worldRef.current?.triggerPulse();
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        <form
          className="sentence-bar glass-panel"
          onSubmit={(e) => {
            e.preventDefault();
            if (!draftPrompt.trim()) return;
            studioSound.playTileTap(2);
            setState((prev) => ({ ...prev, prompt: draftPrompt }));
            worldRef.current?.triggerPulse();
          }}
        >
          <input
            className="sentence-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type a short story sentence..."
            aria-label="Story sentence fed to the miniature transformer"
          />
          <div className="depth-dots" role="group" aria-label="Zoom depth">
            {LEVELS.map((lv) => (
              <button
                key={lv}
                type="button"
                className={`depth-dot ${state.level === lv ? 'is-current' : ''}`}
                title={LEVEL_CONTENT[lv].crumb}
                aria-label={`Zoom to ${LEVEL_CONTENT[lv].crumb}`}
                onClick={() => setLevel(lv)}
              />
            ))}
          </div>
          <button
            type="submit"
            className="cta-btn"
            disabled={busy || !initInfo}
          >
            {busy ? 'Thinking...' : 'Play'}
          </button>
        </form>
      </div>
    </div>
  );
}
