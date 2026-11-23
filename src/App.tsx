import React, { useEffect, useMemo, useRef, useState } from 'react';
import { BookReader } from './components/BookReader';
import {
  getChapterMeta,
  PRESET_PROMPTS,
  TEXTBOOK_CHAPTERS,
  type BookBeat,
} from './content/bookData';
import { createEngineClient, type EngineClient } from './engine/client';
import type {
  EngineInitResult,
  EngineTraceResult,
  TokenizerData,
} from './engine/types';
import { StageSnapshot, TextbookStage } from './scene/TextbookStage';
import {
  BookState,
  DEFAULT_PROMPT,
  hasActiveAblation,
  hasResidualAblation,
  isFinalLnDisabled,
  isHeadDisabled,
  stateFromUrl,
  stateToSearch,
  toggleAllResiduals,
  toggleFinalLn,
  toggleHeadDisabled,
} from './state/bookState';
import {
  computeDotProductBreakdown,
  formatNum,
  formatToken,
  formatTokenExplicit,
  vectorL2Norm,
} from './utils/math';
import {
  computeVectorStats,
  computeWtePca3D,
  findCosineNeighbors,
  traceBpeMerges,
} from './utils/pass2Math';

export default function App() {
  const [state, setState] = useState<BookState>(() =>
    stateFromUrl(window.location.search)
  );
  const [client, setClient] = useState<EngineClient | null>(null);
  const [initInfo, setInitInfo] = useState<EngineInitResult | null>(null);
  const [baselineTrace, setBaselineTrace] = useState<EngineTraceResult | null>(
    null
  );
  const [activeTrace, setActiveTrace] = useState<EngineTraceResult | null>(null);
  const [wteWeights, setWteWeights] = useState<Float32Array | null>(null);
  const [bpeStepIdx, setBpeStepIdx] = useState<number>(999);
  const [ropeSlotA, setRopeSlotA] = useState<number>(3);
  const [ropeSlotB, setRopeSlotB] = useState<number>(11);
  const [showContents, setShowContents] = useState(false);
  const [inputVal, setInputVal] = useState(state.prompt);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const stageMountRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<TextbookStage | null>(null);

  // Sync state to URL bar without reloading
  useEffect(() => {
    const search = stateToSearch(state);
    const nextUrl = `${window.location.pathname}${search}`;
    window.history.replaceState(null, '', nextUrl);
  }, [state]);

  // Init Engine Client Worker
  useEffect(() => {
    let unmounted = false;
    const c = createEngineClient();
    setClient(c);

    c.init()
      .then(async (info) => {
        if (unmounted) return;
        setInitInfo(info);
        try {
          const wte = await c.getTensor('wte.weight');
          if (!unmounted && wte) {
            setWteWeights(wte);
          }
        } catch (e) {
          console.warn('Could not load wte.weight tensor:', e);
        }
      })
      .catch((err) => {
        if (!unmounted) {
          setErrorMsg(err instanceof Error ? err.message : String(err));
        }
      });

    return () => {
      unmounted = true;
      c.destroy();
    };
  }, []);

  // Mount 3D Three.js stage
  useEffect(() => {
    if (!stageMountRef.current) return;
    const stage = new TextbookStage(stageMountRef.current);
    stageRef.current = stage;

    const onResize = () => stage.resize();
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      stage.destroy();
      stageRef.current = null;
    };
  }, []);

  // Run forward pass whenever prompt or ablation changes
  useEffect(() => {
    if (!client || !initInfo) return;
    let cancelled = false;

    // Baseline run (unablated)
    client
      .trace(state.prompt, {})
      .then((bRes) => {
        if (cancelled) return;
        setBaselineTrace(bRes);

        // Active run (with current user ablation)
        if (hasActiveAblation(state.ablation)) {
          return client.trace(state.prompt, state.ablation);
        }
        return bRes;
      })
      .then((aRes) => {
        if (!cancelled && aRes) {
          setActiveTrace(aRes);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setErrorMsg(err instanceof Error ? err.message : String(err));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [client, initInfo, state.prompt, state.ablation]);

  // Precompute 3D PCA cloud once wte tensor and tokenizer vocab are ready
  const pcaCloud = useMemo(() => {
    if (!wteWeights || !initInfo?.tokenizer) return null;
    return computeWtePca3D(wteWeights, initInfo.tokenizer.vocab, 128);
  }, [wteWeights, initInfo]);

  // Compute live step-by-step BPE merges for current prompt
  const bpeSteps = useMemo(() => {
    if (!initInfo?.tokenizer) return [];
    return traceBpeMerges(state.prompt, initInfo.tokenizer);
  }, [state.prompt, initInfo]);

  // Resolved chapter and beat
  const chapter = getChapterMeta(state.chapterId);
  const clampedBeatIdx = Math.max(
    0,
    Math.min(chapter.beats.length - 1, state.beatIndex)
  );
  const activeBeat: BookBeat | null = chapter.beats[clampedBeatIdx] ?? null;
  const activeSceneBeat = activeBeat?.sceneBeat ?? 'ch1-1-extremes';

  // Fallback defaults for Query and Key token indices
  const resolvedQueryIdx = useMemo(() => {
    const len = activeTrace?.tokens.length ?? 1;
    if (state.queryIdx !== null && state.queryIdx < len) {
      return state.queryIdx;
    }
    return Math.max(0, len - 1);
  }, [state.queryIdx, activeTrace]);

  const resolvedKeyIdx = useMemo(() => {
    const len = activeTrace?.tokens.length ?? 1;
    if (state.keyIdx !== null && state.keyIdx < len) {
      return state.keyIdx;
    }
    // Default to a token around ~6 steps before query if possible
    return Math.max(0, resolvedQueryIdx - 6);
  }, [state.keyIdx, resolvedQueryIdx, activeTrace]);

  // Push latest snapshot to 3D Stage
  useEffect(() => {
    if (!stageRef.current || !activeTrace || !initInfo) return;
    const baseline = baselineTrace ?? activeTrace;
    const safeBpeStep = Math.min(bpeStepIdx, Math.max(0, bpeSteps.length - 1));

    stageRef.current.update({
      beat: activeSceneBeat,
      layer: state.layer,
      head: state.head,
      queryIdx: resolvedQueryIdx,
      keyIdx: resolvedKeyIdx,
      showPreSoftmax: state.showPreSoftmax,
      ablation: state.ablation,
      traceRes: activeTrace,
      baselineRes: baseline,
      tokenizer: initInfo.tokenizer,
      wteWeights,
      pcaCloud,
      bpeSteps,
      bpeStepIdx: safeBpeStep,
      ropeSlotA,
      ropeSlotB,
    });
  }, [
    activeSceneBeat,
    state.layer,
    state.head,
    resolvedQueryIdx,
    resolvedKeyIdx,
    state.showPreSoftmax,
    state.ablation,
    activeTrace,
    baselineTrace,
    initInfo,
    wteWeights,
    pcaCloud,
    bpeSteps,
    bpeStepIdx,
    ropeSlotA,
    ropeSlotB,
  ]);

  // Generate real-time telemetry numbers for the BookReader live card
  const liveCardLines = useMemo(() => {
    if (!activeTrace) return [];
    const tokens = activeTrace.tokens;
    const tokenIds = activeTrace.tokenIds;
    const qIdx = resolvedQueryIdx;
    const kIdx = resolvedKeyIdx;
    const qToken = tokens[qIdx] ?? '';
    const kToken = tokens[kIdx] ?? '';

    // Chapter 1 telemetry: Tokenization, BPE, Vocab IDs, Context Window
    if (state.chapterId === 1) {
      const charCount = state.prompt.length;
      const tokCount = tokens.length;
      const compression = (charCount / Math.max(1, tokCount)).toFixed(2);
      const totalSteps = bpeSteps.length;
      const curStep = bpeSteps[Math.min(bpeStepIdx, totalSteps - 1)];

      switch (activeSceneBeat) {
        case 'ch1-1-extremes':
          return [
            `Input text: "${state.prompt.slice(0, 48)}..."`,
            `Character count: ${charCount} chars | Subword tokens: ${tokCount} tokens`,
            `Compression ratio: ${compression} chars/token (Saves attention operations)`,
            `Vocabulary bound: |V| = 512 entries (Prevents parameter explosion)`,
          ];
        case 'ch1-2-bpe-merges':
          return [
            `BPE merge replay: step ${curStep?.step ?? 0} of ${Math.max(0, totalSteps - 1)}`,
            curStep?.mergedResult
              ? `Merged: "${curStep.mergedPair?.[0]}" + "${curStep.mergedPair?.[1]}" → "${curStep.mergedResult}" (Rank #${curStep.mergeRank})`
              : `Status: Base character partition (${curStep?.pieces.length ?? 0} fragments)`,
            `Active fragments: [${(curStep?.pieces ?? []).slice(0, 10).map((p) => `"${p}"`).join(', ')}${(curStep?.pieces.length ?? 0) > 10 ? ', ...' : ''}]`,
          ];
        case 'ch1-3-spaces-blindspot': {
          const withSpaceId = initInfo?.tokenizer.vocab.indexOf(' ' + qToken.trim());
          const noSpaceId = initInfo?.tokenizer.vocab.indexOf(qToken.trim());
          return [
            `Active Token: "${formatTokenExplicit(qToken)}" (ID #${tokenIds[qIdx]})`,
            `With space variant: " ${qToken.trim()}" → ${withSpaceId !== undefined && withSpaceId >= 0 ? `ID #${withSpaceId}` : 'Not in vocab'}`,
            `No space variant: "${qToken.trim()}" → ${noSpaceId !== undefined && noSpaceId >= 0 ? `ID #${noSpaceId}` : 'Not in vocab'}`,
            `Key rule: Space characters are bound to subword prefixes, not separate tokens.`,
          ];
        }
        case 'ch1-4-token-ids': {
          const minId = Math.min(...tokenIds);
          const maxId = Math.max(...tokenIds);
          return [
            `Token IDs tensor x ∈ {0..511}^${tokens.length}:`,
            `[${tokenIds.slice(0, 12).join(', ')}${tokenIds.length > 12 ? ', ...' : ''}]`,
            `Min ID: #${minId} | Max ID: #${maxId} | Model d_model: 128`,
            `Parameter memory lookup: ${tokens.length} rows × 128 cols = ${tokens.length * 128 * 4} bytes`,
          ];
        }
        case 'ch1-5-context-window':
          return [
            `Sequence length T: ${tokens.length} of 64 slots occupied`,
            `Remaining generation budget: ${64 - tokens.length} slots`,
            `Attention matrix size: [${tokens.length} × ${tokens.length}] = ${tokens.length * tokens.length} cells`,
            `Max capacity matrix at T=64: [64 × 64] = 4,096 cells`,
          ];
        default:
          return [];
      }
    }

    // Chapter 2 telemetry: Embeddings, PCA, Order-Blindness, wpe & RoPE
    if (state.chapterId === 2) {
      const activeId = tokenIds[qIdx];
      switch (activeSceneBeat) {
        case 'ch2-1-wte-lookup': {
          if (!wteWeights) return ['Loading W_e weight tensor [512 × 128]...'];
          const base = activeId * 128;
          const slice = Array.from(wteWeights.slice(base, base + 128));
          const stats = computeVectorStats(slice);
          return [
            `Token: "${formatToken(qToken)}" | Vocabulary row: #${activeId}`,
            `Row vector: wte[${activeId}, :] ∈ ℝ¹²⁸`,
            `L2 Norm ||wte||: ${formatNum(stats.l2Norm, 3)} | Mean: ${formatNum(stats.mean, 4)}`,
            `Coordinate range: [${formatNum(stats.min, 3)}, ${formatNum(stats.max, 3)}] (Std: ${formatNum(stats.std, 3)})`,
          ];
        }
        case 'ch2-2-pca-cloud': {
          const topVar = pcaCloud?.explainedVariance ?? [0, 0, 0];
          const sumVar = ((topVar[0] + topVar[1] + topVar[2]) * 100).toFixed(1);
          const neighbors =
            wteWeights && initInfo?.tokenizer
              ? findCosineNeighbors(
                  wteWeights,
                  initInfo.tokenizer.vocab,
                  128,
                  activeId,
                  3
                )
              : [];
          return [
            `Target Token: "${formatToken(qToken)}" (#${activeId})`,
            `PCA Explained Variance: PC1 ${(topVar[0] * 100).toFixed(1)}%, PC2 ${(topVar[1] * 100).toFixed(1)}%, PC3 ${(topVar[2] * 100).toFixed(1)}% (Total: ${sumVar}%)`,
            `Top Cosine Neighbors in ℝ¹²⁸:`,
            ...neighbors.map(
              (n) => `  • "${formatToken(n.text)}" (#${n.id}) → cos = ${formatNum(n.cosine, 3)}`
            ),
          ];
        }
        case 'ch2-3-order-blind':
          return [
            `Permutation test: Sequence of ${tokens.length} tokens`,
            `Self-Attention without position: Attention(P·X) = P·Attention(X)`,
            `Status: Raw word embeddings alone have ZERO directional or syntactic order.`,
            `Required fix: Positional encoding must be injected into the residual stream.`,
          ];
        case 'ch2-4-wpe-add':
          return [
            `Token: "${formatToken(qToken)}" at slot index ${qIdx}`,
            `Embedding formula: h_${qIdx}^(0) = wte[#${activeId}] + wpe[#${qIdx}]`,
            `wte tensor: [512, 128] | wpe tensor: [64, 128]`,
            `Output representation h_${qIdx}^(0) ∈ ℝ¹²⁸ ready for Block 0`,
          ];
        case 'ch2-5-rope': {
          const delta = ropeSlotB - ropeSlotA;
          const deltaDeg = Math.round((Math.abs(delta * 0.35) * 180) / Math.PI);
          return [
            `Rotary Position Embedding (Su et al., 2021):`,
            `Slot A: ${ropeSlotA} | Slot B: ${ropeSlotB} | Relative distance: ${delta} slots`,
            `Rotation offset: Δθ = ${deltaDeg}° across first feature pair`,
            `Advantage: Relative distance invariant <R_m q, R_n k> = g(q, k, m-n)`,
          ];
        }
        default:
          return [];
      }
    }

    // Chapter 3 telemetry: Residual Highway, Break-It Lab, LayerNorm & RMSNorm
    if (state.chapterId === 3) {
      const isCut = hasResidualAblation(state.ablation);
      const isLnOff = isFinalLnDisabled(state.ablation);
      switch (activeSceneBeat) {
        case 'ch3-1-residual-bus': {
          const topToken = activeTrace.tokens[tokens.length - 1];
          const topNext = activeTrace.predictions?.[0]?.token ?? '';
          const topProb = ((activeTrace.predictions?.[0]?.prob ?? 0) * 100).toFixed(1);
          return [
            `Residual state recurrence: h_(l+1) = h_l + SubBlock(LN(h_l))`,
            `Model highway dimension: d_model = 128 across 4 Transformer Blocks`,
            `Final prediction after 4 layers: "${topNext}" (${topProb}%) following "${topToken}"`,
            `Identity shortcut guarantees gradient flow without vanishing or exploding.`,
          ];
        }
        case 'ch3-2-cut-highway': {
          const top1 = activeTrace.predictions?.[0];
          const base1 = baselineTrace?.predictions?.[0];
          return [
            `Residual Status: ${isCut ? 'SEVERED (All shortcuts disabled)' : 'INTACT (Normal)'}`,
            isCut
              ? `CRITICAL FAILURE: Predictions collapsed into gibberish!`
              : `Healthy: Identity shortcut preserves earlier token memory.`,
            `Active top prediction: "${top1?.token ?? ''}" (${((top1?.prob ?? 0) * 100).toFixed(1)}%)`,
            `Baseline unablated: "${base1?.token ?? ''}" (${((base1?.prob ?? 0) * 100).toFixed(1)}%)`,
          ];
        }
        case 'ch3-3-layernorm':
          return [
            `Pre-LayerNorm formula: y = (x − μ) / √(σ² + 1e-5) · γ + β`,
            `Parameters: γ ∈ ℝ¹²⁸ (scale) and β ∈ ℝ¹²⁸ (shift)`,
            `Normalized input guarantee: Mean = 0.000, Variance = 1.000`,
            `Benefit: Stabilizes inputs to Attention and MLP across all 4 layers.`,
          ];
        case 'ch3-4-rmsnorm':
          return [
            `RMSNorm formula: y = x / RMS(x) · γ  where RMS(x) = √(1/d ∑ x_i² + ε)`,
            `Final LN_f status: ${isLnOff ? 'DISABLED' : 'ENABLED'}`,
            `Comparison: RMSNorm skips μ centering, saving GPU memory bandwidth.`,
            `Used in frontier models: LLaMA, Mistral, Gemma 2.`,
          ];
        default:
          return [];
      }
    }

    // Chapter 4 telemetry: Self-Attention flagship
    const qVec = activeTrace.q?.[state.layer]?.[state.head]?.[qIdx] ?? [];
    const kVec = activeTrace.k?.[state.layer]?.[state.head]?.[kIdx] ?? [];
    const breakdown = computeDotProductBreakdown(qVec, kVec, 32);

    const matrix = activeTrace.attention?.[state.layer]?.[state.head] ?? [];
    const currentWeight = matrix[qIdx]?.[kIdx] ?? 0;

    const topPred = activeTrace.predictions?.[0];
    const basePred = baselineTrace?.predictions?.[0];

    const lines: string[] = [
      `Prompt: "${state.prompt.slice(0, 42)}..." (${tokens.length} tokens)`,
      `Layer: ${state.layer} of 3 | Head: ${state.head} of 3 | d_k = 32`,
      `Active Query [${qIdx}]: "${formatToken(qToken)}" | Active Key [${kIdx}]: "${formatToken(kToken)}"`,
      `Raw Dot Product: ${formatNum(breakdown.rawDot, 3)}`,
      `Scaled by 1/√32 (÷5.657): ${formatNum(breakdown.scaledScore, 3)}`,
      `Softmax Attention Weight: ${(currentWeight * 100).toFixed(1)}%`,
    ];

    if (hasActiveAblation(state.ablation)) {
      lines.push(
        `Ablation active! Top prediction: "${topPred?.token ?? ''}" (${((topPred?.prob ?? 0) * 100).toFixed(1)}% vs. base ${((basePred?.prob ?? 0) * 100).toFixed(1)}%)`
      );
    } else {
      lines.push(
        `Top predicted next token: "${topPred?.token ?? ''}" (${((topPred?.prob ?? 0) * 100).toFixed(1)}% confidence)`
      );
    }

    return lines;
  }, [
    state.chapterId,
    activeSceneBeat,
    state.prompt,
    resolvedQueryIdx,
    resolvedKeyIdx,
    state.layer,
    state.head,
    state.ablation,
    activeTrace,
    baselineTrace,
    wteWeights,
    pcaCloud,
    bpeSteps,
    bpeStepIdx,
    ropeSlotA,
    ropeSlotB,
    initInfo,
  ]);

  return (
    <div className="atlas-app">
      {/* Top Application Bar */}
      <header className="app-header">
        <div className="app-brand">
          <span className="brand-title">LLM Atlas</span>
          <span className="brand-tag">Interactive Textbook</span>
        </div>

        <form
          className="prompt-bar"
          onSubmit={(e) => {
            e.preventDefault();
            if (inputVal.trim()) {
              setState((prev) => ({
                ...prev,
                prompt: inputVal.trim(),
                queryIdx: null,
                keyIdx: null,
              }));
            }
          }}
        >
          <select
            className="preset-select"
            aria-label="Example prompt presets"
            onChange={(e) => {
              const val = e.target.value;
              setInputVal(val);
              setState((prev) => ({
                ...prev,
                prompt: val,
                queryIdx: null,
                keyIdx: null,
              }));
            }}
          >
            <option value="">Presets...</option>
            {PRESET_PROMPTS.map((p) => (
              <option key={p.id} value={p.text}>
                {p.label}
              </option>
            ))}
          </select>

          <input
            type="text"
            className="prompt-input"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            placeholder="Type any sentence to see real live tensors..."
          />

          <button type="submit" className="primary-btn">
            Run
          </button>
        </form>

        <div className="header-actions">
          <button
            type="button"
            className="secondary-btn"
            onClick={() => {
              const text = window.location.href;
              navigator.clipboard?.writeText(text);
              alert('Copied direct shareable link to this exact state!');
            }}
          >
            Share Link
          </button>
        </div>
      </header>

      {/* Main Split-Screen Workspace */}
      <main className="workspace">
        <BookReader
          chapter={chapter}
          beat={activeBeat}
          beatIndex={clampedBeatIdx}
          edition={state.edition}
          showContents={showContents}
          liveCardLines={liveCardLines}
          onToggleContents={() => setShowContents((v) => !v)}
          onSelectChapter={(chapterId) => {
            setShowContents(false);
            setState((prev) => ({ ...prev, chapterId, beatIndex: 0 }));
          }}
          onSelectBeat={(beatIndex) =>
            setState((prev) => ({ ...prev, beatIndex }))
          }
          onChangeEdition={(edition) =>
            setState((prev) => ({ ...prev, edition }))
          }
        />

        {/* 3D Visual Stage */}
        <section className="stage-pane" aria-label="3D Interactive Simulation">
          {/* Floating HUD Controls */}
          <div className="stage-hud-top">
            {state.chapterId === 1 && activeSceneBeat === 'ch1-2-bpe-merges' && (
              <div className="hud-group">
                <span className="hud-label">Merge Step:</span>
                <label className="hud-slider">
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, bpeSteps.length - 1)}
                    value={Math.min(
                      bpeStepIdx,
                      Math.max(0, bpeSteps.length - 1)
                    )}
                    onChange={(e) => setBpeStepIdx(Number(e.target.value))}
                  />
                  <span className="mono">
                    {Math.min(bpeStepIdx, Math.max(0, bpeSteps.length - 1))}
                  </span>
                </label>
              </div>
            )}

            {state.chapterId === 2 && activeSceneBeat === 'ch2-5-rope' && (
              <div className="hud-group">
                <span className="hud-label">Slot A:</span>
                <label className="hud-slider">
                  <input
                    type="range"
                    min={0}
                    max={32}
                    value={ropeSlotA}
                    onChange={(e) => setRopeSlotA(Number(e.target.value))}
                  />
                  <span className="mono">{ropeSlotA}</span>
                </label>
                <span className="hud-label" style={{ marginLeft: 6 }}>
                  Slot B:
                </span>
                <label className="hud-slider">
                  <input
                    type="range"
                    min={0}
                    max={32}
                    value={ropeSlotB}
                    onChange={(e) => setRopeSlotB(Number(e.target.value))}
                  />
                  <span className="mono">{ropeSlotB}</span>
                </label>
              </div>
            )}

            {state.chapterId === 3 && activeSceneBeat === 'ch3-2-cut-highway' && (
              <div className="hud-group">
                <button
                  type="button"
                  className={`hud-pill ${hasResidualAblation(state.ablation) ? 'is-active' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleAllResiduals(prev.ablation),
                    }))
                  }
                >
                  {hasResidualAblation(state.ablation)
                    ? 'Restore Residual Highway'
                    : 'Cut Residuals (Break-It Lab)'}
                </button>
              </div>
            )}

            {state.chapterId === 3 && activeSceneBeat === 'ch3-4-rmsnorm' && (
              <div className="hud-group">
                <button
                  type="button"
                  className={`hud-pill ${isFinalLnDisabled(state.ablation) ? 'is-active' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleFinalLn(prev.ablation),
                    }))
                  }
                >
                  {isFinalLnDisabled(state.ablation)
                    ? 'Restore Final LayerNorm'
                    : 'Disable Final LN (LN_f)'}
                </button>
              </div>
            )}

            {state.chapterId === 4 && (
              <>
                <div className="hud-group" role="group" aria-label="Layer selector">
                  <span className="hud-label">Layer:</span>
                  {[0, 1, 2, 3].map((l) => (
                    <button
                      key={l}
                      type="button"
                      className={`hud-pill ${state.layer === l ? 'is-active' : ''}`}
                      onClick={() => setState((prev) => ({ ...prev, layer: l }))}
                    >
                      L{l}
                    </button>
                  ))}
                </div>

                <div className="hud-group" role="group" aria-label="Head selector">
                  <span className="hud-label">Head:</span>
                  {[0, 1, 2, 3].map((h) => {
                    const disabled = isHeadDisabled(
                      state.ablation,
                      state.layer,
                      h
                    );
                    return (
                      <button
                        key={h}
                        type="button"
                        className={`hud-pill ${state.head === h ? 'is-active' : ''} ${
                          disabled ? 'is-disabled' : ''
                        }`}
                        onClick={() =>
                          setState((prev) => ({ ...prev, head: h }))
                        }
                      >
                        H{h}
                      </button>
                    );
                  })}
                </div>

                <div className="hud-group">
                  <button
                    type="button"
                    className={`hud-pill ${state.showPreSoftmax ? 'is-active' : ''}`}
                    onClick={() =>
                      setState((prev) => ({
                        ...prev,
                        showPreSoftmax: !prev.showPreSoftmax,
                      }))
                    }
                  >
                    {state.showPreSoftmax ? 'Logits (Pre-Softmax)' : 'Softmax Weights'}
                  </button>

                  <button
                    type="button"
                    className={`hud-pill ${
                      isHeadDisabled(state.ablation, state.layer, state.head)
                        ? 'is-active'
                        : ''
                    }`}
                    onClick={() =>
                      setState((prev) => ({
                        ...prev,
                        ablation: toggleHeadDisabled(
                          prev.ablation,
                          state.layer,
                          state.head
                        ),
                      }))
                    }
                  >
                    {isHeadDisabled(state.ablation, state.layer, state.head)
                      ? 'Un-ablate Head'
                      : 'Ablate Head'}
                  </button>
                </div>
              </>
            )}

            <div className="camera-hint">Click + drag to rotate 3D camera</div>
          </div>

          <div ref={stageMountRef} className="stage-viewport" />
        </section>
      </main>

      {/* Bottom Status Bar */}
      <footer className="app-footer">
        <div className="footer-left">
          <span>Model: 4 Layers · 4 Heads · d_model=128 · 862,464 Parameters</span>
          <span>
            {initInfo ? 'Verified PyTorch Parity (max diff < 1e-4)' : 'Initializing worker...'}
          </span>
        </div>
        <div className="footer-right">
          <a
            href="https://github.com/ageanton/llm-atlas"
            target="_blank"
            rel="noreferrer"
            className="footer-link"
          >
            GitHub Source &amp; Mathematical Trace
          </a>
        </div>
      </footer>

      {errorMsg && (
        <div
          style={{
            position: 'fixed',
            bottom: 36,
            left: 20,
            background: '#dc2626',
            color: '#fff',
            padding: '8px 12px',
            borderRadius: 6,
            fontSize: '0.8rem',
            zIndex: 99,
          }}
        >
          {errorMsg}
        </div>
      )}
    </div>
  );
}
