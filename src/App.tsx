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
  hasResidualAblation,
  isFinalLnDisabled,
  isHeadDisabled,
  stateFromUrl,
  stateToSearch,
  toggleAllResiduals,
  toggleFinalLn,
  toggleHeadDisabled,
  type BookState,
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
  const [wteWeights, setWteWeights] = useState<Float32Array | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);
  const [bpeStepIdx, setBpeStepIdx] = useState<number>(0);
  const [ropeSlotA, setRopeSlotA] = useState<number>(2);
  const [ropeSlotB, setRopeSlotB] = useState<number>(5);
  const [copied, setCopied] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const chapter =
    TEXTBOOK_CHAPTERS.find((c) => c.id === state.chapterId) ??
    TEXTBOOK_CHAPTERS[0];
  const maxBeatIdx = Math.max(0, chapter.beats.length - 1);
  const clampedBeatIdx = clampInt(state.beatIndex, 0, maxBeatIdx);
  const activeBeat = chapter.beats[clampedBeatIdx] ?? null;
  const activeSceneBeat: SceneBeatId = activeBeat?.sceneBeat ?? 'ch1-1-extremes';

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

  // Boot verified TypeScript engine and load real weights + wte for PCA
  useEffect(() => {
    const client = new TransformerEngineClient();
    clientRef.current = client;
    const modelBaseUrl = new URL(
      `${import.meta.env.BASE_URL}model/`,
      window.location.origin
    ).href;

    client
      .init(modelBaseUrl)
      .then((info) => {
        setInitInfo(info);
        return client.getTensor('wte.weight');
      })
      .then((wte) => setWteWeights(wte))
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

  // Exact 3D PCA cloud computed once from our model's real wte.weight [512, 128]
  const pcaCloud = useMemo(() => {
    if (!wteWeights || !initInfo) return null;
    return computeWtePca3D(
      wteWeights,
      initInfo.tokenizer.vocab,
      initInfo.manifest.config.d_model
    );
  }, [wteWeights, initInfo]);

  // Real step-by-step BPE merge history for Chapter 1 Beat 2
  const bpeSteps = useMemo(() => {
    if (!initInfo) return [];
    return traceBpeMerges(state.prompt, initInfo.tokenizer);
  }, [initInfo, state.prompt]);

  const safeBpeStep = clampInt(bpeStepIdx, 0, Math.max(0, bpeSteps.length - 1));
  const activeTrace = ablated ?? baseline;

  const { resolvedQueryIdx, resolvedKeyIdx } = useMemo(() => {
    if (!activeTrace) return { resolvedQueryIdx: 0, resolvedKeyIdx: 0 };
    const T = activeTrace.tokens.length;
    const q = clampInt(state.queryIdx ?? T - 1, 0, T - 1);

    if (state.keyIdx !== null) {
      return {
        resolvedQueryIdx: q,
        resolvedKeyIdx: clampInt(state.keyIdx, 0, q),
      };
    }

    const lt = activeTrace.trace.layers[state.layer];
    let bestK = Math.max(0, q - 1);
    if (!lt.skipped && q > 0) {
      const row = lt.attn.attnWeights[state.head][q];
      let bestW = -1;
      for (let c = 0; c < q; c++) {
        if (row[c] > bestW) {
          bestW = row[c];
          bestK = c;
        }
      }
    }
    return { resolvedQueryIdx: q, resolvedKeyIdx: bestK };
  }, [activeTrace, state.queryIdx, state.keyIdx, state.layer, state.head]);

  // Push live state & beat choreography to 3D stage
  useEffect(() => {
    if (!stageRef.current || !activeTrace || !baseline || !initInfo) return;
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
    activeTrace,
    baseline,
    initInfo,
    activeSceneBeat,
    state.layer,
    state.head,
    resolvedQueryIdx,
    resolvedKeyIdx,
    state.showPreSoftmax,
    state.ablation,
    wteWeights,
    pcaCloud,
    bpeSteps,
    safeBpeStep,
    ropeSlotA,
    ropeSlotB,
  ]);

  // Keyboard shortcuts: Left/Right step beats, 1/2/3 switch editions
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'ArrowRight') {
        setState((prev) => ({
          ...prev,
          beatIndex: clampInt(prev.beatIndex + 1, 0, maxBeatIdx),
        }));
      } else if (e.key === 'ArrowLeft') {
        setState((prev) => ({
          ...prev,
          beatIndex: clampInt(prev.beatIndex - 1, 0, maxBeatIdx),
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
  }, [maxBeatIdx]);

  // Build real, beat-specific numeric proof lines from the live trace
  const liveCardLines = useMemo(() => {
    if (!activeTrace || !baseline || !initInfo) return [];
    const { tokens, trace } = activeTrace;
    const qTokObj = tokens[resolvedQueryIdx];
    const qTok = formatTokenExplicit(qTokObj.text);
    const kTok = formatTokenExplicit(tokens[resolvedKeyIdx].text);

    // Chapter 1 telemetry
    if (activeSceneBeat === 'ch1-1-extremes') {
      const charCount = state.prompt.length;
      return [
        `Your sentence: ${charCount} characters → ${tokens.length} BPE subword tokens`,
        `Compression ratio: ${(charCount / Math.max(1, tokens.length)).toFixed(2)} chars/token (reduces attention pairs from ${charCount * charCount} to ${tokens.length * tokens.length})`,
      ];
    }
    if (activeSceneBeat === 'ch1-2-bpe-merges') {
      const snap = bpeSteps[safeBpeStep];
      if (!snap) return [];
      return [
        `Merge step ${snap.step} of ${Math.max(0, bpeSteps.length - 1)} · Current count: ${snap.pieces.length} pieces`,
        snap.mergedPair
          ? `Applied rule #${snap.mergeRank}: "${formatTokenExplicit(snap.mergedPair[0])}" + "${formatTokenExplicit(snap.mergedPair[1])}" → "${formatTokenExplicit(snap.mergedResult ?? '')}"`
          : 'Drag the Merge Step slider on stage to watch BPE compress your text.',
      ];
    }
    if (activeSceneBeat === 'ch1-3-spaces-blindspot') {
      const withLeadingSpace = tokens.filter((t) => t.text.startsWith(' ')).length;
      return [
        `Selected token: "${qTok}" (ID #${qTokObj.id}, ${qTokObj.text.length} characters hidden inside 1 ID)`,
        `${withLeadingSpace} of ${tokens.length} tokens in your sentence carry a glued leading space ("·")`,
      ];
    }
    if (activeSceneBeat === 'ch1-4-token-ids') {
      return [
        `Token ID array (first 10): [${tokens.slice(0, 10).map((t) => t.id).join(', ')}${tokens.length > 10 ? ', ...' : ''}]`,
        `Selected "${qTok}" = #${qTokObj.id} of 512 vocabulary entries`,
      ];
    }
    if (activeSceneBeat === 'ch1-5-context-window') {
      const rawCount =
        bpeSteps.length > 0 ? bpeSteps[bpeSteps.length - 1].pieces.length : tokens.length;
      return [
        `Used slots: ${tokens.length} / 64 · Remaining desk capacity: ${64 - tokens.length} tokens`,
        rawCount > 64
          ? `Truncated ${rawCount - 64} tokens past slot 64 (invisible to the model)`
          : 'All tokens fit inside our 64-token context window.',
      ];
    }

    // Chapter 2 telemetry
    if (activeSceneBeat === 'ch2-1-wte-lookup') {
      const st = computeVectorStats(trace.tokEmb[resolvedQueryIdx]);
      return [
        `wte.weight[${qTokObj.id}] ("${qTok}"): 128 floats · ||v|| = ${formatNum(st.l2Norm, 3)}`,
        `Range: [${formatNum(st.min, 3)}, ${formatNum(st.max, 3)}] · mean = ${formatNum(st.mean, 4)}`,
      ];
    }
    if (activeSceneBeat === 'ch2-2-pca-cloud' && wteWeights && pcaCloud) {
      const nbs = findCosineNeighbors(
        wteWeights,
        initInfo.tokenizer.vocab,
        initInfo.manifest.config.d_model,
        qTokObj.id,
        5
      );
      const varPct =
        (pcaCloud.explainedVariance[0] +
          pcaCloud.explainedVariance[1] +
          pcaCloud.explainedVariance[2]) *
        100;
      return [
        `Top 128D cosine neighbors of "${qTok}": ${nbs.map((n) => `"${formatToken(n.text)}" (${n.cosine.toFixed(2)})`).join(', ')}`,
        `3D PCA captures ${varPct.toFixed(1)}% of total variance across all 512 real embeddings`,
      ];
    }
    if (activeSceneBeat === 'ch2-3-order-blind') {
      return [
        'Without wpe/RoPE: permuting token order produces the exact same multiset of [128] vectors.',
      ];
    }
    if (activeSceneBeat === 'ch2-4-wpe-add') {
      const eNorm = vectorL2Norm(trace.tokEmb[resolvedQueryIdx]);
      const pNorm = vectorL2Norm(trace.posEmb[resolvedQueryIdx]);
      const h0Vec = trace.tokEmb[resolvedQueryIdx].map(
        (v, i) => v + trace.posEmb[resolvedQueryIdx][i]
      );
      return [
        `Slot ${resolvedQueryIdx} ("${qTok}"): ||wte|| = ${formatNum(eNorm, 3)} + ||wpe[${resolvedQueryIdx}]|| = ${formatNum(pNorm, 3)} → ||h₀|| = ${formatNum(vectorL2Norm(h0Vec), 3)}`,
        `Starting Residual Stream slab shape entering Block 0: [${tokens.length} × 128]`,
      ];
    }
    if (activeSceneBeat === 'ch2-5-rope') {
      const delta = ropeSlotB - ropeSlotA;
      const rad = (delta * Math.PI) / 10;
      return [
        `Slot A = ${ropeSlotA} (${ropeSlotA * 18}°) · Slot B = ${ropeSlotB} (${ropeSlotB * 18}°) → Distance Δ = ${delta}`,
        `Relative angle Δθ = ${delta * 18}° · cos(Δθ) = ${Math.cos(rad).toFixed(3)} (invariant to shifting both slots equally)`,
      ];
    }

    // Chapter 3 telemetry
    if (activeSceneBeat === 'ch3-1-residual-bus' || activeSceneBeat === 'ch3-2-cut-highway') {
      const baseTop = baseline.trace.greedyNextTokenId;
      const curTop = trace.greedyNextTokenId;
      const cut = hasResidualAblation(state.ablation);
      const lines = [
        `Residual Highway: ${cut ? 'CUT (skip connections disabled)' : 'Intact (x + ΔAttn + ΔMLP)'}`,
        `Prediction: "${formatTokenExplicit(initInfo.tokenizer.vocab[baseTop] ?? '')}" (${(baseline.trace.finalProbs[baseTop] * 100).toFixed(1)}%)${
          ablated
            ? ` → "${formatTokenExplicit(initInfo.tokenizer.vocab[curTop] ?? '')}" (${(trace.finalProbs[curTop] * 100).toFixed(1)}%)`
            : ''
        }`,
      ];
      if (ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null) {
        const dL = ablated.trace.inputLoss - baseline.trace.inputLoss;
        lines.push(
          `Cross-entropy loss: ${baseline.trace.inputLoss.toFixed(3)} → ${ablated.trace.inputLoss.toFixed(3)} (${dL >= 0 ? '+' : ''}${dL.toFixed(3)})`
        );
      }
      return lines;
    }
    if (activeSceneBeat === 'ch3-3-layernorm') {
      const lt = trace.layers[state.layer];
      if (lt.skipped) return [`Block ${state.layer} is disabled.`];
      const rawSt = computeVectorStats(lt.residualBeforeAttn[resolvedQueryIdx]);
      const lnSt = computeVectorStats(lt.ln1Out[resolvedQueryIdx]);
      return [
        `Before LN₁ ("${qTok}", Block ${state.layer}): mean = ${formatNum(rawSt.mean, 3)} · std = ${formatNum(rawSt.std, 3)} · range [${formatNum(rawSt.min, 2)}, ${formatNum(rawSt.max, 2)}]`,
        `After LN₁: mean = ${formatNum(lnSt.mean, 3)} · std = ${formatNum(lnSt.std, 3)} · range [${formatNum(lnSt.min, 2)}, ${formatNum(lnSt.max, 2)}]`,
      ];
    }
    if (activeSceneBeat === 'ch3-4-rmsnorm') {
      const lnOff = isFinalLnDisabled(state.ablation);
      const baseTop = baseline.trace.greedyNextTokenId;
      const curTop = trace.greedyNextTokenId;
      const lines = [
        `Final LN_f: ${lnOff ? 'DISABLED' : 'Active'} · Top guess: "${formatTokenExplicit(initInfo.tokenizer.vocab[curTop] ?? '')}" (${(trace.finalProbs[curTop] * 100).toFixed(1)}%)`,
      ];
      if (ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null) {
        const dL = ablated.trace.inputLoss - baseline.trace.inputLoss;
        lines.push(
          `Loss shift vs. baseline ("${formatTokenExplicit(initInfo.tokenizer.vocab[baseTop] ?? '')}"): ${baseline.trace.inputLoss.toFixed(3)} → ${ablated.trace.inputLoss.toFixed(3)} (${dL >= 0 ? '+' : ''}${dL.toFixed(3)})`
        );
      } else {
        const st = computeVectorStats(trace.lnFOut[resolvedQueryIdx]);
        lines.push(`ln_f output ("${qTok}"): mean = ${formatNum(st.mean, 3)} · rms = ${formatNum(st.rms, 3)}`);
      }
      return lines;
    }

    // Chapter 4 telemetry
    const lt = trace.layers[state.layer];
    if (lt.skipped) return [`Block ${state.layer} is disabled by Ablation.`];
    const active: ActiveLayerTrace = lt;
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
      return [
        `Value blend poured into "${qTok}" (Block ${state.layer} Head ${state.head}):`,
        `${top3.map((x) => `${(x.w * 100).toFixed(0)}% V("${formatTokenExplicit(tokens[x.idx].text)}")`).join(' + ')} + ...`,
      ];
    }

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
    if (ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null) {
      const dLoss = ablated.trace.inputLoss - baseline.trace.inputLoss;
      lines.push(
        `Prompt cross-entropy loss: ${baseline.trace.inputLoss.toFixed(3)} → ${ablated.trace.inputLoss.toFixed(3)} (${dLoss >= 0 ? '+' : ''}${dLoss.toFixed(3)})`
      );
    }
    return lines;
  }, [
    activeTrace,
    baseline,
    ablated,
    initInfo,
    activeSceneBeat,
    state.prompt,
    state.layer,
    state.head,
    state.ablation,
    resolvedQueryIdx,
    resolvedKeyIdx,
    bpeSteps,
    safeBpeStep,
    wteWeights,
    pcaCloud,
    ropeSlotA,
    ropeSlotB,
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

  const headOff = isHeadDisabled(state.ablation, state.layer, state.head);
  const resCut = hasResidualAblation(state.ablation);
  const lnFOff = isFinalLnDisabled(state.ablation);

  return (
    <div className="book-app">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-title">LLM Atlas</span>
          <span className="brand-tag">Interactive Transformer Textbook</span>
        </div>

        <form
          className="prompt-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!draftPrompt.trim()) return;
            setBpeStepIdx(0);
            setState((prev) => ({
              ...prev,
              prompt: draftPrompt,
              queryIdx: null,
              keyIdx: null,
            }));
          }}
        >
          <input
            className="prompt-input"
            value={draftPrompt}
            onChange={(e) => setDraftPrompt(e.target.value)}
            placeholder="Type any sentence to test live..."
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
              setBpeStepIdx(0);
              setState((prev) => ({
                ...prev,
                prompt: p.text,
                queryIdx: null,
                keyIdx: null,
              }));
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

        <div className="topbar-right">
          <button
            type="button"
            className={`mini-btn ${state.lectureMode ? 'is-active' : ''}`}
            onClick={() =>
              setState((prev) => ({ ...prev, lectureMode: !prev.lectureMode }))
            }
          >
            {state.lectureMode ? 'Show Book' : 'Lecture Mode'}
          </button>
          <button type="button" className="mini-btn" onClick={() => void copyLink()}>
            {copied ? 'Copied' : 'Share'}
          </button>
        </div>
      </header>

      <div className={`workspace ${state.lectureMode ? 'is-lecture' : ''}`}>
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

        <section className="stage-pane" aria-label="Interactive 3D Stage">
          <div ref={mountRef} className="canvas-mount" />

          {!initInfo && !error && (
            <div className="loading-pill">
              Loading 862,464 real trained parameters...
            </div>
          )}

          {error && <div className="error-toast">{error}</div>}

          <div className="stage-hud-top">
            <div className="hud-group">
              {activeSceneBeat === 'ch1-2-bpe-merges' && bpeSteps.length > 1 && (
                <>
                  <label className="hud-slider mono">
                    <span>Merge Step {safeBpeStep}/{bpeSteps.length - 1}</span>
                    <input
                      type="range"
                      min={0}
                      max={bpeSteps.length - 1}
                      value={safeBpeStep}
                      onChange={(e) => setBpeStepIdx(Number(e.target.value))}
                    />
                  </label>
                  <button
                    type="button"
                    className="mini-btn"
                    onClick={() =>
                      setBpeStepIdx((s) =>
                        s >= bpeSteps.length - 1 ? 0 : s + 1
                      )
                    }
                  >
                    Step +1
                  </button>
                </>
              )}

              {activeSceneBeat === 'ch2-5-rope' && (
                <>
                  <label className="hud-slider mono">
                    <span>Slot A: {ropeSlotA}</span>
                    <input
                      type="range"
                      min={0}
                      max={24}
                      value={ropeSlotA}
                      onChange={(e) => setRopeSlotA(Number(e.target.value))}
                    />
                  </label>
                  <label className="hud-slider mono">
                    <span>Slot B: {ropeSlotB}</span>
                    <input
                      type="range"
                      min={0}
                      max={24}
                      value={ropeSlotB}
                      onChange={(e) => setRopeSlotB(Number(e.target.value))}
                    />
                  </label>
                </>
              )}

              {activeSceneBeat === 'ch3-2-cut-highway' && (
                <button
                  type="button"
                  className={`mini-btn ${resCut ? 'is-warm' : 'is-active'}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleAllResiduals(prev.ablation),
                    }))
                  }
                >
                  {resCut ? 'Reconnect Residual Highway' : 'Cut Residual Highway'}
                </button>
              )}

              {activeSceneBeat === 'ch3-4-rmsnorm' && (
                <button
                  type="button"
                  className={`mini-btn ${lnFOff ? 'is-warm' : 'is-active'}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleFinalLn(prev.ablation),
                    }))
                  }
                >
                  {lnFOff ? 'Restore Final LN_f' : 'Disable Final LN_f'}
                </button>
              )}

              {(activeSceneBeat === 'ch3-3-layernorm' || state.chapterId === 4) && (
                <>
                  <span className="hud-label">Block</span>
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

              {state.chapterId === 4 && (
                <>
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
                </>
              )}

              {activeSceneBeat === 'attn-4-mask-softmax' && (
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
                  {state.showPreSoftmax ? 'Showing: Raw Scores' : 'Showing: Softmax %'}
                </button>
              )}

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
              <div className="token-bar" aria-label="Select token">
                <span className="hud-label">
                  {activeSceneBeat === 'attn-2-dot-product'
                    ? 'Compare Key:'
                    : 'Inspect Token:'}
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
