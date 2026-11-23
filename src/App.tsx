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
  toggleAttnBlock,
  toggleFinalLn,
  toggleHeadDisabled,
  toggleLayerDisabled,
  toggleMlpBlock,
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
import {
  computeManualWeightLandscape,
  computePerTokenLoss,
  computeSampledDistribution,
  countActiveNeurons,
  geluTanh,
  sampleFromDistribution,
  SCALE_STOPS,
  type TrainingReplayData,
} from './utils/pass3Math';

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
  const [replay, setReplay] = useState<TrainingReplayData | null>(null);
  const [baseline, setBaseline] = useState<EngineTraceResult | null>(null);
  const [ablated, setAblated] = useState<EngineTraceResult | null>(null);

  // Interactive chapter lab states
  const [bpeStepIdx, setBpeStepIdx] = useState<number>(0);
  const [ropeSlotA, setRopeSlotA] = useState<number>(2);
  const [ropeSlotB, setRopeSlotB] = useState<number>(5);
  const [geluProbeX, setGeluProbeX] = useState<number>(1.2);
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(12);
  const [topP, setTopP] = useState<number>(0.9);
  const [sampleBurst, setSampleBurst] = useState<string[]>([]);
  const [weightDelta, setWeightDelta] = useState<number>(0);
  const [replayIdx, setReplayIdx] = useState<number>(0);
  const [scaleIdx, setScaleIdx] = useState<number>(0);
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

  const pcaCloud = useMemo(() => {
    if (!wteWeights || !initInfo) return null;
    return computeWtePca3D(
      wteWeights,
      initInfo.tokenizer.vocab,
      initInfo.manifest.config.d_model
    );
  }, [wteWeights, initInfo]);

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
      return { resolvedQueryIdx: q, resolvedKeyIdx: clampInt(state.keyIdx, 0, q) };
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
      replay,
      geluProbeX,
      temperature,
      topK,
      topP,
      weightDelta,
      replayIdx,
      scaleIdx,
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
    replay,
    geluProbeX,
    temperature,
    topK,
    topP,
    weightDelta,
    replayIdx,
    scaleIdx,
  ]);

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

  const handleSampleBurst = () => {
    if (!activeTrace || !initInfo) return;
    const dist = computeSampledDistribution(
      activeTrace.trace.finalLogits,
      temperature,
      activeSceneBeat === 'ch6-4-nucleus' ? topK : 512,
      activeSceneBeat === 'ch6-4-nucleus' ? topP : 1.0
    );
    const rolls: string[] = [];
    for (let i = 0; i < 10; i++) {
      rolls.push(formatToken(initInfo.tokenizer.vocab[sampleFromDistribution(dist)] ?? ''));
    }
    setSampleBurst(rolls);
  };

  const handleStepAutoregressive = () => {
    if (!activeTrace || !initInfo) return;
    const dist = computeSampledDistribution(
      activeTrace.trace.finalLogits,
      temperature,
      topK,
      topP
    );
    const piece = initInfo.tokenizer.vocab[sampleFromDistribution(dist)] ?? '';
    if (!piece || piece === '<|eos|>') return;
    const nextPrompt = `${state.prompt}${piece}`;
    setDraftPrompt(nextPrompt);
    setState((prev) => ({ ...prev, prompt: nextPrompt, queryIdx: null, keyIdx: null }));
  };

  const liveCardLines = useMemo(() => {
    if (!activeTrace || !baseline || !initInfo) return [];
    const { tokens, trace } = activeTrace;
    const qTokObj = tokens[resolvedQueryIdx];
    const qTok = formatTokenExplicit(qTokObj.text);
    const kTok = formatTokenExplicit(tokens[resolvedKeyIdx].text);
    const vocab = initInfo.tokenizer.vocab;

    // Chapters 1-3
    if (activeSceneBeat === 'ch1-1-extremes') {
      const n = state.prompt.length;
      return [
        `Sentence: ${n} chars → ${tokens.length} BPE tokens (${(n / Math.max(1, tokens.length)).toFixed(2)} chars/token)`,
      ];
    }
    if (activeSceneBeat === 'ch1-2-bpe-merges') {
      const s = bpeSteps[safeBpeStep];
      return s
        ? [
            `Merge ${s.step}/${Math.max(0, bpeSteps.length - 1)} · ${s.pieces.length} pieces`,
            s.mergedPair
              ? `Rule #${s.mergeRank}: "${formatTokenExplicit(s.mergedPair[0])}" + "${formatTokenExplicit(s.mergedPair[1])}" → "${formatTokenExplicit(s.mergedResult ?? '')}"`
              : 'Step 0: Single characters.',
          ]
        : [];
    }
    if (activeSceneBeat === 'ch1-3-spaces-blindspot') {
      return [
        `Selected "${qTok}" (ID #${qTokObj.id}, ${qTokObj.text.length} chars inside 1 atomic token)`,
      ];
    }
    if (activeSceneBeat === 'ch1-4-token-ids') {
      return [`IDs: [${tokens.slice(0, 12).map((t) => t.id).join(', ')}${tokens.length > 12 ? ', ...' : ''}]`];
    }
    if (activeSceneBeat === 'ch1-5-context-window') {
      return [`Context used: ${tokens.length} / 64 slots (${64 - tokens.length} open)`];
    }
    if (activeSceneBeat === 'ch2-1-wte-lookup') {
      const st = computeVectorStats(trace.tokEmb[resolvedQueryIdx]);
      return [`wte[#${qTokObj.id}] ("${qTok}"): ||v||=${formatNum(st.l2Norm, 3)} · range [${formatNum(st.min, 2)}, ${formatNum(st.max, 2)}]`];
    }
    if (activeSceneBeat === 'ch2-2-pca-cloud' && wteWeights && pcaCloud) {
      const nbs = findCosineNeighbors(wteWeights, vocab, initInfo.manifest.config.d_model, qTokObj.id, 5);
      return [
        `128D cosine neighbors of "${qTok}": ${nbs.map((n) => `"${formatToken(n.text)}" (${n.cosine.toFixed(2)})`).join(', ')}`,
      ];
    }
    if (activeSceneBeat === 'ch2-3-order-blind') {
      return ['Without positional stamps, permuting input words leaves the multiset of vectors unchanged.'];
    }
    if (activeSceneBeat === 'ch2-4-wpe-add') {
      return [
        `Slot ${resolvedQueryIdx}: ||wte||=${formatNum(vectorL2Norm(trace.tokEmb[resolvedQueryIdx]), 2)} + ||wpe||=${formatNum(vectorL2Norm(trace.posEmb[resolvedQueryIdx]), 2)} → h₀ [${tokens.length}×128]`,
      ];
    }
    if (activeSceneBeat === 'ch2-5-rope') {
      const d = ropeSlotB - ropeSlotA;
      return [`Slot A=${ropeSlotA}, Slot B=${ropeSlotB} → Δ=${d} slots (${d * 18}°, cos=${Math.cos((d * Math.PI) / 10).toFixed(3)})`];
    }
    if (activeSceneBeat === 'ch3-1-residual-bus' || activeSceneBeat === 'ch3-2-cut-highway') {
      const cut = hasResidualAblation(state.ablation);
      const bTop = baseline.trace.greedyNextTokenId;
      const cTop = trace.greedyNextTokenId;
      return [
        `Highway: ${cut ? 'CUT' : 'Intact'} · Guess: "${formatTokenExplicit(vocab[bTop] ?? '')}" (${(baseline.trace.finalProbs[bTop] * 100).toFixed(1)}%)${ablated ? ` → "${formatTokenExplicit(vocab[cTop] ?? '')}" (${(trace.finalProbs[cTop] * 100).toFixed(1)}%)` : ''}`,
        ...(ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null
          ? [`Loss: ${baseline.trace.inputLoss.toFixed(3)} → ${ablated.trace.inputLoss.toFixed(3)} (+${(ablated.trace.inputLoss - baseline.trace.inputLoss).toFixed(3)})`]
          : []),
      ];
    }
    if (activeSceneBeat === 'ch3-3-layernorm') {
      const lt = trace.layers[state.layer];
      if (lt.skipped) return [`Block ${state.layer} disabled.`];
      const rSt = computeVectorStats(lt.residualBeforeAttn[resolvedQueryIdx]);
      const lSt = computeVectorStats(lt.ln1Out[resolvedQueryIdx]);
      return [
        `Before LN₁: mean=${formatNum(rSt.mean, 3)}, std=${formatNum(rSt.std, 3)} → After LN₁: mean=${formatNum(lSt.mean, 3)}, std=${formatNum(lSt.std, 3)}`,
      ];
    }
    if (activeSceneBeat === 'ch3-4-rmsnorm') {
      const off = isFinalLnDisabled(state.ablation);
      return [
        `Final LN_f: ${off ? 'DISABLED' : 'Active'} · Loss: ${trace.inputLoss?.toFixed(3) ?? '-'}`,
      ];
    }

    // Chapter 4
    if (activeSceneBeat.startsWith('attn-')) {
      const lt = trace.layers[state.layer];
      if (lt.skipped) return [`Block ${state.layer} disabled.`];
      const active: ActiveLayerTrace = lt;
      const rowW = active.attn.attnWeights[state.head][resolvedQueryIdx];
      if (activeSceneBeat === 'attn-2-dot-product') {
        const bd = computeDotProductBreakdown(
          active.attn.q[state.head][resolvedQueryIdx],
          active.attn.k[state.head][resolvedKeyIdx]
        );
        return [
          `Q("${qTok}") · K("${kTok}") = ${formatNum(bd.rawDot, 3)} ÷ √32 = ${formatNum(bd.scaledScore, 3)}`,
        ];
      }
      if (activeSceneBeat === 'attn-4-mask-softmax') {
        return [
          `Masked future tokens: ${tokens.length - 1 - resolvedQueryIdx} · Row sum = ${rowW.reduce((a, b) => a + b, 0).toFixed(6)}`,
        ];
      }
      if (activeSceneBeat === 'attn-5-value-mixer') {
        const top3 = rowW
          .slice(0, resolvedQueryIdx + 1)
          .map((w, idx) => ({ idx, w }))
          .sort((a, b) => b.w - a.w)
          .slice(0, 3);
        return [top3.map((x) => `${(x.w * 100).toFixed(0)}% V("${formatTokenExplicit(tokens[x.idx].text)}")`).join(' + ')];
      }
      const topId = trace.greedyNextTokenId;
      return [
        `Block ${state.layer} H${state.head} · Next guess: "${formatTokenExplicit(vocab[topId] ?? '')}" (${(trace.finalProbs[topId] * 100).toFixed(1)}%)`,
      ];
    }

    // Chapter 5
    if (activeSceneBeat.startsWith('ch5-')) {
      const lt = trace.layers[state.layer];
      if (lt.skipped) return [`Block ${state.layer} disabled.`];
      const firing = countActiveNeurons(lt.mlp.hidden[resolvedQueryIdx], 0.05);
      if (activeSceneBeat === 'ch5-3-gelu') {
        return [
          `Probe: GELU(${geluProbeX.toFixed(2)}) = ${geluTanh(geluProbeX).toFixed(4)}`,
          `Live Block ${state.layer} MLP ("${qTok}"): ${firing} / 512 neurons active (>0.05)`,
        ];
      }
      return [
        `Block ${state.layer} MLP ("${qTok}"): 128 → 512 (${firing} active) → 128 · ||Δmlp|| = ${formatNum(vectorL2Norm(lt.mlp.output[resolvedQueryIdx]), 2)}`,
      ];
    }

    // Chapter 6
    if (activeSceneBeat.startsWith('ch6-')) {
      const topId = trace.greedyNextTokenId;
      const pool = computeSampledDistribution(trace.finalLogits, temperature, topK, topP);
      if (activeSceneBeat === 'ch6-1-logit-lens') {
        return [
          trace.logitLens
            .map((e) => `B${e.afterLayer}:"${formatToken(vocab[e.topTokens[0].id] ?? '')}"(${(e.topTokens[0].prob * 100).toFixed(0)}%)`)
            .join(' → '),
        ];
      }
      return [
        `Top logit: "${formatTokenExplicit(vocab[topId] ?? '')}" z=${formatNum(trace.finalLogits[topId], 2)} (P=${(trace.finalProbs[topId] * 100).toFixed(1)}%)`,
        `Sampling pool: ${pool.length} tokens (τ=${temperature.toFixed(2)}, k=${topK}, p=${topP.toFixed(2)}) · T=${tokens.length}/64`,
      ];
    }

    // Chapter 7
    if (activeSceneBeat === 'ch7-2-cross-entropy' && wteWeights) {
      const { entries } = computePerTokenLoss(
        baseline.trace.lnFOut,
        baseline.trace.tokenIds,
        wteWeights,
        initInfo.tokenizer.vocab_size
      );
      const meanL = baseline.trace.inputLoss ?? 0;
      const worst = entries.reduce((a, b) => (b.loss > a.loss ? b : a), entries[0]);
      return [
        `Mean prompt loss = ${meanL.toFixed(3)} · Perplexity = ${Math.exp(meanL).toFixed(2)}`,
        ...(worst
          ? [`Most surprising token: "${formatTokenExplicit(vocab[worst.targetId] ?? '')}" (loss ${worst.loss.toFixed(2)}, P=${(worst.targetProb * 100).toFixed(1)}%)`]
          : []),
      ];
    }
    if (activeSceneBeat === 'ch7-3-manual-backprop' && wteWeights) {
      const land = computeManualWeightLandscape(
        baseline.trace.lnFOut,
        baseline.trace.tokenIds,
        wteWeights,
        initInfo.tokenizer.vocab_size,
        weightDelta
      );
      return land
        ? [
            `Weight wte["${formatTokenExplicit(vocab[land.targetTokenId] ?? '')}", ${land.dim}] = ${land.currentWeight.toFixed(4)} (Δ ${weightDelta >= 0 ? '+' : ''}${weightDelta.toFixed(2)})`,
            `Live Loss = ${land.currentLoss.toFixed(4)} (trained: ${land.baseLoss.toFixed(4)}) · Gradient ∂L/∂w = ${land.gradient.toFixed(4)}`,
          ]
        : ['Enter at least 2 tokens to compute live loss.'];
    }
    if (activeSceneBeat === 'ch7-4-training-replay' && replay) {
      const ckpt = replay.checkpoints[Math.min(replayIdx, replay.checkpoints.length - 1)];
      const sample = ckpt.prompts[0];
      return [
        `Checkpoint Step ${ckpt.step}: train loss ${ckpt.train_loss.toFixed(3)} · val loss ${ckpt.val_loss.toFixed(3)}`,
        ...(sample ? [`Sample at step ${ckpt.step}: "${sample.prompt}" → ${sample.completion}`] : []),
      ];
    }
    if (activeSceneBeat === 'ch7-5-hallucination') {
      const topId = trace.greedyNextTokenId;
      return [
        `Current top prediction: "${formatTokenExplicit(vocab[topId] ?? '')}" (${(trace.finalProbs[topId] * 100).toFixed(1)}%)`,
      ];
    }

    // Chapter 8
    if (activeSceneBeat === 'ch8-1-ablation') {
      const bTop = baseline.trace.greedyNextTokenId;
      const cTop = trace.greedyNextTokenId;
      const dL =
        ablated && baseline.trace.inputLoss !== null && ablated.trace.inputLoss !== null
          ? ablated.trace.inputLoss - baseline.trace.inputLoss
          : 0;
      return [
        `Baseline: "${formatTokenExplicit(vocab[bTop] ?? '')}" (${(baseline.trace.finalProbs[bTop] * 100).toFixed(1)}%) → Now: "${formatTokenExplicit(vocab[cTop] ?? '')}" (${(trace.finalProbs[cTop] * 100).toFixed(1)}%)`,
        `Cross-entropy loss Δ: ${dL >= 0 ? '+' : ''}${dL.toFixed(3)}`,
      ];
    }
    if (activeSceneBeat === 'ch8-2-scale') {
      const s = SCALE_STOPS[scaleIdx];
      const mult = Math.max(1, Math.round(s.params / initInfo.manifest.total_params));
      return [
        `${s.label}: ${s.params.toLocaleString()} params (${mult.toLocaleString()}× this model)`,
        `${s.layers} layers · d_model ${s.dModel.toLocaleString()} · ${s.heads} heads · ctx ${s.context.toLocaleString()}`,
      ];
    }
    return [
      `${initInfo.manifest.total_params.toLocaleString()} real params · ${tokens.length} tokens · traced in ${activeTrace.elapsedMs}ms`,
    ];
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
    geluProbeX,
    temperature,
    topK,
    topP,
    weightDelta,
    replay,
    replayIdx,
    scaleIdx,
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
  const attnOff = (state.ablation.disabledAttnBlocks ?? []).includes(state.layer);
  const mlpOff = (state.ablation.disabledMlpBlocks ?? []).includes(state.layer);
  const layerOff = (state.ablation.disabledLayers ?? []).includes(state.layer);

  return (
    <div className="book-app">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-title">LLM Atlas</span>
          <span className="brand-tag">8-Chapter Interactive Transformer Textbook</span>
        </div>

        <form
          className="prompt-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!draftPrompt.trim()) return;
            setBpeStepIdx(0);
            setWeightDelta(0);
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
              setWeightDelta(0);
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
                    <span>Merge {safeBpeStep}/{bpeSteps.length - 1}</span>
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
                      setBpeStepIdx((s) => (s >= bpeSteps.length - 1 ? 0 : s + 1))
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
                    <input type="range" min={0} max={24} value={ropeSlotA} onChange={(e) => setRopeSlotA(Number(e.target.value))} />
                  </label>
                  <label className="hud-slider mono">
                    <span>Slot B: {ropeSlotB}</span>
                    <input type="range" min={0} max={24} value={ropeSlotB} onChange={(e) => setRopeSlotB(Number(e.target.value))} />
                  </label>
                </>
              )}

              {activeSceneBeat === 'ch3-2-cut-highway' && (
                <button
                  type="button"
                  className={`mini-btn ${resCut ? 'is-warm' : 'is-active'}`}
                  onClick={() => setState((prev) => ({ ...prev, ablation: toggleAllResiduals(prev.ablation) }))}
                >
                  {resCut ? 'Reconnect Residual Highway' : 'Cut Residual Highway'}
                </button>
              )}

              {activeSceneBeat === 'ch3-4-rmsnorm' && (
                <button
                  type="button"
                  className={`mini-btn ${lnFOff ? 'is-warm' : 'is-active'}`}
                  onClick={() => setState((prev) => ({ ...prev, ablation: toggleFinalLn(prev.ablation) }))}
                >
                  {lnFOff ? 'Restore Final LN_f' : 'Disable Final LN_f'}
                </button>
              )}

              {(activeSceneBeat === 'ch3-3-layernorm' ||
                state.chapterId === 4 ||
                state.chapterId === 5 ||
                activeSceneBeat === 'ch8-1-ablation') && (
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
                        className={`mini-btn mono ${off ? 'is-warm' : state.head === h ? 'is-active' : ''}`}
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
                  onClick={() => setState((prev) => ({ ...prev, showPreSoftmax: !prev.showPreSoftmax }))}
                >
                  {state.showPreSoftmax ? 'Raw Scores' : 'Softmax %'}
                </button>
              )}

              {(activeSceneBeat === 'attn-5-value-mixer' || activeSceneBeat === 'attn-6-multihead-full') && (
                <button
                  type="button"
                  className={`mini-btn ${headOff ? 'is-warm' : ''}`}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      ablation: toggleHeadDisabled(prev.ablation, prev.layer, prev.head),
                    }))
                  }
                >
                  {headOff ? `Re-enable H${state.head}` : `Disable H${state.head}`}
                </button>
              )}

              {activeSceneBeat === 'ch5-3-gelu' && (
                <label className="hud-slider mono">
                  <span>GELU x: {geluProbeX.toFixed(2)}</span>
                  <input type="range" min={-3.2} max={3.2} step={0.1} value={geluProbeX} onChange={(e) => setGeluProbeX(Number(e.target.value))} />
                </label>
              )}

              {(activeSceneBeat === 'ch6-3-temperature' ||
                activeSceneBeat === 'ch6-4-nucleus' ||
                activeSceneBeat === 'ch6-5-autoregression') && (
                <label className="hud-slider mono">
                  <span>τ: {temperature.toFixed(2)}</span>
                  <input type="range" min={0.1} max={2.0} step={0.05} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
                </label>
              )}

              {(activeSceneBeat === 'ch6-4-nucleus' || activeSceneBeat === 'ch6-5-autoregression') && (
                <>
                  <label className="hud-slider mono">
                    <span>k: {topK}</span>
                    <input type="range" min={1} max={40} step={1} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
                  </label>
                  <label className="hud-slider mono">
                    <span>p: {topP.toFixed(2)}</span>
                    <input type="range" min={0.1} max={1.0} step={0.05} value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
                  </label>
                </>
              )}

              {(activeSceneBeat === 'ch6-3-temperature' || activeSceneBeat === 'ch6-4-nucleus') && (
                <button type="button" className="primary-btn" onClick={handleSampleBurst}>
                  Sample 10
                </button>
              )}

              {activeSceneBeat === 'ch6-5-autoregression' && (
                <button type="button" className="primary-btn" disabled={busy} onClick={handleStepAutoregressive}>
                  Step +1 Token
                </button>
              )}

              {activeSceneBeat === 'ch7-3-manual-backprop' && (
                <>
                  <label className="hud-slider mono">
                    <span>Nudge Δw: {weightDelta >= 0 ? '+' : ''}{weightDelta.toFixed(2)}</span>
                    <input type="range" min={-2.0} max={2.0} step={0.05} value={weightDelta} onChange={(e) => setWeightDelta(Number(e.target.value))} />
                  </label>
                  {weightDelta !== 0 && (
                    <button type="button" className="mini-btn" onClick={() => setWeightDelta(0)}>
                      Reset Δw
                    </button>
                  )}
                </>
              )}

              {activeSceneBeat === 'ch7-4-training-replay' && replay && (
                <label className="hud-slider mono">
                  <span>Step {replay.checkpoints[Math.min(replayIdx, replay.checkpoints.length - 1)].step}</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, replay.checkpoints.length - 1)}
                    value={replayIdx}
                    onChange={(e) => setReplayIdx(Number(e.target.value))}
                  />
                </label>
              )}

              {activeSceneBeat === 'ch7-5-hallucination' && (
                <button
                  type="button"
                  className="primary-btn"
                  onClick={() => {
                    const text = 'The capital of France is a little girl named';
                    setDraftPrompt(text);
                    setState((prev) => ({ ...prev, prompt: text, queryIdx: null, keyIdx: null }));
                  }}
                >
                  Load Hallucination Specimen
                </button>
              )}

              {activeSceneBeat === 'ch8-1-ablation' && (
                <>
                  <button
                    type="button"
                    className={`mini-btn ${headOff ? 'is-warm' : ''}`}
                    onClick={() => setState((prev) => ({ ...prev, ablation: toggleHeadDisabled(prev.ablation, prev.layer, 3) }))}
                  >
                    {headOff ? 'H3 Off' : 'Disable H3'}
                  </button>
                  <button
                    type="button"
                    className={`mini-btn ${attnOff ? 'is-warm' : ''}`}
                    onClick={() => setState((prev) => ({ ...prev, ablation: toggleAttnBlock(prev.ablation, prev.layer) }))}
                  >
                    {attnOff ? 'Attn Off' : 'Disable Attn'}
                  </button>
                  <button
                    type="button"
                    className={`mini-btn ${mlpOff ? 'is-warm' : ''}`}
                    onClick={() => setState((prev) => ({ ...prev, ablation: toggleMlpBlock(prev.ablation, prev.layer) }))}
                  >
                    {mlpOff ? 'MLP Off' : 'Disable MLP'}
                  </button>
                  <button
                    type="button"
                    className={`mini-btn ${layerOff ? 'is-warm' : ''}`}
                    onClick={() => setState((prev) => ({ ...prev, ablation: toggleLayerDisabled(prev.ablation, prev.layer) }))}
                  >
                    {layerOff ? 'Block Skipped' : 'Skip Block'}
                  </button>
                </>
              )}

              {activeSceneBeat === 'ch8-2-scale' && (
                <label className="hud-slider mono">
                  <span>Scale: {SCALE_STOPS[scaleIdx].label.split(' ')[0]}</span>
                  <input type="range" min={0} max={SCALE_STOPS.length - 1} value={scaleIdx} onChange={(e) => setScaleIdx(Number(e.target.value))} />
                </label>
              )}
            </div>

            <div className="hud-group">
              {hasActiveAblation(state.ablation) && (
                <button type="button" className="mini-btn is-warm" onClick={() => setState((prev) => ({ ...prev, ablation: {} }))}>
                  Reset Ablation
                </button>
              )}
              <button type="button" className="mini-btn" onClick={() => stageRef.current?.resetCamera()}>
                Reset View
              </button>
            </div>
          </div>

          {activeTrace && (
            <div className="stage-hud-bottom">
              {sampleBurst.length > 0 &&
                (activeSceneBeat === 'ch6-3-temperature' || activeSceneBeat === 'ch6-4-nucleus') && (
                  <div className="sample-burst mono">
                    <span className="hud-label">10 live draws:</span>
                    {sampleBurst.map((s, i) => (
                      <span key={i} className="burst-chip">
                        {s}
                      </span>
                    ))}
                  </div>
                )}

              <div className="token-bar" aria-label="Select token">
                <span className="hud-label">
                  {activeSceneBeat === 'attn-2-dot-product' ? 'Compare Key:' : 'Inspect Token:'}
                </span>
                {activeTrace.tokens.map((tok, idx) => {
                  const isQ = idx === resolvedQueryIdx;
                  const isK = activeSceneBeat === 'attn-2-dot-product' && idx === resolvedKeyIdx;
                  const disabledKey = activeSceneBeat === 'attn-2-dot-product' && idx > resolvedQueryIdx;
                  return (
                    <button
                      key={`${tok.start}-${idx}`}
                      type="button"
                      disabled={disabledKey}
                      className={`tok-btn mono ${isQ ? 'is-query' : ''} ${isK ? 'is-key' : ''}`}
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
