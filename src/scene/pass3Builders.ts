import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { SceneBeatId } from '../content/bookData';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  TokenizerData,
} from '../engine/types';
import {
  createTensorTexture,
  divergingColor,
  formatNum,
  formatToken,
  vectorL2Norm,
} from '../utils/math';
import {
  computeManualWeightLandscape,
  computePerTokenLoss,
  computeSampledDistribution,
  countActiveNeurons,
  FRONTIER_DIFFS,
  geluTanh,
  SCALE_STOPS,
  type TrainingReplayData,
} from '../utils/pass3Math';
import type { Pass2StageContext } from './pass2Builders';

export interface Pass3Snapshot {
  beat: SceneBeatId;
  layer: number;
  head: number;
  queryIdx: number;
  ablation: AblationConfig;
  traceRes: EngineTraceResult;
  baselineRes: EngineTraceResult;
  tokenizer: TokenizerData;
  wteWeights: Float32Array | null;
  replay: TrainingReplayData | null;
  geluProbeX: number;
  temperature: number;
  topK: number;
  topP: number;
  weightDelta: number;
  replayIdx: number;
  scaleIdx: number;
}

function addPlate(
  ctx: Pass2StageContext,
  mat: number[][],
  size: [number, number, number],
  pos: [number, number, number],
  frameHex = '#cbd5e1',
  mode: 'diverging' | 'sequential' = 'diverging'
): void {
  const tex = ctx.trackTex(createTensorTexture(mat, mode, false));
  const side = ctx.trackMat(
    new THREE.MeshStandardMaterial({ color: frameHex, roughness: 0.45 })
  );
  const face = ctx.trackMat(
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.3 })
  );
  const mesh = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(size[0], size[1], size[2], 2, 0.04)),
    [side, side, face, side, face, face]
  );
  mesh.position.set(pos[0], pos[1], pos[2]);
  mesh.castShadow = true;
  ctx.stageRoot.add(mesh);
}

export function buildChapter5678Beat(
  ctx: Pass2StageContext,
  snap: Pass3Snapshot
): boolean {
  switch (snap.beat) {
    case 'ch5-1-mlp-role':
      buildCh5Role(ctx, snap);
      return true;
    case 'ch5-2-expansion':
      buildCh5Expansion(ctx, snap);
      return true;
    case 'ch5-3-gelu':
      buildCh5Gelu(ctx, snap);
      return true;
    case 'ch5-4-wdown':
      buildCh5Wdown(ctx, snap);
      return true;
    case 'ch5-5-swiglu-moe':
      buildCh5Moe(ctx, snap);
      return true;
    case 'ch6-1-logit-lens':
      buildCh6LogitLens(ctx, snap);
      return true;
    case 'ch6-2-unembed':
      buildCh6Unembed(ctx, snap);
      return true;
    case 'ch6-3-temperature':
      buildCh6Sampling(ctx, snap, false);
      return true;
    case 'ch6-4-nucleus':
      buildCh6Sampling(ctx, snap, true);
      return true;
    case 'ch6-5-autoregression':
      buildCh6Autoreg(ctx, snap);
      return true;
    case 'ch6-6-kv-quant':
      buildCh6KvCache(ctx, snap);
      return true;
    case 'ch7-1-step0':
      buildCh7Step0(ctx, snap);
      return true;
    case 'ch7-2-cross-entropy':
      buildCh7CrossEntropy(ctx, snap);
      return true;
    case 'ch7-3-manual-backprop':
      buildCh7ManualBackprop(ctx, snap);
      return true;
    case 'ch7-4-training-replay':
      buildCh7Replay(ctx, snap);
      return true;
    case 'ch7-5-hallucination':
      buildCh7Hallucination(ctx, snap);
      return true;
    case 'ch8-1-ablation':
      buildCh8Ablation(ctx, snap);
      return true;
    case 'ch8-2-scale':
      buildCh8Scale(ctx, snap);
      return true;
    case 'ch8-3-frontier-diff':
      buildCh8FrontierDiff(ctx);
      return true;
    case 'ch8-4-post-training':
      buildCh8PostTraining(ctx);
      return true;
    case 'ch8-5-capstone':
      buildCh8Capstone(ctx, snap);
      return true;
    default:
      return false;
  }
}

// CH 5.1: Parameter Mass — Attention (33.5%) vs. MLP (66.5%)
function buildCh5Role(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const attnBox = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(2.1, 1.45, 1.6, 3, 0.08)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#2563eb', roughness: 0.3 }))
  );
  attnBox.position.set(-2.3, 0.45, 0);
  attnBox.castShadow = true;
  ctx.stageRoot.add(attnBox);
  ctx.addLabel('Self-Attention (33.5%)', '65,536 params · mixes across tokens', [-2.3, 1.8, 0], 2.25, '#1d4ed8');

  const mlpBox = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(3.4, 2.35, 1.9, 3, 0.08)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.3 }))
  );
  mlpBox.position.set(1.9, 0.9, 0);
  mlpBox.castShadow = true;
  ctx.stageRoot.add(mlpBox);
  ctx.addLabel(
    `Block ${snap.layer} MLP (66.5%)`,
    '131,072 params · per-token thinking',
    [1.9, 2.65, 0],
    2.45,
    '#b45309'
  );
}

// CH 5.2: 4x Expansion (128 -> 512)
function buildCh5Expansion(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const lt = snap.traceRes.trace.layers[snap.layer];
  if (lt.skipped) {
    ctx.addLabel(`Block ${snap.layer} disabled`, undefined, [0, 0.6, 0], 2.4, '#b45309');
    return;
  }
  const active: ActiveLayerTrace = lt;
  const T = snap.traceRes.tokens.length;
  addPlate(ctx, active.ln2Out, [1.5, 1.8, 0.22], [-3.6, 0.4, 0], '#93c5fd');
  ctx.addLabel('Input LN₂(u)', `[${T} × 128]`, [-3.6, 1.8, 0], 1.75, '#1d4ed8');

  addPlate(ctx, active.mlp.hidden, [4.6, 2.2, 0.22], [1.4, 0.5, 0], '#f59e0b');
  ctx.addLabel('W_up 4× Expansion', `[${T} × 512] pattern detectors`, [1.4, 2.15, 0], 2.45, '#b45309');

  // Fan-out guide rays
  const lineMat = ctx.trackMat(new THREE.LineBasicMaterial({ color: '#d97706', transparent: true, opacity: 0.7 }));
  for (const dy of [-0.75, 0, 0.75]) {
    const g = ctx.trackGeo(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-2.75, 0.4 + dy * 0.7, 0),
        new THREE.Vector3(-1.0, 0.5 + dy * 1.15, 0),
      ])
    );
    ctx.stageRoot.add(new THREE.Line(g, lineMat));
  }
}

// CH 5.3: Interactive GELU Curve + Live 512-Neuron Sparsity
function buildCh5Gelu(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const lt = snap.traceRes.trace.layers[snap.layer];
  const xVal = snap.geluProbeX;
  const yVal = geluTanh(xVal);

  // Foreground: 3D GELU Curve from x = -3.5 to +3.5
  const curvePts: THREE.Vector3[] = [];
  for (let i = 0; i <= 48; i++) {
    const x = -3.5 + (i / 48) * 7.0;
    const y = geluTanh(x);
    curvePts.push(new THREE.Vector3(x * 0.95, y * 0.65 + 0.15, 1.15));
  }
  const curve = new THREE.CatmullRomCurve3(curvePts);
  const tube = new THREE.Mesh(
    ctx.trackGeo(new THREE.TubeGeometry(curve, 64, 0.055, 10, false)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.25 }))
  );
  ctx.stageRoot.add(tube);

  // Axes for GELU curve
  const axisMat = ctx.trackMat(new THREE.LineBasicMaterial({ color: '#94a3b8' }));
  ctx.stageRoot.add(
    new THREE.Line(
      ctx.trackGeo(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-3.4, 0.15, 1.15),
          new THREE.Vector3(3.4, 0.15, 1.15),
        ])
      ),
      axisMat
    )
  );
  ctx.stageRoot.add(
    new THREE.Line(
      ctx.trackGeo(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, -0.4, 1.15),
          new THREE.Vector3(0, 2.5, 1.15),
        ])
      ),
      axisMat
    )
  );

  // Draggable Probe Sphere on GELU curve
  const probe = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.16, 20, 20)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.25 }))
  );
  probe.position.set(xVal * 0.95, yVal * 0.65 + 0.15, 1.15);
  ctx.stageRoot.add(probe);
  ctx.addLabel(
    `GELU(${xVal.toFixed(2)}) = ${yVal.toFixed(3)}`,
    xVal < 0 ? 'Negative squashed toward 0' : 'Positive match passes through',
    [xVal * 0.95, yVal * 0.65 + 0.82, 1.15],
    2.25,
    xVal < 0 ? '#64748b' : '#b45309'
  );

  // Background: Live 512-neuron sparsity bar strip (64 binned bars)
  if (!lt.skipped) {
    const hidden = lt.mlp.hidden[snap.queryIdx];
    const firing = countActiveNeurons(hidden, 0.05);
    const numBars = 64;
    const bin = hidden.length / numBars;
    let maxA = 1e-5;
    for (const v of hidden) maxA = Math.max(maxA, Math.abs(v));
    const barGeo = ctx.trackGeo(new THREE.BoxGeometry(0.08, 1, 0.18));
    for (let b = 0; b < numBars; b++) {
      let peak = 0;
      for (let j = 0; j < bin; j++) {
        const val = hidden[Math.floor(b * bin + j)] ?? 0;
        if (Math.abs(val) > Math.abs(peak)) peak = val;
      }
      const h = Math.max(0.05, (Math.abs(peak) / maxA) * 1.65);
      const bar = new THREE.Mesh(
        barGeo,
        ctx.trackMat(
          new THREE.MeshStandardMaterial({
            color: peak > 0.05 ? '#d97706' : '#cbd5e1',
            roughness: 0.3,
          })
        )
      );
      bar.scale.y = h;
      bar.position.set(-3.4 + (b / (numBars - 1)) * 6.8, -0.3 + h / 2, -1.55);
      ctx.stageRoot.add(bar);
    }
    ctx.addLabel(
      `Live 512 Neurons (Block ${snap.layer})`,
      `${firing} / 512 firing (>0.05) · ${((firing / 512) * 100).toFixed(0)}% active`,
      [0, 1.85, -1.55],
      2.65,
      '#b45309'
    );
  }
}

// CH 5.4: W_down Compression (512 -> 128) + Residual Add
function buildCh5Wdown(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const lt = snap.traceRes.trace.layers[snap.layer];
  if (lt.skipped) {
    ctx.addLabel(`Block ${snap.layer} disabled`, undefined, [0, 0.6, 0], 2.4, '#b45309');
    return;
  }
  const active: ActiveLayerTrace = lt;
  const T = snap.traceRes.tokens.length;
  addPlate(ctx, active.mlp.hidden, [4.2, 2.0, 0.22], [-2.1, 0.45, 0], '#f59e0b');
  ctx.addLabel('Active GELU Neurons', `[${T} × 512]`, [-2.1, 1.95, 0], 2.0, '#b45309');

  addPlate(ctx, active.mlp.output, [1.5, 1.8, 0.22], [2.7, 0.45, 0], '#93c5fd');
  ctx.addLabel(
    'W_down Output',
    `[${T} × 128] · ||Δ||=${formatNum(vectorL2Norm(active.mlp.output[snap.queryIdx]), 2)}`,
    [2.7, 1.88, 0],
    2.1,
    '#0f766e'
  );

  const plus = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.22, 20, 20)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#c89b54', metalness: 0.65, roughness: 0.25 }))
  );
  plus.position.set(4.35, -0.5, 0);
  ctx.stageRoot.add(plus);
  ctx.addLabel('+ Residual', undefined, [4.35, 0.05, 0], 1.15, '#1d4ed8');
}

// CH 5.5: Mixture of Experts (MoE) Router + 6 Specialist Experts (Top-2 active)
function buildCh5Moe(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const tok = snap.traceRes.tokens[snap.queryIdx];
  const activeA = tok.id % 6;
  const activeB = (tok.id + 2) % 6;

  // Router hub on left
  const router = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(1.1, 1.1, 1.1, 3, 0.14)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.28 }))
  );
  router.position.set(-3.4, 0.35, 0);
  router.castShadow = true;
  ctx.stageRoot.add(router);
  ctx.addLabel('MoE Router', `Token “${formatToken(tok.text)}” → Top-2`, [-3.4, 1.45, 0], 2.0, '#1d4ed8');

  const expertGeo = ctx.trackGeo(new RoundedBoxGeometry(1.3, 0.72, 0.85, 2, 0.06));
  for (let e = 0; e < 6; e++) {
    const z = -2.5 + e * 1.0;
    const isOn = e === activeA || e === activeB;
    const mesh = new THREE.Mesh(
      expertGeo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isOn ? '#d97706' : '#cbd5e1',
          roughness: 0.32,
          transparent: !isOn,
          opacity: isOn ? 1 : 0.45,
        })
      )
    );
    mesh.position.set(1.6, isOn ? 0.55 : 0.15, z);
    mesh.castShadow = isOn;
    ctx.stageRoot.add(mesh);

    if (isOn) {
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-2.75, 0.35, 0),
        new THREE.Vector3(-0.6, 0.75, z * 0.5),
        new THREE.Vector3(0.9, 0.55, z),
      ]);
      const pipe = new THREE.Mesh(
        ctx.trackGeo(new THREE.TubeGeometry(curve, 20, 0.055, 8, false)),
        ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.25 }))
      );
      ctx.stageRoot.add(pipe);
      ctx.addLabel(`Expert ${e} (Active)`, 'SwiGLU FFN', [1.6, 1.35, z], 1.45, '#b45309');
    }
  }
}

// CH 6.1: 4-Block Live Logit Lens Towers
function buildCh6LogitLens(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const { logitLens } = snap.traceRes.trace;
  for (let l = 0; l < logitLens.length; l++) {
    const x = -3.3 + l * 2.2;
    const top = logitLens[l].topTokens[0];
    const prob = top?.prob ?? 0;
    const h = Math.max(0.28, prob * 3.4);
    const bar = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(1.1, h, 1.1, 3, 0.08)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: l === 3 ? '#0f766e' : '#1d4ed8',
          roughness: 0.28,
        })
      )
    );
    bar.position.set(x, h / 2 - 0.5, 0);
    bar.castShadow = true;
    ctx.stageRoot.add(bar);

    const tokText = top ? formatToken(snap.tokenizer.vocab[top.id] ?? '') : '-';
    ctx.addLabel(
      `Block ${l}: “${tokText}”`,
      `${(prob * 100).toFixed(1)}% confidence`,
      [x, h + 0.12, 0],
      1.85,
      l === 3 ? '#0f766e' : '#1d4ed8'
    );
  }
}

// CH 6.2: Unembedding & 512 Raw Signed Logits
function buildCh6Unembed(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const logits = snap.traceRes.trace.finalLogits;
  const sorted = logits
    .map((z, id) => ({ id, z }))
    .sort((a, b) => b.z - a.z);
  const picks = [...sorted.slice(0, 7), ...sorted.slice(-3)];
  let maxAbs = 1e-5;
  for (const item of picks) maxAbs = Math.max(maxAbs, Math.abs(item.z));

  const rail = new THREE.Mesh(
    ctx.trackGeo(new THREE.BoxGeometry(8.6, 0.05, 0.8)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1' }))
  );
  ctx.stageRoot.add(rail);

  picks.forEach((item, idx) => {
    const x = -3.8 + idx * 0.84;
    const h = Math.max(0.12, (Math.abs(item.z) / maxAbs) * 2.1);
    const bar = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(0.5, h, 0.5, 2, 0.04)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: item.z >= 0 ? '#d97706' : '#1d4ed8',
          roughness: 0.28,
        })
      )
    );
    bar.position.set(x, item.z >= 0 ? h / 2 : -h / 2, 0);
    bar.castShadow = true;
    ctx.stageRoot.add(bar);

    if (idx < 5 || idx === 9) {
      ctx.addLabel(
        `“${formatToken(snap.tokenizer.vocab[item.id] ?? '')}”`,
        `z = ${item.z.toFixed(2)}`,
        [x, (item.z >= 0 ? h : 0.2) + 0.5, 0],
        1.05,
        item.z >= 0 ? '#b45309' : '#1d4ed8'
      );
    }
  });
  ctx.addLabel('Raw Signed Logits (Top 7 & Bottom 3 of 512)', 'Before Softmax', [0, 2.75, -0.3], 2.9, '#b45309');
}

// CH 6.3 & 6.4: Temperature & Top-k / Top-p Cutoff Curtain
function buildCh6Sampling(
  ctx: Pass2StageContext,
  snap: Pass3Snapshot,
  showCurtain: boolean
): void {
  const { finalLogits, finalProbs } = snap.traceRes.trace;
  const activePool = computeSampledDistribution(
    finalLogits,
    snap.temperature,
    showCurtain ? snap.topK : 512,
    showCurtain ? snap.topP : 1.0
  );
  const poolMap = new Map(activePool.map((x) => [x.id, x.prob]));

  const top10 = finalProbs
    .map((rawP, id) => ({
      id,
      rawP,
      p: poolMap.get(id) ?? 0,
      kept: poolMap.has(id),
    }))
    .sort((a, b) => b.rawP - a.rawP)
    .slice(0, 10);

  let cutoffIdx = top10.length;
  top10.forEach((item, rank) => {
    if (!item.kept && cutoffIdx === top10.length) cutoffIdx = rank;
    const x = -3.6 + rank * 0.8;
    const prob = item.kept ? item.p : item.rawP;
    const h = Math.max(0.12, prob * 3.6);
    const bar = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(0.5, h, 0.5, 2, 0.05)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: !item.kept ? '#cbd5e1' : rank === 0 ? '#1d4ed8' : '#3b82f6',
          roughness: 0.28,
        })
      )
    );
    bar.position.set(x, -0.7 + h / 2, 0);
    bar.castShadow = true;
    ctx.stageRoot.add(bar);

    if (rank < 6) {
      ctx.addLabel(
        `“${formatToken(snap.tokenizer.vocab[item.id] ?? '')}”`,
        item.kept ? `${(prob * 100).toFixed(1)}%` : 'cut (0%)',
        [x, -0.25 + h + 0.32, 0],
        1.05,
        item.kept ? '#1d4ed8' : '#64748b',
        !item.kept
      );
    }
  });

  if (showCurtain && cutoffIdx > 0 && cutoffIdx < top10.length) {
    const cx = -3.6 + (cutoffIdx - 0.5) * 0.8;
    const curtain = new THREE.Mesh(
      ctx.trackGeo(new THREE.BoxGeometry(0.06, 3.2, 1.6)),
      ctx.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: '#dc2626',
          transparent: true,
          opacity: 0.35,
          roughness: 0.2,
        })
      )
    );
    curtain.position.set(cx, 0.9, 0);
    ctx.stageRoot.add(curtain);
    ctx.addLabel('Nucleus Cutoff', `k=${snap.topK}, p=${snap.topP.toFixed(2)}`, [cx, 2.75, 0], 1.75, '#dc2626');
  }
}

// CH 6.5: Autoregressive Loop Arc
function buildCh6Autoreg(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  buildCh6Sampling(ctx, snap, true);
  const loopCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-3.6, 2.2, 0),
    new THREE.Vector3(-0.2, 3.3, -1.2),
    new THREE.Vector3(3.4, 2.2, 0),
  ]);
  const pipe = new THREE.Mesh(
    ctx.trackGeo(new THREE.TubeGeometry(loopCurve, 32, 0.055, 8, false)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#0f766e', roughness: 0.25 }))
  );
  ctx.stageRoot.add(pipe);
  ctx.addLabel(
    'Autoregressive Feedback Loop',
    `Prompt length = ${snap.traceRes.tokens.length} / 64 · Click "Step +1 Token"`,
    [0, 3.65, -1.2],
    2.9,
    '#0f766e'
  );
}

// CH 6.6: Prefill Frozen KV Cache vs. Single New Decode Column
function buildCh6KvCache(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const T = Math.min(14, Math.max(4, snap.traceRes.tokens.length));
  const spacing = 0.58;
  const startX = -((T - 1) * spacing) / 2;
  const colGeo = ctx.trackGeo(new RoundedBoxGeometry(0.42, 1.9, 0.9, 2, 0.05));

  for (let i = 0; i < T; i++) {
    const isNewest = i === T - 1;
    const mesh = new THREE.Mesh(
      colGeo,
      ctx.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: isNewest ? '#d97706' : '#93c5fd',
          roughness: isNewest ? 0.25 : 0.18,
          transmission: isNewest ? 0 : 0.45,
          transparent: !isNewest,
          opacity: isNewest ? 1 : 0.72,
        })
      )
    );
    mesh.position.set(startX + i * spacing, isNewest ? 0.45 : 0.2, 0);
    mesh.castShadow = true;
    ctx.stageRoot.add(mesh);
  }
  ctx.addLabel('Frozen KV Cache (Past K & V)', 'Never recomputed during Decode!', [startX + ((T - 2) * spacing) / 2, 1.85, 0], 2.6, '#1d4ed8');
  ctx.addLabel('New Token', '1 column O(T)', [startX + (T - 1) * spacing, 2.0, 0], 1.45, '#b45309');
}

// CH 7.1: Step 0 Random Uniform vs. Trained Sharp Distribution
function buildCh7Step0(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const top8 = snap.baselineRes.trace.finalProbs
    .map((p, id) => ({ id, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 8);

  const geo = ctx.trackGeo(new RoundedBoxGeometry(0.54, 1, 0.54, 2, 0.04));
  top8.forEach((item, i) => {
    const x = -3.15 + i * 0.9;
    // Back row: Step 0 blind uniform (1/512 ≈ 0.2%)
    const uBar = new THREE.Mesh(
      geo,
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#94a3b8', roughness: 0.45 }))
    );
    uBar.scale.y = 0.22;
    uBar.position.set(x, 0.11, -1.25);
    ctx.stageRoot.add(uBar);

    // Front row: Trained sharp distribution
    const h = Math.max(0.12, item.p * 3.4);
    const tBar = new THREE.Mesh(
      geo,
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.28 }))
    );
    tBar.scale.y = h;
    tBar.position.set(x, h / 2, 1.15);
    tBar.castShadow = true;
    ctx.stageRoot.add(tBar);
  });

  ctx.addLabel('Step 0: Random Weights (Loss ≈ 6.24)', 'Flat ~0.2% across all 512 tokens', [0, 1.15, -1.25], 2.9, '#64748b');
  ctx.addLabel('After Training: Sharp Predictions', 'Structure emerged from gradients alone', [0, 2.8, 1.15], 2.9, '#1d4ed8');
}

// CH 7.2: Per-Token Live Cross-Entropy Surprise Bars
function buildCh7CrossEntropy(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  if (!snap.wteWeights) return;
  const { entries } = computePerTokenLoss(
    snap.baselineRes.trace.lnFOut,
    snap.baselineRes.trace.tokenIds,
    snap.wteWeights,
    snap.tokenizer.vocab_size
  );
  if (entries.length === 0) return;

  const span = Math.min(9.0, Math.max(4.6, entries.length * 0.65));
  const step = span / entries.length;
  const geo = ctx.trackGeo(new RoundedBoxGeometry(step * 0.72, 1, 0.56, 2, 0.04));

  let worst = entries[0];
  for (const e of entries) if (e.loss > worst.loss) worst = e;

  entries.forEach((e, i) => {
    const x = -span / 2 + (i + 0.5) * step;
    const h = Math.max(0.12, Math.min(3.0, (e.loss / 6.5) * 2.8));
    const isHigh = e.loss > 3.2;
    const bar = new THREE.Mesh(
      geo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isHigh ? '#d97706' : '#2563eb',
          roughness: 0.3,
        })
      )
    );
    bar.scale.y = h;
    bar.position.set(x, h / 2, 0);
    bar.castShadow = true;
    ctx.stageRoot.add(bar);

    if (e === worst || entries.length <= 10) {
      ctx.addLabel(
        formatToken(snap.tokenizer.vocab[e.targetId] ?? ''),
        `L=${e.loss.toFixed(2)}`,
        [x, h + 0.48, 0],
        0.96,
        isHigh ? '#b45309' : '#1d4ed8'
      );
    }
  });
  ctx.addLabel('Per-Token Surprise: L = -log P(true)', 'Tall amber = unexpected · Short blue = predicted well', [0, 3.3, -0.3], 3.1, '#b45309');
}

// CH 7.3: "Be Backpropagation Yourself" — Exact 3D Convex Loss Valley
function buildCh7ManualBackprop(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  if (!snap.wteWeights) return;
  const land = computeManualWeightLandscape(
    snap.baselineRes.trace.lnFOut,
    snap.baselineRes.trace.tokenIds,
    snap.wteWeights,
    snap.tokenizer.vocab_size,
    snap.weightDelta
  );
  if (!land) {
    ctx.addLabel('Type at least 2 tokens', 'Needed to compute Cross-Entropy Loss', [0, 0.6, 0], 2.4);
    return;
  }

  let minL = Infinity;
  let maxL = -Infinity;
  for (const pt of land.curve) {
    if (pt.loss < minL) minL = pt.loss;
    if (pt.loss > maxL) maxL = pt.loss;
  }
  const spanL = Math.max(0.08, maxL - minL);
  const toY = (loss: number) => ((loss - minL) / spanL) * 2.1 + 0.1;

  const pts = land.curve.map(
    (p) => new THREE.Vector3(p.delta * 2.0, toY(p.loss), 0)
  );
  const valleyCurve = new THREE.CatmullRomCurve3(pts);
  const tube = new THREE.Mesh(
    ctx.trackGeo(new THREE.TubeGeometry(valleyCurve, 48, 0.075, 12, false)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.25 }))
  );
  ctx.stageRoot.add(tube);

  // Trained baseline marker at delta = 0
  const baseDot = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.13, 16, 16)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#0f766e' }))
  );
  baseDot.position.set(0, toY(land.baseLoss), 0);
  ctx.stageRoot.add(baseDot);
  ctx.addLabel('Trained Value (Δ=0)', `Loss ${land.baseLoss.toFixed(4)}`, [0, toY(land.baseLoss) - 0.55, 0.35], 1.65, '#0f766e');

  // Rolling user ball at current weightDelta
  const ballX = snap.weightDelta * 2.0;
  const ballY = toY(land.currentLoss);
  const ball = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.22, 24, 24)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.22 }))
  );
  ball.position.set(ballX, ballY, 0);
  ball.castShadow = true;
  ctx.stageRoot.add(ball);

  const dir = land.gradient >= 0 ? '← Step left (-)' : 'Step right (+) →';
  ctx.addLabel(
    `Loss = ${land.currentLoss.toFixed(4)}`,
    `∂L/∂w = ${land.gradient.toFixed(4)} · ${dir}`,
    [ballX, ballY + 0.72, 0],
    2.4,
    '#b45309'
  );
}

// CH 7.4: Real Training Replay 3D Loss Curve
function buildCh7Replay(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const replay = snap.replay;
  if (!replay || replay.loss_curve.length < 2) {
    ctx.addLabel('Loading training_replay.json...', undefined, [0, 0.6, 0], 2.2);
    return;
  }

  const curveData = replay.loss_curve;
  const maxStep = Math.max(1, curveData[curveData.length - 1].step);
  let maxLoss = 1;
  for (const p of curveData) maxLoss = Math.max(maxLoss, p.val_loss);

  const pts = curveData.map(
    (p) => new THREE.Vector3(-4.0 + (p.step / maxStep) * 8.0, (p.val_loss / maxLoss) * 2.6, 0)
  );
  const curve = new THREE.CatmullRomCurve3(pts);
  ctx.stageRoot.add(
    new THREE.Mesh(
      ctx.trackGeo(new THREE.TubeGeometry(curve, 64, 0.065, 10, false)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.25 }))
    )
  );

  const ckpt =
    replay.checkpoints[
      Math.min(snap.replayIdx, Math.max(0, replay.checkpoints.length - 1))
    ];
  const cx = -4.0 + (ckpt.step / maxStep) * 8.0;
  const cy = (ckpt.val_loss / maxLoss) * 2.6;

  const cursor = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.18, 20, 20)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.25 }))
  );
  cursor.position.set(cx, cy, 0);
  ctx.stageRoot.add(cursor);

  const post = new THREE.Mesh(
    ctx.trackGeo(new THREE.CylinderGeometry(0.03, 0.03, cy, 10)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706' }))
  );
  post.position.set(cx, cy / 2, 0);
  ctx.stageRoot.add(post);

  ctx.addLabel(
    `Step ${ckpt.step} / ${maxStep}`,
    `Validation Loss = ${ckpt.val_loss.toFixed(3)}`,
    [cx, cy + 0.68, 0],
    2.25,
    '#b45309'
  );
}

// CH 7.5: Hallucination Specimen Comparison
function buildCh7Hallucination(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  buildCh6Sampling(ctx, snap, false);
  ctx.addLabel(
    'Trained ONLY on TinyStories (No Encyclopedia)',
    'High confidence reflects story grammar, not verified truth',
    [0, 3.25, -0.3],
    3.1,
    '#b45309'
  );
}

// CH 8.1: Surgical Ablation Before vs. After Skyline
function buildCh8Ablation(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const baseProbs = snap.baselineRes.trace.finalProbs;
  const curProbs = snap.traceRes.trace.finalProbs;
  const isAblated = snap.traceRes !== snap.baselineRes;

  const top7 = baseProbs
    .map((bp, id) => ({ id, bp, cp: curProbs[id] }))
    .sort((a, b) => Math.max(b.bp, b.cp) - Math.max(a.bp, a.cp))
    .slice(0, 7);

  top7.forEach((item, i) => {
    const x = -3.0 + i * 1.0;
    const ch = Math.max(0.12, item.cp * 3.5);
    const bar = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(0.52, ch, 0.52, 2, 0.05)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isAblated ? '#d97706' : '#1d4ed8',
          roughness: 0.28,
        })
      )
    );
    bar.position.set(x, -0.65 + ch / 2, 0);
    bar.castShadow = true;
    ctx.stageRoot.add(bar);

    if (isAblated) {
      const gh = Math.max(0.12, item.bp * 3.5);
      const ghost = new THREE.Mesh(
        ctx.trackGeo(new THREE.BoxGeometry(0.6, gh, 0.6)),
        ctx.trackMat(
          new THREE.MeshBasicMaterial({
            color: '#1d4ed8',
            wireframe: true,
            transparent: true,
            opacity: 0.55,
          })
        )
      );
      ghost.position.set(x, -0.65 + gh / 2, 0);
      ctx.stageRoot.add(ghost);
    }

    if (i < 5) {
      ctx.addLabel(
        `“${formatToken(snap.tokenizer.vocab[item.id] ?? '')}”`,
        isAblated
          ? `${(item.bp * 100).toFixed(0)}%→${(item.cp * 100).toFixed(0)}%`
          : `${(item.bp * 100).toFixed(1)}%`,
        [x, -0.2 + Math.max(ch, item.bp * 3.5) + 0.35, 0],
        1.15,
        isAblated ? '#b45309' : '#1d4ed8'
      );
    }
  });
}

// CH 8.2: Scale Cathedral (Logarithmic Parameter Towers)
function buildCh8Scale(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  SCALE_STOPS.forEach((stop, i) => {
    const x = -4.0 + i * 2.0;
    const isSel = i === snap.scaleIdx;
    const logH = 0.45 + (Math.log10(stop.params) - 5.9) * 0.55;
    const w = 0.65 + i * 0.14;
    const tower = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(w, logH, w, 3, 0.06)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isSel ? '#d97706' : i === 0 ? '#0f766e' : '#3b82f6',
          roughness: 0.28,
        })
      )
    );
    tower.position.set(x, -0.8 + logH / 2, 0);
    tower.castShadow = true;
    ctx.stageRoot.add(tower);

    ctx.addLabel(
      stop.label.split(' ')[0],
      i === 0 ? '862K' : i === 1 ? '124M' : i === 2 ? '8B' : i === 3 ? '70B' : '1T+',
      [x, -0.8 + logH + 0.48, 0],
      1.3,
      isSel ? '#b45309' : '#1d4ed8'
    );
  });
}

// CH 8.3: Modern Frontier Diff Paired Columns
function buildCh8FrontierDiff(ctx: Pass2StageContext): void {
  const geo = ctx.trackGeo(new RoundedBoxGeometry(1.15, 0.85, 0.72, 2, 0.06));
  FRONTIER_DIFFS.forEach((item, i) => {
    const x = -4.0 + i * 2.0;
    const mHere = new THREE.Mesh(
      geo,
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.3 }))
    );
    mHere.position.set(x, 0.15, -1.15);
    ctx.stageRoot.add(mHere);
    ctx.addLabel(item.part, item.here, [x, 1.0, -1.15], 1.35, '#1d4ed8');

    const mFront = new THREE.Mesh(
      geo,
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.3 }))
    );
    mFront.position.set(x, 0.45, 1.15);
    ctx.stageRoot.add(mFront);
    ctx.addLabel('Frontier', item.frontier, [x, 1.35, 1.15], 1.45, '#b45309');
  });
}

// CH 8.4: Post-Training 4-Stage Pipeline
function buildCh8PostTraining(ctx: Pass2StageContext): void {
  const stages = [
    { name: '1. Pretraining', sub: 'Raw next-token corpus', color: '#64748b', h: 1.1 },
    { name: '2. SFT', sub: 'Chat dialogue demos', color: '#2563eb', h: 1.65 },
    { name: '3. RLHF / DPO', sub: 'Helpfulness & safety', color: '#0f766e', h: 2.2 },
    { name: '4. Reasoning RL', sub: 'Verifiable step-by-step', color: '#d97706', h: 2.75 },
  ];
  stages.forEach((s, i) => {
    const x = -3.3 + i * 2.2;
    const block = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(1.45, s.h, 1.1, 3, 0.08)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: s.color, roughness: 0.28 }))
    );
    block.position.set(x, -0.75 + s.h / 2, 0);
    block.castShadow = true;
    ctx.stageRoot.add(block);
    ctx.addLabel(s.name, s.sub, [x, -0.75 + s.h + 0.55, 0], 1.85, s.color);
  });
}

// CH 8.5: Complete 4-Block Capstone Cutaway
function buildCh8Capstone(ctx: Pass2StageContext, snap: Pass3Snapshot): void {
  const { trace } = snap.traceRes;
  const bus = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(9.6, 0.16, 0.54, 2, 0.04)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8', roughness: 0.25 }))
  );
  bus.position.set(0, -0.7, 0);
  ctx.stageRoot.add(bus);

  for (let l = 0; l < 4; l++) {
    const x = -3.0 + l * 2.0;
    const lt = trace.layers[l];
    if (!lt.skipped) {
      addPlate(ctx, lt.attn.attnWeights[0], [0.7, 0.7, 0.15], [x - 0.36, 0.35, 0.18], '#93c5fd', 'sequential');
      addPlate(ctx, lt.mlp.hidden, [0.7, 1.2, 0.15], [x + 0.36, 0.45, -0.1], '#f59e0b', 'diverging');
    }
    const top = trace.logitLens[l]?.topTokens[0];
    ctx.addLabel(
      `Block ${l}`,
      top ? `“${formatToken(snap.tokenizer.vocab[top.id] ?? '')}” ${(top.prob * 100).toFixed(0)}%` : 'off',
      [x, 1.65, 0],
      1.4,
      '#1d4ed8'
    );
  }
  ctx.addLabel('Complete 862,464-Parameter Transformer', 'Every piece live & verified in your browser', [0, 2.55, 0], 3.1, '#0f766e');
}
