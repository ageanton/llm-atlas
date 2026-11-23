import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { SceneBeatId } from '../content/bookData';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  TokenizerData,
} from '../engine/types';
import { hasResidualAblation, isFinalLnDisabled } from '../state/bookState';
import {
  createTensorTexture,
  divergingColor,
  formatNum,
  formatToken,
  formatTokenExplicit,
  vectorL2Norm,
} from '../utils/math';
import {
  computeRmsNorm,
  findCosineNeighbors,
  type BpeStepSnapshot,
  type PcaCloudData,
} from '../utils/pass2Math';

export interface Pass2StageContext {
  stageRoot: THREE.Group;
  interactives: THREE.Object3D[];
  trackGeo: <T extends THREE.BufferGeometry>(g: T) => T;
  trackMat: <T extends THREE.Material>(m: T) => T;
  trackTex: <T extends THREE.Texture>(t: T) => T;
  addLabel: (
    primary: string,
    secondary: string | undefined,
    pos: [number, number, number],
    scale?: number,
    accent?: string,
    muted?: boolean
  ) => void;
}

export interface Pass2Snapshot {
  beat: SceneBeatId;
  layer: number;
  queryIdx: number;
  ablation: AblationConfig;
  traceRes: EngineTraceResult;
  baselineRes: EngineTraceResult;
  tokenizer: TokenizerData;
  wteWeights: Float32Array | null;
  pcaCloud: PcaCloudData | null;
  bpeSteps: BpeStepSnapshot[];
  bpeStepIdx: number;
  ropeSlotA: number;
  ropeSlotB: number;
}

interface ClickTag {
  kind: 'token';
  index: number;
}

function createWteWallTexture(
  wte: Float32Array,
  dModel: number,
  highlightRow: number
): THREE.CanvasTexture {
  const rows = Math.floor(wte.length / dModel);
  const h = 256;
  const w = dModel;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;

  let maxAbs = 1e-5;
  for (let i = 0; i < wte.length; i++) {
    const a = Math.abs(wte[i]);
    if (a > maxAbs) maxAbs = a;
  }

  for (let py = 0; py < h; py++) {
    const r = Math.min(rows - 1, Math.floor((py / h) * rows));
    const isHit = Math.abs(r - highlightRow) <= 1;
    const base = r * dModel;
    for (let c = 0; c < w; c++) {
      ctx.fillStyle = isHit
        ? '#1d4ed8'
        : divergingColor((wte[base + c] ?? 0) / maxAbs);
      ctx.fillRect(c, py, 1, 1);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  return tex;
}

function draw128BarStrip(
  ctx: Pass2StageContext,
  vec: number[],
  zPos: number,
  maxAbs: number,
  scaleH: number,
  stripW = 7.8
): void {
  const D = vec.length;
  const step = stripW / D;
  const barGeo = ctx.trackGeo(new THREE.BoxGeometry(step * 0.78, 1, 0.22));
  const rail = new THREE.Mesh(
    ctx.trackGeo(new THREE.BoxGeometry(stripW + 0.2, 0.04, 0.28)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
  );
  rail.position.set(0, 0, zPos);
  ctx.stageRoot.add(rail);

  for (let d = 0; d < D; d++) {
    const v = vec[d];
    const norm = v / maxAbs;
    const h = Math.max(0.04, Math.abs(norm) * scaleH);
    const bar = new THREE.Mesh(
      barGeo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: divergingColor(norm),
          roughness: 0.32,
        })
      )
    );
    bar.scale.y = h;
    bar.position.set(-stripW / 2 + (d + 0.5) * step, v >= 0 ? h / 2 : -h / 2, zPos);
    ctx.stageRoot.add(bar);
  }
}

export function buildChapter123Beat(
  ctx: Pass2StageContext,
  snap: Pass2Snapshot
): boolean {
  switch (snap.beat) {
    case 'ch1-1-extremes':
      buildCh1Extremes(ctx, snap);
      return true;
    case 'ch1-2-bpe-merges':
      buildCh1BpeMerges(ctx, snap);
      return true;
    case 'ch1-3-spaces-blindspot':
      buildCh1Spaces(ctx, snap);
      return true;
    case 'ch1-4-token-ids':
      buildCh1TokenIds(ctx, snap);
      return true;
    case 'ch1-5-context-window':
      buildCh1ContextWindow(ctx, snap);
      return true;
    case 'ch2-1-wte-lookup':
      buildCh2WteLookup(ctx, snap);
      return true;
    case 'ch2-2-pca-cloud':
      buildCh2PcaCloud(ctx, snap);
      return true;
    case 'ch2-3-order-blind':
      buildCh2OrderBlind(ctx);
      return true;
    case 'ch2-4-wpe-add':
      buildCh2WpeAdd(ctx, snap);
      return true;
    case 'ch2-5-rope':
      buildCh2RopeWheel(ctx, snap);
      return true;
    case 'ch3-1-residual-bus':
      buildCh3ResidualBus(ctx, snap, false);
      return true;
    case 'ch3-2-cut-highway':
      buildCh3ResidualBus(ctx, snap, true);
      return true;
    case 'ch3-3-layernorm':
      buildCh3LayerNorm(ctx, snap);
      return true;
    case 'ch3-4-rmsnorm':
      buildCh3RmsNorm(ctx, snap);
      return true;
    default:
      return false;
  }
}

// CH 1.1: Characters vs. Subwords side-by-side
function buildCh1Extremes(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const chars = snap.traceRes.prompt.slice(0, 64).split('');
  const tokens = snap.traceRes.tokens;
  const span = 8.8;

  const charStep = span / Math.max(1, chars.length);
  const charGeo = ctx.trackGeo(
    new THREE.BoxGeometry(Math.max(0.06, charStep * 0.72), 0.24, 0.34)
  );
  const charMat = ctx.trackMat(
    new THREE.MeshStandardMaterial({ color: '#94a3b8', roughness: 0.45 })
  );

  for (let i = 0; i < chars.length; i++) {
    const m = new THREE.Mesh(charGeo, charMat);
    m.position.set(-span / 2 + (i + 0.5) * charStep, -0.1, -1.35);
    ctx.stageRoot.add(m);
  }
  ctx.addLabel(
    `Character-level: ${chars.length} tiny steps`,
    'Short dictionary, but very long sequence',
    [0, 1.05, -1.35],
    2.9,
    '#64748b'
  );

  const tokStep = span / Math.max(1, tokens.length);
  const tokGeo = ctx.trackGeo(
    new RoundedBoxGeometry(Math.min(0.68, tokStep * 0.8), 0.42, 0.66, 2, 0.05)
  );
  for (let i = 0; i < tokens.length; i++) {
    const isSel = i === snap.queryIdx;
    const m = new THREE.Mesh(
      tokGeo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isSel ? '#d97706' : '#1d4ed8',
          roughness: 0.3,
        })
      )
    );
    m.position.set(-span / 2 + (i + 0.5) * tokStep, 0.08, 1.1);
    m.castShadow = true;
    m.userData = { kind: 'token', index: i } satisfies ClickTag;
    ctx.stageRoot.add(m);
    ctx.interactives.push(m);
  }
  ctx.addLabel(
    `BPE Subwords: ${tokens.length} tokens`,
    `${(chars.length / Math.max(1, tokens.length)).toFixed(2)}× fewer steps to attend over`,
    [0, 1.25, 1.1],
    2.9,
    '#1d4ed8'
  );
}

// CH 1.2: Live BPE Merge Step Animator
function buildCh1BpeMerges(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const steps = snap.bpeSteps;
  const cur = steps[Math.min(snap.bpeStepIdx, Math.max(0, steps.length - 1))];
  if (!cur) return;

  const pieces = cur.pieces.slice(0, 64);
  const totalChars = pieces.reduce((acc, p) => acc + Math.max(1, p.length), 0);
  const span = 9.0;
  let cursor = -span / 2;

  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    const width = (Math.max(1, piece.length) / Math.max(1, totalChars)) * span;
    const isFresh = cur.mergedResult !== null && piece === cur.mergedResult;
    const mesh = new THREE.Mesh(
      ctx.trackGeo(
        new RoundedBoxGeometry(Math.max(0.06, width * 0.86), isFresh ? 0.66 : 0.36, 0.62, 2, 0.04)
      ),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isFresh ? '#d97706' : piece.length > 1 ? '#2563eb' : '#cbd5e1',
          roughness: 0.3,
        })
      )
    );
    mesh.position.set(cursor + width / 2, isFresh ? 0.33 : 0.18, 0);
    mesh.castShadow = true;
    ctx.stageRoot.add(mesh);
    cursor += width;
  }

  const subtitle = cur.mergedPair
    ? `Rule #${cur.mergeRank}: "${formatTokenExplicit(cur.mergedPair[0])}" + "${formatTokenExplicit(cur.mergedPair[1])}" → "${formatTokenExplicit(cur.mergedResult ?? '')}"`
    : 'Step 0: Raw individual characters (no merges yet)';
  ctx.addLabel(
    `Merge Step ${cur.step} / ${Math.max(0, steps.length - 1)} · ${cur.pieces.length} pieces`,
    subtitle,
    [0, 1.75, 0],
    3.4,
    cur.mergedPair ? '#b45309' : '#1d4ed8'
  );
}

// CH 1.3: Leading Spaces & Strawberry Blindspot
function buildCh1Spaces(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const tokens = snap.traceRes.tokens;
  const totalChars = tokens.reduce((acc, t) => acc + Math.max(1, t.text.length), 0);
  const span = 9.0;
  let cursor = -span / 2;

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const width = (Math.max(1, tok.text.length) / Math.max(1, totalChars)) * span;
    const multi = tok.text.length >= 3;
    const isSel = i === snap.queryIdx;
    const h = 0.32 + Math.min(1.1, tok.text.length * 0.18);

    const mesh = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(Math.max(0.1, width * 0.88), h, 0.68, 2, 0.05)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isSel ? '#1d4ed8' : multi ? '#d97706' : '#64748b',
          roughness: 0.3,
        })
      )
    );
    mesh.position.set(cursor + width / 2, h / 2, 0);
    mesh.castShadow = true;
    mesh.userData = { kind: 'token', index: i } satisfies ClickTag;
    ctx.stageRoot.add(mesh);
    ctx.interactives.push(mesh);

    if (isSel || tokens.length <= 12 || i % 2 === 0) {
      ctx.addLabel(
        formatTokenExplicit(tok.text),
        `${tok.text.length} ch`,
        [cursor + width / 2, h + 0.48, 0],
        0.92,
        isSel ? '#1d4ed8' : multi ? '#b45309' : '#475569'
      );
    }
    cursor += width;
  }

  ctx.addLabel(
    'Amber blocks hide multiple letters inside 1 atomic ID',
    'Leading dots (·) show spaces glued onto tokens',
    [0, 2.45, -0.4],
    3.2,
    '#b45309'
  );
}

// CH 1.4: Arbitrary Integer IDs
function buildCh1TokenIds(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const tokens = snap.traceRes.tokens;
  const T = tokens.length;
  const span = Math.min(9.0, Math.max(5.0, T * 0.65));
  const step = span / Math.max(1, T);
  const geo = ctx.trackGeo(new RoundedBoxGeometry(step * 0.72, 1, 0.56, 2, 0.04));

  for (let i = 0; i < T; i++) {
    const tok = tokens[i];
    const h = Math.max(0.14, (tok.id / 511) * 2.5);
    const isSel = i === snap.queryIdx;
    const mesh = new THREE.Mesh(
      geo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: isSel ? '#1d4ed8' : '#64748b',
          roughness: 0.32,
        })
      )
    );
    const x = -span / 2 + (i + 0.5) * step;
    mesh.scale.y = h;
    mesh.position.set(x, h / 2, 0);
    mesh.castShadow = true;
    mesh.userData = { kind: 'token', index: i } satisfies ClickTag;
    ctx.stageRoot.add(mesh);
    ctx.interactives.push(mesh);

    if (isSel || T <= 10 || i === T - 1) {
      ctx.addLabel(
        `#${tok.id}`,
        formatToken(tok.text),
        [x, h + 0.48, 0],
        0.95,
        isSel ? '#1d4ed8' : '#475569'
      );
    }
  }

  ctx.addLabel(
    'Raw Vocabulary IDs (0 – 511)',
    'Jagged & arbitrary — nearby IDs have no shared meaning',
    [0, 3.05, -0.3],
    3.1,
    '#1d4ed8'
  );
}

// CH 1.5: Physical 64-Slot Context Desk
function buildCh1ContextWindow(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const filled = snap.traceRes.tokens.length;
  const cols = 16;
  const rows = 4;
  const spacing = 0.52;
  const startX = -((cols - 1) * spacing) / 2;
  const startZ = -((rows - 1) * spacing) / 2;

  const tray = new THREE.Mesh(
    ctx.trackGeo(new RoundedBoxGeometry(cols * spacing + 0.45, 0.14, rows * spacing + 0.45, 2, 0.06)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#e2e8f0', roughness: 0.7 }))
  );
  tray.position.set(0, -0.1, 0);
  tray.receiveShadow = true;
  ctx.stageRoot.add(tray);

  const slotGeo = ctx.trackGeo(new RoundedBoxGeometry(0.4, 0.32, 0.4, 2, 0.04));
  const emptyMat = ctx.trackMat(
    new THREE.MeshStandardMaterial({
      color: '#ffffff',
      roughness: 0.4,
      transparent: true,
      opacity: 0.55,
    })
  );

  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const isFilled = i < filled;
    const isSel = i === snap.queryIdx;
    const mesh = new THREE.Mesh(
      slotGeo,
      isFilled
        ? ctx.trackMat(
            new THREE.MeshStandardMaterial({
              color: isSel ? '#d97706' : '#1d4ed8',
              roughness: 0.3,
            })
          )
        : emptyMat
    );
    mesh.scale.y = isFilled ? 1 : 0.25;
    mesh.position.set(
      startX + c * spacing,
      isFilled ? 0.16 : 0.04,
      startZ + r * spacing
    );
    if (isFilled) {
      mesh.castShadow = true;
      mesh.userData = { kind: 'token', index: i } satisfies ClickTag;
      ctx.interactives.push(mesh);
    }
    ctx.stageRoot.add(mesh);
  }

  const rawTotal =
    snap.bpeSteps.length > 0
      ? snap.bpeSteps[snap.bpeSteps.length - 1].pieces.length
      : filled;
  const overflow = Math.max(0, rawTotal - 64);

  if (overflow > 0) {
    const dropMat = ctx.trackMat(
      new THREE.MeshStandardMaterial({ color: '#dc2626', roughness: 0.4 })
    );
    for (let j = 0; j < Math.min(overflow, 8); j++) {
      const m = new THREE.Mesh(slotGeo, dropMat);
      m.position.set(4.9 + (j % 2) * 0.46, -0.45 - j * 0.06, -0.4 + Math.floor(j / 2) * 0.48);
      m.rotation.set(0.3 * j, 0.4, 0.25 * j);
      ctx.stageRoot.add(m);
    }
  }

  ctx.addLabel(
    `Context Window: ${filled} / 64 slots used`,
    overflow > 0
      ? `${overflow} overflow tokens fell off the desk!`
      : `${64 - filled} open slots remaining on desk`,
    [0, 1.65, -1.2],
    3.1,
    overflow > 0 ? '#dc2626' : '#1d4ed8'
  );
}

// CH 2.1: wte [512, 128] Coordinate Wall + Selected 128-dim Vector
function buildCh2WteLookup(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const tok = snap.traceRes.tokens[snap.queryIdx];
  const vec = snap.traceRes.trace.tokEmb[snap.queryIdx];
  const dModel = vec.length;

  // Left: Tall 512x128 wte matrix wall
  if (snap.wteWeights) {
    const tex = ctx.trackTex(createWteWallTexture(snap.wteWeights, dModel, tok.id));
    const wall = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(2.4, 3.2, 0.2, 2, 0.04)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ map: tex, roughness: 0.35 }))
    );
    wall.position.set(-3.4, 0.45, 0);
    wall.castShadow = true;
    ctx.stageRoot.add(wall);
  }
  ctx.addLabel('wte [512 × 128]', `Row #${tok.id} highlighted`, [-3.4, 2.45, 0], 1.9);

  // Right: Extracted 128-dim bar vector
  let maxAbs = 1e-5;
  for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));

  const stripW = 4.6;
  const step = stripW / dModel;
  const barGeo = ctx.trackGeo(new THREE.BoxGeometry(step * 0.78, 1, 0.26));
  const originX = -0.7;

  const rail = new THREE.Mesh(
    ctx.trackGeo(new THREE.BoxGeometry(stripW + 0.16, 0.05, 0.32)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
  );
  rail.position.set(originX + stripW / 2, 0.25, 0);
  ctx.stageRoot.add(rail);

  for (let d = 0; d < dModel; d++) {
    const norm = vec[d] / maxAbs;
    const h = Math.max(0.04, Math.abs(norm) * 1.35);
    const bar = new THREE.Mesh(
      barGeo,
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: divergingColor(norm),
          roughness: 0.3,
        })
      )
    );
    bar.scale.y = h;
    bar.position.set(
      originX + (d + 0.5) * step,
      0.25 + (vec[d] >= 0 ? h / 2 : -h / 2),
      0
    );
    ctx.stageRoot.add(bar);
  }

  ctx.addLabel(
    `“${formatToken(tok.text)}” → 128 floats`,
    `||wte[#${tok.id}]|| = ${formatNum(vectorL2Norm(vec), 3)}`,
    [originX + stripW / 2, 2.15, 0],
    2.4,
    '#0f766e'
  );
}

// CH 2.2: Real 3D PCA Embedding Cloud (512 real tokens + top-5 128D cosine neighbors)
function buildCh2PcaCloud(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const cloud = snap.pcaCloud;
  if (!cloud || !snap.wteWeights) {
    ctx.addLabel('Computing 3D PCA...', 'From real wte.weight [512 × 128]', [0, 0.5, 0], 2.5);
    return;
  }

  const tok = snap.traceRes.tokens[snap.queryIdx];
  const dModel = snap.traceRes.trace.tokEmb[0].length;
  const neighbors = findCosineNeighbors(
    snap.wteWeights,
    snap.tokenizer.vocab,
    dModel,
    tok.id,
    5
  );
  const neighborMap = new Map(neighbors.map((n) => [n.id, n]));
  const promptIdSet = new Set(snap.traceRes.tokens.map((t) => t.id));

  // Subtle 3D reference axes
  const axisMat = ctx.trackMat(
    new THREE.LineBasicMaterial({ color: '#cbd5e1', transparent: true, opacity: 0.7 })
  );
  const makeAxis = (a: [number, number, number], b: [number, number, number]) => {
    const g = ctx.trackGeo(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...a),
        new THREE.Vector3(...b),
      ])
    );
    ctx.stageRoot.add(new THREE.Line(g, axisMat));
  };
  makeAxis([-3.5, 0, 0], [3.5, 0, 0]);
  makeAxis([0, -2.5, 0], [0, 2.5, 0]);
  makeAxis([0, 0, -3.5], [0, 0, 3.5]);

  // Instanced background constellation of all 512 real vocabulary tokens
  const pts = cloud.points;
  const bgGeo = ctx.trackGeo(new THREE.SphereGeometry(0.065, 10, 10));
  const bgMat = ctx.trackMat(
    new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.05 })
  );
  const inst = new THREE.InstancedMesh(bgGeo, bgMat, pts.length);
  const dummy = new THREE.Object3D();
  const cMuted = new THREE.Color('#94a3b8');
  const cPrompt = new THREE.Color('#60a5fa');
  const cTarget = new THREE.Color('#1d4ed8');
  const cNeighbor = new THREE.Color('#d97706');

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const isTarget = p.id === tok.id;
    const isNeighbor = neighborMap.has(p.id);
    const isPrompt = promptIdSet.has(p.id);
    dummy.position.set(p.x, p.y * 0.75 + 0.3, p.z);
    dummy.scale.setScalar(isTarget ? 2.8 : isNeighbor ? 2.1 : isPrompt ? 1.45 : 0.9);
    dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
    inst.setColorAt(
      i,
      isTarget ? cTarget : isNeighbor ? cNeighbor : isPrompt ? cPrompt : cMuted
    );
  }
  inst.instanceMatrix.needsUpdate = true;
  if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
  ctx.stageRoot.add(inst);

  // Connect target token to its top-5 128D cosine similarity neighbors
  const tp = pts[tok.id];
  if (tp) {
    const tPos = new THREE.Vector3(tp.x, tp.y * 0.75 + 0.3, tp.z);
    ctx.addLabel(
      `“${formatToken(tp.text)}”`,
      'Selected token',
      [tPos.x, tPos.y + 0.42, tPos.z],
      1.35,
      '#1d4ed8'
    );

    const lineMat = ctx.trackMat(
      new THREE.LineBasicMaterial({ color: '#d97706', transparent: true, opacity: 0.85 })
    );
    for (const nb of neighbors) {
      const np = pts[nb.id];
      if (!np) continue;
      const nPos = new THREE.Vector3(np.x, np.y * 0.75 + 0.3, np.z);
      const lGeo = ctx.trackGeo(new THREE.BufferGeometry().setFromPoints([tPos, nPos]));
      ctx.stageRoot.add(new THREE.Line(lGeo, lineMat));
      ctx.addLabel(
        `“${formatToken(np.text)}”`,
        `cos ${nb.cosine.toFixed(2)}`,
        [nPos.x, nPos.y + 0.34, nPos.z],
        1.05,
        '#b45309'
      );
    }
  }
}

// CH 2.3: Order-Blind Bag of Vectors
function buildCh2OrderBlind(ctx: Pass2StageContext): void {
  const rowA = ['The', 'dog', 'chased', 'the', 'ball'];
  const rowB = ['The', 'ball', 'chased', 'the', 'dog'];
  const colors: Record<string, string> = {
    The: '#64748b',
    dog: '#d97706',
    chased: '#3b82f6',
    the: '#64748b',
    ball: '#0f766e',
  };
  const geo = ctx.trackGeo(new RoundedBoxGeometry(1.1, 0.56, 0.65, 2, 0.06));

  const placeRow = (words: string[], z: number, title: string) => {
    words.forEach((w, i) => {
      const x = -3.0 + i * 1.5;
      const mesh = new THREE.Mesh(
        geo,
        ctx.trackMat(
          new THREE.MeshStandardMaterial({
            color: colors[w] ?? '#64748b',
            roughness: 0.3,
          })
        )
      );
      mesh.position.set(x, 0.15, z);
      mesh.castShadow = true;
      ctx.stageRoot.add(mesh);
      ctx.addLabel(w, undefined, [x, 0.82, z], 0.95, colors[w] ?? '#334155');
    });
    ctx.addLabel(title, undefined, [0, 1.55, z], 2.2, '#334155');
  };

  placeRow(rowA, -1.35, 'Sentence A: identical 5 vectors');
  placeRow(rowB, 1.35, 'Sentence B: exact same set, swapped order!');
}

// CH 2.4: wte + wpe = h0 Tensor Slabs
function buildCh2WpeAdd(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const { tokEmb, posEmb } = snap.traceRes.trace;
  const T = tokEmb.length;
  const h0 = tokEmb.map((row, r) => row.map((v, c) => v + posEmb[r][c]));

  const addPlate = (mat: number[][], x: number, frameHex: string) => {
    const tex = ctx.trackTex(createTensorTexture(mat, 'diverging', false));
    const side = ctx.trackMat(
      new THREE.MeshStandardMaterial({ color: frameHex, roughness: 0.45 })
    );
    const face = ctx.trackMat(
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.3 })
    );
    const mesh = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(2.55, 2.0, 0.22, 2, 0.04)),
      [side, side, face, side, face, face]
    );
    mesh.position.set(x, 0.35, 0);
    mesh.castShadow = true;
    ctx.stageRoot.add(mesh);
  };

  addPlate(tokEmb, -3.3, '#cbd5e1');
  ctx.addLabel('Token Meaning (wte)', `[${T} × 128]`, [-3.3, 1.85, 0], 1.95, '#1d4ed8');

  const plus = new THREE.Mesh(
    ctx.trackGeo(new THREE.SphereGeometry(0.2, 20, 20)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#c89b54', metalness: 0.65, roughness: 0.25 }))
  );
  plus.position.set(-1.65, 0.35, 0);
  ctx.stageRoot.add(plus);
  ctx.addLabel('+', undefined, [-1.65, 0.95, 0], 0.7, '#b45309');

  addPlate(posEmb, 0, '#fed7aa');
  ctx.addLabel('Slot Position (wpe)', `[${T} × 128]`, [0, 1.85, 0], 1.95, '#b45309');

  const eq = plus.clone();
  eq.position.set(1.65, 0.35, 0);
  ctx.stageRoot.add(eq);
  ctx.addLabel('=', undefined, [1.65, 0.95, 0], 0.7, '#0f766e');

  addPlate(h0, 3.3, '#93c5fd');
  ctx.addLabel('Starting Stream h₀', `wte + wpe [${T} × 128]`, [3.3, 1.85, 0], 2.05, '#0f766e');
}

// CH 2.5: Interactive 2D RoPE Rotation Clock Wheel
function buildCh2RopeWheel(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const radius = 2.4;
  const angleStep = Math.PI / 10; // 18 deg per slot
  const angA = snap.ropeSlotA * angleStep;
  const angB = snap.ropeSlotB * angleStep;
  const deltaSlots = snap.ropeSlotB - snap.ropeSlotA;
  const deltaDeg = Math.round((deltaSlots * 180) / 10);

  // Clock ring
  const ring = new THREE.Mesh(
    ctx.trackGeo(new THREE.TorusGeometry(radius, 0.06, 16, 64)),
    ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.4 }))
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.05;
  ctx.stageRoot.add(ring);

  // Tick marks
  const tickGeo = ctx.trackGeo(new THREE.BoxGeometry(0.05, 0.06, 0.22));
  const tickMat = ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#94a3b8' }));
  for (let i = 0; i < 20; i++) {
    const a = i * angleStep;
    const tick = new THREE.Mesh(tickGeo, tickMat);
    tick.position.set(Math.cos(a) * radius, 0.06, Math.sin(a) * radius);
    tick.rotation.y = -a + Math.PI / 2;
    ctx.stageRoot.add(tick);
  }

  // Draw rotated Q and K arrows
  const drawArrow = (ang: number, colorHex: string, label: string, sub: string) => {
    const dir = new THREE.Vector3(Math.cos(ang), 0, Math.sin(ang));
    const shaft = new THREE.Mesh(
      ctx.trackGeo(new THREE.CylinderGeometry(0.06, 0.06, radius * 0.88, 16)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.25 }))
    );
    shaft.rotation.z = -Math.PI / 2;
    shaft.rotation.y = -ang;
    shaft.position.set(dir.x * radius * 0.44, 0.2, dir.z * radius * 0.44);
    ctx.stageRoot.add(shaft);

    const tip = new THREE.Mesh(
      ctx.trackGeo(new THREE.ConeGeometry(0.16, 0.36, 16)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.25 }))
    );
    tip.rotation.z = -Math.PI / 2;
    tip.rotation.y = -ang;
    tip.position.set(dir.x * radius * 0.92, 0.2, dir.z * radius * 0.92);
    ctx.stageRoot.add(tip);

    ctx.addLabel(label, sub, [dir.x * (radius + 0.65), 0.72, dir.z * (radius + 0.65)], 1.45, colorHex);
  };

  drawArrow(angA, '#1d4ed8', `Slot A = ${snap.ropeSlotA}`, `${snap.ropeSlotA * 18}°`);
  drawArrow(angB, '#d97706', `Slot B = ${snap.ropeSlotB}`, `${snap.ropeSlotB * 18}°`);

  // Arc connecting angA to angB
  const arcPts: THREE.Vector3[] = [];
  const steps = 24;
  for (let i = 0; i <= steps; i++) {
    const a = angA + ((angB - angA) * i) / steps;
    arcPts.push(new THREE.Vector3(Math.cos(a) * 1.15, 0.2, Math.sin(a) * 1.15));
  }
  if (arcPts.length >= 2) {
    const arcCurve = new THREE.CatmullRomCurve3(arcPts);
    const arcTube = new THREE.Mesh(
      ctx.trackGeo(new THREE.TubeGeometry(arcCurve, 24, 0.04, 8, false)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#0f766e', roughness: 0.3 }))
    );
    ctx.stageRoot.add(arcTube);
  }

  ctx.addLabel(
    `Relative Distance Δ = ${deltaSlots} slots`,
    `Angle between arrows = ${deltaDeg}° · cos(Δθ) = ${Math.cos(angB - angA).toFixed(2)}`,
    [0, 1.85, 0],
    2.9,
    '#0f766e'
  );
}

// CH 3.1 & 3.2: Residual Highway + Break-It Counterfactual Skyline
function buildCh3ResidualBus(
  ctx: Pass2StageContext,
  snap: Pass2Snapshot,
  showBreakIt: boolean
): void {
  const cut = hasResidualAblation(snap.ablation);
  const { trace } = snap.traceRes;
  const last = snap.traceRes.tokens.length - 1;
  const blockX = [-3.3, -1.5, 0.3, 2.1];

  // Unbroken vs Fractured Residual Stream Highway
  if (!cut) {
    const bus = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(7.6, 0.22, 0.66, 2, 0.05)),
      ctx.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: '#1d4ed8',
          emissive: '#1d4ed8',
          emissiveIntensity: 0.25,
          roughness: 0.22,
        })
      )
    );
    bus.position.set(-0.6, -0.55, 0);
    ctx.stageRoot.add(bus);
  } else {
    // Broken dashed segments when skip connections are cut
    const brokenMat = ctx.trackMat(
      new THREE.MeshStandardMaterial({
        color: '#dc2626',
        roughness: 0.5,
        transparent: true,
        opacity: 0.65,
      })
    );
    for (let s = 0; s < 5; s++) {
      const seg = new THREE.Mesh(
        ctx.trackGeo(new RoundedBoxGeometry(0.75, 0.14, 0.52, 2, 0.04)),
        brokenMat
      );
      seg.position.set(-3.8 + s * 1.65, -0.55, 0);
      ctx.stageRoot.add(seg);
    }
  }

  // 4 Blocks writing additive deltas onto the highway
  for (let l = 0; l < 4; l++) {
    const x = blockX[l];
    const lt = trace.layers[l];
    const attnMag = !lt.skipped ? vectorL2Norm(lt.attn.output[last]) : 0.2;
    const mlpMag = !lt.skipped ? vectorL2Norm(lt.mlp.output[last]) : 0.2;

    const frame = new THREE.Mesh(
      ctx.trackGeo(new RoundedBoxGeometry(1.42, 1.7, 1.3, 2, 0.08)),
      ctx.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: cut ? '#ffedd5' : '#ffffff',
          roughness: 0.25,
          transmission: 0.65,
          transparent: true,
          opacity: 0.82,
        })
      )
    );
    frame.position.set(x, 0.72, 0);
    ctx.stageRoot.add(frame);

    const pAttn = new THREE.Mesh(
      ctx.trackGeo(new THREE.BoxGeometry(0.32, Math.min(1.3, 0.25 + attnMag * 0.1), 0.36)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#2563eb', roughness: 0.3 }))
    );
    pAttn.position.set(x - 0.28, 0.55, 0);
    ctx.stageRoot.add(pAttn);

    const pMlp = new THREE.Mesh(
      ctx.trackGeo(new THREE.BoxGeometry(0.32, Math.min(1.3, 0.25 + mlpMag * 0.1), 0.36)),
      ctx.trackMat(new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.3 }))
    );
    pMlp.position.set(x + 0.28, 0.55, 0);
    ctx.stageRoot.add(pMlp);

    const plus = new THREE.Mesh(
      ctx.trackGeo(new THREE.SphereGeometry(0.14, 16, 16)),
      ctx.trackMat(
        new THREE.MeshStandardMaterial({
          color: cut ? '#94a3b8' : '#c89b54',
          metalness: 0.6,
          roughness: 0.3,
        })
      )
    );
    plus.position.set(x, -0.42, 0);
    ctx.stageRoot.add(plus);
  }

  ctx.addLabel(
    cut ? 'Residual Highway CUT!' : 'Unbroken Residual Highway',
    cut ? 'Skip connections disabled — signal shattered' : 'Blocks add (+) ΔAttn and ΔMLP without erasing',
    [-0.6, 2.05, 0],
    2.9,
    cut ? '#dc2626' : '#1d4ed8'
  );

  // Live top-4 probability skyline on the right (with wireframe baseline ghosts when cut)
  if (showBreakIt) {
    const baseProbs = snap.baselineRes.trace.finalProbs;
    const curProbs = trace.finalProbs;
    const top4 = baseProbs
      .map((bp, id) => ({ id, bp, cp: curProbs[id] }))
      .sort((a, b) => b.bp - a.bp)
      .slice(0, 4);

    top4.forEach((item, rank) => {
      const x = 3.55 + rank * 0.56;
      const curH = Math.max(0.1, item.cp * 2.6);
      const bar = new THREE.Mesh(
        ctx.trackGeo(new RoundedBoxGeometry(0.34, curH, 0.34, 2, 0.04)),
        ctx.trackMat(
          new THREE.MeshStandardMaterial({
            color: cut ? '#d97706' : '#1d4ed8',
            roughness: 0.3,
          })
        )
      );
      bar.position.set(x, -0.55 + curH / 2, 0);
      ctx.stageRoot.add(bar);

      if (cut) {
        const gh = Math.max(0.1, item.bp * 2.6);
        const ghost = new THREE.Mesh(
          ctx.trackGeo(new THREE.BoxGeometry(0.4, gh, 0.4)),
          ctx.trackMat(
            new THREE.MeshBasicMaterial({
              color: '#1d4ed8',
              wireframe: true,
              transparent: true,
              opacity: 0.5,
            })
          )
        );
        ghost.position.set(x, -0.55 + gh / 2, 0);
        ctx.stageRoot.add(ghost);
      }

      if (rank < 2) {
        ctx.addLabel(
          `“${formatToken(snap.tokenizer.vocab[item.id] ?? '')}”`,
          `${(item.cp * 100).toFixed(0)}%`,
          [x, -0.2 + Math.max(curH, item.bp * 2.6) + 0.4, 0],
          0.95,
          cut ? '#b45309' : '#1d4ed8'
        );
      }
    });
  }
}

// CH 3.3: Pre-LayerNorm Before vs. After 128-bar Strips
function buildCh3LayerNorm(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const lt = snap.traceRes.trace.layers[snap.layer];
  if (lt.skipped) {
    ctx.addLabel(`Block ${snap.layer} disabled`, undefined, [0, 0.6, 0], 2.4, '#b45309');
    return;
  }
  const active: ActiveLayerTrace = lt;
  const rawVec = active.residualBeforeAttn[snap.queryIdx];
  const normVec = active.ln1Out[snap.queryIdx];

  let maxAbs = 1e-5;
  for (const v of rawVec) maxAbs = Math.max(maxAbs, Math.abs(v));
  for (const v of normVec) maxAbs = Math.max(maxAbs, Math.abs(v));

  draw128BarStrip(ctx, rawVec, -1.45, maxAbs, 1.55);
  ctx.addLabel(
    `Raw Residual h_in (Block ${snap.layer})`,
    `Spiky unnormalized 128 dims · ||h||=${formatNum(vectorL2Norm(rawVec), 2)}`,
    [0, 1.85, -1.45],
    2.9,
    '#b45309'
  );

  draw128BarStrip(ctx, normVec, 1.45, maxAbs, 1.55);
  ctx.addLabel(
    `After Pre-LayerNorm LN₁`,
    `Balanced mean≈0, std≈1 (+ γ, β) · ||ln||=${formatNum(vectorL2Norm(normVec), 2)}`,
    [0, 1.85, 1.45],
    2.9,
    '#0f766e'
  );
}

// CH 3.4: LayerNorm vs. Modern RMSNorm + Final LN_f Break-It
function buildCh3RmsNorm(ctx: Pass2StageContext, snap: Pass2Snapshot): void {
  const lnFOff = isFinalLnDisabled(snap.ablation);
  const lt = snap.traceRes.trace.layers[3];
  const rawLast = lt.skipped
    ? lt.residualOut[snap.queryIdx]
    : lt.residualAfterMlp[snap.queryIdx];
  const lnFVec = snap.traceRes.trace.lnFOut[snap.queryIdx];
  const rmsVec = computeRmsNorm(rawLast);

  let maxAbs = 1e-5;
  for (const v of lnFVec) maxAbs = Math.max(maxAbs, Math.abs(v));
  for (const v of rmsVec) maxAbs = Math.max(maxAbs, Math.abs(v));

  draw128BarStrip(ctx, lnFVec, -1.45, maxAbs, 1.5);
  ctx.addLabel(
    lnFOff ? 'Final LN_f DISABLED (Raw Drift!)' : 'This Model: Final LayerNorm (ln_f)',
    lnFOff ? 'Unnormalized residual hits unembedding!' : 'Mean-centered + variance scaled (+ γ, β)',
    [0, 1.85, -1.45],
    3.0,
    lnFOff ? '#dc2626' : '#1d4ed8'
  );

  draw128BarStrip(ctx, rmsVec, 1.45, maxAbs, 1.5);
  ctx.addLabel(
    'Modern Frontier: RMSNorm',
    'Skips mean subtraction — nearly identical stability, faster GPU kernels',
    [0, 1.85, 1.45],
    3.1,
    '#0f766e'
  );
}
