import * as THREE from 'three';
import type { SceneBeatId } from '../content/bookData';
import type { EngineTraceResult, TokenizerData } from '../engine/types';
import type { AblationConfig } from '../state/bookState';
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

export interface Pass2BuildArgs {
  group: THREE.Group;
  beat: SceneBeatId;
  layer: number;
  head: number;
  queryIdx: number;
  keyIdx: number;
  showPreSoftmax: boolean;
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
  makeLabelSprite: (text: string, color?: string, bg?: string) => THREE.Sprite;
  defaultMat: THREE.Material;
}

const COLOR_ACCENT = 0x2563eb;
const COLOR_EMERALD = 0x059669;
const COLOR_AMBER = 0xd97706;
const COLOR_ROSE = 0xd946ef;
const COLOR_GRAY = 0x94a3b8;
const COLOR_DANGER = 0xdc2626;

function makeTileMesh(
  w: number,
  h: number,
  d: number,
  color: number,
  wire = false
): THREE.Mesh {
  const geom = new THREE.BoxGeometry(w, h, d);
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.35,
    metalness: 0.05,
    wireframe: wire,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function buildChapter123Beat(args: Pass2BuildArgs): boolean {
  const { beat } = args;

  switch (beat) {
    case 'ch1-1-extremes':
      buildCh1Extremes(args);
      return true;
    case 'ch1-2-bpe-merges':
      buildCh1BpeMerges(args);
      return true;
    case 'ch1-3-spaces-blindspot':
      buildCh1SpacesBlindspot(args);
      return true;
    case 'ch1-4-token-ids':
      buildCh1TokenIds(args);
      return true;
    case 'ch1-5-context-window':
      buildCh1ContextWindow(args);
      return true;
    case 'ch2-1-wte-lookup':
      buildCh2WteLookup(args);
      return true;
    case 'ch2-2-pca-cloud':
      buildCh2PcaCloud(args);
      return true;
    case 'ch2-3-order-blind':
      buildCh2OrderBlind(args);
      return true;
    case 'ch2-4-wpe-add':
      buildCh2WpeAdd(args);
      return true;
    case 'ch2-5-rope':
      buildCh2Rope(args);
      return true;
    case 'ch3-1-residual-bus':
      buildCh3ResidualBus(args);
      return true;
    case 'ch3-2-cut-highway':
      buildCh3CutHighway(args);
      return true;
    case 'ch3-3-layernorm':
      buildCh3LayerNorm(args);
      return true;
    case 'ch3-4-rmsnorm':
      buildCh3RmsNorm(args);
      return true;
    default:
      return false;
  }
}

/* -------------------------------------------------------------
 * CHAPTER 1 BUILDERS
 * ----------------------------------------------------------- */

function buildCh1Extremes({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const tokens = traceRes.tokens;
  const promptText = tokens.join('');
  const chars = Array.from(promptText).slice(0, 36);

  // Back row: character-level tiles
  const charSpacing = 0.52;
  const charStart = -((chars.length - 1) * charSpacing) / 2;
  const charLbl = makeLabelSprite(
    `Extreme 1: Pure Characters (${chars.length} steps — long sequence, high O(T²) cost)`,
    '#64748b'
  );
  charLbl.position.set(0, 1.8, -2.6);
  charLbl.scale.set(7.5, 0.75, 1);
  group.add(charLbl);

  for (let i = 0; i < chars.length; i++) {
    const x = charStart + i * charSpacing;
    const mesh = makeTileMesh(0.44, 0.44, 0.22, 0xe2e8f0);
    mesh.position.set(x, 0.6, -2.5);
    group.add(mesh);

    const lbl = makeLabelSprite(
      chars[i] === ' ' ? '␣' : chars[i],
      '#475569',
      'transparent'
    );
    lbl.position.set(x, 1.05, -2.5);
    lbl.scale.set(0.65, 0.45, 1);
    group.add(lbl);
  }

  // Front row: subword tokens
  const tokSpacing = 1.15;
  const tokStart = -((tokens.length - 1) * tokSpacing) / 2;
  const tokLbl = makeLabelSprite(
    `BPE Solution: ${tokens.length} Subwords (Balanced sequence length & finite vocabulary |V|=512)`,
    '#2563eb'
  );
  tokLbl.position.set(0, 1.8, 1.2);
  tokLbl.scale.set(8.5, 0.85, 1);
  group.add(tokLbl);

  for (let i = 0; i < tokens.length; i++) {
    const x = tokStart + i * tokSpacing;
    const mesh = makeTileMesh(1.0, 0.55, 0.35, 0xbfdbfe);
    mesh.position.set(x, 0.6, 1.2);
    group.add(mesh);

    const lbl = makeLabelSprite(
      formatToken(tokens[i]),
      '#1e3a8a',
      'transparent'
    );
    lbl.position.set(x, 1.15, 1.2);
    lbl.scale.set(1.1, 0.5, 1);
    group.add(lbl);

    const idLbl = makeLabelSprite(
      `ID ${traceRes.tokenIds[i]}`,
      '#64748b',
      'transparent'
    );
    idLbl.position.set(x, 0.05, 1.2);
    idLbl.scale.set(0.85, 0.35, 1);
    group.add(idLbl);
  }
}

function buildCh1BpeMerges({
  group,
  bpeSteps,
  bpeStepIdx,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const total = bpeSteps.length;
  const current = bpeSteps[Math.min(bpeStepIdx, total - 1)] ?? bpeSteps[0];
  const pieces = current.pieces;

  const title = makeLabelSprite(
    `BPE Merge Simulator — Step ${current.step} of ${total - 1}` +
      (current.mergedResult
        ? ` (Merged "${current.mergedPair?.[0]}" + "${current.mergedPair?.[1]}" → "${current.mergedResult}", rank #${current.mergeRank})`
        : ' (Initial Characters)'),
    '#d97706'
  );
  title.position.set(0, 2.4, 0);
  title.scale.set(9.5, 0.95, 1);
  group.add(title);

  const spacing = Math.min(1.1, 12 / Math.max(1, pieces.length));
  const startX = -((pieces.length - 1) * spacing) / 2;

  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    const isNew = current.mergedResult && piece === current.mergedResult;
    const x = startX + i * spacing;

    const mesh = makeTileMesh(
      spacing * 0.88,
      0.65,
      0.35,
      isNew ? COLOR_AMBER : 0xecfdf5
    );
    mesh.position.set(x, 0.8, 0);
    group.add(mesh);

    const lbl = makeLabelSprite(
      formatToken(piece),
      isNew ? '#ffffff' : '#065f46',
      'transparent'
    );
    lbl.position.set(x, 1.4, 0);
    lbl.scale.set(Math.min(1.2, spacing * 1.1), 0.5, 1);
    group.add(lbl);

    const lenLbl = makeLabelSprite(
      `${piece.length}B`,
      '#64748b',
      'transparent'
    );
    lenLbl.position.set(x, 0.2, 0);
    lenLbl.scale.set(0.65, 0.3, 1);
    group.add(lenLbl);
  }
}

function buildCh1SpacesBlindspot({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const tokens = traceRes.tokens;
  const count = Math.min(tokens.length, 8);
  const startX = -((count - 1) * 1.6) / 2;

  const header = makeLabelSprite(
    'Leading-Space Blindspot: " dog" vs. "dog" are distinct vocabulary IDs with distinct weights',
    '#2563eb'
  );
  header.position.set(0, 2.6, 0);
  header.scale.set(9.8, 0.9, 1);
  group.add(header);

  for (let i = 0; i < count; i++) {
    const tok = tokens[i];
    const x = startX + i * 1.6;
    const hasSpace = tok.startsWith(' ') || tok.startsWith(' ');

    const mesh = makeTileMesh(
      1.35,
      0.8,
      0.4,
      hasSpace ? 0xdbeafe : 0xfef3c7
    );
    mesh.position.set(x, 0.8, 0);
    group.add(mesh);

    const explicit = formatTokenExplicit(tok);
    const lbl = makeLabelSprite(explicit, hasSpace ? '#1e40af' : '#92400e');
    lbl.position.set(x, 1.45, 0);
    lbl.scale.set(1.4, 0.55, 1);
    group.add(lbl);

    const tag = makeLabelSprite(
      hasSpace ? 'Space Pre-baked' : 'No Leading Space',
      hasSpace ? '#2563eb' : '#d97706',
      'transparent'
    );
    tag.position.set(x, 0.15, 0);
    tag.scale.set(1.2, 0.35, 1);
    group.add(tag);
  }
}

function buildCh1TokenIds({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const { tokens, tokenIds } = traceRes;
  const count = tokens.length;
  const spacing = Math.min(1.25, 14 / Math.max(1, count));
  const startX = -((count - 1) * spacing) / 2;

  const title = makeLabelSprite(
    `Discrete Token IDs: Categorical barcode addresses into W_e [0 to 511]`,
    '#059669'
  );
  title.position.set(0, 2.6, 0);
  title.scale.set(8.5, 0.85, 1);
  group.add(title);

  for (let i = 0; i < count; i++) {
    const x = startX + i * spacing;
    const id = tokenIds[i];

    // Barcode block
    const barH = 0.4 + (id / 512) * 1.1;
    const bar = makeTileMesh(spacing * 0.85, barH, 0.35, COLOR_EMERALD);
    bar.position.set(x, barH / 2, 0);
    group.add(bar);

    const tokLbl = makeLabelSprite(
      formatToken(tokens[i]),
      '#065f46',
      'transparent'
    );
    tokLbl.position.set(x, barH + 0.4, 0);
    tokLbl.scale.set(1.1, 0.45, 1);
    group.add(tokLbl);

    const idLbl = makeLabelSprite(
      `#${id}`,
      '#ffffff',
      'rgba(6,95,70,0.85)'
    );
    idLbl.position.set(x, barH / 2, 0.22);
    idLbl.scale.set(0.75, 0.35, 1);
    group.add(idLbl);
  }
}

function buildCh1ContextWindow({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const maxSlots = 64;
  const activeCount = traceRes.tokens.length;
  const cols = 16;
  const rows = 4;
  const slotW = 0.58;
  const slotD = 0.58;

  const title = makeLabelSprite(
    `Context Window Capacity: ${activeCount} of 64 slots occupied (${64 - activeCount} free)`,
    '#2563eb'
  );
  title.position.set(0, 2.6, -1.8);
  title.scale.set(8.8, 0.85, 1);
  group.add(title);

  const startX = -((cols - 1) * slotW) / 2;
  const startZ = -((rows - 1) * slotD) / 2;

  for (let slot = 0; slot < maxSlots; slot++) {
    const c = slot % cols;
    const r = Math.floor(slot / cols);
    const x = startX + c * slotW;
    const z = startZ + r * slotD;
    const isActive = slot < activeCount;

    const mesh = makeTileMesh(
      slotW * 0.82,
      isActive ? 0.35 : 0.08,
      slotD * 0.82,
      isActive ? COLOR_ACCENT : COLOR_GRAY,
      !isActive
    );
    mesh.position.set(x, isActive ? 0.2 : 0.04, z);
    group.add(mesh);

    if (isActive) {
      const lbl = makeLabelSprite(
        formatToken(traceRes.tokens[slot]),
        '#ffffff',
        'transparent'
      );
      lbl.position.set(x, 0.55, z);
      lbl.scale.set(0.65, 0.3, 1);
      group.add(lbl);
    }
  }
}

/* -------------------------------------------------------------
 * CHAPTER 2 BUILDERS
 * ----------------------------------------------------------- */

function buildCh2WteLookup({
  group,
  traceRes,
  wteWeights,
  queryIdx,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    'Embedding Wall W_e [512 × 128]: Direct row lookup copies 128 float values into the residual stream',
    '#2563eb'
  );
  title.position.set(0, 2.8, -1.5);
  title.scale.set(9.8, 0.9, 1);
  group.add(title);

  // Background wall mesh (512x128 plate representation)
  const wallGeom = new THREE.PlaneGeometry(8.5, 4.2);
  const wallMat = new THREE.MeshBasicMaterial({
    color: 0x0f172a,
    side: THREE.DoubleSide,
  });
  const wall = new THREE.Mesh(wallGeom, wallMat);
  wall.position.set(0, 1.2, -2.0);
  group.add(wall);

  const wallWire = new THREE.LineSegments(
    new THREE.WireframeGeometry(wallGeom),
    new THREE.LineBasicMaterial({ color: 0x334155 })
  );
  wallWire.position.copy(wall.position);
  group.add(wallWire);

  // Active extracted row tiles in foreground
  const safeIdx = Math.min(queryIdx, traceRes.tokens.length - 1);
  const activeId = traceRes.tokenIds[safeIdx];
  const activeToken = traceRes.tokens[safeIdx];

  const rowMesh = makeTileMesh(7.2, 0.55, 0.35, COLOR_ACCENT);
  rowMesh.position.set(0, 0.8, 0.8);
  group.add(rowMesh);

  const rowLbl = makeLabelSprite(
    `Extracted Row #${activeId} for token "${formatToken(activeToken)}" → 128 Float Coordinates`,
    '#ffffff',
    '#1e3a8a'
  );
  rowLbl.position.set(0, 1.35, 0.8);
  rowLbl.scale.set(6.8, 0.65, 1);
  group.add(rowLbl);

  if (wteWeights) {
    const previewCount = 16;
    const startX = -((previewCount - 1) * 0.42) / 2;
    const base = activeId * 128;
    for (let d = 0; d < previewCount; d++) {
      const val = wteWeights[base + d] ?? 0;
      const x = startX + d * 0.42;
      const h = Math.max(0.08, Math.min(1.4, Math.abs(val) * 4.5));
      const col = val >= 0 ? COLOR_EMERALD : COLOR_ROSE;
      const bar = makeTileMesh(0.32, h, 0.22, col);
      bar.position.set(x, 0.2 + h / 2, 1.6);
      group.add(bar);
    }

    const previewLbl = makeLabelSprite(
      'First 16 of 128 embedding dimensions (green = +, pink = −)',
      '#64748b'
    );
    previewLbl.position.set(0, -0.2, 1.6);
    previewLbl.scale.set(5.5, 0.45, 1);
    group.add(previewLbl);
  }
}

function buildCh2PcaCloud({
  group,
  pcaCloud,
  traceRes,
  queryIdx,
  wteWeights,
  tokenizer,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    '3D PCA Space: 512 Vocabulary Embeddings Projected onto Top 3 Principal Components',
    '#2563eb'
  );
  title.position.set(0, 3.2, 0);
  title.scale.set(9.5, 0.9, 1);
  group.add(title);

  if (!pcaCloud || pcaCloud.points.length === 0) {
    const loadingLbl = makeLabelSprite(
      'Computing 3D PCA projection on trained wte...',
      '#64748b'
    );
    loadingLbl.position.set(0, 1.5, 0);
    loadingLbl.scale.set(5, 0.6, 1);
    group.add(loadingLbl);
    return;
  }

  const safeIdx = Math.min(queryIdx, traceRes.tokens.length - 1);
  const activeId = traceRes.tokenIds[safeIdx];

  // Point cloud geometry
  const pts = pcaCloud.points;
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(pts.length * 3);
  const colors = new Float32Array(pts.length * 3);

  const activeColor = new THREE.Color(0xf59e0b);
  const normalColor = new THREE.Color(0x38bdf8);

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    positions[i * 3 + 0] = p.x;
    positions[i * 3 + 1] = p.y + 1.2;
    positions[i * 3 + 2] = p.z;

    const c = p.id === activeId ? activeColor : normalColor;
    colors[i * 3 + 0] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.18,
    vertexColors: true,
  });
  const cloud = new THREE.Points(geom, mat);
  group.add(cloud);

  // Highlight active token with an orb and label
  const activePt = pts[activeId];
  if (activePt) {
    const orb = new THREE.Mesh(
      new THREE.SphereGeometry(0.24, 16, 16),
      new THREE.MeshStandardMaterial({
        color: COLOR_AMBER,
        emissive: 0xd97706,
        emissiveIntensity: 0.5,
      })
    );
    orb.position.set(activePt.x, activePt.y + 1.2, activePt.z);
    group.add(orb);

    const activeLbl = makeLabelSprite(
      `Active: "${formatToken(activePt.text)}" (#${activeId})`,
      '#ffffff',
      '#d97706'
    );
    activeLbl.position.set(activePt.x, activePt.y + 1.7, activePt.z);
    activeLbl.scale.set(2.4, 0.55, 1);
    group.add(activeLbl);

    // Show cosine neighbor lines if weights are ready
    if (wteWeights) {
      const neighbors = findCosineNeighbors(
        wteWeights,
        tokenizer.vocab,
        128,
        activeId,
        4
      );
      for (const n of neighbors) {
        const targetPt = pts[n.id];
        if (!targetPt) continue;
        const lineGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(activePt.x, activePt.y + 1.2, activePt.z),
          new THREE.Vector3(targetPt.x, targetPt.y + 1.2, targetPt.z),
        ]);
        const line = new THREE.Line(
          lineGeom,
          new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.75 })
        );
        group.add(line);

        const nLbl = makeLabelSprite(
          `${formatToken(n.text)} (cos ${formatNum(n.cosine, 2)})`,
          '#10b981',
          'rgba(6,95,70,0.85)'
        );
        nLbl.position.set(targetPt.x, targetPt.y + 1.5, targetPt.z);
        nLbl.scale.set(1.8, 0.45, 1);
        group.add(nLbl);
      }
    }
  }
}

function buildCh2OrderBlind({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const tokens = traceRes.tokens;
  const count = Math.min(tokens.length, 10);
  const title = makeLabelSprite(
    'Order-Blind Bag of Words: Without position coordinates, "Max bit dog" == "Dog bit Max"',
    '#dc2626'
  );
  title.position.set(0, 2.8, 0);
  title.scale.set(9.5, 0.9, 1);
  group.add(title);

  // Scrambled cloud positions in circle
  const radius = 2.4;
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2 + 0.3;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 0.5 + Math.sin(i * 1.7) * 0.4;

    const mesh = makeTileMesh(1.1, 0.5, 0.35, 0xfecaca);
    mesh.position.set(x, y, z);
    group.add(mesh);

    const lbl = makeLabelSprite(
      formatToken(tokens[i]),
      '#991b1b',
      'transparent'
    );
    lbl.position.set(x, y + 0.5, z);
    lbl.scale.set(1.1, 0.45, 1);
    group.add(lbl);

    const qMark = makeLabelSprite('Position ?', '#dc2626');
    qMark.position.set(x, y - 0.4, z);
    qMark.scale.set(0.95, 0.35, 1);
    group.add(qMark);
  }
}

function buildCh2WpeAdd({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const tokens = traceRes.tokens;
  const count = Math.min(tokens.length, 6);
  const spacing = 1.6;
  const startX = -((count - 1) * spacing) / 2;

  const title = makeLabelSprite(
    'Absolute Position Addition: h_t = wte[token] + wpe[slot]',
    '#059669'
  );
  title.position.set(0, 2.8, 0);
  title.scale.set(8.5, 0.85, 1);
  group.add(title);

  for (let i = 0; i < count; i++) {
    const x = startX + i * spacing;

    // Token plate (wte)
    const wteMesh = makeTileMesh(1.2, 0.45, 0.3, 0xbfdbfe);
    wteMesh.position.set(x, 1.6, 0);
    group.add(wteMesh);

    const wteLbl = makeLabelSprite(
      `wte: "${formatToken(tokens[i])}"`,
      '#1e40af',
      'transparent'
    );
    wteLbl.position.set(x, 2.0, 0);
    wteLbl.scale.set(1.3, 0.4, 1);
    group.add(wteLbl);

    // Plus sign
    const plusLbl = makeLabelSprite('+', '#64748b', 'transparent');
    plusLbl.position.set(x, 1.15, 0);
    plusLbl.scale.set(0.5, 0.4, 1);
    group.add(plusLbl);

    // Position plate (wpe)
    const wpeMesh = makeTileMesh(1.2, 0.45, 0.3, 0xfef08a);
    wpeMesh.position.set(x, 0.7, 0);
    group.add(wpeMesh);

    const wpeLbl = makeLabelSprite(
      `wpe: slot ${i}`,
      '#854d0e',
      'transparent'
    );
    wpeLbl.position.set(x, 0.25, 0);
    wpeLbl.scale.set(1.2, 0.35, 1);
    group.add(wpeLbl);
  }
}

function buildCh2Rope({
  group,
  ropeSlotA,
  ropeSlotB,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    `RoPE Clock Wheels: Relative Angle Delta theta = (Slot ${ropeSlotB} - Slot ${ropeSlotA}) * omega`,
    '#2563eb'
  );
  title.position.set(0, 2.8, 0);
  title.scale.set(9.5, 0.85, 1);
  group.add(title);

  // Two clock faces side by side
  const baseFreq = 0.35;
  const angleA = ropeSlotA * baseFreq;
  const angleB = ropeSlotB * baseFreq;

  const clocks = [
    { name: `Slot A (${ropeSlotA})`, x: -2.2, angle: angleA, col: 0x2563eb },
    { name: `Slot B (${ropeSlotB})`, x: 2.2, angle: angleB, col: 0x059669 },
  ];

  for (const c of clocks) {
    // Clock face
    const face = new THREE.Mesh(
      new THREE.CylinderGeometry(1.4, 1.4, 0.1, 32),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.3 })
    );
    face.rotation.x = Math.PI / 2;
    face.position.set(c.x, 1.1, 0);
    group.add(face);

    // Hand
    const handGeom = new THREE.BoxGeometry(0.12, 1.1, 0.14);
    const handMat = new THREE.MeshStandardMaterial({ color: c.col });
    const hand = new THREE.Mesh(handGeom, handMat);
    hand.position.set(c.x, 1.1, 0.15);
    hand.rotation.z = -c.angle;
    group.add(hand);

    const lbl = makeLabelSprite(c.name, '#ffffff', '#0f172a');
    lbl.position.set(c.x, -0.4, 0);
    lbl.scale.set(1.8, 0.5, 1);
    group.add(lbl);
  }

  // Delta angle badge in center
  const deltaSlots = ropeSlotB - ropeSlotA;
  const deltaDeg = Math.round((Math.abs(angleB - angleA) * 180) / Math.PI);
  const deltaBadge = makeLabelSprite(
    `Delta = ${deltaSlots} slots (Angle difference = ${deltaDeg}°)`,
    '#d97706',
    '#fef3c7'
  );
  deltaBadge.position.set(0, 1.1, 0);
  deltaBadge.scale.set(3.8, 0.65, 1);
  group.add(deltaBadge);
}

/* -------------------------------------------------------------
 * CHAPTER 3 BUILDERS
 * ----------------------------------------------------------- */

function buildCh3ResidualBus({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    'The Residual Highway: h_(l+1) = h_l + Attention(LN(h_l)) + MLP(LN(h_l))',
    '#d97706'
  );
  title.position.set(0, 3.2, 0);
  title.scale.set(9.5, 0.9, 1);
  group.add(title);

  // Central highway bar
  const highwayGeom = new THREE.BoxGeometry(10.5, 0.45, 0.55);
  const highwayMat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    metalness: 0.3,
    roughness: 0.2,
  });
  const highway = new THREE.Mesh(highwayGeom, highwayMat);
  highway.position.set(0, 1.2, 0);
  group.add(highway);

  const hwLbl = makeLabelSprite('Central Residual Bus h [128]', '#ffffff', '#b45309');
  hwLbl.position.set(0, 1.75, 0);
  hwLbl.scale.set(3.4, 0.55, 1);
  group.add(hwLbl);

  // 4 layer branch loops off to the side
  const layerSpacing = 2.2;
  const startX = -((4 - 1) * layerSpacing) / 2;

  for (let l = 0; l < 4; l++) {
    const x = startX + l * layerSpacing;

    // Attention loop above
    const attnMesh = makeTileMesh(1.4, 0.5, 0.3, 0x2563eb);
    attnMesh.position.set(x, 2.4, 0);
    group.add(attnMesh);

    const attnLbl = makeLabelSprite(`Attn L${l}`, '#ffffff', 'transparent');
    attnLbl.position.set(x, 2.8, 0);
    attnLbl.scale.set(1.1, 0.35, 1);
    group.add(attnLbl);

    // MLP loop below
    const mlpMesh = makeTileMesh(1.4, 0.5, 0.3, 0x10b981);
    mlpMesh.position.set(x, 0.0, 0);
    group.add(mlpMesh);

    const mlpLbl = makeLabelSprite(`MLP L${l}`, '#ffffff', 'transparent');
    mlpLbl.position.set(x, -0.4, 0);
    mlpLbl.scale.set(1.1, 0.35, 1);
    group.add(mlpLbl);

    // Connecting branch lines
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, 0.25, 0),
      new THREE.Vector3(x, 2.15, 0),
    ]);
    const line = new THREE.Line(
      lineGeom,
      new THREE.LineBasicMaterial({ color: 0x94a3b8 })
    );
    group.add(line);
  }
}

function buildCh3CutHighway({
  group,
  ablation,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const isCut = (ablation.disabledResiduals?.length ?? 0) > 0;

  const title = makeLabelSprite(
    isCut
      ? 'CUT RESIDUAL HIGHWAY: Signals severed! Output predictions collapsing into gibberish'
      : 'Intact Residual Highway: Toggle "Cut Residuals" button to witness catastrophic failure',
    isCut ? '#dc2626' : '#059669'
  );
  title.position.set(0, 3.2, 0);
  title.scale.set(9.8, 0.9, 1);
  group.add(title);

  // Highway with fracture
  const leftGeom = new THREE.BoxGeometry(4.2, 0.45, 0.55);
  const rightGeom = new THREE.BoxGeometry(4.2, 0.45, 0.55);
  const mat = new THREE.MeshStandardMaterial({
    color: isCut ? COLOR_DANGER : 0xf59e0b,
  });

  const leftBar = new THREE.Mesh(leftGeom, mat);
  leftBar.position.set(-2.8, 1.2, 0);
  group.add(leftBar);

  const rightBar = new THREE.Mesh(rightGeom, mat);
  rightBar.position.set(2.8, 1.2, 0);
  group.add(rightBar);

  if (isCut) {
    const cutSign = makeLabelSprite('RESIDUAL SEVERED', '#ffffff', '#dc2626');
    cutSign.position.set(0, 1.2, 0);
    cutSign.scale.set(3.2, 0.75, 1);
    group.add(cutSign);
  } else {
    const bridge = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 0.45, 0.55),
      mat
    );
    bridge.position.set(0, 1.2, 0);
    group.add(bridge);
  }
}

function buildCh3LayerNorm({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    'Pre-LayerNorm: Centering (mean = 0) and scaling (std = 1.0) before every attention/MLP block',
    '#2563eb'
  );
  title.position.set(0, 3.2, 0);
  title.scale.set(9.8, 0.9, 1);
  group.add(title);

  // Side-by-side comparison: Unnormalized vs Normalized
  const dimCount = 16;
  const spacing = 0.38;
  const startX = -((dimCount - 1) * spacing) / 2;

  // Raw strip
  const rawLbl = makeLabelSprite('Unbounded Input (Var can explode)', '#dc2626');
  rawLbl.position.set(0, 2.2, -1.2);
  rawLbl.scale.set(5.5, 0.5, 1);
  group.add(rawLbl);

  for (let i = 0; i < dimCount; i++) {
    const rawVal = Math.sin((i + 1) * 2.3) * 4.2;
    const h = Math.max(0.1, Math.abs(rawVal) * 0.3);
    const bar = makeTileMesh(0.28, h, 0.22, 0xfca5a5);
    bar.position.set(startX + i * spacing, 1.0 + h / 2, -1.2);
    group.add(bar);
  }

  // Normalized strip
  const normLbl = makeLabelSprite('LayerNorm Output: Zero-Mean, Unit-Variance', '#059669');
  normLbl.position.set(0, 0.2, 1.2);
  normLbl.scale.set(5.8, 0.5, 1);
  group.add(normLbl);

  for (let i = 0; i < dimCount; i++) {
    const normVal = Math.sin((i + 1) * 2.3) * 1.0;
    const h = Math.max(0.1, Math.abs(normVal) * 0.6);
    const bar = makeTileMesh(0.28, h, 0.22, 0x86efac);
    bar.position.set(startX + i * spacing, -0.8 + h / 2, 1.2);
    group.add(bar);
  }
}

function buildCh3RmsNorm({
  group,
  traceRes,
  makeLabelSprite,
}: Pass2BuildArgs): void {
  const title = makeLabelSprite(
    'Modern Upgrade: LayerNorm (with mean centering) vs. RMSNorm (pure RMS scaling)',
    '#2563eb'
  );
  title.position.set(0, 3.2, 0);
  title.scale.set(9.8, 0.9, 1);
  group.add(title);

  const cards = [
    {
      title: 'Standard LayerNorm (GPT-2)',
      sub: 'y = (x − μ) / σ * γ + β',
      notes: '2 passes over memory (find μ, then σ). Has bias β.',
      x: -2.8,
      col: 0x93c5fd,
    },
    {
      title: 'Modern RMSNorm (LLaMA / Gemma)',
      sub: 'y = x / RMS(x) * γ',
      notes: '1 pass over memory. Skips μ centering. No bias β.',
      x: 2.8,
      col: 0x86efac,
    },
  ];

  for (const c of cards) {
    const plate = makeTileMesh(4.8, 2.2, 0.35, c.col);
    plate.position.set(c.x, 1.1, 0);
    group.add(plate);

    const tLbl = makeLabelSprite(c.title, '#0f172a', 'transparent');
    tLbl.position.set(c.x, 1.8, 0.22);
    tLbl.scale.set(3.8, 0.45, 1);
    group.add(tLbl);

    const fLbl = makeLabelSprite(c.sub, '#1e3a8a', '#ffffff');
    fLbl.position.set(c.x, 1.2, 0.22);
    fLbl.scale.set(3.2, 0.4, 1);
    group.add(fLbl);

    const nLbl = makeLabelSprite(c.notes, '#334155', 'transparent');
    nLbl.position.set(c.x, 0.6, 0.22);
    nLbl.scale.set(4.2, 0.35, 1);
    group.add(nLbl);
  }
}
