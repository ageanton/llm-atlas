import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { SceneBeatId } from '../content/bookData';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  TokenizerData,
} from '../engine/types';
import { isHeadDisabled } from '../state/bookState';
import { softmax } from '../engine/transformer';
import {
  computeDotProductBreakdown,
  createLabelSpriteTexture,
  divergingColor,
  formatNum,
  formatToken,
  formatTokenExplicit,
  sequentialColor,
} from '../utils/math';
import type { BpeStepSnapshot, PcaCloudData } from '../utils/pass2Math';
import type { TrainingReplayData } from '../utils/pass3Math';
import { buildChapter123Beat } from './pass2Builders';
import { buildChapter5678Beat } from './pass3Builders';

export interface HoverTooltip {
  x: number;
  y: number;
  title: string;
  detail: string;
}

export interface StageCallbacks {
  onPickToken: (index: number) => void;
  onPickHead: (head: number) => void;
  onPickVocabToken: (vocabId: number) => void;
  onHoverInfo: (tip: HoverTooltip | null) => void;
}

export interface StageSnapshot {
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
  pcaFocusVocabId: number | null;
  dotSweepDim: number;
  brokenZeroMask: boolean;
  bpeSteps: BpeStepSnapshot[];
  bpeStepIdx: number;
  ropeSlotA: number;
  ropeSlotB: number;
  replay: TrainingReplayData | null;
  geluProbeX: number;
  temperature: number;
  topK: number;
  topP: number;
  weightDelta: number;
  replayIdx: number;
  scaleIdx: number;
}

interface ClickTag {
  kind: 'token' | 'head' | 'pcaCloud' | 'dotDim';
  index?: number;
}

export class TextbookStage {
  private readonly container: HTMLElement;
  private readonly callbacks: StageCallbacks;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2(-10, -10);
  private readonly envRT: THREE.WebGLRenderTarget;

  private readonly stageRoot = new THREE.Group();
  private interactives: THREE.Object3D[] = [];
  private disposableTextures: THREE.Texture[] = [];
  private disposableGeometries: THREE.BufferGeometry[] = [];
  private disposableMaterials: THREE.Material[] = [];

  private camTargetPos = new THREE.Vector3(0, 3.6, 10.5);
  private camTargetLook = new THREE.Vector3(0, 0.2, 0);
  private needsCameraFlight = false;
  private isDragging = false;

  private lastBeat: SceneBeatId | null = null;
  private currentSnap: StageSnapshot | null = null;
  private animFrameId = 0;
  private clock = new THREE.Clock();
  private pulseMeshes: THREE.Mesh[] = [];
  private disposed = false;

  constructor(container: HTMLElement, callbacks: StageCallbacks) {
    this.container = container;
    this.callbacks = callbacks;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.04;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog('#f3f5f9', 24, 54);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = this.envRT.texture;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(
      36,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      100
    );
    this.camera.position.copy(this.camTargetPos);

    // Full orbit + smooth screen-space panning so nothing ever gets stuck off-screen
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = true;
    this.controls.panSpeed = 0.9;
    this.controls.minDistance = 3.0;
    this.controls.maxDistance = 26;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minPolarAngle = 0.1;
    this.controls.addEventListener('start', () => {
      this.isDragging = true;
      this.needsCameraFlight = false;
    });
    this.controls.addEventListener('end', () => {
      this.isDragging = false;
    });

    this.setupLighting();
    this.scene.add(this.stageRoot);

    this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.addEventListener('pointerleave', this.onPointerLeave);
    this.renderer.domElement.addEventListener('click', this.onClick);
    window.addEventListener('resize', this.onResize);

    this.animate();
  }

  private setupLighting(): void {
    const hemi = new THREE.HemisphereLight('#ffffff', '#dce4f0', 1.12);
    hemi.position.set(0, 14, 0);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight('#fffaf2', 1.3);
    sun.position.set(8, 15, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 2;
    sun.shadow.camera.far = 38;
    sun.shadow.camera.left = -12;
    sun.shadow.camera.right = 12;
    sun.shadow.camera.top = 9;
    sun.shadow.camera.bottom = -9;
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);

    const fill = new THREE.DirectionalLight('#c7dcff', 0.55);
    fill.position.set(-9, 7, -8);
    this.scene.add(fill);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(52, 42),
      new THREE.MeshStandardMaterial({
        color: '#f5f6f8',
        roughness: 0.95,
        metalness: 0.02,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.35;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(36, 36, '#d8e0ea', '#e8edf4');
    grid.position.y = -1.34;
    this.scene.add(grid);
  }

  private trackGeo<T extends THREE.BufferGeometry>(g: T): T {
    this.disposableGeometries.push(g);
    return g;
  }

  private trackMat<T extends THREE.Material>(m: T): T {
    this.disposableMaterials.push(m);
    return m;
  }

  private trackTex<T extends THREE.Texture>(t: T): T {
    this.disposableTextures.push(t);
    return t;
  }

  private clearStage(): void {
    this.stageRoot.clear();
    this.interactives = [];
    this.pulseMeshes = [];
    for (const t of this.disposableTextures) t.dispose();
    for (const g of this.disposableGeometries) g.dispose();
    for (const m of this.disposableMaterials) m.dispose();
    this.disposableTextures = [];
    this.disposableGeometries = [];
    this.disposableMaterials = [];
  }

  resetCamera(): void {
    if (this.lastBeat) {
      this.setCameraForBeat(this.lastBeat);
    }
  }

  update(snap: StageSnapshot): void {
    this.currentSnap = snap;
    if (this.lastBeat !== snap.beat) {
      this.lastBeat = snap.beat;
      this.setCameraForBeat(snap.beat);
    }
    this.rebuild(snap);
  }

  private setCameraForBeat(beat: SceneBeatId): void {
    this.needsCameraFlight = true;
    switch (beat) {
      case 'ch1-1-extremes':
      case 'ch1-2-bpe-merges':
      case 'ch1-3-spaces-blindspot':
      case 'ch1-4-token-ids':
        this.camTargetPos.set(0, 3.8, 11.2);
        this.camTargetLook.set(0, 0.55, 0);
        break;
      case 'ch1-5-context-window':
        this.camTargetPos.set(0.3, 5.4, 9.6);
        this.camTargetLook.set(0.3, 0.1, 0);
        break;
      case 'ch2-1-wte-lookup':
      case 'ch2-4-wpe-add':
        this.camTargetPos.set(0, 2.8, 10.6);
        this.camTargetLook.set(0, 0.45, 0);
        break;
      case 'ch2-2-pca-cloud':
        this.camTargetPos.set(4.6, 3.4, 7.6);
        this.camTargetLook.set(0, 0.3, 0);
        break;
      case 'ch2-3-order-blind':
      case 'ch2-5-rope':
        this.camTargetPos.set(0, 5.0, 8.8);
        this.camTargetLook.set(0, 0.25, 0);
        break;
      case 'ch3-1-residual-bus':
      case 'ch3-2-cut-highway':
        this.camTargetPos.set(0.6, 3.3, 10.8);
        this.camTargetLook.set(0.6, 0.35, 0);
        break;
      case 'ch3-3-layernorm':
      case 'ch3-4-rmsnorm':
        this.camTargetPos.set(0, 4.3, 9.4);
        this.camTargetLook.set(0, 0.35, 0);
        break;
      case 'ch5-1-mlp-role':
      case 'ch5-2-expansion':
      case 'ch5-4-wdown':
      case 'ch6-1-logit-lens':
      case 'ch6-6-kv-quant':
      case 'ch7-4-training-replay':
      case 'ch8-4-post-training':
      case 'ch8-5-capstone':
        this.camTargetPos.set(0.2, 3.2, 10.6);
        this.camTargetLook.set(0.2, 0.6, 0);
        break;
      case 'ch5-3-gelu':
      case 'ch5-5-swiglu-moe':
      case 'ch7-1-step0':
      case 'ch8-3-frontier-diff':
        this.camTargetPos.set(0.2, 4.4, 10.4);
        this.camTargetLook.set(0.2, 0.5, 0);
        break;
      case 'ch6-2-unembed':
      case 'ch6-3-temperature':
      case 'ch6-4-nucleus':
      case 'ch6-5-autoregression':
      case 'ch7-2-cross-entropy':
      case 'ch7-3-manual-backprop':
      case 'ch7-5-hallucination':
      case 'ch8-1-ablation':
      case 'ch8-2-scale':
        this.camTargetPos.set(0, 3.6, 11.2);
        this.camTargetLook.set(0, 0.85, 0);
        break;
      case 'attn-1-problem':
        this.camTargetPos.set(0, 3.2, 10.2);
        this.camTargetLook.set(0, 0.25, 0);
        break;
      case 'attn-2-dot-product':
        this.camTargetPos.set(0, 4.1, 8.8);
        this.camTargetLook.set(0, 0.25, 0);
        break;
      case 'attn-3-scan-row':
      case 'attn-4-mask-softmax':
        this.camTargetPos.set(0, 3.8, 10.6);
        this.camTargetLook.set(0, 0.35, 0);
        break;
      case 'attn-5-value-mixer':
        this.camTargetPos.set(0.2, 4.2, 9.6);
        this.camTargetLook.set(0.2, 0.2, 0);
        break;
      case 'attn-6-multihead-full':
        this.camTargetPos.set(0.2, 5.8, 9.2);
        this.camTargetLook.set(0.2, 0.1, 0);
        break;
    }
  }

  private addLabel(
    primary: string,
    secondary: string | undefined,
    pos: [number, number, number],
    scale = 1.75,
    accent = '#1d4ed8',
    muted = false
  ): void {
    const tex = this.trackTex(
      createLabelSpriteTexture(primary, secondary, accent, muted)
    );
    const mat = this.trackMat(
      new THREE.SpriteMaterial({ map: tex, transparent: true })
    );
    const sp = new THREE.Sprite(mat);
    const aspect = secondary ? 128 / 440 : 88 / 440;
    sp.scale.set(scale, scale * aspect, 1);
    sp.position.set(pos[0], pos[1], pos[2]);
    this.stageRoot.add(sp);
  }

  private rebuild(snap: StageSnapshot): void {
    this.clearStage();
    const handled = buildChapter123Beat(
      {
        stageRoot: this.stageRoot,
        interactives: this.interactives,
        trackGeo: (g) => this.trackGeo(g),
        trackMat: (m) => this.trackMat(m),
        trackTex: (t) => this.trackTex(t),
        addLabel: (p, s, pos, sc, acc, mut) =>
          this.addLabel(p, s, pos, sc, acc, mut),
      },
      snap
    );
    if (handled) return;

    const handledP3 = buildChapter5678Beat(
      {
        stageRoot: this.stageRoot,
        interactives: this.interactives,
        trackGeo: (g) => this.trackGeo(g),
        trackMat: (m) => this.trackMat(m),
        trackTex: (t) => this.trackTex(t),
        addLabel: (p, s, pos, sc, acc, mut) =>
          this.addLabel(p, s, pos, sc, acc, mut),
      },
      snap
    );
    if (handledP3) return;

    const layerTrace = snap.traceRes.trace.layers[snap.layer];
    if (layerTrace.skipped) {
      this.addLabel(
        `Block ${snap.layer} Disabled`,
        'Reset ablation to view Attention',
        [0, 0.6, 0],
        3.2,
        '#b45309',
        true
      );
      return;
    }

    const active: ActiveLayerTrace = layerTrace;
    switch (snap.beat) {
      case 'attn-1-problem':
        this.buildBeat1Problem(snap);
        break;
      case 'attn-2-dot-product':
        this.buildBeat2DotProduct(snap, active);
        break;
      case 'attn-3-scan-row':
        this.buildBeat3Or4Row(snap, active, false);
        break;
      case 'attn-4-mask-softmax':
        this.buildBeat3Or4Row(snap, active, true);
        break;
      case 'attn-5-value-mixer':
        this.buildBeat5ValueMixer(snap, active);
        break;
      case 'attn-6-multihead-full':
        this.buildBeat6FullMatrix(snap, active);
        break;
    }
  }

  // BEAT 1: Isolated Query token on a clean sentence arc — ZERO matrices
  private buildBeat1Problem(snap: StageSnapshot): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const span = Math.min(8.6, Math.max(4.6, T * 0.62));
    const blockW = Math.min(0.62, (span / Math.max(T, 1)) * 0.8);
    const geo = this.trackGeo(new RoundedBoxGeometry(blockW, 0.34, 0.68, 2, 0.05));

    for (let i = 0; i < T; i++) {
      const frac = T === 1 ? 0.5 : i / (T - 1);
      const x = -span / 2 + frac * span;
      const z = Math.sin(frac * Math.PI) * -0.9;
      const isQuery = i === snap.queryIdx;

      const mat = this.trackMat(
        new THREE.MeshStandardMaterial({
          color: isQuery ? '#1d4ed8' : '#ffffff',
          emissive: isQuery ? '#1e40af' : '#000000',
          emissiveIntensity: isQuery ? 0.25 : 0,
          roughness: 0.3,
        })
      );
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, isQuery ? 0.28 : -0.35, z);
      mesh.castShadow = true;
      mesh.userData = { kind: 'token', index: i } satisfies ClickTag;
      this.stageRoot.add(mesh);
      this.interactives.push(mesh);

      if (isQuery) {
        this.pulseMeshes.push(mesh);
        this.addLabel(
          `Query: “${formatToken(tokens[i].text)}”`,
          'Isolated — who does this refer to?',
          [x, 1.38, z],
          2.6,
          '#1d4ed8'
        );
      } else if (T <= 14 || i % 2 === 0 || i === T - 1) {
        this.addLabel(
          formatToken(tokens[i].text),
          undefined,
          [x, -0.82, z + 0.48],
          0.86,
          '#64748b'
        );
      }
    }
  }

  // BEAT 2: ONE single Q · K / sqrt(32) comparison — kinetic & hoverable 32-bar strips
  private buildBeat2DotProduct(
    snap: StageSnapshot,
    active: ActiveLayerTrace
  ): void {
    const tokens = snap.traceRes.tokens;
    const qVec = active.attn.q[snap.head][snap.queryIdx];
    const kVec = active.attn.k[snap.head][snap.keyIdx];
    const breakdown = computeDotProductBreakdown(qVec, kVec);

    const D = breakdown.dim;
    const limit = Math.max(1, Math.min(D, snap.dotSweepDim));
    let runningDot = 0;
    for (let d = 0; d < limit; d++) runningDot += breakdown.products[d];
    const runningScaled = runningDot / breakdown.scaleDivisor;

    const stripW = 7.4;
    const step = stripW / D;
    const barGeo = this.trackGeo(new THREE.BoxGeometry(step * 0.72, 1, 0.22));

    let maxQK = 1e-5;
    let maxProd = 1e-5;
    for (let d = 0; d < D; d++) {
      maxQK = Math.max(maxQK, Math.abs(qVec[d]), Math.abs(kVec[d]));
      maxProd = Math.max(maxProd, Math.abs(breakdown.products[d]));
    }

    const buildStrip = (
      values: number[],
      zPos: number,
      maxAbs: number,
      scaleH: number
    ) => {
      const rail = new THREE.Mesh(
        this.trackGeo(new THREE.BoxGeometry(stripW + 0.2, 0.05, 0.3)),
        this.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
      );
      rail.position.set(0, 0, zPos);
      this.stageRoot.add(rail);

      for (let d = 0; d < D; d++) {
        const v = values[d];
        const norm = v / maxAbs;
        const h = Math.max(0.05, Math.abs(norm) * scaleH);
        const hiddenAhead = d >= limit;
        const isFrontier = limit < D && d === limit - 1;
        const bar = new THREE.Mesh(
          barGeo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: divergingColor(norm),
              emissive: isFrontier ? '#f59e0b' : '#000000',
              emissiveIntensity: isFrontier ? 0.55 : 0,
              roughness: 0.3,
              transparent: hiddenAhead,
              opacity: hiddenAhead ? 0.14 : 1.0,
            })
          )
        );
        const x = -stripW / 2 + (d + 0.5) * step;
        bar.scale.y = h;
        bar.position.set(x, v >= 0 ? h / 2 : -h / 2, zPos);
        bar.castShadow = !hiddenAhead;
        bar.userData = { kind: 'dotDim', index: d } satisfies ClickTag;
        this.stageRoot.add(bar);
        this.interactives.push(bar);
      }
    };

    buildStrip(qVec, -1.55, maxQK, 1.05);
    this.addLabel(
      `Query: “${formatToken(tokens[snap.queryIdx].text)}”`,
      '32 dims · hover any bar to inspect',
      [-2.9, 1.48, -1.55],
      2.4,
      '#1d4ed8'
    );

    buildStrip(kVec, 1.55, maxQK, 1.05);
    this.addLabel(
      `Key: “${formatToken(tokens[snap.keyIdx].text)}”`,
      '32 dims · hover any bar to inspect',
      [-2.9, 1.48, 1.55],
      2.4,
      '#0f766e'
    );

    buildStrip(breakdown.products, 0, maxProd, 1.35);
    this.addLabel(
      limit < D
        ? `Running Score (${limit}/32) = ${formatNum(runningScaled, 3)}`
        : `Match Score = ${formatNum(breakdown.scaledScore, 3)}`,
      `sum(${formatNum(runningDot, 2)}) ÷ √32`,
      [2.5, 1.88, 0],
      2.65,
      runningScaled >= 0 ? '#b45309' : '#1d4ed8'
    );
  }

  // BEATS 3 & 4: One 1D row of scores across the sentence (with Causal Mask + Softmax in Beat 4)
  private buildBeat3Or4Row(
    snap: StageSnapshot,
    active: ActiveLayerTrace,
    applyMaskAndSoftmax: boolean
  ): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const qIdx = snap.queryIdx;
    const rawRow = active.attn.rawScores[snap.head][qIdx];
    const softRow = active.attn.attnWeights[snap.head][qIdx];
    const showSoftmax = applyMaskAndSoftmax && !snap.showPreSoftmax;
    const brokenActive = applyMaskAndSoftmax && snap.brokenZeroMask && qIdx < T - 1;
    const activeSoftRow = brokenActive
      ? softmax(rawRow.map((v, c) => (c > qIdx ? 0 : v)))
      : softRow;

    const span = Math.min(9.0, Math.max(5.0, T * 0.64));
    const step = span / T;
    const barW = step * 0.72;
    const barGeo = this.trackGeo(new RoundedBoxGeometry(barW, 1, 0.56, 2, 0.04));

    let maxAbsRaw = 1e-5;
    for (let c = 0; c < T; c++) {
      if (!applyMaskAndSoftmax || c <= qIdx) {
        maxAbsRaw = Math.max(maxAbsRaw, Math.abs(rawRow[c]));
      }
    }

    // Zero baseline beam
    const baseline = new THREE.Mesh(
      this.trackGeo(new THREE.BoxGeometry(span + 0.5, 0.05, 0.85)),
      this.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
    );
    baseline.position.set(0, 0, 0);
    this.stageRoot.add(baseline);

    let bestCol = 0;
    for (let c = 1; c <= qIdx; c++) {
      if (activeSoftRow[c] > activeSoftRow[bestCol]) bestCol = c;
    }

    for (let c = 0; c < T; c++) {
      const x = -span / 2 + (c + 0.5) * step;
      const isFuture = c > qIdx;

      if (applyMaskAndSoftmax && isFuture) {
        if (brokenActive) {
          const leakedP = activeSoftRow[c];
          const lh = showSoftmax ? Math.max(0.12, leakedP * 3.1) : 0.14;
          const leakBar = new THREE.Mesh(
            barGeo,
            this.trackMat(
              new THREE.MeshStandardMaterial({
                color: '#dc2626',
                emissive: '#991b1b',
                emissiveIntensity: 0.25,
                roughness: 0.3,
              })
            )
          );
          leakBar.scale.y = lh;
          leakBar.position.set(x, lh / 2, 0);
          this.stageRoot.add(leakBar);
          continue;
        }
        // Opaque matte Causal Mask shutter over future tokens
        const shutter = new THREE.Mesh(
          this.trackGeo(new RoundedBoxGeometry(barW * 1.06, 0.9, 0.72, 2, 0.04)),
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: '#cbd5e1',
              roughness: 0.9,
              transparent: true,
              opacity: 0.82,
            })
          )
        );
        shutter.position.set(x, 0.45, 0);
        shutter.userData = { kind: 'token', index: c } satisfies ClickTag;
        this.stageRoot.add(shutter);
        this.interactives.push(shutter);
        continue;
      }

      const raw = rawRow[c];
      const prob = activeSoftRow[c];
      const h = showSoftmax
        ? Math.max(0.06, prob * 3.1)
        : Math.max(0.08, (Math.abs(raw) / maxAbsRaw) * 1.85);

      const colorStr = showSoftmax
        ? sequentialColor(prob)
        : divergingColor(raw / maxAbsRaw);

      const bar = new THREE.Mesh(
        barGeo,
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: colorStr,
            emissive: c === bestCol && showSoftmax ? '#1d4ed8' : '#000000',
            emissiveIntensity: c === bestCol && showSoftmax ? 0.2 : 0,
            roughness: 0.28,
          })
        )
      );
      bar.scale.y = h;
      bar.position.set(x, showSoftmax || raw >= 0 ? h / 2 : -h / 2, 0);
      bar.castShadow = true;
      bar.userData = { kind: 'token', index: c } satisfies ClickTag;
      this.stageRoot.add(bar);
      this.interactives.push(bar);

      if (c === bestCol || c === qIdx || T <= 10) {
        const valStr = showSoftmax
          ? `${(prob * 100).toFixed(0)}%`
          : formatNum(raw, 2);
        this.addLabel(
          formatToken(tokens[c].text),
          valStr,
          [x, (showSoftmax || raw >= 0 ? h : 0.2) + 0.58, 0],
          1.05,
          c === bestCol ? '#1d4ed8' : '#475569'
        );
      }
    }

    if (applyMaskAndSoftmax && qIdx < T - 1) {
      const firstFutureX = -span / 2 + (qIdx + 1.5) * step;
      const lastFutureX = -span / 2 + (T - 0.5) * step;
      if (brokenActive) {
        const stolen = activeSoftRow.slice(qIdx + 1).reduce((a, b) => a + b, 0) * 100;
        this.addLabel(
          'MASK BROKEN (0 ≠ -∞)!',
          `exp(0)=1 → ${stolen.toFixed(1)}% weight leaked to future`,
          [(firstFutureX + lastFutureX) / 2, 1.55, 0],
          2.35,
          '#dc2626'
        );
      } else {
        this.addLabel(
          'Causal Mask (-∞)',
          'Future blocked → 0.0%',
          [(firstFutureX + lastFutureX) / 2, 1.32, 0],
          1.95,
          '#64748b',
          true
        );
      }
    }

    this.addLabel(
      `Query Row: “${formatToken(tokens[qIdx].text)}”`,
      showSoftmax
        ? 'Post-Softmax weights (sum = 100.0%)'
        : 'Raw signed scores QKᵀ / √32',
      [-span / 2 + 1.4, 2.55, -0.4],
      2.6,
      showSoftmax ? '#1d4ed8' : '#b45309'
    );
  }

  // BEAT 5: Value Mixing Bowl (A · V) — Top-4 Value vectors flowing into blended output
  private buildBeat5ValueMixer(
    snap: StageSnapshot,
    active: ActiveLayerTrace
  ): void {
    const tokens = snap.traceRes.tokens;
    const qIdx = snap.queryIdx;
    const weightsRow = active.attn.attnWeights[snap.head][qIdx];
    const vMat = active.attn.v[snap.head];
    const outVec = active.attn.weightedV[snap.head][qIdx];
    const headOff = isHeadDisabled(snap.ablation, snap.layer, snap.head);

    const contributors = weightsRow
      .slice(0, qIdx + 1)
      .map((w, idx) => ({ idx, w, vec: vMat[idx] }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 4);

    let maxAbs = 1e-5;
    for (const item of contributors) {
      for (const v of item.vec) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    for (const v of outVec) maxAbs = Math.max(maxAbs, Math.abs(v));

    const stripLen = 3.2;
    const D = outVec.length;
    const step = stripLen / D;
    const barGeo = this.trackGeo(new THREE.BoxGeometry(step * 0.72, 1, 0.16));

    const drawMiniVec = (
      vec: number[],
      centerX: number,
      centerZ: number,
      alpha: number
    ) => {
      for (let d = 0; d < D; d++) {
        const norm = vec[d] / maxAbs;
        const h = Math.max(0.04, Math.abs(norm) * 0.72);
        const bar = new THREE.Mesh(
          barGeo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: divergingColor(norm),
              roughness: 0.3,
              transparent: alpha < 0.99,
              opacity: alpha,
            })
          )
        );
        bar.scale.y = h;
        bar.position.set(
          centerX - stripLen / 2 + (d + 0.5) * step,
          vec[d] >= 0 ? h / 2 : -h / 2,
          centerZ
        );
        this.stageRoot.add(bar);
      }
    };

    const destPos = new THREE.Vector3(2.45, 0.15, 0);

    contributors.forEach((item, rank) => {
      const z = -2.1 + rank * 1.4;
      const srcX = -2.55;
      drawMiniVec(item.vec, srcX, z, headOff ? 0.25 : Math.max(0.35, 0.4 + item.w));

      this.addLabel(
        `V(“${formatToken(tokens[item.idx].text)}”)`,
        `${(item.w * 100).toFixed(1)}% weight`,
        [srcX, 1.05, z],
        1.7,
        rank === 0 ? '#1d4ed8' : '#475569'
      );

      // Proportional mixing pipe flowing from this Value strip into the A·V output
      const p0 = new THREE.Vector3(srcX + stripLen / 2 + 0.1, 0.15, z);
      const mid = new THREE.Vector3(0.05, 0.45 + item.w * 0.8, z * 0.45);
      const p1 = new THREE.Vector3(destPos.x - stripLen / 2 - 0.1, 0.15, 0);
      const curve = new THREE.CatmullRomCurve3([p0, mid, p1]);
      const radius = headOff ? 0.015 : 0.02 + item.w * 0.11;
      const tube = new THREE.Mesh(
        this.trackGeo(new THREE.TubeGeometry(curve, 24, radius, 10, false)),
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: headOff ? '#cbd5e1' : rank === 0 ? '#d97706' : '#2563eb',
            roughness: 0.25,
            transparent: true,
            opacity: headOff ? 0.3 : Math.min(0.92, 0.25 + item.w * 0.9),
          })
        )
      );
      this.stageRoot.add(tube);
    });

    // Blended output vector Z_h[queryIdx] = (A · V)[queryIdx]
    drawMiniVec(outVec, destPos.x, 0, 1);
    this.addLabel(
      headOff ? 'Blended A·V (Zeroed!)' : `Blended A·V for “${formatToken(tokens[qIdx].text)}”`,
      '32-dim context update → W_o → + Residual',
      [destPos.x, 1.32, 0],
      2.35,
      headOff ? '#b45309' : '#0f766e',
      headOff
    );
  }

  // BEAT 6: Full [T, T] stacked matrix + 4 parallel heads feeding into W_o
  private buildBeat6FullMatrix(
    snap: StageSnapshot,
    active: ActiveLayerTrace
  ): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const weights = active.attn.attnWeights[snap.head];
    const headOff = isHeadDisabled(snap.ablation, snap.layer, snap.head);

    const gridSize = 4.6;
    const cell = gridSize / T;
    const startX = -gridSize / 2 - 0.8;
    const startZ = -gridSize / 2;

    const boxGeo = this.trackGeo(new THREE.BoxGeometry(cell * 0.88, 1, cell * 0.88));
    const maskMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: '#e2e8f0',
        roughness: 0.9,
        transparent: true,
        opacity: 0.55,
      })
    );

    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        const x = startX + (c + 0.5) * cell;
        const z = startZ + (r + 0.5) * cell;
        if (c > r) {
          const m = new THREE.Mesh(boxGeo, maskMat);
          m.scale.y = 0.05;
          m.position.set(x, 0.025, z);
          this.stageRoot.add(m);
          continue;
        }
        const w = weights[r][c];
        const barH = Math.max(0.05, w * 1.25);
        const isRow = r === snap.queryIdx;
        const tile = new THREE.Mesh(
          boxGeo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: headOff ? '#cbd5e1' : sequentialColor(w),
              emissive: isRow && !headOff ? '#1d4ed8' : '#000000',
              emissiveIntensity: isRow ? 0.2 : 0,
              roughness: 0.28,
            })
          )
        );
        tile.scale.y = barH;
        tile.position.set(x, barH / 2, z);
        tile.userData = { kind: 'token', index: r } satisfies ClickTag;
        this.stageRoot.add(tile);
        this.interactives.push(tile);
      }
    }

    // 4 Parallel Head columns standing beside the matrix -> feeding into W_o
    const headGeo = this.trackGeo(new RoundedBoxGeometry(0.68, 0.36, 0.68, 2, 0.05));
    for (let h = 0; h < 4; h++) {
      const z = -1.65 + h * 1.1;
      const off = isHeadDisabled(snap.ablation, snap.layer, h);
      const isCur = h === snap.head;
      const block = new THREE.Mesh(
        headGeo,
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: off ? '#fed7aa' : isCur ? '#1d4ed8' : '#64748b',
            emissive: isCur && !off ? '#1e40af' : '#000000',
            emissiveIntensity: isCur ? 0.3 : 0,
            roughness: 0.28,
          })
        )
      );
      block.position.set(2.9, 0.25, z);
      block.castShadow = true;
      block.userData = { kind: 'head', index: h } satisfies ClickTag;
      this.stageRoot.add(block);
      this.interactives.push(block);

      this.addLabel(
        `Head ${h}`,
        off ? 'Disabled' : isCur ? 'Viewing' : '[T×32]',
        [2.9, 0.92, z],
        1.15,
        off ? '#b45309' : isCur ? '#1d4ed8' : '#475569',
        off
      );
    }

    this.addLabel(
      `Block ${snap.layer} · Head ${snap.head} Matrix`,
      headOff ? 'Head Disabled (Output = 0)' : `${T}×${T} lower-triangular weights`,
      [startX + gridSize / 2, 1.9, startZ - 0.5],
      2.4,
      headOff ? '#b45309' : '#1d4ed8',
      headOff
    );
  }

  private pickHit(
    event: PointerEvent | MouseEvent
  ): { tag: ClickTag; instanceId?: number } | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactives, false);
    for (const hit of hits) {
      const tag = hit.object.userData as ClickTag | undefined;
      if (tag?.kind) return { tag, instanceId: hit.instanceId };
    }
    return null;
  }

  private readonly onPointerLeave = (): void => {
    this.callbacks.onHoverInfo(null);
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    const picked = this.pickHit(event);
    this.renderer.domElement.style.cursor = picked ? 'pointer' : 'grab';
    if (!picked || !this.currentSnap) {
      this.callbacks.onHoverInfo(null);
      return;
    }

    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (
      picked.tag.kind === 'pcaCloud' &&
      typeof picked.instanceId === 'number' &&
      this.currentSnap.pcaCloud
    ) {
      const pt = this.currentSnap.pcaCloud.points[picked.instanceId];
      if (pt) {
        this.callbacks.onHoverInfo({
          x,
          y,
          title: `“${formatTokenExplicit(pt.text)}” · ID #${pt.id}`,
          detail: `||wte|| = ${pt.norm.toFixed(3)} · Click sphere for 128D neighbors`,
        });
        return;
      }
    }

    if (
      picked.tag.kind === 'dotDim' &&
      typeof picked.tag.index === 'number'
    ) {
      const lt = this.currentSnap.traceRes.trace.layers[this.currentSnap.layer];
      if (!lt.skipped) {
        const d = picked.tag.index;
        const qVal = lt.attn.q[this.currentSnap.head][this.currentSnap.queryIdx][d];
        const kVal = lt.attn.k[this.currentSnap.head][this.currentSnap.keyIdx][d];
        this.callbacks.onHoverInfo({
          x,
          y,
          title: `Dimension d = ${d}`,
          detail: `q(${formatNum(qVal, 3)}) × k(${formatNum(kVal, 3)}) = ${formatNum(qVal * kVal, 4)}`,
        });
        return;
      }
    }

    this.callbacks.onHoverInfo(null);
  };

  private readonly onClick = (event: MouseEvent): void => {
    const picked = this.pickHit(event);
    if (!picked) return;
    if (picked.tag.kind === 'token' && typeof picked.tag.index === 'number') {
      this.callbacks.onPickToken(picked.tag.index);
    } else if (picked.tag.kind === 'head' && typeof picked.tag.index === 'number') {
      this.callbacks.onPickHead(picked.tag.index);
    } else if (
      picked.tag.kind === 'pcaCloud' &&
      typeof picked.instanceId === 'number'
    ) {
      this.callbacks.onPickVocabToken(picked.instanceId);
    }
  };

  private readonly onResize = (): void => {
    if (this.disposed) return;
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  private readonly animate = (): void => {
    if (this.disposed) return;
    this.animFrameId = requestAnimationFrame(this.animate);

    const dt = Math.min(0.05, this.clock.getDelta());
    const t = this.clock.elapsedTime;

    if (this.needsCameraFlight && !this.isDragging) {
      const ease = 1 - Math.exp(-5.5 * dt);
      this.camera.position.lerp(this.camTargetPos, ease);
      this.controls.target.lerp(this.camTargetLook, ease);
      if (
        this.camera.position.distanceTo(this.camTargetPos) < 0.03 &&
        this.controls.target.distanceTo(this.camTargetLook) < 0.03
      ) {
        this.needsCameraFlight = false;
      }
    }
    this.controls.update();

    for (const m of this.pulseMeshes) {
      m.position.y = 0.28 + Math.sin(t * 2.4) * 0.06;
    }

    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animFrameId);
    window.removeEventListener('resize', this.onResize);
    this.renderer.domElement.removeEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.removeEventListener('pointerleave', this.onPointerLeave);
    this.renderer.domElement.removeEventListener('click', this.onClick);
    this.clearStage();
    this.controls.dispose();
    this.envRT.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
