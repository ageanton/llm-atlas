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
import {
  computeDotProductBreakdown,
  createLabelSpriteTexture,
  divergingColor,
  formatNum,
  formatToken,
  sequentialColor,
} from '../utils/math';
import type { BpeStepSnapshot, PcaCloudData } from '../utils/pass2Math';
import { buildChapter123Beat } from './pass2Builders';

export interface StageCallbacks {
  onPickToken: (index: number) => void;
  onPickHead: (head: number) => void;
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
  bpeSteps: BpeStepSnapshot[];
  bpeStepIdx: number;
  ropeSlotA: number;
  ropeSlotB: number;
}

interface ClickTag {
  kind: 'token' | 'head';
  index: number;
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
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = this.envRT.texture;

    this.camera = new THREE.PerspectiveCamera(
      38,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.1,
      80
    );
    this.camera.position.set(0, 3.8, 11.2);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.rotateSpeed = 0.75;
    this.controls.zoomSpeed = 0.85;
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
    this.renderer.domElement.addEventListener('click', this.onClick);
    window.addEventListener('resize', this.onResize);

    this.animate();
  }

  private setupLighting(): void {
    const hemi = new THREE.HemisphereLight('#ffffff', '#dce4f0', 1.12);
    hemi.position.set(0, 14, 0);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight('#fffaf0', 1.85);
    sun.position.set(7, 13, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 32;
    sun.shadow.camera.left = -9;
    sun.shadow.camera.right = 9;
    sun.shadow.camera.top = 9;
    sun.shadow.camera.bottom = -9;
    sun.shadow.bias = -0.0003;
    this.scene.add(sun);

    const softFill = new THREE.DirectionalLight('#93c5fd', 0.55);
    softFill.position.set(-8, 6, -6);
    this.scene.add(softFill);

    const floorGeo = new THREE.PlaneGeometry(54, 54);
    const floorMat = new THREE.MeshStandardMaterial({
      color: '#f1f5f9',
      roughness: 0.92,
      metalness: 0.02,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.55;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(36, 36, '#cbd5e1', '#e2e8f0');
    grid.position.y = -0.54;
    this.scene.add(grid);
  }

  private onPointerMove = (e: MouseEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  };

  private onClick = (): void => {
    if (this.isDragging) return;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactives, true);
    if (hits.length === 0) return;

    for (const h of hits) {
      let cur: THREE.Object3D | null = h.object;
      while (cur && cur !== this.stageRoot) {
        const tag = cur.userData as ClickTag | undefined;
        if (tag && tag.kind === 'token') {
          this.callbacks.onPickToken(tag.index);
          return;
        }
        if (tag && tag.kind === 'head') {
          this.callbacks.onPickHead(tag.index);
          return;
        }
        cur = cur.parent;
      }
    }
  };

  private onResize = (): void => {
    if (this.disposed) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  public resetCamera(): void {
    if (this.lastBeat) this.setCameraForBeat(this.lastBeat);
  }

  update(snap: StageSnapshot): void {
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
        this.buildBeat2DotProduct(
          snap,
          active.attn.q[snap.head][snap.queryIdx],
          active.attn.k[snap.head][snap.keyIdx]
        );
        break;
      case 'attn-3-scan-row':
        this.buildBeat3ScanRow(
          snap,
          active.attn.rawScores[snap.head][snap.queryIdx]
        );
        break;
      case 'attn-4-mask-softmax':
        this.buildBeat4MaskSoftmax(snap, active);
        break;
      case 'attn-5-value-mixer':
        this.buildBeat5ValueMixer(
          snap,
          active.attn.attnWeights[snap.head][snap.queryIdx],
          active.attn.v[snap.head],
          active.attn.headOut[snap.head][snap.queryIdx]
        );
        break;
      case 'attn-6-multihead-full':
        this.buildBeat6MultiHead(snap, active);
        break;
    }
  }

  // BEAT 1: Words in Isolation
  private buildBeat1Problem(snap: StageSnapshot): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const span = Math.min(9.0, Math.max(5.0, T * 0.75));
    const step = span / Math.max(1, T);

    const geo = this.trackGeo(new RoundedBoxGeometry(step * 0.72, 0.44, 0.72, 2, 0.05));

    for (let i = 0; i < T; i++) {
      const tok = tokens[i];
      const isQ = i === snap.queryIdx;
      const mat = this.trackMat(
        new THREE.MeshStandardMaterial({
          color: isQ ? '#1d4ed8' : '#e2e8f0',
          roughness: 0.3,
          metalness: 0.05,
        })
      );
      const mesh = new THREE.Mesh(geo, mat);
      const x = -span / 2 + (i + 0.5) * step;
      mesh.position.set(x, 0.22, 0);
      mesh.castShadow = true;
      mesh.userData = { kind: 'token', index: i } satisfies ClickTag;
      this.stageRoot.add(mesh);
      this.interactives.push(mesh);

      this.addLabel(
        formatToken(tok.text),
        `#${tok.id}`,
        [x, 0.85, 0],
        0.98,
        isQ ? '#1d4ed8' : '#334155'
      );
    }

    const qX = -span / 2 + (snap.queryIdx + 0.5) * step;
    this.addLabel(
      'Isolated Query Token',
      'No cross-token links yet',
      [qX, 1.85, 0],
      1.95,
      '#1d4ed8'
    );
  }

  // BEAT 2: One Pairwise Comparison (Q · K)
  private buildBeat2DotProduct(
    snap: StageSnapshot,
    qVec: number[],
    kVec: number[]
  ): void {
    const bd = computeDotProductBreakdown(qVec, kVec);
    const D = 32;
    const stripW = 6.8;
    const step = stripW / D;

    const barGeo = this.trackGeo(new THREE.BoxGeometry(step * 0.78, 1, 0.28));

    const drawVector = (vec: number[], z: number, label: string, accent: string) => {
      let maxAbs = 1e-5;
      for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));

      const rail = new THREE.Mesh(
        this.trackGeo(new THREE.BoxGeometry(stripW + 0.16, 0.04, 0.32)),
        this.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
      );
      rail.position.set(0, 0, z);
      this.stageRoot.add(rail);

      for (let d = 0; d < D; d++) {
        const norm = vec[d] / maxAbs;
        const h = Math.max(0.04, Math.abs(norm) * 0.95);
        const bar = new THREE.Mesh(
          barGeo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: divergingColor(norm),
              roughness: 0.32,
            })
          )
        );
        bar.scale.y = h;
        bar.position.set(-stripW / 2 + (d + 0.5) * step, vec[d] >= 0 ? h / 2 : -h / 2, z);
        this.stageRoot.add(bar);
      }
      this.addLabel(label, `32 dimensions`, [0, 1.35, z], 1.9, accent);
    };

    const qText = formatToken(snap.traceRes.tokens[snap.queryIdx]?.text ?? '');
    const kText = formatToken(snap.traceRes.tokens[snap.keyIdx]?.text ?? '');

    drawVector(qVec, -1.35, `Query: “${qText}”`, '#1d4ed8');
    drawVector(kVec, 1.35, `Key: “${kText}”`, '#0f766e');

    // Product bars in the middle
    let maxProd = 1e-5;
    for (const p of bd.products) maxProd = Math.max(maxProd, Math.abs(p));

    const prodRail = new THREE.Mesh(
      this.trackGeo(new THREE.BoxGeometry(stripW + 0.16, 0.04, 0.32)),
      this.trackMat(new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.7 }))
    );
    prodRail.position.set(0, 0, 0);
    this.stageRoot.add(prodRail);

    for (let d = 0; d < D; d++) {
      const p = bd.products[d];
      const norm = p / maxProd;
      const h = Math.max(0.04, Math.abs(norm) * 1.25);
      const bar = new THREE.Mesh(
        barGeo,
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: divergingColor(norm),
            roughness: 0.28,
          })
        )
      );
      bar.scale.y = h;
      bar.position.set(-stripW / 2 + (d + 0.5) * step, p >= 0 ? h / 2 : -h / 2, 0);
      this.stageRoot.add(bar);
    }

    this.addLabel(
      `Sum = ${formatNum(bd.rawDot, 2)} ÷ √32 = ${formatNum(bd.scaledScore, 2)}`,
      `+${formatNum(bd.positiveVotes, 2)} vs ${formatNum(bd.negativeVotes, 2)}`,
      [0, 1.75, 0],
      2.6,
      bd.scaledScore >= 0 ? '#b45309' : '#1d4ed8'
    );
  }

  // BEAT 3: Scanning One Row of Compatibility
  private buildBeat3ScanRow(snap: StageSnapshot, scores: number[]): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const span = Math.min(9.0, Math.max(5.0, T * 0.75));
    const step = span / Math.max(1, T);

    const geo = this.trackGeo(new RoundedBoxGeometry(step * 0.74, 1, 0.44, 2, 0.04));

    let maxAbs = 1e-5;
    for (let c = 0; c < T; c++) maxAbs = Math.max(maxAbs, Math.abs(scores[c]));

    for (let c = 0; c < T; c++) {
      const sc = scores[c];
      const norm = sc / maxAbs;
      const h = Math.max(0.08, Math.abs(norm) * 1.85);
      const x = -span / 2 + (c + 0.5) * step;
      const isSel = c === snap.queryIdx;

      const bar = new THREE.Mesh(
        geo,
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: divergingColor(norm),
            roughness: 0.3,
          })
        )
      );
      bar.scale.y = h;
      bar.position.set(x, sc >= 0 ? h / 2 : -h / 2, 0);
      bar.castShadow = true;
      bar.userData = { kind: 'token', index: c } satisfies ClickTag;
      this.stageRoot.add(bar);
      this.interactives.push(bar);

      this.addLabel(
        formatToken(tokens[c].text),
        formatNum(sc, 2),
        [x, (sc >= 0 ? h : 0) + 0.52, 0],
        0.88,
        isSel ? '#1d4ed8' : '#334155'
      );
    }

    const qText = formatToken(tokens[snap.queryIdx]?.text ?? '');
    this.addLabel(
      `Raw Attention Row: Query “${qText}”`,
      'Amber = positive alignment · Blue = negative penalty',
      [0, 2.35, -0.4],
      3.1,
      '#1d4ed8'
    );
  }

  // BEAT 4: Causal Mask Shutter + Softmax Weights
  private buildBeat4MaskSoftmax(
    snap: StageSnapshot,
    active: ActiveLayerTrace
  ): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;
    const span = Math.min(9.0, Math.max(5.0, T * 0.75));
    const step = span / Math.max(1, T);

    const scores = active.attn.rawScores[snap.head][snap.queryIdx];
    const weights = active.attn.attnWeights[snap.head][snap.queryIdx];
    const geo = this.trackGeo(new RoundedBoxGeometry(step * 0.74, 1, 0.52, 2, 0.04));

    let maxAbs = 1e-5;
    for (let c = 0; c <= snap.queryIdx; c++) {
      maxAbs = Math.max(maxAbs, Math.abs(scores[c]));
    }

    for (let c = 0; c < T; c++) {
      const isFuture = c > snap.queryIdx;
      const x = -span / 2 + (c + 0.5) * step;

      if (!isFuture) {
        const val = snap.showPreSoftmax ? scores[c] : weights[c];
        const h = Math.max(
          0.06,
          snap.showPreSoftmax ? Math.abs(val / maxAbs) * 1.85 : val * 2.6
        );
        const col = snap.showPreSoftmax
          ? divergingColor(val / maxAbs)
          : sequentialColor(val);

        const bar = new THREE.Mesh(
          geo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: col,
              roughness: 0.3,
            })
          )
        );
        bar.scale.y = h;
        bar.position.set(x, val >= 0 ? h / 2 : -h / 2, 0);
        bar.castShadow = true;
        bar.userData = { kind: 'token', index: c } satisfies ClickTag;
        this.stageRoot.add(bar);
        this.interactives.push(bar);

        this.addLabel(
          formatToken(tokens[c].text),
          snap.showPreSoftmax
            ? formatNum(val, 2)
            : `${(val * 100).toFixed(0)}%`,
          [x, h + 0.48, 0],
          0.86,
          '#1d4ed8'
        );
      }
    }

    // Opaque Causal Mask Shutter over future tokens
    const futureCount = T - 1 - snap.queryIdx;
    if (futureCount > 0) {
      const maskW = futureCount * step;
      const maskX = -span / 2 + (snap.queryIdx + 1 + futureCount / 2) * step - step * 0.08;
      const shutter = new THREE.Mesh(
        this.trackGeo(new RoundedBoxGeometry(maskW, 2.4, 0.85, 2, 0.06)),
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: '#64748b',
            roughness: 0.6,
            metalness: 0.1,
          })
        )
      );
      shutter.position.set(maskX, 1.1, 0);
      shutter.receiveShadow = true;
      this.stageRoot.add(shutter);

      this.addLabel(
        'Causal Mask: -∞',
        'Blocks future peeking',
        [maskX, 2.7, 0],
        2.2,
        '#64748b'
      );
    }

    this.addLabel(
      snap.showPreSoftmax ? 'Pre-Softmax Logits (S)' : 'Softmax Weights (A)',
      snap.showPreSoftmax ? 'Before exponentiation' : 'Normalized to exactly 100% budget',
      [0, 2.35, -0.6],
      2.9,
      snap.showPreSoftmax ? '#b45309' : '#0f766e'
    );
  }

  // BEAT 5: Value Mixing Bowl
  private buildBeat5ValueMixer(
    snap: StageSnapshot,
    weights: number[],
    vMatrix: number[][],
    outVec: number[]
  ): void {
    const tokens = snap.traceRes.tokens;
    const qIdx = snap.queryIdx;
    const D = 32;

    const ranked = weights
      .slice(0, qIdx + 1)
      .map((w, idx) => ({ idx, w }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 3);

    const stripW = 2.4;
    const step = stripW / D;
    const barGeo = this.trackGeo(new THREE.BoxGeometry(step * 0.78, 1, 0.22));

    const drawSubVec = (vec: number[], xPos: number, zPos: number, title: string, weightPct: string) => {
      let maxAbs = 1e-5;
      for (const val of vec) maxAbs = Math.max(maxAbs, Math.abs(val));

      for (let d = 0; d < D; d++) {
        const norm = vec[d] / maxAbs;
        const h = Math.max(0.04, Math.abs(norm) * 0.9);
        const bar = new THREE.Mesh(
          barGeo,
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: divergingColor(norm),
              roughness: 0.32,
            })
          )
        );
        bar.scale.y = h;
        bar.position.set(xPos - stripW / 2 + (d + 0.5) * step, vec[d] >= 0 ? h / 2 : -h / 2, zPos);
        this.stageRoot.add(bar);
      }
      this.addLabel(title, weightPct, [xPos, 1.25, zPos], 1.65, '#0f766e');
    };

    // Up to 3 top source Value strips
    const xPositions = [-3.0, -0.6, 1.8];
    ranked.forEach((item, i) => {
      const tokText = formatToken(tokens[item.idx].text);
      drawSubVec(
        vMatrix[item.idx],
        xPositions[i] ?? 0,
        -1.3,
        `V(“${tokText}”)`,
        `× ${(item.w * 100).toFixed(0)}%`
      );
    });

    // Blended Output Vector A · V
    drawSubVec(
      outVec,
      0,
      1.2,
      `Output: A · V (${formatToken(tokens[qIdx].text)})`,
      '32-dim blended packet'
    );

    this.addLabel(
      'The Value Mixer: Copying Meaning',
      'Earlier Value vectors pour into destination token weighted by Attention %',
      [0, 2.3, 0],
      3.2,
      '#0f766e'
    );
  }

  // BEAT 6: Full Staircase Matrix & 4 Heads
  private buildBeat6MultiHead(
    snap: StageSnapshot,
    active: ActiveLayerTrace
  ): void {
    const tokens = snap.traceRes.tokens;
    const T = tokens.length;

    // 4 Head Selector Cards across the front
    const cardW = 1.6;
    const cardStep = 1.9;
    const originX = -((4 - 1) * cardStep) / 2;
    const headColors = ['#1d4ed8', '#0f766e', '#b45309', '#7c3aed'];

    for (let h = 0; h < 4; h++) {
      const isCur = h === snap.head;
      const isOff = isHeadDisabled(snap.ablation, snap.layer, h);
      const x = originX + h * cardStep;

      const card = new THREE.Mesh(
        this.trackGeo(new RoundedBoxGeometry(cardW, 0.9, 0.45, 2, 0.05)),
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: isOff ? '#fee2e2' : isCur ? headColors[h] : '#ffffff',
            roughness: 0.3,
            metalness: 0.05,
          })
        )
      );
      card.position.set(x, 0.45, 2.2);
      card.castShadow = true;
      card.userData = { kind: 'head', index: h } satisfies ClickTag;
      this.stageRoot.add(card);
      this.interactives.push(card);

      this.addLabel(
        `Head ${h}`,
        isOff ? 'Disabled' : isCur ? 'Active Head' : '32 dims',
        [x, 1.25, 2.2],
        1.55,
        isOff ? '#dc2626' : headColors[h]
      );
    }

    // Lower-triangular Attention Matrix Slab in the background
    const weights = active.attn.attnWeights[snap.head];
    const tex = this.trackTex(createTensorTexture(weights, 'sequential', true));
    const matSlab = new THREE.Mesh(
      this.trackGeo(new RoundedBoxGeometry(3.6, 2.7, 0.22, 2, 0.04)),
      this.trackMat(new THREE.MeshStandardMaterial({ map: tex, roughness: 0.32 }))
    );
    matSlab.position.set(0, 0.8, -1.1);
    matSlab.castShadow = true;
    this.stageRoot.add(matSlab);

    this.addLabel(
      `Attention Matrix [${T} × ${T}] · Head ${snap.head}`,
      'Staircase pattern: lower triangle active, gray upper triangle causally masked',
      [0, 2.65, -1.1],
      3.2,
      headColors[snap.head]
    );
  }

  private clearStage(): void {
    while (this.stageRoot.children.length > 0) {
      const c = this.stageRoot.children[0];
      this.stageRoot.remove(c);
    }
    this.interactives = [];
    this.pulseMeshes = [];

    for (const g of this.disposableGeometries) g.dispose();
    for (const m of this.disposableMaterials) m.dispose();
    for (const t of this.disposableTextures) t.dispose();

    this.disposableGeometries = [];
    this.disposableMaterials = [];
    this.disposableTextures = [];
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

  private animate = (): void => {
    if (this.disposed) return;
    this.animFrameId = requestAnimationFrame(this.animate);

    if (this.needsCameraFlight && !this.isDragging) {
      this.camera.position.lerp(this.camTargetPos, 0.055);
      this.controls.target.lerp(this.camTargetLook, 0.055);
      if (
        this.camera.position.distanceTo(this.camTargetPos) < 0.02 &&
        this.controls.target.distanceTo(this.camTargetLook) < 0.02
      ) {
        this.needsCameraFlight = false;
      }
    }

    const t = this.clock.getElapsedTime();
    for (const m of this.pulseMeshes) {
      const s = 1.0 + 0.05 * Math.sin(t * 3.2);
      m.scale.set(s, s, s);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  public dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animFrameId);
    window.removeEventListener('resize', this.onResize);
    this.renderer.domElement.removeEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.removeEventListener('click', this.onClick);
    this.clearStage();
    this.controls.dispose();
    this.envRT.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
