import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { getAnatomyNode, type NodeSceneView } from '../content/llmAnatomy';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  TokenizerData,
} from '../engine/types';
import { isHeadDisabled } from '../state/atlasState';
import {
  computeSampledDistribution,
  createPlaqueTexture,
  createTensorTexture,
  divergingColor,
  formatNum,
  formatToken,
  sequentialColor,
  vectorL2Norm,
} from '../utils/math';

export interface WorldCallbacks {
  onSelectNode: (nodeId: string, layer?: number) => void;
  onHoverToken: (index: number | null) => void;
}

export interface WorldSnapshot {
  nodeId: string;
  layer: number;
  head: number;
  showPreSoftmax: boolean;
  ablation: AblationConfig;
  baseline: EngineTraceResult;
  ablated: EngineTraceResult | null;
  tokenizer: TokenizerData;
  hoveredTokenIdx: number | null;
  temperature: number;
  topK: number;
  topP: number;
}

interface ClickTag {
  kind: 'node' | 'token';
  nodeId?: string;
  layer?: number;
  tokenIdx?: number;
}

const BLOCK_X = [-3.0, -0.7, 1.6, 3.9];

export class ExplodedWorld {
  private readonly container: HTMLElement;
  private readonly callbacks: WorldCallbacks;
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

  private camTargetPos = new THREE.Vector3(0.4, 4.8, 13.8);
  private camTargetLook = new THREE.Vector3(0.4, 0.25, 0);
  private needsCameraFlight = false;
  private isDragging = false;

  private snapshot: WorldSnapshot | null = null;
  private lastSceneView: NodeSceneView | null = null;
  private lastHoverToken: number | null = null;
  private animFrameId = 0;
  private clock = new THREE.Clock();
  private flowBead: THREE.Mesh | null = null;
  private disposed = false;

  constructor(container: HTMLElement, callbacks: WorldCallbacks) {
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
    this.scene.fog = new THREE.Fog('#f2f5fa', 22, 48);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = this.envRT.texture;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(
      36,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      90
    );
    this.camera.position.copy(this.camTargetPos);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 3.2;
    this.controls.maxDistance = 24;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.minPolarAngle = 0.14;
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
    const hemi = new THREE.HemisphereLight('#ffffff', '#dce4f0', 1.1);
    hemi.position.set(0, 14, 0);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight('#fffaf2', 1.32);
    sun.position.set(8, 15, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 2;
    sun.shadow.camera.far = 36;
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
      new THREE.PlaneGeometry(46, 36),
      new THREE.MeshStandardMaterial({
        color: '#f5f6f8',
        roughness: 0.94,
        metalness: 0.02,
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.65;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(32, 32, '#d8e0ea', '#e8edf4');
    grid.position.y = -1.64;
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
    this.flowBead = null;
    for (const t of this.disposableTextures) t.dispose();
    for (const g of this.disposableGeometries) g.dispose();
    for (const m of this.disposableMaterials) m.dispose();
    this.disposableTextures = [];
    this.disposableGeometries = [];
    this.disposableMaterials = [];
  }

  update(next: WorldSnapshot): void {
    const node = getAnatomyNode(next.nodeId);
    const prevLayer = this.snapshot?.layer;
    this.snapshot = next;

    if (this.lastSceneView !== node.sceneView || prevLayer !== next.layer) {
      this.lastSceneView = node.sceneView;
      this.setCameraForView(node.sceneView);
    }

    this.rebuild(next, node.sceneView, node.sceneFocus);
  }

  private setCameraForView(view: NodeSceneView): void {
    this.needsCameraFlight = true;
    switch (view) {
      case 'overview':
        this.camTargetPos.set(0.4, 4.9, 14.0);
        this.camTargetLook.set(0.4, 0.25, 0);
        break;
      case 'embed':
        this.camTargetPos.set(0, 3.1, 8.8);
        this.camTargetLook.set(0, 0.25, 0);
        break;
      case 'block':
        this.camTargetPos.set(0, 3.5, 9.4);
        this.camTargetLook.set(0, 0.3, 0);
        break;
      case 'attention':
        this.camTargetPos.set(0, 5.2, 8.8);
        this.camTargetLook.set(0, 0.1, 0.1);
        break;
      case 'mlp':
        this.camTargetPos.set(0, 3.4, 9.6);
        this.camTargetLook.set(0, 0.35, 0);
        break;
      case 'output':
        this.camTargetPos.set(0.4, 3.3, 9.8);
        this.camTargetLook.set(0.4, 0.5, 0);
        break;
    }
  }

  private addPlaque(
    title: string,
    subtitle: string | undefined,
    pos: [number, number, number],
    scale = 1.65,
    accent = '#1d4ed8',
    muted = false
  ): void {
    const tex = this.trackTex(
      createPlaqueTexture(title, subtitle, accent, muted)
    );
    const mat = this.trackMat(
      new THREE.SpriteMaterial({ map: tex, transparent: true })
    );
    const sp = new THREE.Sprite(mat);
    const aspect = subtitle ? 124 / 420 : 84 / 420;
    sp.scale.set(scale, scale * aspect, 1);
    sp.position.set(pos[0], pos[1], pos[2]);
    this.stageRoot.add(sp);
  }

  private addTensorPlate(
    matrix: number[][],
    size: [number, number, number],
    pos: [number, number, number],
    opts?: {
      mode?: 'diverging' | 'sequential';
      transpose?: boolean;
      tag?: ClickTag;
      frameColor?: string;
    }
  ): THREE.Mesh {
    const tex = this.trackTex(
      createTensorTexture(
        matrix,
        opts?.mode ?? 'diverging',
        opts?.transpose ?? false
      )
    );
    const geo = this.trackGeo(
      new RoundedBoxGeometry(size[0], size[1], size[2], 2, 0.035)
    );
    const sideMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: opts?.frameColor ?? '#e2e8f0',
        roughness: 0.45,
        metalness: 0.08,
      })
    );
    const faceMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        map: tex,
        roughness: 0.32,
        metalness: 0.02,
      })
    );
    const mesh = new THREE.Mesh(geo, [
      sideMat,
      sideMat,
      faceMat,
      sideMat,
      faceMat,
      faceMat,
    ]);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (opts?.tag) {
      mesh.userData = opts.tag;
      this.interactives.push(mesh);
    }
    this.stageRoot.add(mesh);
    return mesh;
  }

  private addPlusJunction(
    pos: [number, number, number],
    disabled = false
  ): void {
    const sphere = new THREE.Mesh(
      this.trackGeo(new THREE.SphereGeometry(0.17, 20, 20)),
      this.trackMat(
        new THREE.MeshStandardMaterial({
          color: disabled ? '#cbd5e1' : '#c89b54',
          metalness: 0.65,
          roughness: 0.25,
        })
      )
    );
    sphere.position.set(pos[0], pos[1], pos[2]);
    this.stageRoot.add(sphere);
  }

  private rebuild(
    snap: WorldSnapshot,
    view: NodeSceneView,
    focus?: string
  ): void {
    this.clearStage();
    const res = snap.ablated ?? snap.baseline;

    switch (view) {
      case 'overview':
        this.buildOverview(snap, res);
        break;
      case 'embed':
        this.buildEmbeddings(res, focus);
        break;
      case 'block':
        this.buildBlock(snap, res, focus);
        break;
      case 'attention':
        this.buildAttention(snap, res, focus);
        break;
      case 'mlp':
        this.buildMlp(snap, res, focus);
        break;
      case 'output':
        this.buildOutput(snap, res);
        break;
    }
  }

  private buildOverview(snap: WorldSnapshot, res: EngineTraceResult): void {
    const { trace, tokens } = res;
    const T = tokens.length;

    const highway = new THREE.Mesh(
      this.trackGeo(new RoundedBoxGeometry(12.4, 0.14, 0.52, 2, 0.04)),
      this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: '#1d4ed8',
          emissive: '#1e40af',
          emissiveIntensity: 0.2,
          roughness: 0.25,
          clearcoat: 0.8,
        })
      )
    );
    highway.position.set(0.4, -0.78, 0);
    highway.userData = { kind: 'node', nodeId: 'residual-stream' } satisfies ClickTag;
    this.stageRoot.add(highway);
    this.interactives.push(highway);

    this.flowBead = new THREE.Mesh(
      this.trackGeo(new THREE.SphereGeometry(0.11, 16, 16)),
      this.trackMat(new THREE.MeshBasicMaterial({ color: '#60a5fa' }))
    );
    this.flowBead.position.set(-5.4, -0.66, 0);
    this.stageRoot.add(this.flowBead);

    // Embeddings plate
    const h0 = trace.tokEmb.map((row, r) =>
      row.map((v, c) => v + trace.posEmb[r][c])
    );
    this.addTensorPlate(h0, [1.25, 1.35, 0.28], [-5.35, 0.25, 0], {
      tag: { kind: 'node', nodeId: 'embeddings' },
      frameColor: '#cbd5e1',
    });
    this.addPlaque('Embeddings', `wte+wpe [${T}×128]`, [-5.35, 1.48, 0], 1.65);

    // 4 Transformer Blocks
    for (let l = 0; l < 4; l++) {
      const x = BLOCK_X[l];
      const layer = trace.layers[l];
      const isSelected = snap.layer === l;
      const isSkipped = layer.skipped;

      const frame = new THREE.Mesh(
        this.trackGeo(new RoundedBoxGeometry(1.92, 2.35, 2.0, 3, 0.12)),
        this.trackMat(
          new THREE.MeshPhysicalMaterial({
            color: isSkipped ? '#ffedd5' : isSelected ? '#eff6ff' : '#ffffff',
            roughness: 0.22,
            transmission: 0.65,
            thickness: 0.4,
            clearcoat: 0.85,
            transparent: true,
            opacity: 0.85,
          })
        )
      );
      frame.position.set(x, 0.48, 0);
      frame.userData = {
        kind: 'node',
        nodeId: 'transformer-stack',
        layer: l,
      } satisfies ClickTag;
      this.stageRoot.add(frame);
      this.interactives.push(frame);

      if (!layer.skipped) {
        this.addTensorPlate(
          layer.attn.attnWeights[0],
          [0.72, 0.72, 0.16],
          [x - 0.42, 0.52, 0.22],
          {
            mode: 'sequential',
            tag: { kind: 'node', nodeId: 'attention', layer: l },
          }
        );
        this.addTensorPlate(
          layer.mlp.hidden,
          [0.72, 1.25, 0.16],
          [x + 0.42, 0.52, -0.12],
          {
            mode: 'diverging',
            tag: { kind: 'node', nodeId: 'mlp', layer: l },
          }
        );
      }

      this.addPlusJunction([x - 0.42, -0.72, 0], isSkipped);
      this.addPlusJunction([x + 0.42, -0.72, 0], isSkipped);

      const lensTop = trace.logitLens[l]?.topTokens[0];
      const lensText = lensTop
        ? `“${formatToken(snap.tokenizer.vocab[lensTop.id] ?? '')}” ${(lensTop.prob * 100).toFixed(0)}%`
        : 'skipped';
      this.addPlaque(
        `Block ${l}${isSkipped ? ' (off)' : ''}`,
        `Lens: ${lensText}`,
        [x, 2.25, 0],
        1.7,
        isSkipped ? '#b45309' : isSelected ? '#1d4ed8' : '#334155',
        isSkipped
      );
    }

    // Output Skyline
    const top5 = trace.finalProbs
      .map((prob, id) => ({ id, prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5);

    top5.forEach((item, rank) => {
      const h = Math.max(0.18, item.prob * 2.8);
      const bar = new THREE.Mesh(
        this.trackGeo(new RoundedBoxGeometry(0.24, h, 0.24, 2, 0.04)),
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: rank === 0 ? '#1d4ed8' : '#64748b',
            roughness: 0.3,
          })
        )
      );
      bar.position.set(6.05 + rank * 0.32, -0.7 + h / 2, 0);
      bar.castShadow = true;
      bar.userData = { kind: 'node', nodeId: 'output-stage' } satisfies ClickTag;
      this.stageRoot.add(bar);
      this.interactives.push(bar);
    });

    const winner = top5[0];
    this.addPlaque(
      'Output & Softmax',
      `“${formatToken(snap.tokenizer.vocab[winner.id] ?? '')}” ${(winner.prob * 100).toFixed(1)}%`,
      [6.65, 1.95, 0],
      1.85,
      '#0f766e'
    );
  }

  private buildEmbeddings(res: EngineTraceResult, focus?: string): void {
    const { trace, tokens } = res;
    const T = tokens.length;
    const h0 = trace.tokEmb.map((row, r) =>
      row.map((v, c) => v + trace.posEmb[r][c])
    );

    this.addTensorPlate(trace.tokEmb, [2.7, 2.1, 0.24], [-3.2, 0.35, 0.25], {
      tag: { kind: 'node', nodeId: 'wte' },
      frameColor: focus === 'wte' ? '#2563eb' : '#cbd5e1',
    });
    this.addPlaque('Token Embeddings', `wte[x] [${T} × 128]`, [-3.2, 1.95, 0.25], 2.0);

    this.addPlusJunction([-1.45, 0.35, 0.1]);

    this.addTensorPlate(trace.posEmb, [2.7, 2.1, 0.24], [0.25, 0.35, -0.2], {
      tag: { kind: 'node', nodeId: 'wpe' },
      frameColor: focus === 'wpe' ? '#d97706' : '#cbd5e1',
    });
    this.addPlaque(
      'Position Embeddings',
      `wpe[0..${T - 1}] [${T} × 128]`,
      [0.25, 1.95, -0.2],
      2.1,
      '#b45309'
    );

    const arrow = new THREE.Mesh(
      this.trackGeo(new THREE.CylinderGeometry(0.06, 0.06, 0.85, 16)),
      this.trackMat(new THREE.MeshStandardMaterial({ color: '#1d4ed8' }))
    );
    arrow.rotation.z = Math.PI / 2;
    arrow.position.set(2.05, 0.35, 0);
    this.stageRoot.add(arrow);

    this.addTensorPlate(h0, [2.7, 2.1, 0.28], [3.85, 0.35, 0], {
      tag: { kind: 'node', nodeId: 'embed-sum' },
      frameColor: focus === 'h0' ? '#0f766e' : '#93c5fd',
    });
    this.addPlaque(
      'Initial Residual Stream',
      `h₀ = wte + wpe [${T} × 128]`,
      [3.85, 1.95, 0],
      2.15,
      '#0f766e'
    );
  }

  private buildBlock(
    snap: WorldSnapshot,
    res: EngineTraceResult,
    focus?: string
  ): void {
    const l = snap.layer;
    const layer = res.trace.layers[l];
    const T = res.tokens.length;

    const bus = new THREE.Mesh(
      this.trackGeo(new RoundedBoxGeometry(10.2, 0.22, 0.72, 2, 0.05)),
      this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: focus === 'residual' ? '#2563eb' : '#1d4ed8',
          emissive: '#1d4ed8',
          emissiveIntensity: focus === 'residual' ? 0.55 : 0.22,
          roughness: 0.2,
        })
      )
    );
    bus.position.set(0, -0.95, 0);
    bus.userData = { kind: 'node', nodeId: 'residual-stream' } satisfies ClickTag;
    this.stageRoot.add(bus);
    this.interactives.push(bus);

    this.addPlaque(
      `Residual Stream (Block ${l})`,
      `Unbroken [${T} × 128] shared bus`,
      [0, -1.35, 0.85],
      2.4
    );

    if (layer.skipped) {
      this.addPlaque(
        `Block ${l} Disabled by Ablation`,
        'Residual stream passes straight through unchanged',
        [0, 0.65, 0],
        3.2,
        '#b45309',
        true
      );
      return;
    }

    const active: ActiveLayerTrace = layer;
    const last = T - 1;

    this.addTensorPlate(active.residualBeforeAttn, [1.25, 1.45, 0.18], [-4.3, 0.2, 0]);
    this.addPlaque(
      'h_in',
      `||h|| = ${formatNum(vectorL2Norm(active.residualBeforeAttn[last]), 2)}`,
      [-4.3, 1.32, 0],
      1.45
    );

    const ln1 = new THREE.Mesh(
      this.trackGeo(new RoundedBoxGeometry(0.3, 1.5, 0.8, 2, 0.05)),
      this.trackMat(
        new THREE.MeshStandardMaterial({
          color: focus === 'ln1' ? '#d97706' : '#c89b54',
          roughness: 0.3,
          metalness: 0.55,
        })
      )
    );
    ln1.position.set(-3.1, 0.6, 0);
    ln1.userData = { kind: 'node', nodeId: 'ln1' } satisfies ClickTag;
    this.stageRoot.add(ln1);
    this.interactives.push(ln1);
    this.addPlaque('LN₁', 'Pre-Norm', [-3.1, 1.72, 0], 1.15, '#b45309');

    for (let h = 0; h < 4; h++) {
      const off = isHeadDisabled(snap.ablation, l, h);
      this.addTensorPlate(
        active.attn.attnWeights[h],
        [1.45, 1.45, 0.1],
        [-1.55, 0.68, -0.6 + h * 0.4],
        {
          mode: 'sequential',
          tag: { kind: 'node', nodeId: 'attention' },
          frameColor: off ? '#fdba74' : h === snap.head ? '#2563eb' : '#cbd5e1',
        }
      );
    }
    this.addPlaque(
      'Multi-Head Attention',
      `4 heads × [${T}×${T}] · click to explode`,
      [-1.55, 1.95, 0.2],
      2.25,
      '#1d4ed8'
    );

    this.addPlusJunction([-0.3, -0.82, 0]);
    this.addTensorPlate(active.residualAfterAttn, [0.95, 1.35, 0.16], [0.25, 0.15, 0]);
    this.addPlaque(
      'u = h + Attn',
      `||Δ||=${formatNum(vectorL2Norm(active.attn.output[last]), 2)}`,
      [0.25, 1.22, 0],
      1.5
    );

    const ln2 = new THREE.Mesh(
      this.trackGeo(new RoundedBoxGeometry(0.3, 1.5, 0.8, 2, 0.05)),
      this.trackMat(
        new THREE.MeshStandardMaterial({
          color: focus === 'ln2' ? '#d97706' : '#c89b54',
          roughness: 0.3,
          metalness: 0.55,
        })
      )
    );
    ln2.position.set(1.25, 0.6, 0);
    ln2.userData = { kind: 'node', nodeId: 'ln2' } satisfies ClickTag;
    this.stageRoot.add(ln2);
    this.interactives.push(ln2);
    this.addPlaque('LN₂', 'Pre-Norm', [1.25, 1.72, 0], 1.15, '#b45309');

    this.addTensorPlate(active.mlp.hidden, [2.15, 1.75, 0.22], [2.85, 0.68, 0], {
      tag: { kind: 'node', nodeId: 'mlp' },
      frameColor: '#f59e0b',
    });
    this.addPlaque(
      'Feed-Forward MLP',
      '128 → 512 GELU → 128 · click to explode',
      [2.85, 2.02, 0],
      2.35,
      '#b45309'
    );

    this.addPlusJunction([4.25, -0.82, 0]);
    this.addTensorPlate(active.residualAfterMlp, [1.1, 1.45, 0.18], [4.55, 0.2, 0]);
    this.addPlaque(
      'h_out',
      `||h||=${formatNum(vectorL2Norm(active.residualAfterMlp[last]), 2)}`,
      [4.55, 1.32, 0],
      1.45,
      '#0f766e'
    );
  }

  private buildAttention(
    snap: WorldSnapshot,
    res: EngineTraceResult,
    focus?: string
  ): void {
    const l = snap.layer;
    const h = snap.head;
    const layer = res.trace.layers[l];
    const tokens = res.tokens;
    const T = tokens.length;

    if (layer.skipped) {
      this.addPlaque(
        `Block ${l} is disabled`,
        'Re-enable Block in Ablation Lab to inspect Q/K/V',
        [0, 0.5, 0],
        3.0,
        '#b45309',
        true
      );
      return;
    }

    const active: ActiveLayerTrace = layer;
    const headOff = isHeadDisabled(snap.ablation, l, h);
    const Q = active.attn.q[h];
    const K = active.attn.k[h];
    const V = active.attn.v[h];
    const AV = active.attn.weightedV[h];
    const weights = active.attn.attnWeights[h];
    const scores = active.attn.rawScores[h];

    const gridSize = 4.4;
    const cell = gridSize / T;
    const startX = -gridSize / 2;
    const startZ = -gridSize / 2;

    this.addTensorPlate(Q, [0.95, 0.14, gridSize], [startX - 0.95, -0.15, 0], {
      tag: { kind: 'node', nodeId: 'w-q' },
      frameColor: focus === 'qkv' ? '#2563eb' : '#cbd5e1',
    });
    this.addPlaque(`Q (Head ${h})`, `[${T} × 32]`, [startX - 0.95, 0.55, 0], 1.45);

    this.addTensorPlate(K, [gridSize, 0.14, 0.95], [0, -0.15, startZ - 0.95], {
      transpose: true,
      tag: { kind: 'node', nodeId: 'w-k' },
      frameColor: focus === 'qkv' ? '#2563eb' : '#cbd5e1',
    });
    this.addPlaque(`Kᵀ (Head ${h})`, `[32 × ${T}]`, [0, 0.55, startZ - 1.15], 1.55);

    let maxAbsScore = 1e-6;
    for (let r = 0; r < T; r++) {
      for (let c = 0; c <= r; c++) {
        maxAbsScore = Math.max(maxAbsScore, Math.abs(scores[r][c]));
      }
    }

    const focusIdx = snap.hoveredTokenIdx ?? T - 1;
    const boxGeo = this.trackGeo(new THREE.BoxGeometry(cell * 0.88, 1, cell * 0.88));
    const maskMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: focus === 'mask' ? '#fb923c' : '#e2e8f0',
        roughness: 0.85,
        transparent: true,
        opacity: focus === 'mask' ? 0.78 : 0.55,
      })
    );

    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        const x = startX + (c + 0.5) * cell;
        const z = startZ + (r + 0.5) * cell;
        if (c > r) {
          const m = new THREE.Mesh(boxGeo, maskMat);
          m.scale.y = focus === 'mask' ? 0.18 : 0.05;
          m.position.set(x, -0.2 + m.scale.y / 2, z);
          m.userData = { kind: 'node', nodeId: 'causal-mask' } satisfies ClickTag;
          this.stageRoot.add(m);
          this.interactives.push(m);
          continue;
        }

        const w = weights[r][c];
        const s = scores[r][c];
        const barH = snap.showPreSoftmax
          ? Math.max(0.06, (Math.abs(s) / maxAbsScore) * 0.95)
          : Math.max(0.05, w * 1.15);

        const colStr = headOff
          ? '#cbd5e1'
          : snap.showPreSoftmax
            ? divergingColor(s / maxAbsScore)
            : sequentialColor(w);

        const isCross = r === focusIdx || c === focusIdx;
        const mat = this.trackMat(
          new THREE.MeshStandardMaterial({
            color: colStr,
            emissive: isCross && !headOff ? '#1d4ed8' : '#000000',
            emissiveIntensity: isCross ? 0.18 : 0,
            roughness: 0.28,
          })
        );
        const tile = new THREE.Mesh(boxGeo, mat);
        tile.scale.y = barH;
        tile.position.set(x, -0.2 + barH / 2, z);
        tile.userData = { kind: 'token', tokenIdx: r } satisfies ClickTag;
        this.stageRoot.add(tile);
        this.interactives.push(tile);
      }
    }

    if (!headOff) {
      const qZ = startZ + (focusIdx + 0.5) * cell;
      const qPos = new THREE.Vector3(startX - 0.3, 0.22, qZ);
      for (let c = 0; c <= focusIdx; c++) {
        const w = weights[focusIdx][c];
        if (w < 0.06) continue;
        const kX = startX + (c + 0.5) * cell;
        const kPos = new THREE.Vector3(kX, 0.22, startZ - 0.3);
        const mid = qPos.clone().add(kPos).multiplyScalar(0.5);
        mid.y += 0.6 + w * 1.1;
        const curve = new THREE.CatmullRomCurve3([qPos, mid, kPos]);
        const tube = new THREE.Mesh(
          this.trackGeo(
            new THREE.TubeGeometry(curve, 20, 0.015 + w * 0.045, 8, false)
          ),
          this.trackMat(
            new THREE.MeshStandardMaterial({
              color: w > 0.35 ? '#d97706' : '#2563eb',
              emissive: w > 0.35 ? '#b45309' : '#1d4ed8',
              emissiveIntensity: 0.35,
              roughness: 0.2,
              transparent: true,
              opacity: Math.min(0.92, 0.25 + w * 0.8),
            })
          )
        );
        this.stageRoot.add(tube);
      }
    }

    this.addTensorPlate(V, [0.88, 0.14, gridSize], [gridSize / 2 + 1.0, -0.15, 0], {
      tag: { kind: 'node', nodeId: 'w-v' },
      frameColor: focus === 'qkv' ? '#d97706' : '#cbd5e1',
    });
    this.addPlaque(`V (Head ${h})`, `[${T} × 32]`, [gridSize / 2 + 1.0, 0.55, 0], 1.4, '#b45309');

    this.addTensorPlate(AV, [0.88, 0.14, gridSize], [gridSize / 2 + 2.3, -0.15, 0], {
      tag: { kind: 'node', nodeId: 'weighted-v' },
      frameColor: focus === 'av' ? '#0f766e' : '#93c5fd',
    });
    this.addPlaque(
      headOff ? 'A·V (Zeroed)' : 'Output A·V',
      `[${T} × 32]`,
      [gridSize / 2 + 2.3, 0.55, 0],
      1.55,
      headOff ? '#b45309' : '#0f766e',
      headOff
    );

    this.addPlaque(
      'Causal Mask',
      'j > i → -∞',
      [1.15, 0.38, -1.15],
      1.45,
      focus === 'mask' ? '#c2410c' : '#64748b',
      focus !== 'mask'
    );
  }

  private buildMlp(
    snap: WorldSnapshot,
    res: EngineTraceResult,
    focus?: string
  ): void {
    const l = snap.layer;
    const layer = res.trace.layers[l];
    const T = res.tokens.length;

    if (layer.skipped) {
      this.addPlaque(
        `Block ${l} is disabled`,
        'Re-enable Block in Ablation Lab to inspect MLP',
        [0, 0.5, 0],
        3.0,
        '#b45309',
        true
      );
      return;
    }

    const active: ActiveLayerTrace = layer;
    const last = T - 1;
    const lastHidden = active.mlp.hidden[last];

    this.addTensorPlate(active.ln2Out, [1.5, 1.8, 0.22], [-4.2, 0.3, 0], {
      tag: { kind: 'node', nodeId: 'w-up' },
    });
    this.addPlaque('LN₂(u) → W_up', `[${T} × 128]`, [-4.2, 1.65, 0], 1.7);

    this.addTensorPlate(active.mlp.hidden, [4.8, 2.25, 0.22], [0, 0.45, -0.4], {
      tag: { kind: 'node', nodeId: 'gelu' },
      frameColor: focus === 'gelu' ? '#d97706' : '#f59e0b',
    });
    this.addPlaque(
      'GELU Hidden Layer (4× Expansion)',
      `[${T} × 512] · click to inspect GELU`,
      [0, 2.08, -0.4],
      2.5,
      '#b45309'
    );

    const numBars = 64;
    const binSize = lastHidden.length / numBars;
    const barGeo = this.trackGeo(new THREE.BoxGeometry(0.055, 1, 0.12));
    let maxAct = 1e-6;
    for (let i = 0; i < lastHidden.length; i++) {
      maxAct = Math.max(maxAct, Math.abs(lastHidden[i]));
    }

    for (let b = 0; b < numBars; b++) {
      let peak = 0;
      for (let j = 0; j < binSize; j++) {
        const val = lastHidden[Math.floor(b * binSize + j)] ?? 0;
        if (Math.abs(val) > Math.abs(peak)) peak = val;
      }
      const h = Math.max(0.05, (Math.abs(peak) / maxAct) * 1.15);
      const bar = new THREE.Mesh(
        barGeo,
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: peak > 0.05 ? '#d97706' : '#94a3b8',
            roughness: 0.3,
          })
        )
      );
      bar.scale.y = h;
      bar.position.set(-2.35 + (b / (numBars - 1)) * 4.7, -1.15 + h / 2, 0.65);
      this.stageRoot.add(bar);
    }

    this.addPlaque(
      `Final Token “${formatToken(res.tokens[last].text)}” GELU Profile`,
      'Amber = firing neurons · Gray = suppressed',
      [0, -1.35, 1.3],
      2.65,
      '#b45309'
    );

    this.addTensorPlate(active.mlp.output, [1.5, 1.8, 0.22], [4.2, 0.3, 0], {
      tag: { kind: 'node', nodeId: 'w-down' },
      frameColor: '#93c5fd',
    });
    this.addPlaque('W_down Output', `[${T} × 128] → + Residual`, [4.2, 1.65, 0], 1.9, '#0f766e');
  }

  private buildOutput(snap: WorldSnapshot, res: EngineTraceResult): void {
    const { trace } = res;
    const isAblated = snap.ablated !== null;
    const baseProbs = snap.baseline.trace.finalProbs;

    this.addTensorPlate(trace.lnFOut, [1.5, 1.9, 0.22], [-4.5, 0.35, 0], {
      tag: { kind: 'node', nodeId: 'ln-f' },
    });
    this.addPlaque('Final LN_f', `[${res.tokens.length} × 128]`, [-4.5, 1.72, 0], 1.6);

    const activePool = computeSampledDistribution(
      trace.finalLogits,
      snap.temperature,
      snap.topK,
      snap.topP
    );
    const activeMap = new Map(activePool.map((x) => [x.id, x.prob]));

    const top10 = trace.finalProbs
      .map((rawProb, id) => ({
        id,
        rawProb,
        baseProb: baseProbs[id],
        sampledProb: activeMap.get(id) ?? 0,
        inNucleus: activeMap.has(id),
        logit: trace.finalLogits[id],
        text: formatToken(snap.tokenizer.vocab[id] ?? ''),
      }))
      .sort((a, b) => b.rawProb - a.rawProb)
      .slice(0, 10);

    top10.forEach((item, rank) => {
      const x = -2.4 + rank * 0.78;
      const probToShow = item.inNucleus ? item.sampledProb : item.rawProb;
      const h = Math.max(0.14, probToShow * 3.8);

      const bar = new THREE.Mesh(
        this.trackGeo(new RoundedBoxGeometry(0.46, h, 0.46, 2, 0.05)),
        this.trackMat(
          new THREE.MeshStandardMaterial({
            color: !item.inNucleus
              ? '#cbd5e1'
              : rank === 0
                ? isAblated
                  ? '#d97706'
                  : '#1d4ed8'
                : '#3b82f6',
            roughness: 0.28,
          })
        )
      );
      bar.position.set(x, -1.05 + h / 2, 0);
      bar.castShadow = true;
      bar.userData = { kind: 'node', nodeId: 'sampling' } satisfies ClickTag;
      this.stageRoot.add(bar);
      this.interactives.push(bar);

      if (isAblated && Math.abs(item.rawProb - item.baseProb) > 0.01) {
        const gh = Math.max(0.14, item.baseProb * 3.8);
        const ghost = new THREE.Mesh(
          this.trackGeo(new THREE.BoxGeometry(0.52, gh, 0.52)),
          this.trackMat(
            new THREE.MeshBasicMaterial({
              color: '#1d4ed8',
              wireframe: true,
              transparent: true,
              opacity: 0.45,
            })
          )
        );
        ghost.position.set(x, -1.05 + gh / 2, 0);
        this.stageRoot.add(ghost);
      }

      if (rank < 6) {
        this.addPlaque(
          `“${item.text}”`,
          item.inNucleus
            ? `${(probToShow * 100).toFixed(1)}% · z=${formatNum(item.logit, 1)}`
            : `cut · z=${formatNum(item.logit, 1)}`,
          [x, -0.75 + h + 0.35, 0.25],
          1.12,
          item.inNucleus ? '#1d4ed8' : '#64748b',
          !item.inNucleus
        );
      }
    });
  }

  private pickObject(event: PointerEvent | MouseEvent): ClickTag | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactives, false);
    for (const hit of hits) {
      const tag = hit.object.userData as ClickTag | undefined;
      if (tag?.kind) return tag;
    }
    return null;
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    const hit = this.pickObject(event);
    this.renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';
    if (hit?.kind === 'token' && typeof hit.tokenIdx === 'number') {
      if (this.lastHoverToken !== hit.tokenIdx) {
        this.lastHoverToken = hit.tokenIdx;
        this.callbacks.onHoverToken(hit.tokenIdx);
      }
    } else if (this.lastHoverToken !== null) {
      this.lastHoverToken = null;
      this.callbacks.onHoverToken(null);
    }
  };

  private readonly onClick = (event: MouseEvent): void => {
    const hit = this.pickObject(event);
    if (!hit) return;
    if (hit.kind === 'node' && hit.nodeId) {
      this.callbacks.onSelectNode(hit.nodeId, hit.layer);
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

    if (this.flowBead) {
      const frac = (t * 0.26) % 1;
      this.flowBead.position.x = -5.4 + frac * 11.4;
    }

    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animFrameId);
    window.removeEventListener('resize', this.onResize);
    this.renderer.domElement.removeEventListener('pointermove', this.onPointerMove);
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
