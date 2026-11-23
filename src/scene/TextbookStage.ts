import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { SceneBeatId } from '../content/bookData';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
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

  constructor(container: HTMLElement, callbacks: StageCallbacks) {
    this.container = container;
    this.callbacks = callbacks;

    const width = Math.max(container.clientWidth, 320);
    const height = Math.max(container.clientHeight, 240);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf6f8fc);

    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    this.camera.position.set(0, 4.2, 11);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    const envScene = new RoomEnvironment();
    this.envRT = pmrem.fromScene(envScene, 0.04);
    this.scene.environment = this.envRT.texture;
    envScene.dispose();
    pmrem.dispose();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 28;
    this.controls.target.set(0, 0.2, 0);

    this.initLighting();
    this.initGround();
    this.scene.add(this.stageRoot);

    this.renderer.domElement.addEventListener('pointermove', this.handlePointerMove);
    this.renderer.domElement.addEventListener('click', this.handleClick);

    this.animate();
  }

  private initLighting(): void {
    const sun = new THREE.DirectionalLight(0xfff7ed, 1.4);
    sun.position.set(7, 14, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 30;
    sun.shadow.camera.left = -9;
    sun.shadow.camera.right = 9;
    sun.shadow.camera.top = 9;
    sun.shadow.camera.bottom = -9;
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);

    const sky = new THREE.HemisphereLight(0xffffff, 0xe2e8f0, 0.65);
    this.scene.add(sky);
  }

  private initGround(): void {
    const geo = new THREE.PlaneGeometry(60, 60);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      roughness: 0.96,
      metalness: 0.02,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = -0.02;
    mesh.receiveShadow = true;
    this.scene.add(mesh);

    const grid = new THREE.GridHelper(40, 40, 0xd8e2ed, 0xe2e8f0);
    grid.position.y = 0;
    this.scene.add(grid);
  }

  public render(snap: StageSnapshot): void {
    this.clearDynamicScene();

    switch (snap.beat) {
      case 'attn-1-problem':
        this.renderBeat1QKV(snap);
        break;
      case 'attn-2-dot-product':
        this.renderBeat2DotProduct(snap);
        break;
      case 'attn-3-scan-row':
        this.renderBeat3ScanRow(snap);
        break;
      case 'attn-4-mask-softmax':
        this.renderBeat4MaskSoftmax(snap);
        break;
      case 'attn-5-value-mixer':
        this.renderBeat5ValueMixer(snap);
        break;
      case 'attn-6-multihead-full':
        this.renderBeat6Multihead(snap);
        break;
    }
  }

  // -------------------------------------------------------------
  // BEAT 1: Addressing Problem & Q, K, V Projections
  // -------------------------------------------------------------
  private renderBeat1QKV(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 3.8, 10.5), new THREE.Vector3(0, 0.3, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const tok = snap.traceRes.tokens[qIdx];

    const qVec = layer.attn.q[snap.head][qIdx] ?? [];
    const kVec = layer.attn.k[snap.head][qIdx] ?? [];
    const vVec = layer.attn.v[snap.head][qIdx] ?? [];

    this.createStageBanner(
      `Token ${qIdx}: "${formatToken(tok.text)}" · Head ${snap.head} Projections`,
      'One 64-d residual vector linearly splits into three 16-d search & transport roles',
      new THREE.Vector3(0, 4.3, -1)
    );

    // Render Q, K, V as three distinct columns of 16 dimensional slabs
    this.renderVectorTower(
      qVec,
      'Query (Q)',
      'What I Seek',
      new THREE.Vector3(-3.2, 0, 0),
      '#1d4ed8'
    );
    this.renderVectorTower(
      kVec,
      'Key (K)',
      'What I Advertise',
      new THREE.Vector3(0, 0, 0),
      '#059669'
    );
    this.renderVectorTower(
      vVec,
      'Value (V)',
      'What I Transport',
      new THREE.Vector3(3.2, 0, 0),
      '#d97706'
    );

    // Floor origin node showing the input residual token
    this.renderInputTokenPuck(tok.text, qIdx, new THREE.Vector3(0, 0, 3.2));
  }

  // -------------------------------------------------------------
  // BEAT 2: One Comparison — The 16-d Dot Product
  // -------------------------------------------------------------
  private renderBeat2DotProduct(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 3.5, 9.8), new THREE.Vector3(0, 0.4, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const kIdx = clampIndex(snap.keyIdx, numTokens);

    const qVec = layer.attn.q[snap.head][qIdx] ?? [];
    const kVec = layer.attn.k[snap.head][kIdx] ?? [];
    const tokQ = snap.traceRes.tokens[qIdx];
    const tokK = snap.traceRes.tokens[kIdx];

    const breakdown = computeDotProductBreakdown(qVec, kVec);

    this.createStageBanner(
      `Q("${formatToken(tokQ.text)}") · K("${formatToken(tokK.text)}")`,
      `Raw Dot = ${formatNum(breakdown.rawDot)} · Scaled (÷4) = ${formatNum(breakdown.scaledScore)}`,
      new THREE.Vector3(0, 4.2, -1)
    );

    // Render 16 horizontal product rods side-by-side
    const rodGroup = new THREE.Group();
    const rodWidth = 0.28;
    const spacing = 0.38;
    const startX = -((16 * spacing) / 2);

    for (let c = 0; c < 16; c++) {
      const prod = breakdown.products[c];
      const isPos = prod >= 0;
      const h = Math.max(0.08, Math.min(2.4, Math.abs(prod) * 1.6));

      const geo = new RoundedBoxGeometry(rodWidth, h, rodWidth, 2, 0.05);
      const mat = new THREE.MeshStandardMaterial({
        color: isPos ? 0x2563eb : 0xd97706,
        roughness: 0.3,
        metalness: 0.08,
      });
      this.trackDisposable(geo, mat);

      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.position.set(startX + c * spacing, h / 2, 0);
      rodGroup.add(mesh);
    }
    this.stageRoot.add(rodGroup);

    // Ground labels for Query and Key
    this.renderTokenPill(
      `Query [${qIdx}]: ${formatToken(tokQ.text)}`,
      new THREE.Vector3(-2.8, 0.02, 2.2),
      '#1d4ed8'
    );
    this.renderTokenPill(
      `Key [${kIdx}]: ${formatToken(tokK.text)}`,
      new THREE.Vector3(2.8, 0.02, 2.2),
      '#059669'
    );
  }

  // -------------------------------------------------------------
  // BEAT 3: Row Scan & The 1/√d Scale
  // -------------------------------------------------------------
  private renderBeat3ScanRow(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 4.2, 11.5), new THREE.Vector3(0, 0.5, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const tokQ = snap.traceRes.tokens[qIdx];
    const preRow = layer.attn.preSoftmax[snap.head][qIdx] ?? [];

    this.createStageBanner(
      `Scanning Past Words for "${formatToken(tokQ.text)}"`,
      snap.showPreSoftmax
        ? 'Showing RAW unscaled dot products (extreme variance)'
        : 'Showing SCALED dot products (divided by √16 = 4.0)',
      new THREE.Vector3(0, 4.4, -1)
    );

    const step = 0.65;
    const startX = -((numTokens * step) / 2);

    for (let j = 0; j < numTokens; j++) {
      const isMasked = j > qIdx;
      const tok = snap.traceRes.tokens[j];
      const scaledVal = preRow[j] ?? -9999;
      const rawVal = scaledVal * 4;
      const displayVal = snap.showPreSoftmax ? rawVal : scaledVal;

      const posX = startX + j * step;
      if (isMasked) {
        // Causal wall block
        const blockGeo = new RoundedBoxGeometry(0.5, 0.1, 0.5, 2, 0.04);
        const blockMat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          roughness: 0.9,
        });
        this.trackDisposable(blockGeo, blockMat);
        const block = new THREE.Mesh(blockGeo, blockMat);
        block.position.set(posX, 0.05, 0);
        this.stageRoot.add(block);
      } else {
        const h = Math.max(0.1, Math.min(3.2, Math.abs(displayVal) * 0.7));
        const barGeo = new RoundedBoxGeometry(0.48, h, 0.48, 2, 0.06);
        const barMat = new THREE.MeshStandardMaterial({
          color: displayVal >= 0 ? 0x2563eb : 0xd97706,
          roughness: 0.28,
        });
        this.trackDisposable(barGeo, barMat);
        const bar = new THREE.Mesh(barGeo, barMat);
        bar.castShadow = true;
        bar.position.set(posX, h / 2, 0);
        this.stageRoot.add(bar);
      }

      // Token label on floor
      this.renderFloorToken(tok.text, j, j === qIdx, new THREE.Vector3(posX, 0, 1.4));
    }
  }

  // -------------------------------------------------------------
  // BEAT 4: Causal Mask & Softmax (100% Probability Budget)
  // -------------------------------------------------------------
  private renderBeat4MaskSoftmax(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 6.5, 9.2), new THREE.Vector3(0, 0, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const probs = layer.attn.probs[snap.head];
    const pre = layer.attn.preSoftmax[snap.head];

    this.createStageBanner(
      `Full Attention Matrix · Head ${snap.head}`,
      snap.showPreSoftmax
        ? 'Displaying Pre-Softmax Logits (dark slate = -∞ causal wall)'
        : 'Displaying Softmax Attention % (each horizontal row sums to exactly 100%)',
      new THREE.Vector3(0, 4.6, -3.8)
    );

    const cellSize = 0.52;
    const originX = -((numTokens * cellSize) / 2);
    const originZ = -((numTokens * cellSize) / 2);

    for (let r = 0; r < numTokens; r++) {
      for (let c = 0; c < numTokens; c++) {
        const isFuture = c > r;
        const posX = originX + c * cellSize;
        const posZ = originZ + r * cellSize;

        if (isFuture) {
          // Future tile: dark slate tile with -inf
          const tileGeo = new RoundedBoxGeometry(cellSize * 0.9, 0.06, cellSize * 0.9, 2, 0.04);
          const tileMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            roughness: 0.95,
          });
          this.trackDisposable(tileGeo, tileMat);
          const tile = new THREE.Mesh(tileGeo, tileMat);
          tile.position.set(posX, 0.03, posZ);
          this.stageRoot.add(tile);
        } else {
          const p = probs[r]?.[c] ?? 0;
          const raw = pre[r]?.[c] ?? 0;
          const val = snap.showPreSoftmax ? raw : p;
          const h = snap.showPreSoftmax
            ? Math.max(0.08, Math.min(1.8, Math.abs(val) * 0.4))
            : Math.max(0.06, p * 2.8);

          const color = snap.showPreSoftmax
            ? new THREE.Color(divergingColor(val / 4))
            : new THREE.Color(sequentialColor(p));

          const tileGeo = new RoundedBoxGeometry(cellSize * 0.9, h, cellSize * 0.9, 2, 0.05);
          const tileMat = new THREE.MeshStandardMaterial({
            color,
            roughness: 0.25,
            metalness: 0.05,
          });
          this.trackDisposable(tileGeo, tileMat);

          const tile = new THREE.Mesh(tileGeo, tileMat);
          tile.castShadow = true;
          tile.receiveShadow = true;
          tile.position.set(posX, h / 2, posZ);

          // Highlight the currently selected query row with a glowing perimeter
          if (r === qIdx) {
            const edgeGeo = new THREE.EdgesGeometry(tileGeo);
            const edgeMat = new THREE.LineBasicMaterial({
              color: 0x3b82f6,
              linewidth: 2,
            });
            this.trackDisposable(edgeGeo, edgeMat);
            const wire = new THREE.LineSegments(edgeGeo, edgeMat);
            wire.position.copy(tile.position);
            this.stageRoot.add(wire);
          }

          this.stageRoot.add(tile);
        }
      }
    }
  }

  // -------------------------------------------------------------
  // BEAT 5: Value Mixing — Blending Slices into a New Idea
  // -------------------------------------------------------------
  private renderBeat5ValueMixer(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 4.2, 10.8), new THREE.Vector3(0, 0.5, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const tokQ = snap.traceRes.tokens[qIdx];

    const rowProbs = layer.attn.probs[snap.head][qIdx] ?? [];
    const headOut = layer.attn.out[snap.head][qIdx] ?? [];

    this.createStageBanner(
      `Value Mixing for "${formatToken(tokQ.text)}"`,
      'Attention % pulls weighted slices of past Value vectors into this new 16-d vector',
      new THREE.Vector3(0, 4.4, -1)
    );

    // Left side: past tokens and their attention transfer beams
    const step = 0.58;
    const startX = -((Math.min(qIdx + 1, 10) * step) / 2) - 1.5;

    for (let j = 0; j <= qIdx; j++) {
      if (j > 10) break; // keep visible range tidy
      const p = rowProbs[j] ?? 0;
      const tok = snap.traceRes.tokens[j];
      const posX = startX + j * step;

      const beamH = Math.max(0.1, p * 3.4);
      const beamGeo = new RoundedBoxGeometry(0.42, beamH, 0.42, 2, 0.06);
      const beamMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(sequentialColor(p)),
        roughness: 0.2,
      });
      this.trackDisposable(beamGeo, beamMat);
      const beam = new THREE.Mesh(beamGeo, beamMat);
      beam.castShadow = true;
      beam.position.set(posX, beamH / 2, 0.5);
      this.stageRoot.add(beam);

      this.renderFloorToken(tok.text, j, j === qIdx, new THREE.Vector3(posX, 0, 1.8));
    }

    // Right side: resulting 16-d output vector column
    this.renderVectorTower(
      headOut,
      `Output y_${qIdx}`,
      `Head ${snap.head} Context (16-d)`,
      new THREE.Vector3(3.8, 0, 0.5),
      '#0284c7'
    );
  }

  // -------------------------------------------------------------
  // BEAT 6: 4-Head Orchestra & Output Projection (W_o)
  // -------------------------------------------------------------
  private renderBeat6Multihead(snap: StageSnapshot): void {
    const layer = snap.traceRes.layers[snap.layer];
    if (!layer) return;

    this.setCameraTarget(new THREE.Vector3(0, 4.0, 11.5), new THREE.Vector3(0, 0.6, 0));

    const numTokens = snap.traceRes.tokens.length;
    const qIdx = clampIndex(snap.queryIdx, numTokens);
    const tokQ = snap.traceRes.tokens[qIdx];

    this.createStageBanner(
      `4-Head Ensemble for "${formatToken(tokQ.text)}"`,
      'Four 16-d heads concatenate into 64-d and mix through Output Matrix W_o',
      new THREE.Vector3(0, 4.4, -1.5)
    );

    const headColors = ['#2563eb', '#059669', '#d97706', '#7c3aed'];
    const headNames = ['Head 0 (Gender/Subject)', 'Head 1 (Punctuation)', 'Head 2 (Verb/Action)', 'Head 3 (Thematic Target)'];

    for (let h = 0; h < 4; h++) {
      const isOff = isHeadDisabled(snap.ablation, snap.layer, h);
      const outVec = layer.attn.out[h][qIdx] ?? [];
      const posX = -4.5 + h * 3.0;

      this.renderVectorTower(
        outVec,
        `H${h} ${isOff ? '(Ablated)' : ''}`,
        headNames[h],
        new THREE.Vector3(posX, 0, 0),
        isOff ? '#94a3b8' : headColors[h],
        isOff
      );
    }
  }

  // -------------------------------------------------------------
  // Stage Utilities & Primitives
  // -------------------------------------------------------------
  private renderVectorTower(
    vec: number[],
    title: string,
    subtitle: string,
    origin: THREE.Vector3,
    colorHex: string,
    muted = false
  ): void {
    const group = new THREE.Group();
    group.position.copy(origin);

    const titleTex = createLabelSpriteTexture(title, subtitle, colorHex, muted);
    this.trackDisposable(titleTex);
    const titleMat = new THREE.SpriteMaterial({ map: titleTex, transparent: true });
    this.trackDisposable(titleMat);
    const sprite = new THREE.Sprite(titleMat);
    sprite.scale.set(2.4, 0.7, 1);
    sprite.position.set(0, 3.4, 0);
    group.add(sprite);

    // 16 slabs arranged vertically
    const slabW = 1.1;
    const slabD = 0.55;
    const slabH = 0.14;
    const gap = 0.04;

    for (let i = 0; i < 16; i++) {
      const val = vec[i] ?? 0;
      const col = muted
        ? new THREE.Color(0xcbd5e1)
        : new THREE.Color(divergingColor(val / 2.5));

      const geo = new RoundedBoxGeometry(slabW, slabH, slabD, 2, 0.03);
      const mat = new THREE.MeshStandardMaterial({
        color: col,
        roughness: 0.24,
        metalness: 0.08,
      });
      this.trackDisposable(geo, mat);

      const slab = new THREE.Mesh(geo, mat);
      slab.castShadow = true;
      slab.receiveShadow = true;
      slab.position.set(0, 0.1 + i * (slabH + gap), 0);
      group.add(slab);
    }

    this.stageRoot.add(group);
  }

  private renderFloorToken(text: string, index: number, isQuery: boolean, pos: THREE.Vector3): void {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 70;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = isQuery ? '#1d4ed8' : '#ffffff';
    ctx.roundRect(4, 4, 152, 62, 10);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = isQuery ? '#1d4ed8' : '#cbd5e1';
    ctx.stroke();

    ctx.fillStyle = isQuery ? '#ffffff' : '#0f172a';
    ctx.font = '600 24px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(formatToken(text).slice(0, 8), 80, 35);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.trackDisposable(tex);

    const geo = new THREE.PlaneGeometry(0.5, 0.22);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
    this.trackDisposable(geo, mat);

    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(pos);
    mesh.position.y = 0.02;

    (mesh as unknown as { clickTag: ClickTag }).clickTag = { kind: 'token', index };
    this.interactives.push(mesh);
    this.stageRoot.add(mesh);
  }

  private renderTokenPill(text: string, pos: THREE.Vector3, colorHex: string): void {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 70;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = colorHex;
    ctx.roundRect(4, 4, 292, 62, 12);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 24px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text.slice(0, 20), 150, 35);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.trackDisposable(tex);

    const geo = new THREE.PlaneGeometry(1.2, 0.28);
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
    this.trackDisposable(geo, mat);

    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(pos);
    mesh.position.y = 0.02;
    this.stageRoot.add(mesh);
  }

  private renderInputTokenPuck(text: string, idx: number, pos: THREE.Vector3): void {
    const puckGeo = new THREE.CylinderGeometry(0.65, 0.65, 0.18, 32);
    const puckMat = new THREE.MeshStandardMaterial({
      color: 0x1d4ed8,
      roughness: 0.3,
    });
    this.trackDisposable(puckGeo, puckMat);

    const mesh = new THREE.Mesh(puckGeo, puckMat);
    mesh.position.copy(pos);
    mesh.position.y = 0.09;
    mesh.castShadow = true;
    this.stageRoot.add(mesh);

    const labelTex = createLabelSpriteTexture(`Input: "${formatToken(text)}"`, `Token [${idx}]`, '#1d4ed8');
    this.trackDisposable(labelTex);
    const spriteMat = new THREE.SpriteMaterial({ map: labelTex, transparent: true });
    this.trackDisposable(spriteMat);
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.0, 0.6, 1);
    sprite.position.copy(pos);
    sprite.position.y = 0.65;
    this.stageRoot.add(sprite);
  }

  private createStageBanner(primary: string, secondary: string, pos: THREE.Vector3): void {
    const tex = createLabelSpriteTexture(primary, secondary, '#1d4ed8');
    this.trackDisposable(tex);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    this.trackDisposable(mat);
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(4.0, 1.15, 1);
    sprite.position.copy(pos);
    this.stageRoot.add(sprite);
  }

  private setCameraTarget(pos: THREE.Vector3, look: THREE.Vector3): void {
    this.camTargetPos.copy(pos);
    this.camTargetLook.copy(look);
  }

  public resetCamera(): void {
    this.camera.position.set(0, 4.2, 11);
    this.controls.target.set(0, 0.2, 0);
    this.controls.update();
  }

  public resize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private clearDynamicScene(): void {
    this.interactives = [];

    while (this.stageRoot.children.length > 0) {
      const child = this.stageRoot.children[0];
      this.stageRoot.remove(child);
    }

    for (const g of this.disposableGeometries) g.dispose();
    this.disposableGeometries = [];

    for (const m of this.disposableMaterials) m.dispose();
    this.disposableMaterials = [];

    for (const t of this.disposableTextures) t.dispose();
    this.disposableTextures = [];
  }

  private trackDisposable(
    ...items: (THREE.BufferGeometry | THREE.Material | THREE.Texture | undefined)[]
  ): void {
    for (const item of items) {
      if (!item) continue;
      if (item instanceof THREE.BufferGeometry) this.disposableGeometries.push(item);
      else if (item instanceof THREE.Material) this.disposableMaterials.push(item);
      else if (item instanceof THREE.Texture) this.disposableTextures.push(item);
    }
  }

  private handlePointerMove = (e: MouseEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  };

  private handleClick = (): void => {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactives, true);
    if (hits.length === 0) return;

    const hit = hits[0].object as unknown as { clickTag?: ClickTag };
    if (!hit.clickTag) return;

    if (hit.clickTag.kind === 'token') {
      this.callbacks.onPickToken(hit.clickTag.index);
    } else if (hit.clickTag.kind === 'head') {
      this.callbacks.onPickHead(hit.clickTag.index);
    }
  };

  private animate = (): void => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  public dispose(): void {
    this.renderer.domElement.removeEventListener('pointermove', this.handlePointerMove);
    this.renderer.domElement.removeEventListener('click', this.handleClick);
    this.clearDynamicScene();
    this.envRT.dispose();
    this.renderer.dispose();
    this.controls.dispose();
  }
}

function clampIndex(idx: number, len: number): number {
  if (len <= 0) return 0;
  if (!Number.isFinite(idx)) return 0;
  return Math.max(0, Math.min(len - 1, Math.round(idx)));
}
