import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneBeatId } from '../content/bookData';
import type { EngineTraceResult, TokenizerData } from '../engine/types';
import type { AblationConfig } from '../state/bookState';
import {
  computeDotProductBreakdown,
  createLabelSpriteTexture,
  divergingColor,
  formatNum,
  formatToken,
  sequentialColor,
} from '../utils/math';
import {
  type BpeStepSnapshot,
  type PcaCloudData,
} from '../utils/pass2Math';
import { buildChapter123Beat } from './pass2Builders';

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

export class TextbookStage {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private animId: number | null = null;
  private stageRoot: THREE.Group;
  private currentSnapshot: StageSnapshot | null = null;
  private targetCameraPos = new THREE.Vector3(0, 4.5, 9.0);
  private targetLookAt = new THREE.Vector3(0, 1.0, 0);

  constructor(container: HTMLElement) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf8fafc);

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 4.5, 9.0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
    this.controls.minDistance = 2.0;
    this.controls.maxDistance = 28.0;

    this.stageRoot = new THREE.Group();
    this.scene.add(this.stageRoot);

    this.setupLighting();
    this.setupEnvironment();
    this.animate = this.animate.bind(this);
    this.animate();
  }

  private setupLighting(): void {
    const ambient = new THREE.AmbientLight(0xffffff, 0.72);
    this.scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xfffbeb, 1.35);
    sun.position.set(6, 12, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 1024;
    sun.shadow.mapSize.height = 1024;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 30;
    sun.shadow.camera.left = -10;
    sun.shadow.camera.right = 10;
    sun.shadow.camera.top = 10;
    sun.shadow.camera.bottom = -10;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);

    const rim = new THREE.DirectionalLight(0xdbeafe, 0.45);
    rim.position.set(-6, 4, -8);
    this.scene.add(rim);
  }

  private setupEnvironment(): void {
    const floorGeom = new THREE.PlaneGeometry(60, 60);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      roughness: 0.9,
      metalness: 0.05,
    });
    const floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.02;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(40, 40, 0xcfd8dc, 0xe2e8f0);
    grid.position.y = 0;
    this.scene.add(grid);
  }

  public update(snapshot: StageSnapshot): void {
    const prevBeat = this.currentSnapshot?.beat;
    this.currentSnapshot = snapshot;

    if (prevBeat !== snapshot.beat) {
      this.repositionCameraForBeat(snapshot.beat);
    }

    this.rebuildStage(snapshot);
  }

  private repositionCameraForBeat(beat: SceneBeatId): void {
    switch (beat) {
      case 'ch1-1-extremes':
        this.targetCameraPos.set(0, 4.2, 7.8);
        this.targetLookAt.set(0, 0.8, -0.5);
        break;
      case 'ch1-2-bpe-merges':
        this.targetCameraPos.set(0, 3.8, 7.5);
        this.targetLookAt.set(0, 1.0, 0);
        break;
      case 'ch1-3-spaces-blindspot':
        this.targetCameraPos.set(0, 3.6, 6.8);
        this.targetLookAt.set(0, 1.0, 0);
        break;
      case 'ch1-4-token-ids':
        this.targetCameraPos.set(0, 4.0, 7.8);
        this.targetLookAt.set(0, 1.0, 0);
        break;
      case 'ch1-5-context-window':
        this.targetCameraPos.set(0, 5.5, 6.5);
        this.targetLookAt.set(0, 0.5, 0);
        break;
      case 'ch2-1-wte-lookup':
        this.targetCameraPos.set(0, 3.8, 6.8);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'ch2-2-pca-cloud':
        this.targetCameraPos.set(4.2, 4.5, 6.2);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'ch2-3-order-blind':
        this.targetCameraPos.set(0, 5.2, 6.2);
        this.targetLookAt.set(0, 0.8, 0);
        break;
      case 'ch2-4-wpe-add':
        this.targetCameraPos.set(0, 3.6, 6.5);
        this.targetLookAt.set(0, 1.1, 0);
        break;
      case 'ch2-5-rope':
        this.targetCameraPos.set(0, 3.4, 6.5);
        this.targetLookAt.set(0, 1.0, 0);
        break;
      case 'ch3-1-residual-bus':
        this.targetCameraPos.set(0, 4.6, 8.8);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'ch3-2-cut-highway':
        this.targetCameraPos.set(0, 4.0, 8.0);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'ch3-3-layernorm':
        this.targetCameraPos.set(0, 3.8, 7.2);
        this.targetLookAt.set(0, 0.8, 0);
        break;
      case 'ch3-4-rmsnorm':
        this.targetCameraPos.set(0, 3.6, 7.0);
        this.targetLookAt.set(0, 1.0, 0);
        break;
      case 'attn-1-problem':
        this.targetCameraPos.set(0, 3.2, 6.5);
        this.targetLookAt.set(0, 0.8, 0);
        break;
      case 'attn-2-dot-product':
        this.targetCameraPos.set(0, 3.5, 6.8);
        this.targetLookAt.set(0, 1.1, 0);
        break;
      case 'attn-3-scan-row':
        this.targetCameraPos.set(0, 4.2, 7.5);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'attn-4-mask-softmax':
        this.targetCameraPos.set(0, 5.8, 6.8);
        this.targetLookAt.set(0, 0.4, 0);
        break;
      case 'attn-5-value-mixer':
        this.targetCameraPos.set(0, 4.2, 7.8);
        this.targetLookAt.set(0, 1.2, 0);
        break;
      case 'attn-6-multihead-full':
        this.targetCameraPos.set(0, 5.2, 9.2);
        this.targetLookAt.set(0, 1.2, 0);
        break;
    }
  }

  private rebuildStage(snapshot: StageSnapshot): void {
    while (this.stageRoot.children.length > 0) {
      const child = this.stageRoot.children[0];
      this.disposeObject(child);
      this.stageRoot.remove(child);
    }

    // Try Chapter 1, 2, 3 builders first
    const handled = buildChapter123Beat({
      group: this.stageRoot,
      beat: snapshot.beat,
      layer: snapshot.layer,
      head: snapshot.head,
      queryIdx: snapshot.queryIdx,
      keyIdx: snapshot.keyIdx,
      showPreSoftmax: snapshot.showPreSoftmax,
      ablation: snapshot.ablation,
      traceRes: snapshot.traceRes,
      baselineRes: snapshot.baselineRes,
      tokenizer: snapshot.tokenizer,
      wteWeights: snapshot.wteWeights,
      pcaCloud: snapshot.pcaCloud,
      bpeSteps: snapshot.bpeSteps,
      bpeStepIdx: snapshot.bpeStepIdx,
      ropeSlotA: snapshot.ropeSlotA,
      ropeSlotB: snapshot.ropeSlotB,
      makeLabelSprite: this.makeLabelSprite.bind(this),
      defaultMat: new THREE.MeshStandardMaterial({ color: 0x94a3b8 }),
    });

    if (handled) return;

    // Fall back to flagship Chapter 4 builders
    switch (snapshot.beat) {
      case 'attn-1-problem':
        this.buildBeat1Problem(snapshot);
        break;
      case 'attn-2-dot-product':
        this.buildBeat2DotProduct(snapshot);
        break;
      case 'attn-3-scan-row':
        this.buildBeat3ScanRow(snapshot);
        break;
      case 'attn-4-mask-softmax':
        this.buildBeat4MaskSoftmax(snapshot);
        break;
      case 'attn-5-value-mixer':
        this.buildBeat5ValueMixer(snapshot);
        break;
      case 'attn-6-multihead-full':
        this.buildBeat6MultiHead(snapshot);
        break;
    }
  }

  private buildBeat1Problem(s: StageSnapshot): void {
    const tokens = s.traceRes.tokens;
    const count = tokens.length;
    const spacing = 1.05;
    const startX = -((count - 1) * spacing) / 2;

    const title = this.makeLabelSprite(
      'Tokens in Isolation — Without Self-Attention, each word has only its dictionary embedding',
      '#2563eb'
    );
    title.position.set(0, 2.8, 0);
    title.scale.set(8.5, 0.85, 1);
    this.stageRoot.add(title);

    for (let i = 0; i < count; i++) {
      const isQuery = i === s.queryIdx;
      const x = startX + i * spacing;

      const geom = new THREE.BoxGeometry(0.85, 0.55, 0.35);
      const mat = new THREE.MeshStandardMaterial({
        color: isQuery ? 0x2563eb : 0xffffff,
        roughness: 0.3,
        metalness: 0.05,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, 0.6, 0);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.stageRoot.add(mesh);

      const label = this.makeLabelSprite(
        formatToken(tokens[i]),
        isQuery ? '#ffffff' : '#0f172a',
        isQuery ? '#1e40af' : '#e2e8f0'
      );
      label.position.set(x, 1.2, 0);
      label.scale.set(1.1, 0.5, 1);
      this.stageRoot.add(label);

      if (isQuery) {
        const queryTag = this.makeLabelSprite(
          'Target Query Token ("Who am I?")',
          '#2563eb',
          '#dbeafe'
        );
        queryTag.position.set(x, 1.95, 0);
        queryTag.scale.set(2.2, 0.55, 1);
        this.stageRoot.add(queryTag);
      }
    }
  }

  private buildBeat2DotProduct(s: StageSnapshot): void {
    const qIdx = s.queryIdx;
    const kIdx = s.keyIdx;
    const tokens = s.traceRes.tokens;
    const qToken = tokens[qIdx] ?? 'He';
    const kToken = tokens[kIdx] ?? 'Max';

    const title = this.makeLabelSprite(
      `Pairwise Dot Product: Query("${formatToken(qToken)}") · Key("${formatToken(kToken)}")`,
      '#2563eb'
    );
    title.position.set(0, 3.4, 0);
    title.scale.set(7.5, 0.75, 1);
    this.stageRoot.add(title);

    const qBlock = this.createVectorPillar(
      `Query (He) [d_k=32]`,
      -2.4,
      0x2563eb,
      s.traceRes.q?.[s.layer]?.[s.head]?.[qIdx] ?? []
    );
    this.stageRoot.add(qBlock);

    const kBlock = this.createVectorPillar(
      `Key (Max) [d_k=32]`,
      2.4,
      0x059669,
      s.traceRes.k?.[s.layer]?.[s.head]?.[kIdx] ?? []
    );
    this.stageRoot.add(kBlock);

    const qVec = s.traceRes.q?.[s.layer]?.[s.head]?.[qIdx] ?? [];
    const kVec = s.traceRes.k?.[s.layer]?.[s.head]?.[kIdx] ?? [];
    const b = computeDotProductBreakdown(qVec, kVec, 32);

    const bridgeGeom = new THREE.CylinderGeometry(0.06, 0.06, 4.6, 16);
    const bridgeMat = new THREE.MeshStandardMaterial({
      color: 0xd97706,
      emissive: 0xd97706,
      emissiveIntensity: 0.4,
    });
    const bridge = new THREE.Mesh(bridgeGeom, bridgeMat);
    bridge.rotation.z = Math.PI / 2;
    bridge.position.set(0, 1.2, 0);
    this.stageRoot.add(bridge);

    const bridgeCard = this.makeLabelSprite(
      `q·k = ${formatNum(b.rawDot, 2)}  ÷ 5.657 = ${formatNum(b.scaledScore, 2)}`,
      '#92400e',
      '#fef3c7'
    );
    bridgeCard.position.set(0, 1.75, 0);
    bridgeCard.scale.set(3.8, 0.65, 1);
    this.stageRoot.add(bridgeCard);
  }

  private buildBeat3ScanRow(s: StageSnapshot): void {
    const tokens = s.traceRes.tokens;
    const count = tokens.length;
    const qIdx = s.queryIdx;
    const spacing = 1.0;
    const startX = -((count - 1) * spacing) / 2;

    const row =
      s.traceRes.attention?.[s.layer]?.[s.head]?.[qIdx] ??
      new Array(count).fill(0);

    const title = this.makeLabelSprite(
      `Query "${formatToken(tokens[qIdx])}" scanning backward across all available keys`,
      '#2563eb'
    );
    title.position.set(0, 3.4, 0);
    title.scale.set(7.5, 0.75, 1);
    this.stageRoot.add(title);

    for (let k = 0; k < count; k++) {
      const x = startX + k * spacing;
      const weight = row[k] ?? 0;
      const isFuture = k > qIdx;
      const barH = isFuture ? 0.05 : Math.max(0.08, weight * 3.5);

      const geom = new THREE.BoxGeometry(0.75, barH, 0.35);
      const color = isFuture
        ? 0x94a3b8
        : new THREE.Color(sequentialColor(weight)).getHex();
      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.3,
        metalness: 0.1,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, barH / 2, 0);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.stageRoot.add(mesh);

      const tokLabel = this.makeLabelSprite(
        formatToken(tokens[k]),
        isFuture ? '#94a3b8' : '#0f172a'
      );
      tokLabel.position.set(x, -0.4, 0);
      tokLabel.scale.set(0.9, 0.45, 1);
      this.stageRoot.add(tokLabel);

      if (!isFuture) {
        const valLabel = this.makeLabelSprite(
          `${Math.round(weight * 100)}%`,
          '#1e3a8a',
          'transparent'
        );
        valLabel.position.set(x, barH + 0.35, 0);
        valLabel.scale.set(0.75, 0.35, 1);
        this.stageRoot.add(valLabel);
      }
    }
  }

  private buildBeat4MaskSoftmax(s: StageSnapshot): void {
    const tokens = s.traceRes.tokens;
    const T = tokens.length;
    const matrix = s.traceRes.attention?.[s.layer]?.[s.head] ?? [];
    const cellSize = 0.52;
    const startX = -((T - 1) * cellSize) / 2;
    const startZ = -((T - 1) * cellSize) / 2;

    const title = this.makeLabelSprite(
      `Causal Attention Matrix [${T}×${T}] — Upper triangle masked with -∞`,
      '#2563eb'
    );
    title.position.set(0, 3.6, -1.8);
    title.scale.set(7.5, 0.75, 1);
    this.stageRoot.add(title);

    for (let q = 0; q < T; q++) {
      for (let k = 0; k < T; k++) {
        const x = startX + k * cellSize;
        const z = startZ + q * cellSize;
        const isFuture = k > q;
        const weight = matrix[q]?.[k] ?? 0;

        const cellGeom = new THREE.BoxGeometry(
          cellSize * 0.88,
          isFuture ? 0.05 : Math.max(0.08, weight * 1.8),
          cellSize * 0.88
        );
        const color = isFuture
          ? 0x475569
          : new THREE.Color(sequentialColor(weight)).getHex();
        const cellMat = new THREE.MeshStandardMaterial({
          color,
          roughness: 0.4,
          metalness: 0.1,
        });
        const cell = new THREE.Mesh(cellGeom, cellMat);
        const h = isFuture ? 0.025 : Math.max(0.04, (weight * 1.8) / 2);
        cell.position.set(x, h, z);
        cell.castShadow = !isFuture;
        cell.receiveShadow = true;
        this.stageRoot.add(cell);
      }
    }
  }

  private buildBeat5ValueMixer(s: StageSnapshot): void {
    const tokens = s.traceRes.tokens;
    const count = tokens.length;
    const qIdx = s.queryIdx;
    const row =
      s.traceRes.attention?.[s.layer]?.[s.head]?.[qIdx] ??
      new Array(count).fill(0);
    const spacing = 1.05;
    const startX = -((count - 1) * spacing) / 2;

    const title = this.makeLabelSprite(
      `Value Mixing: Out = ∑ (Attention% × Value Vector)`,
      '#2563eb'
    );
    title.position.set(0, 3.4, 0);
    title.scale.set(6.8, 0.7, 1);
    this.stageRoot.add(title);

    for (let k = 0; k <= qIdx; k++) {
      const x = startX + k * spacing;
      const w = row[k] ?? 0;

      const vPillar = this.createMiniPillar(
        formatToken(tokens[k]),
        x,
        -1.5,
        0x7c3aed,
        Math.max(0.3, w * 3.0)
      );
      this.stageRoot.add(vPillar);

      if (w > 0.04) {
        const lineGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, 0.8, -1.5),
          new THREE.Vector3(0, 1.2, 1.5),
        ]);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x7c3aed,
          transparent: true,
          opacity: Math.min(1.0, w * 2.2),
          linewidth: 2,
        });
        const line = new THREE.Line(lineGeom, lineMat);
        this.stageRoot.add(line);
      }
    }

    const outPillar = this.createMiniPillar(
      `Mixed Output for "${formatToken(tokens[qIdx])}"`,
      0,
      1.5,
      0x2563eb,
      2.0
    );
    this.stageRoot.add(outPillar);
  }

  private buildBeat6MultiHead(s: StageSnapshot): void {
    const heads = [
      { id: 0, label: 'Head 0: Previous-Token', color: 0x0284c7 },
      { id: 1, label: 'Head 1: Prefix / Clause', color: 0x059669 },
      { id: 2, label: 'Head 2: Verb / Action', color: 0x7c3aed },
      { id: 3, label: 'Head 3: Pronoun / Max', color: 0xd97706 },
    ];

    const title = this.makeLabelSprite(
      'Multi-Head Self-Attention: 4 specialized heads project to residual stream',
      '#2563eb'
    );
    title.position.set(0, 3.6, 0);
    title.scale.set(7.5, 0.75, 1);
    this.stageRoot.add(title);

    const spacing = 2.1;
    const startX = -((heads.length - 1) * spacing) / 2;

    for (let h = 0; h < heads.length; h++) {
      const info = heads[h];
      const x = startX + h * spacing;
      const isSelected = h === s.head;

      const group = new THREE.Group();
      group.position.set(x, 0, 0);

      const geom = new THREE.BoxGeometry(1.6, 2.2, 0.4);
      const mat = new THREE.MeshStandardMaterial({
        color: info.color,
        roughness: 0.35,
        metalness: 0.1,
        wireframe: !isSelected,
      });
      const box = new THREE.Mesh(geom, mat);
      box.position.y = 1.1;
      box.castShadow = true;
      box.receiveShadow = true;
      group.add(box);

      const lbl = this.makeLabelSprite(
        info.label,
        isSelected ? '#ffffff' : '#0f172a',
        isSelected ? '#1e3a8a' : '#f1f5f9'
      );
      lbl.position.set(0, 2.5, 0);
      lbl.scale.set(1.9, 0.5, 1);
      group.add(lbl);

      const subLbl = this.makeLabelSprite(
        `32 dims → Concat(H₀₋₃) @ W_O`,
        '#64748b'
      );
      subLbl.position.set(0, -0.3, 0);
      subLbl.scale.set(1.7, 0.4, 1);
      group.add(subLbl);

      this.stageRoot.add(group);
    }
  }

  private createVectorPillar(
    title: string,
    x: number,
    color: number,
    vals: number[]
  ): THREE.Group {
    const g = new THREE.Group();
    g.position.set(x, 0, 0);

    const geom = new THREE.BoxGeometry(1.4, 2.4, 0.5);
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.3,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.y = 1.2;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    g.add(mesh);

    const lbl = this.makeLabelSprite(title, '#ffffff', '#0f172a');
    lbl.position.set(0, 2.7, 0);
    lbl.scale.set(2.0, 0.5, 1);
    g.add(lbl);

    return g;
  }

  private createMiniPillar(
    title: string,
    x: number,
    z: number,
    color: number,
    height: number
  ): THREE.Group {
    const g = new THREE.Group();
    g.position.set(x, 0, z);

    const geom = new THREE.BoxGeometry(0.75, height, 0.35);
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.3,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.y = height / 2;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    g.add(mesh);

    const lbl = this.makeLabelSprite(title, '#0f172a');
    lbl.position.set(0, height + 0.35, 0);
    lbl.scale.set(1.2, 0.45, 1);
    g.add(lbl);

    return g;
  }

  private makeLabelSprite(
    text: string,
    textColor = '#0f172a',
    bgColor = '#ffffff'
  ): THREE.Sprite {
    const tex = createLabelSpriteTexture(text, textColor, bgColor);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    return sprite;
  }

  private disposeObject(obj: THREE.Object3D): void {
    if ((obj as THREE.Mesh).geometry) {
      (obj as THREE.Mesh).geometry.dispose();
    }
    if ((obj as THREE.Mesh).material) {
      const mat = (obj as THREE.Mesh).material;
      if (Array.isArray(mat)) {
        mat.forEach((m) => m.dispose());
      } else {
        mat.dispose();
      }
    }
    while (obj.children.length > 0) {
      const c = obj.children[0];
      this.disposeObject(c);
      obj.remove(c);
    }
  }

  public resize(): void {
    if (!this.container) return;
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  private animate(): void {
    this.animId = requestAnimationFrame(this.animate);
    this.camera.position.lerp(this.targetCameraPos, 0.06);
    this.controls.target.lerp(this.targetLookAt, 0.06);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  public destroy(): void {
    if (this.animId !== null) cancelAnimationFrame(this.animId);
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement && this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
