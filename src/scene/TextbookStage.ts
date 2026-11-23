import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { ModelTrace } from '../engine/types';
import type { BookState } from '../state/bookState';
import {
  divergingColor,
  formatToken,
  makeTextSprite,
  sequentialColor,
  threeDivergingColor,
  threeSequentialColor,
} from '../utils/math';

export interface StageRenderData {
  state: BookState;
  trace: ModelTrace | null;
  activeSceneBeat: string;
}

export class TextbookStage {
  private container: HTMLDivElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private contentGroup: THREE.Group;
  private animFrameId: number | null = null;
  private pmremGenerator: THREE.PMREMGenerator;
  private envRT: THREE.WebGLRenderTarget;

  private currentData: StageRenderData | null = null;
  private onSelectTokenCb?: (queryIdx: number, keyIdx: number | null) => void;

  constructor(container: HTMLDivElement) {
    this.container = container;

    const width = Math.max(10, container.clientWidth);
    const height = Math.max(10, container.clientHeight);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfbfcfe);

    this.camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
    this.camera.position.set(0, 18, 38);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    container.appendChild(this.renderer.domElement);

    this.pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    this.pmremGenerator.compileEquirectangularShader();
    const room = new RoomEnvironment();
    this.envRT = this.pmremGenerator.fromScene(room, 0.04);
    this.scene.environment = this.envRT.texture;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    this.controls.maxDistance = 120;
    this.controls.minDistance = 6;

    this.contentGroup = new THREE.Group();
    this.scene.add(this.contentGroup);

    this.setupLighting();
    this.setupGround();
    this.startLoop();
  }

  public setSelectTokenCallback(
    cb: (queryIdx: number, keyIdx: number | null) => void
  ) {
    this.onSelectTokenCb = cb;
  }

  private setupLighting() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0xe2e8f0, 0.85);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight(0xfffaed, 1.25);
    sun.position.set(22, 40, 26);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 120;
    const d = 30;
    sun.shadow.camera.left = -d;
    sun.shadow.camera.right = d;
    sun.shadow.camera.top = d;
    sun.shadow.camera.bottom = -d;
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);

    const fill = new THREE.DirectionalLight(0xe0f2fe, 0.45);
    fill.position.set(-20, 16, -18);
    this.scene.add(fill);
  }

  private setupGround() {
    const groundGeo = new THREE.PlaneGeometry(160, 160);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0xf5f7fa,
      roughness: 0.95,
      metalness: 0.02,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const grid = new THREE.GridHelper(100, 50, 0xd1d9e2, 0xeef2f6);
    grid.position.y = 0;
    this.scene.add(grid);
  }

  public resize(width: number, height: number) {
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  public resetCamera() {
    this.camera.position.set(0, 18, 38);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  public update(data: StageRenderData) {
    this.currentData = data;
    this.rebuildContent();
  }

  private clearGroup(grp: THREE.Group) {
    while (grp.children.length > 0) {
      const obj = grp.children[0];
      grp.remove(obj);
      if ((obj as THREE.Mesh).isMesh || (obj as THREE.Sprite).isSprite) {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((m) => m.dispose());
          } else {
            mesh.material.dispose();
          }
        }
      }
    }
  }

  private rebuildContent() {
    this.clearGroup(this.contentGroup);
    if (!this.currentData || !this.currentData.trace) return;

    const { state, trace, activeSceneBeat } = this.currentData;
    const numTokens = trace.tokens.length;
    const resolvedQ = Math.max(
      0,
      Math.min(numTokens - 1, state.queryIdx ?? numTokens - 1)
    );
    const resolvedK = Math.max(
      0,
      Math.min(numTokens - 1, state.keyIdx ?? resolvedQ)
    );

    switch (activeSceneBeat) {
      case 'attn-1-qkv-projection':
        this.buildBeat1QKV(trace, state, resolvedQ);
        break;
      case 'attn-2-dot-product':
        this.buildBeat2DotProduct(trace, state, resolvedQ, resolvedK);
        break;
      case 'attn-3-scale-by-root-d':
        this.buildBeat3Scale(trace, state, resolvedQ);
        break;
      case 'attn-4-mask-softmax':
        this.buildBeat4MaskSoftmax(trace, state, resolvedQ);
        break;
      case 'attn-5-value-mixer':
        this.buildBeat5ValueMixer(trace, state, resolvedQ);
        break;
      case 'attn-6-multihead-full':
        this.buildBeat6MultiHead(trace, state, resolvedQ);
        break;
      default:
        this.buildBeat1QKV(trace, state, resolvedQ);
        break;
    }
  }

  // --- BEAT 1: QKV Projection ---
  private buildBeat1QKV(
    trace: ModelTrace,
    state: BookState,
    queryIdx: number
  ) {
    const numTokens = trace.tokens.length;
    const head = state.head;
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const qHead = blk.attn.q[head][queryIdx] ?? [];
    const kHead = blk.attn.k[head][queryIdx] ?? [];
    const vHead = blk.attn.v[head][queryIdx] ?? [];

    const labelTok = makeTextSprite(
      `Selected Token [${queryIdx}]: "${formatToken(
        trace.tokens[queryIdx].text
      )}" · Block ${state.layer} · Head ${head}`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    labelTok.position.set(0, 10, 0);
    this.contentGroup.add(labelTok);

    const roles = [
      { name: 'Query (Q)', vec: qHead, colorHex: 0x2563eb, xOff: -11 },
      { name: 'Key (K)', vec: kHead, colorHex: 0x059669, xOff: 0 },
      { name: 'Value (V)', vec: vHead, colorHex: 0xd97706, xOff: 11 },
    ];

    roles.forEach((role) => {
      const header = makeTextSprite(`${role.name} [d=16]`, {
        fontSize: 20,
        bgColor: '#ffffff',
        fontColor: '#0f172a',
      });
      header.position.set(role.xOff, 7.5, 0);
      this.contentGroup.add(header);

      const dHead = 16;
      for (let c = 0; c < dHead; c++) {
        const val = role.vec[c] ?? 0;
        const barH = Math.max(0.12, Math.abs(val) * 1.8);
        const geo = new THREE.BoxGeometry(0.8, barH, 0.8);
        const col = val >= 0 ? role.colorHex : 0xf43f5e;
        const mat = new THREE.MeshStandardMaterial({
          color: col,
          roughness: 0.25,
          metalness: 0.1,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(
          role.xOff - 3.5 + (c % 4) * 2.3,
          barH / 2,
          -3 + Math.floor(c / 4) * 2.2
        );
        this.contentGroup.add(mesh);
      }
    });
  }

  // --- BEAT 2: Dot Product ---
  private buildBeat2DotProduct(
    trace: ModelTrace,
    state: BookState,
    qIdx: number,
    kIdx: number
  ) {
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const qVec = blk.attn.q[state.head][qIdx] ?? [];
    const kVec = blk.attn.k[state.head][kIdx] ?? [];
    const rawDot = blk.attn.preSoftmax[state.head][qIdx]?.[kIdx] ?? 0;

    const tokQ = formatToken(trace.tokens[qIdx].text);
    const tokK = formatToken(trace.tokens[kIdx].text);

    const banner = makeTextSprite(
      `Dot Product: Q("${tokQ}") · K("${tokK}") = ${rawDot.toFixed(3)}`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    banner.position.set(0, 11, 0);
    this.contentGroup.add(banner);

    let runningSum = 0;
    for (let c = 0; c < 16; c++) {
      const qVal = qVec[c] ?? 0;
      const kVal = kVec[c] ?? 0;
      const prod = qVal * kVal;
      runningSum += prod;

      const x = -13 + c * 1.75;

      const qBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, Math.max(0.08, Math.abs(qVal) * 1.5), 0.5),
        new THREE.MeshStandardMaterial({
          color: 0x2563eb,
          roughness: 0.3,
        })
      );
      qBar.position.set(
        x,
        Math.max(0.08, Math.abs(qVal) * 1.5) / 2,
        3.5
      );
      qBar.castShadow = true;
      this.contentGroup.add(qBar);

      const kBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, Math.max(0.08, Math.abs(kVal) * 1.5), 0.5),
        new THREE.MeshStandardMaterial({
          color: 0x059669,
          roughness: 0.3,
        })
      );
      kBar.position.set(
        x,
        Math.max(0.08, Math.abs(kVal) * 1.5) / 2,
        0.5
      );
      kBar.castShadow = true;
      this.contentGroup.add(kBar);

      const prodH = Math.max(0.1, Math.abs(prod) * 2.2);
      const prodBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, prodH, 0.9),
        new THREE.MeshStandardMaterial({
          color: prod >= 0 ? 0x0284c7 : 0xf43f5e,
          roughness: 0.2,
        })
      );
      prodBar.position.set(x, prodH / 2, -3.5);
      prodBar.castShadow = true;
      this.contentGroup.add(prodBar);
    }

    const lblQ = makeTextSprite(`Q [${tokQ}]`, {
      fontSize: 16,
      bgColor: '#dbeafe',
      fontColor: '#1e40af',
    });
    lblQ.position.set(-15, 1.5, 3.5);
    this.contentGroup.add(lblQ);

    const lblK = makeTextSprite(`K [${tokK}]`, {
      fontSize: 16,
      bgColor: '#d1fae5',
      fontColor: '#065f46',
    });
    lblK.position.set(-15, 1.5, 0.5);
    this.contentGroup.add(lblK);

    const lblP = makeTextSprite(`q × k (Products)`, {
      fontSize: 16,
      bgColor: '#f1f5f9',
      fontColor: '#0f172a',
    });
    lblP.position.set(-15, 1.5, -3.5);
    this.contentGroup.add(lblP);
  }

  // --- BEAT 3: Scale by √d ---
  private buildBeat3Scale(
    trace: ModelTrace,
    state: BookState,
    qIdx: number
  ) {
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const row = blk.attn.preSoftmax[state.head][qIdx] ?? [];
    const numTokens = trace.tokens.length;

    const banner = makeTextSprite(
      `Scaling by 1 / √16 = 0.25 · Row ${qIdx} ("${formatToken(
        trace.tokens[qIdx].text
      )}")`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    banner.position.set(0, 11, 0);
    this.contentGroup.add(banner);

    for (let j = 0; j < numTokens; j++) {
      const raw = row[j] ?? 0;
      const unscaled = raw * 4.0;
      const x = -10 + j * 1.5;

      if (j > qIdx) {
        const maskedBlock = new THREE.Mesh(
          new THREE.BoxGeometry(1.1, 0.2, 1.1),
          new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.9,
          })
        );
        maskedBlock.position.set(x, 0.1, 0);
        this.contentGroup.add(maskedBlock);
        continue;
      }

      const unscaledH = Math.max(0.1, Math.abs(unscaled) * 0.7);
      const unscaledBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, unscaledH, 0.5),
        new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          roughness: 0.3,
        })
      );
      unscaledBar.position.set(x - 0.35, unscaledH / 2, 2.5);
      unscaledBar.castShadow = true;
      this.contentGroup.add(unscaledBar);

      const scaledH = Math.max(0.1, Math.abs(raw) * 0.7);
      const scaledBar = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, scaledH, 0.5),
        new THREE.MeshStandardMaterial({
          color: 0x0284c7,
          roughness: 0.2,
        })
      );
      scaledBar.position.set(x + 0.35, scaledH / 2, -2.5);
      scaledBar.castShadow = true;
      this.contentGroup.add(scaledBar);

      const tokLabel = makeTextSprite(formatToken(trace.tokens[j].text), {
        fontSize: 14,
        bgColor: '#ffffff',
        fontColor: '#334155',
      });
      tokLabel.position.set(x, 0.5, 5);
      this.contentGroup.add(tokLabel);
    }

    const legend1 = makeTextSprite('Unscaled Dot Products (Wild Variance)', {
      fontSize: 16,
      bgColor: '#fef3c7',
      fontColor: '#92400e',
    });
    legend1.position.set(-13, 3, 2.5);
    this.contentGroup.add(legend1);

    const legend2 = makeTextSprite('Scaled by 1/4 (Healthy Softmax Gradient)', {
      fontSize: 16,
      bgColor: '#e0f2fe',
      fontColor: '#075985',
    });
    legend2.position.set(-13, 3, -2.5);
    this.contentGroup.add(legend2);
  }

  // --- BEAT 4: Causal Mask & Softmax ---
  private buildBeat4MaskSoftmax(
    trace: ModelTrace,
    state: BookState,
    qIdx: number
  ) {
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const matrix = state.showPreSoftmax
      ? blk.attn.preSoftmax[state.head]
      : blk.attn.probs[state.head];
    const N = trace.tokens.length;

    const banner = makeTextSprite(
      state.showPreSoftmax
        ? `Pre-Softmax Logits (with -∞ Mask)`
        : `Post-Softmax Attention Weights (100% per row)`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    banner.position.set(0, 11, 0);
    this.contentGroup.add(banner);

    const cellSize = 1.35;
    const originX = -((N * cellSize) / 2);
    const originZ = -((N * cellSize) / 2);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = originX + j * cellSize + cellSize / 2;
        const z = originZ + i * cellSize + cellSize / 2;

        if (j > i) {
          const darkSlate = new THREE.Mesh(
            new THREE.BoxGeometry(cellSize * 0.9, 0.2, cellSize * 0.9),
            new THREE.MeshStandardMaterial({
              color: 0x334155,
              roughness: 0.9,
            })
          );
          darkSlate.position.set(x, 0.1, z);
          this.contentGroup.add(darkSlate);
          continue;
        }

        const val = matrix[i]?.[j] ?? 0;
        const h = state.showPreSoftmax
          ? Math.max(0.15, Math.abs(val) * 0.6)
          : Math.max(0.15, val * 8.0);
        const col = state.showPreSoftmax
          ? threeDivergingColor(val, 4.0)
          : threeSequentialColor(val);

        const tile = new THREE.Mesh(
          new THREE.BoxGeometry(cellSize * 0.9, h, cellSize * 0.9),
          new THREE.MeshStandardMaterial({
            color: col,
            roughness: 0.25,
            metalness: 0.05,
          })
        );
        tile.position.set(x, h / 2, z);
        tile.castShadow = true;
        tile.receiveShadow = true;

        if (i === qIdx) {
          const edge = new THREE.BoxHelper(tile, new THREE.Color(0x1d4ed8));
          this.contentGroup.add(edge);
        }

        this.contentGroup.add(tile);
      }
    }
  }

  // --- BEAT 5: Value Mixer ---
  private buildBeat5ValueMixer(
    trace: ModelTrace,
    state: BookState,
    qIdx: number
  ) {
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const probs = blk.attn.probs[state.head][qIdx] ?? [];
    const vMatrix = blk.attn.v[state.head];
    const outHead = blk.attn.out[state.head][qIdx] ?? [];
    const tokQ = formatToken(trace.tokens[qIdx].text);

    const banner = makeTextSprite(
      `Token "${tokQ}" Blending Value Vectors according to Attention %`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    banner.position.set(0, 11, 0);
    this.contentGroup.add(banner);

    for (let j = 0; j <= qIdx; j++) {
      const weight = probs[j] ?? 0;
      const x = -10 + j * 1.6;
      const tokJ = formatToken(trace.tokens[j].text);

      const beamH = Math.max(0.1, weight * 10);
      const beam = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.45, beamH, 16),
        new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          roughness: 0.2,
          transparent: true,
          opacity: Math.max(0.2, weight * 1.5),
        })
      );
      beam.position.set(x, beamH / 2, 2);
      this.contentGroup.add(beam);

      const pLabel = makeTextSprite(`${(weight * 100).toFixed(0)}%`, {
        fontSize: 16,
        bgColor: '#ffffff',
        fontColor: '#0f172a',
      });
      pLabel.position.set(x, beamH + 0.8, 2);
      this.contentGroup.add(pLabel);

      const tokLabel = makeTextSprite(tokJ, {
        fontSize: 14,
        bgColor: '#f1f5f9',
        fontColor: '#334155',
      });
      tokLabel.position.set(x, 0.4, 4);
      this.contentGroup.add(tokLabel);
    }

    const outBanner = makeTextSprite(
      `Resulting Head Vector y_${qIdx} (d=16)`,
      { fontSize: 20, bgColor: '#0284c7', fontColor: '#ffffff' }
    );
    outBanner.position.set(0, 7.5, -4);
    this.contentGroup.add(outBanner);

    for (let c = 0; c < 16; c++) {
      const val = outHead[c] ?? 0;
      const barH = Math.max(0.15, Math.abs(val) * 2.0);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.7, barH, 0.7),
        new THREE.MeshStandardMaterial({
          color: val >= 0 ? 0x0284c7 : 0xf43f5e,
          roughness: 0.25,
        })
      );
      mesh.position.set(-7 + (c % 8) * 1.8, barH / 2, -4 - Math.floor(c / 8) * 2);
      mesh.castShadow = true;
      this.contentGroup.add(mesh);
    }
  }

  // --- BEAT 6: 4-Head Orchestra ---
  private buildBeat6MultiHead(
    trace: ModelTrace,
    state: BookState,
    qIdx: number
  ) {
    const blk = trace.blocks[state.layer];
    if (!blk) return;

    const banner = makeTextSprite(
      `4 Heads in Parallel · Concatenated to 64 dimensions · Multiplied by Wo`,
      { fontSize: 24, bgColor: '#1e293b', fontColor: '#ffffff' }
    );
    banner.position.set(0, 11, 0);
    this.contentGroup.add(banner);

    const headColors = [0x2563eb, 0x059669, 0xd97706, 0x7c3aed];
    const xOffsets = [-12, -4, 4, 12];

    for (let h = 0; h < 4; h++) {
      const off = (state.ablation.disabledHeads ?? []).some(
        ([l, hd]) => l === state.layer && hd === h
      );

      const hLabel = makeTextSprite(
        `Head ${h} ${off ? '(ABLATED - OFF)' : '(Active)'}`,
        {
          fontSize: 18,
          bgColor: off ? '#fee2e2' : '#ffffff',
          fontColor: off ? '#991b1b' : '#0f172a',
        }
      );
      hLabel.position.set(xOffsets[h], 7.5, 0);
      this.contentGroup.add(hLabel);

      const headVec = blk.attn.out[h][qIdx] ?? [];
      for (let c = 0; c < 16; c++) {
        const val = off ? 0 : headVec[c] ?? 0;
        const barH = Math.max(0.12, Math.abs(val) * 1.8);
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(0.7, barH, 0.7),
          new THREE.MeshStandardMaterial({
            color: off ? 0x94a3b8 : headColors[h],
            roughness: 0.3,
            metalness: off ? 0.0 : 0.1,
          })
        );
        mesh.position.set(
          xOffsets[h] - 1.5 + (c % 4) * 1.0,
          barH / 2,
          -2 + Math.floor(c / 4) * 1.1
        );
        mesh.castShadow = true;
        this.contentGroup.add(mesh);
      }
    }

    const woSprite = makeTextSprite('Projection Matrix W_o (64 × 64)', {
      fontSize: 18,
      bgColor: '#1e293b',
      fontColor: '#ffffff',
    });
    woSprite.position.set(0, 2.5, 6);
    this.contentGroup.add(woSprite);
  }

  private startLoop() {
    const loop = () => {
      this.animFrameId = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  public destroy() {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
    }
    this.controls.dispose();
    this.clearGroup(this.contentGroup);
    this.pmremGenerator.dispose();
    this.envRT.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
