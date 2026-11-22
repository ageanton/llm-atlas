import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type {
  AblationConfig,
  ActiveLayerTrace,
  EngineTraceResult,
  TokenizerData,
} from '../engine/types';
import { isHeadDisabled, type Register, type ZoomLevel } from '../state/urlState';
import { formatToken, vectorL2Norm } from '../utils/color';
import { studioSound } from './sound';
import {
  createPillLabelTexture,
  createPorcelainTileTexture,
  createThoughtBubbleTexture,
} from './textures';

export interface WorldCallbacks {
  onSelectLevel: (level: ZoomLevel) => void;
  onSelectChamber: (layer: number, zoomIn: boolean) => void;
  onSelectHead: (head: number) => void;
  onToggleHeadAblation: (layer: number, head: number) => void;
  onHoverToken: (index: number | null) => void;
}

export interface WorldSnapshot {
  level: ZoomLevel;
  layer: number;
  head: number;
  register: Register;
  ablation: AblationConfig;
  baseline: EngineTraceResult;
  ablated: EngineTraceResult | null;
  tokenizer: TokenizerData;
  hoveredTokenIdx: number | null;
}

interface ClickTag {
  kind: 'box' | 'chamber' | 'attnRing' | 'headOrb' | 'token';
  index: number;
}

interface RibbonEntry {
  q: number;
  k: number;
  weight: number;
  mesh: THREE.Mesh;
  mat: THREE.MeshPhysicalMaterial;
  curve: THREE.CatmullRomCurve3;
}

interface PulseBead {
  mesh: THREE.Mesh;
  curve: THREE.CatmullRomCurve3;
  offset: number;
  speed: number;
}

const CHAMBER_X = [-3.0, -1.0, 1.0, 3.0];

export class TransformerWorld {
  private readonly container: HTMLElement;
  private readonly callbacks: WorldCallbacks;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2(-10, -10);
  private readonly envRT: THREE.WebGLRenderTarget;

  // Root groups for seamless continuous transitions
  private readonly studioRoot = new THREE.Group();
  private readonly shellGroup = new THREE.Group();
  private readonly factoryGroup = new THREE.Group();
  private readonly chamberGroup = new THREE.Group();
  private readonly harpGroup = new THREE.Group();
  private readonly bubblesGroup = new THREE.Group();

  // Persistent architectural elements
  private outerShellMesh!: THREE.Mesh;
  private outerShellMat!: THREE.MeshPhysicalMaterial;
  private shellPulseRing!: THREE.Mesh;
  private incomingTilesGroup = new THREE.Group();
  private dynamicGroups: THREE.Group[] = [];
  private interactives: THREE.Object3D[] = [];
  private disposableTextures: THREE.Texture[] = [];
  private disposableGeometries: THREE.BufferGeometry[] = [];
  private disposableMaterials: THREE.Material[] = [];

  // Animated handles
  private ribbons: RibbonEntry[] = [];
  private pulseBeads: PulseBead[] = [];
  private headOrbs: { mesh: THREE.Mesh; head: number; basePos: THREE.Vector3; off: boolean }[] = [];
  private floaters: { obj: THREE.Object3D; baseY: number; phase: number; amp: number }[] = [];
  private factoryPulseLight!: THREE.Mesh;
  private mlpCoreMesh: THREE.Mesh | null = null;
  private pulseTime = 0;

  // Camera spring targets
  private camTargetPos = new THREE.Vector3(0, 3.4, 10.6);
  private camTargetLook = new THREE.Vector3(0, 0.35, 0);
  private isUserDragging = false;
  private needsCameraFlight = false;

  private snapshot: WorldSnapshot | null = null;
  private lastHoverToken: number | null = null;
  private animFrameId = 0;
  private clock = new THREE.Clock();
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
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog('#eff2f7', 16, 36);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = this.envRT.texture;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(
      38,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      80
    );
    this.camera.position.copy(this.camTargetPos);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.enablePan = false;
    this.controls.minDistance = 4.2;
    this.controls.maxDistance = 17;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minPolarAngle = 0.18;
    this.controls.addEventListener('start', () => {
      this.isUserDragging = true;
      this.needsCameraFlight = false;
    });
    this.controls.addEventListener('end', () => {
      this.isUserDragging = false;
    });

    this.setupLightingAndStage();
    this.buildStaticShell();

    this.studioRoot.add(
      this.shellGroup,
      this.factoryGroup,
      this.chamberGroup,
      this.harpGroup,
      this.bubblesGroup
    );
    this.scene.add(this.studioRoot);

    this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.addEventListener('click', this.onClick);
    this.renderer.domElement.addEventListener('dblclick', this.onDoubleClick);
    window.addEventListener('resize', this.onResize);

    this.animate();
  }

  private setupLightingAndStage(): void {
    const hemi = new THREE.HemisphereLight('#ffffff', '#dbe4f0', 1.1);
    hemi.position.set(0, 12, 0);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight('#fff8ee', 1.45);
    sun.position.set(7, 12, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 2;
    sun.shadow.camera.far = 28;
    sun.shadow.camera.left = -9;
    sun.shadow.camera.right = 9;
    sun.shadow.camera.top = 7;
    sun.shadow.camera.bottom = -7;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);

    const rim = new THREE.DirectionalLight('#bfd7ff', 0.65);
    rim.position.set(-8, 6, -7);
    this.scene.add(rim);

    // Sunlit museum pedestal floor receiving soft grounded shadows
    const floorGeo = new THREE.CircleGeometry(18, 64);
    const floorMat = new THREE.MeshStandardMaterial({
      color: '#f3f1ec',
      roughness: 0.92,
      metalness: 0.02,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.75;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  private buildStaticShell(): void {
    // Sleek frosted optical-glass monolith (Level 0 outer casing)
    const shellGeo = new RoundedBoxGeometry(6.8, 2.55, 2.9, 5, 0.36);
    this.outerShellMat = new THREE.MeshPhysicalMaterial({
      color: '#ffffff',
      roughness: 0.22,
      metalness: 0.02,
      transmission: 0.72,
      thickness: 0.9,
      ior: 1.45,
      clearcoat: 1.0,
      clearcoatRoughness: 0.12,
      transparent: true,
      opacity: 0.94,
    });
    this.outerShellMesh = new THREE.Mesh(shellGeo, this.outerShellMat);
    this.outerShellMesh.position.set(0, 0.1, 0);
    this.outerShellMesh.castShadow = true;
    this.outerShellMesh.receiveShadow = true;
    this.outerShellMesh.userData = { kind: 'box', index: 0 } satisfies ClickTag;
    this.shellGroup.add(this.outerShellMesh);

    // Warm brushed-brass bezel rim around the intake & exit
    const brassMat = new THREE.MeshStandardMaterial({
      color: '#c8a46b',
      metalness: 0.78,
      roughness: 0.28,
    });
    const intakeRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.68, 0.05, 16, 48),
      brassMat
    );
    intakeRing.rotation.y = Math.PI / 2;
    intakeRing.position.set(-3.42, 0.1, 0);
    this.shellGroup.add(intakeRing);

    const exitRing = intakeRing.clone();
    exitRing.position.set(3.42, 0.1, 0);
    this.shellGroup.add(exitRing);

    // Sweeping light-pulse band inside the frosted shell
    const pulseGeo = new THREE.BoxGeometry(0.32, 2.2, 2.5);
    const pulseMat = new THREE.MeshBasicMaterial({
      color: '#3b82f6',
      transparent: true,
      opacity: 0.32,
    });
    this.shellPulseRing = new THREE.Mesh(pulseGeo, pulseMat);
    this.shellPulseRing.position.set(-2.8, 0.1, 0);
    this.shellGroup.add(this.shellPulseRing);

    this.shellGroup.add(this.incomingTilesGroup);
  }

  private clearDynamic(): void {
    for (const g of this.dynamicGroups) {
      g.clear();
    }
    this.dynamicGroups = [];
    this.interactives = [this.outerShellMesh];
    this.ribbons = [];
    this.pulseBeads = [];
    this.headOrbs = [];
    this.floaters = [];
    this.mlpCoreMesh = null;

    for (const t of this.disposableTextures) t.dispose();
    for (const g of this.disposableGeometries) g.dispose();
    for (const m of this.disposableMaterials) m.dispose();
    this.disposableTextures = [];
    this.disposableGeometries = [];
    this.disposableMaterials = [];
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

  triggerPulse(): void {
    this.pulseTime = 0;
  }

  update(next: WorldSnapshot): void {
    const prevLevel = this.snapshot?.level;
    const prevLayer = this.snapshot?.layer;
    const prevPrompt = this.snapshot?.baseline.prompt;
    this.snapshot = next;

    if (prevLevel !== next.level || prevLayer !== next.layer) {
      this.setCameraForState(next.level, next.layer);
      if (prevLevel !== undefined) {
        studioSound.playGlide(next.level);
      }
    }

    if (prevPrompt !== next.baseline.prompt) {
      this.pulseTime = 0;
    }

    this.rebuildScene(next);
  }

  setHoveredToken(index: number | null): void {
    if (!this.snapshot) return;
    this.snapshot = { ...this.snapshot, hoveredTokenIdx: index };
    this.applyRibbonHighlight(index);
  }

  private setCameraForState(level: ZoomLevel, layer: number): void {
    this.needsCameraFlight = true;
    if (level === 0) {
      this.camTargetPos.set(0.2, 2.7, 11.2);
      this.camTargetLook.set(0.5, 0.2, 0);
    } else if (level === 1) {
      this.camTargetPos.set(0.4, 4.4, 10.2);
      this.camTargetLook.set(0.4, 0.25, 0);
    } else if (level === 2) {
      const cx = CHAMBER_X[layer] ?? 0;
      this.camTargetPos.set(cx + 0.2, 2.5, 6.4);
      this.camTargetLook.set(cx, 0.25, 0);
    } else {
      this.camTargetPos.set(0, 3.6, 9.4);
      this.camTargetLook.set(0, 0.35, 0);
    }
  }

  private rebuildScene(snap: WorldSnapshot): void {
    this.clearDynamic();
    this.incomingTilesGroup.clear();
    this.factoryGroup.clear();
    this.chamberGroup.clear();
    this.harpGroup.clear();
    this.bubblesGroup.clear();
    this.dynamicGroups.push(
      this.incomingTilesGroup,
      this.factoryGroup,
      this.chamberGroup,
      this.harpGroup,
      this.bubblesGroup
    );

    const activeRes = snap.ablated ?? snap.baseline;

    // Visibility toggles per continuous level
    this.shellGroup.visible = snap.level === 0 || snap.level === 1;
    this.factoryGroup.visible = snap.level === 0 || snap.level === 1;
    this.chamberGroup.visible = snap.level === 2;
    this.harpGroup.visible = snap.level === 3;
    this.bubblesGroup.visible = true;

    // Build shared exit probability bubbles (visible across levels so ablations are always felt!)
    this.buildExitBubbles(snap);

    if (snap.level === 0 || snap.level === 1) {
      this.buildIncomingTiles(activeRes);
      this.buildGlassFactory(snap);
    }
    if (snap.level === 2) {
      this.buildChamberInterior(snap);
    }
    if (snap.level === 3) {
      this.buildHarpOfAttention(snap);
    }
  }

  private buildIncomingTiles(activeRes: EngineTraceResult): void {
    const shown = activeRes.tokens.slice(-5);
    const tileGeo = this.trackGeo(new RoundedBoxGeometry(0.66, 0.42, 0.16, 3, 0.06));
    const sideMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: '#f5f2eb',
        roughness: 0.35,
        metalness: 0.04,
      })
    );

    shown.forEach((tok, idx) => {
      const tex = this.trackTex(
        createPorcelainTileTexture(formatToken(tok.text), `#${tok.id}`, false)
      );
      const faceMat = this.trackMat(
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.3, metalness: 0.02 })
      );
      const mesh = new THREE.Mesh(tileGeo, [
        sideMat,
        sideMat,
        sideMat,
        sideMat,
        faceMat,
        faceMat,
      ]);
      const x = -6.7 + idx * 0.78;
      const y = 0.12 + Math.sin(idx * 0.9) * 0.1;
      mesh.position.set(x, y, 0.15);
      mesh.castShadow = true;
      this.incomingTilesGroup.add(mesh);
      this.floaters.push({ obj: mesh, baseY: y, phase: idx * 0.85, amp: 0.06 });
    });
  }

  private buildGlassFactory(snap: WorldSnapshot): void {
    const activeRes = snap.ablated ?? snap.baseline;
    const { trace } = activeRes;

    // Unbroken luminous central highway (Residual Stream)
    const trackGeo = this.trackGeo(new THREE.CylinderGeometry(0.085, 0.085, 9.2, 24));
    const trackMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#2563eb',
        emissive: '#1d4ed8',
        emissiveIntensity: 0.45,
        roughness: 0.15,
        clearcoat: 1,
      })
    );
    const track = new THREE.Mesh(trackGeo, trackMat);
    track.rotation.z = Math.PI / 2;
    track.position.set(0, 0.08, 0);
    this.factoryGroup.add(track);

    // Animated pulse orb riding the residual highway
    const pulseGeo = this.trackGeo(new THREE.SphereGeometry(0.16, 20, 20));
    const pulseMat = this.trackMat(
      new THREE.MeshBasicMaterial({ color: '#60a5fa' })
    );
    this.factoryPulseLight = new THREE.Mesh(pulseGeo, pulseMat);
    this.factoryPulseLight.position.set(-4.3, 0.08, 0);
    this.factoryGroup.add(this.factoryPulseLight);

    // Embedding Prism on the left showing real last-token embedding bar magnitudes
    const lastEmb = trace.tokEmb[trace.tokEmb.length - 1];
    const barGeo = this.trackGeo(new THREE.BoxGeometry(0.055, 1, 0.055));
    for (let i = 0; i < 12; i++) {
      const val = lastEmb[i * 10] ?? 0;
      const h = Math.max(0.14, Math.min(1.15, Math.abs(val) * 3.8));
      const barMat = this.trackMat(
        new THREE.MeshStandardMaterial({
          color: val >= 0 ? '#d97706' : '#2563eb',
          roughness: 0.25,
          metalness: 0.15,
        })
      );
      const bar = new THREE.Mesh(barGeo, barMat);
      const angle = (i / 12) * Math.PI * 2;
      bar.scale.y = h;
      bar.position.set(-4.45 + Math.cos(angle) * 0.26, 0.08 + h * 0.22, Math.sin(angle) * 0.26);
      this.factoryGroup.add(bar);
    }

    // 4 Crystal-clear Glass Chambers (Transformer Blocks 0..3)
    const chamberGeo = this.trackGeo(new RoundedBoxGeometry(1.48, 1.95, 1.95, 4, 0.22));
    const coreGeo = this.trackGeo(new THREE.OctahedronGeometry(0.34, 1));

    for (let l = 0; l < 4; l++) {
      const x = CHAMBER_X[l];
      const layerTrace = trace.layers[l];
      const isSelected = snap.layer === l;
      const isSkipped = layerTrace.skipped;

      const glassMat = this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: isSkipped ? '#fed7aa' : isSelected ? '#dbeafe' : '#ffffff',
          roughness: 0.16,
          metalness: 0.04,
          transmission: 0.72,
          thickness: 0.6,
          ior: 1.45,
          clearcoat: 1,
          transparent: true,
          opacity: snap.level === 0 ? 0.55 : 0.92,
        })
      );
      const box = new THREE.Mesh(chamberGeo, glassMat);
      box.position.set(x, 0.1, 0);
      box.castShadow = true;
      box.userData = { kind: 'chamber', index: l } satisfies ClickTag;
      this.factoryGroup.add(box);
      this.interactives.push(box);

      // Inner glowing crystal scaled by real residual update norm
      let norm = 0.5;
      if (!layerTrace.skipped) {
        const lastIdx = layerTrace.attn.output.length - 1;
        norm = Math.min(1.3, 0.55 + vectorL2Norm(layerTrace.attn.output[lastIdx]) * 0.07);
      }
      const coreMat = this.trackMat(
        new THREE.MeshStandardMaterial({
          color: isSkipped ? '#c2410c' : isSelected ? '#2563eb' : '#64748b',
          emissive: isSkipped ? '#9a3412' : '#1d4ed8',
          emissiveIntensity: isSelected ? 0.55 : 0.22,
          roughness: 0.25,
          metalness: 0.4,
        })
      );
      const core = new THREE.Mesh(coreGeo, coreMat);
      core.position.set(x, 0.1, 0);
      core.scale.setScalar(norm * 0.75);
      this.factoryGroup.add(core);

      // Live Logit Lens Thought-Bubble hovering above each chamber (visible at Level 1)
      if (snap.level === 1) {
        const lens = trace.logitLens[l];
        const top = lens?.topTokens[0];
        if (top) {
          const tokText = formatToken(snap.tokenizer.vocab[top.id] ?? '');
          const tex = this.trackTex(
            createThoughtBubbleTexture(
              `Chamber ${l}`,
              tokText,
              `${(top.prob * 100).toFixed(0)}%`,
              isSelected ? '#1d4ed8' : '#0f766e'
            )
          );
          const spMat = this.trackMat(
            new THREE.SpriteMaterial({ map: tex, transparent: true })
          );
          const sprite = new THREE.Sprite(spMat);
          sprite.scale.set(1.75, 0.93, 1);
          sprite.position.set(x, 1.92, 0);
          this.factoryGroup.add(sprite);
          this.floaters.push({ obj: sprite, baseY: 1.92, phase: l * 1.1, amp: 0.07 });
        }
      }
    }
  }

  private buildChamberInterior(snap: WorldSnapshot): void {
    const activeRes = snap.ablated ?? snap.baseline;
    const layerIdx = snap.layer;
    const cx = CHAMBER_X[layerIdx] ?? 0;
    const layerTrace = activeRes.trace.layers[layerIdx];

    const group = new THREE.Group();
    group.position.set(cx, 0, 0);
    this.chamberGroup.add(group);

    // Subtle frosted glass pavilion walls framing the chamber
    const frameGeo = this.trackGeo(new RoundedBoxGeometry(6.6, 3.6, 3.8, 3, 0.25));
    const frameMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#ffffff',
        roughness: 0.25,
        transmission: 0.84,
        transparent: true,
        opacity: 0.28,
        side: THREE.BackSide,
      })
    );
    const frame = new THREE.Mesh(frameGeo, frameMat);
    frame.position.set(0, 0.25, 0);
    group.add(frame);

    // Unbroken Residual Highway running right down the center
    const trackGeo = this.trackGeo(new THREE.CylinderGeometry(0.095, 0.095, 7.6, 24));
    const trackMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#1d4ed8',
        emissive: '#1d4ed8',
        emissiveIntensity: 0.5,
        roughness: 0.15,
        clearcoat: 1,
      })
    );
    const highway = new THREE.Mesh(trackGeo, trackMat);
    highway.rotation.z = Math.PI / 2;
    highway.position.set(0, -0.55, 0);
    group.add(highway);

    // Curved glass aqueduct A -> Attention Ring (left side)
    const loopACurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.6, -0.55, 0),
      new THREE.Vector3(-2.2, 0.65, 0.35),
      new THREE.Vector3(-1.15, 0.85, 0),
      new THREE.Vector3(-0.2, -0.55, 0),
    ]);
    const loopAGeo = this.trackGeo(new THREE.TubeGeometry(loopACurve, 40, 0.055, 12, false));
    const glassPipeMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#93c5fd',
        roughness: 0.12,
        transmission: 0.6,
        transparent: true,
        opacity: 0.85,
      })
    );
    group.add(new THREE.Mesh(loopAGeo, glassPipeMat));

    // Pre-LN1 glowing brass ring
    const lnMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: '#c8a46b',
        metalness: 0.8,
        roughness: 0.25,
      })
    );
    const ln1 = new THREE.Mesh(
      this.trackGeo(new THREE.TorusGeometry(0.18, 0.035, 14, 32)),
      lnMat
    );
    ln1.position.set(-2.25, 0.35, 0.25);
    ln1.rotation.y = Math.PI / 3;
    group.add(ln1);

    // Attention Ring (clickable to fly straight into Level 3 Harp!)
    const ringGeo = this.trackGeo(new THREE.TorusGeometry(0.82, 0.11, 20, 48));
    const ringMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#2563eb',
        emissive: '#1d4ed8',
        emissiveIntensity: 0.32,
        roughness: 0.18,
        clearcoat: 1,
      })
    );
    const attnRing = new THREE.Mesh(ringGeo, ringMat);
    attnRing.position.set(-1.25, 0.72, 0);
    attnRing.rotation.x = Math.PI * 0.38;
    attnRing.userData = { kind: 'attnRing', index: layerIdx } satisfies ClickTag;
    group.add(attnRing);
    this.interactives.push(attnRing);

    const attnLabelTex = this.trackTex(
      createPillLabelTexture('Attention Ring', 'Click to enter Harp', '#1d4ed8')
    );
    const attnLabel = new THREE.Sprite(
      this.trackMat(new THREE.SpriteMaterial({ map: attnLabelTex, transparent: true }))
    );
    attnLabel.scale.set(2.0, 0.7, 1);
    attnLabel.position.set(-1.25, 1.78, 0);
    group.add(attnLabel);

    // 4 Orbiting Head Orbs around the Attention Ring (click to inspect, double-click to unplug)
    const orbGeo = this.trackGeo(new THREE.SphereGeometry(0.19, 24, 24));
    for (let h = 0; h < 4; h++) {
      const off = isHeadDisabled(snap.ablation, layerIdx, h);
      const angle = (h / 4) * Math.PI * 2 + Math.PI / 4;
      const basePos = new THREE.Vector3(
        -1.25 + Math.cos(angle) * 0.92,
        0.72 + (off ? 0.55 : 0.12),
        Math.sin(angle) * 0.72
      );
      const orbMat = this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: off ? '#cbd5e1' : snap.head === h ? '#f59e0b' : '#3b82f6',
          emissive: off ? '#000000' : snap.head === h ? '#d97706' : '#1d4ed8',
          emissiveIntensity: off ? 0 : 0.45,
          roughness: 0.2,
          clearcoat: 1,
          transparent: true,
          opacity: off ? 0.45 : 0.96,
        })
      );
      const orb = new THREE.Mesh(orbGeo, orbMat);
      orb.position.copy(basePos);
      orb.userData = { kind: 'headOrb', index: h } satisfies ClickTag;
      group.add(orb);
      this.interactives.push(orb);
      this.headOrbs.push({ mesh: orb, head: h, basePos, off });
    }

    // Curved glass aqueduct B -> Thinking Core (MLP Prism, right side)
    const loopBCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.25, -0.55, 0),
      new THREE.Vector3(0.85, 0.65, -0.25),
      new THREE.Vector3(1.85, 0.85, 0),
      new THREE.Vector3(2.65, -0.55, 0),
    ]);
    const loopBGeo = this.trackGeo(new THREE.TubeGeometry(loopBCurve, 40, 0.055, 12, false));
    group.add(new THREE.Mesh(loopBGeo, glassPipeMat));

    const ln2 = ln1.clone();
    ln2.position.set(0.7, 0.3, -0.2);
    group.add(ln2);

    let mlpEnergy = 0.6;
    if (!layerTrace.skipped) {
      const last = layerTrace.mlp.output.length - 1;
      mlpEnergy = Math.min(1.4, 0.6 + vectorL2Norm(layerTrace.mlp.output[last]) * 0.08);
    }

    const prismGeo = this.trackGeo(new THREE.IcosahedronGeometry(0.66, 0));
    const prismMat = this.trackMat(
      new THREE.MeshPhysicalMaterial({
        color: '#f59e0b',
        emissive: '#b45309',
        emissiveIntensity: 0.28 * mlpEnergy,
        roughness: 0.18,
        metalness: 0.22,
        clearcoat: 1,
        flatShading: true,
      })
    );
    const prism = new THREE.Mesh(prismGeo, prismMat);
    prism.position.set(1.65, 0.72, 0);
    prism.scale.setScalar(0.85 + mlpEnergy * 0.14);
    group.add(prism);
    this.mlpCoreMesh = prism;

    const mlpLabelTex = this.trackTex(
      createPillLabelTexture('Thinking Core', 'MLP 128 → 512 → 128', '#b45309')
    );
    const mlpLabel = new THREE.Sprite(
      this.trackMat(new THREE.SpriteMaterial({ map: mlpLabelTex, transparent: true }))
    );
    mlpLabel.scale.set(2.05, 0.7, 1);
    mlpLabel.position.set(1.65, 1.78, 0);
    group.add(mlpLabel);
  }

  private buildHarpOfAttention(snap: WorldSnapshot): void {
    const activeRes = snap.ablated ?? snap.baseline;
    const layerTrace = activeRes.trace.layers[snap.layer];
    if (layerTrace.skipped) return;

    const active: ActiveLayerTrace = layerTrace;
    const tokens = activeRes.tokens;
    const T = tokens.length;
    const headIdx = snap.head;
    const weights = active.attn.attnWeights[headIdx];
    const headOff = isHeadDisabled(snap.ablation, snap.layer, headIdx);

    // Arrange porcelain word-tiles along a graceful sunlit 3D arc
    const arcRadius = Math.max(3.4, Math.min(5.2, 2.4 + T * 0.18));
    const spanAngle = Math.min(Math.PI * 0.78, Math.max(0.9, T * 0.13));
    const tilePositions: THREE.Vector3[] = [];

    const tileW = Math.max(0.46, Math.min(0.76, 6.4 / Math.max(T, 6)));
    const tileGeo = this.trackGeo(new RoundedBoxGeometry(tileW, 0.42, 0.16, 3, 0.06));
    const sideMat = this.trackMat(
      new THREE.MeshStandardMaterial({ color: '#f5f2eb', roughness: 0.35 })
    );

    for (let i = 0; i < T; i++) {
      const frac = T === 1 ? 0.5 : i / (T - 1);
      const angle = -spanAngle / 2 + frac * spanAngle;
      const x = Math.sin(angle) * arcRadius;
      const z = -Math.cos(angle) * arcRadius + arcRadius * 0.72;
      const pos = new THREE.Vector3(x, -0.55, z);
      tilePositions.push(pos);

      const isFocus =
        snap.hoveredTokenIdx === i || (snap.hoveredTokenIdx === null && i === T - 1);
      const tex = this.trackTex(
        createPorcelainTileTexture(formatToken(tokens[i].text), `#${i}`, isFocus)
      );
      const faceMat = this.trackMat(
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.28 })
      );
      const tile = new THREE.Mesh(tileGeo, [
        sideMat,
        sideMat,
        sideMat,
        sideMat,
        faceMat,
        faceMat,
      ]);
      tile.position.copy(pos);
      tile.lookAt(0, -0.4, 3.8);
      tile.castShadow = true;
      tile.userData = { kind: 'token', index: i } satisfies ClickTag;
      this.harpGroup.add(tile);
      this.interactives.push(tile);
    }

    // Build silky 3D glowing ribbons of light connecting queries to keys
    const beadGeo = this.trackGeo(new THREE.SphereGeometry(0.055, 10, 10));
    const beadMat = this.trackMat(new THREE.MeshBasicMaterial({ color: '#fef08a' }));

    for (let q = 0; q < T; q++) {
      for (let k = 0; k <= q; k++) {
        const w = weights[q][k];
        if (w < 0.045) continue;

        const p0 = tilePositions[q].clone().add(new THREE.Vector3(0, 0.24, 0));
        const p1 = tilePositions[k].clone().add(new THREE.Vector3(0, 0.24, 0));
        const dist = p0.distanceTo(p1);
        const mid = p0
          .clone()
          .add(p1)
          .multiplyScalar(0.5);
        mid.y += Math.max(0.55, 0.45 + dist * 0.34 + w * 0.85);
        if (q === k) {
          mid.y += 0.65;
          mid.z -= 0.35;
        }

        const curve = new THREE.CatmullRomCurve3([p0, mid, p1]);
        const radius = 0.014 + Math.pow(w, 0.85) * 0.068;
        const tubeGeo = this.trackGeo(new THREE.TubeGeometry(curve, 26, radius, 8, false));

        const color = new THREE.Color().lerpColors(
          new THREE.Color('#60a5fa'),
          new THREE.Color('#1d4ed8'),
          Math.min(1, w * 1.4)
        );
        if (w > 0.45) {
          color.lerp(new THREE.Color('#d97706'), (w - 0.45) * 0.85);
        }

        const mat = this.trackMat(
          new THREE.MeshPhysicalMaterial({
            color: headOff ? '#94a3b8' : color,
            emissive: headOff ? '#000000' : color,
            emissiveIntensity: headOff ? 0 : 0.25 + w * 0.55,
            roughness: 0.2,
            transparent: true,
            opacity: headOff ? 0.08 : Math.min(0.92, 0.16 + w * 0.85),
          })
        );
        const mesh = new THREE.Mesh(tubeGeo, mat);
        this.harpGroup.add(mesh);
        this.ribbons.push({ q, k, weight: w, mesh, mat, curve });

        // Traveling spark beads along the strongest attention bridges
        if (!headOff && w > 0.18 && this.pulseBeads.length < 36) {
          const bead = new THREE.Mesh(beadGeo, beadMat);
          this.harpGroup.add(bead);
          this.pulseBeads.push({
            mesh: bead,
            curve,
            offset: (q * 0.31 + k * 0.17) % 1,
            speed: 0.32 + w * 0.45,
          });
        }
      }
    }

    // 4 Orbiting Head Orbs floating above the Harp
    const orbGeo = this.trackGeo(new THREE.SphereGeometry(0.28, 28, 28));
    const socketGeo = this.trackGeo(new THREE.TorusGeometry(0.32, 0.035, 14, 32));
    const socketMat = this.trackMat(
      new THREE.MeshStandardMaterial({
        color: '#c8a46b',
        metalness: 0.78,
        roughness: 0.25,
      })
    );

    for (let h = 0; h < 4; h++) {
      const x = -2.1 + h * 1.4;
      const off = isHeadDisabled(snap.ablation, snap.layer, h);
      const isActiveHead = h === headIdx;

      const socket = new THREE.Mesh(socketGeo, socketMat);
      socket.rotation.x = Math.PI / 2;
      socket.position.set(x, 2.38, -0.5);
      this.harpGroup.add(socket);

      const basePos = new THREE.Vector3(x, off ? 3.05 : 2.52, off ? -0.95 : -0.5);
      const orbMat = this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: off ? '#cbd5e1' : isActiveHead ? '#f59e0b' : '#2563eb',
          emissive: off ? '#000000' : isActiveHead ? '#d97706' : '#1d4ed8',
          emissiveIntensity: off ? 0 : isActiveHead ? 0.55 : 0.28,
          roughness: 0.16,
          clearcoat: 1,
          transparent: true,
          opacity: off ? 0.45 : 0.96,
        })
      );
      const orb = new THREE.Mesh(orbGeo, orbMat);
      orb.position.copy(basePos);
      orb.userData = { kind: 'headOrb', index: h } satisfies ClickTag;
      this.harpGroup.add(orb);
      this.interactives.push(orb);
      this.headOrbs.push({ mesh: orb, head: h, basePos, off });

      const labelTex = this.trackTex(
        createPillLabelTexture(
          `Head ${h}`,
          off ? 'Unplugged' : isActiveHead ? 'Active' : 'Click to view',
          isActiveHead ? '#b45309' : '#1d4ed8',
          off
        )
      );
      const sprite = new THREE.Sprite(
        this.trackMat(new THREE.SpriteMaterial({ map: labelTex, transparent: true }))
      );
      sprite.scale.set(1.22, 0.43, 1);
      sprite.position.set(x, off ? 3.55 : 3.02, -0.5);
      this.harpGroup.add(sprite);
    }

    this.applyRibbonHighlight(snap.hoveredTokenIdx);
  }

  private buildExitBubbles(snap: WorldSnapshot): void {
    const activeRes = snap.ablated ?? snap.baseline;
    const isAblated = snap.ablated !== null;
    const baseProbs = snap.baseline.trace.finalProbs;
    const curProbs = activeRes.trace.finalProbs;

    const topItems = curProbs
      .map((prob, id) => ({
        id,
        prob,
        baseProb: baseProbs[id],
        text: formatToken(snap.tokenizer.vocab[id] ?? ''),
      }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 4);

    const anchorX =
      snap.level === 2
        ? (CHAMBER_X[snap.layer] ?? 0) + 4.35
        : snap.level === 3
          ? 4.75
          : 5.05;

    topItems.forEach((item, rank) => {
      const r = 0.24 + Math.sqrt(Math.max(0.01, item.prob)) * 0.62;
      const y = rank === 0 ? 0.55 : -0.35 + (rank - 1) * 0.72;
      const x = anchorX + (rank === 0 ? 0 : 1.05 + (rank % 2) * 0.35);
      const z = rank === 0 ? 0.2 : -0.45 + rank * 0.32;

      const bubbleGeo = this.trackGeo(new THREE.SphereGeometry(r, 32, 32));
      const bubbleMat = this.trackMat(
        new THREE.MeshPhysicalMaterial({
          color: rank === 0 ? (isAblated ? '#fb923c' : '#3b82f6') : '#93c5fd',
          emissive: rank === 0 ? (isAblated ? '#c2410c' : '#1d4ed8') : '#1e40af',
          emissiveIntensity: rank === 0 ? 0.24 : 0.08,
          roughness: 0.12,
          metalness: 0.05,
          transmission: 0.65,
          thickness: 0.7,
          ior: 1.38,
          clearcoat: 1,
          transparent: true,
          opacity: rank === 0 ? 0.94 : 0.74,
        })
      );
      const sphere = new THREE.Mesh(bubbleGeo, bubbleMat);
      sphere.position.set(x, y, z);
      sphere.castShadow = true;
      this.bubblesGroup.add(sphere);
      this.floaters.push({ obj: sphere, baseY: y, phase: rank * 1.4, amp: 0.09 });

      // Translucent before-ablation ghost ring so the probability shift is unmistakable
      if (isAblated && Math.abs(item.prob - item.baseProb) > 0.01) {
        const ghostR = 0.24 + Math.sqrt(Math.max(0.01, item.baseProb)) * 0.62;
        const ghostGeo = this.trackGeo(new THREE.TorusGeometry(ghostR, 0.022, 12, 40));
        const ghostMat = this.trackMat(
          new THREE.MeshBasicMaterial({
            color: '#1d4ed8',
            transparent: true,
            opacity: 0.55,
          })
        );
        const ghost = new THREE.Mesh(ghostGeo, ghostMat);
        ghost.position.set(x, y, z);
        this.bubblesGroup.add(ghost);
        this.floaters.push({ obj: ghost, baseY: y, phase: rank * 1.4, amp: 0.09 });
      }

      const deltaText =
        isAblated
          ? `${(item.prob * 100).toFixed(1)}% (was ${(item.baseProb * 100).toFixed(1)}%)`
          : `${(item.prob * 100).toFixed(1)}%`;

      const tex = this.trackTex(
        createPillLabelTexture(
          `“${item.text}”`,
          deltaText,
          rank === 0 ? (isAblated ? '#c2410c' : '#1d4ed8') : '#475569'
        )
      );
      const sprite = new THREE.Sprite(
        this.trackMat(new THREE.SpriteMaterial({ map: tex, transparent: true }))
      );
      const scale = rank === 0 ? 1.65 : 1.18;
      sprite.scale.set(scale, scale * 0.35, 1);
      sprite.position.set(x, y + r + 0.32, z);
      this.bubblesGroup.add(sprite);
      this.floaters.push({
        obj: sprite,
        baseY: y + r + 0.32,
        phase: rank * 1.4,
        amp: 0.09,
      });
    });
  }

  private applyRibbonHighlight(hoveredIdx: number | null): void {
    if (!this.snapshot || this.snapshot.level !== 3) return;
    const headOff = isHeadDisabled(
      this.snapshot.ablation,
      this.snapshot.layer,
      this.snapshot.head
    );
    if (headOff) return;

    const T = (this.snapshot.ablated ?? this.snapshot.baseline).tokens.length;
    const focusIdx = hoveredIdx ?? T - 1;

    for (const r of this.ribbons) {
      const connected = r.q === focusIdx || r.k === focusIdx;
      if (connected) {
        r.mat.opacity = Math.min(0.96, 0.32 + r.weight * 0.78);
        r.mat.emissiveIntensity = 0.45 + r.weight * 0.85;
      } else {
        r.mat.opacity = hoveredIdx !== null ? 0.06 : Math.min(0.28, 0.08 + r.weight * 0.25);
        r.mat.emissiveIntensity = 0.08;
      }
    }
  }

  private pickObject(event: PointerEvent | MouseEvent): ClickTag | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.interactives, false);
    for (const hit of hits) {
      if (!hit.object.visible) continue;
      let p: THREE.Object3D | null = hit.object;
      let visible = true;
      while (p) {
        if (!p.visible) {
          visible = false;
          break;
        }
        p = p.parent;
      }
      if (!visible) continue;
      const tag = hit.object.userData as ClickTag | undefined;
      if (tag?.kind) return tag;
    }
    return null;
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    const hit = this.pickObject(event);
    this.renderer.domElement.style.cursor = hit ? 'pointer' : 'grab';

    if (hit?.kind === 'token') {
      if (this.lastHoverToken !== hit.index) {
        this.lastHoverToken = hit.index;
        studioSound.playHarpPluck(0.7, hit.index);
        this.callbacks.onHoverToken(hit.index);
      }
    } else if (this.lastHoverToken !== null) {
      this.lastHoverToken = null;
      this.callbacks.onHoverToken(null);
    }
  };

  private readonly onClick = (event: MouseEvent): void => {
    if (!this.snapshot) return;
    const hit = this.pickObject(event);
    if (!hit) return;

    if (hit.kind === 'box' && this.snapshot.level === 0) {
      this.callbacks.onSelectLevel(1);
    } else if (hit.kind === 'chamber') {
      const alreadySelected = this.snapshot.layer === hit.index;
      this.callbacks.onSelectChamber(hit.index, alreadySelected || this.snapshot.level === 1);
    } else if (hit.kind === 'attnRing') {
      this.callbacks.onSelectLevel(3);
    } else if (hit.kind === 'headOrb') {
      if (event.shiftKey) {
        const currentlyOff = isHeadDisabled(
          this.snapshot.ablation,
          this.snapshot.layer,
          hit.index
        );
        studioSound.playPop(!currentlyOff);
        this.callbacks.onToggleHeadAblation(this.snapshot.layer, hit.index);
      } else {
        studioSound.playTileTap(hit.index);
        this.callbacks.onSelectHead(hit.index);
      }
    } else if (hit.kind === 'token') {
      studioSound.playHarpPluck(0.9, hit.index);
      this.callbacks.onHoverToken(hit.index);
    }
  };

  private readonly onDoubleClick = (event: MouseEvent): void => {
    if (!this.snapshot) return;
    const hit = this.pickObject(event);
    if (hit?.kind === 'headOrb') {
      const currentlyOff = isHeadDisabled(
        this.snapshot.ablation,
        this.snapshot.layer,
        hit.index
      );
      studioSound.playPop(!currentlyOff);
      this.callbacks.onToggleHeadAblation(this.snapshot.layer, hit.index);
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
    this.pulseTime += dt;

    // Smooth camera flight between semantic depths
    if (this.needsCameraFlight && !this.isUserDragging) {
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

    // Smoothly lift and dissolve outer frosted lid when transitioning from Level 0 -> Level 1
    const lidTargetY = this.snapshot?.level === 0 ? 0.1 : 3.6;
    const lidTargetOpacity = this.snapshot?.level === 0 ? 0.94 : 0.0;
    this.outerShellMesh.position.y +=
      (lidTargetY - this.outerShellMesh.position.y) * Math.min(1, dt * 6);
    this.outerShellMat.opacity +=
      (lidTargetOpacity - this.outerShellMat.opacity) * Math.min(1, dt * 6);
    this.outerShellMesh.visible = this.outerShellMat.opacity > 0.03;

    // Luminous pulse sweeping left-to-right through the box & factory
    const waveFrac = (this.pulseTime * 0.42) % 1;
    this.shellPulseRing.position.x = -3.1 + waveFrac * 6.2;
    this.shellPulseRing.visible = this.snapshot?.level === 0;
    if (this.factoryPulseLight) {
      this.factoryPulseLight.position.x = -4.4 + waveFrac * 8.8;
    }

    // Gentle floating buoyancy for bubbles, thought clouds, and porcelain tiles
    for (const f of this.floaters) {
      f.obj.position.y = f.baseY + Math.sin(t * 1.9 + f.phase) * f.amp;
    }

    // Gentle orb hover + slow crystalline prism rotation
    for (const orb of this.headOrbs) {
      orb.mesh.position.y =
        orb.basePos.y + Math.sin(t * 2.2 + orb.head * 1.5) * (orb.off ? 0.03 : 0.07);
    }
    if (this.mlpCoreMesh) {
      this.mlpCoreMesh.rotation.y += dt * 0.55;
      this.mlpCoreMesh.rotation.x = Math.sin(t * 0.8) * 0.22;
    }

    // Animate golden spark beads flowing along attention harp ribbons
    for (const b of this.pulseBeads) {
      const u = (t * b.speed + b.offset) % 1;
      b.curve.getPointAt(u, b.mesh.position);
    }

    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animFrameId);
    window.removeEventListener('resize', this.onResize);
    this.renderer.domElement.removeEventListener('pointermove', this.onPointerMove);
    this.renderer.domElement.removeEventListener('click', this.onClick);
    this.renderer.domElement.removeEventListener('dblclick', this.onDoubleClick);
    this.clearDynamic();
    this.controls.dispose();
    this.envRT.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
