/**
 * Three.js view of the cube.
 *
 * The logical state (`CubieState`) is the single source of truth. The 26
 * cubies never move permanently: a turn temporarily parents one layer to a
 * pivot, animates it, then snaps the cubies back home and repaints every
 * sticker from the new state. Rendering can therefore never drift from the
 * model.
 */
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import {
  FACELET_PLACEMENTS,
  applyMove,
  layerTurnToMove,
  moveRotation,
  solved,
  toFacelets,
  type Axis,
  type CubieState,
  type Move,
  type Vec3,
} from "../cube";

/** Sticker colors, indexed like FACES (U R F D L B). */
export const STICKER_COLORS = ["#f4f4f5", "#dc2626", "#16a34a", "#facc15", "#f97316", "#2563eb"];

export type MoveSource = "user" | "scramble" | "solver";

export interface MoveEventDetail {
  move: Move;
  source: MoveSource;
  state: CubieState;
}

export interface RenderStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  queued: number;
  pixelRatio: number;
}

interface QueuedMove {
  move: Move;
  source: MoveSource;
  duration: number;
}

interface ActiveTurn extends QueuedMove {
  axis: Axis;
  angle: number;
  start: number;
  cubies: THREE.Object3D[];
}

type Pointer =
  | { mode: "orbit"; id: number; x: number; y: number }
  | { mode: "turn"; id: number; x: number; y: number; pos: Vec3; normal: Vec3; point: THREE.Vector3 }
  | { mode: "spent"; id: number };

const QUARTER_TURN_MS = { user: 170, scramble: 70, solver: 280 } as const;
const DRAG_THRESHOLD_PX = 10;
const ORBIT_SPEED = 0.008;
const MAX_PITCH = 1.35;

const AXIS_NAME = ["x", "y", "z"] as const;
const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function roundedRect(size: number, radius: number): THREE.ShapeGeometry {
  const h = size / 2;
  const s = new THREE.Shape();
  s.moveTo(-h + radius, -h);
  s.lineTo(h - radius, -h);
  s.quadraticCurveTo(h, -h, h, -h + radius);
  s.lineTo(h, h - radius);
  s.quadraticCurveTo(h, h, h - radius, h);
  s.lineTo(-h + radius, h);
  s.quadraticCurveTo(-h, h, -h, h - radius);
  s.lineTo(-h, -h + radius);
  s.quadraticCurveTo(-h, -h, -h + radius, -h);
  return new THREE.ShapeGeometry(s, 6);
}

export class CubeView extends EventTarget {
  private state: CubieState = solved();
  private queue: QueuedMove[] = [];
  private active: ActiveTurn | null = null;
  private pointer: Pointer | null = null;
  private turnsLocked = false;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  private readonly cube = new THREE.Group();
  private readonly pivot = new THREE.Group();
  private readonly cubies = new Map<string, THREE.Object3D>();
  private readonly stickers: THREE.Mesh[] = [];
  private readonly stickerMaterials: THREE.MeshStandardMaterial[];
  private readonly raycaster = new THREE.Raycaster();
  private readonly resizeObserver: ResizeObserver;

  private yaw = -0.6;
  private pitch = 0.5;
  private frames = 0;
  private fpsWindowStart = performance.now();
  private fps = 0;

  constructor(private readonly container: HTMLElement) {
    super();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const canvas = this.renderer.domElement;
    canvas.style.touchAction = "none";
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(4, 8, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-6, -3, 4);
    this.scene.add(fill);

    this.stickerMaterials = STICKER_COLORS.map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0 }),
    );
    this.buildCubies();
    this.cube.add(this.pivot);
    this.scene.add(this.cube);
    this.applyOrientation();
    this.paint();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("pointercancel", this.onPointerUp);

    this.renderer.setAnimationLoop(this.frame);
  }

  /** Logical state with all completed moves applied (queued moves excluded). */
  getState(): CubieState {
    return this.state;
  }

  get busy(): boolean {
    return this.active !== null || this.queue.length > 0;
  }

  enqueue(moves: readonly Move[], source: MoveSource = "user"): void {
    for (const move of moves) {
      const quarter = QUARTER_TURN_MS[source];
      this.queue.push({ move, source, duration: move.turns === 2 ? quarter * 1.5 : quarter });
    }
  }

  /** Drops pending moves, cancels the running animation and shows `state`. */
  setState(state: CubieState): void {
    const wasBusy = this.busy;
    this.queue = [];
    if (this.active) this.endTurn(false);
    this.state = state;
    this.paint();
    if (wasBusy) this.dispatchEvent(new Event("idle"));
  }

  /** While locked, dragging a face rotates the view instead of turning it. */
  setTurnsLocked(locked: boolean): void {
    this.turnsLocked = locked;
  }

  stats(): RenderStats {
    const info = this.renderer.info.render;
    return {
      fps: this.fps,
      drawCalls: info.calls,
      triangles: info.triangles,
      queued: this.queue.length + (this.active ? 1 : 0),
      pixelRatio: this.renderer.getPixelRatio(),
    };
  }

  dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // --- Scene construction ---------------------------------------------------

  private buildCubies(): void {
    const body = new RoundedBoxGeometry(0.96, 0.96, 0.96, 3, 0.1);
    const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.55 });
    for (let x = -1; x <= 1; x++)
      for (let y = -1; y <= 1; y++)
        for (let z = -1; z <= 1; z++) {
          if (x === 0 && y === 0 && z === 0) continue;
          const cubie = new THREE.Group();
          cubie.position.set(x, y, z);
          const mesh = new THREE.Mesh(body, bodyMaterial);
          mesh.userData = { pos: [x, y, z] };
          cubie.add(mesh);
          this.cube.add(cubie);
          this.cubies.set(`${x},${y},${z}`, cubie);
        }

    const stickerGeometry = roundedRect(0.84, 0.12);
    const forward = new THREE.Vector3(0, 0, 1);
    for (const { pos, normal } of FACELET_PLACEMENTS) {
      const sticker = new THREE.Mesh(stickerGeometry, this.stickerMaterials[0]);
      const n = new THREE.Vector3(...normal);
      sticker.position.copy(n).multiplyScalar(0.483);
      sticker.quaternion.setFromUnitVectors(forward, n);
      sticker.userData = { pos, normal };
      this.cubies.get(pos.join(","))!.add(sticker);
      this.stickers.push(sticker);
    }
  }

  private paint(): void {
    toFacelets(this.state).forEach((color, i) => {
      this.stickers[i].material = this.stickerMaterials[color];
    });
  }

  private applyOrientation(): void {
    const qPitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), this.pitch);
    const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);
    this.cube.quaternion.copy(qPitch.multiply(qYaw));
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Fit a sphere of radius ~3.1 around the cube in the smaller dimension.
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fit = 3.1 / Math.sin(halfFov) / Math.min(1, this.camera.aspect);
    this.camera.position.set(0, 0, fit);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }

  // --- Animation --------------------------------------------------------------

  private frame = (now: number): void => {
    if (!this.active && this.queue.length > 0) this.startTurn(now);
    if (this.active) {
      const { axis, angle, start, duration } = this.active;
      // Catch up when the user queues many moves.
      const speedup = 1 + Math.min(this.queue.length, 6) * 0.25;
      const t = Math.min(1, ((now - start) * speedup) / duration);
      this.pivot.rotation[AXIS_NAME[axis]] = angle * easeOutCubic(t);
      if (t >= 1) {
        this.endTurn(true);
        if (this.queue.length > 0) this.startTurn(now);
        else this.dispatchEvent(new Event("idle"));
      }
    }

    this.renderer.render(this.scene, this.camera);

    this.frames++;
    if (now - this.fpsWindowStart >= 500) {
      this.fps = (this.frames * 1000) / (now - this.fpsWindowStart);
      this.frames = 0;
      this.fpsWindowStart = now;
    }
  };

  private startTurn(now: number): void {
    const next = this.queue.shift()!;
    const { axis, layer, quarterTurns } = moveRotation(next.move);
    const cubies = [...this.cubies.entries()]
      .filter(([key]) => Number(key.split(",")[axis]) === layer)
      .map(([, c]) => c);
    this.pivot.rotation.set(0, 0, 0);
    for (const c of cubies) this.pivot.attach(c);
    this.active = { ...next, axis, angle: (quarterTurns * Math.PI) / 2, start: now, cubies };
  }

  private endTurn(commit: boolean): void {
    const turn = this.active!;
    this.active = null;
    for (const [key, cubie] of this.cubies) {
      if (!turn.cubies.includes(cubie)) continue;
      this.cube.attach(cubie);
      cubie.position.set(...(key.split(",").map(Number) as Vec3));
      cubie.quaternion.identity();
    }
    this.pivot.rotation.set(0, 0, 0);
    if (!commit) return;
    this.state = applyMove(this.state, turn.move);
    this.paint();
    this.dispatchEvent(
      new CustomEvent<MoveEventDetail>("move", {
        detail: { move: turn.move, source: turn.source, state: this.state },
      }),
    );
  }

  // --- Pointer interaction ----------------------------------------------------

  private onPointerDown = (e: PointerEvent): void => {
    if (this.pointer || e.button !== 0) return;
    const canvas = this.renderer.domElement;
    canvas.setPointerCapture(e.pointerId);
    const hit = this.turnsLocked ? null : this.pick(e);
    this.pointer = hit
      ? { mode: "turn", id: e.pointerId, x: e.clientX, y: e.clientY, ...hit }
      : { mode: "orbit", id: e.pointerId, x: e.clientX, y: e.clientY };
    canvas.style.cursor = "grabbing";
  };

  private onPointerMove = (e: PointerEvent): void => {
    const p = this.pointer;
    if (!p || p.id !== e.pointerId) {
      if (!p) this.renderer.domElement.style.cursor = !this.turnsLocked && this.pick(e) ? "pointer" : "grab";
      return;
    }
    if (p.mode === "orbit") {
      this.yaw += (e.clientX - p.x) * ORBIT_SPEED;
      this.pitch = THREE.MathUtils.clamp(this.pitch + (e.clientY - p.y) * ORBIT_SPEED, -MAX_PITCH, MAX_PITCH);
      p.x = e.clientX;
      p.y = e.clientY;
      this.applyOrientation();
    } else if (p.mode === "turn") {
      const drag = new THREE.Vector2(e.clientX - p.x, e.clientY - p.y);
      if (drag.length() < DRAG_THRESHOLD_PX) return;
      this.enqueue([this.dragToMove(p.pos, p.normal, p.point, drag.normalize())], "user");
      this.pointer = { mode: "spent", id: p.id };
    }
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.pointer?.id !== e.pointerId) return;
    this.pointer = null;
    this.renderer.domElement.style.cursor = "grab";
  };

  /** Finds the outer face of a cubie under the pointer, in cube-local coordinates. */
  private pick(e: PointerEvent): { pos: Vec3; normal: Vec3; point: THREE.Vector3 } | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObject(this.cube, true)[0];
    if (!hit) return null;

    const pos = hit.object.userData.pos as Vec3;
    if (hit.object.userData.normal) return { pos, normal: hit.object.userData.normal, point: hit.point };

    // Body hit (gap between stickers): use the dominant axis of the local hit point.
    const local = this.cube.worldToLocal(hit.point.clone()).sub(new THREE.Vector3(...pos));
    const comps = local.toArray();
    const axis = comps.reduce((best, v, i) => (Math.abs(v) > Math.abs(comps[best]) ? i : best), 0);
    const normal: Vec3 = [0, 0, 0];
    normal[axis] = Math.sign(comps[axis]);
    if (pos[axis] !== normal[axis]) return null; // inner face, seen during a turn
    return { pos, normal, point: hit.point };
  }

  /** Picks the in-plane direction whose screen projection best matches the drag. */
  private dragToMove(pos: Vec3, normal: Vec3, point: THREE.Vector3, drag: THREE.Vector2): Move {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const toScreen = (v: THREE.Vector3) => {
      const p = v.clone().project(this.camera);
      return new THREE.Vector2((p.x * rect.width) / 2, (-p.y * rect.height) / 2);
    };
    const origin = toScreen(point);

    let best = { dir: [0, 0, 0] as Vec3, score: -Infinity };
    for (let axis = 0; axis < 3; axis++) {
      if (normal[axis] !== 0) continue;
      for (const sign of [1, -1]) {
        const dir: Vec3 = [0, 0, 0];
        dir[axis] = sign;
        const worldDir = new THREE.Vector3(...dir).applyQuaternion(this.cube.quaternion);
        const screenDir = toScreen(point.clone().add(worldDir)).sub(origin).normalize();
        const score = screenDir.dot(drag);
        if (score > best.score) best = { dir, score };
      }
    }

    // Rotating around n × d moves the touched face along d.
    const rot = cross(normal, best.dir);
    const axis = rot.findIndex((v) => v !== 0) as Axis;
    return layerTurnToMove(axis, pos[axis], rot[axis] as 1 | -1);
  }
}
