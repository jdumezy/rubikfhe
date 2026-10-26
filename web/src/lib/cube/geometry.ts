/**
 * Mapping between the abstract cube (facelets, moves) and 3D space.
 *
 * Right-handed coordinates, cubie centers on the integer grid {-1, 0, 1}^3:
 * +x = R, +y = U, +z = F.
 */
import { FACES, LAYERS, type Face, type Layer, type Move } from "./moves";

export type Vec3 = [number, number, number];
export type Axis = 0 | 1 | 2;

/** Outward normal of each face. */
export const FACE_NORMAL: Record<Face, Vec3> = {
  U: [0, 1, 0],
  R: [1, 0, 0],
  F: [0, 0, 1],
  D: [0, -1, 0],
  L: [-1, 0, 0],
  B: [0, 0, -1],
};

/**
 * Cubie position of facelet (row, col) on each face, reading the face from
 * outside with U on top (for U: B on top, for D: F on top).
 */
const FACE_GRID: Record<Face, (r: number, c: number) => Vec3> = {
  U: (r, c) => [c - 1, 1, r - 1],
  R: (r, c) => [1, 1 - r, 1 - c],
  F: (r, c) => [c - 1, 1 - r, 1],
  D: (r, c) => [c - 1, -1, 1 - r],
  L: (r, c) => [-1, 1 - r, c - 1],
  B: (r, c) => [1 - c, 1 - r, -1],
};

export interface FaceletPlacement {
  /** Center of the cubie carrying the facelet. */
  pos: Vec3;
  /** Outward normal of the sticker. */
  normal: Vec3;
}

/** Placement of facelet i (0..53, URFDLB order). */
export function faceletPlacement(i: number): FaceletPlacement {
  const face = FACES[Math.floor(i / 9)];
  const k = i % 9;
  return { pos: FACE_GRID[face](Math.floor(k / 3), k % 3), normal: FACE_NORMAL[face] };
}

export const FACELET_PLACEMENTS: readonly FaceletPlacement[] = Array.from({ length: 54 }, (_, i) =>
  faceletPlacement(i),
);

interface LayerSpec {
  axis: Axis;
  /** Coordinate of the layer along `axis` (-1, 0 or 1). */
  coord: number;
  /** Direction of a clockwise quarter turn around +axis (right-hand rule). */
  clockwise: 1 | -1;
}

/**
 * A clockwise face turn is a -90° rotation around the face's outward normal.
 * Slices turn like their reference face: M like L, E like D, S like F.
 */
const LAYER_SPEC: Record<Layer, LayerSpec> = {
  U: { axis: 1, coord: 1, clockwise: -1 },
  R: { axis: 0, coord: 1, clockwise: -1 },
  F: { axis: 2, coord: 1, clockwise: -1 },
  D: { axis: 1, coord: -1, clockwise: 1 },
  L: { axis: 0, coord: -1, clockwise: 1 },
  B: { axis: 2, coord: -1, clockwise: 1 },
  M: { axis: 0, coord: 0, clockwise: 1 },
  E: { axis: 1, coord: 0, clockwise: 1 },
  S: { axis: 2, coord: 0, clockwise: -1 },
};

export interface LayerRotation {
  axis: Axis;
  /** Coordinate of the turning layer along `axis` (-1, 0 or 1). */
  layer: number;
  /** Signed number of quarter turns around +axis (right-hand rule). */
  quarterTurns: number;
}

export function moveRotation(m: Move): LayerRotation {
  const { axis, coord, clockwise } = LAYER_SPEC[m.layer];
  const signedTurns = m.turns === 3 ? -1 : m.turns;
  return { axis, layer: coord, quarterTurns: clockwise * signedTurns };
}

/**
 * Inverse of `moveRotation`: the move equivalent to rotating `layer` by one
 * quarter turn around +axis (direction ±1).
 */
export function layerTurnToMove(axis: Axis, layer: number, direction: 1 | -1): Move {
  const name = LAYERS.find((l) => LAYER_SPEC[l].axis === axis && LAYER_SPEC[l].coord === layer)!;
  return { layer: name, turns: direction === LAYER_SPEC[name].clockwise ? 1 : 3 };
}

/** Rotates an integer vector by `quarterTurns` × 90° around `axis`. */
export function rotateVec(v: Vec3, axis: Axis, quarterTurns: number): Vec3 {
  let out: Vec3 = [...v];
  const q = ((quarterTurns % 4) + 4) % 4;
  const [a, b] = axis === 0 ? [1, 2] : axis === 1 ? [2, 0] : [0, 1];
  for (let i = 0; i < q; i++) {
    const next: Vec3 = [...out];
    next[a] = -out[b];
    next[b] = out[a];
    out = next;
  }
  return out;
}
