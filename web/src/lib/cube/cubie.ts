/**
 * Cubie-level representation of a 3x3x3 cube, following Kociemba's conventions.
 *
 * Positions and cubies share the same names:
 *   corners: URF UFL ULB UBR DFR DLF DBL DRB
 *   edges:   UR UF UL UB DR DF DL DB FR FL BL BR
 *
 * `cp[i]` is the corner cubie sitting at position i, `co[i]` its twist (0..2).
 * `ep[i]` / `eo[i]` are the same for edges (flip 0..1).
 * `ctr[i]` is the center sitting on face i (URFDLB); only slice moves change it.
 *
 * Applying a fixed move is a public permutation of positions plus an addition
 * of orientations mod 3 / mod 2, which is what makes this layout cheap to
 * evaluate homomorphically later on.
 */
import { FACELET_PLACEMENTS, moveRotation, rotateVec, type Axis } from "./geometry";
import { FACES, SLICES, type Face, type Layer, type Move, type Slice } from "./moves";

export interface CubieState {
  cp: number[];
  co: number[];
  ep: number[];
  eo: number[];
  ctr: number[];
}

export const CORNER_NAMES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"] as const;
export const EDGE_NAMES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"] as const;

export function solved(): CubieState {
  return {
    cp: [0, 1, 2, 3, 4, 5, 6, 7],
    co: [0, 0, 0, 0, 0, 0, 0, 0],
    ep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ctr: [0, 1, 2, 3, 4, 5],
  };
}

export function clone(s: CubieState): CubieState {
  return { cp: [...s.cp], co: [...s.co], ep: [...s.ep], eo: [...s.eo], ctr: [...s.ctr] };
}

/** Clockwise quarter turn of each face, in "replaced by" form (centers fixed). */
export const FACE_MOVE_TABLES: Record<Face, Omit<CubieState, "ctr">> = {
  U: {
    cp: [3, 0, 1, 2, 4, 5, 6, 7],
    co: [0, 0, 0, 0, 0, 0, 0, 0],
    ep: [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
    eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  R: {
    cp: [4, 1, 2, 0, 7, 5, 6, 3],
    co: [2, 0, 0, 1, 1, 0, 0, 2],
    ep: [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  F: {
    cp: [1, 5, 2, 3, 0, 4, 6, 7],
    co: [1, 2, 0, 0, 2, 1, 0, 0],
    ep: [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    eo: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
  },
  D: {
    cp: [0, 1, 2, 3, 5, 6, 7, 4],
    co: [0, 0, 0, 0, 0, 0, 0, 0],
    ep: [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
    eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  L: {
    cp: [0, 2, 6, 3, 4, 1, 5, 7],
    co: [0, 1, 2, 0, 0, 2, 1, 0],
    ep: [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
    eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  B: {
    cp: [0, 1, 3, 7, 4, 5, 2, 6],
    co: [0, 0, 1, 2, 0, 0, 2, 1],
    ep: [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
    eo: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
  },
};

/** Returns `a * b`: the state `a` followed by the move/state `b`. */
export function multiply(a: CubieState, b: CubieState): CubieState {
  return {
    cp: b.cp.map((p) => a.cp[p]),
    co: b.cp.map((p, i) => (a.co[p] + b.co[i]) % 3),
    ep: b.ep.map((p) => a.ep[p]),
    eo: b.ep.map((p, i) => (a.eo[p] + b.eo[i]) % 2),
    ctr: b.ctr.map((p) => a.ctr[p]),
  };
}

export function applyMove(s: CubieState, m: Move): CubieState {
  let out = s;
  for (let i = 0; i < m.turns; i++) out = multiply(out, MOVE_TABLES[m.layer]);
  return out;
}

export function applyMoves(s: CubieState, ms: readonly Move[]): CubieState {
  return ms.reduce(applyMove, s);
}

/** True when every face shows a single color, whatever the cube's orientation. */
export function isSolved(s: CubieState): boolean {
  const f = toFacelets(s);
  return f.every((c, i) => c === f[i - (i % 9) + 4]);
}

export function equals(a: CubieState, b: CubieState): boolean {
  const eq = (x: number[], y: number[]) => x.every((v, i) => v === y[i]);
  return eq(a.cp, b.cp) && eq(a.co, b.co) && eq(a.ep, b.ep) && eq(a.eo, b.eo) && eq(a.ctr, b.ctr);
}

// --- Facelet view -----------------------------------------------------------
//
// Facelets are indexed U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9 (0..53), each
// face read row by row as seen from outside the cube (see geometry.ts).

const U = 0, R = 9, F = 18, D = 27, L = 36, B = 45;

const CORNER_FACELETS: readonly (readonly number[])[] = [
  [U + 8, R + 0, F + 2], [U + 6, F + 0, L + 2], [U + 0, L + 0, B + 2], [U + 2, B + 0, R + 2],
  [D + 2, F + 8, R + 6], [D + 0, L + 8, F + 6], [D + 6, B + 8, L + 6], [D + 8, R + 8, B + 6],
];

const EDGE_FACELETS: readonly (readonly number[])[] = [
  [U + 5, R + 1], [U + 7, F + 1], [U + 3, L + 1], [U + 1, B + 1],
  [D + 5, R + 7], [D + 1, F + 7], [D + 3, L + 7], [D + 7, B + 7],
  [F + 5, R + 3], [F + 3, L + 5], [B + 5, L + 3], [B + 3, R + 5],
];

/** Face colors of each cubie, as indices into FACES. */
const CORNER_COLORS = [
  [0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5],
] as const;
const EDGE_COLORS = [
  [0, 1], [0, 2], [0, 4], [0, 5], [3, 1], [3, 2], [3, 4], [3, 5], [2, 1], [2, 4], [5, 4], [5, 1],
] as const;

const CENTER_FACELETS = [U + 4, R + 4, F + 4, D + 4, L + 4, B + 4];

/** Color of each of the 54 facelets, as an index into FACES (0 = U color, ...). */
export function toFacelets(s: CubieState): number[] {
  const out = new Array<number>(54);
  CENTER_FACELETS.forEach((f, i) => (out[f] = s.ctr[i]));
  for (let i = 0; i < 8; i++) {
    for (let n = 0; n < 3; n++) {
      out[CORNER_FACELETS[i][(n + s.co[i]) % 3]] = CORNER_COLORS[s.cp[i]][n];
    }
  }
  for (let i = 0; i < 12; i++) {
    for (let n = 0; n < 2; n++) {
      out[EDGE_FACELETS[i][(n + s.eo[i]) % 2]] = EDGE_COLORS[s.ep[i]][n];
    }
  }
  return out;
}

/** 54-character facelet string in URFDLB order, e.g. "UUUUUUUUURRR...". */
export function toFaceletString(s: CubieState): string {
  return toFacelets(s).map((c) => FACES[c]).join("");
}

// --- Moves derived from geometry ----------------------------------------------

const FACELET_AT = new Map(FACELET_PLACEMENTS.map(({ pos, normal }, i) => [`${pos}|${normal}`, i]));

/**
 * Cubie-level effect of rotating one layer by `quarterTurns` around +axis,
 * derived by following stickers in 3D. A cubie moving from position i to j
 * gets a twist equal to the slot of j that receives slot 0 of i.
 */
export function layerTurnTable(axis: Axis, layer: number, quarterTurns: number): CubieState {
  const turn = (f: number): number => {
    const { pos, normal } = FACELET_PLACEMENTS[f];
    if (pos[axis] !== layer) return f;
    const key = `${rotateVec(pos, axis, quarterTurns)}|${rotateVec(normal, axis, quarterTurns)}`;
    return FACELET_AT.get(key)!;
  };
  const out = solved();
  const follow = (slots: readonly (readonly number[])[], p: number[], o: number[]) =>
    slots.forEach((src, i) => {
      const moved = turn(src[0]);
      const j = slots.findIndex((s) => s.includes(moved));
      p[j] = i;
      o[j] = slots[j].indexOf(moved);
    });
  follow(CORNER_FACELETS, out.cp, out.co);
  follow(EDGE_FACELETS, out.ep, out.eo);
  CENTER_FACELETS.forEach((f, i) => (out.ctr[CENTER_FACELETS.indexOf(turn(f))] = i));
  return out;
}

function sliceTable(slice: Slice): CubieState {
  const { axis, layer, quarterTurns } = moveRotation({ layer: slice, turns: 1 });
  return layerTurnTable(axis, layer, quarterTurns);
}

const MOVE_TABLES = Object.fromEntries([
  ...FACES.map((f) => [f, { ...FACE_MOVE_TABLES[f], ctr: solved().ctr }]),
  ...SLICES.map((s) => [s, sliceTable(s)]),
]) as Record<Layer, CubieState>;
