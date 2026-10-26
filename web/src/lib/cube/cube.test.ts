import { describe, expect, it } from "vitest";
import {
  FACELET_PLACEMENTS,
  FACES,
  FACE_MOVES,
  FACE_MOVE_TABLES,
  SLICE_MOVES,
  applyMove,
  applyMoves,
  equals,
  formatMoves,
  invertMoves,
  isSolved,
  layerTurnTable,
  layerTurnToMove,
  moveRotation,
  parseMoves,
  randomScramble,
  rotateVec,
  solved,
  toFaceletString,
  toFacelets,
  type Move,
} from "./index";

const ALL_MOVES = [...FACE_MOVES, ...SLICE_MOVES];

/** Facelets after `m`, computed purely geometrically by rotating stickers in 3D. */
function geometricFacelets(before: number[], m: Move): number[] {
  const { axis, layer, quarterTurns } = moveRotation(m);
  const key = (p: number[], n: number[]) => `${p}|${n}`;
  const index = new Map(FACELET_PLACEMENTS.map((f, i) => [key(f.pos, f.normal), i]));
  const after = [...before];
  FACELET_PLACEMENTS.forEach(({ pos, normal }, i) => {
    if (pos[axis] !== layer) return;
    const j = index.get(key(rotateVec(pos, axis, quarterTurns), rotateVec(normal, axis, quarterTurns)))!;
    after[j] = before[i];
  });
  return after;
}

const seededRandom = (seed: number) => (n: number) => {
  seed = (seed * 1103515245 + 12345) % 2 ** 31;
  return seed % n;
};

describe("cubie model", () => {
  it("starts solved", () => {
    expect(isSolved(solved())).toBe(true);
    expect(toFaceletString(solved())).toBe(FACES.map((f) => f.repeat(9)).join(""));
  });

  it("every quarter turn has order 4", () => {
    for (const m of ALL_MOVES.filter((m) => m.turns === 1)) {
      const s = applyMoves(solved(), Array(4).fill(m));
      expect(equals(s, solved())).toBe(true);
    }
  });

  it("geometric derivation reproduces Kociemba's face move tables", () => {
    for (const face of FACES) {
      const { axis, layer, quarterTurns } = moveRotation({ layer: face, turns: 1 });
      expect(layerTurnTable(axis, layer, quarterTurns)).toEqual({ ...FACE_MOVE_TABLES[face], ctr: solved().ctr });
    }
  });

  it("slices move centers, and R L' M' is a whole-cube rotation", () => {
    expect(applyMove(solved(), { layer: "M", turns: 1 }).ctr).not.toEqual(solved().ctr);
    const x = applyMoves(solved(), parseMoves("R L' M'"));
    expect(equals(x, solved())).toBe(false);
    expect(isSolved(x)).toBe(true);
    expect(isSolved(applyMoves(solved(), parseMoves("M E S")))).toBe(false);
  });

  it("M2 E2 S2 equals the pons asinorum U2 D2 F2 B2 L2 R2", () => {
    const slices = applyMoves(solved(), parseMoves("M2 E2 S2"));
    expect(equals(slices, applyMoves(solved(), parseMoves("U2 D2 F2 B2 L2 R2")))).toBe(true);
  });

  it("(R U R' U') has order 6", () => {
    const sexy = parseMoves("R U R' U'");
    let s = solved();
    for (let i = 1; i <= 6; i++) {
      s = applyMoves(s, sexy);
      expect(isSolved(s)).toBe(i === 6);
    }
  });

  it("a sequence followed by its inverse is the identity", () => {
    const seq = [...randomScramble(40, seededRandom(42)), ...parseMoves("M E' S2 M' U S")];
    expect(isSolved(applyMoves(solved(), [...seq, ...invertMoves(seq)]))).toBe(true);
  });

  it("X2 equals X X and X' equals X X X", () => {
    for (const m of ALL_MOVES) {
      const quarters = applyMoves(solved(), Array(m.turns).fill({ layer: m.layer, turns: 1 }));
      expect(equals(applyMove(solved(), m), quarters)).toBe(true);
    }
  });

  it("keeps orientation sums invariant", () => {
    const s = applyMoves(solved(), [...randomScramble(100, seededRandom(7)), ...parseMoves("M S E' M2")]);
    expect(s.co.reduce((a, b) => a + b) % 3).toBe(0);
    expect(s.eo.reduce((a, b) => a + b) % 2).toBe(0);
  });
});

describe("geometry", () => {
  it("places the 54 facelets on distinct (cubie, normal) slots", () => {
    const keys = new Set(FACELET_PLACEMENTS.map((f) => `${f.pos}|${f.normal}`));
    expect(keys.size).toBe(54);
  });

  it("agrees with the cubie model for every move from a scrambled state", () => {
    const start = applyMoves(solved(), [...randomScramble(30, seededRandom(1)), ...parseMoves("M E S'")]);
    for (const m of ALL_MOVES) {
      expect(toFacelets(applyMove(start, m))).toEqual(geometricFacelets(toFacelets(start), m));
    }
  });

  it("layerTurnToMove inverts moveRotation for quarter turns", () => {
    for (const m of ALL_MOVES.filter((m) => m.turns !== 2)) {
      const { axis, layer, quarterTurns } = moveRotation(m);
      expect(layerTurnToMove(axis, layer, quarterTurns as 1 | -1)).toEqual(m);
    }
  });
});

describe("notation and scramble", () => {
  it("round-trips notation", () => {
    const seq = "U R2 F' D L2 B' M E2 S'";
    expect(formatMoves(parseMoves(seq))).toBe(seq);
    expect(() => parseMoves("R x")).toThrow();
  });

  it("generates valid scrambles", () => {
    const axis = (m: Move) => "UDRLFB".indexOf(m.layer) >> 1;
    for (let seed = 0; seed < 50; seed++) {
      const s = randomScramble(25, seededRandom(seed));
      expect(s).toHaveLength(25);
      for (let i = 1; i < s.length; i++) {
        expect(FACES).toContain(s[i].layer);
        expect(s[i].layer).not.toBe(s[i - 1].layer);
        if (i >= 2) expect(axis(s[i]) === axis(s[i - 1]) && axis(s[i]) === axis(s[i - 2])).toBe(false);
      }
    }
  });

  it("uses crypto randomness by default", () => {
    expect(randomScramble()).toHaveLength(25);
  });
});
