/** The six outer faces, in Kociemba order (also the facelet string order). */
export const FACES = ["U", "R", "F", "D", "L", "B"] as const;
export type Face = (typeof FACES)[number];

/** Middle slices: M follows L, E follows D, S follows F. */
export const SLICES = ["M", "E", "S"] as const;
export type Slice = (typeof SLICES)[number];

export type Layer = Face | Slice;
export const LAYERS: readonly Layer[] = [...FACES, ...SLICES];

/** Quarter turns clockwise (as seen looking at the face): 1 = X, 2 = X2, 3 = X'. */
export type Turns = 1 | 2 | 3;

export interface Move {
  layer: Layer;
  turns: Turns;
}

const withTurns = (layers: readonly Layer[]): Move[] =>
  layers.flatMap((layer) => ([1, 2, 3] as const).map((turns) => ({ layer, turns })));

/** The 18 face moves (half-turn metric), ordered U U2 U' R R2 R' ... */
export const FACE_MOVES: readonly Move[] = withTurns(FACES);

/** The 9 slice moves M M2 M' E E2 E' S S2 S'. */
export const SLICE_MOVES: readonly Move[] = withTurns(SLICES);

const SUFFIX: Record<Turns, string> = { 1: "", 2: "2", 3: "'" };

export function formatMove(m: Move): string {
  return m.layer + SUFFIX[m.turns];
}

export function formatMoves(ms: readonly Move[]): string {
  return ms.map(formatMove).join(" ");
}

export function parseMove(token: string): Move {
  const match = /^([URFDLBMES])(2|'|2')?$/.exec(token.trim());
  if (!match) throw new Error(`Invalid move: "${token}"`);
  const layer = match[1] as Layer;
  const suffix = match[2];
  const turns: Turns = suffix === "'" ? 3 : suffix ? 2 : 1;
  return { layer, turns };
}

/** Parses a space-separated sequence in Singmaster notation, e.g. "R U R' U'". */
export function parseMoves(seq: string): Move[] {
  return seq.split(/\s+/).filter(Boolean).map(parseMove);
}

export function invertMove(m: Move): Move {
  return { layer: m.layer, turns: (4 - m.turns) as Turns };
}

export function invertMoves(ms: readonly Move[]): Move[] {
  return ms.toReversed().map(invertMove);
}
