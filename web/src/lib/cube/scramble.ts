import { FACES, type Face, type Move, type Turns } from "./moves";

export const SCRAMBLE_LENGTH = 25;

/** Returns a uniform integer in [0, n). */
export type RandomInt = (n: number) => number;

export const cryptoRandomInt: RandomInt = (n) => {
  // Rejection sampling to avoid modulo bias.
  const limit = Math.floor(0x1_0000_0000 / n) * n;
  const buf = new Uint32Array(1);
  do crypto.getRandomValues(buf);
  while (buf[0] >= limit);
  return buf[0] % n;
};

const AXIS: Record<Face, number> = { U: 0, D: 0, R: 1, L: 1, F: 2, B: 2 };

/**
 * Random-move scramble using face moves only, WCA style: never turns the same
 * face twice in a row, and never three moves on the same axis in a row
 * (e.g. no "R L R").
 */
export function randomScramble(length = SCRAMBLE_LENGTH, rand: RandomInt = cryptoRandomInt): Move[] {
  const faces: Face[] = [];
  while (faces.length < length) {
    const face = FACES[rand(FACES.length)];
    const prev = faces.at(-1);
    const prev2 = faces.at(-2);
    if (prev === face) continue;
    if (prev && prev2 && AXIS[prev2] === AXIS[face] && AXIS[prev] === AXIS[face]) continue;
    faces.push(face);
  }
  return faces.map((layer) => ({ layer, turns: (rand(3) + 1) as Turns }));
}
