//! Public plan of the packed circuit: every mask, linear map and token code
//! as plain slot values, derived from the stage tables and the layout.
//!
//! Stage `i` in orientation `O` (see [`Orientation`]):
//!
//! 1. the selector `s` (cubie `i`'s vector) is copied into every block;
//! 2. per block, a linear map gathers it into the 24 × 24 transition matrix
//!    `S[y → z] = Σ_a s_a [π_a(y) = z]` at `(v = y, w = z)`, `π` being the
//!    edge or corner action of correction `c[i, a]`;
//! 3. one product with the state gives `S[y → z] · x[y]` at `(y, z)`;
//! 4. folding along `v` leaves the new vector `x'[z]` at `(0, z)`; a mask
//!    keeps it and replication along `v` copies it: the state is now in
//!    orientation `flip(O)`. The same masked fold also yields the next
//!    selector, isolated in its block.
//!
//! Move tokens are linear in the selector: `Σ_a s_a · code(word[i, a][t])`
//! lands at `(v = 0, w = t)` of a token block after the final fold.

use rubikfhe_cube::{CORNER_COLORS, CORNER_FACELETS, EDGE_COLORS, EDGE_FACELETS};

use super::engine::Diagonals;
use super::layout::{BLOCK, Layout, Orientation, SIDE};
use crate::tables::{Cubie, LOCATIONS, Tables, token_code};

pub struct StagePlan {
    pub orientation: Orientation,
    /// Ciphertext and mask isolating this stage's cubie block (replicated form).
    pub isolate: (usize, Vec<f64>),
    /// Token codes, per token ciphertext of this orientation's family.
    pub codes: Vec<(usize, Vec<f64>)>,
    /// Gather maps, one per state ciphertext (empty for the last stage).
    pub gather: Vec<Diagonals>,
    /// Masks keeping the folded new vectors, one per state ciphertext.
    pub keep_state: Vec<Vec<f64>>,
    /// Ciphertext and mask keeping only the next stage's selector.
    pub keep_next: Option<(usize, Vec<f64>)>,
}

/// Where token `t` of a stage ends up: output family, ciphertext and slot.
#[derive(Clone, Copy, Debug)]
pub struct TokenSlot {
    pub family: Orientation,
    pub ct: usize,
    pub slot: usize,
}

pub struct PackedPlan {
    pub layout: Layout,
    pub state_cts: usize,
    /// Number of token ciphertexts of the `Cols` and `Rows` families.
    pub token_cts: [usize; 2],
    pub stages: Vec<StagePlan>,
    pub token_slots: Vec<Vec<TokenSlot>>,
}

pub fn family_index(o: Orientation) -> usize {
    match o {
        Orientation::Cols => 0,
        Orientation::Rows => 1,
    }
}

/// Block of tracked cubie `r`. With several blocks per ciphertext, edges take
/// the even blocks and corners the odd ones, so every gather map repeats every
/// two blocks: its diagonals are stored and encoded as one 2048-slot period.
fn block_of(layout: &Layout, r: usize) -> usize {
    if layout.blocks_per_ct < 2 {
        return r;
    }
    match Cubie::TRACKED[r] {
        Cubie::Edge(e) => 2 * e,
        Cubie::Corner(c) => 2 * c + 1,
    }
}

impl PackedPlan {
    pub fn new(tables: &Tables, slots: usize) -> PackedPlan {
        let layout = Layout::new(slots);
        let blocks = (0..Cubie::TRACKED.len()).map(|r| block_of(&layout, r)).max().unwrap() + 1;
        let state_cts = layout.cts_for(blocks);
        let zeros = || vec![0.0; slots];

        // Token blocks are handed out per family, in stage order.
        let mut next_token_block = [0usize; 2];
        let mut token_slots = Vec::new();
        let token_start: Vec<usize> = tables
            .stages
            .iter()
            .enumerate()
            .map(|(i, stage)| {
                let f = family_index(Orientation::of_stage(i));
                let start = next_token_block[f];
                next_token_block[f] += stage.word_len.div_ceil(SIDE);
                start
            })
            .collect();
        let token_cts = next_token_block.map(|blocks| layout.cts_for(blocks));

        let stages = tables
            .stages
            .iter()
            .enumerate()
            .map(|(i, stage)| {
                let o = Orientation::of_stage(i);

                let (ct, base) = layout.place(block_of(&layout, i));
                let mut isolate = zeros();
                for v in 0..LOCATIONS {
                    for w in 0..SIDE {
                        isolate[base + o.at(v, w)] = 1.0;
                    }
                }

                let mut codes: Vec<(usize, Vec<f64>)> = Vec::new();
                let mut slots_of_stage = Vec::new();
                for t in 0..stage.word_len {
                    let (tct, tbase) = layout.place(token_start[i] + t / SIDE);
                    if codes.last().is_none_or(|(c, _)| *c != tct) {
                        codes.push((tct, zeros()));
                    }
                    let values = &mut codes.last_mut().unwrap().1;
                    for (k, &a) in stage.orbit.iter().enumerate() {
                        let code = token_code(stage.words[k].get(t).copied());
                        values[tbase + o.at(a as usize, t % SIDE)] = code as f64;
                    }
                    slots_of_stage.push(TokenSlot { family: o, ct: tct, slot: tbase + o.at(0, t % SIDE) });
                }
                token_slots.push(slots_of_stage);

                let last = i + 1 == tables.stages.len();
                let mut gather = Vec::new();
                let mut keep_state = Vec::new();
                if !last {
                    // One period of the block pattern: edge then corner when
                    // interleaved, else the ciphertext's single cubie.
                    let period = if layout.blocks_per_ct < 2 { 1 } else { 2 };
                    for c in 0..state_cts {
                        let mut diagonals = Diagonals::new();
                        for b in 0..period {
                            let edge = if period == 2 { b == 0 } else { Cubie::TRACKED[c].is_edge() };
                            let base = b * BLOCK;
                            for (k, &a) in stage.orbit.iter().enumerate() {
                                let map = stage.location_map(k, edge);
                                for (y, &z) in map.iter().enumerate() {
                                    let z = z as usize;
                                    let offset = (a as i64 - y as i64) * o.step() as i64;
                                    let d = offset.rem_euclid(slots as i64) as usize;
                                    diagonals.entry(d).or_insert_with(|| vec![0.0; period * BLOCK])
                                        [base + o.at(y, z)] = 1.0;
                                }
                            }
                        }
                        let mut keep = zeros();
                        for block in 0..layout.blocks_per_ct {
                            for z in 0..LOCATIONS {
                                keep[block * BLOCK + o.at(0, z)] = 1.0;
                            }
                        }
                        gather.push(diagonals);
                        keep_state.push(keep);
                    }
                }
                let keep_next = (!last).then(|| {
                    let (ct, base) = layout.place(block_of(&layout, i + 1));
                    let mut keep = zeros();
                    for z in 0..LOCATIONS {
                        keep[base + o.at(0, z)] = 1.0;
                    }
                    (ct, keep)
                });

                StagePlan { orientation: o, isolate: (ct, isolate), codes, gather, keep_state, keep_next }
            })
            .collect();

        PackedPlan { layout, state_cts, token_cts, stages, token_slots }
    }

    /// Client side: the two ciphertexts' worth of slots whose product is the
    /// one-hot location of every tracked cubie (spec §5.1), in `Cols`
    /// orientation. Each slot is one bit of the one-hot sticker colors, copied
    /// to a public position: no cube reasoning happens on the client.
    pub fn pack_input(&self, colors: &[u8; 54]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let bit = |f: usize, c: u8| (colors[f] == c) as u8 as f64;
        let mut u = vec![vec![0.0; self.layout.slots]; self.state_cts];
        let mut v = u.clone();
        for (r, cubie) in Cubie::TRACKED.iter().enumerate() {
            let (ct, base) = self.layout.place(block_of(&self.layout, r));
            for x in 0..LOCATIONS {
                let (first, second) = match *cubie {
                    Cubie::Edge(p) => {
                        let (q, o) = (x / 2, x % 2);
                        ((EDGE_FACELETS[q][o], EDGE_COLORS[p][0]), (EDGE_FACELETS[q][1 - o], EDGE_COLORS[p][1]))
                    }
                    Cubie::Corner(p) => {
                        let (q, o) = (x / 3, x % 3);
                        let f = CORNER_FACELETS[q];
                        ((f[o], CORNER_COLORS[p][0]), (f[(o + 1) % 3], CORNER_COLORS[p][1]))
                    }
                };
                for w in 0..SIDE {
                    let slot = base + Orientation::Cols.at(x, w);
                    u[ct][slot] = bit(first.0, first.1);
                    v[ct][slot] = bit(second.0, second.1);
                }
            }
        }
        (u, v)
    }

    /// Client side: rounds the decrypted output slots to move tokens.
    pub fn unpack_tokens(&self, cols: &[Vec<f64>], rows: &[Vec<f64>]) -> Vec<i64> {
        self.token_slots
            .iter()
            .flatten()
            .map(|t| {
                let family = if t.family == Orientation::Cols { cols } else { rows };
                family[t.ct][t.slot].round() as i64
            })
            .collect()
    }
}
