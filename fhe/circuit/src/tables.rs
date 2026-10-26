//! Public tables of the one-cubie stabilizer-chain circuit (spec §3–§4, §6–§7).
//!
//! Stage `i` reads the oriented location of tracked cubie `i`, selects the
//! public correction for it, emits that correction's move word and applies its
//! action to every later cubie.

use std::collections::BTreeMap;

use rubikfhe_cube::{CORNER_FACELETS, EDGE_FACELETS, Face, Move, simplify};
use rubikfhe_solver::Thistlethwaite;

use crate::chain::StabChain;
use crate::perm::Perm;

/// Oriented locations of one cubie: 12 positions × 2 for edges, 8 × 3 for corners.
pub const LOCATIONS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cubie {
    Edge(usize),
    Corner(usize),
}

impl Cubie {
    /// The 18 tracked cubies in stage order; BR and DRB are forced by the others.
    pub const TRACKED: [Cubie; 18] = [
        Cubie::Edge(0),
        Cubie::Edge(1),
        Cubie::Edge(2),
        Cubie::Edge(3),
        Cubie::Edge(4),
        Cubie::Edge(5),
        Cubie::Edge(6),
        Cubie::Edge(7),
        Cubie::Edge(8),
        Cubie::Edge(9),
        Cubie::Edge(10),
        Cubie::Corner(0),
        Cubie::Corner(1),
        Cubie::Corner(2),
        Cubie::Corner(3),
        Cubie::Corner(4),
        Cubie::Corner(5),
        Cubie::Corner(6),
    ];

    pub fn is_edge(self) -> bool {
        matches!(self, Cubie::Edge(_))
    }

    /// The marked sticker: the cubie's first reference facelet.
    pub fn marked_sticker(self) -> u8 {
        match self {
            Cubie::Edge(i) => EDGE_FACELETS[i][0] as u8,
            Cubie::Corner(i) => CORNER_FACELETS[i][0] as u8,
        }
    }

    /// Facelet of oriented location `x`: position `x / k`, reference slot `x % k`.
    pub fn location_facelet(self, x: u8) -> u8 {
        let x = x as usize;
        if self.is_edge() { EDGE_FACELETS[x / 2][x % 2] as u8 } else { CORNER_FACELETS[x / 3][x % 3] as u8 }
    }

    pub fn location_of_facelet(self, f: u8) -> u8 {
        (0..LOCATIONS as u8).find(|&x| self.location_facelet(x) == f).expect("facelet of the cubie's type")
    }

    pub fn home(self) -> u8 {
        self.location_of_facelet(self.marked_sticker())
    }
}

/// One stage of the circuit.
pub struct Stage {
    pub cubie: Cubie,
    /// The orbit `Ω_i`, as locations of the stage's cubie.
    pub orbit: Vec<u8>,
    /// `c[i, a]` for each orbit point `a` (same order as `orbit`).
    pub corrections: Vec<Perm>,
    /// A move word implementing each correction (unpadded).
    pub words: Vec<Vec<Move>>,
    /// Padded word length `L_i`.
    pub word_len: usize,
    /// For each orbit point: the new location of an edge / a corner at each location.
    pub edge_maps: Vec<[u8; LOCATIONS]>,
    pub corner_maps: Vec<[u8; LOCATIONS]>,
}

impl Stage {
    pub fn location_map(&self, k: usize, edge: bool) -> &[u8; LOCATIONS] {
        if edge { &self.edge_maps[k] } else { &self.corner_maps[k] }
    }
}

pub struct Tables {
    pub chain: StabChain,
    pub stages: Vec<Stage>,
}

impl Tables {
    pub fn new(solver: &Thistlethwaite) -> Tables {
        let base: Vec<u8> = Cubie::TRACKED.iter().map(|c| c.marked_sticker()).collect();
        let chain = StabChain::new(&Perm::generators(), &base);
        let short = short_corrections(&base, SHORT_WORD_DEPTH);

        let stages = Cubie::TRACKED
            .iter()
            .zip(&chain.levels)
            .map(|(&cubie, level)| {
                let mut orbit = Vec::new();
                let mut corrections = Vec::new();
                let mut words = Vec::new();
                for (&facelet, u) in &level.transversal {
                    // Any element of H_i bringing `facelet` home is a valid
                    // correction: prefer the shortest word found by search,
                    // else solve the Schreier–Sims transversal element (solving
                    // the state `u` yields a word for `u⁻¹`).
                    let word = short[level_index(&base, level.base)].get(&facelet).cloned().unwrap_or_else(|| {
                        simplify(&solver.solve(&u.to_state().expect("transversal elements are cube states")))
                    });
                    orbit.push(cubie.location_of_facelet(facelet));
                    corrections.push(Perm::of_moves(&word));
                    words.push(word);
                }
                let word_len = words.iter().map(Vec::len).max().unwrap_or(0);
                let maps = |c: Cubie| -> Vec<[u8; LOCATIONS]> {
                    corrections
                        .iter()
                        .map(|p| std::array::from_fn(|x| c.location_of_facelet(p.image(c.location_facelet(x as u8)))))
                        .collect()
                };
                let edge_maps = maps(Cubie::Edge(0));
                let corner_maps = maps(Cubie::Corner(0));
                Stage { cubie, orbit, corrections, words, word_len, edge_maps, corner_maps }
            })
            .collect();
        Tables { chain, stages }
    }

    /// Number of move tokens in the output (`Σ L_i`).
    pub fn output_len(&self) -> usize {
        self.stages.iter().map(|s| s.word_len).sum()
    }
}

/// Search depth for short correction words (≈ 2·10⁷ words, well under a second).
const SHORT_WORD_DEPTH: usize = 6;

fn level_index(base: &[u8], point: u8) -> usize {
    base.iter().position(|&b| b == point).unwrap()
}

/// Shortest words of length ≤ `depth` acting as a correction, per stage and
/// orbit point: `w` qualifies for stage `i` and point `a` when it fixes
/// `base[..i]` and sends facelet `a` to `base[i]`.
fn short_corrections(base: &[u8], depth: usize) -> Vec<BTreeMap<u8, Vec<Move>>> {
    let gens: Vec<(Move, Perm)> = Move::ALL.iter().map(|&m| (m, Perm::of_move(m))).collect();
    let mut found: Vec<BTreeMap<u8, Vec<Move>>> = vec![BTreeMap::new(); base.len()];
    let mut word = Vec::with_capacity(depth);

    fn visit(
        p: &Perm,
        word: &mut Vec<Move>,
        depth: usize,
        base: &[u8],
        gens: &[(Move, Perm)],
        found: &mut [BTreeMap<u8, Vec<Move>>],
    ) {
        let inv = p.inverse();
        for (i, &b) in base.iter().enumerate() {
            let slot = found[i].entry(inv.image(b));
            match slot {
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(word.clone());
                }
                std::collections::btree_map::Entry::Occupied(mut e) if e.get().len() > word.len() => {
                    e.insert(word.clone());
                }
                _ => {}
            }
            if p.image(b) != b {
                break;
            }
        }
        if word.len() == depth {
            return;
        }
        for &(m, g) in gens {
            if let Some(last) = word.last() {
                // Canonical sequences: no repeated face, opposite faces in one order.
                let opposite = |f: Face| Face::ALL[(f.index() + 3) % 6];
                if last.face == m.face || (opposite(last.face) == m.face && m.face.index() < 3) {
                    continue;
                }
            }
            word.push(m);
            visit(&g.after(p), word, depth, base, gens, found);
            word.pop();
        }
    }

    visit(&Perm::IDENTITY, &mut word, depth, base, &gens, &mut found);
    found
}

/// Token codes (spec §2.2): 0 = NOP, 1..=18 = `Move::ALL[code - 1]`.
pub fn token_code(m: Option<Move>) -> u8 {
    m.map_or(0, |m| m.index() + 1)
}

pub fn token_move(code: u8) -> Option<Move> {
    code.checked_sub(1).and_then(Move::from_index)
}
