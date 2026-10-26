//! Plaintext reference solver: Thistlethwaite's algorithm as table descent.
//!
//! The cube is brought through the nested subgroups
//!
//!   G0 = <U, D, R, L, F, B>
//!   G1 = <U, D, R, L, F2, B2>     edges oriented
//!   G2 = <U, D, R2, L2, F2, B2>   corners oriented, E-slice edges in the E slice
//!   G3 = <U2, D2, R2, L2, F2, B2> corners in their tetrads, edges in their slices
//!   G4 = {solved}
//!
//! Phase i only uses the generators of G(i). For each phase a breadth-first
//! search from the solved cube labels every coset of G(i+1) with its distance
//! to G(i+1). Solving is then pure descent: at every step, pick the first
//! allowed move whose resulting coset is one step closer.
//!
//! Each step is a lookup of a coset label followed by one of a fixed set of
//! moves, and every phase is padded to its maximal length, so the output has a
//! fixed shape regardless of the input. This is the structure the encrypted
//! version will follow.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use rubikfhe_cube::{CubieState, Face, Move};

type LabelFn = fn(&Labels, &CubieState) -> u64;

/// A phase of the algorithm: allowed moves, coset label, distance table.
pub struct Phase {
    pub name: &'static str,
    pub moves: Vec<Move>,
    /// Longest optimal solution within this phase, i.e. its padded length.
    pub max_depth: usize,
    dist: IdMap,
    label: LabelFn,
}

impl Phase {
    /// Number of cosets of G(i+1) reachable in G(i).
    pub fn size(&self) -> usize {
        self.dist.len()
    }
}

pub struct Thistlethwaite {
    pub phases: [Phase; 4],
    labels: Labels,
    move_states: [CubieState; 18],
}

/// Precomputed helpers for coset labels.
struct Labels {
    /// For every corner permutation (by rank), the smallest rank in its right
    /// coset under the corner group of G3.
    corner_coset: Vec<u16>,
}

impl Thistlethwaite {
    /// Builds the four distance tables (about a second in release mode).
    pub fn new() -> Thistlethwaite {
        let move_states = Move::ALL.map(|m| CubieState::SOLVED.apply(m));
        let labels = Labels { corner_coset: corner_cosets() };

        use Face::*;
        let all = |faces: &[Face]| faces.iter().flat_map(|&f| (1..=3).map(move |t| Move::new(f, t))).collect();
        let half = |faces: &[Face]| faces.iter().map(|&f| Move::new(f, 2)).collect::<Vec<_>>();
        let concat = |a: Vec<Move>, b: Vec<Move>| [a, b].concat();

        let specs: [(&'static str, Vec<Move>, LabelFn); 4] = [
            ("edge orientation", all(&[U, R, F, D, L, B]), label_g1),
            ("corner orientation, E slice", concat(all(&[U, D, R, L]), half(&[F, B])), label_g2),
            ("tetrads and slices", concat(all(&[U, D]), half(&[R, L, F, B])), label_g3),
            ("half turns", half(&[U, D, R, L, F, B]), label_solved),
        ];
        let phases = specs.map(|(name, moves, label)| {
            let (dist, max_depth) = bfs(&moves, &move_states, |s| label(&labels, s));
            Phase { name, moves, max_depth, dist, label }
        });
        Thistlethwaite { phases, labels, move_states }
    }

    /// Total number of move slots in a padded solution.
    pub fn padded_length(&self) -> usize {
        self.phases.iter().map(|p| p.max_depth).sum()
    }

    /// Solves `state`, returning exactly `padded_length()` slots: each phase
    /// fills its slots with moves, then `None` padding.
    pub fn solve_padded(&self, state: &CubieState) -> Vec<Option<Move>> {
        let mut s = *state;
        let mut out = Vec::with_capacity(self.padded_length());
        for phase in &self.phases {
            let start = out.len();
            let mut d = self.distance(phase, &s);
            while d > 0 {
                let (m, next) = phase
                    .moves
                    .iter()
                    .map(|&m| (m, s.multiply(&self.move_states[m.index() as usize])))
                    .find(|(_, t)| self.distance(phase, t) == d - 1)
                    .expect("distance tables are consistent");
                out.push(Some(m));
                s = next;
                d -= 1;
            }
            out.resize(start + phase.max_depth, None);
        }
        debug_assert!(s.is_solved());
        out
    }

    /// Solves `state`, returning only the moves.
    pub fn solve(&self, state: &CubieState) -> Vec<Move> {
        self.solve_padded(state).into_iter().flatten().collect()
    }

    fn distance(&self, phase: &Phase, s: &CubieState) -> u8 {
        *phase.dist.get(&(phase.label)(&self.labels, s)).expect("state is in the phase's group")
    }
}

impl Default for Thistlethwaite {
    fn default() -> Self {
        Self::new()
    }
}

// --- Coset labels ---------------------------------------------------------------
//
// A label must be invariant under left multiplication by G(i+1), i.e. under
// relabeling pieces with an element of the target group, so that it names the
// coset G(i+1)·s. That is what makes descent valid.

/// Edge orientations (G1 keeps them all 0).
fn label_g1(_: &Labels, s: &CubieState) -> u64 {
    s.eo.iter().enumerate().fold(0, |acc, (i, &o)| acc | (o as u64) << i)
}

/// Corner orientations and the positions holding E-slice edges (FR FL BL BR).
fn label_g2(_: &Labels, s: &CubieState) -> u64 {
    let co = s.co.iter().fold(0, |acc, &o| acc * 3 + o as u64);
    let e_slice = s.ep.iter().enumerate().fold(0, |acc, (i, &e)| acc | ((e >= 8) as u64) << i);
    co << 12 | e_slice
}

/// Corner permutation up to the corner group of G3, and the positions
/// holding M-slice edges (UF UB DF DB).
fn label_g3(labels: &Labels, s: &CubieState) -> u64 {
    let corners = labels.corner_coset[rank(&s.cp)] as u64;
    let m_slice = s.ep.iter().enumerate().fold(0, |acc, (i, &e)| acc | ((e < 8 && e % 2 == 1) as u64) << i);
    corners << 12 | m_slice
}

/// The full permutation state (orientations are already solved in G3).
fn label_solved(_: &Labels, s: &CubieState) -> u64 {
    let edges = s.ep.iter().fold(0, |acc, &e| acc << 4 | e as u64);
    (rank(&s.cp) as u64) << 48 | edges
}

/// Lexicographic rank of a permutation of 0..8.
fn rank(p: &[u8; 8]) -> usize {
    let mut r = 0;
    for i in 0..8 {
        let smaller_after = p[i + 1..].iter().filter(|&&x| x < p[i]).count();
        r = r * (8 - i) + smaller_after;
    }
    r
}

fn unrank(mut r: usize) -> [u8; 8] {
    let mut digits = [0usize; 8];
    for i in (0..8).rev() {
        digits[i] = r % (8 - i);
        r /= 8 - i;
    }
    let mut pool: Vec<u8> = (0..8).collect();
    digits.map(|d| pool.remove(d))
}

/// Right coset label of every corner permutation under G3's corner group.
fn corner_cosets() -> Vec<u16> {
    let compose = |a: &[u8; 8], b: &[u8; 8]| -> [u8; 8] { std::array::from_fn(|i| a[b[i] as usize]) };
    let gens: Vec<[u8; 8]> = Face::ALL.iter().map(|&f| CubieState::SOLVED.apply(Move::new(f, 2)).cp).collect();

    // Closure of the half-turn corner permutations (96 elements).
    let mut group = vec![CubieState::SOLVED.cp];
    let mut i = 0;
    while i < group.len() {
        for g in &gens {
            let p = compose(g, &group[i]);
            if !group.contains(&p) {
                group.push(p);
            }
        }
        i += 1;
    }

    (0..40320)
        .map(|r| {
            let p = unrank(r);
            group.iter().map(|h| rank(&compose(h, &p))).min().unwrap() as u16
        })
        .collect()
}

// --- Breadth-first search ---------------------------------------------------------

/// Multiplicative hash for already well-spread integer keys.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed")
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (n ^ (n >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type IdMap = HashMap<u64, u8, BuildHasherDefault<IdHasher>>;

/// Distance of every reachable label to the solved label, and the maximum distance.
fn bfs(moves: &[Move], move_states: &[CubieState; 18], label: impl Fn(&CubieState) -> u64) -> (IdMap, usize) {
    let mut dist = IdMap::default();
    dist.insert(label(&CubieState::SOLVED), 0);
    let mut frontier = vec![CubieState::SOLVED];
    let mut depth = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for s in &frontier {
            for m in moves {
                let t = s.multiply(&move_states[m.index() as usize]);
                dist.entry(label(&t)).or_insert_with(|| {
                    next.push(t);
                    depth as u8 + 1
                });
            }
        }
        if !next.is_empty() {
            depth += 1;
        }
        frontier = next;
    }
    (dist, depth)
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    fn solver() -> &'static Thistlethwaite {
        static SOLVER: OnceLock<Thistlethwaite> = OnceLock::new();
        SOLVER.get_or_init(Thistlethwaite::new)
    }

    /// Deterministic pseudo-random scrambles.
    fn scrambles(count: usize) -> impl Iterator<Item = CubieState> {
        let mut seed: u64 = 0x5EED;
        (0..count).map(move |_| {
            let moves: Vec<Move> = (0..40)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    Move::ALL[(seed >> 33) as usize % 18]
                })
                .collect();
            CubieState::SOLVED.apply_all(&moves)
        })
    }

    #[test]
    fn rank_round_trips() {
        for r in [0, 1, 4242, 40319] {
            assert_eq!(rank(&unrank(r)), r);
        }
    }

    #[test]
    fn phases_have_the_textbook_coset_counts() {
        // |G0|/|G1|, |G1|/|G2|, |G2|/|G3|, |G3|: the labels name cosets exactly.
        let sizes: Vec<usize> = solver().phases.iter().map(Phase::size).collect();
        assert_eq!(sizes, [2048, 1_082_565, 29_400, 663_552]);
    }

    #[test]
    fn phases_have_the_textbook_maximal_lengths() {
        let depths: Vec<usize> = solver().phases.iter().map(|p| p.max_depth).collect();
        assert_eq!(depths, [7, 10, 13, 15]);
        assert_eq!(solver().padded_length(), 45);
    }

    #[test]
    fn solves_random_cubes() {
        for s in scrambles(500) {
            let padded = solver().solve_padded(&s);
            assert_eq!(padded.len(), 45);
            let moves: Vec<Move> = padded.into_iter().flatten().collect();
            assert!(s.apply_all(&moves).is_solved());
        }
    }

    #[test]
    fn solved_cube_needs_no_moves() {
        assert!(solver().solve(&CubieState::SOLVED).is_empty());
    }
}
