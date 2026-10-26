//! Permutations of the 54 facelets (centers always fixed), and their link to
//! cube states.
//!
//! Convention (spec §3): a configuration `g` maps each sticker `s` (named by
//! its home facelet) to its current facelet `g(s)`. Applying a physical move
//! `m` yields `m ∘ g`, so a correction `c` solves `g` when `c ∘ g = id`.

use rubikfhe_cube::{CORNER_FACELETS, CubieState, EDGE_FACELETS, Face, Move};

pub const POINTS: usize = 54;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Perm(pub [u8; POINTS]);

impl Perm {
    pub const IDENTITY: Perm = {
        let mut p = [0u8; POINTS];
        let mut i = 0;
        while i < POINTS {
            p[i] = i as u8;
            i += 1;
        }
        Perm(p)
    };

    pub fn image(&self, x: u8) -> u8 {
        self.0[x as usize]
    }

    /// `self ∘ inner`: apply `inner` first.
    pub fn after(&self, inner: &Perm) -> Perm {
        Perm(std::array::from_fn(|x| self.0[inner.0[x] as usize]))
    }

    pub fn inverse(&self) -> Perm {
        let mut out = [0u8; POINTS];
        for (x, &y) in self.0.iter().enumerate() {
            out[y as usize] = x as u8;
        }
        Perm(out)
    }

    pub fn is_identity(&self) -> bool {
        *self == Perm::IDENTITY
    }

    pub fn fixes(&self, points: &[u8]) -> bool {
        points.iter().all(|&p| self.image(p) == p)
    }

    /// Sticker permutation of a cube state: where each home sticker now sits.
    pub fn of_state(s: &CubieState) -> Perm {
        let mut g = Perm::IDENTITY;
        for i in 0..8 {
            for n in 0..3 {
                let sticker = CORNER_FACELETS[s.cp[i] as usize][n];
                g.0[sticker] = CORNER_FACELETS[i][(n + s.co[i] as usize) % 3] as u8;
            }
        }
        for i in 0..12 {
            for n in 0..2 {
                let sticker = EDGE_FACELETS[s.ep[i] as usize][n];
                g.0[sticker] = EDGE_FACELETS[i][(n + s.eo[i] as usize) % 2] as u8;
            }
        }
        g
    }

    /// Inverse of [`Perm::of_state`]; `None` if `self` does not move whole cubies.
    pub fn to_state(&self) -> Option<CubieState> {
        let at = self.inverse(); // facelet -> sticker sitting there
        let mut s = CubieState::SOLVED;
        for (i, facelets) in CORNER_FACELETS.iter().enumerate() {
            let (o, sticker) = (0..3)
                .map(|o| (o, at.image(facelets[o] as u8) as usize))
                .find(|&(_, st)| CORNER_FACELETS.iter().any(|f| f[0] == st))?;
            s.cp[i] = CORNER_FACELETS.iter().position(|f| f[0] == sticker)? as u8;
            s.co[i] = o as u8;
        }
        for (i, facelets) in EDGE_FACELETS.iter().enumerate() {
            let (o, sticker) = (0..2)
                .map(|o| (o, at.image(facelets[o] as u8) as usize))
                .find(|&(_, st)| EDGE_FACELETS.iter().any(|f| f[0] == st))?;
            s.ep[i] = EDGE_FACELETS.iter().position(|f| f[0] == sticker)? as u8;
            s.eo[i] = o as u8;
        }
        (Perm::of_state(&s) == *self).then_some(s)
    }

    pub fn of_move(m: Move) -> Perm {
        Perm::of_state(&CubieState::SOLVED.apply(m))
    }

    pub fn of_moves(moves: &[Move]) -> Perm {
        Perm::of_state(&CubieState::SOLVED.apply_all(moves))
    }

    /// The six clockwise quarter turns.
    pub fn generators() -> Vec<Perm> {
        Face::ALL.iter().map(|&f| Perm::of_move(Move::new(f, 1))).collect()
    }
}
