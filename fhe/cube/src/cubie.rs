//! Cubie-level representation, following Kociemba's conventions (mirrors
//! `web/src/lib/cube/cubie.ts`).
//!
//! corners: URF UFL ULB UBR DFR DLF DBL DRB
//! edges:   UR UF UL UB DR DF DL DB FR FL BL BR
//!
//! `cp[i]` is the corner cubie at position i and `co[i]` its twist (0..2);
//! `ep[i]` / `eo[i]` are the same for edges (flip 0..1).

use crate::{CubeError, Face, Move};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CubieState {
    pub cp: [u8; 8],
    pub co: [u8; 8],
    pub ep: [u8; 12],
    pub eo: [u8; 12],
}

/// Size of the serialized state: cp, co, ep, eo, one byte per entry.
pub const STATE_BYTES: usize = 40;

impl CubieState {
    pub const SOLVED: CubieState = CubieState {
        cp: [0, 1, 2, 3, 4, 5, 6, 7],
        co: [0; 8],
        ep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        eo: [0; 12],
    };

    /// Returns `self * b`: this state followed by the move/state `b`.
    pub fn multiply(&self, b: &CubieState) -> CubieState {
        let mut out = CubieState::SOLVED;
        for i in 0..8 {
            let p = b.cp[i] as usize;
            out.cp[i] = self.cp[p];
            out.co[i] = (self.co[p] + b.co[i]) % 3;
        }
        for i in 0..12 {
            let p = b.ep[i] as usize;
            out.ep[i] = self.ep[p];
            out.eo[i] = (self.eo[p] + b.eo[i]) % 2;
        }
        out
    }

    pub fn apply(&self, m: Move) -> CubieState {
        let table = &FACE_MOVE_TABLES[m.face.index()];
        (0..m.turns).fold(*self, |s, _| s.multiply(table))
    }

    pub fn apply_all(&self, moves: &[Move]) -> CubieState {
        moves.iter().fold(*self, |s, &m| s.apply(m))
    }

    pub fn is_solved(&self) -> bool {
        *self == CubieState::SOLVED
    }

    pub fn to_bytes(&self) -> [u8; STATE_BYTES] {
        let mut out = [0; STATE_BYTES];
        out[..8].copy_from_slice(&self.cp);
        out[8..16].copy_from_slice(&self.co);
        out[16..28].copy_from_slice(&self.ep);
        out[28..].copy_from_slice(&self.eo);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<CubieState, CubeError> {
        if bytes.len() != STATE_BYTES {
            return Err(CubeError::Invalid(format!("expected {STATE_BYTES} bytes, got {}", bytes.len())));
        }
        let mut s = CubieState::SOLVED;
        s.cp.copy_from_slice(&bytes[..8]);
        s.co.copy_from_slice(&bytes[8..16]);
        s.ep.copy_from_slice(&bytes[16..28]);
        s.eo.copy_from_slice(&bytes[28..]);
        s.validate()?;
        Ok(s)
    }

    /// Checks that the state is reachable from solved by legal moves.
    pub fn validate(&self) -> Result<(), CubeError> {
        let is_perm = |p: &[u8]| {
            let mut seen = vec![false; p.len()];
            p.iter().all(|&v| (v as usize) < p.len() && !std::mem::replace(&mut seen[v as usize], true))
        };
        if !is_perm(&self.cp) || !is_perm(&self.ep) {
            return Err(CubeError::Invalid("pieces are not a permutation".into()));
        }
        if self.co.iter().any(|&o| o > 2) || self.eo.iter().any(|&o| o > 1) {
            return Err(CubeError::Invalid("orientation out of range".into()));
        }
        if self.co.iter().map(|&o| o as u32).sum::<u32>() % 3 != 0 {
            return Err(CubeError::Invalid("a corner is twisted".into()));
        }
        if self.eo.iter().map(|&o| o as u32).sum::<u32>() % 2 != 0 {
            return Err(CubeError::Invalid("an edge is flipped".into()));
        }
        if parity(&self.cp) != parity(&self.ep) {
            return Err(CubeError::Invalid("two pieces are swapped".into()));
        }
        Ok(())
    }
}

/// Permutation parity: 0 for even, 1 for odd.
pub fn parity(p: &[u8]) -> u8 {
    let mut inversions = 0;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            inversions += (p[i] > p[j]) as u8;
        }
    }
    inversions % 2
}

/// Clockwise quarter turn of each face (indexed like `Face`), in "replaced by" form.
pub const FACE_MOVE_TABLES: [CubieState; 6] = [
    // U
    CubieState {
        cp: [3, 0, 1, 2, 4, 5, 6, 7],
        co: [0, 0, 0, 0, 0, 0, 0, 0],
        ep: [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
        eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    // R
    CubieState {
        cp: [4, 1, 2, 0, 7, 5, 6, 3],
        co: [2, 0, 0, 1, 1, 0, 0, 2],
        ep: [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
        eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    // F
    CubieState {
        cp: [1, 5, 2, 3, 0, 4, 6, 7],
        co: [1, 2, 0, 0, 2, 1, 0, 0],
        ep: [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
        eo: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    },
    // D
    CubieState {
        cp: [0, 1, 2, 3, 5, 6, 7, 4],
        co: [0, 0, 0, 0, 0, 0, 0, 0],
        ep: [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
        eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    // L
    CubieState {
        cp: [0, 2, 6, 3, 4, 1, 5, 7],
        co: [0, 1, 2, 0, 0, 2, 1, 0],
        ep: [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
        eo: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    // B
    CubieState {
        cp: [0, 1, 3, 7, 4, 5, 2, 6],
        co: [0, 0, 1, 2, 0, 0, 2, 1],
        ep: [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
        eo: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    },
];

impl Face {
    pub fn move_table(self) -> &'static CubieState {
        &FACE_MOVE_TABLES[self.index()]
    }
}
