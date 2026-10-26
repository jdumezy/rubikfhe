use std::fmt;
use std::str::FromStr;

use crate::CubeError;

/// The six faces, in Kociemba order (also the facelet string order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    U,
    R,
    F,
    D,
    L,
    B,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::U, Face::R, Face::F, Face::D, Face::L, Face::B];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn letter(self) -> char {
        b"URFDLB"[self.index()] as char
    }

    pub fn from_letter(c: char) -> Option<Face> {
        "URFDLB".find(c).map(|i| Face::ALL[i])
    }
}

/// A face move: `turns` clockwise quarter turns (1 = X, 2 = X2, 3 = X').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub face: Face,
    pub turns: u8,
}

impl Move {
    /// The 18 face moves, ordered U U2 U' R R2 R' ... (index = face * 3 + turns - 1).
    pub const ALL: [Move; 18] = {
        let mut out = [Move { face: Face::U, turns: 1 }; 18];
        let mut i = 0;
        while i < 18 {
            out[i] = Move { face: Face::ALL[i / 3], turns: (i % 3) as u8 + 1 };
            i += 1;
        }
        out
    };

    pub fn new(face: Face, turns: u8) -> Move {
        debug_assert!((1..=3).contains(&turns));
        Move { face, turns }
    }

    pub fn index(self) -> u8 {
        (self.face.index() * 3) as u8 + self.turns - 1
    }

    pub fn from_index(i: u8) -> Option<Move> {
        Move::ALL.get(i as usize).copied()
    }

    pub fn inverse(self) -> Move {
        Move { face: self.face, turns: 4 - self.turns }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = ["", "2", "'"][self.turns as usize - 1];
        write!(f, "{}{}", self.face.letter(), suffix)
    }
}

impl FromStr for Move {
    type Err = CubeError;

    fn from_str(s: &str) -> Result<Move, CubeError> {
        let mut chars = s.chars();
        let face = chars.next().and_then(Face::from_letter);
        let turns = match chars.as_str() {
            "" => Some(1),
            "2" | "2'" => Some(2),
            "'" => Some(3),
            _ => None,
        };
        match (face, turns) {
            (Some(face), Some(turns)) => Ok(Move { face, turns }),
            _ => Err(CubeError::InvalidMove(s.to_string())),
        }
    }
}

pub fn parse_moves(seq: &str) -> Result<Vec<Move>, CubeError> {
    seq.split_whitespace().map(str::parse).collect()
}

pub fn format_moves(moves: &[Move]) -> String {
    moves.iter().map(Move::to_string).collect::<Vec<_>>().join(" ")
}

/// Merges consecutive turns of the same face (e.g. "U U2" -> "U'", "R R'" -> "").
pub fn simplify(moves: &[Move]) -> Vec<Move> {
    let mut out: Vec<Move> = Vec::with_capacity(moves.len());
    for &m in moves {
        match out.last_mut() {
            Some(last) if last.face == m.face => {
                let turns = (last.turns + m.turns) % 4;
                if turns == 0 {
                    out.pop();
                } else {
                    last.turns = turns;
                }
            }
            _ => out.push(m),
        }
    }
    out
}
