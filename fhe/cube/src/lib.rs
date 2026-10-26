//! Cube model shared by the client and the server.

mod cubie;
mod facelets;
mod moves;

use std::fmt;

pub use cubie::{CubieState, FACE_MOVE_TABLES, STATE_BYTES, parity};
pub use facelets::{
    CORNER_COLORS, CORNER_FACELETS, EDGE_COLORS, EDGE_FACELETS, facelet_colors, from_facelets, to_facelet_colors,
    to_facelets,
};
pub use moves::{Face, Move, format_moves, parse_moves, simplify};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CubeError {
    InvalidMove(String),
    Invalid(String),
}

impl fmt::Display for CubeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubeError::InvalidMove(m) => write!(f, "invalid move: {m:?}"),
            CubeError::Invalid(msg) => write!(f, "invalid cube: {msg}"),
        }
    }
}

impl std::error::Error for CubeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turns_have_order_four() {
        for face in Face::ALL {
            let m = Move::new(face, 1);
            assert!(CubieState::SOLVED.apply_all(&[m; 4]).is_solved());
        }
    }

    #[test]
    fn sexy_move_has_order_six() {
        let seq = parse_moves("R U R' U'").unwrap();
        let mut s = CubieState::SOLVED;
        for i in 1..=6 {
            s = s.apply_all(&seq);
            assert_eq!(s.is_solved(), i == 6);
        }
    }

    #[test]
    fn move_indices_and_notation_round_trip() {
        for (i, m) in Move::ALL.iter().enumerate() {
            assert_eq!(m.index() as usize, i);
            assert_eq!(m.to_string().parse::<Move>().unwrap(), *m);
        }
        assert!("M".parse::<Move>().is_err());
    }

    #[test]
    fn facelets_round_trip() {
        let s = CubieState::SOLVED.apply_all(&parse_moves("R U2 F' D L2 B R' U F2").unwrap());
        assert_eq!(from_facelets(&to_facelets(&s)).unwrap(), s);
        assert_eq!(to_facelets(&CubieState::SOLVED), "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
    }

    #[test]
    fn facelets_match_the_web_model() {
        // `toFaceletString(applyMoves(solved(), parseMoves("R U F")))` in web/src/lib/cube.
        let s = CubieState::SOLVED.apply_all(&parse_moves("R U F").unwrap());
        assert_eq!(to_facelets(&s), "UUUUUULLDFBBFRRFRRFFRFFRDDRRRUDDBDDBFFDLLDLLBLLLUBBUBB");
    }

    #[test]
    fn parsing_follows_center_colors() {
        // A cube turned upside down by slice moves is still solved.
        let mut f: Vec<char> = to_facelets(&CubieState::SOLVED).chars().collect();
        let swap = |f: &mut Vec<char>, a: char, b: char| {
            for c in f.iter_mut() {
                *c = if *c == a {
                    b
                } else if *c == b {
                    a
                } else {
                    *c
                };
            }
        };
        swap(&mut f, 'U', 'D');
        swap(&mut f, 'F', 'B');
        assert!(from_facelets(&f.iter().collect::<String>()).unwrap().is_solved());
    }

    #[test]
    fn rejects_impossible_cubes() {
        let mut s = CubieState::SOLVED;
        s.co[0] = 1;
        assert!(s.validate().is_err());
        let mut s = CubieState::SOLVED;
        s.ep.swap(0, 1);
        assert!(s.validate().is_err());
        assert!(from_facelets("UUU").is_err());
    }

    #[test]
    fn simplify_merges_same_face_turns() {
        let moves = parse_moves("U U2 R R' F F F2 B").unwrap();
        assert_eq!(format_moves(&simplify(&moves)), "U' B");
    }
}
