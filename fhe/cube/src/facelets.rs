//! Facelet strings: 54 characters, U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9,
//! each face read row by row from outside (see `web/src/lib/cube/geometry.ts`).
//!
//! Characters name sticker *colors* by the face they belong to when solved.
//! Parsing identifies each face by its center color, so a cube whose centers
//! were moved by slice turns is read relative to its current orientation, and
//! the resulting moves apply to the faces as they are now.

use crate::{CubeError, CubieState, Face};

const U: usize = 0;
const R: usize = 9;
const F: usize = 18;
const D: usize = 27;
const L: usize = 36;
const B: usize = 45;

const CORNER_FACELETS: [[usize; 3]; 8] = [
    [U + 8, R, F + 2],
    [U + 6, F, L + 2],
    [U, L, B + 2],
    [U + 2, B, R + 2],
    [D + 2, F + 8, R + 6],
    [D, L + 8, F + 6],
    [D + 6, B + 8, L + 6],
    [D + 8, R + 8, B + 6],
];

const EDGE_FACELETS: [[usize; 2]; 12] = [
    [U + 5, R + 1],
    [U + 7, F + 1],
    [U + 3, L + 1],
    [U + 1, B + 1],
    [D + 5, R + 7],
    [D + 1, F + 7],
    [D + 3, L + 7],
    [D + 7, B + 7],
    [F + 5, R + 3],
    [F + 3, L + 5],
    [B + 5, L + 3],
    [B + 3, R + 5],
];

/// Faces of each cubie, as indices into `Face::ALL`.
const CORNER_COLORS: [[u8; 3]; 8] =
    [[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5]];
const EDGE_COLORS: [[u8; 2]; 12] =
    [[0, 1], [0, 2], [0, 4], [0, 5], [3, 1], [3, 2], [3, 4], [3, 5], [2, 1], [2, 4], [5, 4], [5, 1]];

pub fn to_facelets(s: &CubieState) -> String {
    let mut out = [0u8; 54];
    for (face, chunk) in out.chunks_mut(9).enumerate() {
        chunk.fill(face as u8);
    }
    for i in 0..8 {
        for n in 0..3 {
            out[CORNER_FACELETS[i][(n + s.co[i] as usize) % 3]] = CORNER_COLORS[s.cp[i] as usize][n];
        }
    }
    for i in 0..12 {
        for n in 0..2 {
            out[EDGE_FACELETS[i][(n + s.eo[i] as usize) % 2]] = EDGE_COLORS[s.ep[i] as usize][n];
        }
    }
    out.iter().map(|&c| Face::ALL[c as usize].letter()).collect()
}

pub fn from_facelets(facelets: &str) -> Result<CubieState, CubeError> {
    let invalid = |msg: &str| CubeError::Invalid(msg.to_string());
    let colors: Vec<char> = facelets.chars().collect();
    if colors.len() != 54 {
        return Err(invalid("expected 54 facelets"));
    }

    // Map each color to the face whose center currently shows it.
    let centers: Vec<char> = (0..6).map(|f| colors[f * 9 + 4]).collect();
    let face_of = |c: char| centers.iter().position(|&x| x == c).map(|i| i as u8);
    let mut f = [0u8; 54];
    for (i, &c) in colors.iter().enumerate() {
        f[i] = face_of(c).ok_or_else(|| invalid("unknown color"))?;
    }
    for face in 0..6 {
        if f.iter().filter(|&&c| c == face).count() != 9 {
            return Err(invalid("each color must appear 9 times"));
        }
    }

    let mut s = CubieState::SOLVED;
    for i in 0..8 {
        let ori = (0..3)
            .find(|&o| matches!(f[CORNER_FACELETS[i][o]], 0 | 3))
            .ok_or_else(|| invalid("corner without U/D sticker"))?;
        let c1 = f[CORNER_FACELETS[i][(ori + 1) % 3]];
        let c2 = f[CORNER_FACELETS[i][(ori + 2) % 3]];
        let j = (0..8)
            .find(|&j| CORNER_COLORS[j][1] == c1 && CORNER_COLORS[j][2] == c2)
            .ok_or_else(|| invalid("impossible corner"))?;
        s.cp[i] = j as u8;
        s.co[i] = ori as u8;
    }
    for i in 0..12 {
        let (a, b) = (f[EDGE_FACELETS[i][0]], f[EDGE_FACELETS[i][1]]);
        let (j, flip) = (0..12)
            .find_map(|j| match EDGE_COLORS[j] {
                [x, y] if x == a && y == b => Some((j, 0)),
                [x, y] if x == b && y == a => Some((j, 1)),
                _ => None,
            })
            .ok_or_else(|| invalid("impossible edge"))?;
        s.ep[i] = j as u8;
        s.eo[i] = flip;
    }
    s.validate()?;
    Ok(s)
}
