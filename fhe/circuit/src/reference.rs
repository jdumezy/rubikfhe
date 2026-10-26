//! Exact plaintext evaluation of the one-cubie circuit (spec §5–§7, §8.2).
//!
//! Every value is an integer combination of products of one-hot input bits;
//! there are no branches or lookups on data. The CKKS evaluator computes the
//! same expressions on encrypted slots.

use rubikfhe_cube::{CORNER_COLORS, CORNER_FACELETS, EDGE_COLORS, EDGE_FACELETS, Move};

use crate::tables::{Cubie, LOCATIONS, Tables, token_code, token_move};

/// One-hot facelet colors: `x[f][c] = 1` iff facelet `f` has color `c`.
pub type OneHot = [[i64; 6]; 54];

pub fn one_hot(colors: &[u8; 54]) -> OneHot {
    std::array::from_fn(|f| std::array::from_fn(|c| (colors[f] as usize == c) as i64))
}

/// Indicator `Z_p[x]` of each tracked cubie being at oriented location `x`
/// (spec §5.1): a product of two one-hot bits, depth 1.
pub fn decode(x: &OneHot) -> Vec<[i64; LOCATIONS]> {
    Cubie::TRACKED
        .iter()
        .map(|&cubie| {
            std::array::from_fn(|loc| match cubie {
                Cubie::Edge(p) => {
                    let ([a, b], (q, o)) = (EDGE_COLORS[p], (loc / 2, loc % 2));
                    let f = EDGE_FACELETS[q];
                    x[f[o]][a as usize] * x[f[1 - o]][b as usize]
                }
                Cubie::Corner(p) => {
                    let ([a, b, _], (q, o)) = (CORNER_COLORS[p], (loc / 3, loc % 3));
                    let f = CORNER_FACELETS[q];
                    x[f[o]][a as usize] * x[f[(o + 1) % 3]][b as usize]
                }
            })
        })
        .collect()
}

/// Evaluates the circuit and returns the `Σ L_i` move tokens.
pub fn evaluate(tables: &Tables, x: &OneHot) -> Vec<i64> {
    let mut w = decode(x);
    let mut tokens = Vec::with_capacity(tables.output_len());
    for (i, stage) in tables.stages.iter().enumerate() {
        let s: Vec<i64> = stage.orbit.iter().map(|&a| w[i][a as usize]).collect();

        for t in 0..stage.word_len {
            let codes = stage.words.iter().map(|word| token_code(word.get(t).copied()) as i64);
            tokens.push(s.iter().zip(codes).map(|(s, c)| s * c).sum());
        }

        // W'_r = Σ_a s_a · P[i, a](W_r) for every later cubie (spec §6).
        for (r, cubie) in Cubie::TRACKED.iter().enumerate().skip(i + 1) {
            let mut next = [0i64; LOCATIONS];
            for (k, &s_k) in s.iter().enumerate() {
                let map = stage.location_map(k, cubie.is_edge());
                for (loc, &v) in w[r].iter().enumerate() {
                    next[map[loc] as usize] += s_k * v;
                }
            }
            w[r] = next;
        }
    }
    tokens
}

/// Decoded tokens without the NOPs.
pub fn tokens_to_moves(tokens: &[i64]) -> Vec<Move> {
    tokens.iter().filter_map(|&t| token_move(u8::try_from(t).ok()?)).collect()
}
