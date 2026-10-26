use std::sync::OnceLock;

use rubikfhe_cube::{CubieState, Move, to_facelet_colors};
use rubikfhe_solver::Thistlethwaite;

use crate::perm::Perm;
use crate::reference::{decode, evaluate, one_hot, tokens_to_moves};
use crate::tables::{Cubie, LOCATIONS, Tables};

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| Tables::new(&Thistlethwaite::new()))
}

fn random_moves(seed: &mut u64, len: usize) -> Vec<Move> {
    (0..len)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Move::ALL[(*seed >> 33) as usize % 18]
        })
        .collect()
}

fn scrambles(count: usize) -> Vec<CubieState> {
    let mut seed = 0xC0FFEE;
    (0..count)
        .map(|i| {
            let len = [5, 20, 40, 120][i % 4];
            CubieState::SOLVED.apply_all(&random_moves(&mut seed, len))
        })
        .collect()
}

/// Solves `s` with the reference circuit and checks the replay.
fn assert_circuit_solves(s: &CubieState) {
    let tokens = evaluate(tables(), &one_hot(&to_facelet_colors(s)));
    assert_eq!(tokens.len(), tables().output_len());
    assert!(tokens.iter().all(|&t| (0..=18).contains(&t)), "token out of range");
    assert!(s.apply_all(&tokens_to_moves(&tokens)).is_solved(), "replay does not solve {s:?}");
}

// --- A. Group and tables -------------------------------------------------------

#[test]
fn move_permutations_follow_the_spec_convention() {
    for g in Perm::generators() {
        assert!(g.after(&g).after(&g).after(&g).is_identity());
    }
    for s in scrambles(20) {
        for m in Move::ALL {
            assert_eq!(Perm::of_state(&s.apply(m)), Perm::of_move(m).after(&Perm::of_state(&s)));
        }
        assert_eq!(Perm::of_state(&s).to_state(), Some(s));
    }
}

#[test]
fn chain_has_the_cube_group_order_and_spec_orbits() {
    let chain = &tables().chain;
    assert_eq!(chain.order(), 43_252_003_274_489_856_000);
    assert_eq!(chain.orbit_sizes(), [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 24, 21, 18, 15, 12, 9, 3]);
}

#[test]
fn corrections_fix_earlier_marks_and_bring_the_cubie_home() {
    let marks: Vec<u8> = Cubie::TRACKED.iter().map(|c| c.marked_sticker()).collect();
    for (i, stage) in tables().stages.iter().enumerate() {
        for (k, (&a, c)) in stage.orbit.iter().zip(&stage.corrections).enumerate() {
            assert!(c.fixes(&marks[..i]), "stage {i} correction {k} moves an earlier mark");
            assert_eq!(c.image(stage.cubie.location_facelet(a)), marks[i]);
            assert_eq!(Perm::of_moves(&stage.words[k]), *c, "stage {i} word {k} differs from its correction");
            if a == stage.cubie.home() {
                assert!(c.is_identity() && stage.words[k].is_empty());
            }
            for edge in [true, false] {
                let mut seen = [false; LOCATIONS];
                for &y in stage.location_map(k, edge) {
                    assert!(!std::mem::replace(&mut seen[y as usize], true), "location map is not a bijection");
                }
            }
        }
    }
}

#[test]
fn report_output_length() {
    let lens: Vec<usize> = tables().stages.iter().map(|s| s.word_len).collect();
    println!("word lengths per stage: {lens:?}, total {} tokens", tables().output_len());
}

// --- B. Sticker decoder --------------------------------------------------------

#[test]
fn decoder_finds_every_tracked_cubie() {
    for s in scrambles(500) {
        let z = decode(&one_hot(&to_facelet_colors(&s)));
        let g = Perm::of_state(&s);
        for (r, cubie) in Cubie::TRACKED.iter().enumerate() {
            let want = cubie.location_of_facelet(g.image(cubie.marked_sticker()));
            let expected: [i64; LOCATIONS] = std::array::from_fn(|x| (x == want as usize) as i64);
            assert_eq!(z[r], expected);
        }
    }
}

// --- C. Circuit ------------------------------------------------------------------

#[test]
fn solves_solved_and_short_scrambles() {
    assert_circuit_solves(&CubieState::SOLVED);
    for a in Move::ALL {
        assert_circuit_solves(&CubieState::SOLVED.apply(a));
        for b in Move::ALL {
            assert_circuit_solves(&CubieState::SOLVED.apply_all(&[a, b]));
        }
    }
}

#[test]
fn solves_random_scrambles() {
    for s in scrambles(2_000) {
        assert_circuit_solves(&s);
    }
}

#[test]
#[ignore = "slow: run with --ignored for the spec's 10,000 scrambles"]
fn solves_ten_thousand_scrambles() {
    for s in scrambles(10_000) {
        assert_circuit_solves(&s);
    }
}

#[test]
fn stages_solve_their_cubie_in_order() {
    let marks: Vec<u8> = Cubie::TRACKED.iter().map(|c| c.marked_sticker()).collect();
    for s in scrambles(200) {
        let mut state = s;
        for (i, stage) in tables().stages.iter().enumerate() {
            let g = Perm::of_state(&state);
            let at = stage.cubie.location_of_facelet(g.image(marks[i]));
            let k = stage.orbit.iter().position(|&a| a == at).expect("cubie lies in the stage orbit");
            state = state.apply_all(&stage.words[k]);
            assert!(Perm::of_state(&state).fixes(&marks[..=i]), "stage {i} left a cubie unsolved");
        }
        assert!(state.is_solved());
    }
}

// --- Packed circuit (slot simulator) ----------------------------------------------

mod packed {
    use super::*;
    use crate::packed::{PackedPlan, Prepared, SlotSim, evaluate};

    fn packed_tokens(slots: usize, s: &CubieState) -> Vec<i64> {
        let plan = PackedPlan::new(tables(), slots);
        let engine = SlotSim { slots };
        let prepared = Prepared::new(&engine, &plan).unwrap();
        let (u, v) = plan.pack_input(&to_facelet_colors(s));
        let (out, _) = evaluate(&engine, &prepared, &u, &v).unwrap();
        plan.unpack_tokens(&out.cols, &out.rows)
    }

    #[test]
    fn matches_the_reference_with_one_block_per_ciphertext() {
        for s in scrambles(30) {
            let want = evaluate_reference(s);
            assert_eq!(packed_tokens(1 << 10, &s), want);
        }
    }

    #[test]
    fn matches_the_reference_with_all_blocks_in_one_ciphertext() {
        for s in scrambles(10) {
            assert_eq!(packed_tokens(1 << 15, &s), evaluate_reference(s));
        }
    }

    fn evaluate_reference(s: CubieState) -> Vec<i64> {
        let tokens = evaluate_ref(tables(), &one_hot(&to_facelet_colors(&s)));
        assert!(s.apply_all(&tokens_to_moves(&tokens)).is_solved());
        tokens
    }

    use crate::reference::evaluate as evaluate_ref;
}
