use std::sync::OnceLock;
use std::time::Instant;

use rubikfhe_circuit::packed::{Engine, Event, PackedPlan, Prepared, SlotSim, evaluate, evaluate_inspected};
use rubikfhe_circuit::reference::{evaluate as evaluate_reference, one_hot, tokens_to_moves};
use rubikfhe_circuit::tables::Tables;
use rubikfhe_cube::{CubieState, Move, to_facelet_colors};
use rubikfhe_solver::Thistlethwaite;

use crate::{CkksEngine, Params, Ring};

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| Tables::new(&Thistlethwaite::new()))
}

fn scramble(seed: u64, len: usize) -> CubieState {
    let mut s = seed;
    let moves: Vec<Move> = (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Move::ALL[(s >> 33) as usize % 18]
        })
        .collect();
    CubieState::SOLVED.apply_all(&moves)
}

/// A keyed engine and the client's secret for `params`; the keys go
/// through their byte format, as between the processes.
fn setup(params: Params, plan: &PackedPlan) -> (CkksEngine, crate::SecretKey) {
    let ring = Ring::new(params.clone()).unwrap();
    let sk = ring.generate_secret([1; 32]);
    let galois = ring.circuit_galois_elements(plan);
    let keys = ring.generate_eval_keys(&sk, &galois, [2; 32]).unwrap();
    let mut bytes = Vec::new();
    crate::wire::write_eval_keys(&mut bytes, &keys).unwrap();
    drop(keys);
    let server_ring = Ring::new(params).unwrap();
    let mut engine = CkksEngine::new(server_ring);
    let keys = engine.prepare_keys_from(&mut bytes.as_slice()).unwrap();
    engine.set_keys(Some(keys));
    (engine, sk)
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn operations_match_the_simulator() {
    let params = Params::insecure_test(11);
    let slots = params.slots();
    let plan = PackedPlan::new(tables(), slots);
    let (engine, sk) = setup(params, &plan);
    let sim = SlotSim { slots };

    let a: Vec<f64> = (0..slots).map(|i| ((i * 7) % 5) as f64 / 4.0).collect();
    let b: Vec<f64> = (0..slots).map(|i| ((i * 3) % 2) as f64).collect();
    let ca = engine.ring.encrypt(&sk, &a, [3; 32]).unwrap();
    let cb = engine.ring.encrypt(&sk, &b, [4; 32]).unwrap();
    let check = |name: &str, ct: &crate::engine::Ciphertext, want: Vec<f64>| {
        let got = engine.ring.decrypt(&sk, ct).unwrap();
        let err = max_err(&got, &want);
        assert!(err < 1e-3, "{name}: max error {err:e}");
    };

    check("mul", &engine.mul(&ca, &cb).unwrap(), sim.mul(&a, &b).unwrap());
    let pt = engine.encode(&b).unwrap();
    check("mul_pt", &engine.mul_pt(&ca, &pt).unwrap(), sim.mul_pt(&a, &b).unwrap());
    for (lb, lc) in [(0, 5), (5, 5)] {
        check("inner_sum", &engine.inner_sum(&ca, lb, lc).unwrap(), sim.inner_sum(&a, lb, lc).unwrap());
        check("replicate", &engine.replicate(&ca, lb, lc).unwrap(), sim.replicate(&a, lb, lc).unwrap());
    }
    for stage in &plan.stages[..2] {
        let lt = engine.encode_lt(&stage.gather[0]).unwrap();
        check("apply_lt", &engine.apply_lt(&ca, &lt).unwrap(), sim.apply_lt(&a, &stage.gather[0]).unwrap());
    }
}

/// Runs the packed circuit under CKKS for `s` and returns the decrypted tokens.
fn ckks_tokens(
    engine: &CkksEngine,
    sk: &crate::SecretKey,
    plan: &PackedPlan,
    prepared: &Prepared<CkksEngine>,
    s: &CubieState,
) -> Vec<i64> {
    let (u, v) = plan.pack_input(&to_facelet_colors(s));
    let enc = |x: &Vec<Vec<f64>>, seed: u8| -> Vec<_> {
        x.iter().enumerate().map(|(i, v)| engine.ring.encrypt(sk, v, [seed + i as u8; 32]).unwrap()).collect()
    };
    let through_wire = |cts: Vec<_>| -> Vec<_> {
        cts.iter()
            .map(|c| {
                crate::wire::ciphertext_from_bytes(&engine.ring, &crate::wire::ciphertext_to_bytes(c).unwrap()).unwrap()
            })
            .collect()
    };
    let (u, v) = (through_wire(enc(&u, 10)), through_wire(enc(&v, 100)));
    let t = Instant::now();
    let (out, report) = evaluate(engine, prepared, &u, &v).unwrap();
    eprintln!("evaluated in {:.2?} ({} refreshes)", t.elapsed(), report.refreshes);
    let dec = |cts: &Vec<_>| -> Vec<Vec<f64>> { cts.iter().map(|c| engine.ring.decrypt(sk, c).unwrap()).collect() };
    plan.unpack_tokens(&dec(&out.cols), &dec(&out.rows))
}

#[test]
fn circuit_matches_the_reference_under_ckks() {
    let params = Params::insecure_test(12);
    let plan = PackedPlan::new(tables(), params.slots());
    let (engine, sk) = setup(params, &plan);
    let prepared = Prepared::new(&engine, &plan).unwrap();
    for seed in [1, 2] {
        let s = scramble(seed, 30);
        let want = evaluate_reference(tables(), &one_hot(&to_facelet_colors(&s)));
        assert_eq!(ckks_tokens(&engine, &sk, &plan, &prepared, &s), want);
        assert!(s.apply_all(&tokens_to_moves(&want)).is_solved());
    }
}

#[test]
#[ignore = "secure parameters with bootstrapping: several GB of keys, run with --ignored --nocapture"]
fn circuit_solves_under_secure_parameters() {
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let t = Instant::now();
    let (engine, sk) = setup(params, &plan);
    eprintln!("keys generated and prepared in {:.2?}", t.elapsed());
    let t = Instant::now();
    let prepared = Prepared::new(&engine, &plan).unwrap();
    eprintln!("plan encoded in {:.2?}", t.elapsed());
    let s = scramble(7, 40);
    let tokens = ckks_tokens(&engine, &sk, &plan, &prepared, &s);
    assert_eq!(tokens, evaluate_reference(tables(), &one_hot(&to_facelet_colors(&s))));
}

#[test]
#[ignore = "diagnostic: per-stage drift between CKKS and the simulator"]
fn stage_by_stage_drift() {
    let params = Params::insecure_test(12);
    let slots = params.slots();
    let plan = PackedPlan::new(tables(), slots);
    let (engine, sk) = setup(params, &plan);
    let prepared = Prepared::new(&engine, &plan).unwrap();
    let sim = SlotSim { slots };
    let sim_prepared = Prepared::new(&sim, &plan).unwrap();
    let s = scramble(1, 30);
    let (u, v) = plan.pack_input(&to_facelet_colors(&s));

    let mut sim_states = Vec::new();
    evaluate_inspected(&sim, &sim_prepared, &u, &v, |e| {
        if let Event::Stage { selector, state, .. } = e {
            sim_states.push((selector.clone(), state.to_vec()));
        }
    })
    .unwrap();

    let enc = |x: &Vec<Vec<f64>>, seed: u8| -> Vec<_> {
        x.iter().enumerate().map(|(i, v)| engine.ring.encrypt(&sk, v, [seed + i as u8; 32]).unwrap()).collect()
    };
    let (cu, cv) = (enc(&u, 10), enc(&v, 100));
    evaluate_inspected(&engine, &prepared, &cu, &cv, |e| {
        let Event::Stage { stage: i, selector: sel, state } = e else { return };
        let got_sel = engine.ring.decrypt(&sk, sel).unwrap();
        let (want_sel, want_state) = &sim_states[i];
        let sum_got: f64 = got_sel.iter().sum::<f64>();
        let sum_want: f64 = want_sel.iter().sum::<f64>();
        let state_err = state.iter().zip(want_state).map(|(c, w)| max_err(&engine.ring.decrypt(&sk, c).unwrap(), w)).fold(0.0, f64::max);
        eprintln!(
            "stage {i:2}: selector err {:.2e}, sum {sum_got:.4} (want {sum_want}), state err {state_err:.2e}, budget {}",
            max_err(&got_sel, want_sel), engine.log_budget(sel)
        );
    })
    .unwrap();
}
