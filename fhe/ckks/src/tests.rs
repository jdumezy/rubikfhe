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
    let mut engine = CkksEngine::new(server_ring).unwrap();
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
    let ca = engine.upload(&engine.ring.encrypt(&sk, &a, [3; 32]).unwrap());
    let cb = engine.upload(&engine.ring.encrypt(&sk, &b, [4; 32]).unwrap());
    let check = |name: &str, ct: &crate::engine::DeviceCiphertext, want: Vec<f64>| {
        let got = engine.ring.decrypt(&sk, &engine.download(ct).unwrap()).unwrap();
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
        x.iter()
            .enumerate()
            .map(|(i, v)| engine.upload(&engine.ring.encrypt(sk, v, [seed + i as u8; 32]).unwrap()))
            .collect()
    };
    let through_wire = |cts: Vec<_>| -> Vec<_> {
        cts.iter()
            .map(|c| {
                let bytes = crate::wire::ciphertext_to_bytes(&engine.download(c).unwrap()).unwrap();
                engine.upload(&crate::wire::ciphertext_from_bytes(&engine.ring, &bytes).unwrap())
            })
            .collect()
    };
    let (u, v) = (through_wire(enc(&u, 10)), through_wire(enc(&v, 100)));
    let t = Instant::now();
    let (out, report) = evaluate(engine, prepared, &u, &v).unwrap();
    eprintln!("evaluated in {:.2?} ({} refreshes)", t.elapsed(), report.refreshes);
    // The outputs go back through the byte format too, as the server sends them.
    let dec = |cts: &Vec<_>| -> Vec<Vec<f64>> {
        cts.iter()
            .map(|c| {
                let bytes = crate::wire::ciphertext_to_bytes(&engine.download(c).unwrap()).unwrap();
                engine.ring.decrypt(sk, &crate::wire::ciphertext_from_bytes(&engine.ring, &bytes).unwrap()).unwrap()
            })
            .collect()
    };
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
    drift(Params::insecure_test(12));
}

#[test]
#[ignore = "diagnostic: per-stage drift under the secure parameters, refreshes included"]
fn secure_stage_drift() {
    drift(Params::secure().unwrap());
}

fn drift(params: Params) {
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
        x.iter()
            .enumerate()
            .map(|(i, v)| engine.upload(&engine.ring.encrypt(&sk, v, [seed + i as u8; 32]).unwrap()))
            .collect()
    };
    let (cu, cv) = (enc(&u, 10), enc(&v, 100));
    let (outputs, report) = evaluate_inspected(&engine, &prepared, &cu, &cv, |e| {
        let Event::Stage { stage: i, selector: sel, state } = e else {
            if let Event::Refresh { stage } = e {
                eprintln!("refresh before stage {stage}");
            }
            return;
        };
        let got_sel = engine.ring.decrypt(&sk, &engine.download(sel).unwrap()).unwrap();
        let (want_sel, want_state) = &sim_states[i];
        let sum_got: f64 = got_sel.iter().sum::<f64>();
        let sum_want: f64 = want_sel.iter().sum::<f64>();
        let state_err = state.iter().zip(want_state).map(|(c, w)| max_err(&engine.ring.decrypt(&sk, &engine.download(c).unwrap()).unwrap(), w)).fold(0.0, f64::max);
        eprintln!(
            "stage {i:2}: selector err {:.2e}, sum {sum_got:.4} (want {sum_want}), state err {state_err:.2e}, budget {}",
            max_err(&got_sel, want_sel), engine.log_budget(sel)
        );
    })
    .unwrap();
    // How far the token slots are from their integers.
    let dec = |cts: &[_]| -> Vec<Vec<f64>> {
        cts.iter().map(|c| engine.ring.decrypt(&sk, &engine.download(c).unwrap()).unwrap()).collect()
    };
    let (cols, rows) = (dec(&outputs.cols), dec(&outputs.rows));
    let token_err = plan
        .token_slots
        .iter()
        .flatten()
        .map(|t| {
            let v = if t.family == rubikfhe_circuit::packed::layout::Orientation::Cols {
                cols[t.ct][t.slot]
            } else {
                rows[t.ct][t.slot]
            };
            (v - v.round()).abs()
        })
        .fold(0.0, f64::max);
    eprintln!("{report:?}; token error {token_err:.2e}");
}

/// Forwards to an engine and records the time spent in each kind of operation,
/// waiting for the device after each so that queued work is charged to its call.
struct Timed<'a, E> {
    inner: &'a E,
    sync: &'a (dyn Fn() + Sync),
    times: std::sync::Mutex<std::collections::BTreeMap<&'static str, (u32, std::time::Duration)>>,
}

impl<'a, E: Engine> Timed<'a, E> {
    fn new(inner: &'a E, sync: &'a (dyn Fn() + Sync)) -> Self {
        Timed { inner, sync, times: Default::default() }
    }
    fn time<T>(&self, op: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        (self.sync)();
        let mut times = self.times.lock().unwrap();
        let entry = times.entry(op).or_default();
        entry.0 += 1;
        entry.1 += t.elapsed();
        out
    }
}

impl<E: Engine> Engine for Timed<'_, E> {
    type Ct = E::Ct;
    type Pt = E::Pt;
    type Lt = E::Lt;
    fn slots(&self) -> usize {
        self.inner.slots()
    }
    fn encode(&self, v: &[f64]) -> anyhow::Result<E::Pt> {
        self.inner.encode(v)
    }
    fn encode_lt(&self, d: &rubikfhe_circuit::packed::Diagonals) -> anyhow::Result<E::Lt> {
        self.inner.encode_lt(d)
    }
    fn mul(&self, a: &E::Ct, b: &E::Ct) -> anyhow::Result<E::Ct> {
        self.time("mul (ct x ct)", || self.inner.mul(a, b))
    }
    fn mul_pt(&self, a: &E::Ct, b: &E::Pt) -> anyhow::Result<E::Ct> {
        self.time("mul_pt (masks, codes)", || self.inner.mul_pt(a, b))
    }
    fn copy(&self, a: &E::Ct) -> anyhow::Result<E::Ct> {
        self.time("add/sub", || self.inner.copy(a))
    }
    fn add_assign(&self, a: &mut E::Ct, b: &E::Ct) -> anyhow::Result<()> {
        self.time("add/sub", || self.inner.add_assign(a, b))
    }
    fn sub_assign(&self, a: &mut E::Ct, b: &E::Ct) -> anyhow::Result<()> {
        self.time("add/sub", || self.inner.sub_assign(a, b))
    }
    fn apply_lt(&self, a: &E::Ct, lt: &E::Lt) -> anyhow::Result<E::Ct> {
        self.time("gather (linear transform)", || self.inner.apply_lt(a, lt))
    }
    fn inner_sum(&self, a: &E::Ct, lb: usize, lc: usize) -> anyhow::Result<E::Ct> {
        self.time("inner_sum (folds, broadcast)", || self.inner.inner_sum(a, lb, lc))
    }
    fn replicate(&self, a: &E::Ct, lb: usize, lc: usize) -> anyhow::Result<E::Ct> {
        self.time("replicate", || self.inner.replicate(a, lb, lc))
    }
    fn log_budget(&self, a: &E::Ct) -> usize {
        self.inner.log_budget(a)
    }
    fn stage_cost(&self) -> usize {
        self.inner.stage_cost()
    }
    fn clean_cost(&self) -> usize {
        self.inner.clean_cost()
    }
    fn plaintext_cost(&self) -> usize {
        self.inner.plaintext_cost()
    }
    fn refresh_reserve(&self) -> usize {
        self.inner.refresh_reserve()
    }
    fn refresh_cleans(&self) -> bool {
        self.inner.refresh_cleans()
    }
    fn refresh(&self, a: &E::Ct) -> anyhow::Result<E::Ct> {
        self.time("bootstrap", || self.inner.refresh(a))
    }
}

fn rss_gb() -> f64 {
    let out =
        std::process::Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output().unwrap();
    String::from_utf8(out.stdout).unwrap().trim().parse::<f64>().unwrap() / 1e6
}

#[test]
#[ignore = "profile: time per operation and memory per component at secure parameters"]
fn profile_secure() {
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let base = rss_gb();
    let ring = Ring::new(params.clone()).unwrap();
    let sk = ring.generate_secret([1; 32]);
    let galois = ring.circuit_galois_elements(&plan);
    let keys = ring.generate_eval_keys(&sk, &galois, [2; 32]).unwrap();
    let mut circuit_bytes = Vec::new();
    crate::wire::write_eval_keys(
        &mut circuit_bytes,
        &crate::EvalKeys {
            tensor_key: keys.tensor_key.clone(),
            rotation_keys: keys.rotation_keys.clone(),
            bootstrap: None,
        },
    )
    .unwrap();
    let mut bytes = Vec::new();
    crate::wire::write_eval_keys(&mut bytes, &keys).unwrap();
    drop(keys);
    eprintln!(
        "circuit keys: {} rotations, {:.2} GB serialized; all keys {:.2} GB",
        galois.len(),
        circuit_bytes.len() as f64 / 1e9,
        bytes.len() as f64 / 1e9
    );
    drop(circuit_bytes);

    let mut engine = CkksEngine::new(Ring::new(params).unwrap()).unwrap();
    let before_keys = rss_gb();
    let keys = engine.prepare_keys_from(&mut bytes.as_slice()).unwrap();
    let after_keys = rss_gb();
    drop(bytes);
    engine.set_keys(Some(keys));
    let sync = || engine.synchronize();
    let timed = Timed::new(&engine, &sync);
    let before_plan = rss_gb();
    let prepared = Prepared::new(&timed, &plan).unwrap();
    let after_plan = rss_gb();
    eprintln!(
        "memory: baseline {base:.1} GB, prepared keys {:.1} GB, prepared plan {:.1} GB",
        after_keys - before_keys,
        after_plan - before_plan
    );

    let s = scramble(7, 40);
    let (u, v) = plan.pack_input(&to_facelet_colors(&s));
    let u: Vec<_> = u.iter().map(|x| engine.upload(&engine.ring.encrypt(&sk, x, [3; 32]).unwrap())).collect();
    let v: Vec<_> = v.iter().map(|x| engine.upload(&engine.ring.encrypt(&sk, x, [4; 32]).unwrap())).collect();
    // A first run compiles the device's kernels; the profile is of the next.
    evaluate(&timed, &prepared, &u, &v).unwrap();
    timed.times.lock().unwrap().clear();
    let t = Instant::now();
    let (out, report) = evaluate(&timed, &prepared, &u, &v).unwrap();
    let total = t.elapsed();
    eprintln!("evaluation {total:.2?} ({} bootstraps, {} cleanings):", report.refreshes, report.cleanings);
    for (op, (n, d)) in timed.times.lock().unwrap().iter() {
        eprintln!(
            "  {op:<30} {n:>4} calls  {:>8.0} ms  {:>5.1}%",
            d.as_secs_f64() * 1e3,
            100.0 * d.as_secs_f64() / total.as_secs_f64()
        );
    }
    let dec = |cts: &Vec<_>| -> Vec<Vec<f64>> {
        cts.iter().map(|c| engine.ring.decrypt(&sk, &engine.download(c).unwrap()).unwrap()).collect()
    };
    assert_eq!(
        plan.unpack_tokens(&dec(&out.cols), &dec(&out.rows)),
        evaluate_reference(tables(), &one_hot(&to_facelet_colors(&s)))
    );
}

#[test]
#[ignore = "diagnostic: resident memory after each server construction step"]
fn server_memory_steps() {
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let r0 = rss_gb();
    let ring = Ring::new(params).unwrap();
    let r1 = rss_gb();
    let engine = CkksEngine::new(ring).unwrap();
    let r2 = rss_gb();
    let prepared = Prepared::new(&engine, &plan).unwrap();
    let r3 = rss_gb();
    eprintln!(
        "tables+plan {r0:.2} GB | ring + bootstrap context +{:.2} GB | engine scratch +{:.2} GB | encoded plan +{:.2} GB",
        r1 - r0,
        r2 - r1,
        r3 - r2
    );
    eprintln!("scratch {:.2} GB", engine.scratch_size() as f64 / 1e9);
    drop(prepared);
}

#[test]
fn secure_parameters_fit_the_modulus_bounds() {
    let params = Params::secure().unwrap();
    let preset = params.bootstrap.as_deref().unwrap();
    eprintln!(
        "{} (radix {}): fresh {} bits, refreshed {}, {preset:?}",
        params.name,
        params.base2k,
        params.fresh_k,
        params.refreshed_k()
    );
    // Six stages after the first product, or after a refresh and the selector's isolation.
    let budget = params.refreshed_k() - params.log_delta;
    assert!(budget >= 6 * params.stage_cost() + params.pt_log_delta + params.refresh_reserve());
}

#[test]
#[ignore = "probe: the secure functional bootstrap rounds noisy bits"]
fn functional_preset_cleans_noisy_bits() {
    use poulpy_ckks::api::CKKSBootstrappingOps;
    use poulpy_ckks::layouts::BootstrappingContextPrepared;
    use poulpy_ckks::prelude::*;
    use poulpy_core::layouts::{GLWELayout, GLWESecretAlloc, LWEInfos, Rank};
    use poulpy_core::{GLWESecretPreparedFactory, GLWESecretSampling};
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
    use poulpy_hal::layouts::{Module, ScratchOwned};
    use poulpy_hal::source::Source;

    type Be = crate::backend::Be;
    let params = Params::secure().unwrap();
    let preset = params.bootstrap.as_deref().unwrap();
    let (n, base2k) = (1usize << preset.log_n, preset.base2k);
    let keys = &preset.keys_layout;
    eprintln!(
        "input k {}, output k {}, bootstrap k {}; {} rotation key layouts; d2s k {} s2d k {}",
        preset.input_k,
        preset.output_k,
        preset.bootstrap_k,
        keys.rotation_key_layouts().len(),
        keys.encapsulation.as_ref().unwrap().dense_to_sparse.k().as_usize(),
        keys.encapsulation.as_ref().unwrap().sparse_to_dense.as_ref().unwrap().k().as_usize(),
    );
    let module = Module::<Be>::new(n as u64);
    let lut = preset.lut.transfer_to(&module);
    let mut scratch = ScratchOwned::<Be>::alloc(1 << 31);
    let t = Instant::now();
    let ctx =
        BootstrappingContextPrepared::<Be, f64>::compile(&module, base2k.into(), &preset.plan, &mut scratch.borrow())
            .unwrap();
    let sk_layout = GLWELayout { n: n.into(), base2k: base2k.into(), k: preset.bootstrap_k.into(), rank: Rank(1) };
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, preset.dense_hamming_weight, &mut Source::new([5; 32]));
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&sk_layout);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    let raw_keys = ctx
        .generate_keys(
            &module,
            &sk_raw,
            keys,
            &mut Source::new([7; 32]),
            &mut Source::new([2; 32]),
            &mut Source::new([1; 32]),
            &mut scratch.borrow(),
        )
        .unwrap();
    let mut bytes = Vec::new();
    for k in raw_keys.rotation_keys.values() {
        poulpy_hal::layouts::WriterTo::write_to(k, &mut bytes).unwrap();
    }
    eprintln!("{} rotation keys, {:.2} GB", raw_keys.rotation_keys.len(), bytes.len() as f64 / 1e9);
    let prepared = raw_keys.prepare(&module, &mut scratch.borrow());
    eprintln!("context + keys in {:.1?}", t.elapsed());

    let m = n / 2;
    let bits: Vec<f64> = (0..m).map(|i| (i % 3 == 0) as u8 as f64).collect();
    let in_layout = GLWELayout { k: preset.input_k.into(), ..sk_layout };
    let mut errors = Vec::new();
    for (run, noise) in [0.0, 0.01, 0.05, 0.1, 0.2].into_iter().enumerate() {
        let noisy: Vec<f64> =
            bits.iter().enumerate().map(|(i, b)| b + noise * (((i * 37) % 11) as f64 / 5.0 - 1.0)).collect();
        let mut pt = module.ckks_plaintext_alloc_from_slots(m, base2k.into(), preset.input_k.into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(preset.log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        module.ckks_encode_slots_host_into(&mut pt, &noisy, &vec![0.0; m], &mut scratch.borrow()).unwrap();
        let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&in_layout);
        module
            .ckks_encrypt_sk(
                &mut ct,
                &pt,
                &sk,
                &mut Source::new([4; 32]),
                &mut Source::new([3; 32]),
                &mut scratch.borrow(),
            )
            .unwrap();
        ct.set_slots(SlotsKind::Real);

        // The bootstrap narrows its output to `output_k`: start at the working width.
        let mut outputs = vec![module.ckks_ciphertext_alloc(base2k.into(), preset.bootstrap_k.into())];
        let t = Instant::now();
        module
            .ckks_functional_bootstrap(
                &mut outputs,
                &ct,
                &ctx,
                std::slice::from_ref(&lut),
                &prepared,
                &mut scratch.borrow(),
            )
            .unwrap();
        let ms = t.elapsed().as_millis();
        let out = &outputs[0];
        let mut dec = module.ckks_plaintext_alloc_from_slots(m, base2k.into(), (out.log_delta() + 8).into());
        dec.set_meta(out.meta());
        module.ckks_decrypt(&mut dec, out, &sk, &mut scratch.borrow()).unwrap();
        let (mut re, mut im) = (vec![0.0; m], vec![0.0; m]);
        module.ckks_decode_slots_host_into(&dec, &mut re, &mut im, &mut scratch.borrow()).unwrap();
        let err = re.iter().zip(&bits).map(|(r, b)| (r - b).abs()).fold(0.0, f64::max);
        if run == 0 {
            eprintln!("output k {} log_delta {} budget {}", out.k().as_usize(), out.log_delta(), out.log_budget());
        }
        let mean = re.iter().zip(&bits).map(|(r, b)| (r - b).abs()).sum::<f64>() / m as f64;
        eprintln!("input noise {noise:.0e}: max error {err:.2e} ({:.1} bits), mean {mean:.2e}, {ms} ms", -err.log2());
        errors.push(err);
    }
    assert!(errors[2] < 1e-2);
}

#[test]
#[ignore = "probe: the secure engine refresh on fresh bits"]
fn secure_refresh_rounds_bits() {
    use poulpy_ckks::CKKSInfos;
    use poulpy_core::layouts::LWEInfos;
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let (engine, sk) = setup(params, &plan);
    let m = engine.slots();
    let bits: Vec<f64> = (0..m).map(|i| (i % 3 == 0) as u8 as f64).collect();
    let noisy: Vec<f64> =
        bits.iter().enumerate().map(|(i, b)| b + 0.01 * (((i * 37) % 11) as f64 / 5.0 - 1.0)).collect();
    let ct = engine.upload(&engine.ring.encrypt(&sk, &noisy, [9; 32]).unwrap());
    eprintln!("fresh: k {} log_delta {} budget {}", ct.k().as_usize(), ct.log_delta(), ct.log_budget());
    let fresh = engine.ring.decrypt(&sk, &engine.download(&ct).unwrap()).unwrap();
    eprintln!("fresh error {:.2e}", max_err(&fresh, &bits));
    let t = Instant::now();
    let out = engine.refresh(&ct).unwrap();
    eprintln!(
        "refresh {:?}: k {} log_delta {} budget {}",
        t.elapsed(),
        out.k().as_usize(),
        out.log_delta(),
        out.log_budget()
    );
    let got = engine.ring.decrypt(&sk, &engine.download(&out).unwrap()).unwrap();
    let err = max_err(&got, &bits);
    eprintln!("refreshed error {err:.2e}");
    assert!(err < 1e-3);
}

#[test]
#[ignore = "probe: a cold and a warm run of the circuit on the device"]
fn device_cold_and_warm() {
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let (engine, sk) = setup(params, &plan);
    let prepared = Prepared::new(&engine, &plan).unwrap();
    let (u, v) = plan.pack_input(&to_facelet_colors(&scramble(7, 40)));
    let enc = |x: &Vec<Vec<f64>>| -> Vec<_> {
        x.iter().map(|v| engine.upload(&engine.ring.encrypt(&sk, v, [5; 32]).unwrap())).collect()
    };
    let (u, v) = (enc(&u), enc(&v));
    for run in ["cold", "warm", "warm"] {
        let t = Instant::now();
        evaluate(&engine, &prepared, &u, &v).unwrap();
        engine.synchronize();
        eprintln!("{run} run {:.2?}", t.elapsed());
    }
}

#[test]
#[ignore = "probe: each operation of the circuit alone on the device, synchronized"]
fn device_op_costs() {
    use rubikfhe_circuit::packed::layout::LOG_SIDE;
    let params = Params::secure().unwrap();
    let plan = PackedPlan::new(tables(), params.slots());
    let (engine, sk) = setup(params, &plan);
    let x: Vec<f64> = (0..engine.slots()).map(|i| (i % 2) as f64).collect();
    let a = engine.upload(&engine.ring.encrypt(&sk, &x, [3; 32]).unwrap());
    let b = engine.upload(&engine.ring.encrypt(&sk, &x, [4; 32]).unwrap());
    let reps = 5;
    let time = |name: &str, f: &mut dyn FnMut()| {
        f();
        engine.synchronize();
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        engine.synchronize();
        eprintln!("{name:<28} {:>9.2?}", t.elapsed() / reps);
    };
    let pt = engine.encode(&x).unwrap();
    let lt = engine.encode_lt(&plan.stages[0].gather[0]).unwrap();
    let t = engine.apply_lt(&a, &lt).unwrap();
    time("mul fresh x fresh", &mut || drop(engine.mul(&a, &b).unwrap()));
    time("mul gathered x fresh", &mut || drop(engine.mul(&t, &b).unwrap()));
    time("mul_pt", &mut || drop(engine.mul_pt(&a, &pt).unwrap()));
    time("apply_lt", &mut || drop(engine.apply_lt(&a, &lt).unwrap()));
    time("inner_sum 5x32", &mut || drop(engine.inner_sum(&a, LOG_SIDE, LOG_SIDE).unwrap()));
    time("replicate 5x32", &mut || drop(engine.replicate(&a, LOG_SIDE, LOG_SIDE).unwrap()));
    time("refresh", &mut || drop(engine.refresh(&a).unwrap()));
    let prepared = Prepared::new(&engine, &plan).unwrap();
    eprintln!("-- with the plan encoded");
    time("mul gathered x fresh", &mut || drop(engine.mul(&t, &b).unwrap()));
    time("apply_lt", &mut || drop(engine.apply_lt(&a, &lt).unwrap()));
    time("refresh", &mut || drop(engine.refresh(&a).unwrap()));
    let (u, v) = plan.pack_input(&to_facelet_colors(&scramble(7, 40)));
    let enc = |x: &Vec<Vec<f64>>| -> Vec<_> {
        x.iter().map(|v| engine.upload(&engine.ring.encrypt(&sk, v, [5; 32]).unwrap())).collect()
    };
    let (u, v) = (enc(&u), enc(&v));
    let t0 = Instant::now();
    evaluate(&engine, &prepared, &u, &v).unwrap();
    engine.synchronize();
    eprintln!("whole circuit {:.2?}", t0.elapsed());
    eprintln!("-- after the circuit");
    time("mul gathered x fresh", &mut || drop(engine.mul(&t, &b).unwrap()));
    time("apply_lt", &mut || drop(engine.apply_lt(&a, &lt).unwrap()));
    time("refresh", &mut || drop(engine.refresh(&a).unwrap()));
}
