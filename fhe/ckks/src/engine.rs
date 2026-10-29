//! CKKS implementation of the circuit [`Engine`], on poulpy.
//!
//! The client's side ([`Ring`]: secret, keys, encryption) runs on the CPU
//! backend [`Be`], which also holds the keys and ciphertexts the processes
//! exchange. The server's [`CkksEngine`] runs the circuit and its bootstrap on
//! the device backend [`Dev`], a GPU when the build enables one: keys move to
//! the device as they are prepared, ciphertexts on the way in and out.

use std::collections::BTreeMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use poulpy_ckks::api::{
    CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSInnerSumOps,
    CKKSLinearTransformationOps,
};
use poulpy_ckks::encoding::ckks_encode_linear_transformation_from_diagonals;
use poulpy_ckks::layouts::{
    BootstrappingContext, BootstrappingKeySet, BootstrappingKeysLayout, BootstrappingKeysPrepared, CKKSCiphertextOwned,
    CKKSPlaintextOwned, ComplexDiagonals, EncapsulationKeys, EncodedLut, LinearTransformationPrepared,
};
use poulpy_ckks::prelude::*;
use poulpy_core::layouts::prepared::{GLWEAutomorphismKeyPrepared, GLWESwitchingKeyPrepared, GLWETensorKeyPrepared};
use poulpy_core::layouts::{
    BackendGLWEAutomorphismKey, BackendGLWEAutomorphismKeyPrepared, BackendGLWESecret, BackendGLWESecretPrepared,
    BackendGLWETensorKeyPrepared, Degree, Diagonals as CoreDiagonals, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendRef,
    GGLWEViewMut, GLWEInfos, GLWESwitchingKeyDegrees, GLWETensorKey, GetGaloisElement, LWEInfos,
    LinearTransformationLayout, LinearTransformationStrategy, ModuleCoreAlloc, optimal_bsgs_giant_step,
};
use poulpy_core::layouts::{GLWEAutomorphismKey, GLWECIEmbedKey, GLWECITraceKey, GLWESwitchingKey};
use poulpy_core::layouts::{GetAutomorphismKey, TorusPrecision, prepared::GLWEAutomorphismKeyPreparedBackendRef};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWEAutomorphismKeyPreparedFactory, GLWEBytesOf, GLWESecretPreparedFactory,
    GLWESecretSampling, GLWESwitchingKeyEncryptSk, GLWESwitchingKeyPreparedFactory, GLWETensorKeyEncryptSk,
    GLWETensorKeyPreparedFactory, ScratchArenaTakeCore, TransferInto,
};
use poulpy_hal::api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchArena, ScratchOwned};
use poulpy_hal::source::Source;
use rubikfhe_circuit::packed::layout::{LOG_BLOCK, LOG_SIDE};
use rubikfhe_circuit::packed::{Diagonals, Engine, PackedPlan, Prepared, evaluate};

use crate::backend::{Be, Dev};
use crate::params::Params;
use crate::wire::{KeySink, read_eval_keys_into};

/// A ciphertext as the client encrypts it and the processes exchange it.
pub type Ciphertext = CKKSCiphertextOwned<Be>;
/// A ciphertext on the server's device.
pub type DeviceCiphertext = CKKSCiphertextOwned<Dev>;
pub type Plaintext = CKKSPlaintextOwned<Dev>;
type Word = <Be as Backend>::ZnxWord;
type Buf = <Be as Backend>::OwnedBuf;
type DevBuf = <Dev as Backend>::OwnedBuf;

/// Extra bits above the plaintext precision, for the integer part of codes (≤ 18).
const PT_INTEGER_BITS: usize = 8;

fn pt_meta(log_delta: usize) -> GLWEPlaintextMeta {
    GLWEPlaintextMeta { scale: Scale::Log(log_delta), slots: SlotsKind::Complex, log_sparsity: 0 }
}

/// Bootstrapping context and LUT, compiled from the preset on the device.
struct Bootstrap {
    context: BootstrappingContext<Dev, f64>,
    lut: EncodedLut<CKKSPlaintextOwned<Dev>>,
}

impl Bootstrap {
    fn new(module: &Module<Dev>, params: &Params) -> Result<Option<Bootstrap>> {
        let Some(preset) = &params.bootstrap else { return Ok(None) };
        let bytes = <Module<Dev> as CKKSEncodingHostOps<Dev, f64>>::ckks_reim_tmp_bytes(module, module.n());
        let unprepared = BootstrappingContext::<Dev, f64>::compile_unprepared(
            module,
            preset.base2k.into(),
            &preset.plan,
            &mut ScratchOwned::<Dev>::alloc(bytes).borrow(),
        )?;
        let bytes = unprepared.prepare_tmp_bytes(module);
        let context = unprepared.prepare(module, &mut ScratchOwned::<Dev>::alloc(bytes).borrow())?;
        Ok(Some(Bootstrap { context, lut: preset.lut.transfer_to(module) }))
    }
}

/// Scratch to generate the bootstrapping keys of `layout`.
fn bootstrap_keygen_tmp_bytes(module: &Module<Be>, layout: &BootstrappingKeysLayout) -> usize {
    let mut bytes = module.glwe_tensor_key_encrypt_sk_tmp_bytes(&layout.tensor_key);
    for key in layout.rotation_key_layouts() {
        bytes = bytes.max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&key));
    }
    if let Some(e) = &layout.encapsulation {
        for key in [Some(e.dense_to_sparse), e.sparse_to_dense].into_iter().flatten() {
            bytes = bytes.max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key));
        }
    }
    bytes
}

/// Public evaluation keys, as the client generates them.
pub struct EvalKeys {
    pub tensor_key: GLWETensorKey<Buf, Word>,
    pub rotation_keys: BTreeMap<i64, BackendGLWEAutomorphismKey<Be>>,
    pub bootstrap: Option<BootstrappingKeySet<Buf, Word>>,
}

/// The circuit's ring on the CPU, keyless: what both sides derive from the
/// parameters, and where the client works.
pub struct Ring {
    pub params: Params,
    pub module: Module<Be>,
}

impl Ring {
    pub fn new(params: Params) -> Result<Ring> {
        let module = Module::<Be>::new(params.n() as u64);
        Ok(Ring { params, module })
    }

    fn cyclotomic_order(&self) -> i64 {
        self.module.cyclotomic_order()
    }

    fn scratch(&self) -> ScratchOwned<Be> {
        ScratchOwned::<Be>::alloc(self.scratch_bytes())
    }

    fn scratch_bytes(&self) -> usize {
        let p = &self.params;
        let ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&p.glwe_layout());
        let atk = p.rotation_key_layout();
        self.module
            .ckks_all_ops_tmp_bytes(&ct, &ct, &p.tensor_key_layout())
            .max(self.module.ckks_inner_sum_tmp_bytes(&ct, &ct, &atk))
    }

    fn lt_strategy(indexes: &[i64], slots: usize) -> LinearTransformationStrategy {
        LinearTransformationStrategy::BsgsPowersOfTwo {
            giant_step: optimal_bsgs_giant_step(indexes.iter().copied(), slots),
        }
    }

    /// Galois elements of every rotation the circuit performs, for `plan`.
    pub fn circuit_galois_elements(&self, plan: &PackedPlan) -> Vec<i64> {
        let n = Degree(self.params.n() as u32);
        let mut out = std::collections::BTreeSet::new();
        for (log_batch, log_count) in
            [(0, LOG_SIDE), (LOG_SIDE, LOG_SIDE), (LOG_BLOCK, plan.layout.log_blocks_per_ct())]
        {
            if log_count > 0 {
                out.extend(self.module.ckks_inner_sum_galois_elements(n, log_batch, log_count));
                out.extend(self.module.ckks_replicate_galois_elements(n, log_batch, log_count));
            }
        }
        for stage in &plan.stages {
            for diagonals in &stage.gather {
                let indexes: Vec<i64> = diagonals.keys().map(|&i| i as i64).collect();
                let slots = self.params.slots();
                let layout = LinearTransformationLayout {
                    indexes: indexes.clone(),
                    slots,
                    strategy: Self::lt_strategy(&indexes, slots),
                };
                out.extend(layout.galois_elements(self.cyclotomic_order()));
            }
        }
        out.into_iter().collect()
    }
}

/// The client's secret key, on the circuit ring.
pub struct SecretKey {
    pub raw: BackendGLWESecret<Be>,
    pub prepared: BackendGLWESecretPrepared<Be>,
}

impl Ring {
    pub fn generate_secret(&self, seed: [u8; 32]) -> SecretKey {
        let layout = self.params.glwe_layout();
        let mut raw = self.module.glwe_secret_alloc_from_infos(&layout);
        self.module.glwe_secret_fill_ternary_hw(&mut raw, self.params.hamming_weight, &mut Source::new(seed));
        let mut prepared = self.module.glwe_secret_prepared_alloc_from_infos(&layout);
        self.module.glwe_secret_prepare(&mut prepared, &raw);
        SecretKey { raw, prepared }
    }

    /// Evaluation keys: relinearization, the circuit's rotations, and the
    /// bootstrapping keys when the parameters bootstrap.
    pub fn generate_eval_keys(&self, sk: &SecretKey, galois_elements: &[i64], seed: [u8; 32]) -> Result<EvalKeys> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let mut xa = Source::new(seed);
        let mut xe = Source::new([seed, [0x5a; 32]].concat()[16..48].try_into().unwrap());

        let mut tensor_key = self.module.glwe_tensor_key_alloc_from_infos(&p.tensor_key_layout());
        self.module.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk.raw, &mut xa, &mut xe, &mut scratch.borrow());

        let bootstrap = match &p.bootstrap {
            Some(preset) => {
                let layout = &preset.keys_layout;
                let bytes = bootstrap_keygen_tmp_bytes(&self.module, layout);
                let mut xs = Source::new([seed, [0xc3; 32]].concat()[8..40].try_into().unwrap());
                Some(preset.plan.generate_keys(
                    &self.module,
                    &sk.raw,
                    layout,
                    &mut xs,
                    &mut xe,
                    &mut xa,
                    &mut ScratchOwned::<Be>::alloc(bytes).borrow(),
                )?)
            }
            None => None,
        };
        let mut rotation_keys = BTreeMap::new();
        // The bootstrapping keys also serve the circuit's rotations, where their
        // gadget covers the circuit's widest ciphertexts (some are narrower).
        let shared = |g: i64| {
            let Some((b, preset)) = bootstrap.as_ref().zip(p.bootstrap.as_ref()) else { return false };
            let layout = preset.keys_layout.rotation_key(g);
            b.rotation_keys.contains_key(&g) && (layout.dnum.0 * layout.k_digit.0) as usize >= p.fresh_k
        };
        for &g in galois_elements.iter().filter(|&&g| !shared(g)) {
            let mut key = self.module.glwe_automorphism_key_alloc_from_infos(&p.rotation_key_layout());
            self.module.glwe_automorphism_key_encrypt_sk(&mut key, g, &sk.raw, &mut xe, &mut xa, &mut scratch.borrow());
            rotation_keys.insert(g, key);
        }

        Ok(EvalKeys { tensor_key, rotation_keys, bootstrap })
    }

    /// Encrypts slot values at the fresh width and scale.
    pub fn encrypt(&self, sk: &SecretKey, values: &[f64], seed: [u8; 32]) -> Result<Ciphertext> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let mut pt = self.module.ckks_pt_vec_alloc(p.slots(), p.base2k.into(), p.fresh_k.into());
        pt.set_meta(pt_meta(p.log_delta));
        self.module.ckks_encode_reim_into(&mut pt, values, &vec![0.0; values.len()], &mut scratch.borrow())?;
        let mut ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&p.glwe_layout());
        let mut xe = Source::new([seed, [0x17; 32]].concat()[24..56].try_into().unwrap());
        self.module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &sk.prepared,
            &mut xe,
            &mut Source::new(seed),
            &mut scratch.borrow(),
        )?;
        Ok(ct)
    }

    /// Decrypts slot values (real parts).
    pub fn decrypt(&self, sk: &SecretKey, ct: &Ciphertext) -> Result<Vec<f64>> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let budget = ct.log_budget().min(PT_INTEGER_BITS + 8);
        let mut pt = self.module.ckks_pt_vec_alloc(p.slots(), p.base2k.into(), (ct.log_delta() + budget).into());
        pt.set_meta(ct.meta());
        self.module.ckks_decrypt(&mut pt, ct, &sk.prepared, &mut scratch.borrow())?;
        let (mut re, mut im) = (vec![0.0; p.slots()], vec![0.0; p.slots()]);
        self.module.ckks_decode_reim_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
        Ok(re)
    }
}

/// Evaluation keys prepared on the device.
pub struct PreparedKeys {
    tensor_key: BackendGLWETensorKeyPrepared<Dev>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    bootstrap: Option<BootstrappingKeysPrepared<DevBuf, Dev>>,
}

/// The CKKS [`Engine`]: the circuit ring on the device, with prepared
/// evaluation keys.
pub struct CkksEngine {
    pub ring: Ring,
    module: Module<Dev>,
    bootstrap: Option<Bootstrap>,
    keys: Option<PreparedKeys>,
    /// One scratch arena, and its size, for everything that runs on the
    /// device: circuit operations, bootstrapping and key preparation never
    /// overlap, so they share the largest arena any of them needs.
    scratch: Mutex<(usize, ScratchOwned<Dev>)>,
}

impl CkksEngine {
    pub fn new(ring: Ring) -> Result<CkksEngine> {
        let module = Module::<Dev>::new(ring.params.n() as u64);
        let bootstrap = Bootstrap::new(&module, &ring.params)?;
        let mut engine =
            CkksEngine { ring, module, bootstrap, keys: None, scratch: Mutex::new((0, ScratchOwned::alloc(0))) };
        let p = &engine.ring.params;
        // The circuit's keys are prepared beside their copy on the device.
        let circuit_keys = engine.moved_bytes(&p.tensor_key_layout())
            + engine.module.glwe_tensor_key_prepare_tmp_bytes(&p.tensor_key_layout());
        let rotation_keys = engine.moved_bytes(&p.rotation_key_layout())
            + engine.module.glwe_automorphism_key_prepare_tmp_bytes(&p.rotation_key_layout());
        let mut bytes = engine.ops_scratch_bytes().max(circuit_keys).max(rotation_keys);
        if let (Some(b), Some(preset)) = (&engine.bootstrap, p.bootstrap.as_deref()) {
            let boot = engine.module.ckks_functional_bootstrap_tmp_bytes(
                &engine.alloc_like(preset.bootstrap_k),
                &engine.alloc_like(preset.input_k),
                &b.context,
                std::slice::from_ref(&b.lut),
                &preset.keys_layout,
            );
            bytes = bytes.max(boot).max(preset.keys_layout.prepare_tmp_bytes(&engine.module));
        }
        engine.scratch = Mutex::new((bytes, ScratchOwned::<Dev>::alloc(bytes)));
        Ok(engine)
    }

    pub fn has_keys(&self) -> bool {
        self.keys.is_some()
    }

    /// Bytes of the scratch arena.
    pub fn scratch_size(&self) -> usize {
        self.scratch.lock().unwrap().0
    }

    /// Scratch of the circuit's operations.
    fn ops_scratch_bytes(&self) -> usize {
        let p = &self.ring.params;
        let ct = self.alloc_like(p.fresh_k);
        let atk = p.rotation_key_layout();
        self.module
            .ckks_all_ops_tmp_bytes(&ct, &ct, &p.tensor_key_layout())
            .max(self.module.ckks_inner_sum_tmp_bytes(&ct, &ct, &atk))
    }

    /// Scratch a key of `layout` takes once moved to the device.
    fn moved_bytes(&self, layout: &impl GGLWEInfos) -> usize {
        Dev::scratch_aligned(self.module.gglwe_bytes_of_from_infos(layout))
    }

    /// Grows the scratch arena to at least `bytes` on top of the base arena.
    fn reserve_scratch(&self, bytes: usize) {
        let mut guard = self.scratch.lock().unwrap();
        if guard.0 < bytes {
            *guard = (bytes, ScratchOwned::<Dev>::alloc(bytes.max(self.ops_scratch_bytes())));
        }
    }

    /// The circuit's rotation keys, falling back on the bootstrapping keys.
    fn rotations(&self) -> Result<Rotations<'_>> {
        let keys = self.keys()?;
        Ok(Rotations { own: &keys.rotation_keys, shared: keys.bootstrap.as_ref().map(|b| &b.rotation_keys) })
    }

    fn keys(&self) -> Result<&PreparedKeys> {
        self.keys.as_ref().context("no evaluation keys installed")
    }

    /// A ciphertext of width `k`. Its storage spans the fresh width at least,
    /// whatever `k`: the device reuses a freed buffer only for a request of
    /// the same size, and a ciphertext narrowing stage by stage would
    /// otherwise make every allocation a new, zero-filled device buffer.
    fn alloc_like(&self, k: usize) -> DeviceCiphertext {
        let mut layout = self.ring.params.glwe_layout();
        layout.k = k.max(self.ring.params.fresh_k).into();
        let mut ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        ct.set_k(k.into());
        ct
    }

    /// Runs the circuit once on encryptions of zero. A GPU compiles each
    /// kernel the first time it runs, which made the first solve take four
    /// times as long as the next ones; this moves that cost to key setup.
    pub fn warm_up(&self, prepared: &Prepared<CkksEngine>) -> Result<()> {
        let p = &self.ring.params;
        let zeros: Vec<_> = (0..prepared.state_cts)
            .map(|_| {
                let mut ct = self.alloc_like(p.fresh_k);
                ct.set_meta(pt_meta(p.log_delta));
                ct
            })
            .collect();
        evaluate(self, prepared, &zeros, &zeros)?;
        self.synchronize();
        Ok(())
    }

    /// Waits for the work queued on the device.
    pub fn synchronize(&self) {
        self.module.synchronize();
    }

    /// Moves a received ciphertext to the device.
    pub fn upload(&self, ct: &Ciphertext) -> DeviceCiphertext {
        let mut out = self.alloc_like(ct.k().as_usize());
        (**ct).transfer_into(&mut *out);
        out.set_meta(ct.meta());
        out
    }

    /// Brings a ciphertext back from the device, to send or decrypt, with the
    /// storage of its width: device ciphertexts hold more (see [`Self::alloc_like`]).
    pub fn download(&self, ct: &DeviceCiphertext) -> Result<Ciphertext> {
        let module = &self.ring.module;
        let mut layout = ct.glwe_layout();
        layout.k = (ct.data().size() * self.ring.params.base2k).into();
        let mut held = module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        held.set_k(ct.k());
        (**ct).transfer_into(&mut *held);
        held.set_meta(ct.meta());
        let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&ct.glwe_layout());
        module.ckks_copy(&mut out, &held, &mut self.ring.scratch().borrow())?;
        Ok(out)
    }
}

impl Engine for CkksEngine {
    type Ct = DeviceCiphertext;
    type Pt = Plaintext;
    type Lt = LinearTransformationPrepared<Dev>;

    fn slots(&self) -> usize {
        self.ring.params.slots()
    }

    fn encode(&self, values: &[f64]) -> Result<Plaintext> {
        let p = &self.ring.params;
        let mut pt =
            self.module.ckks_pt_vec_alloc(p.slots(), p.base2k.into(), (p.pt_log_delta + PT_INTEGER_BITS).into());
        pt.set_meta(pt_meta(p.pt_log_delta));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_encode_reim_into(&mut pt, values, &vec![0.0; values.len()], &mut scratch.borrow())?;
        Ok(pt)
    }

    fn encode_lt(&self, diagonals: &Diagonals) -> Result<Self::Lt> {
        let p = &self.ring.params;
        let slots = p.slots();
        let (mut re, im) = (CoreDiagonals::<f64>::new(slots), CoreDiagonals::<f64>::new(slots));
        for (&i, values) in diagonals {
            re.set(i as i64, values.clone());
        }
        let indexes: Vec<i64> = diagonals.keys().map(|&i| i as i64).collect();
        let module = &self.module;
        let (lt, plan) = {
            let mut guard = self.scratch.lock().unwrap();
            let scratch = &mut guard.1;
            let lt = ckks_encode_linear_transformation_from_diagonals(
                module,
                p.base2k.into(),
                CoeffsMeta::from_delta_budget(p.pt_log_delta, PT_INTEGER_BITS),
                &ComplexDiagonals::new(re, im),
                Ring::lt_strategy(&indexes, slots),
                false,
                &mut scratch.borrow(),
            )?;
            let first = lt.first_diagonal_plaintext().context("empty gather map")?;
            let mut prepared =
                LinearTransformationPrepared::<Dev>::alloc_prepared_from_index(module, &lt.index(), first);
            module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow())?;
            (prepared, lt.index())
        };
        let ct = self.alloc_like(p.fresh_k);
        let atk = p.rotation_key_layout();
        self.reserve_scratch(
            module
                .ckks_eval_linear_transformation_tmp_bytes(&ct, &plan, &atk)
                .max(module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &atk))
                + self.ops_scratch_bytes(),
        );
        Ok(lt)
    }

    fn copy(&self, a: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_copy(&mut out, a, &mut scratch.borrow())?;
        Ok(out)
    }

    fn mul(&self, a: &DeviceCiphertext, b: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize().min(b.k().as_usize()));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_mul_into(&mut out, a, b, &self.keys()?.tensor_key, &mut scratch.borrow())?;
        Ok(out)
    }

    fn mul_pt(&self, a: &DeviceCiphertext, b: &Plaintext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_mul_pt_vec_into(&mut out, a, b, &mut scratch.borrow())?;
        Ok(out)
    }

    fn add_assign(&self, a: &mut DeviceCiphertext, b: &DeviceCiphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_add_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn sub_assign(&self, a: &mut DeviceCiphertext, b: &DeviceCiphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_sub_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn apply_lt(&self, a: &DeviceCiphertext, lt: &Self::Lt) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_eval_linear_transformation_self_into(
            &mut out,
            a,
            lt,
            &self.rotations()?,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn inner_sum(&self, a: &DeviceCiphertext, log_batch: usize, log_count: usize) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_inner_sum(
            log_batch,
            log_count,
            false,
            &mut out,
            a,
            &self.rotations()?,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn replicate(&self, a: &DeviceCiphertext, log_batch: usize, log_count: usize) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_replicate(log_batch, log_count, &mut out, a, &self.rotations()?, &mut scratch.borrow())?;
        Ok(out)
    }

    fn log_budget(&self, a: &DeviceCiphertext) -> usize {
        a.log_budget()
    }

    fn stage_cost(&self) -> usize {
        self.ring.params.stage_cost()
    }

    fn clean_cost(&self) -> usize {
        self.ring.params.clean_cost()
    }

    fn plaintext_cost(&self) -> usize {
        self.ring.params.pt_log_delta
    }

    fn refresh_reserve(&self) -> usize {
        self.ring.params.refresh_reserve()
    }

    fn refresh_cleans(&self) -> bool {
        self.bootstrap.is_some()
    }

    /// Functional bootstrap: relabels the input at a higher scale, where the
    /// bootstrap's own error is smaller, refreshes it through the LUT
    /// `0 → 0, 1 → 1`, and relabels the output back at the circuit's scale.
    fn refresh(&self, a: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let (Some(b), Some(preset), Some(keys)) =
            (&self.bootstrap, &self.ring.params.bootstrap, &self.keys()?.bootstrap)
        else {
            anyhow::bail!("out of capacity and these parameters do not bootstrap");
        };
        let log_delta = self.ring.params.log_delta;
        let scratch = &mut self.scratch.lock().unwrap().1;
        // The budget the bootstrap reads, at the circuit's scale; the relabel
        // keeps the budget and widens the ciphertext to the bootstrap's input.
        let mut input = self.alloc_like(preset.input_k - (preset.log_delta - log_delta));
        self.module.ckks_copy(&mut input, a, &mut scratch.borrow())?;
        input.set_log_delta(preset.log_delta);
        // The values are real: the bootstrap skips the imaginary half.
        input.set_slots(SlotsKind::Real);
        let mut outputs = [self.alloc_like(preset.bootstrap_k)];
        self.module.ckks_functional_bootstrap(
            &mut outputs,
            &input,
            &b.context,
            std::slice::from_ref(&b.lut),
            keys,
            &mut scratch.borrow(),
        )?;
        let [mut out] = outputs;
        out.set_log_delta(log_delta);
        out.set_slots(SlotsKind::Complex);
        Ok(out)
    }
}

/// A received key's matrix moved into a carve of the device scratch, with what
/// its preparation reads besides: its Galois element and its degrees.
struct Moved<'a> {
    matrix: GGLWEViewMut<'a, Dev>,
    p: i64,
    degrees: (Degree, Degree),
}

impl<'a> Moved<'a> {
    /// Moves `key` into the front of `scratch`, returning the rest.
    fn of<K>(
        key: &K,
        p: i64,
        degrees: (Degree, Degree),
        scratch: ScratchArena<'a, Dev>,
    ) -> (Self, ScratchArena<'a, Dev>)
    where
        K: GGLWEInfos + TransferInto<GGLWEViewMut<'a, Dev>>,
    {
        let (mut matrix, rest) = scratch.take_gglwe_scratch(key);
        key.transfer_into(&mut matrix);
        (Moved { matrix, p, degrees }, rest)
    }
}

impl GGLWEToBackendRef<Dev> for Moved<'_> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, Dev> {
        self.matrix.to_backend_ref()
    }
}

impl GetGaloisElement for Moved<'_> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl GLWESwitchingKeyDegrees for Moved<'_> {
    fn input_degree(&self) -> &Degree {
        &self.degrees.0
    }

    fn output_degree(&self) -> &Degree {
        &self.degrees.1
    }
}

/// Prepares the keys of a stream one by one, as they arrive, so that raw keys
/// never pile up next to their prepared form.
struct Preparing<'a> {
    engine: &'a CkksEngine,
    scratch: &'a mut ScratchOwned<Dev>,
    tensor_key: Option<BackendGLWETensorKeyPrepared<Dev>>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    boot_rotation_keys: std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<DevBuf, Dev>>,
    boot_tensor_key: Option<GLWETensorKeyPrepared<DevBuf, Dev>>,
    encapsulation: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<DevBuf, Dev>>>,
    first_factor_keys: std::collections::HashMap<i64, GLWESwitchingKeyPrepared<DevBuf, Dev>>,
}

impl<'a> Preparing<'a> {
    fn new(engine: &'a CkksEngine, scratch: &'a mut ScratchOwned<Dev>) -> Self {
        Preparing {
            engine,
            scratch,
            tensor_key: None,
            rotation_keys: Default::default(),
            boot_rotation_keys: Default::default(),
            boot_tensor_key: None,
            encapsulation: None,
            first_factor_keys: Default::default(),
        }
    }

    /// The module, for a bootstrapping key.
    fn boot(&self) -> Result<&'a Module<Dev>> {
        self.engine.bootstrap.as_ref().context("bootstrapping keys for parameters without bootstrapping")?;
        Ok(&self.engine.module)
    }

    fn tensor(&mut self, key: &GLWETensorKey<Buf, Word>) -> GLWETensorKeyPrepared<DevBuf, Dev> {
        let module = &self.engine.module;
        let (mut moved, mut rest) = self.scratch.borrow().take_gglwe_scratch(key);
        key.transfer_into(&mut moved);
        let mut prepared = module.alloc_tensor_key_prepared_from_infos(key);
        module.glwe_tensor_key_prepare(&mut prepared, &moved, &mut rest);
        prepared
    }

    fn automorphism(&mut self, key: &GLWEAutomorphismKey<Buf, Word>) -> GLWEAutomorphismKeyPrepared<DevBuf, Dev> {
        let module = &self.engine.module;
        let (moved, mut rest) = Moved::of(key, key.p(), (key.n(), key.n()), self.scratch.borrow());
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(key);
        module.glwe_automorphism_key_prepare(&mut prepared, &moved, &mut rest);
        prepared
    }

    fn switching_key(&mut self, key: &GLWESwitchingKey<Buf, Word>) -> Result<GLWESwitchingKeyPrepared<DevBuf, Dev>> {
        let module = self.boot()?;
        let degrees = (*key.input_degree(), *key.output_degree());
        let (moved, mut rest) = Moved::of(key, 0, degrees, self.scratch.borrow());
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
        module.glwe_switching_key_prepare(&mut prepared, &moved, &mut rest);
        Ok(prepared)
    }

    fn finish(self, bootstrap: bool) -> Result<PreparedKeys> {
        let bootstrap = if bootstrap {
            Some(BootstrappingKeysPrepared {
                rotation_keys: self.boot_rotation_keys,
                tensor_key: self.boot_tensor_key.context("missing bootstrapping tensor key")?,
                encapsulation_keys: self.encapsulation,
                first_factor_keys: self.first_factor_keys,
                conversion_keys: None,
            })
        } else {
            None
        };
        if bootstrap.is_some() != self.engine.bootstrap.is_some() {
            anyhow::bail!("bootstrapping keys do not match the parameters");
        }
        Ok(PreparedKeys {
            tensor_key: self.tensor_key.context("missing tensor key")?,
            rotation_keys: self.rotation_keys,
            bootstrap,
        })
    }
}

impl KeySink for Preparing<'_> {
    fn tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> Result<()> {
        self.tensor_key = Some(self.tensor(&key));
        Ok(())
    }

    fn rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        let prepared = self.automorphism(&key);
        self.rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        self.boot()?;
        let prepared = self.automorphism(&key);
        self.boot_rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> Result<()> {
        self.boot()?;
        self.boot_tensor_key = Some(self.tensor(&key));
        Ok(())
    }

    fn encapsulation_keys(
        &mut self,
        dense_to_sparse: GLWESwitchingKey<Buf, Word>,
        sparse_to_dense: Option<GLWESwitchingKey<Buf, Word>>,
    ) -> Result<()> {
        let dense_to_sparse = self.switching_key(&dense_to_sparse)?;
        let sparse_to_dense = sparse_to_dense.map(|k| self.switching_key(&k)).transpose()?;
        self.encapsulation = Some((dense_to_sparse, sparse_to_dense));
        Ok(())
    }

    fn first_factor_key(&mut self, rotation: i64, key: GLWESwitchingKey<Buf, Word>) -> Result<()> {
        let prepared = self.switching_key(&key)?;
        self.first_factor_keys.insert(rotation, prepared);
        Ok(())
    }

    fn conversion_keys(&mut self, _: GLWECITraceKey<Buf, Word>, _: GLWECIEmbedKey<Buf, Word>) -> Result<()> {
        anyhow::bail!("conversion keys for a bootstrap on the standard ring")
    }
}

impl CkksEngine {
    /// Reads a key stream and prepares every key on the device as it arrives.
    pub fn prepare_keys_from<R: std::io::Read>(&self, r: &mut R) -> Result<PreparedKeys> {
        let mut guard = self.scratch.lock().unwrap();
        let mut sink = Preparing::new(self, &mut guard.1);
        let bootstrap = read_eval_keys_into(r, &self.ring, &mut sink)?;
        sink.finish(bootstrap)
    }

    /// Installs prepared keys, or drops the current ones with `None`.
    pub fn set_keys(&mut self, keys: Option<PreparedKeys>) {
        self.keys = keys;
    }
}

/// Rotation keys of the circuit, then those of the bootstrap.
struct Rotations<'a> {
    own: &'a std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    shared: Option<&'a std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<DevBuf, Dev>>>,
}

impl GetAutomorphismKey<Dev> for Rotations<'_> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, Dev>> {
        match self.shared {
            Some(shared) if !self.own.contains_key(&p) => shared.lookup_automorphism_key(p, k),
            _ => self.own.lookup_automorphism_key(p, k),
        }
    }
}
