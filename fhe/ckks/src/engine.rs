//! CKKS implementation of the circuit [`Engine`], on poulpy.
//!
//! The circuit runs on a module of degree `N`. Bootstrapping, when enabled,
//! runs on the preset's modules of degree `2N` and `N` (conjugate-invariant).

use std::collections::BTreeMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use poulpy_ckks::api::{
    CKKSAllOpsTmpBytes, CKKSBootstrappingCIOps, CKKSCopyOps, CKKSInnerSumOps, CKKSLinearTransformationOps,
};
use poulpy_ckks::layouts::{
    BootstrappingContext, BootstrappingKeySet, BootstrappingKeysPrepared, CKKSCiphertextOwned, CKKSPlaintextOwned,
    ComplexDiagonals, EncapsulationKeys, LinearTransformationPrepared,
};
use poulpy_ckks::prelude::*;
use poulpy_ckks::reference::ckks_encode_linear_transformation_from_diagonals;
use poulpy_core::layouts::prepared::{
    GLWEAutomorphismKeyPrepared, GLWECIEmbedKeyPrepared, GLWECIKeyPreparedFactory, GLWECITraceKeyPrepared,
    GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWETensorKeyPrepared,
};
use poulpy_core::layouts::{
    BackendGLWEAutomorphismKey, BackendGLWEAutomorphismKeyPrepared, BackendGLWESecret, BackendGLWESecretPrepared,
    BackendGLWETensorKeyPrepared, Degree, Diagonals as CoreDiagonals, GLWEAutomorphismKeyPreparedFactory,
    GLWESecretSampling, GLWETensorKey, GLWETensorKeyPreparedFactory, LWEInfos, LinearTransformationLayout,
    LinearTransformationStrategy, ModuleCoreAlloc, optimal_bsgs_giant_step, prepared::GLWESecretPreparedFactory,
};
use poulpy_core::layouts::{GLWEAutomorphismKey, GLWECIEmbedKey, GLWECITraceKey, GLWESwitchingKey};
use poulpy_core::{GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk};
use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module, ScratchOwned};
use poulpy_hal::source::Source;
use rubikfhe_circuit::packed::layout::{LOG_BLOCK, LOG_SIDE};
use rubikfhe_circuit::packed::{Diagonals, Engine, PackedPlan};

use crate::backend::{Be, BeCi};
use crate::params::Params;
use crate::wire::{KeySink, read_eval_keys_into};

pub type Ciphertext = CKKSCiphertextOwned<Be>;
pub type Plaintext = CKKSPlaintextOwned<Be>;
type Word = <Be as Backend>::ZnxWord;
type Buf = <Be as Backend>::OwnedBuf;

/// Extra bits above the plaintext precision, for the integer part of codes (≤ 18).
const PT_INTEGER_BITS: usize = 8;

fn pt_meta(log_delta: usize) -> GLWEPlaintextMeta {
    GLWEPlaintextMeta { scale: Scale::Log(log_delta), slots: SlotsKind::Complex, log_sparsity: 0 }
}

/// Bootstrapping context, compiled from the preset (public).
pub struct Bootstrap {
    pub module: Module<Be>,
    pub ci: Module<BeCi>,
    pub context: BootstrappingContext<Be, f64>,
}

impl Bootstrap {
    fn new(params: &Params) -> Result<Option<Bootstrap>> {
        let Some(preset) = &params.bootstrap else { return Ok(None) };
        let n = preset.n() as u64;
        let (module, ci) = (Module::<Be>::new(2 * n), Module::<BeCi>::new(n));
        let (bytes, bytes_ci) = BootstrappingContext::<Be, f64>::compile_ci_unprepared_tmp_bytes(&module, &ci);
        let unprepared = BootstrappingContext::<Be, f64>::compile_ci_unprepared(
            &module,
            &ci,
            preset.base2k().into(),
            preset.plan(),
            &mut ScratchOwned::<Be>::alloc(bytes).borrow(),
            &mut ScratchOwned::<BeCi>::alloc(bytes_ci).borrow(),
        )?;
        let bytes = unprepared.prepare_tmp_bytes(&module);
        let context = unprepared.prepare(&module, &mut ScratchOwned::<Be>::alloc(bytes).borrow());
        Ok(Some(Bootstrap { module, ci, context }))
    }
}

/// Public evaluation keys, as the client generates them.
pub struct EvalKeys {
    pub tensor_key: GLWETensorKey<Buf, Word>,
    pub rotation_keys: BTreeMap<i64, BackendGLWEAutomorphismKey<Be>>,
    pub bootstrap: Option<BootstrappingKeySet<Buf, Word>>,
}

/// The circuit's ring, keyless: what both sides derive from the parameters.
pub struct Ring {
    pub params: Params,
    pub module: Module<Be>,
    pub bootstrap: Option<Bootstrap>,
}

impl Ring {
    pub fn new(params: Params) -> Result<Ring> {
        let module = Module::<Be>::new(params.n() as u64);
        let bootstrap = Bootstrap::new(&params)?;
        Ok(Ring { params, module, bootstrap })
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
            .ckks_all_ops_tmp_bytes(&ct, &p.tensor_key_layout(), &ct)
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

        let mut rotation_keys = BTreeMap::new();
        for &g in galois_elements {
            let mut key = self.module.glwe_automorphism_key_alloc_from_infos(&p.rotation_key_layout());
            self.module.glwe_automorphism_key_encrypt_sk(&mut key, g, &sk.raw, &mut xe, &mut xa, &mut scratch.borrow());
            rotation_keys.insert(g, key);
        }

        let bootstrap = match (&self.bootstrap, &p.bootstrap) {
            (Some(b), Some(preset)) => {
                let layout = preset.keys_layout();
                let (bytes, bytes_ci) = layout.generate_ci_tmp_bytes(&b.module, &b.ci);
                let mut xs = Source::new([seed, [0xc3; 32]].concat()[8..40].try_into().unwrap());
                Some(b.context.generate_keys_ci(
                    &b.module,
                    &b.ci,
                    &sk.raw,
                    layout,
                    &mut xs,
                    &mut xe,
                    &mut xa,
                    &mut ScratchOwned::<Be>::alloc(bytes).borrow(),
                    &mut ScratchOwned::<BeCi>::alloc(bytes_ci).borrow(),
                )?)
            }
            _ => None,
        };
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

/// Evaluation keys prepared for computation.
pub struct PreparedKeys {
    tensor_key: BackendGLWETensorKeyPrepared<Be>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Be>>,
    bootstrap: Option<BootstrappingKeysPrepared<Buf, Be, BeCi>>,
}

/// The CKKS [`Engine`]: a ring with prepared evaluation keys.
pub struct CkksEngine {
    pub ring: Ring,
    keys: Option<PreparedKeys>,
    /// Scratch arena and its size, grown to the largest linear transformation.
    scratch: Mutex<(usize, ScratchOwned<Be>)>,
    bootstrap_scratch: Option<Mutex<(ScratchOwned<Be>, ScratchOwned<BeCi>)>>,
}

impl CkksEngine {
    pub fn new(ring: Ring) -> CkksEngine {
        let scratch = Mutex::new((0, ring.scratch()));
        let bootstrap_scratch = ring.bootstrap.as_ref().zip(ring.params.bootstrap.as_ref()).map(|(b, preset)| {
            let (bytes, bytes_ci) = b.module.ckks_bootstrap_ci_tmp_bytes(
                &b.ci,
                &preset.bootstrap_layout(),
                &preset.input_layout(),
                &b.context,
                preset.keys_layout(),
            );
            Mutex::new((ScratchOwned::<Be>::alloc(bytes), ScratchOwned::<BeCi>::alloc(bytes_ci)))
        });
        CkksEngine { ring, keys: None, scratch, bootstrap_scratch }
    }

    pub fn has_keys(&self) -> bool {
        self.keys.is_some()
    }

    /// Grows the scratch arena to at least `bytes` on top of the base arena.
    fn reserve_scratch(&self, bytes: usize) {
        let mut guard = self.scratch.lock().unwrap();
        if guard.0 < bytes {
            *guard = (bytes, ScratchOwned::<Be>::alloc(bytes.max(self.ring.scratch_bytes())));
        }
    }

    fn keys(&self) -> Result<&PreparedKeys> {
        self.keys.as_ref().context("no evaluation keys installed")
    }

    fn alloc_like(&self, k: usize) -> Ciphertext {
        let mut layout = self.ring.params.glwe_layout();
        layout.k = k.into();
        self.ring.module.ckks_ciphertext_alloc_from_glwe_infos(&layout)
    }
}

impl Engine for CkksEngine {
    type Ct = Ciphertext;
    type Pt = Plaintext;
    type Lt = LinearTransformationPrepared<Be>;

    fn slots(&self) -> usize {
        self.ring.params.slots()
    }

    fn encode(&self, values: &[f64]) -> Result<Plaintext> {
        let p = &self.ring.params;
        let mut pt =
            self.ring.module.ckks_pt_vec_alloc(p.slots(), p.base2k.into(), (p.pt_log_delta + PT_INTEGER_BITS).into());
        pt.set_meta(pt_meta(p.pt_log_delta));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_encode_reim_into(&mut pt, values, &vec![0.0; values.len()], &mut scratch.borrow())?;
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
        let (prepared, plan) = {
            let mut guard = self.scratch.lock().unwrap();
            let scratch = &mut guard.1;
            let lt = ckks_encode_linear_transformation_from_diagonals(
                &self.ring.module,
                p.base2k.into(),
                CoeffsMeta::from_delta_budget(p.pt_log_delta, PT_INTEGER_BITS),
                &ComplexDiagonals::new(re, im),
                Ring::lt_strategy(&indexes, slots),
                false,
                &mut scratch.borrow(),
            )?;
            let first = lt.first_diagonal_plaintext().context("empty gather map")?;
            let plan = lt.index();
            let mut prepared =
                LinearTransformationPrepared::<Be>::alloc_prepared_from_index(&self.ring.module, &plan, first);
            self.ring.module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
            (prepared, plan)
        };
        let ct = self.ring.module.ckks_ciphertext_alloc_from_glwe_infos(&p.glwe_layout());
        let atk = p.rotation_key_layout();
        let module = &self.ring.module;
        self.reserve_scratch(
            module
                .ckks_eval_linear_transformation_tmp_bytes(&ct, &plan, &atk)
                .max(module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &atk))
                + self.ring.scratch_bytes(),
        );
        Ok(prepared)
    }

    fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        let mut out = self.alloc_like(a.k().as_usize().min(b.k().as_usize()));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_mul_into(&mut out, a, b, &self.keys()?.tensor_key, &mut scratch.borrow())?;
        Ok(out)
    }

    fn mul_pt(&self, a: &Ciphertext, b: &Plaintext) -> Result<Ciphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_mul_pt_vec_into(&mut out, a, b, &mut scratch.borrow())?;
        Ok(out)
    }

    fn add_assign(&self, a: &mut Ciphertext, b: &Ciphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_add_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn sub_assign(&self, a: &mut Ciphertext, b: &Ciphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_sub_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn apply_lt(&self, a: &Ciphertext, lt: &Self::Lt) -> Result<Ciphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_eval_linear_transformation_self_into(
            &mut out,
            a,
            lt,
            &self.keys()?.rotation_keys,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn inner_sum(&self, a: &Ciphertext, log_batch: usize, log_count: usize) -> Result<Ciphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_inner_sum(
            &mut out,
            a,
            log_batch,
            log_count,
            false,
            &self.keys()?.rotation_keys,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn replicate(&self, a: &Ciphertext, log_batch: usize, log_count: usize) -> Result<Ciphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.ring.module.ckks_replicate(
            &mut out,
            a,
            log_batch,
            log_count,
            &self.keys()?.rotation_keys,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn log_budget(&self, a: &Ciphertext) -> usize {
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

    fn refresh(&self, a: &Ciphertext) -> Result<Ciphertext> {
        let (Some(b), Some(preset), Some(keys)) =
            (&self.ring.bootstrap, &self.ring.params.bootstrap, &self.keys()?.bootstrap)
        else {
            anyhow::bail!("out of capacity and these parameters do not bootstrap");
        };
        let scratch = self.bootstrap_scratch.as_ref().unwrap();
        let (scratch, scratch_ci) = &mut *scratch.lock().unwrap();
        let mut input = b.module.ckks_ciphertext_alloc_from_glwe_infos(&preset.input_layout());
        b.module.ckks_copy(&mut input, a, &mut scratch.borrow())?;
        let mut output = b.module.ckks_ciphertext_alloc_from_glwe_infos(&preset.bootstrap_layout());
        output.set_k(preset.bootstrap_k().into());
        b.module.ckks_bootstrap_ci_into(
            &b.ci,
            &mut output,
            &input,
            &b.context,
            keys,
            &mut scratch.borrow(),
            &mut scratch_ci.borrow(),
        )?;
        Ok(output)
    }
}

/// Prepares the keys of a stream one by one, as they arrive, so that raw keys
/// never pile up next to their prepared form.
struct Preparing<'a> {
    ring: &'a Ring,
    scratch: ScratchOwned<Be>,
    boot_scratch: Option<(ScratchOwned<Be>, ScratchOwned<BeCi>)>,
    tensor_key: Option<BackendGLWETensorKeyPrepared<Be>>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Be>>,
    boot_rotation_keys: std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<Buf, Be>>,
    boot_tensor_key: Option<GLWETensorKeyPrepared<Buf, BeCi>>,
    encapsulation: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<Buf, Be>>>,
    first_factor_keys: std::collections::HashMap<i64, GLWESwitchingKeyPrepared<Buf, Be>>,
    conversion: Option<(GLWECITraceKeyPrepared<Buf, Be>, GLWECIEmbedKeyPrepared<Buf, Be>)>,
}

impl<'a> Preparing<'a> {
    fn new(ring: &'a Ring) -> Preparing<'a> {
        let boot_scratch = ring.bootstrap.as_ref().zip(ring.params.bootstrap.as_ref()).map(|(b, preset)| {
            let (bytes, bytes_ci) = preset.keys_layout().prepare_ci_tmp_bytes(&b.module, &b.ci);
            (ScratchOwned::<Be>::alloc(bytes), ScratchOwned::<BeCi>::alloc(bytes_ci))
        });
        Preparing {
            ring,
            scratch: ring.scratch(),
            boot_scratch,
            tensor_key: None,
            rotation_keys: Default::default(),
            boot_rotation_keys: Default::default(),
            boot_tensor_key: None,
            encapsulation: None,
            first_factor_keys: Default::default(),
            conversion: None,
        }
    }

    fn boot(&mut self) -> Result<(&'a Bootstrap, &mut ScratchOwned<Be>, &mut ScratchOwned<BeCi>)> {
        let b = self.ring.bootstrap.as_ref().context("bootstrapping keys for parameters without bootstrapping")?;
        let (scratch, scratch_ci) = self.boot_scratch.as_mut().context("no bootstrapping scratch")?;
        Ok((b, scratch, scratch_ci))
    }

    fn switching_key(&mut self, key: &GLWESwitchingKey<Buf, Word>) -> Result<GLWESwitchingKeyPrepared<Buf, Be>> {
        let (b, scratch, _) = self.boot()?;
        let mut prepared = b.module.glwe_switching_key_prepared_alloc_from_infos(key);
        b.module.glwe_switching_key_prepare(&mut prepared, key, &mut scratch.borrow());
        Ok(prepared)
    }

    fn finish(self, bootstrap: bool) -> Result<PreparedKeys> {
        let bootstrap = if bootstrap {
            Some(BootstrappingKeysPrepared {
                rotation_keys: self.boot_rotation_keys,
                tensor_key: self.boot_tensor_key.context("missing bootstrapping tensor key")?,
                encapsulation_keys: self.encapsulation,
                first_factor_keys: self.first_factor_keys,
                conversion_keys: Some(self.conversion.context("missing conversion keys")?),
            })
        } else {
            None
        };
        if bootstrap.is_some() != self.ring.bootstrap.is_some() {
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
        let module = &self.ring.module;
        let mut prepared = module.alloc_tensor_key_prepared_from_infos(&self.ring.params.tensor_key_layout());
        module.prepare_tensor_key(&mut prepared, &key, &mut self.scratch.borrow());
        self.tensor_key = Some(prepared);
        Ok(())
    }

    fn rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        let module = &self.ring.module;
        let mut prepared =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&self.ring.params.rotation_key_layout());
        module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut self.scratch.borrow());
        self.rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        let (b, scratch, _) = self.boot()?;
        let mut prepared = b.module.glwe_automorphism_key_prepared_alloc_from_infos(&key);
        b.module.glwe_automorphism_key_prepare(&mut prepared, &key, &mut scratch.borrow());
        self.boot_rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> Result<()> {
        let (b, _, scratch_ci) = self.boot()?;
        let mut prepared = b.ci.alloc_tensor_key_prepared_from_infos(&key);
        b.ci.prepare_tensor_key(&mut prepared, &key, &mut scratch_ci.borrow());
        self.boot_tensor_key = Some(prepared);
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

    fn conversion_keys(&mut self, trace: GLWECITraceKey<Buf, Word>, embed: GLWECIEmbedKey<Buf, Word>) -> Result<()> {
        let (b, scratch, _) = self.boot()?;
        let mut trace_key = b.module.glwe_ci_trace_key_prepared_alloc_from_infos(&trace);
        b.module.glwe_ci_key_prepare(&mut trace_key, &trace, &mut scratch.borrow());
        let mut embed_key = b.module.glwe_ci_embed_key_prepared_alloc_from_infos(&embed);
        b.module.glwe_ci_key_prepare(&mut embed_key, &embed, &mut scratch.borrow());
        self.conversion = Some((trace_key, embed_key));
        Ok(())
    }
}

impl CkksEngine {
    /// Reads a key stream and prepares every key as it arrives.
    pub fn prepare_keys_from<R: std::io::Read>(&self, r: &mut R) -> Result<PreparedKeys> {
        let mut sink = Preparing::new(&self.ring);
        let bootstrap = read_eval_keys_into(r, &self.ring, &mut sink)?;
        sink.finish(bootstrap)
    }

    /// Installs prepared keys, or drops the current ones with `None`.
    pub fn set_keys(&mut self, keys: Option<PreparedKeys>) {
        self.keys = keys;
    }
}
