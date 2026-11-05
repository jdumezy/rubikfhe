//! A functional bootstrapping parameter set: S2C-first bootstrapping that
//! evaluates the binary LUT `0 → 0, 1 → 1` on the way, so every refresh also
//! rounds the circuit's one-hot values back to 0/1.
//!
//! poulpy ships no functional preset. This one takes the plan of an S2C-first
//! preset (its DFTs and radix), adds the LUT, and derives the widths and key
//! layouts the way poulpy's preset builder does, checking every key against
//! the 128-bit modulus bounds of its secret.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_ckks::CoeffsMeta;
use poulpy_ckks::layouts::{
    BootstrappingKeysLayout, BootstrappingPlan, CKKSPlaintextOwned, EncapsulationKeysLayout, EncodedLut,
};
use poulpy_ckks::polynomial::SplitStrategy;
use poulpy_ckks::presets::bootstrapping::BootstrappingPreset;
use poulpy_core::layouts::{Base2K, Degree, GGLWEInfos, GGLWELayout, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::{HostBytesBackend, Module};

/// Largest modulus, in bits, for 128-bit security at `N = 2^16`, by secret
/// Hamming weight (sparse-key-estimate table).
const DENSE_HAMMING_WEIGHT: usize = 1024;
const DENSE_MAX_MODULUS: usize = 1707;
const SPARSE_HAMMING_WEIGHT: usize = 32;
const SPARSE_MAX_MODULUS: usize = 345;

/// Bits a key keeps between its modulus and the input plus one digit (as in
/// poulpy's presets).
const KEY_NOISE_MARGIN: usize = 2;

/// The binary LUT's cosine series, as in poulpy's functional tests. Its own
/// error is far below the bootstrap's, which the input scale sets.
const LUT_DEGREE: usize = 31;
const LUT_K_INTERVAL: usize = 16;
const LUT_LOG_INTERVAL_REDUCTION: usize = 3;
const LUT_LOG_DELTA: usize = 55;

pub struct FunctionalPreset {
    pub name: String,
    pub log_n: usize,
    pub base2k: usize,
    /// Scale of the ciphertexts entering the bootstrap; they leave one bit higher.
    pub log_delta: usize,
    pub plan: BootstrappingPlan,
    pub lut: EncodedLut<CKKSPlaintextOwned<HostBytesBackend>>,
    pub keys_layout: BootstrappingKeysLayout,
    /// Width of a ciphertext entering the bootstrap.
    pub input_k: usize,
    /// Width of a ciphertext leaving it.
    pub output_k: usize,
    /// Working width of the bootstrap, in whole limbs.
    pub bootstrap_k: usize,
    pub dense_hamming_weight: usize,
}

impl std::fmt::Debug for FunctionalPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionalPreset")
            .field("name", &self.name)
            .field("input_k", &self.input_k)
            .field("output_k", &self.output_k)
            .field("bootstrap_k", &self.bootstrap_k)
            .finish_non_exhaustive()
    }
}

impl FunctionalPreset {
    /// Bootstraps inputs of scale `2^log_delta` to outputs of at least
    /// `min_output_k` bits, with the plan and radix of `base`.
    pub fn new(base: &BootstrappingPreset, log_delta: usize, min_output_k: usize) -> Result<FunctionalPreset> {
        let (log_n, base2k) = (base.n().trailing_zeros() as usize, base.base2k());
        ensure!(log_n == 16, "the modulus bounds are those of N = 2^16");
        ensure!(
            base.dense_secret_hamming_weight() == DENSE_HAMMING_WEIGHT
                && base.plan().sparse_secret_hamming_weight() == Some(SPARSE_HAMMING_WEIGHT),
            "the modulus bounds are those of Hamming weights {DENSE_HAMMING_WEIGHT} and {SPARSE_HAMMING_WEIGHT}"
        );
        let host = Module::<HostBytesBackend>::new(1 << log_n);
        let lut = EncodedLut::binary(
            &host,
            0.0f64,
            1.0,
            LUT_DEGREE,
            LUT_K_INTERVAL,
            LUT_LOG_INTERVAL_REDUCTION,
            base2k.into(),
            CoeffsMeta::from_delta_budget(LUT_LOG_DELTA, base2k),
            SplitStrategy::MinDepth,
        )?;
        let plan = base.plan().clone().with_functional_bootstrap(&lut)?;

        let log_modulus = log_delta + lut.log_msg_ratio();
        let input_k = plan.input_k(log_modulus);
        let overhead = plan.functional_bootstrap_k(0, log_delta, &lut)?;
        let bootstrap_k = (min_output_k + overhead).next_multiple_of(base2k);
        let output_k = bootstrap_k - overhead;
        ensure!(output_k > input_k, "output width {output_k} must exceed the input width {input_k}");
        ensure!(
            bootstrap_k <= DENSE_MAX_MODULUS,
            "bootstrap width {bootstrap_k} exceeds the dense-secret bound {DENSE_MAX_MODULUS}"
        );

        let keys = Shape { log_n, base2k };
        // A key only used on narrower ciphertexts takes the digit of fewest rows those admit.
        let widths = plan.key_widths(log_n, bootstrap_k, input_k, log_delta);
        let shape = |name, k| keys.key_layout(name, k, keys.widest_k_digit(k, DENSE_MAX_MODULUS), DENSE_MAX_MODULUS);
        let rotation_keys = widths
            .rotation_keys
            .iter()
            .map(|(&p, &k)| Ok((p, shape("rotation", k)?.into())))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let keys_layout = BootstrappingKeysLayout {
            rotation_keys,
            tensor_key: shape("tensor", widths.tensor_key)?.into(),
            encapsulation: Some(EncapsulationKeysLayout {
                dense_to_sparse: keys.key_layout("dense-to-sparse", log_modulus, base2k, SPARSE_MAX_MODULUS)?.into(),
                sparse_to_dense: Some(
                    keys.sparse_to_dense_layout(plan.raised_window(log_modulus), bootstrap_k, DENSE_MAX_MODULUS)?
                        .into(),
                ),
            }),
            first_factor_key: None,
            conversion_keys: None,
        };
        let preset = FunctionalPreset {
            name: format!("n16_d{log_delta}_k{bootstrap_k}_s2c_binary_lut"),
            log_n,
            base2k,
            log_delta,
            plan,
            lut,
            keys_layout,
            input_k,
            output_k,
            bootstrap_k,
            dense_hamming_weight: DENSE_HAMMING_WEIGHT,
        };
        preset.validate()?;
        Ok(preset)
    }

    /// Every key within the modulus bound of the secret it is encrypted under.
    fn validate(&self) -> Result<()> {
        let keys = &self.keys_layout;
        for key in keys.rotation_keys.values() {
            validate_key("rotation", key, DENSE_MAX_MODULUS)?;
        }
        validate_key("tensor", &keys.tensor_key, DENSE_MAX_MODULUS)?;
        let enc = keys.encapsulation.as_ref().context("missing encapsulation keys")?;
        validate_key("dense-to-sparse", &enc.dense_to_sparse, SPARSE_MAX_MODULUS)?;
        if let Some(key) = &enc.sparse_to_dense {
            validate_key("sparse-to-dense", key, DENSE_MAX_MODULUS)?;
        }
        Ok(())
    }
}

fn validate_key<K: GGLWEInfos + LWEInfos>(name: &str, key: &K, limit: usize) -> Result<()> {
    let gadget = key.gglwe_layout().gadget_k().as_usize();
    ensure!(gadget <= limit, "{name} key gadget modulus {gadget} exceeds {limit}");
    ensure!(key.k_aux().as_usize() <= limit, "{name} key auxiliary modulus {} exceeds {limit}", key.k_aux());
    ensure!(key.k().as_usize() <= limit, "{name} key modulus {} exceeds {limit}", key.k());
    Ok(())
}

/// The key-shape rules of poulpy's preset builder.
struct Shape {
    log_n: usize,
    base2k: usize,
}

impl Shape {
    /// Guard behind the digits: a digit and `log_n` bits, or what the bound leaves.
    fn guard(&self, k_digit: usize, gadget: usize, limit: usize) -> usize {
        (k_digit + self.log_n).min(limit.saturating_sub(gadget))
    }

    fn fits(&self, input_k: usize, k_digit: usize, limit: usize) -> bool {
        let gadget = input_k.div_ceil(k_digit) * k_digit;
        let guard = self.guard(k_digit, gadget, limit);
        guard >= k_digit && gadget + guard >= input_k + k_digit + KEY_NOISE_MARGIN
    }

    /// Whole-limb digit of the key of fewest rows that fits, narrowest at that count.
    fn widest_k_digit(&self, input_k: usize, limit: usize) -> usize {
        (1..=input_k.max(1).div_ceil(self.base2k))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| self.fits(input_k, k_digit, limit))
            .min_by_key(|&k_digit| (input_k.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k)
    }

    fn key_layout(&self, name: &str, input_k: usize, k_digit: usize, limit: usize) -> Result<GGLWELayout> {
        let dnum = GGLWELayout::dnum_for_input(TorusPrecision(input_k as u32), TorusPrecision(k_digit as u32));
        let gadget = dnum.as_usize() * k_digit;
        ensure!(
            self.fits(input_k, k_digit, limit),
            "{name} key: {k_digit}-bit digits on a {input_k}-bit input do not fit {limit} bits"
        );
        Ok(GGLWELayout {
            n: Degree(1 << self.log_n),
            base2k: Base2K(self.base2k as u32),
            dnum,
            k_aux: TorusPrecision(self.guard(k_digit, gadget, limit) as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(k_digit as u32),
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    }

    /// The sparse-to-dense key switches the raise: its rows sit under the raised
    /// ciphertext, `window` bits at the bottom of the working width, and end at it.
    fn sparse_to_dense_layout(&self, window: usize, bootstrap_k: usize, limit: usize) -> Result<GGLWELayout> {
        let shape = |k_digit: usize| {
            let gadget = window.div_ceil(k_digit) * k_digit;
            bootstrap_k.checked_sub(gadget).map(|k_offset| (gadget, k_offset))
        };
        let fits = |k_digit: usize| {
            shape(k_digit).is_some_and(|(gadget, k_offset)| self.fits(gadget, k_digit, limit.saturating_sub(k_offset)))
        };
        let k_digit = (1..=window.div_ceil(self.base2k))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| fits(k_digit))
            .min_by_key(|&k_digit| (window.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k);
        let (gadget, k_offset) = shape(k_digit).context("sparse-to-dense key: digits overshoot the working width")?;
        Ok(GGLWELayout {
            k_offset: TorusPrecision(k_offset as u32),
            ..self.key_layout("sparse-to-dense", gadget, k_digit, limit.saturating_sub(k_offset))?
        })
    }
}
