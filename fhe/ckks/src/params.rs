//! CKKS parameters: the secure set with bootstrapping, and small insecure sets
//! for tests.

use anyhow::Result;
use poulpy_ckks::presets::bootstrapping::{BootstrappingPreset, n16_d35_k685_p19_s2c_ffr_ci};
use poulpy_core::layouts::{GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank};

use crate::backend::{Be, BeCi};

/// Failure probability `2^-128` per bootstrap when adapting the radix to the backend.
const BOOTSTRAP_FAILURE_BITS: usize = 128;

#[derive(Clone, Debug)]
pub struct Params {
    pub name: String,
    pub log_n: usize,
    pub base2k: usize,
    /// Scale of the ciphertexts, `2^log_delta`.
    pub log_delta: usize,
    /// Precision of the public plaintexts (masks, gather maps, token codes).
    pub pt_log_delta: usize,
    /// Width of fresh and bootstrapped ciphertexts.
    pub fresh_k: usize,
    pub hamming_weight: usize,
    pub bootstrap: Option<BootstrappingPreset>,
}

/// The preset at the largest radix the backend affords for a `2^-128`
/// bootstrap failure probability, as poulpy's `preset_for_backend` derives it:
/// above the nominal radix the keys are re-derived, below it the high-modulus
/// keys take the widest digits that fit the modulus bounds.
fn for_backend(preset: BootstrappingPreset) -> Result<BootstrappingPreset> {
    let max = preset.max_backend_base2k::<Be, BeCi>(BOOTSTRAP_FAILURE_BITS);
    if max > preset.base2k() {
        return preset.with_backend_base2k::<Be, BeCi>(BOOTSTRAP_FAILURE_BITS);
    }
    if max == preset.base2k() {
        return Ok(preset);
    }
    let preset = preset.with_base2k(max)?;
    for dsize in (2..=7).rev() {
        if let Ok(adapted) = preset.with_dsizes(dsize, 1) {
            return Ok(adapted);
        }
    }
    preset.with_dsizes(1, 1)
}

impl Params {
    /// 128-bit secure parameters: logN 16, refreshed with the
    /// `n16_d35_k685_p19_s2c_ffr_ci` bootstrapping preset.
    pub fn secure() -> Result<Params> {
        let preset = for_backend(n16_d35_k685_p19_s2c_ffr_ci()?)?;
        Ok(Params {
            name: preset.name().to_string(),
            log_n: preset.n().trailing_zeros() as usize,
            base2k: preset.base2k(),
            log_delta: preset.log_delta(),
            pt_log_delta: 20,
            fresh_k: preset.output_k(),
            hamming_weight: preset.dense_secret_hamming_weight(),
            bootstrap: Some(preset),
        })
    }

    /// Small, **insecure** parameters with enough modulus for the whole circuit
    /// without bootstrapping. For tests only.
    pub fn insecure_test(log_n: usize) -> Params {
        Params {
            name: format!("insecure-test-logn{log_n}"),
            log_n,
            base2k: 52,
            log_delta: 35,
            pt_log_delta: 20,
            fresh_k: 34 * 52,
            hamming_weight: 192,
            bootstrap: None,
        }
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    /// Complex slots of a ciphertext.
    pub fn slots(&self) -> usize {
        self.n() / 2
    }

    /// Capacity one circuit stage consumes: a gather, a product and a mask.
    pub fn stage_cost(&self) -> usize {
        2 * self.pt_log_delta + self.log_delta
    }

    /// Capacity a noise cleaning consumes: two products, then the selector mask.
    pub fn clean_cost(&self) -> usize {
        2 * self.log_delta + self.pt_log_delta
    }

    /// Capacity a ciphertext must keep to be bootstrapped (`input_k − log_delta`).
    pub fn refresh_reserve(&self) -> usize {
        self.bootstrap.as_ref().map_or(0, |p| p.input_k() - p.log_delta())
    }

    pub fn glwe_layout(&self) -> GLWELayout {
        GLWELayout { n: self.n().into(), base2k: self.base2k.into(), k: self.fresh_k.into(), rank: Rank(1) }
    }

    /// One gadget digit covering a fresh ciphertext.
    fn key_shape(&self) -> (usize, usize) {
        (self.fresh_k, self.fresh_k + self.log_n)
    }

    pub fn tensor_key_layout(&self) -> GLWETensorKeyLayout {
        let (k_digit, k_aux) = self.key_shape();
        GLWETensorKeyLayout {
            n: self.n().into(),
            base2k: self.base2k.into(),
            dnum: 1usize.into(),
            k_aux: k_aux.into(),
            rank: Rank(1),
            k_digit: k_digit.into(),
        }
    }

    pub fn rotation_key_layout(&self) -> GLWEAutomorphismKeyLayout {
        let (k_digit, k_aux) = self.key_shape();
        GLWEAutomorphismKeyLayout {
            n: self.n().into(),
            base2k: self.base2k.into(),
            dnum: 1usize.into(),
            k_aux: k_aux.into(),
            rank: Rank(1),
            k_digit: k_digit.into(),
        }
    }
}
