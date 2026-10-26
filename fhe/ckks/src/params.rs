//! CKKS parameters: the secure set with bootstrapping, and small insecure sets
//! for tests.

use std::sync::Arc;

use anyhow::{Result, ensure};
use poulpy_ckks::presets::bootstrapping::{BootstrappingPreset, n16_d35_k733_p20_s2c};
use poulpy_core::layouts::{GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank};

use crate::backend::Be;
use crate::functional::FunctionalPreset;

/// Failure probability `2^-128` per bootstrap when adapting the radix to the backend.
const BOOTSTRAP_FAILURE_BITS: usize = 128;

/// Bits a ciphertext's scale is raised by before it is bootstrapped. The
/// bootstrap's own error shrinks by about one bit per bit of input scale: its
/// worst slot keeps 11 bits at `2^35`, 16 at `2^40` and 21 at `2^45`.
const REFRESH_LIFT: usize = 7;

/// Circuit stages between refreshes. Errors double every stage and a refresh
/// squares them: an error `e` leaves as about `2.5 e²` on fresh bits, and
/// `10 e²` on the circuit's state, so a few stages are safe.
const STAGES_PER_REFRESH: usize = 6;

#[derive(Clone, Debug)]
pub struct Params {
    pub name: String,
    pub log_n: usize,
    pub base2k: usize,
    /// Scale of the ciphertexts, `2^log_delta`.
    pub log_delta: usize,
    /// Precision of the public plaintexts (masks, gather maps, token codes).
    pub pt_log_delta: usize,
    /// Width of fresh ciphertexts, the widest the circuit sees.
    pub fresh_k: usize,
    pub hamming_weight: usize,
    /// Functional bootstrapping: every refresh also rounds the values to 0/1.
    pub bootstrap: Option<Arc<FunctionalPreset>>,
}

/// The preset at the largest radix the backend affords for a `2^-128`
/// bootstrap failure probability, as poulpy's `preset_for_backend` derives it:
/// above the nominal radix the keys are re-derived, below it the high-modulus
/// keys take the widest digits that fit the modulus bounds. (EvalMod runs on
/// the standard ring, so the conjugate-invariant backend is `Be` itself.)
fn for_backend(preset: BootstrappingPreset) -> Result<BootstrappingPreset> {
    let max = preset.max_backend_base2k::<Be, Be>(BOOTSTRAP_FAILURE_BITS);
    if max > preset.base2k() {
        return preset.with_backend_base2k::<Be, Be>(BOOTSTRAP_FAILURE_BITS);
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
    /// 128-bit secure parameters: logN 16, refreshed by a functional bootstrap
    /// on the plan of the `n16_d35_k733_p20_s2c` preset, sized for
    /// [`STAGES_PER_REFRESH`] stages between refreshes.
    pub fn secure() -> Result<Params> {
        // Power-of-two rotation keys: fewer bootstrapping keys, which also
        // serve most of the circuit's rotations.
        let base = for_backend(n16_d35_k733_p20_s2c()?.with_power_of_two_keys()?)?;
        // Encoding a plaintext rounds its coefficients at `2^-pt_log_delta`,
        // about `√N · 2^-pt_log_delta` per slot: 24 bits keep the errors a
        // refresh receives near 1e-3, which it squares away.
        let (log_delta, pt_log_delta) = (base.log_delta(), 24);
        let stage_cost = 2 * pt_log_delta + log_delta;
        // The bootstrap reads its input at the lifted scale, plus one bit of message ratio.
        let reserve = base.plan().input_k(log_delta + REFRESH_LIFT + 1) - log_delta;
        // After a refresh, the selector is isolated again: one plaintext product.
        let refreshed_k = log_delta + reserve + STAGES_PER_REFRESH * stage_cost + pt_log_delta;
        let preset = FunctionalPreset::new(&base, log_delta + REFRESH_LIFT, refreshed_k + REFRESH_LIFT + 1)?;
        ensure!(preset.input_k - log_delta == reserve, "the functional plan reads {} bits", preset.input_k);
        let params = Params {
            name: format!("{} (power-of-two keys)", preset.name),
            log_n: preset.log_n,
            base2k: preset.base2k,
            log_delta,
            pt_log_delta,
            fresh_k: 0,
            hamming_weight: preset.dense_hamming_weight,
            bootstrap: Some(Arc::new(preset)),
        };
        // Fresh ciphertexts also pay for the first product, of the two halves of the input.
        Ok(Params { fresh_k: params.refreshed_k() + log_delta, ..params })
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
        self.bootstrap.as_ref().map_or(0, |p| p.input_k - self.log_delta)
    }

    /// Width of a refreshed ciphertext, back at the circuit's scale: the
    /// bootstrap's output, less the bits it is shifted down by.
    pub fn refreshed_k(&self) -> usize {
        self.bootstrap.as_ref().map_or(self.fresh_k, |p| p.output_k - (p.log_delta + 1 - self.log_delta))
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
