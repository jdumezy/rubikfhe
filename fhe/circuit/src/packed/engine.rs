//! The operations the packed circuit is made of, and an exact slot simulator.
//!
//! The evaluator only talks to an [`Engine`], so the CKKS evaluation and the
//! plaintext simulation run the very same sequence of operations.

use std::collections::BTreeMap;

use anyhow::Result;

/// Diagonals of a slot linear map, poulpy's convention:
/// `out[j] = Σ_i diag_i[j] · in[(j + i) mod slots]`, with `i` in `[0, slots)`.
pub type Diagonals = BTreeMap<usize, Vec<f64>>;

pub trait Engine: Sync {
    type Ct: Clone + Send + Sync;
    type Pt: Send + Sync;
    type Lt: Send + Sync;

    fn slots(&self) -> usize;

    /// Encodes public slot values (0/1 masks, small integers).
    fn encode(&self, values: &[f64]) -> Result<Self::Pt>;
    fn encode_lt(&self, diagonals: &Diagonals) -> Result<Self::Lt>;

    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;
    fn mul_pt(&self, a: &Self::Ct, b: &Self::Pt) -> Result<Self::Ct>;
    fn add_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> Result<()>;
    fn sub_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> Result<()>;
    fn apply_lt(&self, a: &Self::Ct, lt: &Self::Lt) -> Result<Self::Ct>;
    /// `Σ_{j < 2^log_count} rot(a, j · 2^log_batch)`.
    fn inner_sum(&self, a: &Self::Ct, log_batch: usize, log_count: usize) -> Result<Self::Ct>;
    /// `Σ_{j < 2^log_count} rot(a, -j · 2^log_batch)`.
    fn replicate(&self, a: &Self::Ct, log_batch: usize, log_count: usize) -> Result<Self::Ct>;

    /// Remaining capacity, in bits.
    fn log_budget(&self, a: &Self::Ct) -> usize;
    /// Capacity a stage costs: `2 · plaintext precision + scale`.
    fn stage_cost(&self) -> usize;
    /// Capacity a noise cleaning costs: two products and the selector mask.
    fn clean_cost(&self) -> usize;
    /// Capacity a product with a public plaintext costs.
    fn plaintext_cost(&self) -> usize;
    /// Capacity that must remain for a refresh to be possible.
    fn refresh_reserve(&self) -> usize;
    /// Whether a refresh also rounds the values back to 0/1, so the circuit
    /// needs no separate cleaning.
    fn refresh_cleans(&self) -> bool {
        false
    }
    fn refresh(&self, a: &Self::Ct) -> Result<Self::Ct>;
}

/// Exact evaluation on plaintext slot vectors, with unbounded capacity.
pub struct SlotSim {
    pub slots: usize,
}

fn rotate(a: &[f64], k: i64) -> Vec<f64> {
    let n = a.len() as i64;
    (0..n).map(|j| a[(j + k).rem_euclid(n) as usize]).collect()
}

impl Engine for SlotSim {
    type Ct = Vec<f64>;
    type Pt = Vec<f64>;
    type Lt = Diagonals;

    fn slots(&self) -> usize {
        self.slots
    }

    fn encode(&self, values: &[f64]) -> Result<Vec<f64>> {
        Ok(values.to_vec())
    }

    fn encode_lt(&self, diagonals: &Diagonals) -> Result<Diagonals> {
        Ok(diagonals.clone())
    }

    fn mul(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<Vec<f64>> {
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    fn mul_pt(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<Vec<f64>> {
        self.mul(a, b)
    }

    fn add_assign(&self, a: &mut Vec<f64>, b: &Vec<f64>) -> Result<()> {
        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        Ok(())
    }

    fn sub_assign(&self, a: &mut Vec<f64>, b: &Vec<f64>) -> Result<()> {
        a.iter_mut().zip(b).for_each(|(x, y)| *x -= y);
        Ok(())
    }

    fn apply_lt(&self, a: &Vec<f64>, lt: &Diagonals) -> Result<Vec<f64>> {
        let mut out = vec![0.0; a.len()];
        for (&i, diag) in lt {
            for (j, (o, r)) in out.iter_mut().zip(rotate(a, i as i64)).enumerate() {
                *o += diag[j % diag.len()] * r;
            }
        }
        Ok(out)
    }

    fn inner_sum(&self, a: &Vec<f64>, log_batch: usize, log_count: usize) -> Result<Vec<f64>> {
        let mut out = vec![0.0; a.len()];
        for j in 0..1i64 << log_count {
            self.add_assign(&mut out, &rotate(a, j << log_batch))?;
        }
        Ok(out)
    }

    fn replicate(&self, a: &Vec<f64>, log_batch: usize, log_count: usize) -> Result<Vec<f64>> {
        let mut out = vec![0.0; a.len()];
        for j in 0..1i64 << log_count {
            self.add_assign(&mut out, &rotate(a, -(j << log_batch)))?;
        }
        Ok(out)
    }

    fn log_budget(&self, _: &Vec<f64>) -> usize {
        usize::MAX
    }

    fn stage_cost(&self) -> usize {
        0
    }

    fn clean_cost(&self) -> usize {
        0
    }

    fn plaintext_cost(&self) -> usize {
        0
    }

    fn refresh_reserve(&self) -> usize {
        0
    }

    fn refresh(&self, a: &Vec<f64>) -> Result<Vec<f64>> {
        Ok(a.clone())
    }
}
