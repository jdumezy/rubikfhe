//! CKKS evaluation of the oblivious solver circuit on poulpy.

pub mod backend;
pub mod engine;
pub mod params;
pub mod system;
pub mod wire;

pub use engine::{CkksEngine, EvalKeys, Ring, SecretKey};
pub use params::Params;

#[cfg(test)]
mod tests;
