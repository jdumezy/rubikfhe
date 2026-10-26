//! Oblivious solver circuit: a stabilizer-chain correction circuit that maps
//! one-hot sticker colors to a fixed-length sequence of move tokens using only
//! additions, products and public constants.

pub mod chain;
pub mod packed;
pub mod perm;
pub mod reference;
pub mod tables;

#[cfg(test)]
mod tests;
