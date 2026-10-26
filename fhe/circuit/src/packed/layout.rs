//! Slot layout of the packed circuit.
//!
//! Every tracked cubie owns a block of 32 × 32 slots. Its 24-entry location
//! vector `x` is stored along one axis of the block (`v`) and replicated
//! along the other (`w`); which axis is which alternates from stage to stage,
//! so the circuit never needs a transpose:
//!
//! - `Cols`: `x[v]` sits in column `v`, copied down every row;
//! - `Rows`: `x[v]` sits in row `v`, copied along every column.
//!
//! A ciphertext holds `slots / 1024` blocks; cubies fill blocks in order,
//! spilling into further ciphertexts when one is too small (small test rings).

pub const LOG_SIDE: usize = 5;
pub const SIDE: usize = 1 << LOG_SIDE;
pub const LOG_BLOCK: usize = 2 * LOG_SIDE;
pub const BLOCK: usize = 1 << LOG_BLOCK;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Cols,
    Rows,
}

impl Orientation {
    /// Stage 0 starts in `Cols`; every stage flips the orientation.
    pub fn of_stage(stage: usize) -> Orientation {
        if stage.is_multiple_of(2) { Orientation::Cols } else { Orientation::Rows }
    }

    pub fn flip(self) -> Orientation {
        match self {
            Orientation::Cols => Orientation::Rows,
            Orientation::Rows => Orientation::Cols,
        }
    }

    /// Slot (within a block) of vector entry `v`, copy `w`.
    pub fn at(self, v: usize, w: usize) -> usize {
        match self {
            Orientation::Cols => w * SIDE + v,
            Orientation::Rows => v * SIDE + w,
        }
    }

    /// Slot distance between consecutive vector entries.
    pub fn step(self) -> usize {
        match self {
            Orientation::Cols => 1,
            Orientation::Rows => SIDE,
        }
    }

    /// `log_batch` of the inner sum / replicate that folds along the vector axis.
    pub fn log_batch(self) -> usize {
        match self {
            Orientation::Cols => 0,
            Orientation::Rows => LOG_SIDE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub slots: usize,
    pub blocks_per_ct: usize,
}

impl Layout {
    pub fn new(slots: usize) -> Layout {
        assert!(slots.is_power_of_two() && slots >= BLOCK, "need at least one 32x32 block of slots");
        Layout { slots, blocks_per_ct: slots / BLOCK }
    }

    /// Number of ciphertexts needed for `blocks` blocks.
    pub fn cts_for(&self, blocks: usize) -> usize {
        blocks.div_ceil(self.blocks_per_ct)
    }

    /// Ciphertext and first slot of block number `index`.
    pub fn place(&self, index: usize) -> (usize, usize) {
        (index / self.blocks_per_ct, (index % self.blocks_per_ct) * BLOCK)
    }

    pub fn log_blocks_per_ct(&self) -> usize {
        self.blocks_per_ct.trailing_zeros() as usize
    }
}
