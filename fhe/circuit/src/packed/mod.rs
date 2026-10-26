//! The circuit packed into SIMD slots, as evaluated under CKKS.

pub mod engine;
pub mod eval;
pub mod layout;
pub mod plan;

pub use engine::{Diagonals, Engine, SlotSim};
pub use eval::{Event, Outputs, Prepared, Report, evaluate, evaluate_inspected};
pub use layout::Layout;
pub use plan::PackedPlan;
