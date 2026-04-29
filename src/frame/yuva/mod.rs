//! Validated YUVA source frames split by sub-family.
//!
//! Re-exports every item at the original `crate::frame::*` path so
//! existing call sites resolve unchanged. Keeping the per-sub-family
//! files siblings (rather than forcing each into a deeper path)
//! preserves the public surface while putting each ~800-line block
//! in its own file.

mod sub_4_2_0;
mod sub_4_2_2;
mod sub_4_4_4;

pub use sub_4_2_0::*;
pub use sub_4_2_2::*;
pub use sub_4_4_4::*;
