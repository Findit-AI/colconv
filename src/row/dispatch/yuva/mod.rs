//! YUVA dispatchers — the Yuva444p family (`Yuva444p9` / `p10` /
//! `p12` / `p14` / `p16`) and the Yuva420p family (`Yuva420p` / `p9`
//! / `p10` / `p12` / `p16`), for both 8-bit RGBA and native-depth
//! `u16` RGBA outputs. The 12-bit and 14-bit dispatchers ride the
//! same BITS-generic kernel templates (`yuv_444p_n_*` / `yuv_420p_n_*`)
//! that already cover the lower depths, so per-arch SIMD comes free.
//! 16-bit goes through the dedicated i64 4:4:4 / 4:2:0 kernel
//! family. Split per-sub-family so neither sub-file exceeds ~1.5 KLoC.
//!
//! The Yuva422p family does not have its own row dispatcher: per-row
//! the chroma layout is identical to 4:2:0 (half-width U / V), so
//! `MixedSinker<Yuva422p*>` delegates row-level work to the
//! `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers (including the
//! new `yuva420p12_*` pair, which is reused by `Yuva422p12`). The
//! 4:2:0 vs 4:2:2 difference is purely in the vertical walker
//! (chroma row index `r / 2` vs `r`) and is handled in the walker /
//! sinker layer.

mod sub_4_2_0;
mod sub_4_4_4;

pub use sub_4_2_0::*;
pub use sub_4_4_4::*;
