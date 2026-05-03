use super::*;
use crate::{ColorMatrix, frame::*, raw::*, yuv::*};

// Per-format-family submodules. Each houses tests + format-local
// helpers (`solid_*_frame` builders); cross-cutting helpers
// (`pseudo_random_u8`, `pseudo_random_u16_low_n_bits`) live at
// module scope below and are re-exported as `pub(super)` so the
// submodules can pull them via `use super::*;`.
mod ayuv64;
mod bayer;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod packed_yuv_8bit;
mod planar_other_8bit_9bit;
mod semi_planar_8bit;
mod subsampled_4_2_0_high_bit;
mod subsampled_high_bit_pn;
mod v210;
mod v30x;
mod v410;
mod vuya;
mod vuyx;
mod xv36;
mod y210;
mod y212;
mod y216;
mod yuv420p_8bit;
mod yuva;

pub(super) fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
  let mask = ((1u32 << bits) - 1) as u16;
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) as u16) & mask;
  }
}

pub(super) fn pseudo_random_u8(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}
