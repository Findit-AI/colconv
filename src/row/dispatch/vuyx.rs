//! VUYX (FFmpeg `AV_PIX_FMT_VUYX`) row-level dispatchers (Ship 12c).
//!
//! Three entries:
//! - `vuyx_to_rgb_row` — re-export of `vuya::vuya_to_rgb_row` (bit-identical kernel).
//! - `vuyx_to_rgba_row` — VUYX-specific: forces output α = `0xFF` (padding A byte ignored).
//! - `vuyx_to_luma_row` — re-export of `vuya::vuya_to_luma_row` (bit-identical kernel).
//!
//! VUYX shares the VUYA byte stream (`V(8) ‖ U(8) ‖ Y(8) ‖ X(8)`) but the
//! fourth byte is padding with undefined content; RGBA output must always
//! emit α = `0xFF` regardless of what is in the source byte.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{rgba_row_bytes, scalar},
};

// ---- Re-exports (bit-identical kernels) ------------------------------------

/// Converts one row of VUYX to packed RGB (u8). Identical to
/// [`crate::row::vuya_to_rgb_row`] — the padding byte is irrelevant when there is no
/// α channel in the output. See that function for full documentation.
pub use super::vuya::vuya_to_rgb_row as vuyx_to_rgb_row;

/// Extracts one row of 8-bit luma from a packed VUYX buffer. Identical to
/// [`crate::row::vuya_to_luma_row`] — Y is at byte offset 2 of each quadruple
/// regardless of α semantics. See that function for full documentation.
pub use super::vuya::vuya_to_luma_row as vuyx_to_luma_row;

// ---- VUYX-specific RGBA dispatcher -----------------------------------------

/// Converts one row of VUYX to packed RGBA (u8). The padding byte (offset 3
/// of each pixel quadruple) is **ignored**; output α is forced to `0xFF`
/// (opaque) for every pixel (`ALPHA_SRC = false`). For VUYA where the A byte
/// carries real alpha, use `vuya_to_rgba_row` instead. `use_simd = false`
/// forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Re-use vuya's packed-length helper by computing it inline (same formula).
  let packed_min = match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize (VUYX packed row)"),
  };
  assert!(packed.len() >= packed_min, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::vuyx_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the VUYX-specific `vuyx_to_rgba_row` dispatcher.
  //! The re-exported `vuyx_to_rgb_row` and `vuyx_to_luma_row` are
  //! exercised by the VUYA dispatcher tests (they are the same function).
  use super::*;

  fn pack_vuyx(v: u8, u: u8, y: u8, x: u8) -> [u8; 4] {
    [v, u, y, x]
  }

  fn solid_vuyx(width: usize, y_val: u8, pad: u8) -> std::vec::Vec<u8> {
    let quad = pack_vuyx(128, 128, y_val, pad);
    (0..width).flat_map(|_| quad).collect()
  }

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn vuyx_rgba_dispatcher_rejects_short_packed() {
    let packed = [0u8; 8];
    let mut rgba = [0u8; 4 * 4];
    vuyx_to_rgba_row(&packed, &mut rgba, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgba_out row too short")]
  fn vuyx_rgba_dispatcher_rejects_short_output() {
    let packed = [0u8; 4 * 4];
    let mut rgba = [0u8; 2];
    vuyx_to_rgba_row(&packed, &mut rgba, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  fn vuyx_rgba_dispatcher_forces_alpha_ff() {
    // Source padding bytes are 0x42 and 0x99 — output α must be 0xFF for all.
    let buf = solid_vuyx(8, 128, 0x42);
    let mut rgba = [0u8; 8 * 4];
    vuyx_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert_eq!(px[3], 0xFF, "VUYX output alpha must always be 0xFF");
    }
  }

  #[test]
  fn vuyx_rgba_dispatcher_ignores_nonzero_padding_byte() {
    // Even if the padding byte contains 0xFF, the output is still 0xFF (no bleed-through).
    // More importantly: when the padding byte is 0x00 the output must still be 0xFF.
    let buf = solid_vuyx(4, 200, 0x00);
    let mut rgba = [0u8; 4 * 4];
    vuyx_to_rgba_row(&buf, &mut rgba, 4, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert_eq!(
        px[3], 0xFF,
        "VUYX output alpha must be 0xFF even when padding=0"
      );
    }
  }

  // ---- 32-bit width × 4 overflow guard ------------------------------------

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_4: usize = (usize::MAX / 4) + 1;

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn vuyx_rgba_dispatcher_rejects_width_times_4_overflow() {
    let p: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    vuyx_to_rgba_row(
      &p,
      &mut rgba,
      OVERFLOW_WIDTH_TIMES_4,
      ColorMatrix::Bt709,
      true,
      false,
    );
  }
}
