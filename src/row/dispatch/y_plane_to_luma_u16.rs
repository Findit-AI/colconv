//! Runtime SIMD dispatcher for `y_plane_to_luma_u16_row`. Mirrors the
//! pattern used by other crate-internal dispatchers (e.g.
//! `dispatch::alpha_extract`).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::y_plane_to_luma_u16 as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

/// Runtime-dispatched zero-extension of a u8 Y plane to u16.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false` (`MixedSinker::with_simd(false)`), the
/// SIMD cascade is bypassed and scalar runs directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize, use_simd: bool) {
  // Release-mode safety boundary before any `unsafe` SIMD dispatch.
  // Per-arch helpers only `debug_assert!` these bounds; without these
  // unconditional checks, a short caller slice would silently turn
  // into out-of-bounds reads/writes in release builds. Mirrors the
  // guard pattern in `dispatch::packed_yuv422` and `dispatch::vuya`.
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");

  if !use_simd {
    return scalar::y_plane_to_luma_u16_row(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime.
        unsafe {
          arch::neon::y_plane_to_luma_u16_row(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::y_plane_to_luma_u16_row(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::y_plane_to_luma_u16_row(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::y_plane_to_luma_u16_row(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::y_plane_to_luma_u16_row(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::y_plane_to_luma_u16_row(plane, out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  #[should_panic(expected = "plane too short")]
  fn dispatcher_panics_on_short_plane() {
    let plane = std::vec![0u8; 4];
    let mut out = std::vec![0u16; 8];
    y_plane_to_luma_u16_row(&plane, &mut out, 8, true);
  }

  #[test]
  #[should_panic(expected = "out too short")]
  fn dispatcher_panics_on_short_out() {
    let plane = std::vec![0u8; 8];
    let mut out = std::vec![0u16; 4];
    y_plane_to_luma_u16_row(&plane, &mut out, 8, true);
  }

  #[test]
  #[should_panic(expected = "plane too short")]
  fn dispatcher_panics_on_short_plane_use_simd_false() {
    // Same guard fires before the use_simd shortcut.
    let plane = std::vec![0u8; 4];
    let mut out = std::vec![0u16; 8];
    y_plane_to_luma_u16_row(&plane, &mut out, 8, false);
  }
}
