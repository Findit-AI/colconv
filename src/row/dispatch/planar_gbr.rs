//! Runtime SIMD dispatchers for planar GBR sources (Tier 10).
//!
//! Three kernels:
//! - [`gbr_to_rgb_row`] — interleave G/B/R → packed `R, G, B`.
//! - [`gbra_to_rgba_row`] — interleave G/B/R/A → packed `R, G, B, A`
//!   (real source α).
//! - [`gbr_to_rgba_opaque_row`] — interleave G/B/R → packed
//!   `R, G, B, 0xFF` (`Gbrp` standalone with_rgba path).
//!
//! Each function follows the same `cfg_select!` pattern used across
//! `dispatch::*`: platform arm at compile time, best available backend
//! at runtime via the `*_available()` helpers.

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
use crate::row::{rgb_row_bytes, rgba_row_bytes, scalar};

/// Interleaves three planar G/B/R rows into packed `R, G, B` bytes.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgb_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbr_to_rgb_row(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgb_row(g, b, r, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgb_row(g, b, r, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgb_row(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgb_row(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgb_row(g, b, r, rgb_out, width);
}

/// Interleaves four planar G/B/R/A rows into packed `R, G, B, A` bytes
/// (real source α).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbra_to_rgba_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbra_to_rgba_row(g, b, r, a, rgba_out, width);
}

/// Interleaves three planar G/B/R rows into packed `R, G, B, A` bytes
/// with constant α = `0xFF` (used by `Gbrp` for the standalone
/// `with_rgba` path).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgba_opaque_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgba_opaque_row(g, b, r, rgba_out, width);
}
