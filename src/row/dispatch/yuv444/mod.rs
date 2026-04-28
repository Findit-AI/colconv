//! YUV 4:4:4 dispatchers, split per source format for readability.
//!
//! - `yuv_444` — 8-bit YUV 4:4:4 → RGB / RGBA.
//! - `yuv444p9` / `yuv444p10` / `yuv444p12` / `yuv444p14` —
//!   high-bit planar (4 variants per format). RGB / RGB-u16 paths
//!   are thin wrappers over the BITS-generic helpers below; the
//!   RGBA / RGBA-u16 paths are full dispatchers.
//! - `yuv444p16` — 16-bit planar with its own dedicated dispatchers
//!   (the BITS-generic template is pinned to {9, 10, 12, 14}).
//!
//! `yuv_444p_n_to_rgb_row<BITS>` / `yuv_444p_n_to_rgb_u16_row<BITS>`
//! are the BITS-generic dispatchers shared by the 9 / 10 / 12 / 14-bit
//! RGB wrappers above. They stay `pub(crate)` and live here at the
//! `yuv444` module root so siblings can reach them via `super::*`.

use crate::row::scalar;
use crate::row::{arch, rgb_row_bytes, rgb_row_elems};
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
use crate::ColorMatrix;


/// YUV 4:4:4 planar 10/12/14-bit → **u8** RGB dispatcher. Const
/// generic over `BITS ∈ {10, 12, 14}`. Dispatches to the best
/// available backend for the current target (NEON / SSE4.1 / AVX2 /
/// AVX-512 / wasm simd128), falling back to scalar when no SIMD
/// backend is available or `use_simd` is false.
///
/// Crate-private — external callers use the concrete
/// [`yuv444p10_to_rgb_row`] / [`yuv444p12_to_rgb_row`] /
/// [`yuv444p14_to_rgb_row`] wrappers, which pin `BITS` to a
/// supported value. This avoids the 16-bit footgun (`(1 << 16) - 1`
/// truncates to `-1` when cast to `i16` in the SIMD clamp), and
/// matches the [`yuv420p10_to_rgb_row`] family's convention of
/// keeping the `<BITS>` generic internal.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar 10/12/14-bit → **native-depth u16** RGB dispatcher.
/// Const generic over `BITS ∈ {10, 12, 14}`. Low-bit-packed output.
/// Dispatches to the best available backend (NEON / SSE4.1 / AVX2 /
/// AVX-512 / wasm simd128), falling back to scalar when no SIMD
/// backend is available or `use_simd` is false.
///
/// Crate-private — see the note on [`yuv_444p_n_to_rgb_row`]. The
/// 16-bit path is [`yuv444p16_to_rgb_u16_row`], which uses a
/// dedicated i64-chroma kernel family.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
}

pub(super) mod yuv444p10;
pub(super) mod yuv444p12;
pub(super) mod yuv444p14;
pub(super) mod yuv444p16;
pub(super) mod yuv444p9;
pub(super) mod yuv_444;

pub use yuv444p10::*;
pub use yuv444p12::*;
pub use yuv444p14::*;
pub use yuv444p16::*;
pub use yuv444p9::*;
pub use yuv_444::*;
