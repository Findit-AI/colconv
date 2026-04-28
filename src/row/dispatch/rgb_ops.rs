//! RGB→HSV and BGR↔RGB swap dispatchers extracted from `row::mod` for
//! organization. All three route through the standard
//! `cfg_select!` per-arch block; `use_simd = false` forces scalar.

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
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of packed RGB to planar HSV (OpenCV 8‑bit
/// encoding). See `scalar::rgb_to_hsv_row` for semantics.
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend (same semantics as `yuv_420_to_rgb_row`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb.len() >= rgb_min, "rgb row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD HSV backend fall through to scalar.
      }
    }
  }

  scalar::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
}

/// Derives **luma** (Y') from packed RGB. Used by Tier 6 packed-RGB
/// source sinkers' `with_luma` path (the source has no Y plane to
/// memcpy, so luma must be computed from R/G/B). See
/// `scalar::rgb_to_luma_row` for semantics — `matrix` selects the
/// BT.* coefficient set, `full_range` chooses Y' ∈ `[0, 255]` vs
/// `[16, 235]`.
///
/// `use_simd` is currently a no-op — scalar is the only available
/// path for this kernel today. SIMD wiring lands in a follow-up
/// once enough Tier 6 callers exist to justify the per-arch work.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb_to_luma_row(
  rgb: &[u8],
  luma_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  _use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb.len() >= rgb_min, "rgb row too short");
  assert!(luma_out.len() >= width, "luma row too short");

  scalar::rgb_to_luma_row(rgb, luma_out, width, matrix, full_range);
}

/// Drops the alpha byte from packed `R, G, B, A` input, producing
/// packed `R, G, B` output (`4 * width` → `3 * width` bytes). Used
/// by [`Rgba`](crate::yuv::Rgba) sinker's RGB / luma / HSV paths
/// (Ship 9b).
///
/// `use_simd` is currently a no-op — scalar is the only available
/// path today.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba_to_rgb_row(rgba: &[u8], rgb_out: &mut [u8], width: usize, _use_simd: bool) {
  let rgba_min = rgba_row_bytes(width);
  let rgb_min = rgb_row_bytes(width);
  assert!(rgba.len() >= rgba_min, "rgba row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
  scalar::rgba_to_rgb_row(rgba, rgb_out, width);
}

/// Swaps R↔B in packed `B, G, R, A` input, producing packed
/// `R, G, B, A` (alpha preserved). Self-inverse, so the same
/// dispatcher can be called for either direction. Used by
/// [`Bgra`](crate::yuv::Bgra) sinker's RGBA-output path.
///
/// `use_simd` is currently a no-op — scalar is the only available
/// path today.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra_to_rgba_row(bgra: &[u8], rgba_out: &mut [u8], width: usize, _use_simd: bool) {
  let rgba_min = rgba_row_bytes(width);
  assert!(bgra.len() >= rgba_min, "bgra row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");
  scalar::bgra_to_rgba_row(bgra, rgba_out, width);
}

/// Swaps R↔B and drops alpha from packed `B, G, R, A` input,
/// producing packed `R, G, B` (`4 * width` → `3 * width` bytes).
/// Used by [`Bgra`](crate::yuv::Bgra) sinker's RGB / luma / HSV
/// paths.
///
/// `use_simd` is currently a no-op — scalar is the only available
/// path today.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra_to_rgb_row(bgra: &[u8], rgb_out: &mut [u8], width: usize, _use_simd: bool) {
  let rgba_min = rgba_row_bytes(width);
  let rgb_min = rgb_row_bytes(width);
  assert!(bgra.len() >= rgba_min, "bgra row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
  scalar::bgra_to_rgb_row(bgra, rgb_out, width);
}

/// Rewrites a row of packed BGR to packed RGB by swapping the outer
/// two channels (byte 0 ↔ byte 2) of every triple. `input` and
/// `output` must not alias.
///
/// The underlying transformation is self‑inverse, so
/// [`rgb_to_bgr_row`] shares the same implementation — use whichever
/// name reads more naturally at the call site.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr_to_rgb_row(bgr: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  swap_rb_channels_row(bgr, rgb_out, width, use_simd);
}

/// Rewrites a row of packed RGB to packed BGR by swapping the outer
/// two channels. See [`bgr_to_rgb_row`] — this is an alias that reads
/// more naturally for the opposite direction.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb_to_bgr_row(rgb: &[u8], bgr_out: &mut [u8], width: usize, use_simd: bool) {
  swap_rb_channels_row(rgb, bgr_out, width, use_simd);
}

/// Shared dispatcher behind `bgr_to_rgb_row` / `rgb_to_bgr_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn swap_rb_channels_row(input: &[u8], output: &mut [u8], width: usize, use_simd: bool) {
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  let rgb_min = rgb_row_bytes(width);
  assert!(input.len() >= rgb_min, "input row too short");
  assert!(output.len() >= rgb_min, "output row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 just verified.
          unsafe {
            arch::x86_avx2::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 just verified.
          unsafe {
            arch::x86_sse41::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::bgr_rgb_swap_row(input, output, width);
}
