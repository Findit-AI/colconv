//! YUV 4:2:0 dispatchers, split per source format for readability.
//!
//! - `yuv_410` — 8-bit YUV 4:1:0 → RGB / RGBA (Cinepak / Sorenson
//!   legacy). Co-located here because 4:1:0 shares the vertical-
//!   subsampling walker shape with 4:2:0.
//! - `yuv_420` — 8-bit YUV 4:2:0 → RGB / RGBA.
//! - `yuv420p9` / `yuv420p10` / `yuv420p12` / `yuv420p14` /
//!   `yuv420p16` — high-bit planar 4:2:0 (4 variants per format:
//!   RGB, RGB-u16, RGBA, RGBA-u16).
//! - `p010` / `p012` / `p016` — high-bit semi-planar 4:2:0
//!   (4 variants per format).
//!
//! Public functions re-exported up to `crate::row::*` via parent
//! `dispatch/mod.rs`.

#[cfg(all(
  feature = "yuv-planar",
  any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
  )
))]
use crate::row::arch;
#[cfg(all(feature = "yuv-planar", target_arch = "aarch64"))]
use crate::row::neon_available;
#[cfg(all(feature = "yuv-planar", target_arch = "wasm32"))]
use crate::row::simd128_available;
#[cfg(all(feature = "yuv-planar", target_arch = "x86_64"))]
use crate::row::{avx2_available, avx512_available, sse41_available};
#[cfg(feature = "yuv-planar")]
use crate::{KernelMatrix, row::scalar};

/// YUV 4:2:0 planar 9/10/12/14-bit → planar **HSV** (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`)
/// dispatcher. Const generic over `BITS ∈ {9, 10, 12, 14}` and `BE`.
/// Direct: no source-width RGB row is materialized — the SIMD backends
/// stage a fixed 64-pixel **8-bit** RGB chunk internally over the
/// existing `yuv_420p_n_to_rgb_row` kernel + `rgb_to_hsv_row`, so the
/// output is byte-identical to `rgb_to_hsv_row(yuv_420p_n_to_rgb_row::
/// <BITS, BE>(...))` within the selected tier (the same 8-bit RGB
/// intermediate the high-bit→RGB→HSV path uses). Also serves 4:2:2
/// (identical per-row chroma shape).
///
/// Crate-private — external callers use the concrete
/// [`yuv420p10_to_hsv_row_endian`] family wrappers, which pin `BITS`;
/// the 16-bit path is [`yuv420p16_to_hsv_row_endian`]. `use_simd =
/// false` forces the scalar reference path.
#[cfg(feature = "yuv-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_hsv_row::<BITS, BE>(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_hsv_row::<BITS, BE>(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_hsv_row::<BITS, BE>(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_hsv_row::<BITS, BE>(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_hsv_row::<BITS, BE>(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_hsv_row::<BITS, BE>(
    y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
  );
}

#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p010;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p012;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p016;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p10;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p12;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p14;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p16;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p9;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv_410;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv_420;

#[cfg(feature = "yuv-semi-planar")]
pub use p010::*;
#[cfg(feature = "yuv-semi-planar")]
pub use p012::*;
#[cfg(feature = "yuv-semi-planar")]
pub use p016::*;
#[cfg(feature = "yuv-planar")]
pub use yuv_410::*;
#[cfg(feature = "yuv-planar")]
pub use yuv_420::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p9::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p10::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p12::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p14::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p16::*;
