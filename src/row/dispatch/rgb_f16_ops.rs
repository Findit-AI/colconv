//! Tier 9 — packed half-precision float RGB (`Rgbf16`) source-side row
//! dispatchers.
//!
//! Each entry point converts one row of packed `R, G, B` `half::f16` input
//! to the requested output format. The conversion always starts with
//! hardware-accelerated f16 → f32 widening (AArch64 `vcvt_f32_f16`, x86
//! F16C `_mm{,256,512}_cvtph_ps`) or scalar `half::f16::to_f32`, then
//! delegates downstream conversion to the existing `rgbf32_to_*_row`
//! kernels.
//!
//! The `with_rgb_f16` lossless memcpy and `with_rgb_f32` lossless widening
//! paths skip the downstream `f32` kernels entirely.
//!
//! Backends:
//! - AArch64: NEON FCVT widening + existing NEON f32 kernels (gated on
//!   `neon` AND `fp16` runtime detection — `fp16` is a separate CPU-feature
//!   bit on AArch64; falls back to scalar without it).
//! - x86_64: AVX-512F+F16C → AVX2+F16C → SSE4.1+F16C → fully scalar (F16C
//!   detection is a runtime guard *in addition to* the SIMD tier check; on
//!   machines without F16C the dispatcher routes to the scalar
//!   `rgbf16_to_*_row` reference instead of mixing scalar widening with
//!   SIMD downstream — the latter would copy widened f32 to a scratch
//!   buffer twice and benchmarks haven't shown a win over plain scalar).
//! - wasm32: scalar widen + wasm-simd128 downstream.
//!
//! `use_simd = false` forces the scalar reference path.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, f16c_available, sse41_available};
#[cfg(target_arch = "aarch64")]
use crate::row::{fp16_available, neon_available};
use crate::row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar};

/// Converts packed `R, G, B` `half::f16` input to packed `R, G, B` `u8`
/// output with `[0, 1]` saturation and ×255 scaling.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_bytes(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          // SAFETY: AVX-512F + F16C verified.
          unsafe {
            arch::x86_avx512::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          // SAFETY: AVX2 + F16C verified.
          unsafe {
            arch::x86_avx2::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          // SAFETY: SSE4.1 + F16C verified.
          unsafe {
            arch::x86_sse41::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgb_row::<BE>(rgb_in, rgb_out, width);
}

/// Converts packed `R, G, B` `half::f16` input to packed `R, G, B, A` `u8`
/// output (`A = 0xFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgba_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_bytes(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgba_row::<BE>(rgb_in, rgba_out, width);
}

/// Converts packed `R, G, B` `half::f16` input to packed `R, G, B` `u16`
/// output with `[0, 1]` saturation and ×65535 scaling.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
}

/// Converts packed `R, G, B` `half::f16` input to packed `R, G, B, A` `u16`
/// output (`A = 0xFFFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
}

/// **Lossless** half-float pass-through: copies packed `R, G, B` `half::f16`
/// from input into output verbatim. HDR values > 1.0 and negatives are
/// preserved bit-exact.
///
/// `use_simd = false` forces the scalar reference path (which is also just
/// `copy_from_slice` — the compiler will vectorize it regardless).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_f16_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
}

/// Lossless widening pass: converts packed `R, G, B` `half::f16` input to
/// packed `R, G, B` `f32` output. HDR values > 1.0 and negatives are
/// preserved (no clamping).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf16 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_f32_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf16_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
}

// ---- Tier 9 — packed half-float RGBA (`Rgbaf16`) source-side dispatchers --
//
// The alpha-bearing twin of the `rgbf16_to_*_row` dispatchers. Input is
// packed `R, G, B, A` `half::f16` (`4 * width` elements); the `*_to_rgb*`
// variants drop alpha, the `*_to_rgba*` variants carry it. Each SIMD
// backend delegates to the sibling `rgbf16_*` widen-then-convert kernel,
// so the F16C / fp16 runtime gating mirrors the `Rgbf16` path exactly.

/// Packed `R, G, B, A` `f16` → packed `R, G, B` `u8` (drop alpha).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgb_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          // SAFETY: NEON + fp16 verified.
          unsafe {
            arch::neon::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgb_row::<BE>(rgba_in, rgb_out, width);
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B, A` `u8` (real alpha).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgba_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgba_row::<BE>(rgba_in, rgba_out, width);
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B` `u16` (drop alpha).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B, A` `u16` (real alpha).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
}

/// **Lossless** drop-alpha: packed `R, G, B, A` `f16` → packed `R, G, B`
/// `f16` (alpha dropped, bit-exact pass-through).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgb_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_f16_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
}

/// **Lossless** packed `R, G, B, A` `f16` pass-through (4 channels, alpha
/// preserved bit-exact).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgba_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_f16_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
}

/// Widening drop-alpha: packed `R, G, B, A` `f16` → packed `R, G, B`
/// `f32` (alpha dropped, HDR/negatives preserved).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_f32_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
}

/// Widening pass-through: packed `R, G, B, A` `f16` → packed `R, G, B, A`
/// `f32` (4 channels, alpha widened too).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf16_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf16 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_f32_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          unsafe {
            arch::neon::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          unsafe {
            arch::x86_avx512::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() && f16c_available() {
          unsafe {
            arch::x86_avx2::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() && f16c_available() {
          unsafe {
            arch::x86_sse41::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf16_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
}
