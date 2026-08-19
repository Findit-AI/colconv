//! Tier 9 — packed float RGB (`Rgbf32`) source-side row dispatchers.
//!
//! Each entry point converts one row of packed `R, G, B` `f32` input
//! to the requested output format. Outputs targeting integer types
//! (u8 / u16) clamp the source value to `[0, 1]` and scale by the
//! output range (255 / 65535); the `f32` output (`rgbf32_to_rgb_f32_row`)
//! is a lossless memcpy that preserves HDR values > 1.0 and negative
//! values bit-exact.
//!
//! Backends: native NEON / SSE4.1 / AVX2 / AVX-512 / wasm-simd128
//! kernels operate on `f32x4` / `f32x8` / `f32x16` registers; `use_simd
//! = false` forces the scalar reference path.

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
use crate::row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar};

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u8`
/// output with `[0, 1]` saturation and ×255 scaling.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf32_to_rgb_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_bytes(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf32 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F verified.
          unsafe {
            arch::x86_avx512::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf32_to_rgb_row::<BE>(rgb_in, rgb_out, width);
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u8`
/// output (`A = 0xFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_bytes(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf32 row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf32_to_rgba_row::<BE>(rgb_in, rgba_out, width);
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u16`
/// output with `[0, 1]` saturation and ×65535 scaling.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf32 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf32_to_rgb_u16_row::<BE>(rgb_in, rgb_out, width);
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u16`
/// output (`A = 0xFFFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf32 row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_out, width);
}

/// **Lossless** float pass-through: copies packed `R, G, B` `f32`
/// from input into output verbatim. HDR values > 1.0 and negatives
/// are preserved bit-exact.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbf32_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let rgb_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(rgb_in.len() >= rgb_in_min, "rgbf32 row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_f32_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbf32_to_rgb_f32_row::<BE>(rgb_in, rgb_out, width);
}

// ---- Tier 9 — packed float RGBA (`Rgbaf32`) source-side dispatchers -----
//
// The alpha-bearing twin of the `rgbf32_to_*_row` dispatchers. Input is
// packed `R, G, B, A` `f32` (`4 * width` elements); the `*_to_rgb*`
// variants drop alpha, the `*_to_rgba*` variants carry it. Each SIMD
// backend delegates to the sibling `rgbf32_*` kernel (a flat 12-lane feed
// for the real-alpha forms, a gather-then-convert for the drop-alpha
// forms), so the backend selection mirrors the `Rgbf32` path exactly.

/// Packed `R, G, B, A` `f32` → packed `R, G, B` `u8` (drop alpha,
/// `[0, 1]` × 255). `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgb_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F verified.
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_out, width);
}

/// Packed `R, G, B, A` `f32` → packed `R, G, B, A` `u8` (real alpha,
/// `[0, 1]` × 255 per channel). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgba_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_out, width);
}

/// Packed `R, G, B, A` `f32` → packed `R, G, B` `u16` (drop alpha,
/// `[0, 1]` × 65535). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgb_u16_row::<BE>(rgba_in, rgb_out, width);
}

/// Packed `R, G, B, A` `f32` → packed `R, G, B, A` `u16` (real alpha,
/// `[0, 1]` × 65535 per channel). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_out, width);
}

/// **Lossless** drop-alpha: packed `R, G, B, A` `f32` → packed `R, G, B`
/// `f32` (alpha dropped, HDR/negatives preserved). `use_simd = false`
/// forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_f32_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgb_f32_row::<BE>(rgba_in, rgb_out, width);
}

/// **Lossless** packed `R, G, B, A` `f32` pass-through (4 channels, alpha
/// preserved; HDR/negatives bit-exact). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgbaf32_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  assert!(
    rgba_in.len() >= rgba_row_elems(width),
    "rgbaf32 row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_f32_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }
  scalar::rgbaf32_to_rgba_f32_row::<BE>(rgba_in, rgba_out, width);
}
