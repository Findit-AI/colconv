//! Runtime SIMD dispatchers for 32-bit packed RGB / RGBA kernels
//! (Rgb96 / Rgba128).
//!
//! All dispatchers validate input/output bounds via the shared
//! `rgb_row_bytes` / `rgba_row_bytes` / `rgb_row_elems` / `rgba_row_elems`
//! helpers, then route to the best available SIMD backend.
//!
//! **SIMD dispatch order (x86_64):** AVX-512 → AVX2 → SSE4.1 → scalar.
//! **SIMD dispatch (aarch64):** NEON → scalar.
//! **SIMD dispatch (wasm32):** wasm-simd128 → scalar.
//!
//! **Input element-strides**
//! - Rgb96:   source row is `width x 3` u32 elements.
//! - Rgba128: source row is `width x 4` u32 elements.
//!
//! **Luma / HSV signatures** take an extra `rgb_scratch: &mut [u8]` parameter
//! (length ≥ `width x 3` bytes). The dispatcher first narrows the source to
//! u8 RGB into that scratch, then applies the luma or HSV kernel.
// Luma / HSV dispatchers are wired into sinker impls alongside the conversion
// dispatchers; suppress dead_code until then.
#![allow(dead_code)]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  all(target_arch = "wasm32", target_feature = "simd128")
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  KernelMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

// ---- input-side element-count helpers -----------------------------------

/// Minimum u32-element count of one packed 3-channel row (`width x 3`).
/// Panics if `width x 3` overflows `usize`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb96_packed_elems(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) x 3 overflows usize (Rgb96 packed row)"),
  }
}

// =============================================================================
// Rgb96 (R, G, B — 3 u32 elements per pixel)
// =============================================================================

/// Converts one row of `Rgb96` to packed u8 RGB. Each 32-bit channel is
/// narrowed via `>> 24`. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgb_row_endian<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb96_to_rgb_row::<BE>(rgb96, rgb_out, width);
}

/// LE-only wrapper around [`rgb96_to_rgb_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgb_row(rgb96: &[u32], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  rgb96_to_rgb_row_endian::<false>(rgb96, rgb_out, width, use_simd)
}

/// Converts one row of `Rgb96` to packed u8 RGBA. Alpha forced to `0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgba_row_endian<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb96_to_rgba_row::<BE>(rgb96, rgba_out, width);
}

/// LE-only wrapper around [`rgb96_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgba_row(rgb96: &[u32], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  rgb96_to_rgba_row_endian::<false>(rgb96, rgba_out, width, use_simd)
}

/// Converts one row of `Rgb96` to native-depth u16 RGB (narrow `>> 16`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgb_u16_row_endian<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb96_to_rgb_u16_row::<BE>(rgb96, rgb_out, width);
}

/// LE-only wrapper around [`rgb96_to_rgb_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgb_u16_row(rgb96: &[u32], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  rgb96_to_rgb_u16_row_endian::<false>(rgb96, rgb_out, width, use_simd)
}

/// Converts one row of `Rgb96` to native-depth u16 RGBA (narrow `>> 16`,
/// alpha forced to `0xFFFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgba_u16_row_endian<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb96_to_rgba_u16_row::<BE>(rgb96, rgba_out, width);
}

/// LE-only wrapper around [`rgb96_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_rgba_u16_row(rgb96: &[u32], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  rgb96_to_rgba_u16_row_endian::<false>(rgb96, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Rgb96`. Narrows to u8 RGB via
/// `rgb96_to_rgb_row` into `rgb_scratch` (length ≥ `width x 3`), then applies
/// `rgb_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb96_to_luma_row_endian<const BE: bool>(
  rgb96: &[u32],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgb96_to_rgb_row_endian::<BE>(rgb96, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgb96_to_luma_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_luma_row(
  rgb96: &[u32],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgb96_to_luma_row_endian::<false>(
    rgb96,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Rgb96` (Y' computed at 8-bit precision
/// and zero-extended). Narrows to u8 RGB via `rgb96_to_rgb_row` into
/// `rgb_scratch`, then applies `rgb_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb96_to_luma_u16_row_endian<const BE: bool>(
  rgb96: &[u32],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgb96_to_rgb_row_endian::<BE>(rgb96, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgb96_to_luma_u16_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_luma_u16_row(
  rgb96: &[u32],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgb96_to_luma_u16_row_endian::<false>(
    rgb96,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Rgb96` (OpenCV 8-bit encoding).
/// Narrows to u8 RGB via `rgb96_to_rgb_row` into `rgb_scratch`, then applies
/// `rgb_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb96_to_hsv_row_endian<const BE: bool>(
  rgb96: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb96_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb96.len() >= in_min, "rgb96 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  rgb96_to_rgb_row_endian::<BE>(rgb96, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`rgb96_to_hsv_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb96_to_hsv_row(
  rgb96: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  rgb96_to_hsv_row_endian::<false>(rgb96, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}

/// Minimum u32-element count of one packed 4-channel row (`width x 4`).
/// Panics if `width x 4` overflows `usize`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgba128_packed_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) x 4 overflows usize (Rgba128 packed row)"),
  }
}

// =============================================================================
// Rgba128 (R, G, B, A — 4 u32 elements per pixel, source alpha real)
// =============================================================================

/// Converts one row of `Rgba128` to packed u8 RGB. Source alpha is discarded;
/// R/G/B narrowed via `>> 24`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgb_row_endian<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba128_to_rgb_row::<BE>(rgba128, rgb_out, width);
}

/// LE-only wrapper around [`rgba128_to_rgb_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgb_row(rgba128: &[u32], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  rgba128_to_rgb_row_endian::<false>(rgba128, rgb_out, width, use_simd)
}

/// Converts one row of `Rgba128` to packed u8 RGBA. All 4 channels narrowed
/// via `>> 24`; source alpha passes through.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgba_row_endian<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba128_to_rgba_row::<BE>(rgba128, rgba_out, width);
}

/// LE-only wrapper around [`rgba128_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgba_row(rgba128: &[u32], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  rgba128_to_rgba_row_endian::<false>(rgba128, rgba_out, width, use_simd)
}

/// Converts one row of `Rgba128` to native-depth u16 RGB. Source alpha
/// discarded; R/G/B narrowed `>> 16`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgb_u16_row_endian<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba128_to_rgb_u16_row::<BE>(rgba128, rgb_out, width);
}

/// LE-only wrapper around [`rgba128_to_rgb_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgb_u16_row(rgba128: &[u32], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  rgba128_to_rgb_u16_row_endian::<false>(rgba128, rgb_out, width, use_simd)
}

/// Converts one row of `Rgba128` to native-depth u16 RGBA (all 4 channels
/// narrowed `>> 16`; source alpha preserved).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgba_u16_row_endian<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba128_to_rgba_u16_row::<BE>(rgba128, rgba_out, width);
}

/// LE-only wrapper around [`rgba128_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_rgba_u16_row(
  rgba128: &[u32],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  rgba128_to_rgba_u16_row_endian::<false>(rgba128, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Rgba128` (alpha ignored). Narrows to u8
/// RGB via `rgba128_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba128_to_luma_row_endian<const BE: bool>(
  rgba128: &[u32],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgba128_to_rgb_row_endian::<BE>(rgba128, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgba128_to_luma_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_luma_row(
  rgba128: &[u32],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgba128_to_luma_row_endian::<false>(
    rgba128,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Rgba128` (alpha ignored; Y' computed at
/// 8-bit precision and zero-extended).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba128_to_luma_u16_row_endian<const BE: bool>(
  rgba128: &[u32],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgba128_to_rgb_row_endian::<BE>(rgba128, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgba128_to_luma_u16_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_luma_u16_row(
  rgba128: &[u32],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgba128_to_luma_u16_row_endian::<false>(
    rgba128,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Rgba128` (alpha ignored). Narrows to u8
/// RGB via `rgba128_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba128_to_hsv_row_endian<const BE: bool>(
  rgba128: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba128_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba128.len() >= in_min, "rgba128 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  rgba128_to_rgb_row_endian::<BE>(rgba128, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`rgba128_to_hsv_row_endian`].
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba128_to_hsv_row(
  rgba128: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  rgba128_to_hsv_row_endian::<false>(rgba128, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}
