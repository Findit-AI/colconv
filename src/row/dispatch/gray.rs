//! Runtime SIMD dispatchers for gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! Each dispatcher selects the highest available SIMD backend and falls back
//! to scalar. `use_simd = false` bypasses the SIMD cascade and calls scalar
//! directly.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

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
use crate::row::{
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::gray as scalar,
};

// ---- Gray8 ------------------------------------------------------------------

/// Dispatch `gray8_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  // Compute the overflow-checking helper BEFORE the y_plane bounds
  // assert so 32-bit `width × N` overflow surfaces as the documented
  // "overflows usize" panic instead of a misleading "too short" message.
  let out_min = rgb_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray8_to_rgb_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray8_to_rgb_row(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray8_to_rgb_row(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray8_to_rgb_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray8_to_rgb_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
}

/// Dispatch `gray8_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray8_to_rgba_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray8_to_rgba_row(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray8_to_rgba_row(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray8_to_rgba_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray8_to_rgba_row(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
}

/// Dispatch `gray8_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
}

// ---- GrayN (const BITS) ------------------------------------------------

/// Dispatch `gray_n_to_rgb_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray_n_to_rgba_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray_n_to_rgb_u16_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray_n_to_rgba_u16_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray_n_to_luma_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_luma_row::<BITS, BE>(y_plane, out, width);
}

/// Dispatch `gray_n_to_luma_u16_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_luma_u16_row::<BITS, BE>(y_plane, out, width);
}

/// Dispatch `gray_n_to_hsv_row<BITS>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::gray_n_to_hsv_row::<BITS, BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray_n_to_hsv_row::<BITS, BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray_n_to_hsv_row::<BITS, BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray_n_to_hsv_row::<BITS, BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray_n_to_hsv_row::<BITS, BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray_n_to_hsv_row::<BITS, BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray_n_to_hsv_row::<BITS, BE>(y_plane, h_out, s_out, v_out, width, full_range);
}

// ---- Gray16 ----------------------------------------------------------------

/// Dispatch `gray16_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray16_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray16_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray16_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray16_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray16_to_luma_row::<BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_luma_row::<BE>(y_plane, out, width);
}

/// Dispatch `gray16_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_luma_u16_row::<BE>(y_plane, out, width);
}

/// Dispatch `gray16_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_hsv_row<const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray16_to_hsv_row::<BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray16_to_hsv_row::<BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
}

// ---- Gray32 ----------------------------------------------------------------

/// Dispatch `gray32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgb_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgba_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray32_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= out_min, "out too short");
  if !use_simd {
    return scalar::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
}

/// Dispatch `gray32_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_luma_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray32_to_luma_row::<BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_luma_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_luma_row::<BE>(y_plane, out, width);
}

/// Dispatch `gray32_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_luma_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_luma_u16_row::<BE>(y_plane, out, width);
}

/// Dispatch `gray32_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray32_to_hsv_row<const BE: bool>(
  y_plane: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
  full_range: bool,
) {
  assert!(y_plane.len() >= width, "y_plane too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::gray32_to_hsv_row::<BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::gray32_to_hsv_row::<BE>(
            y_plane, h_out, s_out, v_out, width, full_range,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
}
