//! Runtime SIMD dispatchers for Grayf32 → {RGB, RGBA, RGBu16, RGBAu16, RGBf32,
//! luma, luma_u16, luma_f32, HSV} kernels.
//!
//! `use_simd = false` bypasses the SIMD cascade and calls scalar directly.

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
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::grayf32 as scalar,
};

// ---- grayf32_to_rgb_row -------------------------------------------------------

/// Dispatch `grayf32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::grayf32_to_rgb_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_rgb_row::<BE>(plane, out, width);
}

// ---- grayf32_to_rgba_row ------------------------------------------------------

/// Dispatch `grayf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgba_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::grayf32_to_rgba_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_rgba_row::<BE>(plane, out, width);
}

// ---- grayf32_to_rgb_u16_row ---------------------------------------------------

/// Dispatch `grayf32_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_u16_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  if !use_simd {
    return scalar::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_rgb_u16_row::<BE>(plane, out, width);
}

// ---- grayf32_to_rgba_u16_row --------------------------------------------------

/// Dispatch `grayf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_u16_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgba_row_elems(width), "out too short");
  if !use_simd {
    return scalar::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_rgba_u16_row::<BE>(plane, out, width);
}

// ---- grayf32_to_rgb_f32_row ---------------------------------------------------

/// Dispatch `grayf32_to_rgb_f32_row` (lossless replicate, all backends delegate to scalar).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_f32_row<const BE: bool>(
  plane: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  scalar::grayf32_to_rgb_f32_row::<BE>(plane, out, width);
}

// ---- grayf32_to_luma_row ------------------------------------------------------

/// Dispatch `grayf32_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::grayf32_to_luma_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_luma_row::<BE>(plane, out, width);
}

// ---- grayf32_to_luma_u16_row --------------------------------------------------

/// Dispatch `grayf32_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_u16_row<const BE: bool>(
  plane: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::grayf32_to_luma_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_luma_u16_row::<BE>(plane, out, width);
}

// ---- grayf32_to_luma_f32_row --------------------------------------------------

/// Dispatch `grayf32_to_luma_f32_row` (lossless memcpy, no SIMD needed).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_f32_row<const BE: bool>(
  plane: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  scalar::grayf32_to_luma_f32_row::<BE>(plane, out, width);
}

// ---- grayf32_to_hsv_row -------------------------------------------------------

/// Dispatch `grayf32_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_hsv_row<const BE: bool>(
  plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf32_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
}
