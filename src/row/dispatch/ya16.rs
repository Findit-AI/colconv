//! Runtime SIMD dispatchers for Ya16 → {RGB, RGBA, RGBu16, RGBAu16, luma,
//! luma_u16, HSV} kernels.
//!
//! Source is a packed `[Y0, A0, Y1, A1, ...]` u16 plane.
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
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::ya16 as scalar,
  ya_row_elems,
};

// ---- ya16_to_rgb_row ----------------------------------------------------------

/// Dispatch `ya16_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::ya16_to_rgb_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_rgb_row::<BE>(packed, out, width);
}

// ---- ya16_to_rgba_row ---------------------------------------------------------

/// Dispatch `ya16_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::ya16_to_rgba_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_rgba_row::<BE>(packed, out, width);
}

// ---- ya16_to_rgb_u16_row ------------------------------------------------------

/// Dispatch `ya16_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  if !use_simd {
    return scalar::ya16_to_rgb_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_rgb_u16_row::<BE>(packed, out, width);
}

// ---- ya16_to_rgba_u16_row -----------------------------------------------------

/// Dispatch `ya16_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_elems(width), "out too short");
  if !use_simd {
    return scalar::ya16_to_rgba_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_rgba_u16_row::<BE>(packed, out, width);
}

// ---- ya16_to_luma_row ---------------------------------------------------------

/// Dispatch `ya16_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::ya16_to_luma_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_luma_row::<BE>(packed, out, width);
}

// ---- ya16_to_luma_u16_row -----------------------------------------------------

/// Dispatch `ya16_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::ya16_to_luma_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_luma_u16_row::<BE>(packed, out, width);
}

// ---- ya16_to_hsv_row ----------------------------------------------------------

/// Dispatch `ya16_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(h_out.len() >= width, "H out too short");
  assert!(s_out.len() >= width, "S out too short");
  assert!(v_out.len() >= width, "V out too short");
  if !use_simd {
    return scalar::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
}
