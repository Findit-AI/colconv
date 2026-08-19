//! Runtime SIMD dispatchers for Yaf32 → {RGB, RGBA, RGBu16, RGBAu16, RGBf32,
//! luma, luma_u16, luma_f32, HSV} kernels.
//!
//! Source is a packed `[Y0, A0, Y1, A1, ...]` f32 plane (2 f32 per pixel).
//! `use_simd = false` bypasses the SIMD cascade and calls scalar directly. The
//! `rgb_f32` / `luma_f32` lossless paths always use scalar (mirroring the
//! `grayf32` dispatcher's lossless-path routing).

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
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::yaf32 as scalar,
  ya_row_elems,
};

// ---- yaf32_to_rgb_row ---------------------------------------------------------

/// Dispatch `yaf32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::yaf32_to_rgb_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_rgb_row::<BE>(packed, out, width);
}

// ---- yaf32_to_rgba_row --------------------------------------------------------

/// Dispatch `yaf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgba_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::yaf32_to_rgba_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_rgba_row::<BE>(packed, out, width);
}

// ---- yaf32_to_rgb_u16_row -----------------------------------------------------

/// Dispatch `yaf32_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  if !use_simd {
    return scalar::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
}

// ---- yaf32_to_rgba_u16_row ----------------------------------------------------

/// Dispatch `yaf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgba_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_elems(width), "out too short");
  if !use_simd {
    return scalar::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
}

// ---- yaf32_to_rgb_f32_row -----------------------------------------------------

/// Dispatch `yaf32_to_rgb_f32_row` (lossless replicate, all backends delegate
/// to scalar).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_f32_row<const BE: bool>(
  packed: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  scalar::yaf32_to_rgb_f32_row::<BE>(packed, out, width);
}

// ---- yaf32_to_luma_row --------------------------------------------------------

/// Dispatch `yaf32_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::yaf32_to_luma_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_luma_row::<BE>(packed, out, width);
}

// ---- yaf32_to_luma_u16_row ----------------------------------------------------

/// Dispatch `yaf32_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::yaf32_to_luma_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_luma_u16_row::<BE>(packed, out, width);
}

// ---- yaf32_to_luma_f32_row ----------------------------------------------------

/// Dispatch `yaf32_to_luma_f32_row` (lossless Y pass-through, no SIMD needed).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_f32_row<const BE: bool>(
  packed: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  scalar::yaf32_to_luma_f32_row::<BE>(packed, out, width);
}

// ---- yaf32_to_hsv_row ---------------------------------------------------------

/// Dispatch `yaf32_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_hsv_row<const BE: bool>(
  packed: &[f32],
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
    return scalar::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
}
