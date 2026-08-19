//! Runtime SIMD dispatchers for Grayf16 → {RGB, RGBA, RGBu16, RGBAu16, RGBf32,
//! luma, luma_u16, luma_f32, HSV} kernels.
//!
//! The source is a `&[half::f16]` luma plane. Each SIMD backend widens the f16
//! samples to f32 (AArch64 `vcvt_f32_f16` gated on NEON + `fp16`; x86 F16C
//! `_mm{,256,512}_cvtph_ps` gated on the SIMD tier + `f16c`; wasm + the f16→f32
//! lossless paths fall back to scalar) and then routes through the existing
//! `grayf32` kernels — so the f16 → integer rounding is byte-identical to
//! `grayf32` once the (lossless) widen is applied.
//!
//! `use_simd = false` bypasses the SIMD cascade and calls scalar directly. The
//! `rgb_f32` / `luma_f32` widening paths always use scalar (mirroring the
//! `grayf32` dispatcher's lossless-path routing).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

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
use crate::row::{
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::grayf16 as scalar,
};

// ---- grayf16_to_rgb_row -------------------------------------------------------

/// Dispatch `grayf16_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::grayf16_to_rgb_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_rgb_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_rgb_row::<BE>(plane, out, width);
}

// ---- grayf16_to_rgba_row ------------------------------------------------------

/// Dispatch `grayf16_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgba_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgba_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::grayf16_to_rgba_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_rgba_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_rgba_row::<BE>(plane, out, width);
}

// ---- grayf16_to_rgb_u16_row ---------------------------------------------------

/// Dispatch `grayf16_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_u16_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  if !use_simd {
    return scalar::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_rgb_u16_row::<BE>(plane, out, width);
}

// ---- grayf16_to_rgba_u16_row --------------------------------------------------

/// Dispatch `grayf16_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgba_u16_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgba_row_elems(width), "out too short");
  if !use_simd {
    return scalar::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_rgba_u16_row::<BE>(plane, out, width);
}

// ---- grayf16_to_rgb_f32_row ---------------------------------------------------

/// Dispatch `grayf16_to_rgb_f32_row` (lossless f16 → f32 widen + replicate; all
/// backends delegate to scalar, mirroring the `grayf32` lossless-path routing).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_f32_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  scalar::grayf16_to_rgb_f32_row::<BE>(plane, out, width);
}

// ---- grayf16_to_luma_row ------------------------------------------------------

/// Dispatch `grayf16_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::grayf16_to_luma_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_luma_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_luma_row::<BE>(plane, out, width);
}

// ---- grayf16_to_luma_u16_row --------------------------------------------------

/// Dispatch `grayf16_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_u16_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::grayf16_to_luma_u16_row::<BE>(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_luma_u16_row::<BE>(plane, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_luma_u16_row::<BE>(plane, out, width);
}

// ---- grayf16_to_luma_f32_row --------------------------------------------------

/// Dispatch `grayf16_to_luma_f32_row` (lossless f16 → f32 widen; all backends
/// delegate to scalar, mirroring the `grayf32` lossless-path routing).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_f32_row<const BE: bool>(
  plane: &[half::f16],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(plane.len() >= width, "plane too short");
  assert!(out.len() >= width, "out too short");
  scalar::grayf16_to_luma_f32_row::<BE>(plane, out, width);
}

// ---- grayf16_to_hsv_row -------------------------------------------------------

/// Dispatch `grayf16_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_hsv_row<const BE: bool>(
  plane: &[half::f16],
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
    return scalar::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::grayf16_to_hsv_row::<BE>(plane, h_out, s_out, v_out, width);
}
