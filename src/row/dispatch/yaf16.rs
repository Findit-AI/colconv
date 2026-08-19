//! Runtime SIMD dispatchers for Yaf16 → {RGB, RGBA, RGBu16, RGBAu16, RGBf32,
//! luma, luma_u16, luma_f32, HSV} kernels.
//!
//! Source is a packed `[Y0, A0, Y1, A1, ...]` `half::f16` plane (2 f16 per
//! pixel). Each SIMD backend widens the f16 samples to f32 (AArch64
//! `vcvt_f32_f16` gated on NEON + `fp16`; x86 F16C `_mm{,256,512}_cvtph_ps`
//! gated on the SIMD tier + `f16c`; wasm + the f16→f32 lossless paths fall back
//! to scalar) and then routes through the existing `yaf32` kernels — so the f16
//! → integer rounding is byte-identical to `yaf32` once the (lossless) widen is
//! applied.
//!
//! `use_simd = false` bypasses the SIMD cascade and calls scalar directly. The
//! `rgb_f32` / `luma_f32` widening paths always use scalar (mirroring the
//! `grayf16` dispatcher's lossless-path routing).

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
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::yaf16 as scalar,
  ya_row_elems,
};

// ---- yaf16_to_rgb_row ---------------------------------------------------------

/// Dispatch `yaf16_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_rgb_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::yaf16_to_rgb_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_rgb_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_rgb_row::<BE>(packed, out, width);
}

// ---- yaf16_to_rgba_row --------------------------------------------------------

/// Dispatch `yaf16_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_rgba_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_bytes(width), "out too short");
  if !use_simd {
    return scalar::yaf16_to_rgba_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_rgba_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_rgba_row::<BE>(packed, out, width);
}

// ---- yaf16_to_rgb_u16_row -----------------------------------------------------

/// Dispatch `yaf16_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_rgb_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  if !use_simd {
    return scalar::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
}

// ---- yaf16_to_rgba_u16_row ----------------------------------------------------

/// Dispatch `yaf16_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_rgba_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgba_row_elems(width), "out too short");
  if !use_simd {
    return scalar::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
}

// ---- yaf16_to_rgb_f32_row -----------------------------------------------------

/// Dispatch `yaf16_to_rgb_f32_row` (lossless f16 → f32 widen + replicate; all
/// backends delegate to scalar, mirroring the `grayf16` lossless-path routing).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_rgb_f32_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= rgb_row_elems(width), "out too short");
  scalar::yaf16_to_rgb_f32_row::<BE>(packed, out, width);
}

// ---- yaf16_to_luma_row --------------------------------------------------------

/// Dispatch `yaf16_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_luma_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::yaf16_to_luma_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_luma_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_luma_row::<BE>(packed, out, width);
}

// ---- yaf16_to_luma_u16_row ----------------------------------------------------

/// Dispatch `yaf16_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_luma_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  if !use_simd {
    return scalar::yaf16_to_luma_u16_row::<BE>(packed, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_luma_u16_row::<BE>(packed, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_luma_u16_row::<BE>(packed, out, width);
}

// ---- yaf16_to_luma_f32_row ----------------------------------------------------

/// Dispatch `yaf16_to_luma_f32_row` (lossless f16 → f32 widen; all backends
/// delegate to scalar, mirroring the `grayf16` lossless-path routing).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_luma_f32_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  assert!(packed.len() >= ya_row_elems(width), "packed too short");
  assert!(out.len() >= width, "out too short");
  scalar::yaf16_to_luma_f32_row::<BE>(packed, out, width);
}

// ---- yaf16_to_hsv_row ---------------------------------------------------------

/// Dispatch `yaf16_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf16_to_hsv_row<const BE: bool>(
  packed: &[half::f16],
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
    return scalar::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && fp16_available() {
        unsafe {
          arch::neon::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() && f16c_available() {
        unsafe {
          arch::x86_avx512::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if avx2_available() && f16c_available() {
        unsafe {
          arch::x86_avx2::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
      if sse41_available() && f16c_available() {
        unsafe {
          arch::x86_sse41::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
}
