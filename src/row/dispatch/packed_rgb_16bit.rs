//! Runtime SIMD dispatchers for 16-bit packed RGB/BGR/RGBA/BGRA kernels
//! (Tier 8 finish).
//!
//! 28 public functions:
//!   4 source formats (Rgb48, Bgr48, Rgba64, Bgra64)
//!   x 7 outputs (rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, hsv)
//!
//! All dispatchers validate input/output bounds via the shared
//! `rgb_row_bytes` / `rgba_row_bytes` / `rgb_row_elems` / `rgba_row_elems`
//! helpers, then route to the best available SIMD backend.
//!
//! **SIMD dispatch order (x86_64):** AVX-512 → AVX2 → SSE4.1 → scalar.
//! **SIMD dispatch (aarch64):** NEON → scalar.
//! **SIMD dispatch (wasm32):** wasm-simd128 → scalar (wired in later task).
//!
//! **Input element-strides**
//! - Rgb48 / Bgr48: source row is `width x 3` u16 elements.
//! - Rgba64 / Bgra64: source row is `width x 4` u16 elements.
//!
//! **Luma / HSV signatures** take an extra `rgb_scratch: &mut [u8]` parameter
//! (length ≥ `width x 3` bytes). The dispatcher first narrows the source to
//! u8 RGB into that scratch, then applies the luma or HSV kernel. This lets
//! the sinker reuse its managed scratch rather than forcing a heap allocation
//! at the row level.
// Luma / HSV dispatchers are wired into sinker impls in Task 9; suppress
// dead_code until then.
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

/// Minimum u16-element count of one packed 3-channel row (`width x 3`).
/// Panics if `width x 3` overflows `usize` (only possible on 32-bit targets
/// with extreme widths). Result feeds the dispatcher input-side `assert!`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb48_packed_elems(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) x 3 overflows usize (Rgb48/Bgr48 packed row)"),
  }
}

/// Minimum u16-element count of one packed 4-channel row (`width x 4`).
/// Panics if `width x 4` overflows `usize`. Result feeds the dispatcher
/// input-side `assert!`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgba64_packed_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) x 4 overflows usize (Rgba64/Bgra64 packed row)"),
  }
}

// =============================================================================
// Rgb48 (R, G, B — 3 u16 elements per pixel)
// =============================================================================

/// Converts one row of `Rgb48` to packed u8 RGB. Each 16-bit channel is
/// narrowed via `>> 8`. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgb_row_endian<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb48_to_rgb_row::<BE>(rgb48, rgb_out, width);
}

/// LE-only wrapper around [`rgb48_to_rgb_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_rgb_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgb_row(rgb48: &[u16], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  rgb48_to_rgb_row_endian::<false>(rgb48, rgb_out, width, use_simd)
}

/// Converts one row of `Rgb48` to packed u8 RGBA. Alpha forced to `0xFF`.
/// `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgba_row_endian<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb48_to_rgba_row::<BE>(rgb48, rgba_out, width);
}

/// LE-only wrapper around [`rgb48_to_rgba_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_rgba_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgba_row(rgb48: &[u16], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  rgb48_to_rgba_row_endian::<false>(rgb48, rgba_out, width, use_simd)
}

/// Converts one row of `Rgb48` to native-depth u16 RGB (identity copy).
/// `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgb_u16_row_endian<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb48_to_rgb_u16_row::<BE>(rgb48, rgb_out, width);
}

/// LE-only wrapper around [`rgb48_to_rgb_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_rgb_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgb_u16_row(rgb48: &[u16], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  rgb48_to_rgb_u16_row_endian::<false>(rgb48, rgb_out, width, use_simd)
}

/// Converts one row of `Rgb48` to native-depth u16 RGBA. Alpha forced to
/// `0xFFFF`. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgba_u16_row_endian<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgb48_to_rgba_u16_row::<BE>(rgb48, rgba_out, width);
}

/// LE-only wrapper around [`rgb48_to_rgba_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_rgba_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_rgba_u16_row(rgb48: &[u16], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  rgb48_to_rgba_u16_row_endian::<false>(rgb48, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Rgb48` source. Narrows to u8 RGB via
/// `rgb48_to_rgb_row` into `rgb_scratch` (length ≥ `width x 3`), then applies
/// `rgb_to_luma_row`. `use_simd = false` forces the scalar path for both steps.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb48_to_luma_row_endian<const BE: bool>(
  rgb48: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgb48_to_rgb_row_endian::<BE>(rgb48, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgb48_to_luma_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_luma_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_luma_row(
  rgb48: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgb48_to_luma_row_endian::<false>(
    rgb48,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Rgb48` source (Y' is computed at 8-bit
/// precision and zero-extended). Narrows to u8 RGB via `rgb48_to_rgb_row` into
/// `rgb_scratch`, then applies `rgb_to_luma_u16_row`. `use_simd = false` forces
/// the scalar path for both steps.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb48_to_luma_u16_row_endian<const BE: bool>(
  rgb48: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgb48_to_rgb_row_endian::<BE>(rgb48, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgb48_to_luma_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_luma_u16_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_luma_u16_row(
  rgb48: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgb48_to_luma_u16_row_endian::<false>(
    rgb48,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Rgb48` source (OpenCV 8-bit encoding).
/// Narrows to u8 RGB via `rgb48_to_rgb_row` into `rgb_scratch`, then applies
/// `rgb_to_hsv_row`. `use_simd = false` forces the scalar path for both steps.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgb48_to_hsv_row_endian<const BE: bool>(
  rgb48: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgb48.len() >= in_min, "rgb48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  rgb48_to_rgb_row_endian::<BE>(rgb48, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`rgb48_to_hsv_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgb48_to_hsv_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb48_to_hsv_row(
  rgb48: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  rgb48_to_hsv_row_endian::<false>(rgb48, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}

// =============================================================================
// Bgr48 (B, G, R — 3 u16 elements per pixel)
// =============================================================================

/// Converts one row of `Bgr48` to packed u8 RGB (B↔R swap, narrow via `>> 8`).
/// `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgb_row_endian<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgr48_to_rgb_row::<BE>(bgr48, rgb_out, width);
}

/// LE-only wrapper around [`bgr48_to_rgb_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_rgb_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgb_row(bgr48: &[u16], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  bgr48_to_rgb_row_endian::<false>(bgr48, rgb_out, width, use_simd)
}

/// Converts one row of `Bgr48` to packed u8 RGBA (B↔R swap, alpha forced to
/// `0xFF`). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgba_row_endian<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgr48_to_rgba_row::<BE>(bgr48, rgba_out, width);
}

/// LE-only wrapper around [`bgr48_to_rgba_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_rgba_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgba_row(bgr48: &[u16], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  bgr48_to_rgba_row_endian::<false>(bgr48, rgba_out, width, use_simd)
}

/// Converts one row of `Bgr48` to native-depth u16 RGB (B↔R swap, values
/// unchanged). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgb_u16_row_endian<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgr48_to_rgb_u16_row::<BE>(bgr48, rgb_out, width);
}

/// LE-only wrapper around [`bgr48_to_rgb_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_rgb_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgb_u16_row(bgr48: &[u16], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  bgr48_to_rgb_u16_row_endian::<false>(bgr48, rgb_out, width, use_simd)
}

/// Converts one row of `Bgr48` to native-depth u16 RGBA (B↔R swap, alpha
/// forced to `0xFFFF`). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgba_u16_row_endian<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgr48_to_rgba_u16_row::<BE>(bgr48, rgba_out, width);
}

/// LE-only wrapper around [`bgr48_to_rgba_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_rgba_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_rgba_u16_row(bgr48: &[u16], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  bgr48_to_rgba_u16_row_endian::<false>(bgr48, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Bgr48` source. Narrows to u8 RGB via
/// `bgr48_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgr48_to_luma_row_endian<const BE: bool>(
  bgr48: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  bgr48_to_rgb_row_endian::<BE>(bgr48, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`bgr48_to_luma_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_luma_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_luma_row(
  bgr48: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  bgr48_to_luma_row_endian::<false>(
    bgr48,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Bgr48` source. Narrows to u8 RGB via
/// `bgr48_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgr48_to_luma_u16_row_endian<const BE: bool>(
  bgr48: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  bgr48_to_rgb_row_endian::<BE>(bgr48, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`bgr48_to_luma_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_luma_u16_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_luma_u16_row(
  bgr48: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  bgr48_to_luma_u16_row_endian::<false>(
    bgr48,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Bgr48` source. Narrows to u8 RGB via
/// `bgr48_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgr48_to_hsv_row_endian<const BE: bool>(
  bgr48: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgb48_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgr48.len() >= in_min, "bgr48 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  bgr48_to_rgb_row_endian::<BE>(bgr48, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`bgr48_to_hsv_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgr48_to_hsv_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr48_to_hsv_row(
  bgr48: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  bgr48_to_hsv_row_endian::<false>(bgr48, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}

// =============================================================================
// Rgba64 (R, G, B, A — 4 u16 elements per pixel, source alpha real)
// =============================================================================

/// Converts one row of `Rgba64` to packed u8 RGB. Source alpha is discarded;
/// R/G/B narrowed via `>> 8`. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgb_row_endian<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba64_to_rgb_row::<BE>(rgba64, rgb_out, width);
}

/// LE-only wrapper around [`rgba64_to_rgb_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_rgb_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgb_row(rgba64: &[u16], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  rgba64_to_rgb_row_endian::<false>(rgba64, rgb_out, width, use_simd)
}

/// Converts one row of `Rgba64` to packed u8 RGBA. All 4 channels narrowed via
/// `>> 8`; source alpha passes through. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgba_row_endian<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba64_to_rgba_row::<BE>(rgba64, rgba_out, width);
}

/// LE-only wrapper around [`rgba64_to_rgba_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_rgba_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgba_row(rgba64: &[u16], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  rgba64_to_rgba_row_endian::<false>(rgba64, rgba_out, width, use_simd)
}

/// Converts one row of `Rgba64` to native-depth u16 RGB. Source alpha
/// discarded; R/G/B copied as-is. `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgb_u16_row_endian<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba64_to_rgb_u16_row::<BE>(rgba64, rgb_out, width);
}

/// LE-only wrapper around [`rgba64_to_rgb_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_rgb_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgb_u16_row(rgba64: &[u16], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  rgba64_to_rgb_u16_row_endian::<false>(rgba64, rgb_out, width, use_simd)
}

/// Converts one row of `Rgba64` to native-depth u16 RGBA (identity copy of all
/// 4 channels; source alpha preserved). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgba_u16_row_endian<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::rgba64_to_rgba_u16_row::<BE>(rgba64, rgba_out, width);
}

/// LE-only wrapper around [`rgba64_to_rgba_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_rgba_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_rgba_u16_row(rgba64: &[u16], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  rgba64_to_rgba_u16_row_endian::<false>(rgba64, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Rgba64` source. Narrows to u8 RGB via
/// `rgba64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba64_to_luma_row_endian<const BE: bool>(
  rgba64: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgba64_to_rgb_row_endian::<BE>(rgba64, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgba64_to_luma_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_luma_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_luma_row(
  rgba64: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgba64_to_luma_row_endian::<false>(
    rgba64,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Rgba64` source. Narrows to u8 RGB via
/// `rgba64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_u16_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba64_to_luma_u16_row_endian<const BE: bool>(
  rgba64: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  rgba64_to_rgb_row_endian::<BE>(rgba64, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`rgba64_to_luma_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_luma_u16_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_luma_u16_row(
  rgba64: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  rgba64_to_luma_u16_row_endian::<false>(
    rgba64,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Rgba64` source. Narrows to u8 RGB via
/// `rgba64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_hsv_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn rgba64_to_hsv_row_endian<const BE: bool>(
  rgba64: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(rgba64.len() >= in_min, "rgba64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  rgba64_to_rgb_row_endian::<BE>(rgba64, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`rgba64_to_hsv_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `rgba64_to_hsv_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgba64_to_hsv_row(
  rgba64: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  rgba64_to_hsv_row_endian::<false>(rgba64, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}

// =============================================================================
// Bgra64 (B, G, R, A — 4 u16 elements per pixel, source alpha real)
// =============================================================================

/// Converts one row of `Bgra64` to packed u8 RGB (B↔R swap, drop alpha,
/// narrow via `>> 8`). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgb_row_endian<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgb_row_bytes(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgra64_to_rgb_row::<BE>(bgra64, rgb_out, width);
}

/// LE-only wrapper around [`bgra64_to_rgb_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_rgb_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgb_row(bgra64: &[u16], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  bgra64_to_rgb_row_endian::<false>(bgra64, rgb_out, width, use_simd)
}

/// Converts one row of `Bgra64` to packed u8 RGBA (B↔R swap, all 4 channels
/// narrowed via `>> 8`; source alpha passes through). `use_simd = false` forces
/// the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgba_row_endian<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgba_row_bytes(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgra64_to_rgba_row::<BE>(bgra64, rgba_out, width);
}

/// LE-only wrapper around [`bgra64_to_rgba_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_rgba_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgba_row(bgra64: &[u16], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  bgra64_to_rgba_row_endian::<false>(bgra64, rgba_out, width, use_simd)
}

/// Converts one row of `Bgra64` to native-depth u16 RGB (B↔R swap, drop alpha,
/// values copied as-is). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgb_u16_row_endian<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgb_row_elems(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgra64_to_rgb_u16_row::<BE>(bgra64, rgb_out, width);
}

/// LE-only wrapper around [`bgra64_to_rgb_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_rgb_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgb_u16_row(bgra64: &[u16], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  bgra64_to_rgb_u16_row_endian::<false>(bgra64, rgb_out, width, use_simd)
}

/// Converts one row of `Bgra64` to native-depth u16 RGBA (B↔R swap; source
/// alpha preserved at position 3). `use_simd = false` forces the scalar path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgba_u16_row_endian<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let out_min = rgba_row_elems(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe { arch::neon::neon_bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe { arch::x86_avx512::avx512_bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
        if avx2_available() {
          unsafe { arch::x86_avx2::avx2_bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
        if sse41_available() {
          unsafe { arch::x86_sse41::sse41_bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width); }
          return;
        }
      },
      all(target_arch = "wasm32", target_feature = "simd128") => {
        unsafe { arch::wasm_simd128::wasm_bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width); }
        return;
      },
      _ => {}
    }
  }
  scalar::bgra64_to_rgba_u16_row::<BE>(bgra64, rgba_out, width);
}

/// LE-only wrapper around [`bgra64_to_rgba_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_rgba_u16_row_endian::<false>(...)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_rgba_u16_row(bgra64: &[u16], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  bgra64_to_rgba_u16_row_endian::<false>(bgra64, rgba_out, width, use_simd)
}

/// Derives 8-bit luma from one row of `Bgra64` source. Narrows to u8 RGB via
/// `bgra64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgra64_to_luma_row_endian<const BE: bool>(
  bgra64: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  bgra64_to_rgb_row_endian::<BE>(bgra64, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`bgra64_to_luma_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_luma_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_luma_row(
  bgra64: &[u16],
  luma_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  bgra64_to_luma_row_endian::<false>(
    bgra64,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives u16 luma from one row of `Bgra64` source. Narrows to u8 RGB via
/// `bgra64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_luma_u16_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgra64_to_luma_u16_row_endian<const BE: bool>(
  bgra64: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  bgra64_to_rgb_row_endian::<BE>(bgra64, rgb_scratch, width, use_simd);
  scalar::rgb_to_luma_u16_row(rgb_scratch, luma_out, width, matrix, full_range);
}

/// LE-only wrapper around [`bgra64_to_luma_u16_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_luma_u16_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_luma_u16_row(
  bgra64: &[u16],
  luma_out: &mut [u16],
  rgb_scratch: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  bgra64_to_luma_u16_row_endian::<false>(
    bgra64,
    luma_out,
    rgb_scratch,
    width,
    matrix,
    full_range,
    use_simd,
  )
}

/// Derives planar HSV from one row of `Bgra64` source. Narrows to u8 RGB via
/// `bgra64_to_rgb_row` into `rgb_scratch`, then applies `rgb_to_hsv_row`.
/// Source alpha is discarded.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bgra64_to_hsv_row_endian<const BE: bool>(
  bgra64: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let in_min = rgba64_packed_elems(width);
  let scratch_min = rgb_row_bytes(width);
  assert!(bgra64.len() >= in_min, "bgra64 row too short");
  assert!(rgb_scratch.len() >= scratch_min, "rgb_scratch too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");
  bgra64_to_rgb_row_endian::<BE>(bgra64, rgb_scratch, width, use_simd);
  scalar::rgb_to_hsv_row(rgb_scratch, h_out, s_out, v_out, width);
}

/// LE-only wrapper around [`bgra64_to_hsv_row_endian`]; preserves the pre-endian-
/// generic public signature so existing little-endian callers compile
/// unchanged. Equivalent to `bgra64_to_hsv_row_endian::<false>(...)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgra64_to_hsv_row(
  bgra64: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  rgb_scratch: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  bgra64_to_hsv_row_endian::<false>(bgra64, h_out, s_out, v_out, rgb_scratch, width, use_simd)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the 28 public packed-16-bit-RGB dispatchers.
  //!
  //! Each dispatcher's scalar fallback is exercised via `use_simd = false`.
  //! Overflow-guard tests are gated on 32-bit targets where `usize` is 32 bits.
  //!
  //! Fixtures use the `as_le_u16` helper so the kernel's `from_le` recovers
  //! the intended logical values on both LE (no-op) and BE (byte-swap) hosts.
  use super::*;

  // ---- helpers -------------------------------------------------------------

  /// Re-encode a host-native u16 slice as LE-encoded byte storage, packed back
  /// into `Vec<u16>`. Kernels called via `*_endian::<false>` recover the
  /// intended logical values via `u16::from_le` on both hosts.
  fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  /// Build a `width`-pixel Rgb48 / Bgr48 row with every channel = `val`,
  /// LE-encoded.
  fn solid_rgb48(width: usize, val: u16) -> std::vec::Vec<u16> {
    as_le_u16(&std::vec![val; width * 3])
  }

  /// Build a `width`-pixel Rgba64 / Bgra64 row with every channel = `val`,
  /// LE-encoded.
  fn solid_rgba64(width: usize, val: u16) -> std::vec::Vec<u16> {
    as_le_u16(&std::vec![val; width * 4])
  }

  // ---- Rgb48 ---------------------------------------------------------------

  #[test]
  fn rgb48_dispatcher_to_rgb_scalar_path() {
    // All-white Rgb48: each u16 channel = 0xFFFF; narrowed >> 8 = 0xFF.
    let src = solid_rgb48(4, 0xFFFF);
    let mut rgb = std::vec![0u8; 4 * 3];
    rgb48_to_rgb_row_endian::<false>(&src, &mut rgb, 4, false);
    assert!(
      rgb.iter().all(|&v| v == 0xFF),
      "expected all 0xFF, got {rgb:?}"
    );
  }

  #[test]
  fn rgb48_dispatcher_to_rgba_scalar_path() {
    let src = solid_rgb48(4, 0x1200);
    let mut rgba = std::vec![0u8; 4 * 4];
    rgb48_to_rgba_row_endian::<false>(&src, &mut rgba, 4, false);
    for px in rgba.chunks(4) {
      assert_eq!(px[0], 0x12, "R channel");
      assert_eq!(px[3], 0xFF, "alpha forced to 0xFF");
    }
  }

  #[test]
  fn rgb48_dispatcher_to_rgb_u16_scalar_path() {
    let src = solid_rgb48(4, 0xABCD);
    let mut rgb_u16 = std::vec![0u16; 4 * 3];
    rgb48_to_rgb_u16_row_endian::<false>(&src, &mut rgb_u16, 4, false);
    assert!(
      rgb_u16.iter().all(|&v| v == 0xABCD),
      "expected identity copy"
    );
  }

  #[test]
  fn rgb48_dispatcher_to_rgba_u16_scalar_path() {
    let src = solid_rgb48(4, 0x1234);
    let mut rgba_u16 = std::vec![0u16; 4 * 4];
    rgb48_to_rgba_u16_row_endian::<false>(&src, &mut rgba_u16, 4, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[0], 0x1234, "R channel");
      assert_eq!(px[3], 0xFFFF, "alpha forced to 0xFFFF");
    }
  }

  #[test]
  fn rgb48_dispatcher_to_luma_scalar_path() {
    // All-white Rgb48 (all channels = 0xFF00) → near-white luma in full-range BT.709.
    let src = solid_rgb48(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u8; 4];
    rgb48_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(y >= 240, "full-range white luma must be near 255, got {y}");
    }
  }

  #[test]
  fn rgb48_dispatcher_to_luma_u16_scalar_path() {
    let src = solid_rgb48(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u16; 4];
    rgb48_to_luma_u16_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(
        y >= 240,
        "full-range white luma_u16 must be near 255, got {y}"
      );
    }
  }

  #[test]
  fn rgb48_dispatcher_to_hsv_scalar_path() {
    // Pure red: R=0xFF00, G=0, B=0 → H=0, S=255, V≈255 in OpenCV encoding.
    let src = as_le_u16(&[0xFF00u16, 0x0000, 0x0000]); // 1 pixel
    let mut scratch = std::vec![0u8; 3];
    let mut h = std::vec![0u8; 1];
    let mut s = std::vec![0u8; 1];
    let mut v = std::vec![0u8; 1];
    rgb48_to_hsv_row_endian::<false>(&src, &mut h, &mut s, &mut v, &mut scratch, 1, false);
    assert_eq!(h[0], 0, "H for pure red must be 0");
    assert_eq!(s[0], 255, "S for pure red must be 255");
    assert!(v[0] >= 254, "V for pure red must be near 255, got {}", v[0]);
  }

  // ---- Bgr48 ---------------------------------------------------------------

  #[test]
  fn bgr48_dispatcher_to_rgb_scalar_path() {
    // Bgr48 pixel [B=0x1100, G=0x2200, R=0x3300] → rgb [R=0x33, G=0x22, B=0x11].
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300]);
    let mut rgb = [0u8; 3];
    bgr48_to_rgb_row_endian::<false>(&src, &mut rgb, 1, false);
    assert_eq!(rgb[0], 0x33, "R");
    assert_eq!(rgb[1], 0x22, "G");
    assert_eq!(rgb[2], 0x11, "B");
  }

  #[test]
  fn bgr48_dispatcher_to_rgba_scalar_path() {
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300]);
    let mut rgba = [0u8; 4];
    bgr48_to_rgba_row_endian::<false>(&src, &mut rgba, 1, false);
    assert_eq!(rgba[0], 0x33, "R");
    assert_eq!(rgba[3], 0xFF, "alpha forced to 0xFF");
  }

  #[test]
  fn bgr48_dispatcher_to_rgb_u16_scalar_path() {
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333]); // B, G, R
    let mut rgb_u16 = [0u16; 3];
    bgr48_to_rgb_u16_row_endian::<false>(&src, &mut rgb_u16, 1, false);
    assert_eq!(rgb_u16[0], 0x3333, "R (from position 2)");
    assert_eq!(rgb_u16[1], 0x2222, "G");
    assert_eq!(rgb_u16[2], 0x1111, "B (from position 0)");
  }

  #[test]
  fn bgr48_dispatcher_to_rgba_u16_scalar_path() {
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333]); // B, G, R
    let mut rgba_u16 = [0u16; 4];
    bgr48_to_rgba_u16_row_endian::<false>(&src, &mut rgba_u16, 1, false);
    assert_eq!(rgba_u16[0], 0x3333, "R");
    assert_eq!(rgba_u16[3], 0xFFFF, "alpha forced to 0xFFFF");
  }

  #[test]
  fn bgr48_dispatcher_to_luma_scalar_path() {
    let src = solid_rgb48(4, 0xFF00); // all channels = 0xFF00
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u8; 4];
    bgr48_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(y >= 240, "full-range white luma must be near 255, got {y}");
    }
  }

  #[test]
  fn bgr48_dispatcher_to_luma_u16_scalar_path() {
    let src = solid_rgb48(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u16; 4];
    bgr48_to_luma_u16_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(y >= 240, "full-range white luma_u16 must be near 255");
    }
  }

  #[test]
  fn bgr48_dispatcher_to_hsv_scalar_path() {
    // Pure blue in Bgr48 layout: B=0xFF00, G=0, R=0.
    // After B↔R swap → rgb=[R=0, G=0, B=0xFF] → H=120, S=255, V≈255.
    let src = as_le_u16(&[0xFF00u16, 0x0000, 0x0000]); // B, G, R
    let mut scratch = std::vec![0u8; 3];
    let mut h = std::vec![0u8; 1];
    let mut s = std::vec![0u8; 1];
    let mut v = std::vec![0u8; 1];
    bgr48_to_hsv_row_endian::<false>(&src, &mut h, &mut s, &mut v, &mut scratch, 1, false);
    assert_eq!(h[0], 120, "H for pure blue must be 120 in OpenCV encoding");
    assert_eq!(s[0], 255, "S for pure blue must be 255");
    assert!(
      v[0] >= 254,
      "V for pure blue must be near 255, got {}",
      v[0]
    );
  }

  // ---- Rgba64 --------------------------------------------------------------

  #[test]
  fn rgba64_dispatcher_to_rgb_scalar_path() {
    // Source alpha should be dropped; R/G/B narrowed.
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xDEAD]); // R, G, B, A
    let mut rgb = [0u8; 3];
    rgba64_to_rgb_row_endian::<false>(&src, &mut rgb, 1, false);
    assert_eq!(rgb[0], 0x11, "R");
    assert_eq!(rgb[1], 0x22, "G");
    assert_eq!(rgb[2], 0x33, "B");
  }

  #[test]
  fn rgba64_dispatcher_to_rgba_scalar_path() {
    // Source alpha 0xABCD → 0xAB after >> 8.
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xABCD]);
    let mut rgba = [0u8; 4];
    rgba64_to_rgba_row_endian::<false>(&src, &mut rgba, 1, false);
    assert_eq!(rgba[3], 0xAB, "source alpha depth-converted >> 8");
  }

  #[test]
  fn rgba64_dispatcher_to_rgb_u16_scalar_path() {
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xDEAD]);
    let mut rgb_u16 = [0u16; 3];
    rgba64_to_rgb_u16_row_endian::<false>(&src, &mut rgb_u16, 1, false);
    assert_eq!(rgb_u16[0], 0x1111, "R");
    assert_eq!(rgb_u16[1], 0x2222, "G");
    assert_eq!(rgb_u16[2], 0x3333, "B");
  }

  #[test]
  fn rgba64_dispatcher_to_rgba_u16_scalar_path() {
    // Identity copy; source alpha preserved.
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xABCD]);
    let mut rgba_u16 = [0u16; 4];
    rgba64_to_rgba_u16_row_endian::<false>(&src, &mut rgba_u16, 1, false);
    assert_eq!(rgba_u16[0], 0x1111, "R");
    assert_eq!(rgba_u16[3], 0xABCD, "source alpha preserved");
  }

  #[test]
  fn rgba64_dispatcher_to_luma_scalar_path() {
    // All-white Rgba64 (alpha irrelevant for luma path).
    let src = solid_rgba64(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u8; 4];
    rgba64_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(y >= 240, "full-range white luma must be near 255, got {y}");
    }
  }

  #[test]
  fn rgba64_dispatcher_to_luma_u16_scalar_path() {
    let src = solid_rgba64(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u16; 4];
    rgba64_to_luma_u16_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(
        y >= 240,
        "full-range white luma_u16 must be near 255, got {y}"
      );
    }
  }

  #[test]
  fn rgba64_dispatcher_to_hsv_scalar_path() {
    // Pure green Rgba64: R=0, G=0xFF00, B=0, A=anything → H=60, S=255, V≈255.
    let src = as_le_u16(&[0x0000u16, 0xFF00, 0x0000, 0x1234]);
    let mut scratch = std::vec![0u8; 3];
    let mut h = std::vec![0u8; 1];
    let mut s = std::vec![0u8; 1];
    let mut v = std::vec![0u8; 1];
    rgba64_to_hsv_row_endian::<false>(&src, &mut h, &mut s, &mut v, &mut scratch, 1, false);
    assert_eq!(h[0], 60, "H for pure green must be 60 in OpenCV encoding");
    assert_eq!(s[0], 255, "S for pure green must be 255");
    assert!(
      v[0] >= 254,
      "V for pure green must be near 255, got {}",
      v[0]
    );
  }

  // ---- Bgra64 --------------------------------------------------------------

  #[test]
  fn bgra64_dispatcher_to_rgb_scalar_path() {
    // Bgra64: B=0x1100, G=0x2200, R=0x3300, A=0xDEAD → RGB [R=0x33, G=0x22, B=0x11].
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xDEAD]);
    let mut rgb = [0u8; 3];
    bgra64_to_rgb_row_endian::<false>(&src, &mut rgb, 1, false);
    assert_eq!(rgb[0], 0x33, "R");
    assert_eq!(rgb[1], 0x22, "G");
    assert_eq!(rgb[2], 0x11, "B");
  }

  #[test]
  fn bgra64_dispatcher_to_rgba_scalar_path() {
    // Source alpha 0xABCD → 0xAB after >> 8; channels swapped.
    let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xABCD]);
    let mut rgba = [0u8; 4];
    bgra64_to_rgba_row_endian::<false>(&src, &mut rgba, 1, false);
    assert_eq!(rgba[0], 0x33, "R (from position 2)");
    assert_eq!(rgba[3], 0xAB, "source alpha depth-converted >> 8");
  }

  #[test]
  fn bgra64_dispatcher_to_rgb_u16_scalar_path() {
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xDEAD]); // B, G, R, A
    let mut rgb_u16 = [0u16; 3];
    bgra64_to_rgb_u16_row_endian::<false>(&src, &mut rgb_u16, 1, false);
    assert_eq!(rgb_u16[0], 0x3333, "R (from position 2)");
    assert_eq!(rgb_u16[1], 0x2222, "G");
    assert_eq!(rgb_u16[2], 0x1111, "B (from position 0)");
  }

  #[test]
  fn bgra64_dispatcher_to_rgba_u16_scalar_path() {
    let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xABCD]); // B, G, R, A
    let mut rgba_u16 = [0u16; 4];
    bgra64_to_rgba_u16_row_endian::<false>(&src, &mut rgba_u16, 1, false);
    assert_eq!(rgba_u16[0], 0x3333, "R (from position 2)");
    assert_eq!(rgba_u16[3], 0xABCD, "source alpha preserved");
  }

  #[test]
  fn bgra64_dispatcher_to_luma_scalar_path() {
    let src = solid_rgba64(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u8; 4];
    bgra64_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(y >= 240, "full-range white luma must be near 255, got {y}");
    }
  }

  #[test]
  fn bgra64_dispatcher_to_luma_u16_scalar_path() {
    let src = solid_rgba64(4, 0xFF00);
    let mut scratch = std::vec![0u8; 4 * 3];
    let mut luma = std::vec![0u16; 4];
    bgra64_to_luma_u16_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      4,
      KernelMatrix::Bt709,
      true,
      false,
    );
    for &y in &luma {
      assert!(
        y >= 240,
        "full-range white luma_u16 must be near 255, got {y}"
      );
    }
  }

  #[test]
  fn bgra64_dispatcher_to_hsv_scalar_path() {
    // Pure blue in Bgra64 layout: B=0xFF00, G=0, R=0, A=any.
    // After B↔R swap → rgb=[R=0, G=0, B=0xFF] → H=120, S=255, V≈255.
    let src = as_le_u16(&[0xFF00u16, 0x0000, 0x0000, 0x1234]);
    let mut scratch = std::vec![0u8; 3];
    let mut h = std::vec![0u8; 1];
    let mut s = std::vec![0u8; 1];
    let mut v = std::vec![0u8; 1];
    bgra64_to_hsv_row_endian::<false>(&src, &mut h, &mut s, &mut v, &mut scratch, 1, false);
    assert_eq!(h[0], 120, "H for pure blue must be 120 in OpenCV encoding");
    assert_eq!(s[0], 255, "S for pure blue must be 255");
    assert!(
      v[0] >= 254,
      "V for pure blue must be near 255, got {}",
      v[0]
    );
  }

  // ---- panic guards --------------------------------------------------------

  #[test]
  #[should_panic(expected = "rgb48 row too short")]
  fn rgb48_to_rgb_row_rejects_short_input() {
    let src = [0u16; 2]; // needs 3 for width=1
    let mut out = [0u8; 3];
    rgb48_to_rgb_row_endian::<false>(&src, &mut out, 1, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn rgb48_to_rgb_row_rejects_short_output() {
    let src = [0u16; 3];
    let mut out = [0u8; 2]; // needs 3
    rgb48_to_rgb_row_endian::<false>(&src, &mut out, 1, false);
  }

  #[test]
  #[should_panic(expected = "rgba64 row too short")]
  fn rgba64_to_rgb_row_rejects_short_input() {
    let src = [0u16; 3]; // needs 4 for width=1
    let mut out = [0u8; 3];
    rgba64_to_rgb_row_endian::<false>(&src, &mut out, 1, false);
  }

  #[test]
  #[should_panic(expected = "rgba_out row too short")]
  fn rgba64_to_rgba_row_rejects_short_output() {
    let src = [0u16; 4];
    let mut out = [0u8; 3]; // needs 4
    rgba64_to_rgba_row_endian::<false>(&src, &mut out, 1, false);
  }

  #[test]
  #[should_panic(expected = "luma_out row too short")]
  fn rgb48_to_luma_row_rejects_short_luma_output() {
    let src = [0u16; 3];
    let mut scratch = [0u8; 3];
    let mut luma: [u8; 0] = [];
    rgb48_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      1,
      KernelMatrix::Bt709,
      true,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "rgb_scratch too short")]
  fn rgb48_to_luma_row_rejects_short_scratch() {
    let src = [0u16; 3];
    let mut scratch = [0u8; 2]; // needs 3
    let mut luma = [0u8; 1];
    rgb48_to_luma_row_endian::<false>(
      &src,
      &mut luma,
      &mut scratch,
      1,
      KernelMatrix::Bt709,
      true,
      false,
    );
  }

  // ---- 32-bit width overflow guards ----------------------------------------
  //
  // On 32-bit targets (wasm32, i686) `usize` is 32 bits. A caller could pass a
  // `width` value that causes `width x 3` or `width x 4` to silently wrap to
  // a small value, making an undersized buffer pass the bound check and
  // potentially reaching unsafe SIMD loads downstream. The `rgb48_packed_elems`
  // and `rgba64_packed_elems` helpers use `checked_mul` and panic on overflow.

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_3: usize = (usize::MAX / 3) + 1;

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_4: usize = (usize::MAX / 4) + 1;

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb48_dispatcher_rejects_width_times_3_overflow() {
    let p: [u16; 0] = [];
    let mut out: [u8; 0] = [];
    rgb48_to_rgb_row_endian::<false>(&p, &mut out, OVERFLOW_WIDTH_TIMES_3, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn bgr48_dispatcher_rejects_width_times_3_overflow() {
    let p: [u16; 0] = [];
    let mut out: [u8; 0] = [];
    bgr48_to_rgb_row_endian::<false>(&p, &mut out, OVERFLOW_WIDTH_TIMES_3, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgba64_dispatcher_rejects_width_times_4_overflow() {
    let p: [u16; 0] = [];
    let mut out: [u8; 0] = [];
    rgba64_to_rgb_row_endian::<false>(&p, &mut out, OVERFLOW_WIDTH_TIMES_4, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn bgra64_dispatcher_rejects_width_times_4_overflow() {
    let p: [u16; 0] = [];
    let mut out: [u8; 0] = [];
    bgra64_to_rgb_row_endian::<false>(&p, &mut out, OVERFLOW_WIDTH_TIMES_4, false);
  }
}
