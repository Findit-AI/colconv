//! NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) dispatchers.
//!
//! NV20 is the low-bit-packed twin of [`P210`](crate::source::P210): the
//! same Y + half-width interleaved-UV plane shape and the same per-row UV
//! layout as P010/P210, but each `u16` carries its 10 active bits in the
//! **low** 10 (`value & 0x03FF`) instead of P210's high 10 (`value >> 6`).
//! Every kernel below is therefore the `BITS = 10`, `LOW_PACKED = true`
//! monomorphization of the shared `p_n_to_*` Pn family — byte-identical
//! to P010/P210 once the de-pack extraction differs (mask vs shift). The
//! row kernels de-interleave the UV plane and nearest-neighbour upsample
//! chroma horizontally in-register, exactly like the P010/P210 path.
//!
//! These functions are endian-aware (`big_endian` / `<const BE>`): the
//! kernel byte-swaps each wire `u16` before masking the active low bits.

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
use crate::{
  KernelMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Converts one row of **NV20** (semi-planar 4:2:2, 10-bit, low-bit-
/// packed — 10 active bits in the low 10 of each `u16`) to packed
/// **8-bit** RGB.
///
/// See `scalar::nv20_to_rgb_row::<false>` for the reference. The only
/// difference from P210 (which reuses `p010_to_rgb_row_endian`) is the
/// `& 0x03FF` low-bit de-pack instead of the `>> 6` high-bit shift.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_to_rgb_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "NV20 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::neon::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::nv20_to_rgb_row::<false>(y, uv_half, rgb_out, width, matrix, full_range),
    scalar::nv20_to_rgb_row::<true>(y, uv_half, rgb_out, width, matrix, full_range)
  );
}

/// Converts one row of **NV20** to **native-depth `u16`** packed RGB
/// (10 active bits in the **low** 10 of each output `u16`, matching the
/// `yuv420p10le` convention). See `scalar::nv20_to_rgb_u16_row::<false>`.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_to_rgb_u16_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "NV20 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::neon::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::nv20_to_rgb_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range),
    scalar::nv20_to_rgb_u16_row::<true>(y, uv_half, rgb_out, width, matrix, full_range)
  );
}

/// Converts one row of **NV20** to packed **8-bit** **RGBA**. Alpha
/// defaults to `0xFF` (opaque). See `scalar::nv20_to_rgba_row::<false>`.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_to_rgba_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "NV20 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::nv20_to_rgba_row::<false>(y, uv_half, rgba_out, width, matrix, full_range),
    scalar::nv20_to_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range)
  );
}

/// Converts one row of **NV20** to **native-depth `u16`** packed
/// **RGBA** — output is low-bit-packed; alpha element is `(1 << 10) - 1`.
/// See `scalar::nv20_to_rgba_u16_row::<false>`. `use_simd = false`
/// forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_to_rgba_u16_row_endian(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "NV20 requires even width");
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::nv20_to_rgba_u16_row::<false>(y, uv_half, rgba_out, width, matrix, full_range),
    scalar::nv20_to_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range)
  );
}

/// Converts one row of **NV20** **directly** to planar HSV bytes
/// (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`), without materializing a source-width RGB row.
/// Byte-identical to `rgb_to_hsv_row(nv20_to_rgb_row_endian(...))` within
/// the selected tier. See `scalar::nv20_to_hsv_row::<false>`.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_to_hsv_row_endian(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  assert_eq!(width & 1, 0, "NV20 requires even width");
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::neon::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::nv20_to_hsv_row::<false>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range),
    scalar::nv20_to_hsv_row::<true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range)
  );
}

/// Extracts one row of **NV20** native luma: the de-packed Y range-
/// reduced to 8 bits (`(value & 0x03FF) >> 2` after host-native
/// normalization). Unlike the high-bit P-formats the `>> 8` top-byte
/// shortcut does not apply (NV20's active bits are the low 10). A trivial
/// per-element mask + shift over a contiguous Y plane, so there is no
/// SIMD variant and no `use_simd` knob. See
/// `scalar::nv20_to_luma_row::<false>`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn nv20_to_luma_row_endian(y: &[u16], luma_out: &mut [u8], width: usize, big_endian: bool) {
  assert!(y.len() >= width, "y row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  if big_endian {
    scalar::nv20_to_luma_row::<true>(y, luma_out, width);
  } else {
    scalar::nv20_to_luma_row::<false>(y, luma_out, width);
  }
}
