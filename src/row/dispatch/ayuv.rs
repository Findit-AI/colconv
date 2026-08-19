//! AYUV (FFmpeg `AV_PIX_FMT_AYUV`) row-level dispatchers.
//!
//! AYUV is packed 8-bit 4:4:4 with real source alpha. Each pixel is a
//! 4-byte quadruple `[A(8), Y(8), U(8), V(8)]` — the alpha-first channel
//! re-ordering of `Vuya`. Buffer length is `width × 4` bytes — no
//! even-width restriction. The byte stream is byte-order-fixed (single
//! bytes, no endianness).
//!
//! Five entries mirror the `Vuya` family with the AYUV channel order:
//! `ayuv_to_rgb_row`, `ayuv_to_rgba_row` (source α pass-through from
//! offset 0), `ayuv_to_hsv_row`, `ayuv_to_luma_row`, and
//! `ayuv_to_luma_u16_row`. `use_simd = false` forces scalar.

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
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Minimum byte count of one packed AYUV row (`width × 4`) with overflow
/// checking. Panics if `width × 4` cannot be represented as `usize`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn ayuv_packed_bytes(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) x 4 overflows usize (AYUV packed row)"),
  }
}

/// Converts one row of AYUV to packed RGB (u8). The source alpha byte
/// (offset 0) is discarded — RGB output has no alpha channel.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv_packed_bytes(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe {
            arch::neon::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::ayuv_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of AYUV **directly** to planar HSV bytes (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding). Byte-identical to
/// `rgb_to_hsv_row(ayuv_to_rgb_row(...))` within the selected tier. The
/// alpha byte (offset 0) is dropped (HSV is colour-only).
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn ayuv_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv_packed_bytes(width),
    "packed row too short"
  );
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe {
            arch::neon::ayuv_to_hsv_row(packed, h_out, s_out, v_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::ayuv_to_hsv_row(
              packed, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::ayuv_to_hsv_row(packed, h_out, s_out, v_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::ayuv_to_hsv_row(
              packed, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::ayuv_to_hsv_row(
              packed, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::ayuv_to_hsv_row(packed, h_out, s_out, v_out, width, matrix, full_range);
}

/// Converts one row of AYUV to packed RGBA (u8). The source alpha byte at
/// offset 0 of each pixel quadruple is passed through verbatim to RGBA
/// slot 3. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv_packed_bytes(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::ayuv_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed AYUV buffer. Y is at byte
/// offset 1 of each pixel quadruple; the A, U, and V bytes are ignored.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    packed.len() >= ayuv_packed_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::ayuv_to_luma_row(packed, luma_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::ayuv_to_luma_row(packed, luma_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::ayuv_to_luma_row(packed, luma_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::ayuv_to_luma_row(packed, luma_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::ayuv_to_luma_row(packed, luma_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::ayuv_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of u16 luma (zero-extended Y bytes) from a packed AYUV
/// buffer. Y is at byte offset 1 of each pixel quadruple.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ayuv_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize, use_simd: bool) {
  assert!(
    packed.len() >= ayuv_packed_bytes(width),
    "packed row too short"
  );
  assert!(out.len() >= width, "out too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe {
            arch::neon::ayuv_to_luma_u16_row(packed, out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::ayuv_to_luma_u16_row(packed, out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::ayuv_to_luma_u16_row(packed, out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::ayuv_to_luma_u16_row(packed, out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::ayuv_to_luma_u16_row(packed, out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::ayuv_to_luma_u16_row(packed, out, width);
}
