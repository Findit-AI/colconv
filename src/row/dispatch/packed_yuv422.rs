//! Packed YUV 4:2:2 (8-bit) dispatchers — Tier 3 source-side support
//! for `yuyv422` / `uyvy422` / `yvyu422` (Ship 10).
//!
//! Six row-conversion entries (3 formats × {RGB, RGBA}) plus three
//! luma-extraction entries. Routes through the standard
//! `cfg_select!` per-arch block; `use_simd = false` forces scalar.

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
  ColorMatrix,
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of YUYV422 to packed RGB. See
/// [`scalar::yuyv422_to_rgb_row`] for byte layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yuyv422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuyv422_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of YUYV422 to packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yuyv422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuyv422_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of UYVY422 to packed RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyvy422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyvy422_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of UYVY422 to packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyvy422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyvy422_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of YVYU422 to packed RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yvyu422_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yvyu422_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of YVYU422 to packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yvyu422_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yvyu422_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed YUYV422 buffer.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yuyv422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::yuyv422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yuyv422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yuyv422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yuyv422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yuyv422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuyv422_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of 8-bit luma from a packed UYVY422 buffer.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyvy422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::uyvy422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyvy422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyvy422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyvy422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyvy422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyvy422_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of 8-bit luma from a packed YVYU422 buffer.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn yvyu422_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert_eq!(width & 1, 0, "packed YUV 4:2:2 requires even width");
  assert!(packed.len() >= width * 2, "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::yvyu422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::yvyu422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::yvyu422_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::yvyu422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::yvyu422_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yvyu422_to_luma_row(packed, luma_out, width);
}
