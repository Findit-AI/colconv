//! Y216 (Tier 4 packed YUV 4:2:2 16-bit, full-range u16 quadruples)
//! dispatchers (Ship 11d).
//!
//! Six entries: `y216_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `y216_to_luma_row` / `y216_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! Unlike Y210 / Y212, Y216 uses full-width u16 samples (BITS = 16)
//! and has its own dedicated kernel family — the public dispatchers
//! call `y216_to_*` directly without a BITS const-generic.

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar, y2xx_row_elems},
};

/// Converts one row of Y216 to packed RGB (u8). See
/// [`scalar::y216_to_rgb_or_rgba_row`] for sample layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
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
          unsafe { arch::neon::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of Y216 to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
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
          unsafe { arch::neon::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of Y216 to packed `u16` RGB at native 16-bit
/// depth (full-range, `[0, 65535]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of Y216 to packed `u16` RGBA at native 16-bit
/// depth with `α = 0xFFFF` (16-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed Y216 buffer.
/// Y values are downshifted from 16-bit to 8-bit via `>> 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y216_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed Y216
/// buffer (full-range: each `u16` carries the 16-bit Y value directly).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn y216_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "Y216 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= y2xx_row_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::y216_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::y216_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::y216_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::y216_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::y216_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::y216_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public Y216 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests and the scalar reference's
  //! own inline tests; this block verifies the dispatcher correctly
  //! reaches its scalar fallback when SIMD is disabled and panics on
  //! invalid inputs.
  use super::*;

  /// Largest `width` value such that `width * 2` wraps to 0 on a
  /// 64-bit target. Used to trigger the overflow path in
  /// `y2xx_row_elems`.
  const OVERFLOW_WIDTH_TIMES_2: usize = usize::MAX / 2 + 1;

  /// Build one Y216-shaped u16 quadruple `[Y0, U, Y1, V]` with each
  /// sample as a full 16-bit value.
  fn y216_quad(y0: u16, u: u16, y1: u16, v: u16) -> [u16; 4] {
    [y0, u, y1, v]
  }

  /// Build a `Vec<u16>` Y216 row of `width` pixels with `(Y, U, V)`
  /// repeated. Width must be even.
  fn solid_y216(width: usize, y: u16, u: u16, v: u16) -> std::vec::Vec<u16> {
    let mut buf = std::vec::Vec::with_capacity(width * 2);
    for _ in 0..(width / 2) {
      buf.extend_from_slice(&y216_quad(y, u, y, v));
    }
    buf
  }

  #[test]
  fn y216_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=32768, U=V=32768 at 16-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    let buf = solid_y216(8, 32768, 32768, 32768);

    // u8 RGB
    let mut rgb = [0u8; 8 * 3];
    y216_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 8 * 4];
    y216_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 16-bit depth.
    let mut rgb_u16 = [0u16; 8 * 3];
    y216_to_rgb_u16_row(&buf, &mut rgb_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(32768) <= 4);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 0xFFFF.
    let mut rgba_u16 = [0u16; 8 * 4];
    y216_to_rgba_u16_row(&buf, &mut rgba_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 0xFFFF);
    }

    // u8 luma — Y=32768 → 128 after `>> 8`.
    let mut luma = [0u8; 8];
    y216_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, (32768u16 >> 8) as u8);
    }

    // u16 luma — full 16-bit Y value.
    let mut luma_u16 = [0u16; 8];
    y216_to_luma_u16_row(&buf, &mut luma_u16, 8, false);
    for &y in &luma_u16 {
      assert_eq!(y, 32768);
    }
  }

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn y216_dispatcher_rejects_short_packed() {
    // packed buffer has only 2 elements for width=4 (needs 8).
    let packed = [0u16; 2];
    let mut rgb = [0u8; 4 * 3];
    y216_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn y216_dispatcher_rejects_short_output() {
    // output buffer has only 2 bytes for width=4 (needs 12).
    let packed = [0u16; 8];
    let mut rgb = [0u8; 2];
    y216_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "Y216 requires even width (4:2:2 chroma pair)")]
  fn y216_dispatcher_rejects_odd_width() {
    let packed = [0u16; 6];
    let mut rgb = [0u8; 9];
    y216_to_rgb_row(&packed, &mut rgb, 3, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn y216_dispatcher_rejects_width_times_2_overflow() {
    // y2xx_row_elems(OVERFLOW_WIDTH_TIMES_2) panics with "overflows usize".
    let packed = [0u16; 4];
    let mut rgb = [0u8; 4];
    // width must be even to pass the even-width check, so use OVERFLOW_WIDTH_TIMES_2
    // which is even (usize::MAX / 2 + 1 — on 64-bit, MAX is odd so MAX/2 is even,
    // +1 makes it odd... let's use a value that is definitely even and overflows).
    // Actually OVERFLOW_WIDTH_TIMES_2 = usize::MAX/2+1; on 64-bit usize::MAX = 2^64-1,
    // MAX/2 = 2^63-1 (odd), +1 = 2^63 (even). So it IS even — good.
    y216_to_rgb_row(
      &packed,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt709,
      true,
      false,
    );
  }
}
