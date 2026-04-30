//! v210 (Tier 4 packed YUV 4:2:2 10-bit) dispatchers (Ship 11a).
//!
//! Six entries: `v210_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `v210_to_luma_row` / `v210_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! The per-format SIMD kernels are const-generic on `ALPHA`
//! (`v210_to_rgb_or_rgba_row::<ALPHA>` /
//! `v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`) — the public
//! dispatchers split them into RGB vs. RGBA entries by hard-wiring
//! `ALPHA = false` / `true`.

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar, v210_row_bytes},
};

/// Converts one row of v210 to packed RGB (u8). See
/// [`scalar::v210_to_rgb_or_rgba_row`] for byte layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
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
          unsafe { arch::neon::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of v210 to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
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
          unsafe { arch::neon::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of v210 to packed `u16` RGB at native 10-bit
/// depth (low-bit-packed, `[0, 1023]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgb_u16_row(
  packed: &[u8],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
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
          unsafe { arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of v210 to packed `u16` RGBA at native 10-bit
/// depth with `α = 1023` (10-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_rgba_u16_row(
  packed: &[u8],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
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
          unsafe { arch::neon::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed v210 buffer.
/// Y values are downshifted from 10-bit to 8-bit via `>> 2`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v210_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed v210
/// buffer (low-bit-packed: each `u16` carries the 10-bit Y value in
/// its low 10 bits).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn v210_to_luma_u16_row(packed: &[u8], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(
    width.is_multiple_of(2),
    "v210 requires even width (4:2:2 chroma pair)"
  );
  assert!(
    packed.len() >= v210_row_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::v210_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::v210_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::v210_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::v210_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::v210_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::v210_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public v210 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests
  //! (`src/row/arch/*/tests/v210.rs`) and the scalar reference's
  //! own inline tests; this block only verifies the dispatcher
  //! correctly reaches its scalar fallback when SIMD is disabled.
  use super::*;

  /// Build a v210 word from 12 logical samples in v210 standard
  /// order: `[Cb0, Y0, Cr0, Y1, Cb1, Y2, Cr1, Y3, Cb2, Y4, Cr2, Y5]`.
  fn pack_v210_word(samples: [u16; 12]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let w0 = (samples[0] as u32 & 0x3FF)
      | ((samples[1] as u32 & 0x3FF) << 10)
      | ((samples[2] as u32 & 0x3FF) << 20);
    let w1 = (samples[3] as u32 & 0x3FF)
      | ((samples[4] as u32 & 0x3FF) << 10)
      | ((samples[5] as u32 & 0x3FF) << 20);
    let w2 = (samples[6] as u32 & 0x3FF)
      | ((samples[7] as u32 & 0x3FF) << 10)
      | ((samples[8] as u32 & 0x3FF) << 20);
    let w3 = (samples[9] as u32 & 0x3FF)
      | ((samples[10] as u32 & 0x3FF) << 10)
      | ((samples[11] as u32 & 0x3FF) << 20);
    out[0..4].copy_from_slice(&w0.to_le_bytes());
    out[4..8].copy_from_slice(&w1.to_le_bytes());
    out[8..12].copy_from_slice(&w2.to_le_bytes());
    out[12..16].copy_from_slice(&w3.to_le_bytes());
    out
  }

  #[test]
  fn v210_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=512, U=V=512 at 10-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    let word = pack_v210_word([512; 12]);

    // u8 RGB
    let mut rgb = [0u8; 6 * 3];
    v210_to_rgb_row(&word, &mut rgb, 6, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 6 * 4];
    v210_to_rgba_row(&word, &mut rgba, 6, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 10-bit depth.
    let mut rgb_u16 = [0u16; 6 * 3];
    v210_to_rgb_u16_row(&word, &mut rgb_u16, 6, ColorMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(512) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 1023.
    let mut rgba_u16 = [0u16; 6 * 4];
    v210_to_rgba_u16_row(&word, &mut rgba_u16, 6, ColorMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 1023);
    }

    // u8 luma — Y=512 → 128 after `>> 2`.
    let mut luma = [0u8; 6];
    v210_to_luma_row(&word, &mut luma, 6, false);
    for &y in &luma {
      assert_eq!(y, (512u16 >> 2) as u8);
    }

    // u16 luma — low-packed 10-bit Y.
    let mut luma_u16 = [0u16; 6];
    v210_to_luma_u16_row(&word, &mut luma_u16, 6, false);
    for &y in &luma_u16 {
      assert_eq!(y, 512);
    }
  }
}
