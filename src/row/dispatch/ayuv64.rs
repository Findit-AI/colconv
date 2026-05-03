//! AYUV64 (FFmpeg `AV_PIX_FMT_AYUV64LE`) row-level dispatchers
//! (Ship 12d, Task 9).
//!
//! Six entries: `ayuv64_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `ayuv64_to_luma_row` / `ayuv64_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! AYUV64 is 4:4:4 (no chroma subsampling): each pixel is a u16
//! quadruple `[A(16), Y(16), U(16), V(16)]`. All channels are 16-bit
//! native — no padding bits, no shift required. Buffer length is
//! `width × 4` u16 elements — no even-width restriction.
//!
//! Source α is real (depth-converted u16 → u8 via `>> 8` for u8 RGBA;
//! written direct as u16 for u16 RGBA).

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
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar},
};

/// Returns the minimum u16-element count of one packed AYUV64 row
/// (`width × 4`) with overflow checking. Panics if `width × 4` cannot
/// be represented as `usize` (only reachable on 32-bit targets with
/// extreme widths).
#[cfg_attr(not(tarpaulin), inline(always))]
fn ayuv64_packed_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize (AYUV64 packed row)"),
  }
}

/// Converts one row of AYUV64 to packed RGB (u8). Source α is discarded.
/// See [`scalar::ayuv64_to_rgb_or_rgba_row`] for pixel layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
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
          unsafe { arch::neon::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of AYUV64 to packed RGBA (u8). The source A u16 at slot 0
/// of each pixel quadruple is depth-converted to u8 via `>> 8`. `use_simd =
/// false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
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
          unsafe { arch::neon::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of AYUV64 to packed `u16` RGB at native 16-bit
/// depth. Source α is discarded. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
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
          unsafe { arch::neon::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_rgb_u16_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of AYUV64 to packed `u16` RGBA at native 16-bit
/// depth. The source A u16 at slot 0 of each pixel quadruple is written
/// direct (no conversion). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
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
          unsafe { arch::neon::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_rgba_u16_row(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed AYUV64 buffer. Y is at slot 1
/// of each pixel quadruple; extracted via `>> 8` (high byte). `use_simd =
/// false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::ayuv64_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed AYUV64 buffer.
/// Y is at slot 1 of each pixel quadruple; written direct (no shift — 16-bit
/// native). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(
    packed.len() >= ayuv64_packed_elems(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::ayuv64_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::ayuv64_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::ayuv64_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::ayuv64_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::ayuv64_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::ayuv64_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public AYUV64 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests and the scalar reference's
  //! own inline tests; this block verifies the dispatcher correctly
  //! reaches its scalar fallback when SIMD is disabled and panics on
  //! invalid inputs.
  use super::*;

  /// Pack one AYUV64 pixel from explicit A / Y / U / V samples (16-bit
  /// native, no shift required).
  fn pack_ayuv64(a: u16, y: u16, u: u16, v: u16) -> [u16; 4] {
    [a, y, u, v]
  }

  /// Build a `Vec<u16>` AYUV64 row of `width` pixels with neutral
  /// chroma (U=V=32768) and the given Y / alpha values. Any positive
  /// width is valid (4:4:4, no chroma subsampling).
  fn solid_ayuv64(width: usize, y: u16, a: u16) -> std::vec::Vec<u16> {
    let quad = pack_ayuv64(a, y, 32768, 32768);
    (0..width).flat_map(|_| quad).collect()
  }

  // ---- panic guards -------------------------------------------------------

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn ayuv64_dispatcher_rejects_short_packed() {
    // packed buffer has only 2×4=8 u16 elements for width=4 (needs 4×4=16).
    let packed = [0u16; 8];
    let mut rgb = [0u8; 4 * 3];
    ayuv64_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn ayuv64_dispatcher_rejects_short_rgb_output() {
    let packed = [0u16; 4 * 4];
    let mut rgb = [0u8; 2];
    ayuv64_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgba_out row too short")]
  fn ayuv64_dispatcher_rejects_short_rgba_output() {
    let packed = [0u16; 4 * 4];
    let mut rgba = [0u8; 2];
    ayuv64_to_rgba_row(&packed, &mut rgba, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn ayuv64_dispatcher_rejects_short_rgb_u16_output() {
    let packed = [0u16; 4 * 4];
    let mut rgb = [0u16; 2];
    ayuv64_to_rgb_u16_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgba_out row too short")]
  fn ayuv64_dispatcher_rejects_short_rgba_u16_output() {
    let packed = [0u16; 4 * 4];
    let mut rgba = [0u16; 2];
    ayuv64_to_rgba_u16_row(&packed, &mut rgba, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "luma_out row too short")]
  fn ayuv64_dispatcher_rejects_short_luma_output() {
    let packed = [0u16; 4 * 4];
    let mut luma = [0u8; 2];
    ayuv64_to_luma_row(&packed, &mut luma, 4, false);
  }

  #[test]
  #[should_panic(expected = "luma_out row too short")]
  fn ayuv64_dispatcher_rejects_short_luma_u16_output() {
    let packed = [0u16; 4 * 4];
    let mut luma = [0u16; 2];
    ayuv64_to_luma_u16_row(&packed, &mut luma, 4, false);
  }

  // ---- functional smoke ---------------------------------------------------

  #[test]
  fn ayuv64_dispatchers_route_with_simd_false() {
    // Limited-range BT.709: Y=60160 = 235*256 is limited-range white;
    // neutral chroma U=V=32768. With use_simd=false the scalar path is
    // exercised. Source α = 0xABCD tests depth-conversion pass-through.
    let buf = solid_ayuv64(8, 60160, 0xABCD);

    // u8 RGB — limited-range white → near 255 on every channel
    let mut rgb = [0u8; 8 * 3];
    ayuv64_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, false, false);
    for px in rgb.chunks(3) {
      assert!(
        px[0].abs_diff(255) <= 2,
        "R near-white expected, got {}",
        px[0]
      );
      assert_eq!(px[0], px[1], "R ≠ G for neutral chroma");
      assert_eq!(px[1], px[2], "G ≠ B for neutral chroma");
    }

    // u8 RGBA — source α 0xABCD >> 8 = 0xAB in output α channel
    let mut rgba = [0u8; 8 * 4];
    ayuv64_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, false, false);
    for px in rgba.chunks(4) {
      assert!(
        px[0].abs_diff(255) <= 2,
        "R near-white expected, got {}",
        px[0]
      );
      assert_eq!(px[3], 0xABu8, "source α must be depth-converted (>> 8) for u8 RGBA");
    }

    // u16 RGB — near-white (65535 or close)
    let mut rgb_u16 = [0u16; 8 * 3];
    ayuv64_to_rgb_u16_row(&buf, &mut rgb_u16, 8, ColorMatrix::Bt709, false, false);
    for px in rgb_u16.chunks(3) {
      assert!(
        px[0].abs_diff(0xFFFF) <= 256,
        "R u16 near-white expected, got {}",
        px[0]
      );
      assert_eq!(px[0], px[1], "R ≠ G for neutral chroma (u16)");
      assert_eq!(px[1], px[2], "G ≠ B for neutral chroma (u16)");
    }

    // u16 RGBA — source α 0xABCD must appear direct in output α channel
    let mut rgba_u16 = [0u16; 8 * 4];
    ayuv64_to_rgba_u16_row(&buf, &mut rgba_u16, 8, ColorMatrix::Bt709, false, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(
        px[3], 0xABCDu16,
        "source α must be written direct for u16 RGBA"
      );
    }

    // u8 luma — Y=60160; >> 8 = 234 (0xEA)
    let mut luma = [0u8; 8];
    ayuv64_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, (60160u16 >> 8) as u8, "luma u8 must be Y >> 8");
    }

    // u16 luma — Y=60160 written direct
    let mut luma_u16 = [0u16; 8];
    ayuv64_to_luma_u16_row(&buf, &mut luma_u16, 8, false);
    for &y in &luma_u16 {
      assert_eq!(y, 60160u16, "luma u16 must be Y direct");
    }
  }

  // ---- 32-bit width × 4 overflow guard ------------------------------------
  //
  // AYUV64 packed rows consume `4 * width` u16 elements. Without the
  // [`ayuv64_packed_elems`] helper a 32-bit caller could overflow `width × 4`
  // to a small value, pass the input-side `assert!` with an undersized
  // slice, and reach unsafe SIMD loads.

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_4: usize = {
    // Smallest width whose `width × 4` overflows 32-bit `usize`.
    (usize::MAX / 4) + 1
  };

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn ayuv64_dispatcher_rejects_width_times_4_overflow() {
    let p: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    ayuv64_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_4,
      ColorMatrix::Bt709,
      true,
      false,
    );
  }
}
