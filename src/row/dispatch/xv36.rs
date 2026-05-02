//! XV36 (Tier 5 packed YUV 4:4:4 12-bit, `AV_PIX_FMT_XV36LE`)
//! dispatchers (Ship 12b).
//!
//! Six entries: `xv36_to_{rgb,rgba}_row` (u8) and the matching
//! `_u16` variants for native-depth output, plus
//! `xv36_to_luma_row` / `xv36_to_luma_u16_row` for direct luma
//! extraction. Routes through the standard `cfg_select!` per-arch
//! block; `use_simd = false` forces scalar.
//!
//! XV36 is 4:4:4 (no chroma subsampling): each pixel is a u16
//! quadruple `[U, Y, V, A]` MSB-aligned at 12-bit (low 4 bits zero
//! per sample). Buffer length is `width × 4` u16 elements — no
//! even-width restriction.

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

/// Returns the minimum u16-element count of one packed XV36 row
/// (`width × 4`) with overflow checking. Panics if `width × 4` cannot
/// be represented as `usize` (only reachable on 32-bit targets with
/// extreme widths).
#[cfg_attr(not(tarpaulin), inline(always))]
fn xv36_packed_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize (XV36 packed row)"),
  }
}

/// Converts one row of XV36 to packed RGB (u8). See
/// [`scalar::xv36_to_rgb_or_rgba_row`] for pixel layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of XV36 to packed RGBA (u8) with `α = 0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Converts one row of XV36 to packed `u16` RGB at native 12-bit
/// depth (low-bit-packed, `[0, 4095]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(
    rgb_out.len() >= rgb_row_elems(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_rgb_u16_or_rgba_u16_row::<false>(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of XV36 to packed `u16` RGBA at native 12-bit
/// depth with `α = 4095` (12-bit opaque maximum).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(
    rgba_out.len() >= rgba_row_elems(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_rgb_u16_or_rgba_u16_row::<true>(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed XV36 buffer.
/// Y values are downshifted from 12-bit MSB-aligned to 8-bit via `>> 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::xv36_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of native-depth `u16` luma from a packed XV36
/// buffer (low-bit-packed: each `u16` carries the 12-bit Y value in
/// its low 12 bits).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xv36_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize, use_simd: bool) {
  assert!(packed.len() >= xv36_packed_elems(width), "packed row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::xv36_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::xv36_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::xv36_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::xv36_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::xv36_to_luma_u16_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::xv36_to_luma_u16_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public XV36 dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests and the scalar reference's
  //! own inline tests; this block verifies the dispatcher correctly
  //! reaches its scalar fallback when SIMD is disabled and panics on
  //! invalid inputs.
  use super::*;

  /// Pack one XV36 pixel from explicit U / Y / V / A samples (12-bit
  /// each, MSB-aligned into u16: value is `sample << 4`).
  fn pack_xv36(u: u16, y: u16, v: u16, a: u16) -> [u16; 4] {
    debug_assert!(u <= 0xFFF && y <= 0xFFF && v <= 0xFFF && a <= 0xFFF);
    [u << 4, y << 4, v << 4, a << 4]
  }

  /// Build a `Vec<u16>` XV36 row of `width` pixels with `(U, Y, V, A)`
  /// repeated. Any positive width is valid (4:4:4, no chroma subsampling).
  fn solid_xv36(width: usize, u: u16, y: u16, v: u16) -> std::vec::Vec<u16> {
    let quad = pack_xv36(u, y, v, 0);
    (0..width).flat_map(|_| quad).collect()
  }

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn xv36_dispatcher_rejects_short_packed() {
    // packed buffer has only 2*4=8 u16 elements for width=4 (needs 4*4=16).
    let packed = [0u16; 8];
    let mut rgb = [0u8; 4 * 3];
    xv36_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn xv36_dispatcher_rejects_short_output() {
    // output buffer has only 2 bytes for width=4 (needs 12).
    let packed = [0u16; 4 * 4];
    let mut rgb = [0u8; 2];
    xv36_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  fn xv36_dispatchers_route_with_simd_false() {
    // Full-range gray (Y=0x800, U=V=0x800 at 12-bit). Every dispatcher
    // should reach its scalar fallback when `use_simd = false`,
    // produce the documented gray output, and not panic.
    // 0x800 = 2048 in 12-bit; MSB-aligned: 2048 << 4 = 0x8000 in u16.
    let buf = solid_xv36(8, 0x800, 0x800, 0x800);

    // u8 RGB — full-range gray 0x800/0xFFF * 255 ≈ 128
    let mut rgb = [0u8; 8 * 3];
    xv36_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 2);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u8 RGBA — alpha = 0xFF
    let mut rgba = [0u8; 8 * 4];
    xv36_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 2);
      assert_eq!(px[3], 0xFF);
    }

    // u16 RGB at native 12-bit depth.
    let mut rgb_u16 = [0u16; 8 * 3];
    xv36_to_rgb_u16_row(&buf, &mut rgb_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgb_u16.chunks(3) {
      assert!(px[0].abs_diff(0x800) <= 4);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }

    // u16 RGBA — alpha = 4095 (12-bit opaque maximum).
    let mut rgba_u16 = [0u16; 8 * 4];
    xv36_to_rgba_u16_row(&buf, &mut rgba_u16, 8, ColorMatrix::Bt709, true, false);
    for px in rgba_u16.chunks(4) {
      assert_eq!(px[3], 0x0FFF);
    }

    // u8 luma — Y=0x800 MSB-aligned → u16 value 0x8000; >> 8 = 128.
    let mut luma = [0u8; 8];
    xv36_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, 0x80u8);
    }

    // u16 luma — low-packed 12-bit Y value: 0x8000 >> 4 = 0x800.
    let mut luma_u16 = [0u16; 8];
    xv36_to_luma_u16_row(&buf, &mut luma_u16, 8, false);
    for &y in &luma_u16 {
      assert_eq!(y, 0x800);
    }
  }

  // ---- 32-bit width × 4 overflow guard ------------------------------------
  //
  // XV36 packed rows consume `4 * width` u16 elements. Without the
  // [`xv36_packed_elems`] helper a 32-bit caller could overflow `width × 4`
  // to a small value, pass the input-side `assert!` with an undersized
  // slice, and reach unsafe SIMD loads.

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_4: usize = {
    // Smallest width whose `width × 4` overflows 32-bit `usize`.
    // `usize::MAX / 4` on 32-bit is `(2^32 - 1) / 4 = 1073741823`, so
    // `+ 1` gives `1073741824` which × 4 wraps to 0.
    (usize::MAX / 4) + 1
  };

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn xv36_dispatcher_rejects_width_times_4_overflow() {
    let p: [u16; 0] = [];
    let mut rgb: [u8; 0] = [];
    xv36_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_4,
      ColorMatrix::Bt709,
      true,
      false,
    );
  }
}
