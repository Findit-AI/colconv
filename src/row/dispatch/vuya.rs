//! VUYA (FFmpeg `AV_PIX_FMT_VUYA`) row-level dispatchers (Ship 12c).
//!
//! Three entries: `vuya_to_rgb_row`, `vuya_to_rgba_row`,
//! `vuya_to_luma_row`. Routes through the standard `cfg_select!`
//! per-arch block; `use_simd = false` forces scalar.
//!
//! VUYA is 4:4:4 (no chroma subsampling): each pixel is a 4-byte
//! quadruple `[V(8), U(8), Y(8), A(8)]`. Buffer length is `width × 4`
//! bytes — no even-width restriction.
//!
//! The RGB dispatcher (`vuya_to_rgb_row`) is shared with VUYX because
//! the byte stream is bit-identical for the RGB path (alpha is
//! discarded). `vuyx_to_rgb_row` in `dispatch::vuyx` is a re-export.
//! Likewise `vuya_to_luma_row` is re-exported as `vuyx_to_luma_row`.

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

/// Returns the minimum byte count of one packed VUYA / VUYX row
/// (`width × 4`) with overflow checking. Panics if `width × 4` cannot
/// be represented as `usize` (only reachable on 32-bit targets with
/// extreme widths).
#[cfg_attr(not(tarpaulin), inline(always))]
fn vuya_packed_bytes(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize (VUYA packed row)"),
  }
}

/// Converts one row of VUYA or VUYX to packed RGB (u8). The alpha byte in
/// the source is discarded — RGB output has no alpha channel. This kernel
/// is bit-identical for both VUYA and VUYX (the A byte difference is
/// irrelevant when there is no α store), so `vuyx_to_rgb_row` in
/// `dispatch::vuyx` is a re-export of this function. `use_simd = false`
/// forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= vuya_packed_bytes(width),
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
          unsafe { arch::neon::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::vuya_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of VUYA to packed RGBA (u8). The source alpha byte at
/// offset 3 of each pixel quadruple is passed through verbatim
/// (`ALPHA_SRC = true`). For VUYX (where the A byte is padding and output
/// α should be `0xFF`), use `vuyx_to_rgba_row` instead. `use_simd = false`
/// forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(
    packed.len() >= vuya_packed_bytes(width),
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
          unsafe { arch::neon::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::vuya_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed VUYA or VUYX buffer. Y is
/// at byte offset 2 of each pixel quadruple; the V, U, and A bytes are
/// ignored. Avoids the full YUV→RGB pipeline when only luma is needed.
/// This function is shared with VUYX (`vuyx_to_luma_row` in
/// `dispatch::vuyx` re-exports it). `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert!(
    packed.len() >= vuya_packed_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::vuya_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::vuya_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::vuya_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::vuya_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::vuya_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::vuya_to_luma_row(packed, luma_out, width);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  //! Smoke tests for the public VUYA dispatchers. Walker / kernel
  //! correctness lives in the per-arch tests and the scalar reference's
  //! own inline tests; this block verifies the dispatcher correctly
  //! reaches its scalar fallback when SIMD is disabled and panics on
  //! invalid inputs.
  use super::*;

  /// Pack one VUYA pixel from explicit V / U / Y / A samples.
  fn pack_vuya(v: u8, u: u8, y: u8, a: u8) -> [u8; 4] {
    [v, u, y, a]
  }

  /// Build a `Vec<u8>` VUYA row of `width` neutral-gray pixels
  /// (V=128, U=128, Y=y_val, A=a_val). Any positive width is valid.
  fn solid_vuya(width: usize, y_val: u8, a_val: u8) -> std::vec::Vec<u8> {
    let quad = pack_vuya(128, 128, y_val, a_val);
    (0..width).flat_map(|_| quad).collect()
  }

  #[test]
  #[should_panic(expected = "packed row too short")]
  fn vuya_dispatcher_rejects_short_packed() {
    // packed buffer has only 2 × 4 = 8 bytes for width = 4 (needs 16).
    let packed = [0u8; 8];
    let mut rgb = [0u8; 4 * 3];
    vuya_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgb_out row too short")]
  fn vuya_dispatcher_rejects_short_rgb_output() {
    let packed = [0u8; 4 * 4];
    let mut rgb = [0u8; 2];
    vuya_to_rgb_row(&packed, &mut rgb, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "rgba_out row too short")]
  fn vuya_dispatcher_rejects_short_rgba_output() {
    let packed = [0u8; 4 * 4];
    let mut rgba = [0u8; 2];
    vuya_to_rgba_row(&packed, &mut rgba, 4, ColorMatrix::Bt709, true, false);
  }

  #[test]
  #[should_panic(expected = "luma_out row too short")]
  fn vuya_dispatcher_rejects_short_luma_output() {
    let packed = [0u8; 4 * 4];
    let mut luma = [0u8; 2];
    vuya_to_luma_row(&packed, &mut luma, 4, false);
  }

  #[test]
  fn vuya_dispatchers_route_with_simd_false() {
    // Full-range gray: Y=128, U=V=128. With full_range=true and neutral
    // chroma the output should be close to 128 on every RGB channel.
    let buf = solid_vuya(8, 128, 0xAB);

    // RGB — full-range gray ≈ 128
    let mut rgb = [0u8; 8 * 3];
    vuya_to_rgb_row(&buf, &mut rgb, 8, ColorMatrix::Bt709, true, false);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 2, "R channel off: {}", px[0]);
      assert_eq!(px[0], px[1], "R ≠ G");
      assert_eq!(px[1], px[2], "G ≠ B");
    }

    // RGBA — source alpha byte 0xAB must pass through verbatim (VUYA).
    let mut rgba = [0u8; 8 * 4];
    vuya_to_rgba_row(&buf, &mut rgba, 8, ColorMatrix::Bt709, true, false);
    for px in rgba.chunks(4) {
      assert!(px[0].abs_diff(128) <= 2, "R channel off: {}", px[0]);
      assert_eq!(px[3], 0xAB, "alpha must be source value for VUYA");
    }

    // Luma — Y=128 must appear directly.
    let mut luma = [0u8; 8];
    vuya_to_luma_row(&buf, &mut luma, 8, false);
    for &y in &luma {
      assert_eq!(y, 128u8, "luma must equal source Y byte");
    }
  }

  // ---- 32-bit width × 4 overflow guard ------------------------------------

  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_4: usize = {
    (usize::MAX / 4) + 1
  };

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn vuya_dispatcher_rejects_width_times_4_overflow() {
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    vuya_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_4,
      ColorMatrix::Bt709,
      true,
      false,
    );
  }
}
