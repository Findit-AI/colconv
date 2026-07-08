//! Runtime SIMD dispatcher for the centered (#302 phase-0.5) 2:1
//! horizontal chroma-upsample reconstruct kernels. Mirrors the crate's
//! dispatcher pattern (`dispatch::y_plane_to_luma_u16`): the highest
//! available SIMD backend wins, every tier bit-identical to the scalar
//! reference, and `use_simd == false` (`MixedSinker::with_simd(false)`)
//! bypasses the SIMD cascade entirely.

#![cfg_attr(
  not(all(feature = "std", feature = "yuv-planar", feature = "yuv-semi-planar")),
  allow(dead_code)
)]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

/// Runtime-dispatched u8 centered 2:1 horizontal chroma upsample —
/// [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
/// with a SIMD fast path.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  any(
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuva"
  )
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_center_h_row(
  c_half: &[u8],
  c_full: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  // Release-mode safety boundary before any `unsafe` SIMD dispatch; the
  // per-arch kernels only `debug_assert!` these bounds.
  assert!(c_half.len() >= width / 2, "c_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_2to1_center_h(c_half, c_full, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe { arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe { arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe { arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe { arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe { arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::chroma_upsample_2to1_center_h(c_half, c_full, width);
}

/// Runtime-dispatched u16 centered 2:1 horizontal chroma upsample —
/// [`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16)
/// with a SIMD fast path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_center_h_u16_row<const BITS: u32>(
  c_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(c_half.len() >= width / 2, "c_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_2to1_center_h_u16::<BITS>(c_half, c_full, width, big_endian);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe { arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(c_half, c_full, width, big_endian); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe { arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(c_half, c_full, width, big_endian); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe { arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(c_half, c_full, width, big_endian); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe { arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(c_half, c_full, width, big_endian); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe { arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(c_half, c_full, width, big_endian); }
        return;
      }
    },
    _ => {}
  }
  scalar::chroma_upsample_2to1_center_h_u16::<BITS>(c_half, c_full, width, big_endian);
}

/// Runtime-dispatched semi-planar P-format centered 2:1 horizontal chroma
/// upsample —
/// [`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx)
/// with a SIMD fast path.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_center_h_p0xx_row<const BITS: u32, const LOW_PACKED: bool>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, LOW_PACKED>(
      uv_half, uv_full, width, big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe { arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(uv_half, uv_full, width, big_endian); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe { arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(uv_half, uv_full, width, big_endian); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe { arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(uv_half, uv_full, width, big_endian); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe { arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(uv_half, uv_full, width, big_endian); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe { arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(uv_half, uv_full, width, big_endian); }
        return;
      }
    },
    _ => {}
  }
  scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, LOW_PACKED>(
    uv_half, uv_full, width, big_endian,
  );
}

#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
mod tests {
  use super::*;

  #[test]
  #[should_panic(expected = "c_half row too short")]
  fn dispatcher_panics_on_short_c_half() {
    let c_half = std::vec![0u8; 3];
    let mut c_full = std::vec![0u8; 8];
    chroma_upsample_2to1_center_h_row(&c_half, &mut c_full, 8, true);
  }

  #[test]
  #[should_panic(expected = "c_full row too short")]
  fn dispatcher_panics_on_short_c_full() {
    let c_half = std::vec![0u8; 4];
    let mut c_full = std::vec![0u8; 4];
    chroma_upsample_2to1_center_h_row(&c_half, &mut c_full, 8, true);
  }

  #[test]
  #[should_panic(expected = "uv_full row too short")]
  fn dispatcher_panics_on_short_uv_full() {
    let uv_half = std::vec![0u16; 8];
    let mut uv_full = std::vec![0u16; 8];
    chroma_upsample_2to1_center_h_p0xx_row::<10, false>(&uv_half, &mut uv_full, 8, false, true);
  }

  // The `use_simd == false` path and the SIMD path produce byte-identical
  // output — the per-backend equivalence tests assert the latter; here we
  // only confirm the dispatcher wires both to the same result.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn dispatcher_simd_matches_scalar() {
    let c_half: std::vec::Vec<u8> = (0..64u32)
      .map(|x| (x.wrapping_mul(37) >> 1) as u8)
      .collect();
    let w = 128;
    let mut a = std::vec![0u8; w];
    let mut b = std::vec![0u8; w];
    chroma_upsample_2to1_center_h_row(&c_half, &mut a, w, true);
    chroma_upsample_2to1_center_h_row(&c_half, &mut b, w, false);
    assert_eq!(a, b);
  }
}
