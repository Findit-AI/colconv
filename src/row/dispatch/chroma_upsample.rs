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
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
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
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_row(
            c_half, c_full, width,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_row(c_half, c_full, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_row(
            c_half, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_row(
            c_half, c_full, width,
          );
        }
        return;
      }
    }
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
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(
            c_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(
            c_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(
            c_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(
            c_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_u16_row::<BITS>(
            c_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
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
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(
            uv_half, uv_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<
            BITS,
            LOW_PACKED,
          >(uv_half, uv_full, width, big_endian);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<BITS, LOW_PACKED>(
            uv_half, uv_full, width, big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<
            BITS,
            LOW_PACKED,
          >(uv_half, uv_full, width, big_endian);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_2to1_center_h_p0xx_row::<
            BITS,
            LOW_PACKED,
          >(uv_half, uv_full, width, big_endian);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, LOW_PACKED>(
    uv_half, uv_full, width, big_endian,
  );
}

/// Runtime-dispatched u8 bottom-sited even-row 4:2:0 chroma upsample —
/// [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h)
/// with a SIMD fast path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottom_even_h_row(
  prev_half: &[u8],
  cur_half: &[u8],
  c_full: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(prev_half.len() >= width / 2, "prev_half row too short");
  assert!(cur_half.len() >= width / 2, "cur_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottom_even_h(prev_half, cur_half, c_full, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottom_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottom_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottom_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottom_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottom_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottom_even_h(prev_half, cur_half, c_full, width);
}

/// Runtime-dispatched u8 bottom-left-sited even-row 4:2:0 chroma upsample —
/// [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h)
/// with a SIMD fast path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottomleft_even_h_row(
  prev_half: &[u8],
  cur_half: &[u8],
  c_full: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(prev_half.len() >= width / 2, "prev_half row too short");
  assert!(cur_half.len() >= width / 2, "cur_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottomleft_even_h(prev_half, cur_half, c_full, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottomleft_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottomleft_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottomleft_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottomleft_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottomleft_even_h_row(
            prev_half, cur_half, c_full, width,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottomleft_even_h(prev_half, cur_half, c_full, width);
}

/// Runtime-dispatched u16 bottom-sited even-row 4:2:0 chroma upsample —
/// [`chroma_upsample_420_bottom_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottom_even_h_u16)
/// with a SIMD fast path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottom_even_h_u16_row<const BITS: u32>(
  prev_half: &[u16],
  cur_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(prev_half.len() >= width / 2, "prev_half row too short");
  assert!(cur_half.len() >= width / 2, "cur_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
      prev_half, cur_half, c_full, width, big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottom_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
    prev_half, cur_half, c_full, width, big_endian,
  );
}

/// Runtime-dispatched u16 bottom-left-sited even-row 4:2:0 chroma upsample —
/// [`chroma_upsample_420_bottomleft_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16)
/// with a SIMD fast path.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottomleft_even_h_u16_row<const BITS: u32>(
  prev_half: &[u16],
  cur_half: &[u16],
  c_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(prev_half.len() >= width / 2, "prev_half row too short");
  assert!(cur_half.len() >= width / 2, "cur_half row too short");
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
      prev_half, cur_half, c_full, width, big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottomleft_even_h_u16_row::<BITS>(
            prev_half, cur_half, c_full, width, big_endian,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
    prev_half, cur_half, c_full, width, big_endian,
  );
}

/// Runtime-dispatched semi-planar P-format bottom-sited even-row 4:2:0 chroma
/// upsample —
/// [`chroma_upsample_420_bottom_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottom_even_h_p0xx)
/// with a SIMD fast path. High-bit-packed (P010/P012/P016) only.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottom_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottom_even_h_p0xx::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      width,
      big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottom_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottom_even_h_p0xx::<BITS>(
    prev_uv_half,
    cur_uv_half,
    uv_full,
    width,
    big_endian,
  );
}

/// Runtime-dispatched semi-planar P-format bottom-left-sited even-row 4:2:0
/// chroma upsample —
/// [`chroma_upsample_420_bottomleft_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h_p0xx)
/// with a SIMD fast path. High-bit-packed (P010/P012/P016) only.
#[cfg(all(
  any(feature = "std", feature = "alloc"),
  feature = "yuv-planar",
  feature = "yuv-semi-planar"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottomleft_even_h_p0xx_row<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_420_bottomleft_even_h_p0xx::<BITS>(
      prev_uv_half,
      cur_uv_half,
      uv_full,
      width,
      big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 (hence SSE4.1) verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_420_bottomleft_even_h_p0xx_row::<BITS>(
            prev_uv_half,
            cur_uv_half,
            uv_full,
            width,
            big_endian,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_420_bottomleft_even_h_p0xx::<BITS>(
    prev_uv_half,
    cur_uv_half,
    uv_full,
    width,
    big_endian,
  );
}

/// Runtime-dispatched u8 full-width bottom-sited **4:4:0** even-row vertical
/// chroma upsample —
/// [`chroma_upsample_440_bottom_v`](crate::row::scalar::chroma_upsample_440_bottom_v)
/// with a SIMD fast path. 4:4:0 keeps full-width chroma, so this is the pure
/// vertical rounding-average `out[j] = (prev[j] + cur[j] + 1) >> 1` — no
/// horizontal reconstruction.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_440_bottom_v_row(
  prev: &[u8],
  cur: &[u8],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(prev.len() >= width, "prev row too short");
  assert!(cur.len() >= width, "cur row too short");
  assert!(out.len() >= width, "out row too short");

  if !use_simd {
    return scalar::chroma_upsample_440_bottom_v(prev, cur, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_440_bottom_v_row(prev, cur, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_440_bottom_v_row(
            prev, cur, out, width,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_440_bottom_v_row(prev, cur, out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_440_bottom_v_row(prev, cur, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_440_bottom_v_row(
            prev, cur, out, width,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_440_bottom_v(prev, cur, out, width);
}

/// Runtime-dispatched u16 full-width bottom-sited **4:4:0** even-row vertical
/// chroma upsample for the high-bit planar sink —
/// [`chroma_upsample_440_bottom_v_u16_wire`](crate::row::scalar::chroma_upsample_440_bottom_v_u16_wire)
/// with a SIMD fast path. `prev` / `cur` / `out` stay in the source's wire byte
/// order (`big_endian`); each backend fuses the wire → host normalization and
/// low-`BITS` mask, so the result is bit-identical per tier.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_440_bottom_v_u16_row<const BITS: u32>(
  prev: &[u16],
  cur: &[u16],
  out: &mut [u16],
  width: usize,
  big_endian: bool,
  use_simd: bool,
) {
  assert!(prev.len() >= width, "prev row too short");
  assert!(cur.len() >= width, "cur row too short");
  assert!(out.len() >= width, "out row too short");

  if !use_simd {
    return scalar::chroma_upsample_440_bottom_v_u16_wire::<BITS>(
      prev, cur, out, width, big_endian,
    );
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev, cur, out, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev, cur, out, width, big_endian,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev, cur, out, width, big_endian,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev, cur, out, width, big_endian,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_440_bottom_v_u16_row::<BITS>(
            prev, cur, out, width, big_endian,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_440_bottom_v_u16_wire::<BITS>(prev, cur, out, width, big_endian);
}

/// Runtime-dispatched u8 centered **1→4** horizontal chroma upsample (4:1:1 /
/// 4:1:0) —
/// [`chroma_upsample_4to1_center_h`](crate::row::scalar::chroma_upsample_4to1_center_h)
/// with a SIMD fast path. Each quarter-width sample expands to four output
/// columns at the centered phase; `c_quarter` carries `width.div_ceil(4)`
/// samples.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_4to1_center_h_row(
  c_quarter: &[u8],
  c_full: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(
    c_quarter.len() >= width.div_ceil(4),
    "c_quarter row too short"
  );
  assert!(c_full.len() >= width, "c_full row too short");

  if !use_simd {
    return scalar::chroma_upsample_4to1_center_h(c_quarter, c_full, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime; bounds asserted.
        unsafe {
          arch::neon::chroma_upsample::chroma_upsample_4to1_center_h_row(c_quarter, c_full, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + BW verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx512::chroma_upsample::chroma_upsample_4to1_center_h_row(
            c_quarter, c_full, width,
          );
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_avx2::chroma_upsample::chroma_upsample_4to1_center_h_row(
            c_quarter, c_full, width,
          );
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime; bounds asserted.
        unsafe {
          arch::x86_sse41::chroma_upsample::chroma_upsample_4to1_center_h_row(
            c_quarter, c_full, width,
          );
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time; bounds asserted.
        unsafe {
          arch::wasm_simd128::chroma_upsample::chroma_upsample_4to1_center_h_row(
            c_quarter, c_full, width,
          );
        }
        return;
      }
    }
    _ => {}
  }
  scalar::chroma_upsample_4to1_center_h(c_quarter, c_full, width);
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
