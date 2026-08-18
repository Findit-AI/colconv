//! Runtime SIMD dispatchers for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Five kernel variants, all const-generic over `BE` (big-endian input when
//! `true`):
//! - [`gbr32_to_rgb_row`] — G/B/R/A planar → packed `R, G, B` bytes (`>> 24`).
//! - [`gbr32_to_rgb_u16_row`] — G/B/R/A planar → packed `R, G, B` u16 (`>> 16`).
//! - [`gbra32_to_rgba_row`] — G/B/R/A planar → packed `R, G, B, A` bytes
//!   (`>> 24`, real source α).
//! - [`gbra32_to_rgba_u16_row`] — same, u16 output (`>> 16`).
//! - [`gbr32_to_luma_u16_row`] — native-precision Q15 luma from `>> 16`-narrowed
//!   G/B/R; scalar-only (mirrors the high-bit luma dispatcher).
//!
//! The four narrow/interleave kernels follow the `cfg_select!` pattern from
//! `dispatch::planar_gbr_high_bit`: platform arm at compile time, best
//! available backend at runtime, scalar fallback otherwise.

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

// 1. G/B/R/A → packed R,G,B  (u8 output, alpha dropped, `>> 24`).
/// Interleaves planar G/B/R/A `u32` rows into packed `R, G, B` **bytes**,
/// narrowing each channel `>> 24`. `use_simd = false` forces scalar.
/// When `BE = true`, input u32 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr32_to_rgb_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe { arch::x86_avx512::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe { arch::wasm_simd128::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr32_to_rgb_row::<BE>(g, b, r, rgb_out, width);
}

// 2. G/B/R/A → packed R,G,B  (u16 output, alpha dropped, `>> 16`).
/// Interleaves planar G/B/R/A `u32` rows into packed `R, G, B` **u16**
/// elements, narrowing each channel `>> 16`. `use_simd = false` forces scalar.
/// When `BE = true`, input u32 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr32_to_rgb_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgb_u16_out.len() >= out_min, "rgb_u16_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe { arch::x86_avx512::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe { arch::wasm_simd128::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr32_to_rgb_u16_row::<BE>(g, b, r, rgb_u16_out, width);
}

// 3. G/B/R/A → packed R,G,B,A  (u8 output, real source α, `>> 24`).
/// Interleaves planar G/B/R/A `u32` rows into packed `R, G, B, A` **bytes**,
/// narrowing all four channels `>> 24`. `use_simd = false` forces scalar.
/// When `BE = true`, input u32 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbra32_to_rgba_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe { arch::x86_avx512::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe { arch::wasm_simd128::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbra32_to_rgba_row::<BE>(g, b, r, a, rgba_out, width);
}

// 4. G/B/R/A → packed R,G,B,A  (u16 output, real source α, `>> 16`).
/// Interleaves planar G/B/R/A `u32` rows into packed `R, G, B, A` **u16**
/// elements, narrowing all four channels `>> 16`. `use_simd = false` forces
/// scalar. When `BE = true`, input u32 samples are big-endian and byte-swapped
/// first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbra32_to_rgba_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_u16_out.len() >= out_min, "rgba_u16_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe { arch::x86_avx512::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbra32_to_rgba_u16_row::<BE>(g, b, r, a, rgba_u16_out, width);
}

// 5. G/B/R → luma Y'  (u16 output, native precision, Q15).
/// Derives luma (Y') from three planar G/B/R `u32` rows narrowed `>> 16` to
/// native u16 precision. Scalar-only (mirrors `gbr_to_luma_u16_high_bit_row`);
/// `_use_simd` is accepted for signature consistency with the rest of the row
/// dispatcher family. When `BE = true`, input u32 samples are byte-swapped.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbr32_to_luma_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  luma_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  _use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  scalar::gbr32_to_luma_u16_row::<BE>(g, b, r, luma_out, width, matrix, full_range);
}
