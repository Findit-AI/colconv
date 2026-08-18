//! Runtime SIMD dispatchers for MSB-aligned high-bit planar GBR sources
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of [`planar_gbr_high_bit`](super::planar_gbr_high_bit).
//! These formats carry no alpha plane (three planes — G, B, R), so only the
//! 3-plane kernel variants exist, all const-generic over `BITS ∈ {10, 12}`
//! and `BE` (big-endian input when `true`):
//! - [`gbr_to_rgb_msb_row`] — interleave G/B/R → packed `R, G, B` bytes.
//! - [`gbr_to_rgb_u16_msb_row`] — interleave G/B/R → packed `R, G, B` u16.
//! - [`gbr_to_rgba_opaque_msb_row`] — interleave G/B/R → packed
//!   `R, G, B, 0xFF` bytes (opaque α).
//! - [`gbr_to_rgba_opaque_u16_msb_row`] — same, u16 output with
//!   `(1 << BITS) - 1` opaque α.
//! - [`gbr_to_luma_u16_msb_row`] — native-precision luma from planar
//!   G/B/R u16 inputs; scalar-only (mirrors the low-bit family).

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

// 1. G/B/R → packed R,G,B  (u8 output).
/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B` **bytes**. Recovers each sample (`>> (16 - BITS)`) then downshifts
/// by `BITS - 8`. `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgb_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
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
          unsafe { arch::neon::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe { arch::x86_avx512::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe { arch::wasm_simd128::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr_to_rgb_msb_row::<BITS, BE>(g, b, r, rgb_out, width);
}

// 2. G/B/R → packed R,G,B  (u16 output, native depth).
/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B` **u16** elements. Recovers each sample (`>> (16 - BITS)`); values
/// stay in `[0, (1 << BITS) - 1]`. `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgb_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
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
          unsafe {
            arch::neon::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr_to_rgb_u16_msb_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
}

// 3. G/B/R → packed R,G,B,0xFF  (u8 output, opaque α).
/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B, A` **bytes** with constant α = `0xFF`. `use_simd = false` forces
/// scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgba_opaque_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr_to_rgba_opaque_msb_row::<BITS, BE>(g, b, r, rgba_out, width);
}

// 4. G/B/R → packed R,G,B,(1<<BITS)-1  (u16 output, opaque α).
/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B, A` **u16** elements with constant α = `(1 << BITS) - 1`.
/// `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgba_opaque_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(rgba_u16_out.len() >= out_min, "rgba_u16_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(g, b, r, rgba_u16_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(
              g, b, r, rgba_u16_out, width,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(
              g, b, r, rgba_u16_out, width,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(
              g, b, r, rgba_u16_out, width,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(
              g, b, r, rgba_u16_out, width,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(g, b, r, rgba_u16_out, width);
}

// 5. G/B/R → luma Y'  (u16 output, native depth). Scalar-only (mirrors the
// low-bit family).
/// Derives luma (Y') from three MSB-aligned planar G/B/R `u16` rows at native
/// bit depth. Scalar-only — the `use_simd` flag is accepted for signature
/// consistency with the rest of the row dispatcher family.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbr_to_luma_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  _use_simd: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  scalar::gbr_to_luma_u16_msb_row::<BITS, BE>(g, b, r, luma_out, width, matrix, full_range);
}
