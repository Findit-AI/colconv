//! Runtime SIMD dispatchers for high-bit-depth planar GBR sources (Tier 10b).
//!
//! Seven kernel variants, all const-generic over `BITS ∈ {9, 10, 12, 14, 16}`
//! and `BE` (big-endian input when `true`):
//! - [`gbr_to_rgb_high_bit_row`] — interleave G/B/R → packed `R, G, B` bytes.
//! - [`gbr_to_rgb_u16_high_bit_row`] — interleave G/B/R → packed `R, G, B` u16.
//! - [`gbr_to_rgba_opaque_high_bit_row`] — interleave G/B/R → packed
//!   `R, G, B, 0xFF` bytes (opaque α).
//! - [`gbr_to_rgba_opaque_u16_high_bit_row`] — same, u16 output with
//!   `(1 << BITS) - 1` opaque α.
//! - [`gbra_to_rgba_high_bit_row`] — interleave G/B/R/A → packed
//!   `R, G, B, A` bytes (real source α, downshifted by `BITS - 8`).
//! - [`gbra_to_rgba_u16_high_bit_row`] — same, u16 output (no depth conv).
//! - [`gbr_to_luma_u16_high_bit_row`] — native-precision luma from planar
//!   G/B/R u16 inputs; scalar-only for now (SIMD can follow later).
//!
//! Each function follows the `cfg_select!` pattern from `dispatch::planar_gbr`:
//! platform arm at compile time, best available backend at runtime.

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
/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8`. `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgb_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
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
          unsafe {
            arch::neon::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgb_high_bit_row::<BITS, BE>(g, b, r, rgb_out, width);
}

// 2. G/B/R → packed R,G,B  (u16 output, native depth).
/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B` **u16**
/// elements. Samples are copied as-is (no depth conversion); values stay in
/// `[0, (1 << BITS) - 1]`. `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgb_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
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
            arch::neon::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgb_u16_out,
              width,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g, b, r, rgb_u16_out, width);
}

// 3. G/B/R → packed R,G,B,0xFF  (u8 output, opaque α).
/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A` **bytes**
/// with constant α = `0xFF`. Used by `GbrpN` for standalone `with_rgba` path.
/// `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
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
            arch::neon::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g, b, r, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(
              g, b, r, rgba_out, width,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g, b, r, rgba_out, width);
}

// 4. G/B/R → packed R,G,B,(1<<BITS)-1  (u16 output, opaque α).
/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A`
/// **u16** elements with constant α = `(1 << BITS) - 1` (native-depth
/// opaque). Used by `GbrpN` for standalone `with_rgba_u16` path.
/// `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
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
            arch::neon::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(g, b, r, rgba_u16_out, width);
}

// 5. G/B/R/A → packed R,G,B,A  (u8 output, real source α).
/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **bytes**. Alpha is downshifted by `BITS - 8` (real source α, not
/// constant). `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbra_to_rgba_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
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
          unsafe {
            arch::neon::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbra_to_rgba_high_bit_row::<BITS, BE>(g, b, r, a, rgba_out, width);
}

// 6. G/B/R/A → packed R,G,B,A  (u16 output, real source α, no depth conv).
/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **u16** elements. Alpha is copied directly without depth conversion (values
/// stay in `[0, (1 << BITS) - 1]`). `use_simd = false` forces scalar.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbra_to_rgba_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
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
          unsafe {
            arch::neon::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(g, b, r, a, rgba_u16_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified available.
          unsafe {
            arch::x86_avx512::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              a,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              a,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              a,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time enabled.
          unsafe {
            arch::wasm_simd128::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
              g,
              b,
              r,
              a,
              rgba_u16_out,
              width,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(g, b, r, a, rgba_u16_out, width);
}

// 7. G/B/R → luma Y'  (u16 output, native depth).
/// Derives luma (Y') from three planar G/B/R `u16` rows at native bit
/// depth. Scalar-only for now — SIMD can be added later.
///
/// `matrix` selects the BT.* coefficient set. `full_range = true`
/// produces Y' ∈ `[0, (1 << BITS) - 1]`; `full_range = false` produces
/// limited-range Y' ∈ `[16 << (BITS - 8), 235 << (BITS - 8)]`.
///
/// `use_simd` accepted for signature consistency with the rest of the
/// row dispatcher family. Currently no SIMD path is wired (kernel is
/// scalar-only); the flag is reserved for future backends.
/// When `BE = true`, input u16 samples are big-endian and byte-swapped first.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn gbr_to_luma_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  _use_simd: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(luma_out.len() >= width, "luma_out row too short");
  scalar::gbr_to_luma_u16_high_bit_row::<BITS, BE>(g, b, r, luma_out, width, matrix, full_range);
}
