//! Scalar reference kernels for MSB-aligned high-bit-depth planar GBR
//! sources (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! These are the MSB-aligned twins of the low-bit-packed
//! [`planar_gbr_high_bit`](super::planar_gbr_high_bit) family. The active
//! sample lives in the **high** `BITS` bits of each `u16` element (low
//! `16 - BITS` bits zero, FFmpeg `shift = 16 - BITS`), so sample recovery is
//! a right-shift by `16 - BITS` rather than a low-bit mask. Once recovered the
//! sample is in `[0, (1 << BITS) - 1]` — identical to the low-bit family — so
//! every downstream step (channel reorder, u8 downshift, native u16 copy,
//! luma) is the same.
//!
//! `gbr_*` kernels are const-generic over `BITS ∈ {10, 12}` (FFmpeg only ships
//! `gbrp10msb` / `gbrp12msb`) **and** `BE: bool` (endianness of the source
//! planes). No runtime branching on `BITS`: every `16 - BITS` / `BITS - 8`
//! shift is a const-eval expression resolved at monomorphisation, and the `BE`
//! branch is const-folded away.
//!
//! These formats carry no alpha plane (three planes — G, B, R), so the
//! 4-plane `gbra_*` kernels of the low-bit family have no MSB analog; the
//! `with_rgba` outputs use the **opaque** constant-alpha kernels.
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the packed output
//! convention is **R, G, B**. Every kernel performs this reorder.
//!
//! # u8 downshift
//!
//! u8-output kernels recover the sample (`>> (16 - BITS)`) then apply
//! `>> (BITS - 8)` (plain truncation, matching FFmpeg `swscale`). The net
//! effect is the high byte (`raw >> 8`), but the two-step form keeps these
//! kernels line-for-line mirrors of the low-bit family.
//!
//! # Big-endian (`BE = true`) mode
//!
//! When `BE = true` each `u16` sample is byte-swapped before the shift. The
//! swap is a compile-time branch: the `BE = false` path compiles to a no-op.

/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B` **bytes**, recovering each sample (`>> (16 - BITS)`) then
/// downshifting by `BITS - 8`.
///
/// Output order is **R, G, B** per pixel (FFmpeg `RGB24` convention).
///
/// When `BE = true` each source element is byte-swapped before processing.
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_out` has at least `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let align = 16 - BITS;
  let shift = BITS - 8;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw >> align;
    let g_val = g_raw >> align;
    let b_val = b_raw >> align;
    let dst = x * 3;
    rgb_out[dst] = (r_val >> shift) as u8;
    rgb_out[dst + 1] = (g_val >> shift) as u8;
    rgb_out[dst + 2] = (b_val >> shift) as u8;
  }
}

/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B` **`u16`** samples at native depth. Recovers each sample
/// (`>> (16 - BITS)`) — output values are in `[0, (1 << BITS) - 1]`.
///
/// When `BE = true` each source element is byte-swapped before processing.
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_u16_out` has at least `width * 3` samples.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  let align = 16 - BITS;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let dst = x * 3;
    rgb_u16_out[dst] = r_raw >> align;
    rgb_u16_out[dst + 1] = g_raw >> align;
    rgb_u16_out[dst + 2] = b_raw >> align;
  }
}

/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B, A` **bytes** with a constant **opaque** alpha (`0xFF`). Used for
/// the `Gbrp*Msb` sources (no alpha plane) when `with_rgba` is requested.
///
/// Each sample is recovered (`>> (16 - BITS)`) then downshifted by `BITS - 8`.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let align = 16 - BITS;
  let shift = BITS - 8;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw >> align;
    let g_val = g_raw >> align;
    let b_val = b_raw >> align;
    let dst = x * 4;
    rgba_out[dst] = (r_val >> shift) as u8;
    rgba_out[dst + 1] = (g_val >> shift) as u8;
    rgba_out[dst + 2] = (b_val >> shift) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Interleaves three MSB-aligned planar G/B/R `u16` rows into packed
/// `R, G, B, A` **`u16`** samples with a constant **opaque** alpha
/// (`(1 << BITS) - 1`). Used for the `Gbrp*Msb` sources (no alpha plane) when
/// `with_rgba_u16` is requested. Recovers each sample (`>> (16 - BITS)`).
///
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let align = 16 - BITS;
  let opaque: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let dst = x * 4;
    rgba_u16_out[dst] = r_raw >> align;
    rgba_u16_out[dst + 1] = g_raw >> align;
    rgba_u16_out[dst + 2] = b_raw >> align;
    rgba_u16_out[dst + 3] = opaque;
  }
}

/// Derives luma (Y') from three MSB-aligned planar G/B/R `u16` rows directly
/// at native bit depth, avoiding the 256-level banding that would result from
/// staging through u8.
///
/// Samples are recovered (`>> (16 - BITS)`) into `[0, (1 << BITS) - 1]`, then
/// fed into the same Q15 / i64 luma path as the low-bit family.
///
/// `full_range = true` → Y' ∈ `[0, (1 << BITS) - 1]` (full).
/// `full_range = false` → Y' ∈ `[16 << (BITS - 8), 235 << (BITS - 8)]`
/// (limited / studio swing).
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_luma_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: crate::KernelMatrix,
  full_range: bool,
) {
  const { assert!(matches!(BITS, 10 | 12), "BITS must be 10 or 12") };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (k_r, k_g, k_b) = super::luma_coefficients_q15(matrix);
  let k_r = k_r as i64;
  let k_g = k_g as i64;
  let k_b = k_b as i64;
  const RND: i64 = 1 << 14;
  let native_max: u16 = ((1u32 << BITS) - 1) as u16;
  let align = 16 - BITS;

  if full_range {
    for x in 0..width {
      let r_raw = if BE {
        u16::from_be(r[x])
      } else {
        u16::from_le(r[x])
      };
      let g_raw = if BE {
        u16::from_be(g[x])
      } else {
        u16::from_le(g[x])
      };
      let b_raw = if BE {
        u16::from_be(b[x])
      } else {
        u16::from_le(b[x])
      };
      let rv = (r_raw >> align) as i64;
      let gv = (g_raw >> align) as i64;
      let bv = (b_raw >> align) as i64;
      let y = ((k_r * rv + k_g * gv + k_b * bv + RND) >> 15) as i32;
      luma_out[x] = y.clamp(0, native_max as i32) as u16;
    }
  } else {
    // Limited-range luma at native depth — mirrors the low-bit family's
    // `gbr_to_luma_u16_high_bit_row`. See that kernel for the derivation of
    // the exact `range / native_max` ratio (i64 throughout).
    let y_off = (16i64) << (BITS - 8);
    let range = (219i64) << (BITS - 8);
    let native_max_i64 = native_max as i64;
    let y_max = (235i64) << (BITS - 8);
    let y_min = y_off;
    for x in 0..width {
      let r_raw = if BE {
        u16::from_be(r[x])
      } else {
        u16::from_le(r[x])
      };
      let g_raw = if BE {
        u16::from_be(g[x])
      } else {
        u16::from_le(g[x])
      };
      let b_raw = if BE {
        u16::from_be(b[x])
      } else {
        u16::from_le(b[x])
      };
      let rv = (r_raw >> align) as i64;
      let gv = (g_raw >> align) as i64;
      let bv = (b_raw >> align) as i64;
      let y_full = (k_r * rv + k_g * gv + k_b * bv + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, native_max_i64);
      let y_lim = y_off + (y_full_clamped * range + native_max_i64 / 2) / native_max_i64;
      luma_out[x] = y_lim.clamp(y_min, y_max) as u16;
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
