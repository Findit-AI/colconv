//! Scalar reference kernels for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Input planes are four full-width `&[u32]` slices in **G, B, R, A** order
//! (FFmpeg planar convention); all 32 bits of each element are active and the
//! alpha plane is real per-pixel α. Each `u32` is either LE- or BE-encoded on
//! disk/wire; the `<const BE: bool>` const-generic selects the interpretation
//! and the value is normalised to host-native order on load via
//! `u32::from_le` / `u32::from_be` (a no-op when the source byte order matches
//! the host, a `swap_bytes` otherwise). This mirrors the SIMD
//! `load_endian_u32x*` helpers and keeps the scalar reference correct on
//! big-endian hosts (s390x).
//!
//! These are the full-bit `u32` twins of the 16-bit
//! [`super::planar_gbr_high_bit`] `Gbrap16` kernels, sharing the same packed
//! output layout (`R, G, B[, A]`, FFmpeg `RGB24` channel order) so the binning
//! / resample tails are reused unchanged at `BITS = 16`.
//!
//! # Depth-conversion convention (matches Gray32 / Rgb96 / Rgba128)
//!
//! - u32 → u8:  `(v >> 24) as u8`  (high-byte extraction).
//! - u32 → u16: `(v >> 16) as u16` (high-halfword extraction).
//!
//! `luma_u16` is computed at native u16 precision from the `>> 16`-narrowed
//! G/B/R via Q15 coefficients and i64 intermediates — the `Gbrap16`
//! `gbr_to_luma_u16_high_bit_row::<16>` formula applied to the narrowed
//! planes.

/// Load one `u32` element from a source whose byte order is selected by `BE`,
/// returning the value in host-native byte order. The `if BE` branch is
/// monomorphised away, so the unused arm is eliminated from the binary.
#[inline(always)]
fn load_u32<const BE: bool>(v: u32) -> u32 {
  if BE { u32::from_be(v) } else { u32::from_le(v) }
}

/// G/B/R/A planar `u32` → packed `R, G, B` **bytes**: drop alpha, narrow each
/// channel via `>> 24`, reorder to R, G, B.
///
/// Input stride: `width` `u32` elements per plane; output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr32_to_rgb_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = (load_u32::<BE>(r[x]) >> 24) as u8;
    rgb_out[dst + 1] = (load_u32::<BE>(g[x]) >> 24) as u8;
    rgb_out[dst + 2] = (load_u32::<BE>(b[x]) >> 24) as u8;
  }
}

/// G/B/R/A planar `u32` → packed `R, G, B` **`u16`** elements: drop alpha,
/// narrow each channel via `>> 16`, reorder to R, G, B. This is the
/// source-width staging row fed to the high-bit packed-RGB resample tail.
///
/// Input stride: `width` `u32` elements per plane; output: `width * 3` u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr32_to_rgb_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_u16_out[dst] = (load_u32::<BE>(r[x]) >> 16) as u16;
    rgb_u16_out[dst + 1] = (load_u32::<BE>(g[x]) >> 16) as u16;
    rgb_u16_out[dst + 2] = (load_u32::<BE>(b[x]) >> 16) as u16;
  }
}

/// G/B/R/A planar `u32` → packed `R, G, B, A` **bytes**: narrow all four
/// channels via `>> 24` (real source α), reorder color to R, G, B.
///
/// Input stride: `width` `u32` elements per plane; output: `width * 4` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra32_to_rgba_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = (load_u32::<BE>(r[x]) >> 24) as u8;
    rgba_out[dst + 1] = (load_u32::<BE>(g[x]) >> 24) as u8;
    rgba_out[dst + 2] = (load_u32::<BE>(b[x]) >> 24) as u8;
    rgba_out[dst + 3] = (load_u32::<BE>(a[x]) >> 24) as u8;
  }
}

/// G/B/R/A planar `u32` → packed `R, G, B, A` **`u16`** elements: narrow all
/// four channels via `>> 16` (real source α), reorder color to R, G, B. This
/// is the canonical host-native RGBA staging row fed to the alpha-aware
/// high-bit packed-RGBA resample tail.
///
/// Input stride: `width` `u32` elements per plane; output: `width * 4` u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra32_to_rgba_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let dst = x * 4;
    rgba_u16_out[dst] = (load_u32::<BE>(r[x]) >> 16) as u16;
    rgba_u16_out[dst + 1] = (load_u32::<BE>(g[x]) >> 16) as u16;
    rgba_u16_out[dst + 2] = (load_u32::<BE>(b[x]) >> 16) as u16;
    rgba_u16_out[dst + 3] = (load_u32::<BE>(a[x]) >> 16) as u16;
  }
}

/// G/B/R/A planar `u32` → packed host-native `u32` `R, G, B` (drop alpha, NO
/// narrow): reorder to R, G, B and swap each surviving channel to host order.
/// The source-width staging row fed to the native-`u32` packed-RGB resample
/// tier so binning runs at full `u32` precision (0-ULP, issue #289).
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr32_to_rgb_u32_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_u32_out: &mut [u32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u32_out.len() >= width * 3, "rgb_u32_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_u32_out[dst] = load_u32::<BE>(r[x]);
    rgb_u32_out[dst + 1] = load_u32::<BE>(g[x]);
    rgb_u32_out[dst + 2] = load_u32::<BE>(b[x]);
  }
}

/// G/B/R/A planar `u32` → packed host-native `u32` `R, G, B, A` (real α, NO
/// narrow): reorder color to R, G, B and swap every channel to host order. The
/// canonical host-native RGBA staging row fed to the native-`u32` alpha-aware
/// packed-RGBA resample tier (0-ULP, issue #289).
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra32_to_rgba_u32_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_u32_out: &mut [u32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u32_out.len() >= width * 4,
    "rgba_u32_out row too short"
  );
  for x in 0..width {
    let dst = x * 4;
    rgba_u32_out[dst] = load_u32::<BE>(r[x]);
    rgba_u32_out[dst + 1] = load_u32::<BE>(g[x]);
    rgba_u32_out[dst + 2] = load_u32::<BE>(b[x]);
    rgba_u32_out[dst + 3] = load_u32::<BE>(a[x]);
  }
}

/// Derives luma (Y') from three planar G/B/R `u32` rows at native u16
/// precision: each channel is narrowed `>> 16` to u16, then combined via Q15
/// coefficients with i64 intermediates. This is the `Gbrap16`
/// `gbr_to_luma_u16_high_bit_row::<16>` formula applied to the narrowed
/// planes (so a direct `Gbrap32` `luma_u16` is byte-identical to feeding the
/// `>> 16`-narrowed planes through the high-bit luma path).
///
/// `full_range = true` → Y' ∈ `[0, 65535]`; `full_range = false` →
/// Y' ∈ `[4096, 60160]` (`[16 << 8, 235 << 8]`, limited / studio swing).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr32_to_luma_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  luma_out: &mut [u16],
  width: usize,
  matrix: crate::KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (k_r, k_g, k_b) = super::luma_coefficients_q15(matrix);
  let k_r = k_r as i64;
  let k_g = k_g as i64;
  let k_b = k_b as i64;
  const RND: i64 = 1 << 14;
  const NATIVE_MAX: i64 = 65535;

  if full_range {
    for x in 0..width {
      let rv = (load_u32::<BE>(r[x]) >> 16) as i64;
      let gv = (load_u32::<BE>(g[x]) >> 16) as i64;
      let bv = (load_u32::<BE>(b[x]) >> 16) as i64;
      let y = (k_r * rv + k_g * gv + k_b * bv + RND) >> 15;
      luma_out[x] = y.clamp(0, NATIVE_MAX) as u16;
    }
  } else {
    // Limited-range luma at native u16 depth (BITS = 16):
    //   Y_lim = 16<<8 + clamp(Y_full) * (219<<8) / 65535
    // with round-half-up bias. See `gbr_to_luma_u16_high_bit_row` for the
    // derivation of the exact native ratio (vs the 8-bit 219/255 ratio).
    const Y_OFF: i64 = 16 << 8;
    const RANGE: i64 = 219 << 8;
    const Y_MAX: i64 = 235 << 8;
    const Y_MIN: i64 = Y_OFF;
    for x in 0..width {
      let rv = (load_u32::<BE>(r[x]) >> 16) as i64;
      let gv = (load_u32::<BE>(g[x]) >> 16) as i64;
      let bv = (load_u32::<BE>(b[x]) >> 16) as i64;
      let y_full = (k_r * rv + k_g * gv + k_b * bv + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, NATIVE_MAX);
      let y_lim = Y_OFF + (y_full_clamped * RANGE + NATIVE_MAX / 2) / NATIVE_MAX;
      luma_out[x] = y_lim.clamp(Y_MIN, Y_MAX) as u16;
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
