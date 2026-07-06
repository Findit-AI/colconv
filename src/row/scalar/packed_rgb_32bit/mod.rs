//! Scalar reference kernels for 32-bit packed RGB / RGBA sources
//! (`AV_PIX_FMT_RGB96{LE,BE}` / `AV_PIX_FMT_RGBA128{LE,BE}`).
//!
//! Input planes are `&[u32]`. Each u32 sample is either LE- or BE-encoded on
//! disk/wire; the `<const BE: bool>` const-generic parameter selects the
//! interpretation. When `BE = false` the input is LE-encoded; when `BE = true`
//! the input is BE-encoded. In both cases each element is converted to
//! host-native byte order on load via `u32::from_le` / `u32::from_be`, which
//! are no-ops when the source byte order already matches the host. This
//! mirrors the SIMD `load_endian_u32x*` helpers and keeps the scalar reference
//! correct on big-endian hosts (s390x).
//!
//! The full-bit `u32` twins of the 16-bit [`super::packed_rgb_16bit`]
//! `Rgb48` / `Rgba64` families — all 32 bits per channel are active (no
//! stray-bit contract); the Rgba128 alpha channel is real.
//!
//! # Format layouts
//!
//! | Format   | Elements per pixel | Channel order |
//! |----------|--------------------|---------------|
//! | Rgb96    | 3                  | R, G, B       |
//! | Rgba128  | 4                  | R, G, B, A    |
//!
//! The Rgba128 alpha is real and is passed through (depth-converted) by the
//! `*_to_rgba*` outputs; the `*_to_rgb*` outputs drop it.
//!
//! # Depth-conversion convention
//!
//! - u32 → u8:  `(v >> 24) as u8`  (high-byte extraction).
//! - u32 → u16: `(v >> 16) as u16` (high-halfword extraction).

// ---- Endian load helper ------------------------------------------------------

/// Load one u32 element from a source whose byte order is selected by `BE`,
/// returning the value in host-native byte order.
///
/// `u32::from_be` / `u32::from_le` are target-endian aware: each is a no-op
/// when the source byte order matches the host, and a `swap_bytes` otherwise.
/// This matches the SIMD `load_endian_u32x*` helpers and keeps the scalar
/// reference correct on big-endian hosts (s390x).
///
/// The `if BE` branch is evaluated at compile time (monomorphization), so the
/// unused branch is entirely eliminated from the generated binary.
#[inline(always)]
fn load_u32<const BE: bool>(v: u32) -> u32 {
  if BE { u32::from_be(v) } else { u32::from_le(v) }
}

// ---- Rgb96 family (3 u32 elements per pixel: R, G, B) -----------------------

/// Rgb96 → packed u8 RGB: narrow each 32-bit channel via `>> 24`.
///
/// When `BE = true` each u32 element is byte-swapped on load so the channel
/// value is in host-native order before narrowing.
///
/// Input stride: `width * 3` u32 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb96_to_rgb_row<const BE: bool>(rgb96: &[u32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_out[dst] = (load_u32::<BE>(rgb96[src]) >> 24) as u8;
    rgb_out[dst + 1] = (load_u32::<BE>(rgb96[src + 1]) >> 24) as u8;
    rgb_out[dst + 2] = (load_u32::<BE>(rgb96[src + 2]) >> 24) as u8;
  }
}

/// Rgb96 → packed u16 RGB: narrow each 32-bit channel via `>> 16`.
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input stride: `width * 3` u32 elements, output: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb96_to_rgb_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_u16_out[dst] = (load_u32::<BE>(rgb96[src]) >> 16) as u16;
    rgb_u16_out[dst + 1] = (load_u32::<BE>(rgb96[src + 1]) >> 16) as u16;
    rgb_u16_out[dst + 2] = (load_u32::<BE>(rgb96[src + 2]) >> 16) as u16;
  }
}

/// Rgb96 → packed u8 RGBA: narrow each 32-bit channel via `>> 24`, force
/// alpha = `0xFF` (no source alpha in Rgb96).
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input stride: `width * 3` u32 elements, output: `width * 4` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb96_to_rgba_row<const BE: bool>(rgb96: &[u32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_out[dst] = (load_u32::<BE>(rgb96[src]) >> 24) as u8;
    rgba_out[dst + 1] = (load_u32::<BE>(rgb96[src + 1]) >> 24) as u8;
    rgba_out[dst + 2] = (load_u32::<BE>(rgb96[src + 2]) >> 24) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Rgb96 → packed u16 RGBA: narrow each 32-bit channel via `>> 16`, force
/// alpha = `0xFFFF` (no source alpha in Rgb96).
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input stride: `width * 3` u32 elements, output: `width * 4` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb96_to_rgba_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_u16_out[dst] = (load_u32::<BE>(rgb96[src]) >> 16) as u16;
    rgba_u16_out[dst + 1] = (load_u32::<BE>(rgb96[src + 1]) >> 16) as u16;
    rgba_u16_out[dst + 2] = (load_u32::<BE>(rgb96[src + 2]) >> 16) as u16;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Rgba128 family (4 u32 elements per pixel: R, G, B, A) ------------------

/// Rgba128 → packed u8 RGB: drop alpha, narrow each R/G/B channel via `>> 24`.
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input stride: `width * 4` u32 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgb_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = (load_u32::<BE>(rgba128[src]) >> 24) as u8;
    rgb_out[dst + 1] = (load_u32::<BE>(rgba128[src + 1]) >> 24) as u8;
    rgb_out[dst + 2] = (load_u32::<BE>(rgba128[src + 2]) >> 24) as u8;
  }
}

/// Rgba128 → packed u16 RGB: drop alpha, narrow each R/G/B channel via `>> 16`.
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input stride: `width * 4` u32 elements, output: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgb_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u16_out[dst] = (load_u32::<BE>(rgba128[src]) >> 16) as u16;
    rgb_u16_out[dst + 1] = (load_u32::<BE>(rgba128[src + 1]) >> 16) as u16;
    rgb_u16_out[dst + 2] = (load_u32::<BE>(rgba128[src + 2]) >> 16) as u16;
  }
}

/// Rgba128 → packed u8 RGBA: narrow all 4 channels via `>> 24` (source alpha
/// passes through).
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input and output stride: `width * 4` elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgba_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = (load_u32::<BE>(rgba128[i]) >> 24) as u8;
    rgba_out[i + 1] = (load_u32::<BE>(rgba128[i + 1]) >> 24) as u8;
    rgba_out[i + 2] = (load_u32::<BE>(rgba128[i + 2]) >> 24) as u8;
    rgba_out[i + 3] = (load_u32::<BE>(rgba128[i + 3]) >> 24) as u8;
  }
}

/// Rgba128 → packed u16 RGBA: narrow all 4 channels via `>> 16` (source alpha
/// passes through).
///
/// When `BE = true` each u32 element is byte-swapped on load.
///
/// Input and output stride: `width * 4` u32 / u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgba_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let i = x * 4;
    rgba_u16_out[i] = (load_u32::<BE>(rgba128[i]) >> 16) as u16;
    rgba_u16_out[i + 1] = (load_u32::<BE>(rgba128[i + 1]) >> 16) as u16;
    rgba_u16_out[i + 2] = (load_u32::<BE>(rgba128[i + 2]) >> 16) as u16;
    rgba_u16_out[i + 3] = (load_u32::<BE>(rgba128[i + 3]) >> 16) as u16;
  }
}

// ---- Native-u32 staging (no depth narrow) — 0-ULP resample tier -------------
//
// The wire row converts to **host-native `u32`** (the `BE` swap only, NO `>> 16`
// narrow) so the area / filter resample bins at full `u32` precision and each
// output narrows only afterwards — exact 0-ULP for both ranges (issue #289).
// Scalar-only (the `u32` resample tier ships no SIMD): no `use_simd` parameter.

/// Rgb96 → host-native `u32` RGB (canonical `R, G, B`, no narrow). Each of the
/// `width * 3` wire elements is swapped to host order; the channel order is
/// already canonical, so this is a pure endian normalization.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb96_to_rgb_u32_row<const BE: bool>(
  rgb96: &[u32],
  rgb_u32_out: &mut [u32],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_u32_out.len() >= width * 3, "rgb_u32_out row too short");
  for (o, &raw) in rgb_u32_out[..width * 3].iter_mut().zip(rgb96.iter()) {
    *o = load_u32::<BE>(raw);
  }
}

/// Rgba128 → host-native `u32` RGBA (canonical `R, G, B, A`, no narrow). Pure
/// endian normalization of the `width * 4` wire elements (α passes through).
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgba_u32_row<const BE: bool>(
  rgba128: &[u32],
  rgba_u32_out: &mut [u32],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(
    rgba_u32_out.len() >= width * 4,
    "rgba_u32_out row too short"
  );
  for (o, &raw) in rgba_u32_out[..width * 4].iter_mut().zip(rgba128.iter()) {
    *o = load_u32::<BE>(raw);
  }
}

/// Rgba128 → host-native `u32` RGB (drop alpha, no narrow). Reorders the `4`→`3`
/// element stride; the surviving R/G/B are swapped to host order.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba128_to_rgb_u32_row<const BE: bool>(
  rgba128: &[u32],
  rgb_u32_out: &mut [u32],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_u32_out.len() >= width * 3, "rgb_u32_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u32_out[dst] = load_u32::<BE>(rgba128[src]);
    rgb_u32_out[dst + 1] = load_u32::<BE>(rgba128[src + 1]);
    rgb_u32_out[dst + 2] = load_u32::<BE>(rgba128[src + 2]);
  }
}

// ---- Unit tests -------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests;
