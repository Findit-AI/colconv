//! Scalar f32-source and f16-source kernels for planar GBR float formats.
//!
//! Source planes are `&[f32]` (or `&[half::f16]` widened to f32 at dispatch).
//! Nominal sample range `[0.0, 1.0]`; HDR values > 1.0 are permitted and
//! handled as follows:
//!
//! - **Integer-output paths** (`*_u8`, `*_u16`): clamped via
//!   `.clamp(0.0, 1.0)` before scaling. NaN is not clamped to a valid value —
//!   `f32::clamp(NaN, 0.0, 1.0)` returns NaN (Rust 1.50+), and the subsequent
//!   saturating `as u8` / `as u16` cast gives 0. Callers must not rely on a
//!   specific NaN result.
//! - **Lossless float-output paths** (`*_f32`): HDR, NaN,
//!   and Inf are preserved bit-exact (`gbrpf32_to_rgb_f32_row`,
//!   `gbrpf32_to_rgba_f32_row`).
//! - **f16-output paths** (`*_f16`): HDR values exceeding the f16 maximum
//!   (~65504) saturate to `f16::INFINITY` / `f16::NEG_INFINITY`. This is the
//!   documented caller-visible behaviour; callers needing full HDR range use
//!   the f32 pass-through accessors.
//!
//! # Endian support
//!
//! All `<const BE: bool>` kernels take the source planes as `&[f32]`
//! (Gbrpf32) — the same host-native typed slice on every target. Each
//! sample is loaded via `f32::to_bits()` → `u32::from_be` / `u32::from_le`
//! → `f32::from_bits` (see `load_f32`), so the on-disk byte order encoded
//! in `BE` is reinterpreted into the host-native f32 value before any
//! arithmetic. `BE = false` (LE wire format) is a no-op on LE hosts and
//! a 4-byte swap on BE hosts; `BE = true` is the inverse. The dead
//! branch is eliminated at monomorphisation. (The f16-input dispatch
//! widens `&[half::f16]` to f32 upstream of these kernels — endian
//! handling for the f16 source happens in the dispatcher, not here.)
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up),
//! MXCSR-independent. Matches the Grayf32 and Rgbf32 scalar contracts.
//!
//! # f32 → f16 rounding
//!
//! IEEE-754 round-to-nearest-even via `half::f16::from_f32` (the `half`
//! crate default). No override needed.
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the packed
//! output convention is **R, G, B** (matching `AV_PIX_FMT_RGB24`). Every
//! kernel performs this reorder.

// Kernels are not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#![cfg_attr(not(test), allow(dead_code))]

use crate::KernelMatrix;

// ---- shared helpers --------------------------------------------------------

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(y: f32) -> u8 {
  (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(y: f32) -> u16 {
  (y.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// f32 → half::f16 via IEEE-754 round-to-nearest-even.
/// HDR values exceeding f16 max (~65504) saturate to ±Inf.
#[inline(always)]
fn f32_to_f16(y: f32) -> half::f16 {
  half::f16::from_f32(y)
}

/// Load a single f32 sample from a `&[f32]` plane, target-endian aware.
///
/// The source plane is the raw on-disk / on-wire byte stream reinterpreted
/// as `&[f32]`. Each f32 read therefore picks up four bytes in **host-native**
/// order. We then convert that host-native u32 to the value the encoded
/// stream represents:
///
/// - `BE = true`: bytes on disk are big-endian → `u32::from_be` is a no-op
///   on BE hosts and a byte-swap on LE hosts.
/// - `BE = false`: bytes on disk are little-endian → `u32::from_le` is a
///   no-op on LE hosts and a byte-swap on BE hosts.
///
/// **Both** branches go through `from_be` / `from_le` so the
/// LE-data-on-BE-host case is handled correctly too. An unconditional
/// `swap_bytes` would corrupt rows on big-endian hosts (e.g. s390x).
#[inline(always)]
fn load_f32<const BE: bool>(plane: &[f32], i: usize) -> f32 {
  let raw = plane[i];
  if BE {
    f32::from_bits(u32::from_be(raw.to_bits()))
  } else {
    f32::from_bits(u32::from_le(raw.to_bits()))
  }
}

// ---- Gbrpf32 → u8 RGB ------------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **bytes**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 255]`
/// with round-half-up. Output order is **R, G, B** per pixel.
///
/// When `BE = true` each f32 element is loaded as a big-endian u32 bit
/// pattern (4-byte swap before reinterpret). `BE = false` is LE / host-native.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) -----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **bytes**
/// with constant opaque α = `0xFF`. Used for `Gbrpf32` sources (no α plane).
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = 0xFF;
  }
}

// ---- Gbrpf32 → u16 RGB -----------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`u16`**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 65535]`
/// with round-half-up (full-range).
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) ----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`u16`**
/// with constant opaque α = `0xFFFF`.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = 0xFFFF;
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`f32`**.
///
/// Lossless interleave — no clamping, no rounding. HDR values > 1.0,
/// NaN, and Inf are preserved bit-exact.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap)
/// before being written to the output. The output is always host-native f32.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = load_f32::<BE>(r, x);
    rgb_out[dst + 1] = load_f32::<BE>(g, x);
    rgb_out[dst + 2] = load_f32::<BE>(b, x);
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`f32`**
/// with α = `1.0` (opaque). Lossless — HDR, NaN, and Inf preserved.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f32::<BE>(r, x);
    rgba_out[dst + 1] = load_f32::<BE>(g, x);
    rgba_out[dst + 2] = load_f32::<BE>(b, x);
    rgba_out[dst + 3] = 1.0;
  }
}

// ---- Gbrpf32 → f16 RGB (fused narrow + interleave) -------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`half::f16`**.
///
/// Fused planar-gather, IEEE-754 round-to-nearest-even f32→f16 narrow, and
/// interleave in a single pass. HDR values exceeding the f16 maximum (~65504)
/// saturate to `half::f16::INFINITY`. Callers needing full HDR range use
/// `gbrpf32_to_rgb_f32_row` instead.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → f16 RGBA (fused narrow, α = f16(1.0)) ----------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`half::f16`**
/// with α = `half::f16::from_f32(1.0)`. HDR > ~65504 saturates to f16 ±Inf.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let one_f16 = half::f16::from_f32(1.0);
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = one_f16;
  }
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) ----------------------------

/// Derives luma (Y') from planar G/B/R `f32` rows by staging through an
/// 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The intermediate u8 RGB uses round-half-up clamping; luma is then computed
/// by `rgb_to_luma_row`. `matrix` and `full_range` control the luma weighting.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ---------------------------

/// Derives luma (Y') in `u16` from planar G/B/R `f32` rows by staging through
/// an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The u16 luma value has the same dynamic range as the u8 path (0–255), zero-
/// extended into the u16 carrier — matching the convention of packed-YUV
/// `*_to_luma_u16_row` kernels for 8-bit-equivalent sources.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → HSV (staged via RGB scratch) --------------------------------

/// Converts planar G/B/R `f32` rows to planar HSV **bytes** by staging
/// through an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// Matches OpenCV `cv2.COLOR_RGB2HSV` semantics: `H ∈ [0, 179]`, `S, V ∈
/// [0, 255]`. f32 values are clamped via `f32_to_u8` before the RGB→HSV step.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_hsv_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **bytes**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by 255
/// with round-half-up.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
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
    rgba_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_u8(load_f32::<BE>(a, x));
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`u16`**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by
/// 65535 with round-half-up.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_u16(load_f32::<BE>(a, x));
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`f32`**.
///
/// Lossless — HDR, NaN, and Inf are preserved bit-exact in all four channels
/// including α.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f32::<BE>(r, x);
    rgba_out[dst + 1] = load_f32::<BE>(g, x);
    rgba_out[dst + 2] = load_f32::<BE>(b, x);
    rgba_out[dst + 3] = load_f32::<BE>(a, x);
  }
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, source α) ---------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A`
/// **`half::f16`** with source α.
///
/// Fused narrow: all four channels converted via IEEE-754 round-to-nearest-even
/// in a single pass. HDR > ~65504 saturates to f16 ±Inf.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_f16(load_f32::<BE>(a, x));
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
