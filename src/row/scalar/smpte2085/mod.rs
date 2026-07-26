//! Scalar reference kernels for the **SMPTE ST 2085** (ITU-T H.273
//! `MatrixCoefficients = 11`, "Y'D'zD'x") non-affine colour decode — the
//! SMPTE ST 2085 colour-difference model for an `X'Y'Z'` signal.
//!
//! SMPTE 2085 is **not** an affine YCbCr matrix: unlike every
//! [`Coefficients::for_matrix`](super::Coefficients::for_matrix) decode (a
//! single Q15 matrix + offset), recovering RGB from `Y', D'z, D'x` requires a
//! per-channel non-linear transfer in the middle of the pipeline. It is the
//! direct analogue of [`super::ictcp`] / [`super::iptc2`]
//! (`decode → recover → EOTF → matmul → OETF → narrow`); the difference is
//! that the two 3×3 rotations of those decodes collapse to a **non-affine**
//! `Y'D'zD'x → X'Y'Z'` recovery (eqs 76-78) followed by a single fixed
//! `XYZ → RGB` gamut matrix.
//!
//! # Signal model
//!
//! SMPTE 2085 carries an **`X'Y'Z'`** signal (PQ-encoded CIE XYZ) in the
//! YCbCr container: the `X/Y/Z` tristimulus values ride in the `R/G/B` slots
//! of the H.273 forward relations. The decode reconstructs linear `XYZ`, then
//! maps it into the **BT.2020** RGB output gamut. It is defined **only** for
//! the SMPTE ST 2084 PQ transfer.
//!
//! # Decode pipeline (per pixel)
//!
//! ```text
//! Y',D'z,D'x (int)  ──dequant──▶  Y',D'z,D'x (norm, f32)
//!                   ──recover───▶  X',Y',Z'       (inverse H.273 eqs 76-78)
//!                   ──PQ EOTF───▶  X,Y,Z (linear) (per-channel SMPTE ST 2084)
//!                   ──M_XYZ→RGB─▶  RGB   (linear, BT.2020 primaries)
//!                   ──PQ OETF───▶  R'G'B'         (per-channel SMPTE ST 2084)
//!                   ──narrow────▶  u8 / u16 output
//! ```
//!
//! 1. **Dequantize** the integer `Y',D'z,D'x` samples to the normalized
//!    domain (`Y' ∈ [0,1]` luma-like, `D'z,D'x` signed chroma-like), using the
//!    identical studio/full-range scaling pixon's affine YCbCr decode uses
//!    ([`super::range_params_n`]) — `Y'D'zD'x` is carried in a YCbCr container
//!    and shares its H.273 quantization (the same encoding as [`super::ictcp`]
//!    / [`super::iptc2`], `Y'` in the Y-slot, `D'z`/`D'x` in the Cb/Cr slots).
//! 2. **Recover `X'Y'Z'`** by inverting the H.273 eq 76-78 forward relations
//!    ([`recover_xyz_prime`]): `Y' = E'_Y`, then solve for `X'` and `Z'`.
//! 3. **PQ EOTF** lifts the non-linear `X'Y'Z'` to linear `XYZ` per channel,
//!    via the BT.2100 transfer math of [`crate::resample::pq_hlg`].
//! 4. **[`M_XYZ_TO_RGB_REC2020`]** maps linear `XYZ → RGB` in the BT.2020
//!    output gamut.
//! 5. **PQ OETF** re-encodes linear `RGB → R'G'B'` per channel, yielding the
//!    `R'G'B'` display signal. This matches pixon's transfer-preserving
//!    convention (the affine YCbCr decode likewise emits `R'G'B'` in the
//!    source's transfer domain). The integer-output kernels narrow `R'G'B'`;
//!    out-of-gamut excursions clamp at the narrow.
//!
//! # Verification
//!
//! No external library is needed: the H.273 §8.3 code 11 eqs 76-78 ARE the
//! oracle. [`recover_xyz_prime`] is the exact analytic inverse of the forward
//! relations, and the end-to-end decode is pinned by an `X'Y'Z'` encode →
//! decode round-trip and the neutral-axis structural anchors in [`tests`].
//!
//! # No SIMD
//!
//! Scalar-only by design (as [`super::ictcp`] / [`super::iptc2`]): the
//! per-channel transcendental EOTF/OETF (`powf`) do not vectorize into the
//! integer-lane shape the affine YCbCr kernels use, and the transcendental
//! cost dwarfs any lane parallelism win. Routing therefore always takes the
//! scalar path for `ColorMatrix::Smpte2085`, regardless of the `use_simd`
//! hint.

use crate::{Transfer, resample::pq_hlg};

use super::bits_mask;

/// Which transfer system a [`Smpte2085`](self) source is encoded in. SMPTE
/// 2085 is defined **only** for the SMPTE ST 2084 PQ transfer (like
/// [`IptC2`](super::iptc2), and unlike [`ICtCp`](super::ictcp), which
/// additionally has an HLG rotation), so this carries a single variant; it
/// exists to mirror ICtCp's transfer-resolution gate
/// ([`IctcpTransfer::for_transfer`](super::ictcp::IctcpTransfer::for_transfer))
/// and give the dispatch a resolvable predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Smpte2085Transfer {
  /// SMPTE ST 2084 Perceptual Quantizer. The only transfer SMPTE 2085 is
  /// defined for.
  Pq,
}

impl Smpte2085Transfer {
  /// Lowercase identifier for the transfer (`"pq"`). The mandated unit-enum
  /// accessor; consumed by diagnostics (no production caller yet, hence the
  /// `dead_code` allowance).
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn as_str(self) -> &'static str {
    match self {
      Self::Pq => "pq",
    }
  }

  /// Resolves the SMPTE 2085 transfer variant from a source's signalled H.273
  /// [`Transfer`] characteristics.
  ///
  /// Returns [`Some`] **only** for [`Transfer::SmpteSt2084Pq`] → [`Self::Pq`],
  /// the single transfer SMPTE 2085 defines a derivation for. Any other
  /// transfer (including [`Transfer::Unspecified`]) returns [`None`]: SMPTE
  /// 2085 is undefined outside PQ, so the caller must fall back to the affine
  /// matrix path rather than apply an unverifiable transfer. A source tagged
  /// `ColorMatrix::Smpte2085` with no PQ transfer is malformed; the affine
  /// fallback is the defined, non-panicking policy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn for_transfer(transfer: Transfer) -> Option<Self> {
    match transfer {
      Transfer::SmpteSt2084Pq => Some(Self::Pq),
      _ => None,
    }
  }
}

// ---- SMPTE 2085 white-point normalizers (H.273 §8.3 code 11) -------------
//
// The forward color-difference relations of ITU-T H.273 V4 (07/2024) §8.3
// Table 4 code 11 (eqs 76-78), where the `R/G/B` slots carry `X/Y/Z`:
//   E'_Y = E'_G                              (the Y channel is X'Y'Z''s Y')
//   D'z  = (0.986566 · E'_B − E'_Y) / 2      (eq 77)
//   D'x  = (E'_R − 0.991902 · E'_Y) / 2      (eq 78)
// The two literals are the H.273 white-point normalizers folded into eqs
// 77/78; `recover_xyz_prime` inverts these relations exactly.

/// H.273 eq 77 `Z'` white-point normalizer (`0.986566`). In the forward
/// relation `D'z = (0.986566 · E'_B − E'_Y) / 2`, so the decode recovers
/// `Z' = (2·D'z + Y') / 0.986566`.
const WHITE_NORM_Z: f32 = 0.986_566_f32;

/// H.273 eq 78 `X'` white-point normalizer (`0.991902`). In the forward
/// relation `D'x = (E'_R − 0.991902 · E'_Y) / 2`, so the decode recovers
/// `X' = 2·D'x + 0.991902 · Y'`.
const WHITE_NORM_X: f32 = 0.991_902_f32;

/// `XYZ → RGB` (linear, **BT.2020** primaries, D65 output white). Identical
/// to `xyz12`'s `M_XYZ_TO_RGB_REC2020`, derived from the ITU-R BT.2020-2
/// chromaticities R=(0.708, 0.292), G=(0.170, 0.797), B=(0.131, 0.046),
/// W=D65=(0.3127, 0.3290) via the standard primary-scaling construction. Held
/// locally so the module is self-contained (the `xyz12` table is gated on the
/// `xyz` feature, this decode on `yuv-planar`).
///
/// f64 source values:
///
/// ```text
/// [  1.7166511880,  -0.3556707838,  -0.2533662814]
/// [ -0.6666843518,   1.6164812366,   0.0157685458]
/// [  0.0176398574,  -0.0427706133,   0.9421031212]
/// ```
const M_XYZ_TO_RGB_REC2020: [[f32; 3]; 3] = [
  [1.716_651_f32, -0.355_670_78_f32, -0.253_366_3_f32],
  [-0.666_684_3_f32, 1.616_481_2_f32, 0.015_768_547_f32],
  [0.017_639_857_f32, -0.042_770_613_f32, 0.942_103_15_f32],
];

/// Applies a 3×3 matrix to a column vector.
#[cfg_attr(not(tarpaulin), inline(always))]
fn matmul3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
  [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ]
}

/// Endian-aware load of a wire `u16` masked to the active low `BITS`
/// (the low-bit-packed `Yuv*pN` convention, `MSB = false`). Mirrors the
/// affine high-bit kernel's `load_u16` + low-bit mask.
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_sample<const BITS: u32, const BE: bool>(s: u16) -> u16 {
  let raw = if BE { u16::from_be(s) } else { u16::from_le(s) };
  raw & bits_mask::<BITS>()
}

/// Dequantizes one integer `Y',D'z,D'x` triple to the normalized SMPTE 2085
/// domain (`Y'` luma-like in `[0, 1]`, `D'z`/`D'x` signed chroma-like centred
/// on `0`), using the **same** H.273 studio/full-range scaling the affine
/// YCbCr decode applies ([`super::range_params_n`] + the `128 << (BITS-8)`
/// chroma bias). `Y'D'zD'x` shares the YCbCr integer encoding with
/// [`super::ictcp`] / [`super::iptc2`], so this is identical to those decodes'
/// dequantization.
///
/// `Y' = (DY' − y_off) / y_range`, `D'z = (DD'z − 2^(BITS-1)) / c_range`,
/// where `(y_off, y_range, c_range)` are `(0, in_max, in_max)` for full range
/// and `(16·k, 219·k, 224·k)` (`k = 2^(BITS-8)`) for studio range — the
/// unscaled, normalized form of [`super::range_params_n`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn dequant_smpte2085<const BITS: u32>(y: u16, dz: u16, dx: u16, full_range: bool) -> [f32; 3] {
  let k: i32 = 1 << (BITS - 8);
  let chroma_bias = 128 * k; // = 2^(BITS-1), the chroma zero point
  let (y_off, y_range, c_range): (i32, f32, f32) = if full_range {
    let in_max = ((1u32 << BITS) - 1) as f32;
    (0, in_max, in_max)
  } else {
    (16 * k, (219 * k) as f32, (224 * k) as f32)
  };
  [
    (y as i32 - y_off) as f32 / y_range,
    (dz as i32 - chroma_bias) as f32 / c_range,
    (dx as i32 - chroma_bias) as f32 / c_range,
  ]
}

/// Recovers the normalized `X'Y'Z'` signal from a normalized `Y',D'z,D'x`
/// triple — the exact analytic inverse of the H.273 eq 76-78 forward
/// relations (see the module-level white-point normalizer notes):
///
/// ```text
/// Y' = E'_Y
/// Z' = (2·D'z + Y') / 0.986566
/// X' = 2·D'x + 0.991902 · Y'
/// ```
///
/// The `X/Y/Z` tristimulus values ride in the `R/G/B` slots of the forward
/// relations, so the output is `[X', Y', Z']`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn recover_xyz_prime(norm: [f32; 3]) -> [f32; 3] {
  let yp = norm[0];
  let dz = norm[1];
  let dx = norm[2];
  let zp = (2.0_f32 * dz + yp) / WHITE_NORM_Z;
  let xp = 2.0_f32 * dx + WHITE_NORM_X * yp;
  [xp, yp, zp]
}

/// Decodes one normalized SMPTE 2085 triple to the linear `RGB` (BT.2020
/// primaries) of steps 2–4 of the pipeline: `recover → PQ EOTF → M_XYZ→RGB`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn smpte2085_norm_to_rgb_linear(norm: [f32; 3]) -> [f32; 3] {
  let xyz_prime = recover_xyz_prime(norm);
  let xyz = [
    pq_hlg::pq_eotf(xyz_prime[0]),
    pq_hlg::pq_eotf(xyz_prime[1]),
    pq_hlg::pq_eotf(xyz_prime[2]),
  ];
  matmul3(&M_XYZ_TO_RGB_REC2020, xyz)
}

/// Decodes one normalized SMPTE 2085 triple all the way to the `R'G'B'`
/// display signal (the full pipeline steps 2–5): linear `RGB` then the
/// per-channel PQ OETF re-encode. This is the value the integer-output kernels
/// narrow.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte2085_norm_to_rgb_prime(norm: [f32; 3]) -> [f32; 3] {
  let lin = smpte2085_norm_to_rgb_linear(norm);
  [
    pq_hlg::pq_oetf(lin[0]),
    pq_hlg::pq_oetf(lin[1]),
    pq_hlg::pq_oetf(lin[2]),
  ]
}

/// Decodes one integer SMPTE 2085 triple to the `R'G'B'` display signal —
/// [`dequant_smpte2085`] then [`smpte2085_norm_to_rgb_prime`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn smpte2085_pixel_to_rgb_prime<const BITS: u32, const BE: bool>(
  y: u16,
  dz: u16,
  dx: u16,
  full_range: bool,
) -> [f32; 3] {
  let norm = dequant_smpte2085::<BITS>(
    load_sample::<BITS, BE>(y),
    load_sample::<BITS, BE>(dz),
    load_sample::<BITS, BE>(dx),
    full_range,
  );
  smpte2085_norm_to_rgb_prime(norm)
}

/// Round-half-up `f32 → u8` narrow with `[0, 1]` clamp (mirrors the ICtCp /
/// IPT-C2 non-affine kernels' `narrow_unit_to_u8`).
#[cfg_attr(not(tarpaulin), inline(always))]
fn narrow_unit_to_u8(c: f32) -> u8 {
  let scaled = c.clamp(0.0_f32, 1.0_f32) * 255.0_f32 + 0.5_f32;
  scaled.clamp(0.0_f32, 255.0_f32) as u8
}

/// Round-half-up `f32 → u16` narrow to the **native** `BITS`-depth range
/// `[0, (1 << BITS) - 1]` (low-bit-packed), matching the affine
/// `yuv_444p_n_to_rgb_u16_row` native-depth output contract — the
/// `Yuv444pN` u16 outputs are native `BITS`-bit, **not** full 16-bit.
#[cfg_attr(not(tarpaulin), inline(always))]
fn narrow_unit_to_u16_native<const BITS: u32>(c: f32) -> u16 {
  let out_max = ((1u32 << BITS) - 1) as f32;
  let scaled = c.clamp(0.0_f32, 1.0_f32) * out_max + 0.5_f32;
  scaled.clamp(0.0_f32, out_max) as u16
}

// ---- Planar 4:4:4 high-bit SMPTE 2085 → packed RGB/RGBA kernels ---------
//
// The representative high-bit family for the #303 wiring. `BITS = 12` in
// practice (`Yuv444p12`); the kernels are const-generic over any `BITS` the
// affine 4:4:4 family accepts.

/// One row of high-bit planar 4:4:4 SMPTE 2085 → packed **u8 RGB** (`ALPHA =
/// false`) or **RGBA** (`ALPHA = true`, opaque `0xFF`). `BITS` is the active
/// input bit depth; `BE` the wire byte order; `full_range` the YCbCr-style
/// quantization range.
#[cfg_attr(not(tarpaulin), inline(always))]
fn smpte2085_444p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = smpte2085_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range);
    out[x * bpp] = narrow_unit_to_u8(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u8(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u8(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// One row of high-bit planar 4:4:4 SMPTE 2085 → packed **native-depth u16
/// RGB** (`ALPHA = false`) or **RGBA** (`ALPHA = true`, opaque alpha
/// `(1 << BITS) - 1`). The `R'G'B'` display signal is narrowed to the
/// `Yuv444pN` native range `[0, (1 << BITS) - 1]` (low-bit-packed) — the same
/// contract as the affine `yuv_444p_n_to_rgb_u16_row` family, **not** a
/// full-16-bit scale. The opaque alpha is `(1 << BITS) - 1`, matching both the
/// affine RGBA kernel and [`expand_rgb_u16_to_rgba_u16_row`](crate::row::scalar::rgb_expand::expand_rgb_u16_to_rgba_u16_row),
/// so the `rgba_u16`-only and `rgb_u16 + rgba_u16` sink routes are identical.
#[cfg_attr(not(tarpaulin), inline(always))]
fn smpte2085_444p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = smpte2085_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range);
    out[x * bpp] = narrow_unit_to_u16_native::<BITS>(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u16_native::<BITS>(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u16_native::<BITS>(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = ((1u32 << BITS) - 1) as u16;
    }
  }
}

/// High-bit planar 4:4:4 SMPTE 2085 → packed **u8 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte2085_444p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  smpte2085_444p_n_to_rgb_or_rgba_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range);
}

/// High-bit planar 4:4:4 SMPTE 2085 → packed **u8 RGBA** (opaque `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte2085_444p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  smpte2085_444p_n_to_rgb_or_rgba_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range);
}

/// High-bit planar 4:4:4 SMPTE 2085 → packed **u16 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte2085_444p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  smpte2085_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range);
}

/// High-bit planar 4:4:4 SMPTE 2085 → packed native-depth **u16 RGBA** (opaque
/// alpha `(1 << BITS) - 1`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte2085_444p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  smpte2085_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range);
}

#[cfg(all(test, feature = "std"))]
mod tests;
