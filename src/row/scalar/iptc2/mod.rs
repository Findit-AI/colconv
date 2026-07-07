//! Scalar reference kernels for the **IPT-C2** (ITU-T H.273
//! `MatrixCoefficients = 15`, "IPT-C2") non-affine colour decode — the base
//! colour space of the Dolby Vision Profile 5 signal.
//!
//! IPT-C2 is **not** an affine YCbCr matrix: unlike every
//! [`Coefficients::for_matrix`](super::Coefficients::for_matrix) decode (a
//! single Q15 matrix + offset), recovering RGB from `I, P, T` requires a
//! per-channel non-linear transfer in the middle of the pipeline. This
//! module is the dedicated, scalar-only decode for it — the direct analogue
//! of [`super::ictcp`] (`decode → matmul → EOTF → matmul → OETF → narrow`),
//! whose PQ pipeline this mirrors exactly; only the two 3×3 matrices differ.
//!
//! # Decode pipeline (per pixel)
//!
//! ```text
//! I,P,T (int)  ──dequant──▶  I,P,T (norm, f32)
//!              ──M⁻¹_IPT──▶  L'M'S'         (inverse IPT-C2 rotation)
//!              ──PQ EOTF──▶  LMS  (linear)  (per-channel SMPTE ST 2084)
//!              ──M_LMS→RGB─▶  RGB  (linear, IPT-C2 RGB primaries)
//!              ──PQ OETF──▶  R'G'B'         (per-channel SMPTE ST 2084)
//!              ──narrow───▶  u8 / u16 output
//! ```
//!
//! 1. **Dequantize** the integer `I,P,T` samples to the normalized domain
//!    (`I ∈ [0,1]` luma-like, `P,T` signed chroma-like), using the identical
//!    studio/full-range scaling colconv's affine YCbCr decode uses
//!    ([`super::range_params_n`]) — IPT-C2 is carried in a YCbCr container and
//!    shares its H.273 quantization (the same encoding as [`super::ictcp`]).
//! 2. **Inverse IPT-C2 rotation** [`IPTC2_IPT_TO_LMSP`] maps `I,P,T → L'M'S'`.
//!    IPT-C2 has a single rotation (no PQ/HLG split): the decode is defined
//!    only for the SMPTE ST 2084 PQ transfer.
//! 3. **PQ EOTF** lifts the non-linear `L'M'S'` to linear `LMS` per channel,
//!    via the BT.2100 transfer math of [`crate::resample::pq_hlg`].
//! 4. **[`IPTC2_LMS_TO_RGB`]** (the inverse of IPT-C2's own `RGB→LMS`
//!    crosstalk matrix) maps linear `LMS → RGB`.
//! 5. **PQ OETF** re-encodes linear `RGB → R'G'B'` per channel, yielding the
//!    `R'G'B'` display signal. This matches colconv's transfer-preserving
//!    convention (the affine YCbCr decode likewise emits `R'G'B'` in the
//!    source's transfer domain). The integer-output kernels narrow `R'G'B'`;
//!    out-of-gamut excursions clamp at the narrow.
//!
//! # Verification
//!
//! The two matrices are the exact rational inverses of the published H.273
//! integer forward matrices (Table 4 code 15); the end-to-end decode is
//! pinned by a spec-integer encode/decode round-trip and the neutral-axis
//! structural anchor in [`tests`] — no external library, the spec integers
//! are the oracle.
//!
//! # No SIMD
//!
//! Scalar-only by design (as [`super::ictcp`]): the per-channel
//! transcendental EOTF/OETF (`powf`) do not vectorize into the integer-lane
//! shape the affine YCbCr kernels use, and the transcendental cost dwarfs any
//! lane parallelism win. Routing therefore always takes the scalar path for
//! `ColorMatrix::IptC2`, regardless of the `use_simd` hint.

use crate::{Transfer, resample::pq_hlg};

use super::bits_mask;

/// Which transfer system an [`IptC2`](self) source is encoded in. IPT-C2 is
/// defined **only** for the SMPTE ST 2084 PQ transfer (unlike
/// [`ICtCp`](super::ictcp), which additionally has an HLG rotation), so this
/// carries a single variant; it exists to mirror ICtCp's transfer-resolution
/// gate ([`IctcpTransfer::for_transfer`](super::ictcp::IctcpTransfer::for_transfer))
/// and give the dispatch a resolvable predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IptC2Transfer {
  /// SMPTE ST 2084 Perceptual Quantizer (the Dolby Vision Profile 5 base
  /// transfer). The only transfer IPT-C2 is defined for.
  Pq,
}

impl IptC2Transfer {
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

  /// Resolves the IPT-C2 transfer variant from a source's signalled H.273
  /// [`Transfer`] characteristics.
  ///
  /// Returns [`Some`] **only** for [`Transfer::SmpteSt2084Pq`] → [`Self::Pq`],
  /// the single transfer IPT-C2 defines a derivation for. Any other transfer
  /// (including [`Transfer::Unspecified`]) returns [`None`]: IPT-C2 is
  /// undefined outside PQ, so the caller must fall back to the affine matrix
  /// path rather than apply an unverifiable transfer. A source tagged
  /// `ColorMatrix::IptC2` with no PQ transfer is malformed; the affine
  /// fallback is the defined, non-panicking policy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn for_transfer(transfer: Transfer) -> Option<Self> {
    match transfer {
      Transfer::SmpteSt2084Pq => Some(Self::Pq),
      _ => None,
    }
  }
}

// ---- Verified IPT-C2 decode matrices ------------------------------------
//
// Both are the exact rational inverses of the published ITU-T H.273 V4
// (07/2024) §8.3 Table 4 code 15 integer forward matrices:
//   RGB→LMS       (eq 17-19) = [[1747,2169,180],[673,3029,394],[50,207,3839]]/4096
//   L'M'S'→IPT    (eq 85-87) = [[1638,1638,820],[18248,-19870,1622],[3300,1463,-4763]]/4096
// The decode inverts these. The `RGB→LMS` rows each sum to 4096/4096 = 1
// (so gray maps to gray), and the `L'M'S'→IPT` P and T rows each sum to 0
// (so an equal L'=M'=S' has P = T = 0) — the structural anchors in `tests`.

/// `IPT → L'M'S'`: the inverse of the H.273 eq 85-87 `L'M'S'→IPT` rotation.
/// Each row is `[1, x, y]`, so a neutral `I` (with `P = T = 0`) yields
/// `L' = M' = S' = I`.
const IPTC2_IPT_TO_LMSP: [[f32; 3]; 3] = [
  [1.0_f32, 0.097_557_89_f32, 0.205_382_93_f32],
  [1.0_f32, -0.113_883_62_f32, 0.133_378_28_f32],
  [1.0_f32, 0.032_611_65_f32, -0.676_696_2_f32],
];

/// `LMS → RGB` (linear, IPT-C2 RGB primaries): the inverse of the H.273 eq
/// 17-19 `RGB→LMS` crosstalk matrix. Distinct from
/// [`ICtCp`](super::ictcp)'s `LMS_TO_RGB` (IPT-C2 has its own `RGB→LMS`
/// integer matrix).
const IPTC2_LMS_TO_RGB: [[f32; 3]; 3] = [
  [3.237_466_f32, -2.324_205_8_f32, 0.086_739_56_f32],
  [-0.718_875_47_f32, 1.877_899_9_f32, -0.159_024_48_f32],
  [-0.003_403_51_f32, -0.070_985_93_f32, 1.074_389_5_f32],
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

/// Dequantizes one integer `I,P,T` triple to the normalized IPT-C2 domain
/// (`I` luma-like in `[0, 1]`, `P`/`T` signed chroma-like centred on `0`),
/// using the **same** H.273 studio/full-range scaling the affine YCbCr decode
/// applies ([`super::range_params_n`] + the `128 << (BITS-8)` chroma bias).
/// IPT-C2 shares the YCbCr integer encoding with [`super::ictcp`], so this is
/// identical to that decode's dequantization.
///
/// `I = (DI' − y_off) / y_range`, `P = (DP' − 2^(BITS-1)) / c_range`, where
/// `(y_off, y_range, c_range)` are `(0, in_max, in_max)` for full range and
/// `(16·k, 219·k, 224·k)` (`k = 2^(BITS-8)`) for studio range — the
/// unscaled, normalized form of [`super::range_params_n`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn dequant_iptc2<const BITS: u32>(i: u16, p: u16, t: u16, full_range: bool) -> [f32; 3] {
  let k: i32 = 1 << (BITS - 8);
  let chroma_bias = 128 * k; // = 2^(BITS-1), the chroma zero point
  let (y_off, y_range, c_range): (i32, f32, f32) = if full_range {
    let in_max = ((1u32 << BITS) - 1) as f32;
    (0, in_max, in_max)
  } else {
    (16 * k, (219 * k) as f32, (224 * k) as f32)
  };
  [
    (i as i32 - y_off) as f32 / y_range,
    (p as i32 - chroma_bias) as f32 / c_range,
    (t as i32 - chroma_bias) as f32 / c_range,
  ]
}

/// Decodes one normalized IPT-C2 triple to the linear `RGB` (IPT-C2 RGB
/// primaries) of steps 2–4 of the pipeline: `M⁻¹ → PQ EOTF → M_LMS→RGB`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn iptc2_norm_to_rgb_linear(norm: [f32; 3]) -> [f32; 3] {
  let lms_p = matmul3(&IPTC2_IPT_TO_LMSP, norm);
  let lms = [
    pq_hlg::pq_eotf(lms_p[0]),
    pq_hlg::pq_eotf(lms_p[1]),
    pq_hlg::pq_eotf(lms_p[2]),
  ];
  matmul3(&IPTC2_LMS_TO_RGB, lms)
}

/// Decodes one normalized IPT-C2 triple all the way to the `R'G'B'` display
/// signal (the full pipeline steps 2–5): linear `RGB` then the per-channel PQ
/// OETF re-encode. This is the value the integer-output kernels narrow.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn iptc2_norm_to_rgb_prime(norm: [f32; 3]) -> [f32; 3] {
  let lin = iptc2_norm_to_rgb_linear(norm);
  [
    pq_hlg::pq_oetf(lin[0]),
    pq_hlg::pq_oetf(lin[1]),
    pq_hlg::pq_oetf(lin[2]),
  ]
}

/// Decodes one integer IPT-C2 triple to the `R'G'B'` display signal —
/// [`dequant_iptc2`] then [`iptc2_norm_to_rgb_prime`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn iptc2_pixel_to_rgb_prime<const BITS: u32, const BE: bool>(
  i: u16,
  p: u16,
  t: u16,
  full_range: bool,
) -> [f32; 3] {
  let norm = dequant_iptc2::<BITS>(
    load_sample::<BITS, BE>(i),
    load_sample::<BITS, BE>(p),
    load_sample::<BITS, BE>(t),
    full_range,
  );
  iptc2_norm_to_rgb_prime(norm)
}

/// Round-half-up `f32 → u8` narrow with `[0, 1]` clamp (mirrors the ICtCp
/// non-affine kernel's `narrow_unit_to_u8`).
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

// ---- Planar 4:4:4 high-bit IPT-C2 → packed RGB/RGBA kernels -------------
//
// The representative high-bit family for the #303 wiring. `BITS = 12` in
// practice (`Yuv444p12`, the Dolby Vision Profile 5 base); the kernels are
// const-generic over any `BITS` the affine 4:4:4 family accepts.

/// One row of high-bit planar 4:4:4 IPT-C2 → packed **u8 RGB** (`ALPHA =
/// false`) or **RGBA** (`ALPHA = true`, opaque `0xFF`). `BITS` is the active
/// input bit depth; `BE` the wire byte order; `full_range` the YCbCr-style
/// quantization range.
#[cfg_attr(not(tarpaulin), inline(always))]
fn iptc2_444p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
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
    let rgb = iptc2_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range);
    out[x * bpp] = narrow_unit_to_u8(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u8(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u8(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// One row of high-bit planar 4:4:4 IPT-C2 → packed **native-depth u16 RGB**
/// (`ALPHA = false`) or **RGBA** (`ALPHA = true`, opaque alpha `(1 << BITS) -
/// 1`). The `R'G'B'` display signal is narrowed to the `Yuv444pN` native
/// range `[0, (1 << BITS) - 1]` (low-bit-packed) — the same contract as the
/// affine `yuv_444p_n_to_rgb_u16_row` family, **not** a full-16-bit scale.
/// The opaque alpha is `(1 << BITS) - 1`, matching both the affine RGBA
/// kernel and [`expand_rgb_u16_to_rgba_u16_row`](crate::row::scalar::rgb_expand::expand_rgb_u16_to_rgba_u16_row),
/// so the `rgba_u16`-only and `rgb_u16 + rgba_u16` sink routes are identical.
#[cfg_attr(not(tarpaulin), inline(always))]
fn iptc2_444p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
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
    let rgb = iptc2_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range);
    out[x * bpp] = narrow_unit_to_u16_native::<BITS>(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u16_native::<BITS>(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u16_native::<BITS>(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = ((1u32 << BITS) - 1) as u16;
    }
  }
}

/// High-bit planar 4:4:4 IPT-C2 → packed **u8 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn iptc2_444p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  iptc2_444p_n_to_rgb_or_rgba_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range);
}

/// High-bit planar 4:4:4 IPT-C2 → packed **u8 RGBA** (opaque `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn iptc2_444p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  iptc2_444p_n_to_rgb_or_rgba_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range);
}

/// High-bit planar 4:4:4 IPT-C2 → packed **u16 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn iptc2_444p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  iptc2_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range);
}

/// High-bit planar 4:4:4 IPT-C2 → packed native-depth **u16 RGBA** (opaque
/// alpha `(1 << BITS) - 1`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn iptc2_444p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  iptc2_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range);
}

#[cfg(all(test, feature = "std"))]
mod tests;
