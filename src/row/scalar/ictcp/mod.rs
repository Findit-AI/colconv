//! Scalar reference kernels for the **ICtCp** (ITU-R BT.2100, H.273
//! `MatrixCoefficients = 14`) non-affine colour decode.
//!
//! ICtCp is **not** an affine YCbCr matrix: unlike every
//! [`Coefficients::for_matrix`](super::Coefficients::for_matrix) decode (a
//! single Q15 matrix + offset), recovering RGB from `I, Ct, Cp` requires a
//! per-channel non-linear transfer in the middle of the pipeline. This
//! module is the dedicated, scalar-only decode for it — the direct analogue
//! of [`super::xyz12`] (`decode → matmul → OETF → narrow`), the crate's
//! existing non-affine source.
//!
//! # Decode pipeline (per pixel)
//!
//! ```text
//! I,Ct,Cp (int)  ──dequant──▶  I,Ct,Cp (norm, f32)
//!                ──M⁻¹_ICtCp──▶  L'M'S'         (inverse ICtCp matrix; PQ vs HLG)
//!                ──EOTF───────▶  LMS  (linear)  (per-channel; PQ vs HLG)
//!                ──M_LMS→RGB──▶  RGB  (linear, BT.2020 primaries)
//!                ──OETF───────▶  R'G'B'         (per-channel; PQ vs HLG)
//!                ──narrow─────▶  u8 / u16 output
//! ```
//!
//! 1. **Dequantize** the integer `I,Ct,Cp` samples to the normalized
//!    domain (`I ∈ [0,1]` luma-like, `Ct,Cp` signed chroma-like), using the
//!    identical studio/full-range scaling pixon's affine YCbCr decode uses
//!    ([`super::range_params_n`]) — ICtCp is carried in a YCbCr container
//!    and shares its H.273 quantization.
//! 2. **Inverse ICtCp matrix** `M⁻¹` maps `I,Ct,Cp → L'M'S'`. The matrix
//!    **differs between PQ and HLG** (the SMPTE-2085-class transfer-dependent
//!    trap), selected by [`IctcpTransfer`].
//! 3. **EOTF** lifts the non-linear `L'M'S'` to linear `LMS` per channel —
//!    PQ (SMPTE ST 2084) or HLG (ARIB STD-B67) — via the BT.2100 transfer
//!    math of [`crate::resample::pq_hlg`] (the verified #313 foundation).
//! 4. **`M_LMS→RGB`** (the inverse of the BT.2020 `RGB→LMS` crosstalk
//!    matrix) maps linear `LMS → RGB`.
//! 5. **OETF** re-encodes linear `RGB → R'G'B'` per channel (PQ or HLG),
//!    yielding the BT.2100 `R'G'B'` display signal. This matches pixon's
//!    transfer-preserving convention (the affine YCbCr decode likewise
//!    emits `R'G'B'` in the source's transfer domain) and BT.2100's own
//!    definition of `R'G'B'` as the PQ/HLG-encoded RGB. The integer-output
//!    kernels narrow `R'G'B'`; out-of-gamut excursions clamp at the narrow.
//!
//! # Verification
//!
//! Every matrix and the end-to-end decode are pinned against the
//! `colour-science` 0.4.7 reference (`colour.ICtCp_to_RGB`, PQ and HLG) and
//! BT.2100-2 structural anchors in [`tests`]; the inverse matrices are the
//! exact rational inverses of the published BT.2100 forward matrices.
//!
//! # No SIMD
//!
//! Scalar-only by design: the per-channel transcendental EOTF/OETF
//! (`powf`/`exp`/`ln`) do not vectorize into the integer-lane shape the
//! affine YCbCr kernels use, and the transcendental cost dwarfs any lane
//! parallelism win. Routing therefore always takes the scalar path for
//! `ColorMatrix::Ictcp`, regardless of the `use_simd` hint.

use crate::{Transfer, resample::pq_hlg};

use super::bits_mask;

/// Which BT.2100 transfer system an [`ICtCp`](self) source is encoded in —
/// the axis that selects **both** the inverse `ICtCp` matrix **and** the
/// per-channel EOTF/OETF. This is the transfer-dependent selection BT.2100
/// makes between its PQ and HLG `ICtCp` variants (the matrices genuinely
/// differ; see [`ICTCP_TO_LMSP_PQ`] vs [`ICTCP_TO_LMSP_HLG`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IctcpTransfer {
  /// SMPTE ST 2084 Perceptual Quantizer (HDR10 / Dolby Vision). The
  /// default and most common `ICtCp` transfer.
  Pq,
  /// ARIB STD-B67 Hybrid Log-Gamma.
  Hlg,
}

impl IctcpTransfer {
  /// Lowercase identifier for the transfer (`"pq"` / `"hlg"`). The mandated
  /// unit-enum accessor; consumed by diagnostics and the reference tests
  /// (no production caller yet, hence the `dead_code` allowance).
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn as_str(self) -> &'static str {
    match self {
      Self::Pq => "pq",
      Self::Hlg => "hlg",
    }
  }

  /// Resolves the `ICtCp` transfer variant from a source's signalled H.273
  /// [`Transfer`] characteristics — the bridge from
  /// [`ColorSpec::transfer`](crate::ColorSpec::transfer) deferred from #313.
  ///
  /// Returns [`Some`] **only** for the two transfers BT.2100 defines an
  /// `ICtCp` derivation for:
  ///
  /// - [`Transfer::SmpteSt2084Pq`] → [`Self::Pq`]
  /// - [`Transfer::AribStdB67Hlg`] → [`Self::Hlg`]
  ///
  /// Any other transfer (including [`Transfer::Unspecified`]) returns
  /// [`None`]: `ICtCp` is undefined outside PQ/HLG, so the caller must fall
  /// back to the affine matrix path rather than apply an unverifiable
  /// transfer. A source tagged `ColorMatrix::Ictcp` with no PQ/HLG transfer
  /// is malformed; the affine fallback is the defined, non-panicking policy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn for_transfer(transfer: Transfer) -> Option<Self> {
    match transfer {
      Transfer::SmpteSt2084Pq => Some(Self::Pq),
      Transfer::AribStdB67Hlg => Some(Self::Hlg),
      _ => None,
    }
  }
}

// ---- Verified BT.2100 decode matrices -----------------------------------
//
// All three are pinned against `colour-science` 0.4.7 (the published
// `MATRIX_ICTCP_*` constants) and are the exact rational inverses of the
// BT.2100-2 forward matrices documented in #313:
//   RGB→LMS (BT.2020)   = [[1688,2146,262],[683,2951,462],[99,309,3688]]/4096
//   L'M'S'→ICtCp  (PQ)  = [[2048,2048,0],[6610,-13613,7003],[17933,-17390,-543]]/4096
//   L'M'S'→ICtCp (HLG)  = [[2048,2048,0],[3625,-7465,3840],[9500,-9212,-288]]/4096
// The decode inverts these.

/// `LMS → RGB` (linear, BT.2020 primaries): the inverse of the BT.2100
/// `RGB→LMS` crosstalk matrix. Shared by the PQ and HLG decodes (only the
/// `ICtCp` matrix + transfer differ between them). Matches
/// `colour.models.rgb.ictcp.MATRIX_ICTCP_LMS_TO_RGB`.
const LMS_TO_RGB: [[f32; 3]; 3] = [
  [3.436_606_6_f32, -2.506_452_f32, 0.069_845_42_f32],
  [-0.791_329_56_f32, 1.9836005_f32, -0.19227089_f32],
  [-0.025_949_9_f32, -0.098_913_714_f32, 1.124_863_6_f32],
];

/// `ICtCp → L'M'S'` for the **PQ** transfer: the inverse of the BT.2100
/// PQ `L'M'S'→ICtCp` matrix. Matches
/// `colour.models.rgb.ictcp.MATRIX_ICTCP_ICTCP_TO_LMS_P`.
const ICTCP_TO_LMSP_PQ: [[f32; 3]; 3] = [
  [1.0_f32, 0.008_609_037_f32, 0.111029625_f32],
  [1.0_f32, -0.008_609_037_f32, -0.111029625_f32],
  [1.0_f32, 0.560_031_35_f32, -0.320_627_18_f32],
];

/// `ICtCp → L'M'S'` for the **HLG** transfer: the inverse of the BT.2100
/// HLG `L'M'S'→ICtCp` matrix. Genuinely different from
/// [`ICTCP_TO_LMSP_PQ`] (the BT.2100 HLG `ICtCp` matrix has its own
/// coefficients) — this is the transfer-dependent selection that must track
/// [`IctcpTransfer`]. Matches
/// `colour.models.rgb.ictcp.MATRIX_ICTCP_ICTCP_TO_LMS_P_BT2100_HLG_2`.
const ICTCP_TO_LMSP_HLG: [[f32; 3]; 3] = [
  [1.0_f32, 0.015_718_58_f32, 0.209_581_06_f32],
  [1.0_f32, -0.015_718_58_f32, -0.209_581_06_f32],
  [1.0_f32, 1.021_271_1_f32, -0.605_274_5_f32],
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

/// Dequantizes one integer `I,Ct,Cp` triple to the normalized `ICtCp`
/// domain (`I` luma-like in `[0, 1]`, `Ct`/`Cp` signed chroma-like centred
/// on `0`), using the **same** H.273 studio/full-range scaling the affine
/// YCbCr decode applies ([`super::range_params_n`] + the `128 << (BITS-8)`
/// chroma bias). `ICtCp` shares the YCbCr integer encoding, so this is the
/// correct dequantization — verified end-to-end against `colour-science` in
/// [`tests`].
///
/// `I = (DI' − y_off) / y_range`, `Ct = (DCt' − 2^(BITS-1)) / c_range`,
/// where `(y_off, y_range, c_range)` are `(0, in_max, in_max)` for full
/// range and `(16·k, 219·k, 224·k)` (`k = 2^(BITS-8)`) for studio range —
/// the unscaled, normalized form of [`super::range_params_n`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn dequant_ictcp<const BITS: u32>(i: u16, ct: u16, cp: u16, full_range: bool) -> [f32; 3] {
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
    (ct as i32 - chroma_bias) as f32 / c_range,
    (cp as i32 - chroma_bias) as f32 / c_range,
  ]
}

/// Decodes one normalized `ICtCp` triple to the linear `RGB` (BT.2020
/// primaries) of steps 2–4 of the pipeline: `M⁻¹ → EOTF → M_LMS→RGB`. The
/// transfer selects both the inverse matrix and the per-channel EOTF.
#[cfg_attr(not(tarpaulin), inline(always))]
fn ictcp_norm_to_rgb_linear(norm: [f32; 3], transfer: IctcpTransfer) -> [f32; 3] {
  let lms_p = match transfer {
    IctcpTransfer::Pq => matmul3(&ICTCP_TO_LMSP_PQ, norm),
    IctcpTransfer::Hlg => matmul3(&ICTCP_TO_LMSP_HLG, norm),
  };
  let lms = match transfer {
    IctcpTransfer::Pq => [
      pq_hlg::pq_eotf(lms_p[0]),
      pq_hlg::pq_eotf(lms_p[1]),
      pq_hlg::pq_eotf(lms_p[2]),
    ],
    IctcpTransfer::Hlg => [
      pq_hlg::hlg_eotf(lms_p[0]),
      pq_hlg::hlg_eotf(lms_p[1]),
      pq_hlg::hlg_eotf(lms_p[2]),
    ],
  };
  matmul3(&LMS_TO_RGB, lms)
}

/// Decodes one normalized `ICtCp` triple all the way to the BT.2100
/// `R'G'B'` display signal (the full pipeline steps 2–5): linear `RGB` then
/// the per-channel OETF re-encode. This is the value the integer-output
/// kernels narrow.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ictcp_norm_to_rgb_prime(norm: [f32; 3], transfer: IctcpTransfer) -> [f32; 3] {
  let lin = ictcp_norm_to_rgb_linear(norm, transfer);
  match transfer {
    IctcpTransfer::Pq => [
      pq_hlg::pq_oetf(lin[0]),
      pq_hlg::pq_oetf(lin[1]),
      pq_hlg::pq_oetf(lin[2]),
    ],
    IctcpTransfer::Hlg => [
      pq_hlg::hlg_oetf(lin[0]),
      pq_hlg::hlg_oetf(lin[1]),
      pq_hlg::hlg_oetf(lin[2]),
    ],
  }
}

/// Decodes one integer `ICtCp` triple to the `R'G'B'` display signal —
/// [`dequant_ictcp`] then [`ictcp_norm_to_rgb_prime`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn ictcp_pixel_to_rgb_prime<const BITS: u32, const BE: bool>(
  i: u16,
  ct: u16,
  cp: u16,
  full_range: bool,
  transfer: IctcpTransfer,
) -> [f32; 3] {
  let norm = dequant_ictcp::<BITS>(
    load_sample::<BITS, BE>(i),
    load_sample::<BITS, BE>(ct),
    load_sample::<BITS, BE>(cp),
    full_range,
  );
  ictcp_norm_to_rgb_prime(norm, transfer)
}

/// Round-half-up `f32 → u8` narrow with `[0, 1]` clamp (mirrors the xyz12
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

// ---- Planar 4:4:4 high-bit ICtCp → packed RGB/RGBA kernels --------------
//
// The representative high-bit family for the #303 wiring. `BITS ∈ {10, 12}`
// in practice (PQ/HLG `ICtCp` is HDR); the kernels are const-generic over
// any `BITS` the affine 4:4:4 family accepts.

/// One row of high-bit planar 4:4:4 `ICtCp` → packed **u8 RGB** (`ALPHA =
/// false`) or **RGBA** (`ALPHA = true`, opaque `0xFF`). `BITS` is the active
/// input bit depth; `BE` the wire byte order; `full_range` the YCbCr-style
/// quantization range; `transfer` the PQ/HLG selection.
#[cfg_attr(not(tarpaulin), inline(always))]
fn ictcp_444p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = ictcp_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range, transfer);
    out[x * bpp] = narrow_unit_to_u8(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u8(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u8(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// One row of high-bit planar 4:4:4 `ICtCp` → packed **native-depth u16
/// RGB** (`ALPHA = false`) or **RGBA** (`ALPHA = true`, opaque alpha
/// `(1 << BITS) - 1`). The `R'G'B'` display signal is narrowed to the
/// `Yuv444pN` native range `[0, (1 << BITS) - 1]` (low-bit-packed) — the
/// same contract as the affine `yuv_444p_n_to_rgb_u16_row` family, **not**
/// a full-16-bit scale. The opaque alpha is `(1 << BITS) - 1`, matching
/// both the affine RGBA kernel and [`expand_rgb_u16_to_rgba_u16_row`](crate::row::scalar::rgb_expand::expand_rgb_u16_to_rgba_u16_row),
/// so the `rgba_u16`-only and `rgb_u16 + rgba_u16` sink routes are identical.
#[cfg_attr(not(tarpaulin), inline(always))]
fn ictcp_444p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = ictcp_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range, transfer);
    out[x * bpp] = narrow_unit_to_u16_native::<BITS>(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u16_native::<BITS>(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u16_native::<BITS>(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = ((1u32 << BITS) - 1) as u16;
    }
  }
}

/// High-bit planar 4:4:4 `ICtCp` → packed **u8 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ictcp_444p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  ictcp_444p_n_to_rgb_or_rgba_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range, transfer);
}

/// High-bit planar 4:4:4 `ICtCp` → packed **u8 RGBA** (opaque `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ictcp_444p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  ictcp_444p_n_to_rgb_or_rgba_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range, transfer);
}

/// High-bit planar 4:4:4 `ICtCp` → packed **u16 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ictcp_444p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  ictcp_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE>(
    y, u, v, rgb_out, width, full_range, transfer,
  );
}

/// High-bit planar 4:4:4 `ICtCp` → packed native-depth **u16 RGBA** (opaque
/// alpha `(1 << BITS) - 1`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ictcp_444p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  full_range: bool,
  transfer: IctcpTransfer,
) {
  ictcp_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE>(
    y, u, v, rgba_out, width, full_range, transfer,
  );
}

#[cfg(all(test, feature = "std"))]
mod tests;
