//! Scalar reference kernels for the **constant-luminance** `YcCbcCrc` colour
//! decode (ITU-R BT.2020 CL, H.273 `MatrixCoefficients = 13`,
//! [`ColorMatrix::ChromaDerivedCl`](crate::ColorMatrix::ChromaDerivedCl)).
//!
//! CL is **not** an affine YCbCr matrix. Every
//! [`Coefficients::for_matrix`](super::Coefficients::for_matrix) decode — and
//! the chromaticity-derived *non*-constant-luminance sibling
//! [`ColorMatrix::ChromaDerivedNcl`](crate::ColorMatrix::ChromaDerivedNcl),
//! H.273 `= 12` — recovers `R'G'B'` from `Y',Cb,Cr` with a single Q15 matrix:
//! the luma `Y'` is a weighted sum of the *gamma-encoded* `R'G'B'`. The
//! constant-luminance system instead carries the luma in the **linear** domain
//! (`Yc = Kr·R + Kg·G + Kb·B` from linear light, then gamma-encoded to `Y'c`)
//! and bends the chroma normalisers through the BT.2020 OETF, so recovering RGB
//! needs a per-channel non-linear transfer in the middle of the pipeline. This
//! module is the dedicated, scalar-only decode for it — the direct analogue of
//! [`super::ictcp`] (the BT.2100 ICtCp non-affine decode) and [`super::xyz12`].
//!
//! # Why BT.2020-only (and not chromaticity-general like NCL)
//!
//! `ChromaDerivedNcl` is purely affine, so it generalises to *any* primaries:
//! [`chroma_derived_luma_weights`](super::chroma_derived_luma_weights) yields
//! `(Kr, Kg, Kb)` from the chromaticities and the standard YCbCr formula does
//! the rest. CL cannot: its chroma normalisers `(Nb, Pb, Nr, Pr)` are the
//! BT.2020 OETF evaluated at the gamut boundary (`Pb = 1 − oetf(Kb)`,
//! `Nb = −oetf(1 − Kb)`, and likewise for red), so the decode is intrinsically
//! tied to the BT.2020 luma weights **and** the BT.2020 transfer. There is no
//! published reference for "CL with arbitrary primaries", so this decode is
//! defined and verified for BT.2020 only — `chroma_derived_luma_weights` is
//! reused purely to *anchor* that the weights are the BT.2020 `0.2627 / 0.6780
//! / 0.0593` (see [`tests`]); the per-pixel math uses the published constants.
//! Routing therefore gates on BT.2020 primaries (the [`ClSystem`] resolution),
//! mirroring how ICtCp gates on a PQ/HLG transfer.
//!
//! # Decode pipeline (per pixel)
//!
//! ```text
//! Y'c,Cbc,Crc (int)  ──dequant──▶  Y'c,Cbc,Crc (norm, f32)
//!   ──chroma recover──▶  Y'c, B', R'   (piecewise ±2·{Nb,Pb} / ±2·{Nr,Pr})
//!   ──inverse-OETF────▶  Yc, B, R      (linear, BT.2020 camera curve)
//!   ──solve green─────▶  G             (G = (Yc − Kb·B − Kr·R) / Kg, linear)
//!   ──OETF────────────▶  R'G'B'        (re-encode to the display signal)
//!   ──narrow──────────▶  u8 / u16 output
//! ```
//!
//! 1. **Dequantize** the integer `Y'c,Cbc,Crc` to the normalized domain
//!    (`Y'c ∈ [0,1]`, `Cbc/Crc` signed, centred on `0`), using the **same**
//!    H.273 studio/full-range scaling pixon's affine YCbCr decode applies
//!    (the unscaled form of [`super::range_params_n`]); CL shares the YCbCr
//!    integer encoding. Verified against `colour-science`'s `ranges_YCbCr`.
//! 2. **Chroma recovery** maps `Cbc → B'` and `Crc → R'` about `Y'c` with the
//!    BT.2020 piecewise normalisers (`B' = Y'c + Cbc·(−2Nb)` for `Cbc ≤ 0`,
//!    `+ Cbc·(2Pb)` otherwise; likewise red). These four factors are the
//!    published BT.2020 CL constants.
//! 3. **Inverse-OETF** lifts the encoded `Y'c, B', R'` to linear `Yc, B, R`
//!    via the BT.2020 camera curve ([`crate::resample::bt2020_oetf`]).
//! 4. **Solve green** from the linear luminance identity
//!    `Yc = Kr·R + Kg·G + Kb·B` → `G = (Yc − Kb·B − Kr·R) / Kg` (linear light;
//!    the defining property of the constant-luminance system).
//! 5. **OETF** re-encodes linear `R, G, B → R'G'B'` (the BT.2020 display
//!    signal). This matches pixon's transfer-preserving convention — the
//!    affine YCbCr decode likewise emits `R'G'B'` in the source's transfer
//!    domain — and BT.2020's definition of `R'G'B'` as the OETF-encoded RGB.
//!    The integer-output kernels narrow `R'G'B'`; out-of-gamut excursions
//!    clamp at the narrow. (`colour-science`'s `YcCbcCrc_to_RGB` stops at the
//!    *linear* RGB of step 4; [`tests`] pin pixon against that stage, then
//!    re-encode, exactly as the ICtCp tests do.)
//!
//! # Verification
//!
//! The end-to-end decode is pinned against `colour-science` 0.4.7 — the
//! authoritative BT.2020 CL implementation (`colour.models.rgb.ycbcr.
//! YcCbcCrc_to_RGB`) — plus BT.2020-2 structural anchors (gray-axis
//! neutrality, OETF round-trip) that hold independently of any library, in
//! [`tests`].
//!
//! # No SIMD
//!
//! Scalar-only by design, for the same reason as [`super::ictcp`]: the
//! per-channel transcendental inverse-OETF / OETF (`powf`) does not vectorize
//! into the integer-lane shape the affine YCbCr kernels use, and the
//! transcendental cost dwarfs any lane-parallelism win. Routing always takes
//! the scalar path for `ColorMatrix::ChromaDerivedCl`, regardless of the
//! `use_simd` hint.

use crate::{Primaries, Transfer, resample::bt2020_oetf};

use super::bits_mask;

/// The constant-luminance system an [`YcCbcCrc`](self) source decodes under —
/// the gate that selects the BT.2020 luma weights, the OETF-derived chroma
/// normalisers, and the OETF bit-depth (`α/β`). Resolved from the signalled
/// colour [`Primaries`] **and** [`Transfer`]; CL is defined for BT.2020 only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ClSystem {
  /// Whether the BT.2020 OETF uses its 12-bit `(α, β) = (1.0993, 0.0181)`
  /// constants (`Transfer::Bt2020_12Bit`) rather than the 10-bit default.
  is_12_bit: bool,
}

impl ClSystem {
  /// Lowercase identifier (`"bt2020-cl-10"` / `"bt2020-cl-12"`). The mandated
  /// unit-style accessor; consumed by the reference tests (no production
  /// caller yet beyond the kernels, hence the `dead_code` allowance).
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn as_str(self) -> &'static str {
    if self.is_12_bit {
      "bt2020-cl-12"
    } else {
      "bt2020-cl-10"
    }
  }

  /// Whether the OETF uses the 12-bit-system `(α, β)` constants.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn is_12_bit(self) -> bool {
    self.is_12_bit
  }

  /// Resolves the CL system from a source's signalled colour [`Primaries`] and
  /// [`Transfer`] — the bridge from
  /// [`ColorSpec`](crate::ColorSpec) deferred from #313.
  ///
  /// Returns [`Some`] **only** for BT.2020 primaries paired with a BT.2020
  /// transfer characteristic — the gamut *and* the camera gamma the CL
  /// `YcCbcCrc` derivation is published and verifiable for:
  /// [`Transfer::Bt2020_12Bit`] selects the 12-bit `(α, β)` constants and
  /// [`Transfer::Bt2020_10Bit`] the 10-bit ones.
  ///
  /// **Every other** combination returns [`None`] and the caller falls back to
  /// the affine matrix path: a non-BT.2020 primary set, and — critically — any
  /// non-BT.2020 transfer, including [`Transfer::Unspecified`] and the PQ/HLG
  /// HDR curves, which are *not* the CL camera gamma. Decoding a PQ- or
  /// HLG-tagged source through the BT.2020 camera inverse-OETF would emit
  /// deterministic but wrong RGB (and falsely trip the non-affine resample
  /// reject), so an unresolved transfer takes the defined, non-panicking
  /// affine fallback instead — the same explicit transfer allow-list shape
  /// ICtCp uses (it resolves only its PQ/HLG transfers and returns [`None`]
  /// otherwise).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn resolve(primaries: Primaries, transfer: Transfer) -> Option<Self> {
    match (primaries, transfer) {
      (Primaries::Bt2020, Transfer::Bt2020_12Bit) => Some(Self { is_12_bit: true }),
      (Primaries::Bt2020, Transfer::Bt2020_10Bit) => Some(Self { is_12_bit: false }),
      _ => None,
    }
  }
}

// ---- Verified BT.2020 constant-luminance constants ----------------------
//
// All five are the published ITU-R BT.2020-2 (Table 4) values, cross-checked
// against `colour-science` 0.4.7 (`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` /
// `RGB_to_YcCbcCrc`); [`tests`] pin them.

/// BT.2020 CL luma weights `(Kr, Kg, Kb)` — the linear-light luminance
/// coefficients (`Yc = Kr·R + Kg·G + Kb·B`). Identical to the BT.2020
/// non-constant-luminance weights and to
/// [`chroma_derived_luma_weights(Primaries::Bt2020)`](super::chroma_derived_luma_weights),
/// which [`tests`] assert; the decode uses the published literals so the
/// per-pixel green solve is exact and `libm`-free.
const KR: f32 = 0.2627;
const KG: f32 = 0.6780;
const KB: f32 = 0.0593;

/// Blue-chroma recovery factor for the **negative** half (`Cbc ≤ 0`):
/// `−2·Nb` with `Nb = −oetf_BT2020(1 − Kb) = −0.9702`. Matches
/// `colour-science`'s `1.9404`.
const NEG_2NB: f32 = 1.9404;
/// Blue-chroma recovery factor for the **positive** half (`Cbc > 0`):
/// `2·Pb` with `Pb = 1 − oetf_BT2020(Kb) = 0.7908`. Matches
/// `colour-science`'s `1.5816`.
const POS_2PB: f32 = 1.5816;
/// Red-chroma recovery factor for the **negative** half (`Crc ≤ 0`):
/// `−2·Nr` with `Nr = −oetf_BT2020(1 − Kr) = −0.8592`. Matches
/// `colour-science`'s `1.7184`.
const NEG_2NR: f32 = 1.7184;
/// Red-chroma recovery factor for the **positive** half (`Crc > 0`):
/// `2·Pr` with `Pr = 1 − oetf_BT2020(Kr) = 0.4968`. Matches
/// `colour-science`'s `0.9936`.
const POS_2PR: f32 = 0.9936;

/// Endian-aware load of a wire `u16` masked to the active low `BITS`
/// (the low-bit-packed `Yuv*pN` convention, `MSB = false`). Mirrors the
/// affine high-bit kernel's `load_u16` + low-bit mask.
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_sample<const BITS: u32, const BE: bool>(s: u16) -> u16 {
  let raw = if BE { u16::from_be(s) } else { u16::from_le(s) };
  raw & bits_mask::<BITS>()
}

/// Dequantizes one integer `Y'c,Cbc,Crc` triple to the normalized CL domain
/// (`Y'c` luma-like in `[0, 1]`, `Cbc`/`Crc` signed chroma-like centred on
/// `0`), using the **same** H.273 studio/full-range scaling the affine YCbCr
/// decode applies ([`super::range_params_n`] + the `128 << (BITS-8)` chroma
/// bias). Identical in shape to [`super::ictcp::dequant_ictcp`]; verified
/// against `colour-science`'s `ranges_YCbCr` in [`tests`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn dequant_cl<const BITS: u32>(yc: u16, cbc: u16, crc: u16, full_range: bool) -> [f32; 3] {
  let k: i32 = 1 << (BITS - 8);
  let chroma_bias = 128 * k; // = 2^(BITS-1), the chroma zero point
  let (y_off, y_range, c_range): (i32, f32, f32) = if full_range {
    let in_max = ((1u32 << BITS) - 1) as f32;
    (0, in_max, in_max)
  } else {
    (16 * k, (219 * k) as f32, (224 * k) as f32)
  };
  [
    (yc as i32 - y_off) as f32 / y_range,
    (cbc as i32 - chroma_bias) as f32 / c_range,
    (crc as i32 - chroma_bias) as f32 / c_range,
  ]
}

/// Decodes one normalized `Y'c,Cbc,Crc` triple to **linear** BT.2020 RGB
/// (pipeline steps 2–4): chroma recovery → inverse-OETF → green solve. This is
/// the stage `colour-science`'s `YcCbcCrc_to_RGB` returns and the reference
/// tests pin.
#[cfg_attr(not(tarpaulin), inline(always))]
fn cl_norm_to_rgb_linear(norm: [f32; 3], system: ClSystem) -> [f32; 3] {
  let [yc_prime, cbc, crc] = norm;
  // Chroma recovery: B'/R' about Y'c, piecewise by the sign of the chroma.
  let b_prime = yc_prime + cbc * if cbc <= 0.0 { NEG_2NB } else { POS_2PB };
  let r_prime = yc_prime + crc * if crc <= 0.0 { NEG_2NR } else { POS_2PR };
  // Inverse-OETF to linear light.
  let is12 = system.is_12_bit();
  let yc = bt2020_oetf::oetf_inverse(yc_prime, is12);
  let b = bt2020_oetf::oetf_inverse(b_prime, is12);
  let r = bt2020_oetf::oetf_inverse(r_prime, is12);
  // Green from the linear-luminance identity (the CL defining property).
  let g = (yc - KB * b - KR * r) / KG;
  [r, g, b]
}

/// Decodes one normalized `Y'c,Cbc,Crc` triple all the way to the BT.2020
/// `R'G'B'` display signal (steps 2–5): linear RGB then the per-channel OETF
/// re-encode. This is the value the integer-output kernels narrow.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cl_norm_to_rgb_prime(norm: [f32; 3], system: ClSystem) -> [f32; 3] {
  let lin = cl_norm_to_rgb_linear(norm, system);
  let is12 = system.is_12_bit();
  [
    bt2020_oetf::oetf(lin[0], is12),
    bt2020_oetf::oetf(lin[1], is12),
    bt2020_oetf::oetf(lin[2], is12),
  ]
}

/// Decodes one integer `Y'c,Cbc,Crc` triple to the `R'G'B'` display signal —
/// [`dequant_cl`] then [`cl_norm_to_rgb_prime`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn cl_pixel_to_rgb_prime<const BITS: u32, const BE: bool>(
  yc: u16,
  cbc: u16,
  crc: u16,
  full_range: bool,
  system: ClSystem,
) -> [f32; 3] {
  let norm = dequant_cl::<BITS>(
    load_sample::<BITS, BE>(yc),
    load_sample::<BITS, BE>(cbc),
    load_sample::<BITS, BE>(crc),
    full_range,
  );
  cl_norm_to_rgb_prime(norm, system)
}

/// Round-half-up `f32 → u8` narrow with `[0, 1]` clamp (mirrors the ICtCp /
/// xyz12 non-affine kernels' `narrow_unit_to_u8`).
#[cfg_attr(not(tarpaulin), inline(always))]
fn narrow_unit_to_u8(c: f32) -> u8 {
  let scaled = c.clamp(0.0_f32, 1.0_f32) * 255.0_f32 + 0.5_f32;
  scaled.clamp(0.0_f32, 255.0_f32) as u8
}

/// Round-half-up `f32 → u16` narrow to the **native** `BITS`-depth range
/// `[0, (1 << BITS) - 1]` (low-bit-packed), matching the affine
/// `yuv_444p_n_to_rgb_u16_row` native-depth output contract.
#[cfg_attr(not(tarpaulin), inline(always))]
fn narrow_unit_to_u16_native<const BITS: u32>(c: f32) -> u16 {
  let out_max = ((1u32 << BITS) - 1) as f32;
  let scaled = c.clamp(0.0_f32, 1.0_f32) * out_max + 0.5_f32;
  scaled.clamp(0.0_f32, out_max) as u16
}

// ---- Planar 4:4:4 high-bit CL → packed RGB/RGBA kernels -----------------
//
// The representative high-bit family for the #303 wiring, mirroring the ICtCp
// kernels. `BITS ∈ {10, 12}` in practice; const-generic over any `BITS` the
// affine 4:4:4 family accepts.

/// One row of high-bit planar 4:4:4 CL `YcCbcCrc` → packed **u8 RGB** (`ALPHA
/// = false`) or **RGBA** (`ALPHA = true`, opaque `0xFF`). `BITS` is the active
/// input bit depth; `BE` the wire byte order; `full_range` the YCbCr-style
/// quantization range; `system` the BT.2020 CL resolution.
#[cfg_attr(not(tarpaulin), inline(always))]
fn cl_444p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = cl_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range, system);
    out[x * bpp] = narrow_unit_to_u8(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u8(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u8(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// One row of high-bit planar 4:4:4 CL `YcCbcCrc` → packed **native-depth u16
/// RGB** (`ALPHA = false`) or **RGBA** (`ALPHA = true`, opaque alpha
/// `(1 << BITS) - 1`). Narrowed to the `Yuv444pN` native range, the same
/// contract as the affine `yuv_444p_n_to_rgb_u16_row` family.
#[cfg_attr(not(tarpaulin), inline(always))]
fn cl_444p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  for x in 0..width {
    let rgb = cl_pixel_to_rgb_prime::<BITS, BE>(y[x], u[x], v[x], full_range, system);
    out[x * bpp] = narrow_unit_to_u16_native::<BITS>(rgb[0]);
    out[x * bpp + 1] = narrow_unit_to_u16_native::<BITS>(rgb[1]);
    out[x * bpp + 2] = narrow_unit_to_u16_native::<BITS>(rgb[2]);
    if ALPHA {
      out[x * bpp + 3] = ((1u32 << BITS) - 1) as u16;
    }
  }
}

/// High-bit planar 4:4:4 CL `YcCbcCrc` → packed **u8 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cl_444p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  cl_444p_n_to_rgb_or_rgba_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range, system);
}

/// High-bit planar 4:4:4 CL `YcCbcCrc` → packed **u8 RGBA** (opaque `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cl_444p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  cl_444p_n_to_rgb_or_rgba_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range, system);
}

/// High-bit planar 4:4:4 CL `YcCbcCrc` → packed native-depth **u16 RGB**.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cl_444p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  cl_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE>(y, u, v, rgb_out, width, full_range, system);
}

/// High-bit planar 4:4:4 CL `YcCbcCrc` → packed native-depth **u16 RGBA**
/// (opaque alpha `(1 << BITS) - 1`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn cl_444p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  full_range: bool,
  system: ClSystem,
) {
  cl_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE>(y, u, v, rgba_out, width, full_range, system);
}

#[cfg(all(test, feature = "std"))]
mod tests;
