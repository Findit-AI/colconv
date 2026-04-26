//! Shared parameter types for the RAW (Bayer) family.
//!
//! Bayer demosaic kernels need three caller-supplied parameters that
//! aren't part of the source frame itself:
//!
//! - [`BayerPattern`] — how the four-color repeat tiles the sensor.
//! - [`WhiteBalance`] — per-channel gains (R / G / B) the camera (or
//!   the user) computed from a white reference.
//! - [`ColorCorrectionMatrix`] — the 3×3 RGB→RGB transform from the
//!   sensor's native primaries into a working space (sRGB / Rec.709 /
//!   Rec.2020). RED, BMD, and Nikon SDKs all hand this back as a
//!   3×3.
//!
//! The walker fuses `wb` and `ccm` into a single 3×3 transform
//! (`M = CCM · diag(wb)`) before dispatching to the per-row kernel,
//! so the per-pixel arithmetic is one 3×3 matmul, not two passes.

use derive_more::IsVariant;
use thiserror::Error;

/// Bayer pattern — which sensor color sits at the top-left of the
/// repeating 2×2 tile.
///
/// In BGGR / RGGB the green diagonal runs top-left → bottom-right; in
/// GRBG / GBRG the green diagonal runs top-right → bottom-left. Each
/// 2×2 cell carries two greens (one on the red row, one on the blue
/// row), one red, and one blue.
///
/// Source: read from the camera's metadata (R3D `ImagerCFA`, BRAW
/// `cfa_pattern`, NRAW SDK accessor). FFmpeg's bayer pixel formats
/// (`AV_PIX_FMT_BAYER_BGGR8` / `RGGB8` / `GRBG8` / `GBRG8` and the
/// `*_16LE` siblings) carry the pattern in the format identifier
/// itself.
///
/// **Scope.** This enum covers the four standard 2×2 Bayer
/// arrangements only. Other CFA families used by modern
/// professional cameras (Quad Bayer / Sony, X-Trans / Fujifilm,
/// RGBW / BMD URSA 12K, Foveon stacked photosites / Sigma,
/// monochrome / Leica) are tracked separately as future RAW
/// pixel-buffer types — they need different walker shapes
/// and / or completely different demosaic algorithms, so they
/// won't ride on this enum. See
/// `docs/color-conversion-functions.md` § "Cleanup follow-ups
/// → Tier 14 RAW family extensions" for the full roadmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum BayerPattern {
  /// `B G / G R` — top-left is **B**, bottom-right is **R**.
  Bggr,
  /// `R G / G B` — top-left is **R**, bottom-right is **B**.
  Rggb,
  /// `G R / B G` — top-left is **G** (on the red row), top-right is
  /// **R**.
  Grbg,
  /// `G B / R G` — top-left is **G** (on the blue row), top-right is
  /// **B**.
  Gbrg,
}

/// Demosaic algorithm.
///
/// Selects the per-pixel reconstruction kernel the walker uses to
/// fill in the two missing color channels at each Bayer site.
///
/// Currently only [`BayerDemosaic::Bilinear`] is wired up. The enum
/// is `#[non_exhaustive]` so future variants (Malvar-He-Cutler /
/// MHC for sharper output, DCB / VNG / AHD for edge-aware
/// high-quality reconstruction) can land without a breaking
/// change. The MHC variant is the smallest next step (5-row
/// window, ~3× bilinear cost); DCB / VNG / AHD are larger
/// follow-ups that need a different walker shape than the per-row
/// model. See `docs/color-conversion-functions.md` §
/// "Cleanup follow-ups → Higher-quality Bayer demosaic algorithms"
/// for the full design notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IsVariant)]
#[non_exhaustive]
pub enum BayerDemosaic {
  /// Bilinear demosaic — 3×3 row window, 4-tap horizontal/vertical
  /// average for the missing color channels. Soft but fast and
  /// numerically stable; the standard "first pass" reconstruction.
  #[default]
  Bilinear,
}

/// Per-channel white-balance gains.
///
/// Each gain is a **finite, non-negative** `f32` multiplier applied
/// to the corresponding raw color channel before the
/// [`ColorCorrectionMatrix`] is applied. Source: camera metadata
/// (`WB_RGGB_LEVELS` family, RED `Kelvin` / `Tint` resolved to
/// gains by the SDK, BRAW `whiteBalanceKelvin` resolved similarly).
/// [`WhiteBalance::try_new`] enforces the invariant; any NaN, ±∞,
/// or negative gain is rejected via [`WhiteBalanceError`].
///
/// Zero is permitted (zeroes that channel — degenerate but
/// well-defined).
///
/// A neutral [`WhiteBalance::neutral`] (`R = G = B = 1.0`) means
/// "no white-balance correction" — the sensor's native primaries are
/// passed through unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhiteBalance {
  r: f32,
  g: f32,
  b: f32,
}

impl WhiteBalance {
  /// Constructs a [`WhiteBalance`] from explicit R / G / B gains,
  /// validating that each is **finite and non-negative**. Camera
  /// metadata pipelines occasionally surface NaN / ±∞ (failed Kelvin
  /// → gain conversions, missing sensor metadata) and a single such
  /// value would propagate through the fused 3×3 transform and
  /// produce silently-corrupt output (NaN clamps to 0 on cast,
  /// turning unrelated channels black). Reject upstream instead.
  ///
  /// Returns [`WhiteBalanceError`] if any gain is non-finite or
  /// negative. A gain of `0` is permitted (zeroes out that channel —
  /// degenerate but well-defined).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(r: f32, g: f32, b: f32) -> Result<Self, WhiteBalanceError> {
    if !r.is_finite() {
      return Err(WhiteBalanceError::NonFinite {
        channel: WbChannel::R,
        value: r,
      });
    }
    if !g.is_finite() {
      return Err(WhiteBalanceError::NonFinite {
        channel: WbChannel::G,
        value: g,
      });
    }
    if !b.is_finite() {
      return Err(WhiteBalanceError::NonFinite {
        channel: WbChannel::B,
        value: b,
      });
    }
    if r < 0.0 {
      return Err(WhiteBalanceError::Negative {
        channel: WbChannel::R,
        value: r,
      });
    }
    if g < 0.0 {
      return Err(WhiteBalanceError::Negative {
        channel: WbChannel::G,
        value: g,
      });
    }
    if b < 0.0 {
      return Err(WhiteBalanceError::Negative {
        channel: WbChannel::B,
        value: b,
      });
    }
    // Magnitude bound. Real WB gains rarely exceed 10× (extreme
    // tungsten correction); the bound is generous (`1e6`) but
    // closes the door on finite-but-pathological metadata that
    // would overflow per-pixel f32 math during the matmul. With
    // gains ≤ 1e6 and 16-bit samples (≤ 65535) and CCM coefficients
    // bounded by [`ColorCorrectionMatrix::MAX_COEFFICIENT`],
    // the largest per-channel sum stays well under `f32::MAX`,
    // so the kernel can never produce Inf or NaN from validated
    // inputs.
    if r > Self::MAX_GAIN {
      return Err(WhiteBalanceError::OutOfBounds {
        channel: WbChannel::R,
        value: r,
        max: Self::MAX_GAIN,
      });
    }
    if g > Self::MAX_GAIN {
      return Err(WhiteBalanceError::OutOfBounds {
        channel: WbChannel::G,
        value: g,
        max: Self::MAX_GAIN,
      });
    }
    if b > Self::MAX_GAIN {
      return Err(WhiteBalanceError::OutOfBounds {
        channel: WbChannel::B,
        value: b,
        max: Self::MAX_GAIN,
      });
    }
    Ok(Self { r, g, b })
  }

  /// Maximum permitted gain magnitude. `1e6` is far above any
  /// realistic camera-metadata value (real WB gains are O(1–10))
  /// and far below the value at which per-pixel f32 matmul could
  /// overflow given sample range `[0, 65535]` and CCM coefficient
  /// bounds — see [`Self::try_new`] for the full overflow analysis.
  pub const MAX_GAIN: f32 = 1.0e6;

  /// Constructs a [`WhiteBalance`], panicking on invalid input.
  /// Prefer [`Self::try_new`] when gains may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(r: f32, g: f32, b: f32) -> Self {
    match Self::try_new(r, g, b) {
      Ok(wb) => wb,
      Err(_) => panic!("invalid WhiteBalance gains (non-finite, negative, or > MAX_GAIN)"),
    }
  }

  /// Neutral white-balance (`R = G = B = 1.0`) — sensor primaries
  /// pass through unchanged.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn neutral() -> Self {
    Self {
      r: 1.0,
      g: 1.0,
      b: 1.0,
    }
  }

  /// Red-channel gain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> f32 {
    self.r
  }

  /// Green-channel gain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> f32 {
    self.g
  }

  /// Blue-channel gain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> f32 {
    self.b
  }
}

impl Default for WhiteBalance {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::neutral()
  }
}

/// 3×3 color-correction matrix applied after white balance.
///
/// Maps the sensor's white-balanced RGB into a target working space
/// (sRGB / Rec.709 / Rec.2020). Stored row-major: `m[i][j]` is the
/// coefficient of the input column `j` contributing to the output
/// channel `i`. Applying the matrix to an input vector
/// `[R_in, G_in, B_in]` yields:
///
/// ```text
///   R_out = m[0][0]*R_in + m[0][1]*G_in + m[0][2]*B_in
///   G_out = m[1][0]*R_in + m[1][1]*G_in + m[1][2]*B_in
///   B_out = m[2][0]*R_in + m[2][1]*G_in + m[2][2]*B_in
/// ```
///
/// A neutral [`ColorCorrectionMatrix::identity`] (1.0 on the
/// diagonal, 0 off) means "no color correction" — the
/// white-balanced sensor RGB is passed through.
///
/// Source: RED / BMD / Nikon SDKs hand a 3×3 back natively.
///
/// **Color-space note.** This matrix is *opaque* about the target
/// gamut — the caller decides whether the output is in Rec.709 /
/// Rec.2020 / DCI-P3 / ACES AP0 or AP1 / sensor-native primaries
/// by choosing the coefficients accordingly. The output is always
/// **scene-linear** (no transfer-function / log / gamma encoding
/// applied; the demosaic kernel does linear arithmetic).
/// Downstream gamut transforms and transfer-function encoding
/// (sRGB, Rec.709 OETF, log, HLG, PQ) are not in `colconv`'s
/// current scope — typically handled via OCIO or a dedicated
/// tonemap layer. See `docs/color-conversion-functions.md` §
/// "Cleanup follow-ups → Color-space handling" for the deferred
/// in-crate convenience-layer roadmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCorrectionMatrix {
  m: [[f32; 3]; 3],
}

impl ColorCorrectionMatrix {
  /// Constructs a [`ColorCorrectionMatrix`] from a row-major 3×3,
  /// validating that every element is **finite** (not NaN, not
  /// ±∞) and bounded by `|value| <= [`Self::MAX_COEFFICIENT_ABS`]
  /// (= 1e6). CCM elements may legitimately be negative (color
  /// matrices regularly subtract crosstalk), and the magnitude
  /// bound is well above any realistic camera value (real CCMs
  /// are O(1–5)) but closes the door on finite-but-pathological
  /// metadata that would overflow per-pixel f32 math.
  ///
  /// Returns [`ColorCorrectionMatrixError`] on the first
  /// out-of-spec element, naming its `(row, col)` coordinates.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(m: [[f32; 3]; 3]) -> Result<Self, ColorCorrectionMatrixError> {
    let mut row = 0;
    while row < 3 {
      let mut col = 0;
      while col < 3 {
        let v = m[row][col];
        if !v.is_finite() {
          return Err(ColorCorrectionMatrixError::NonFinite { row, col, value: v });
        }
        // Magnitude bound — see the type-level docs for the
        // overflow analysis. With `|coeff| <= 1e6`, gain ≤ 1e6,
        // and sample range `[0, 65535]`, the largest per-channel
        // sum is `3 * 1e6 * 1e6 * 65535 ≈ 1.97e17`, ~21 orders
        // of magnitude under `f32::MAX ≈ 3.4e38`. No Inf, no NaN.
        if !(v >= -Self::MAX_COEFFICIENT_ABS && v <= Self::MAX_COEFFICIENT_ABS) {
          return Err(ColorCorrectionMatrixError::OutOfBounds {
            row,
            col,
            value: v,
            max_abs: Self::MAX_COEFFICIENT_ABS,
          });
        }
        col += 1;
      }
      row += 1;
    }
    Ok(Self { m })
  }

  /// Maximum permitted absolute value of any CCM element. `1e6`
  /// is far above any realistic camera-metadata value (real CCMs
  /// are O(1–5)) and closes the door on finite-but-pathological
  /// metadata. See [`Self::try_new`] for the overflow analysis.
  pub const MAX_COEFFICIENT_ABS: f32 = 1.0e6;

  /// Constructs a [`ColorCorrectionMatrix`], panicking on invalid
  /// input. Prefer [`Self::try_new`] when matrix elements may be
  /// invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(m: [[f32; 3]; 3]) -> Self {
    match Self::try_new(m) {
      Ok(ccm) => ccm,
      Err(_) => panic!(
        "invalid ColorCorrectionMatrix element (non-finite or |value| > MAX_COEFFICIENT_ABS)"
      ),
    }
  }

  /// The identity matrix — no color correction. Equivalent to
  /// passing the white-balanced sensor RGB straight through.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn identity() -> Self {
    Self {
      m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
  }

  /// Borrows the underlying row-major 3×3.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_array(&self) -> &[[f32; 3]; 3] {
    &self.m
  }
}

impl Default for ColorCorrectionMatrix {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::identity()
  }
}

/// Identifies which white-balance channel failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum WbChannel {
  /// Red gain.
  R,
  /// Green gain.
  G,
  /// Blue gain.
  B,
}

/// Errors returned by [`WhiteBalance::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum WhiteBalanceError {
  /// A gain is non-finite (NaN, +∞, or -∞).
  #[error("WhiteBalance.{channel:?} is non-finite (got {value})")]
  NonFinite {
    /// Which channel failed validation.
    channel: WbChannel,
    /// The offending gain value.
    value: f32,
  },
  /// A gain is negative. Zero is allowed (zeroes the channel).
  #[error("WhiteBalance.{channel:?} is negative (got {value})")]
  Negative {
    /// Which channel failed validation.
    channel: WbChannel,
    /// The offending gain value.
    value: f32,
  },
  /// A gain exceeds [`WhiteBalance::MAX_GAIN`] (`1e6`). The bound
  /// is far above any realistic camera value but closes the door
  /// on finite-but-pathological metadata that would overflow
  /// per-pixel f32 matmul.
  #[error("WhiteBalance.{channel:?} = {value} exceeds the magnitude bound ({max})")]
  OutOfBounds {
    /// Which channel failed validation.
    channel: WbChannel,
    /// The offending gain value.
    value: f32,
    /// The bound that was exceeded ([`WhiteBalance::MAX_GAIN`]).
    max: f32,
  },
}

/// Errors returned by [`ColorCorrectionMatrix::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum ColorCorrectionMatrixError {
  /// An element is non-finite (NaN, +∞, or -∞).
  #[error("ColorCorrectionMatrix[{row}][{col}] is non-finite (got {value})")]
  NonFinite {
    /// Row index of the offending element (0..3).
    row: usize,
    /// Column index of the offending element (0..3).
    col: usize,
    /// The offending value.
    value: f32,
  },
  /// An element's absolute value exceeds
  /// [`ColorCorrectionMatrix::MAX_COEFFICIENT_ABS`] (`1e6`). The
  /// bound is far above any realistic camera value but closes the
  /// door on finite-but-pathological metadata.
  #[error(
    "ColorCorrectionMatrix[{row}][{col}] = {value} exceeds the magnitude bound (|coeff| ≤ {max_abs})"
  )]
  OutOfBounds {
    /// Row index of the offending element (0..3).
    row: usize,
    /// Column index of the offending element (0..3).
    col: usize,
    /// The offending value.
    value: f32,
    /// The bound that was exceeded
    /// ([`ColorCorrectionMatrix::MAX_COEFFICIENT_ABS`]).
    max_abs: f32,
  },
}

/// Internal: fuse white-balance and CCM into a single 3×3 transform
/// `M = CCM · diag(wb)`. The walker calls this once per frame; the
/// per-row kernel applies a single 3×3 matmul per pixel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn fuse_wb_ccm(wb: &WhiteBalance, ccm: &ColorCorrectionMatrix) -> [[f32; 3]; 3] {
  let m = ccm.as_array();
  let (wr, wg, wb_) = (wb.r(), wb.g(), wb.b());
  [
    [m[0][0] * wr, m[0][1] * wg, m[0][2] * wb_],
    [m[1][0] * wr, m[1][1] * wg, m[1][2] * wb_],
    [m[2][0] * wr, m[2][1] * wg, m[2][2] * wb_],
  ]
}

#[cfg(test)]
mod tests;
