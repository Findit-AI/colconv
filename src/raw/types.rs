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
/// is `#[non_exhaustive]` so future variants (e.g. Malvar-He-Cutler
/// for sharper output) can land without a breaking change.
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
/// Each gain is a positive `f32` multiplier applied to the
/// corresponding raw color channel before the [`ColorCorrectionMatrix`]
/// is applied. Source: camera metadata (`WB_RGGB_LEVELS` family, RED
/// `Kelvin` / `Tint` resolved to gains by the SDK, BRAW
/// `whiteBalanceKelvin` resolved similarly).
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
  /// Constructs a [`WhiteBalance`] from explicit R / G / B gains.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(r: f32, g: f32, b: f32) -> Self {
    Self { r, g, b }
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCorrectionMatrix {
  m: [[f32; 3]; 3],
}

impl ColorCorrectionMatrix {
  /// Constructs a [`ColorCorrectionMatrix`] from a row-major 3×3.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(m: [[f32; 3]; 3]) -> Self {
    Self { m }
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
mod tests {
  use super::*;

  #[test]
  fn white_balance_neutral_is_default() {
    assert_eq!(WhiteBalance::default(), WhiteBalance::neutral());
    assert_eq!(WhiteBalance::neutral().r(), 1.0);
    assert_eq!(WhiteBalance::neutral().g(), 1.0);
    assert_eq!(WhiteBalance::neutral().b(), 1.0);
  }

  #[test]
  fn ccm_identity_is_default() {
    assert_eq!(
      ColorCorrectionMatrix::default(),
      ColorCorrectionMatrix::identity()
    );
    let id = ColorCorrectionMatrix::identity();
    let m = id.as_array();
    assert_eq!(m[0], [1.0, 0.0, 0.0]);
    assert_eq!(m[1], [0.0, 1.0, 0.0]);
    assert_eq!(m[2], [0.0, 0.0, 1.0]);
  }

  #[test]
  fn fuse_wb_ccm_with_neutral_wb_returns_ccm() {
    let ccm = ColorCorrectionMatrix::new([[1.0, 0.5, 0.25], [0.0, 0.8, 0.2], [0.1, 0.1, 0.7]]);
    let m = fuse_wb_ccm(&WhiteBalance::neutral(), &ccm);
    assert_eq!(&m, ccm.as_array());
  }

  #[test]
  fn fuse_wb_ccm_with_identity_ccm_returns_diag_wb() {
    let wb = WhiteBalance::new(1.5, 1.0, 2.0);
    let m = fuse_wb_ccm(&wb, &ColorCorrectionMatrix::identity());
    assert_eq!(m, [[1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
  }

  #[test]
  fn fuse_wb_ccm_scales_columns_by_wb() {
    // M = CCM · diag(wb) ⇒ column j of M is column j of CCM × wb_j.
    let ccm =
      ColorCorrectionMatrix::new([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0]]);
    let wb = WhiteBalance::new(0.5, 1.0, 0.25);
    let m = fuse_wb_ccm(&wb, &ccm);
    assert_eq!(m[0], [0.5, 2.0, 1.0]);
    assert_eq!(m[1], [4.0, 16.0, 8.0]);
    assert_eq!(m[2], [32.0, 128.0, 64.0]);
  }
}
