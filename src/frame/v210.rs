//! Packed `v210` 4:2:2 frame type — 10-bit YUV in a custom 32-bit
//! word packing. Each 16-byte word holds 12 × 10-bit samples = 6
//! pixels (4:2:2: 6 Y + 3 Cb + 3 Cr).
//!
//! Layout per word (little-endian):
//!
//! | Word | bits[9:0] | bits[19:10] | bits[29:20] | bits[31:30] |
//! |------|-----------|-------------|-------------|-------------|
//! | 0    | Cb₀       | Y₀          | Cr₀         | unused      |
//! | 1    | Y₁        | Cb₁         | Y₂          | unused      |
//! | 2    | Cr₁       | Y₃          | Cb₂         | unused      |
//! | 3    | Y₄        | Cr₂         | Y₅          | unused      |
//!
//! De-facto pro-broadcast standard for 10-bit SDI capture (DeckLink,
//! Kona, AJA, etc.). Used in BlackmagicDesign tooling, ProRes
//! intermediate workflows, and most DIT pipelines.

use derive_more::IsVariant;
use thiserror::Error;

/// Validated wrapper around a packed `v210` plane.
///
/// Construct via [`Self::try_new`] (fallible) or [`Self::new`]
/// (panics on invalid input).
#[derive(Debug, Clone, Copy)]
pub struct V210Frame<'a> {
  v210: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`V210Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum V210FrameError {
  /// `width == 0` or `height == 0`.
  #[error("V210Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width % 6 != 0`. v210 packs 6 pixels per 16-byte word, so the
  /// kernel only handles widths divisible by 6.
  #[error("V210Frame: width {width} is not a multiple of 6 (v210 packs 6 pixels per 16-byte word)")]
  WidthNotMultipleOf6 {
    /// Configured width.
    width: u32,
  },
  /// `stride < (width / 6) * 16`. Each row needs at least
  /// `(width / 6) * 16` bytes to hold all pixels.
  #[error("V210Frame: stride {stride} is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in bytes (`(width / 6) * 16`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `v210.len() < expected`. The packed plane is too short for the
  /// declared geometry.
  #[error("V210Frame: plane too short: expected >= {expected} bytes, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in bytes (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length.
    actual: usize,
  },
  /// `stride * height` overflows `u32`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("V210Frame: stride×height overflows u32 (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
  /// `(width / 6) * 16` overflows `u32`. Only reachable on 32-bit
  /// targets with extreme widths.
  #[error("V210Frame: row size in bytes ((width / 6) × 16) overflows u32")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
}

impl<'a> V210Frame<'a> {
  /// Validates and constructs a [`V210Frame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    v210: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, V210FrameError> {
    if width == 0 || height == 0 {
      return Err(V210FrameError::ZeroDimension { width, height });
    }
    if !width.is_multiple_of(6) {
      return Err(V210FrameError::WidthNotMultipleOf6 { width });
    }
    let words = width / 6;
    let min_stride = match words.checked_mul(16) {
      Some(n) => n,
      None => return Err(V210FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(V210FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(V210FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if v210.len() < plane_min {
      return Err(V210FrameError::PlaneTooShort {
        expected: plane_min,
        actual: v210.len(),
      });
    }
    Ok(Self {
      v210,
      width,
      height,
      stride,
    })
  }

  /// Panicking convenience over [`Self::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(v210: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(v210, width, height, stride) {
      Ok(f) => f,
      Err(e) => {
        // const-context-compatible panic message.
        match e {
          V210FrameError::ZeroDimension { .. } => panic!("invalid V210Frame: zero dimension"),
          V210FrameError::WidthNotMultipleOf6 { .. } => {
            panic!("invalid V210Frame: width not multiple of 6")
          }
          V210FrameError::StrideTooSmall { .. } => panic!("invalid V210Frame: stride too small"),
          V210FrameError::PlaneTooShort { .. } => panic!("invalid V210Frame: plane too short"),
          V210FrameError::GeometryOverflow { .. } => {
            panic!("invalid V210Frame: geometry overflow")
          }
          V210FrameError::WidthOverflow { .. } => panic!("invalid V210Frame: width overflow"),
        }
      }
    }
  }

  /// Packed plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v210(&self) -> &'a [u8] {
    self.v210
  }
  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Stride in bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
