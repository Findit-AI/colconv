//! Packed YUV 4:4:4 high-bit-depth frame types — Tier 5.
//!
//! This module is the container for the Tier 5 packed-YUV-4:4:4
//! family (`v410`, `xv36`, `vuya` / `vuyx`, `ayuv64`). Ship 12a
//! adds [`V410Frame`] and [`V30XFrame`] (sibling formats with opposite
//! padding positions); siblings land in 12b / 12c / 12d.

use derive_more::IsVariant;
use thiserror::Error;

/// Validated wrapper around a packed YUV 4:4:4 10-bit `V410` plane.
///
/// `V410` (FFmpeg `AV_PIX_FMT_V410`, also known as `XV30`) packs **one
/// pixel per 32-bit word** with the following little-endian layout
/// (MSB → LSB):
///
/// | Bits  | Field |
/// |-------|-------|
/// | 31:30 | padding (zero) |
/// | 29:20 | V (10 bits) |
/// | 19:10 | Y (10 bits) |
/// | 9:0   | U (10 bits) |
///
/// Each row holds exactly `width` u32 words (`stride >= width`); the
/// plane occupies `stride * height` u32 elements.
#[derive(Debug, Clone, Copy)]
pub struct V410Frame<'a> {
  packed: &'a [u32],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`V410Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum V410FrameError {
  /// `width == 0` or `height == 0`.
  #[error("V410Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `stride < width`. Each row needs at least `width` u32 words.
  #[error("V410Frame: stride {stride} u32 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride (= `width`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry.
  #[error("V410Frame: plane too short: expected >= {expected} u32 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u32 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u32 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("V410Frame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a> V410Frame<'a> {
  /// Validates and constructs a [`V410Frame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u32],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, V410FrameError> {
    if width == 0 || height == 0 {
      return Err(V410FrameError::ZeroDimension { width, height });
    }
    if stride < width {
      return Err(V410FrameError::StrideTooSmall {
        min_stride: width,
        stride,
      });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(V410FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(V410FrameError::PlaneTooShort {
        expected: plane_min,
        actual: packed.len(),
      });
    }
    Ok(Self {
      packed,
      width,
      height,
      stride,
    })
  }

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V210Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u32], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        V410FrameError::ZeroDimension { .. } => panic!("invalid V410Frame: zero dimension"),
        V410FrameError::StrideTooSmall { .. } => panic!("invalid V410Frame: stride too small"),
        V410FrameError::PlaneTooShort { .. } => panic!("invalid V410Frame: plane too short"),
        V410FrameError::GeometryOverflow { .. } => panic!("invalid V410Frame: geometry overflow"),
      },
    }
  }

  /// Packed plane: `stride * height` total u32 elements, with
  /// `width` active pixels per row and `stride` u32 elements per
  /// row. Each word holds one pixel `(U, Y, V, padding)` per the
  /// V410 layout described above.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u32] {
    self.packed
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
  /// Stride in u32 elements (NOT bytes — the number of u32 slots
  /// per row, ≥ `width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Validated wrapper around a packed YUV 4:4:4 10-bit `V30X` plane.
///
/// `V30X` (FFmpeg `AV_PIX_FMT_V30XLE`) packs **one pixel per 32-bit word**
/// with the following little-endian layout (MSB → LSB):
///
/// | Bits  | Field |
/// |-------|-------|
/// | 31:22 | V (10 bits) |
/// | 21:12 | Y (10 bits) |
/// | 11:2  | U (10 bits) |
/// | 1:0   | padding (zero) |
///
/// This is a sibling of [`V410Frame`]: the pixel data is identical but
/// V30X places the 2-bit padding at the **LSB** (bits \[1:0\]), whereas V410
/// places it at the **MSB** (bits \[31:30\]). Bit-extraction shifts differ by
/// exactly 2.
///
/// Each row holds exactly `width` u32 words (`stride >= width`); the
/// plane occupies `stride * height` u32 elements.
#[derive(Debug, Clone, Copy)]
pub struct V30XFrame<'a> {
  packed: &'a [u32],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`V30XFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum V30XFrameError {
  /// `width == 0` or `height == 0`.
  #[error("V30XFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `stride < width`. Each row needs at least `width` u32 words.
  #[error("V30XFrame: stride {stride} u32 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride (= `width`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry.
  #[error("V30XFrame: plane too short: expected >= {expected} u32 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u32 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u32 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("V30XFrame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a> V30XFrame<'a> {
  /// Validates and constructs a [`V30XFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u32],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, V30XFrameError> {
    if width == 0 || height == 0 {
      return Err(V30XFrameError::ZeroDimension { width, height });
    }
    if stride < width {
      return Err(V30XFrameError::StrideTooSmall {
        min_stride: width,
        stride,
      });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(V30XFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(V30XFrameError::PlaneTooShort {
        expected: plane_min,
        actual: packed.len(),
      });
    }
    Ok(Self {
      packed,
      width,
      height,
      stride,
    })
  }

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V210Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u32], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        V30XFrameError::ZeroDimension { .. } => panic!("invalid V30XFrame: zero dimension"),
        V30XFrameError::StrideTooSmall { .. } => panic!("invalid V30XFrame: stride too small"),
        V30XFrameError::PlaneTooShort { .. } => panic!("invalid V30XFrame: plane too short"),
        V30XFrameError::GeometryOverflow { .. } => panic!("invalid V30XFrame: geometry overflow"),
      },
    }
  }

  /// Packed plane: `stride * height` total u32 elements, with
  /// `width` active pixels per row and `stride` u32 elements per
  /// row. Each word holds one pixel `(U, Y, V, padding)` per the
  /// V30X layout described above.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u32] {
    self.packed
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
  /// Stride in u32 elements (NOT bytes — the number of u32 slots
  /// per row, ≥ `width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
