use derive_more::IsVariant;
use thiserror::Error;

/// Errors returned by [`X2Rgb10Frame::try_new`]. Variant shape mirrors
/// the [`RgbaFrameError`] family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum X2Rgb10FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 4 * width`.
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("X2RGB10 plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **X2RGB10** frame
/// (`AV_PIX_FMT_X2RGB10LE`) — 4 bytes per pixel, 32-bit
/// little-endian word with `(MSB) 2X | 10R | 10G | 10B (LSB)`. The
/// top 2 bits are **ignored padding**; each colour channel carries
/// 10 active bits.
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct X2Rgb10Frame<'a> {
  x2rgb10: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> X2Rgb10Frame<'a> {
  /// Constructs a new [`X2Rgb10Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    x2rgb10: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, X2Rgb10FrameError> {
    if width == 0 || height == 0 {
      return Err(X2Rgb10FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(X2Rgb10FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(X2Rgb10FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(X2Rgb10FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if x2rgb10.len() < plane_min {
      return Err(X2Rgb10FrameError::PlaneTooShort {
        expected: plane_min,
        actual: x2rgb10.len(),
      });
    }
    Ok(Self {
      x2rgb10,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`X2Rgb10Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(x2rgb10: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(x2rgb10, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid X2Rgb10Frame dimensions or plane length"),
    }
  }

  /// Packed X2RGB10 plane bytes — each 4-byte group is one
  /// little-endian `u32` word.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn x2rgb10(&self) -> &'a [u8] {
    self.x2rgb10
  }
  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride (`>= 4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Errors returned by [`X2Bgr10Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum X2Bgr10FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 4 * width`.
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("X2BGR10 plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **X2BGR10** frame
/// (`AV_PIX_FMT_X2BGR10LE`) — 4 bytes per pixel, 32-bit
/// little-endian word with `(MSB) 2X | 10B | 10G | 10R (LSB)`.
/// Channel positions are reversed relative to [`X2Rgb10Frame`].
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct X2Bgr10Frame<'a> {
  x2bgr10: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> X2Bgr10Frame<'a> {
  /// Constructs a new [`X2Bgr10Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    x2bgr10: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, X2Bgr10FrameError> {
    if width == 0 || height == 0 {
      return Err(X2Bgr10FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(X2Bgr10FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(X2Bgr10FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(X2Bgr10FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if x2bgr10.len() < plane_min {
      return Err(X2Bgr10FrameError::PlaneTooShort {
        expected: plane_min,
        actual: x2bgr10.len(),
      });
    }
    Ok(Self {
      x2bgr10,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`X2Bgr10Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(x2bgr10: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(x2bgr10, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid X2Bgr10Frame dimensions or plane length"),
    }
  }

  /// Packed X2BGR10 plane bytes — each 4-byte group is one
  /// little-endian `u32` word.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn x2bgr10(&self) -> &'a [u8] {
    self.x2bgr10
  }
  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride (`>= 4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
