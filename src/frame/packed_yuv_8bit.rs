use derive_more::IsVariant;
use thiserror::Error;

// ============================================================
// Tier 3 — Packed YUV 4:2:2 8-bit source-side frames (Ship 10)
// ============================================================
//
// Three formats share the same layout: a single plane of 8-bit
// bytes, 2 pixels per 4-byte block, horizontal chroma 2:1
// subsampling. They differ only in byte ordering inside each block:
//
// - YUYV422 (YUY2)  — `[Y0, U0, Y1, V0]` per 2-pixel block
// - UYVY422 (UYVY)  — `[U0, Y0, V0, Y1]` per 2-pixel block
// - YVYU422 (YVYU)  — `[Y0, V0, Y1, U0]` per 2-pixel block
//
// Stride is in bytes and must be ≥ `2 * width`; width must be even
// (one chroma sample serves two adjacent Y pixels). The `try_new`
// constructors share an enum shape with the existing `Tier 6`
// packed-RGB frame types.

/// Errors returned by [`Yuyv422Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuyv422FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. Packed YUV 4:2:2 pairs two Y samples per
  /// chroma pair, so each 2-pixel block needs exactly 4 bytes —
  /// odd widths can't form a complete final block.
  #[error("width ({width}) is odd; packed YUV 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `stride < 2 * width`. Each row needs `2 * width` bytes
  /// (4 bytes per 2-pixel block).
  #[error("stride ({stride}) is smaller than 2 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride (`2 * width`).
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("YUYV plane has {actual} bytes but at least {expected} are required")]
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
  /// `2 * width` overflows `u32`.
  #[error("2 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **YUYV422** frame (`AV_PIX_FMT_YUYV422`,
/// also known as YUY2). Single plane, 4 bytes per 2-pixel block,
/// byte order `Y0, U0, Y1, V0` — Y in even byte positions, U/V in
/// odd positions with U preceding V.
///
/// `stride` is in **bytes** (≥ `2 * width`). `width` must be even.
#[derive(Debug, Clone, Copy)]
pub struct Yuyv422Frame<'a> {
  yuyv: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Yuyv422Frame<'a> {
  /// Constructs a new [`Yuyv422Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    yuyv: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Yuyv422FrameError> {
    if width == 0 || height == 0 {
      return Err(Yuyv422FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuyv422FrameError::OddWidth { width });
    }
    let min_stride = match width.checked_mul(2) {
      Some(v) => v,
      None => return Err(Yuyv422FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Yuyv422FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuyv422FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if yuyv.len() < plane_min {
      return Err(Yuyv422FrameError::PlaneTooShort {
        expected: plane_min,
        actual: yuyv.len(),
      });
    }
    Ok(Self {
      yuyv,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Yuyv422Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(yuyv: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(yuyv, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuyv422Frame dimensions or plane length"),
    }
  }

  /// Packed YUYV plane bytes (`Y0, U0, Y1, V0, Y2, U2, Y3, V2, …`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn yuyv(&self) -> &'a [u8] {
    self.yuyv
  }
  /// Frame width in pixels (even).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Errors returned by [`Uyvy422Frame::try_new`]. Variant shape
/// mirrors [`Yuyv422FrameError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Uyvy422FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd.
  #[error("width ({width}) is odd; packed YUV 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `stride < 2 * width`.
  #[error("stride ({stride}) is smaller than 2 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("UYVY plane has {actual} bytes but at least {expected} are required")]
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
  /// `2 * width` overflows `u32`.
  #[error("2 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **UYVY422** frame (`AV_PIX_FMT_UYVY422`).
/// Single plane, 4 bytes per 2-pixel block, byte order
/// `U0, Y0, V0, Y1` — Y in odd byte positions, U/V in even
/// positions. The de-facto SDI capture format on Apple
/// QuickTime / VideoToolbox 8-bit paths.
///
/// `stride` is in **bytes** (≥ `2 * width`). `width` must be even.
#[derive(Debug, Clone, Copy)]
pub struct Uyvy422Frame<'a> {
  uyvy: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Uyvy422Frame<'a> {
  /// Constructs a new [`Uyvy422Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    uyvy: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Uyvy422FrameError> {
    if width == 0 || height == 0 {
      return Err(Uyvy422FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Uyvy422FrameError::OddWidth { width });
    }
    let min_stride = match width.checked_mul(2) {
      Some(v) => v,
      None => return Err(Uyvy422FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Uyvy422FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Uyvy422FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if uyvy.len() < plane_min {
      return Err(Uyvy422FrameError::PlaneTooShort {
        expected: plane_min,
        actual: uyvy.len(),
      });
    }
    Ok(Self {
      uyvy,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Uyvy422Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(uyvy: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(uyvy, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Uyvy422Frame dimensions or plane length"),
    }
  }

  /// Packed UYVY plane bytes (`U0, Y0, V0, Y1, U2, Y2, V2, Y3, …`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uyvy(&self) -> &'a [u8] {
    self.uyvy
  }
  /// Frame width in pixels (even).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Errors returned by [`Yvyu422Frame::try_new`]. Variant shape
/// mirrors [`Yuyv422FrameError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yvyu422FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd.
  #[error("width ({width}) is odd; packed YUV 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `stride < 2 * width`.
  #[error("stride ({stride}) is smaller than 2 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("YVYU plane has {actual} bytes but at least {expected} are required")]
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
  /// `2 * width` overflows `u32`.
  #[error("2 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **YVYU422** frame (`AV_PIX_FMT_YVYU422`).
/// Single plane, 4 bytes per 2-pixel block, byte order
/// `Y0, V0, Y1, U0` — same Y positions as YUYV but with V/U
/// swapped relative to YUYV (V precedes U). Common on Android
/// camera HAL outputs.
///
/// `stride` is in **bytes** (≥ `2 * width`). `width` must be even.
#[derive(Debug, Clone, Copy)]
pub struct Yvyu422Frame<'a> {
  yvyu: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Yvyu422Frame<'a> {
  /// Constructs a new [`Yvyu422Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    yvyu: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Yvyu422FrameError> {
    if width == 0 || height == 0 {
      return Err(Yvyu422FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yvyu422FrameError::OddWidth { width });
    }
    let min_stride = match width.checked_mul(2) {
      Some(v) => v,
      None => return Err(Yvyu422FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Yvyu422FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yvyu422FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if yvyu.len() < plane_min {
      return Err(Yvyu422FrameError::PlaneTooShort {
        expected: plane_min,
        actual: yvyu.len(),
      });
    }
    Ok(Self {
      yvyu,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Yvyu422Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(yvyu: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(yvyu, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yvyu422Frame dimensions or plane length"),
    }
  }

  /// Packed YVYU plane bytes (`Y0, V0, Y1, U0, Y2, V2, Y3, U2, …`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn yvyu(&self) -> &'a [u8] {
    self.yvyu
  }
  /// Frame width in pixels (even).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
