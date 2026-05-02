//! Packed YUV 4:4:4 high-bit-depth frame types — Tier 5.
//!
//! This module is the container for the Tier 5 packed-YUV-4:4:4
//! family (`v410`, `xv36`, `vuya` / `vuyx`, `ayuv64`). Ship 12a
//! adds [`V410Frame`] and [`V30XFrame`] (sibling formats with opposite
//! padding positions); Ship 12b adds [`Xv36Frame`]; siblings land in
//! 12c / 12d.

use derive_more::IsVariant;
use thiserror::Error;

/// Validated wrapper around a packed YUV 4:4:4 10-bit `V410` plane.
///
/// `V410` is the **MSB-padded** packed YUV 4:4:4 layout — the same
/// bits Microsoft V410 fourcc, NVIDIA Video Codec SDK, Apple
/// AVFoundation, and the FFmpeg `AV_CODEC_ID_V410` codec all describe.
/// Current FFmpeg (8.1+) exposes this layout as `AV_PIX_FMT_XV30LE`
/// (the `AV_PIX_FMT_V410` symbol was renamed to `XV30` — same bit
/// pattern, new name). Each pixel occupies one 32-bit word with the
/// following little-endian layout (MSB → LSB):
///
/// | Bits  | Field |
/// |-------|-------|
/// | 31:30 | padding (zero) |
/// | 29:20 | V (10 bits) |
/// | 19:10 | Y (10 bits) |
/// | 9:0   | U (10 bits) |
///
/// **If your data uses LSB padding instead** (`AV_PIX_FMT_V30XLE`,
/// `(msb) 10V 10Y 10U 2X (lsb)`), use [`V30XFrame`] — it is a
/// type-distinct sibling with the same shape but shifted bit
/// positions.
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

/// Validated wrapper around a packed YUV 4:4:4 12-bit `XV36` plane.
///
/// `XV36` (FFmpeg `AV_PIX_FMT_XV36LE`) packs **four u16 channels per
/// pixel** as `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` little-endian. Each
/// channel uses the high 12 bits of its u16 with the low 4 bits zero
/// (MSB-aligned at 12-bit, same encoding as `Y212`). The `X` prefix
/// means the A slot is **padding** — reads are tolerated but values
/// are discarded; RGBA outputs always force α = max (`0xFF` u8 /
/// `0x0FFF` u16 native-depth).
///
/// Per-pixel layout (LE, MSB → LSB inside each channel u16):
///
/// | u16 slot | Field | Active bits |
/// |----------|-------|-------------|
/// | 0        | U     | bits[15:4]  |
/// | 1        | Y     | bits[15:4]  |
/// | 2        | V     | bits[15:4]  |
/// | 3        | A     | bits[15:4] (padding) |
///
/// Each row holds exactly `width × 4` u16 elements (`stride >=
/// width × 4`); the plane occupies `stride * height` u16 elements.
#[derive(Debug, Clone, Copy)]
pub struct Xv36Frame<'a> {
  packed: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`Xv36Frame::try_new`] and
/// [`Xv36Frame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Xv36FrameError {
  /// `width == 0` or `height == 0`.
  #[error("Xv36Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width × 4` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("Xv36Frame: width {width} × 4 overflows u32 (per-row u16 element count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `stride < width × 4` (u16 elements). Each row needs at least
  /// `width × 4` u16 elements (= `width × 8` bytes) to hold all
  /// pixels.
  #[error("Xv36Frame: stride {stride} u16 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in u16 elements (`width × 4`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short.
  #[error("Xv36Frame: plane too short: expected >= {expected} u16 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u16 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u16 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("Xv36Frame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
  /// `try_new_checked` only: a sample's low 4 bits are non-zero.
  /// Diagnoses callers feeding low-bit-packed data (e.g.
  /// `yuv444p12le` mistakenly handed to an XV36 path).
  #[error("Xv36Frame: sample with non-zero low 4 bits found; expected MSB-aligned data")]
  SampleLowBitsSet,
}

impl<'a> Xv36Frame<'a> {
  /// Validates and constructs an [`Xv36Frame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Xv36FrameError> {
    if width == 0 || height == 0 {
      return Err(Xv36FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(n) => n,
      None => return Err(Xv36FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Xv36FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(Xv36FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(Xv36FrameError::PlaneTooShort {
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

  /// Like [`Self::try_new`] but additionally rejects samples whose
  /// low 4 bits are non-zero. Validates the MSB-alignment invariant
  /// (low 4 bits zero per the XV36 encoding).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Xv36FrameError> {
    let frame = Self::try_new(packed, width, height, stride)?;
    let row_elems = (width * 4) as usize;
    let h = height as usize;
    let stride_us = stride as usize;
    for row in 0..h {
      let start = row * stride_us;
      for &sample in &packed[start..start + row_elems] {
        if sample & 0x000F != 0 {
          return Err(Xv36FrameError::SampleLowBitsSet);
        }
      }
    }
    Ok(frame)
  }

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V410Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        Xv36FrameError::ZeroDimension { .. } => panic!("invalid Xv36Frame: zero dimension"),
        Xv36FrameError::WidthOverflow { .. } => panic!("invalid Xv36Frame: width overflow"),
        Xv36FrameError::StrideTooSmall { .. } => panic!("invalid Xv36Frame: stride too small"),
        Xv36FrameError::PlaneTooShort { .. } => panic!("invalid Xv36Frame: plane too short"),
        Xv36FrameError::GeometryOverflow { .. } => panic!("invalid Xv36Frame: geometry overflow"),
        // SampleLowBitsSet is only emitted by try_new_checked.
        Xv36FrameError::SampleLowBitsSet => {
          panic!("invalid Xv36Frame: sample low bits set (unreachable from try_new)")
        }
      },
    }
  }

  /// Packed plane: `stride * height` total u16 elements, with
  /// `width × 4` active u16 elements per row (4 channels per pixel)
  /// and `stride` u16 elements per row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u16] {
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
  /// Stride in u16 elements (NOT bytes — the number of u16 slots per
  /// row, ≥ `width × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
