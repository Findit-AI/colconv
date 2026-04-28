//! Validated source-frame types.
//!
//! Each pixel family has its own frame struct carrying the backing
//! plane slice(s), pixel dimensions, and byte strides. Construction
//! validates strides vs. widths and that each plane covers its
//! declared area.

use derive_more::{Display, IsVariant};
use thiserror::Error;

/// A validated YUV 4:2:0 planar frame.
///
/// Three planes:
/// - `y` — full-size luma, `y_stride >= width`, length `>= y_stride * height`.
/// - `u` / `v` — half-width, half-height chroma,
///   `u_stride >= (width + 1) / 2`, length `>= u_stride * ((height + 1) / 2)`.
///
/// `width` must be even (4:2:0 subsamples chroma 2:1 in width, and the
/// SIMD kernels assume `width & 1 == 0`). `height` may be odd — chroma
/// row sizing uses `height.div_ceil(2)` and the row walker maps Y row
/// `r` to chroma row `r / 2`, so the final Y row of an odd-height
/// frame shares chroma with its single chroma row. Odd-width input is
/// rejected at construction.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a> Yuv420pFrame<'a> {
  /// Constructs a new [`Yuv420pFrame`], validating dimensions and
  /// plane lengths.
  ///
  /// Returns [`Yuv420pFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd (odd height is allowed and handled via
  ///   `height.div_ceil(2)` in chroma-row sizing),
  /// - `y_stride < width`, `u_stride < (width + 1) / 2`, or
  ///   `v_stride < (width + 1) / 2`,
  /// - any plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  // The 3-plane × (slice, stride, dim) shape is intrinsic to YUV 4:2:0;
  // `div_ceil` on u32 isn't const-stable yet, so the `(x + 1) / 2`
  // idiom stays.
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuv420pFrameError::ZeroDimension { width, height });
    }
    // 4:2:0 subsamples chroma 2:1 in width (one chroma sample covers
    // two Y columns), so odd widths have no paired chroma for the
    // rightmost column and the SIMD kernels assume `width & 1 == 0`.
    // Height is allowed to be odd: plane sizing uses
    // `height.div_ceil(2)` and the row walker maps every Y row `r`
    // to chroma row `r / 2`, so a frame like 640x481 works — the last
    // Y row shares chroma with the final chroma row alone.
    if width & 1 != 0 {
      return Err(Yuv420pFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuv420pFrameError::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuv420pFrameError::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuv420pFrameError::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }

    // Plane sizes use `checked_mul` because `stride * height` can
    // wrap `usize` on 32‑bit targets (wasm32, i686) for large inputs
    // — without this guard, an undersized plane could pass validation
    // and panic later during row slicing. The declared geometry must
    // fit in `usize` to be usable at all.
    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv420pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv420pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv420pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv420pFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv420pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// U (Cb) plane bytes. Row `r` starts at byte offset `r * u_stride()`.
  /// U has half the width and half the height of the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }

  /// V (Cr) plane bytes. Row `r` starts at byte offset `r * v_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the U plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }

  /// Byte stride of the V plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
}

/// A validated YUV 4:2:2 planar frame.
///
/// Three planes. Same per-row kernel contract as [`Yuv420pFrame`] —
/// the 4:2:0 → 4:2:2 difference is purely vertical. [`Nv16Frame`]
/// has the same axis difference versus [`Nv12Frame`].
///
/// - `y` — full-size luma, `y_stride >= width`, length
///   `>= y_stride * height`.
/// - `u` / `v` — **half-width**, **full-height** chroma,
///   `u_stride >= (width + 1) / 2`, length `>= u_stride * height`.
///
/// `width` must be even (4:2:2 still pairs chroma columns 2:1). No
/// height parity constraint — chroma is full-height.
///
/// Canonical for `libx264 -pix_fmt yuv422p`, pro-video intermediates,
/// and ProRes SW decode at 8 bits.
#[derive(Debug, Clone, Copy)]
pub struct Yuv422pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a> Yuv422pFrame<'a> {
  /// Constructs a new [`Yuv422pFrame`], validating dimensions and
  /// plane lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv422pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuv422pFrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuv422pFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuv422pFrameError::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuv422pFrameError::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuv422pFrameError::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv422pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv422pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2: chroma is **full-height** — no `div_ceil(2)`.
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv422pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv422pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv422pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv422pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv422pFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv422pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// U (Cb) plane bytes. Half-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }

  /// V (Cr) plane bytes. Half-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the U plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }

  /// Byte stride of the V plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
}

/// Errors returned by [`Yuv422pFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuv422pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. 4:2:2 subsamples chroma 2:1 in width.
  #[error("width ({width}) is odd; 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `u_stride` is smaller than the half-width chroma row.
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// Required minimum U‑plane stride (`= width / 2`).
    chroma_width: u32,
    /// The supplied U‑plane stride.
    u_stride: u32,
  },
  /// `v_stride` is smaller than the half-width chroma row.
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// Required minimum V‑plane stride.
    chroma_width: u32,
    /// The supplied V‑plane stride.
    v_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` bytes.
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated YUV 4:4:4 planar frame.
///
/// Three planes, all full-size. Same per-row arithmetic as
/// [`Nv24Frame`] / [`Nv42Frame`] but with U and V read from separate
/// slices instead of an interleaved plane.
///
/// - `y` / `u` / `v` — full-size, `*_stride >= width`, length
///   `>= *_stride * height`.
///
/// No width parity constraint (4:4:4 chroma is 1:1 with Y).
///
/// Canonical for ProRes 4444 SW decode, CUDA/NVDEC hardware-decode
/// download of 4:4:4 content, and `libx264 -pix_fmt yuv444p`.
#[derive(Debug, Clone, Copy)]
pub struct Yuv444pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a> Yuv444pFrame<'a> {
  /// Constructs a new [`Yuv444pFrame`], validating dimensions and
  /// plane lengths. Odd widths are accepted — 4:4:4 chroma pairs
  /// nothing.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv444pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuv444pFrameError::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuv444pFrameError::YStrideTooSmall { width, y_stride });
    }
    if u_stride < width {
      return Err(Yuv444pFrameError::UStrideTooSmall { width, u_stride });
    }
    if v_stride < width {
      return Err(Yuv444pFrameError::VStrideTooSmall { width, v_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv444pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv444pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv444pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv444pFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv444pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// U (Cb) plane bytes. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }

  /// V (Cr) plane bytes. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }

  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the U plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }

  /// Byte stride of the V plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
}

/// Errors returned by [`Yuv444pFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuv444pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `u_stride < width`.
  #[error("u_stride ({u_stride}) is smaller than width ({width})")]
  UStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied U‑plane stride.
    u_stride: u32,
  },
  /// `v_stride < width`.
  #[error("v_stride ({v_stride}) is smaller than width ({width})")]
  VStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied V‑plane stride.
    v_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` bytes.
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated YUV 4:4:0 planar frame.
///
/// **4:4:0 = full-width chroma, half-height chroma.** Axis-flipped
/// counterpart to 4:2:2: chroma is fully sampled horizontally
/// (1:1 with Y) but subsampled 2:1 vertically (one chroma row per
/// two Y rows). FFmpeg names: `yuv440p`, `yuvj440p`. Mostly seen
/// from JPEG decoders that subsampled vertically only.
///
/// Three planes:
/// - `y` — full-size luma.
/// - `u` / `v` — full-width, **half-height** chroma. `u_stride >=
///   width`, length `>= u_stride * ((height + 1) / 2)`.
///
/// `width` accepts any value (4:4:0 has no horizontal subsampling
/// — same as 4:4:4). `height` may be odd: chroma row sizing uses
/// `height.div_ceil(2)` and the row walker maps Y row `r` to
/// chroma row `r / 2`, so a frame like 1280x481 works.
///
/// Per-row kernel reuses [`Yuv444pFrame`]'s `yuv_444_to_rgb_row`:
/// per-row math is identical (full-width chroma, no horizontal
/// duplication); only the walker reads chroma row `r / 2` instead
/// of `r`.
///
/// Validation errors surface as [`Yuv440pFrameError`] (a transparent
/// alias of [`Yuv444pFrameError`] — same variants apply since 4:4:0
/// uses the same chroma-width and overflow contracts as 4:4:4).
#[derive(Debug, Clone, Copy)]
pub struct Yuv440pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a> Yuv440pFrame<'a> {
  /// Constructs a new [`Yuv440pFrame`], validating dimensions and
  /// plane lengths. Errors surface as [`Yuv440pFrameError`] (a
  /// transparent alias of [`Yuv444pFrameError`] — same variants apply
  /// since 4:4:0 has full-width chroma like 4:4:4 and no width-parity
  /// constraint).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv440pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuv444pFrameError::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuv444pFrameError::YStrideTooSmall { width, y_stride });
    }
    if u_stride < width {
      return Err(Yuv444pFrameError::UStrideTooSmall { width, u_stride });
    }
    if v_stride < width {
      return Err(Yuv444pFrameError::VStrideTooSmall { width, v_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv444pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:4:0: chroma is half-height (same as 4:2:0 vertical axis).
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv444pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv444pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv444pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv440pFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv440pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }
  /// U (Cb) plane bytes. **Full-width, half-height** — one row per
  /// two Y rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }
  /// V (Cr) plane bytes. Full-width, half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }
  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels. May be odd.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y plane stride.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U plane stride (full-width).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V plane stride (full-width).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
}

/// Errors returned by [`Yuv440pFrame::try_new`]. Transparent alias of
/// [`Yuv444pFrameError`] — 4:4:0 has the same full-width chroma and
/// no width-parity constraint, so the variants apply unchanged. The
/// alias keeps the public API self-descriptive.
pub type Yuv440pFrameError = Yuv444pFrameError;

/// A validated NV12 (semi‑planar 4:2:0) frame.
///
/// Two planes:
/// - `y` — full‑size luma, `y_stride >= width`, length `>= y_stride * height`.
/// - `uv` — interleaved chroma (`U0, V0, U1, V1, …`) at half width and
///   half height, so each UV row is `2 * ceil(width / 2) = width` bytes
///   of payload; `uv_stride >= width`, length
///   `>= uv_stride * ceil(height / 2)`.
///
/// `width` must be even (same 4:2:0 rationale as [`Yuv420pFrame`]);
/// `height` may be odd — chroma row sizing uses `height.div_ceil(2)`
/// and the walker reuses chroma with `row / 2`. This matters in
/// practice: 640x481 outputs from macroblock-aligned decoders are
/// representable. Odd-width input is rejected at construction.
///
/// This is the canonical layout emitted by Apple VideoToolbox, VA‑API,
/// NVDEC, D3D11VA, and Android MediaCodec for 8‑bit decoded frames.
#[derive(Debug, Clone, Copy)]
pub struct Nv12Frame<'a> {
  y: &'a [u8],
  uv: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a> Nv12Frame<'a> {
  /// Constructs a new [`Nv12Frame`], validating dimensions and plane
  /// lengths.
  ///
  /// Returns [`Nv12FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd (4:2:0 subsamples chroma 2:1 in width; odd
  ///   height is allowed and handled via `height.div_ceil(2)`),
  /// - `y_stride < width`,
  /// - `uv_stride < width` (the UV row holds `width / 2` interleaved
  ///   pairs = `width` bytes of payload),
  /// - either plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, Nv12FrameError> {
    if width == 0 || height == 0 {
      return Err(Nv12FrameError::ZeroDimension { width, height });
    }
    // Same odd‑width rationale as [`Yuv420pFrame::try_new`]. Height
    // is allowed to be odd — chroma row sizing uses `div_ceil(2)` and
    // the walker maps Y row `r` to chroma row `r / 2`, so NV12 frames
    // like 640x481 (the decoder output for a 640x480 source cropped
    // from an encoded 480-row‑plus‑edge MB grid) are representable.
    if width & 1 != 0 {
      return Err(Nv12FrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Nv12FrameError::YStrideTooSmall { width, y_stride });
    }
    // Each chroma row carries `width / 2` interleaved UV pairs = `width`
    // bytes of payload.
    let uv_row_bytes = width;
    if uv_stride < uv_row_bytes {
      return Err(Nv12FrameError::UvStrideTooSmall {
        uv_row_bytes,
        uv_stride,
      });
    }

    // Plane sizes use `checked_mul` because `stride * rows` can wrap
    // `usize` on 32‑bit targets (wasm32, i686) — see
    // [`Yuv420pFrame::try_new`] for the same rationale.
    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv12FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Nv12FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let uv_min = match (uv_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv12FrameError::GeometryOverflow {
          stride: uv_stride,
          rows: chroma_height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(Nv12FrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`Nv12Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Nv12Frame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved UV plane. Each chroma row starts at offset
  /// `chroma_row * uv_stride()` and contains `width` bytes of payload
  /// laid out as `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`. The chroma
  /// row index for an output row `r` is `r / 2` (4:2:0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u8] {
    self.uv
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the interleaved UV plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }
}

/// Errors returned by [`Nv12Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Nv12FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. 4:2:0 subsamples chroma 2:1 in width, so each
  /// chroma column pairs two Y columns — odd widths leave the last Y
  /// column without a paired chroma sample, and the SIMD kernels
  /// assume `width & 1 == 0`. Height is allowed to be odd (handled by
  /// `height.div_ceil(2)` in chroma‑row sizing).
  #[error("width ({width}) is odd; 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `uv_stride` is smaller than the `width` bytes of interleaved UV
  /// payload one chroma row must hold.
  #[error("uv_stride ({uv_stride}) is smaller than UV row payload ({uv_row_bytes} bytes)")]
  UvStrideTooSmall {
    /// Required minimum UV‑plane stride (`= width`).
    uv_row_bytes: u32,
    /// The supplied UV‑plane stride.
    uv_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// UV plane is shorter than `uv_stride * ceil(height / 2)` bytes.
  #[error("UV plane has {actual} bytes but at least {expected} are required")]
  UvPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (can only fire on 32‑bit
  /// targets — wasm32, i686 — with extreme dimensions).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated NV16 (semi‑planar 4:2:2) frame.
///
/// Same interleaved‑UV layout as [`Nv12Frame`] but with 4:2:2 chroma
/// subsampling — chroma is half‑width, **full‑height**. Each chroma row
/// pairs with exactly one Y row (vs. 4:2:0, where two Y rows share one
/// chroma row). The row primitive itself is identical to NV12's
/// (`nv12_to_rgb_row`) — the difference is in the walker, which
/// advances chroma every row instead of every two rows.
///
/// Two planes:
/// - `y` — full‑size luma, `y_stride >= width`, length
///   `>= y_stride * height`.
/// - `uv` — interleaved chroma (`U0, V0, U1, V1, …`) at half width and
///   **full height**, so each UV row is `width` bytes of payload;
///   `uv_stride >= width`, length `>= uv_stride * height`.
///
/// `width` must be even (4:2:2 still subsamples chroma 2:1 in width).
/// `height` is unrestricted — no parity constraint. Odd‑width input is
/// rejected at construction.
///
/// Emitted by some professional capture hardware and by FFmpeg's
/// `AV_PIX_FMT_NV16` (relatively uncommon compared to NV12, but shows
/// up in pro-video pipelines).
#[derive(Debug, Clone, Copy)]
pub struct Nv16Frame<'a> {
  y: &'a [u8],
  uv: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a> Nv16Frame<'a> {
  /// Constructs a new [`Nv16Frame`], validating dimensions and plane
  /// lengths.
  ///
  /// Returns [`Nv16FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd (4:2:2 subsamples chroma 2:1 in width),
  /// - `y_stride < width`,
  /// - `uv_stride < width` (the UV row holds `width / 2` interleaved
  ///   pairs = `width` bytes of payload),
  /// - either plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, Nv16FrameError> {
    if width == 0 || height == 0 {
      return Err(Nv16FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Nv16FrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Nv16FrameError::YStrideTooSmall { width, y_stride });
    }
    // Each chroma row carries `width / 2` interleaved UV pairs = `width`
    // bytes of payload — same as NV12.
    let uv_row_bytes = width;
    if uv_stride < uv_row_bytes {
      return Err(Nv16FrameError::UvStrideTooSmall {
        uv_row_bytes,
        uv_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv16FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Nv16FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2 chroma is full‑height — no `div_ceil(2)` here (this is the
    // only structural difference from [`Nv12Frame::try_new`]).
    let uv_min = match (uv_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv16FrameError::GeometryOverflow {
          stride: uv_stride,
          rows: height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(Nv16FrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`Nv16Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Nv16Frame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved UV plane. Each chroma row starts at offset
  /// `row * uv_stride()` (4:2:2: one UV row per Y row) and contains
  /// `width` bytes of payload laid out as
  /// `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u8] {
    self.uv
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the interleaved UV plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }
}

/// Errors returned by [`Nv16Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Nv16FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. 4:2:2 subsamples chroma 2:1 in width.
  #[error("width ({width}) is odd; 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `uv_stride` is smaller than the `width` bytes of interleaved UV
  /// payload one chroma row must hold.
  #[error("uv_stride ({uv_stride}) is smaller than UV row payload ({uv_row_bytes} bytes)")]
  UvStrideTooSmall {
    /// Required minimum UV‑plane stride (`= width`).
    uv_row_bytes: u32,
    /// The supplied UV‑plane stride.
    uv_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// UV plane is shorter than `uv_stride * height` bytes.
  #[error("UV plane has {actual} bytes but at least {expected} are required")]
  UvPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (can only fire on 32‑bit
  /// targets — wasm32, i686 — with extreme dimensions).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated NV24 (semi‑planar 4:4:4) frame.
///
/// Same interleaved‑UV layout family as [`Nv12Frame`] / [`Nv16Frame`]
/// but with **4:4:4** chroma — no subsampling. Chroma is full‑width
/// and full‑height; each Y pixel has its own UV pair. Width has no
/// parity constraint (chroma is 1:1 with Y, not 2:1).
///
/// Two planes:
/// - `y` — full‑size luma, `y_stride >= width`, length
///   `>= y_stride * height`.
/// - `uv` — interleaved chroma (`U0, V0, U1, V1, …`) at **full width**
///   and full height, so each UV row is `2 * width` bytes of payload;
///   `uv_stride >= 2 * width`, length `>= uv_stride * height`.
#[derive(Debug, Clone, Copy)]
pub struct Nv24Frame<'a> {
  y: &'a [u8],
  uv: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a> Nv24Frame<'a> {
  /// Constructs a new [`Nv24Frame`], validating dimensions and plane
  /// lengths.
  ///
  /// Returns [`Nv24FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `y_stride < width`,
  /// - `uv_stride < 2 * width`,
  /// - the `2 * width` product overflows `u32`,
  /// - either plane is too short to cover its declared rows.
  ///
  /// Unlike [`Nv12Frame`] / [`Nv16Frame`], odd widths are accepted —
  /// 4:4:4 does not pair chroma columns.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, Nv24FrameError> {
    if width == 0 || height == 0 {
      return Err(Nv24FrameError::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Nv24FrameError::YStrideTooSmall { width, y_stride });
    }
    // Each chroma row carries `width` UV pairs = `2 * width` bytes of
    // payload. Use `checked_mul` — `2 * width` could overflow `u32` at
    // `width >= 2^31`.
    let uv_row_bytes = match width.checked_mul(2) {
      Some(v) => v,
      None => {
        return Err(Nv24FrameError::GeometryOverflow {
          stride: width,
          rows: 2,
        });
      }
    };
    if uv_stride < uv_row_bytes {
      return Err(Nv24FrameError::UvStrideTooSmall {
        uv_row_bytes,
        uv_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv24FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Nv24FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let uv_min = match (uv_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv24FrameError::GeometryOverflow {
          stride: uv_stride,
          rows: height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(Nv24FrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`Nv24Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u8],
    uv: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Nv24Frame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved UV plane. Each chroma row starts at offset
  /// `row * uv_stride()` and contains `2 * width` bytes of payload
  /// laid out as `U0, V0, U1, V1, …, U_{w-1}, V_{w-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u8] {
    self.uv
  }

  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the interleaved UV plane (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }
}

/// Errors returned by [`Nv24Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Nv24FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `uv_stride` is smaller than the `2 * width` bytes of interleaved
  /// UV payload one chroma row must hold.
  #[error("uv_stride ({uv_stride}) is smaller than UV row payload ({uv_row_bytes} bytes)")]
  UvStrideTooSmall {
    /// Required minimum UV‑plane stride (`= 2 * width`).
    uv_row_bytes: u32,
    /// The supplied UV‑plane stride.
    uv_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// UV plane is shorter than `uv_stride * height` bytes.
  #[error("UV plane has {actual} bytes but at least {expected} are required")]
  UvPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// Size arithmetic overflowed. Fires for either
  /// `stride * rows` exceeding `usize::MAX` (the usual case) **or**
  /// the `width * 2` computation for the UV-row-payload length
  /// exceeding `u32::MAX` at extreme widths.
  #[error("declared geometry overflows: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride (or `width`, for the `width * 2` overflow case) of
    /// the dimension whose product overflowed.
    stride: u32,
    /// Row count (or `2`, for the `width * 2` overflow case) that
    /// overflowed against the stride.
    rows: u32,
  },
}

/// A validated NV42 (semi‑planar 4:4:4, VU‑ordered) frame.
///
/// NV24's byte‑order twin: chroma layout is `V0, U0, V1, U1, …`
/// instead of NV24's `U0, V0, U1, V1, …`. All validation rules are
/// identical to [`Nv24Frame`]; only the kernel‑level interpretation of
/// even / odd bytes in the interleaved plane differs.
#[derive(Debug, Clone, Copy)]
pub struct Nv42Frame<'a> {
  y: &'a [u8],
  vu: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  vu_stride: u32,
}

impl<'a> Nv42Frame<'a> {
  /// Constructs a new [`Nv42Frame`], validating dimensions and plane
  /// lengths. Same rules as [`Nv24Frame::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u8],
    vu: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    vu_stride: u32,
  ) -> Result<Self, Nv42FrameError> {
    if width == 0 || height == 0 {
      return Err(Nv42FrameError::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Nv42FrameError::YStrideTooSmall { width, y_stride });
    }
    let vu_row_bytes = match width.checked_mul(2) {
      Some(v) => v,
      None => {
        return Err(Nv42FrameError::GeometryOverflow {
          stride: width,
          rows: 2,
        });
      }
    };
    if vu_stride < vu_row_bytes {
      return Err(Nv42FrameError::VuStrideTooSmall {
        vu_row_bytes,
        vu_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv42FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Nv42FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let vu_min = match (vu_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv42FrameError::GeometryOverflow {
          stride: vu_stride,
          rows: height,
        });
      }
    };
    if vu.len() < vu_min {
      return Err(Nv42FrameError::VuPlaneTooShort {
        expected: vu_min,
        actual: vu.len(),
      });
    }

    Ok(Self {
      y,
      vu,
      width,
      height,
      y_stride,
      vu_stride,
    })
  }

  /// Constructs a new [`Nv42Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u8],
    vu: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    vu_stride: u32,
  ) -> Self {
    match Self::try_new(y, vu, width, height, y_stride, vu_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Nv42Frame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved VU plane. Each chroma row starts at offset
  /// `row * vu_stride()` and contains `2 * width` bytes of payload
  /// laid out as `V0, U0, V1, U1, …, V_{w-1}, U_{w-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vu(&self) -> &'a [u8] {
    self.vu
  }

  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the interleaved VU plane (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vu_stride(&self) -> u32 {
    self.vu_stride
  }
}

/// Errors returned by [`Nv42Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Nv42FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `vu_stride` is smaller than the `2 * width` bytes of interleaved
  /// VU payload one chroma row must hold.
  #[error("vu_stride ({vu_stride}) is smaller than VU row payload ({vu_row_bytes} bytes)")]
  VuStrideTooSmall {
    /// Required minimum VU‑plane stride (`= 2 * width`).
    vu_row_bytes: u32,
    /// The supplied VU‑plane stride.
    vu_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// VU plane is shorter than `vu_stride * height` bytes.
  #[error("VU plane has {actual} bytes but at least {expected} are required")]
  VuPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// Size arithmetic overflowed. Fires for either
  /// `stride * rows` exceeding `usize::MAX` (the usual case) **or**
  /// the `width * 2` computation for the VU-row-payload length
  /// exceeding `u32::MAX` at extreme widths.
  #[error("declared geometry overflows: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride (or `width`, for the `width * 2` overflow case) of
    /// the dimension whose product overflowed.
    stride: u32,
    /// Row count (or `2`, for the `width * 2` overflow case) that
    /// overflowed against the stride.
    rows: u32,
  },
}

/// A validated P010 (semi‑planar 4:2:0, 10‑bit `u16`) frame.
///
/// The canonical layout emitted by Apple VideoToolbox, VA‑API, NVDEC,
/// D3D11VA, and Intel QSV for 10‑bit HDR hardware‑decoded output. Same
/// plane shape as [`Nv12Frame`] — one full‑size luma plane plus one
/// interleaved UV plane at half width and half height — but sample
/// width is **`u16`** and the 10 active bits sit in the **high** 10 of
/// each element (`sample = value << 6`, low 6 bits zero). That matches
/// Microsoft's P010 convention and FFmpeg's `AV_PIX_FMT_P010LE`.
///
/// This is **not** the [`Yuv420p10Frame`] layout — yuv420p10le puts the
/// 10 bits in the **low** 10 of each `u16`. Callers holding a P010
/// buffer must use [`P010Frame`]; callers holding yuv420p10le must use
/// [`Yuv420p10Frame`]. Kernels mask/shift appropriately for each.
///
/// Stride is in **samples** (`u16` elements), not bytes. Users holding
/// an FFmpeg byte buffer should cast via [`bytemuck::cast_slice`] and
/// divide `linesize[i]` by 2 before constructing.
///
/// Two planes:
/// - `y` — full‑size luma, `y_stride >= width`, length
///   `>= y_stride * height` (all in `u16` samples).
/// - `uv` — interleaved chroma (`U0, V0, U1, V1, …`) at half width and
///   half height, so each UV row carries `2 * ceil(width / 2) = width`
///   `u16` elements; `uv_stride >= width`, length
///   `>= uv_stride * ceil(height / 2)`.
///
/// `width` must be even (same 4:2:0 rationale as the other frame
/// types); `height` may be odd (handled via `height.div_ceil(2)` in
/// chroma‑row sizing).
///
/// # Input sample range and packing sanity
///
/// Each `u16` sample's `BITS` active bits live in the high `BITS`
/// positions; the low `16 - BITS` bits are expected to be zero.
/// [`Self::try_new`] validates geometry only.
///
/// [`Self::try_new_checked`] additionally scans every sample and
/// rejects any with non‑zero low `16 - BITS` bits — a **necessary
/// but not sufficient** packing sanity check. Its catch rate
/// weakens as `BITS` grows: at `BITS == 10` it rejects 63/64 random
/// samples and is a strong signal; at `BITS == 12` it only rejects
/// 15/16, and **common flat‑region values in decoder output are
/// exactly the ones that slip through** (`Y = 256/1024` limited
/// black, `UV = 2048` neutral chroma are all multiples of 16 in
/// both layouts). See [`Self::try_new_checked`] for the full
/// table. For strict provenance, callers must rely on their source
/// format metadata and pick the right frame type ([`PnFrame`] vs
/// [`Yuv420pFrame16`]) at construction.
///
/// Kernels shift each load right by `16 - BITS` to extract the
/// active value, so mispacked input (e.g. a `yuv420p12le` buffer
/// handed to the P012 kernel) produces deterministic, backend‑
/// independent output — wrong colors, but consistently wrong across
/// scalar + every SIMD backend, which is visible in any output diff.
#[derive(Debug, Clone, Copy)]
pub struct PnFrame<'a, const BITS: u32> {
  y: &'a [u16],
  uv: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a, const BITS: u32> PnFrame<'a, BITS> {
  /// Constructs a new [`P010Frame`], validating dimensions and plane
  /// lengths. Strides are in `u16` **samples**.
  ///
  /// Returns [`P010FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd,
  /// - `y_stride < width`,
  /// - `uv_stride < width` (the UV row holds `width / 2` interleaved
  ///   pairs = `width` `u16` elements),
  /// - either plane is too short, or
  /// - `stride * rows` overflows `usize` (32‑bit targets only).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    // Guard the `BITS` parameter at the top. 10 and 12 use the Q15
    // i32 kernel family (`p_n_to_rgb_*<BITS>`); 16 uses the parallel
    // i64 kernel family (`p16_to_rgb_*`). 14 has no high-bit-packed
    // hardware format. All three supported depths funnel through the
    // same `PnFrame` struct; kernel selection is at the public
    // dispatcher boundary.
    if BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(PnFrameError::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(PnFrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(PnFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(PnFrameError::YStrideTooSmall { width, y_stride });
    }
    let uv_row_elems = width;
    if uv_stride < uv_row_elems {
      return Err(PnFrameError::UvStrideTooSmall {
        uv_row_elems,
        uv_stride,
      });
    }
    // Interleaved UV is consecutive `(U, V)` u16 pairs. An odd
    // u16-element stride would start every other chroma row on the
    // V element of the previous pair, swapping U / V interpretation
    // deterministically and producing wrong colors on alternate rows.
    if uv_stride & 1 != 0 {
      return Err(PnFrameError::UvStrideOdd { uv_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(PnFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let uv_min = match (uv_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: uv_stride,
          rows: chroma_height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(PnFrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`P010Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid PnFrame dimensions, plane lengths, or BITS value"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample and
  /// rejects any whose **low `16 - BITS` bits** are non‑zero. A valid
  /// high‑bit‑packed sample has its `BITS` active bits in the high
  /// `BITS` positions and zero below, so non‑zero low bits is
  /// evidence the buffer isn't Pn‑shaped.
  ///
  /// **This is a packing sanity check, not a provenance validator.**
  /// The check catches noisy low‑bit‑packed data (where most samples
  /// have low‑bit content), but it **cannot** distinguish Pn from a
  /// low‑bit‑packed buffer whose samples all happen to be multiples
  /// of `1 << (16 - BITS)`. The catch rate scales with `BITS`:
  ///
  /// - `BITS == 10` (P010): 6 low bits must be zero. Random u16
  ///   samples pass with probability `1/64`; noisy `yuv420p10le`
  ///   data is almost always caught.
  /// - `BITS == 12` (P012): only 4 low bits. Pass probability is
  ///   `1/16` — 4× weaker. **Common limited‑range flat‑region values
  ///   (`Y = 256` limited black, `UV = 2048` neutral chroma,
  ///   `Y = 1024` full black) are all multiples of 16 in both
  ///   layouts**, so flat `yuv420p12le` content passes **every
  ///   time**. The `>> 4` extraction in the Pn kernels then
  ///   discards the real signal and produces badly darkened
  ///   output. For P012, prefer format metadata over this check.
  ///
  /// Callers who need strict provenance must rely on their source
  /// format metadata and pick the right frame type at construction
  /// ([`PnFrame`] vs [`Yuv420pFrame16`]); no runtime check on opaque
  /// `u16` data can reliably tell the two layouts apart, and the
  /// weakness is proportionally worse the higher the `BITS` value.
  /// The regression test
  /// `p012_try_new_checked_accepts_low_packed_flat_content_by_design`
  /// in `frame::tests` pins this limitation in code.
  ///
  /// Cost: one O(plane_size) scan per plane. The default
  /// [`Self::try_new`] skips this so the hot path stays O(1).
  ///
  /// Returns [`PnFrameError::SampleLowBitsSet`] on the first
  /// offending sample — carries the plane, element index, offending
  /// value, and the number of low bits expected to be zero.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    let frame = Self::try_new(y, uv, width, height, y_stride, uv_stride)?;
    let low_bits = 16 - BITS;
    let low_mask: u16 = ((1u32 << low_bits) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let uv_w = w; // interleaved: `width / 2` pairs × 2 elements
    let chroma_h = height.div_ceil(2) as usize;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Y,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * uv_stride as usize;
      for (col, &s) in uv[start..start + uv_w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Uv,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y (luma) plane samples. Row `r` starts at sample offset
  /// `r * y_stride()`. Each sample's 10 active bits sit in the **high**
  /// 10 positions of the `u16` (low 6 bits zero).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }

  /// Interleaved UV plane samples. Each chroma row starts at sample
  /// offset `chroma_row * uv_stride()` and contains `width` `u16`
  /// elements laid out as `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  /// Each element's 10 active bits sit in the high 10 positions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u16] {
    self.uv
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Sample stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Sample stride of the interleaved UV plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }

  /// Active bit depth — 10, 12, or 16. Mirrors the `BITS` const parameter
  /// so generic code can read it without naming the type.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// Type alias for a validated P010 frame (10‑bit, high‑bit‑packed).
/// Use this name at call sites for readability.
pub type P010Frame<'a> = PnFrame<'a, 10>;

/// Type alias for a validated P012 frame (12‑bit, high‑bit‑packed).
/// Same layout as [`P010Frame`] but with 12 active bits in the high
/// 12 of each `u16` (`sample = value << 4`, low 4 bits zero).
pub type P012Frame<'a> = PnFrame<'a, 12>;

/// Type alias for a validated P016 frame (16‑bit, no high-vs-low
/// distinction — the full `u16` range is active). Tight wrapper over
/// [`PnFrame`] with `BITS == 16`.
///
/// **Uses a parallel i64 kernel family** — scalar + SIMD kernels
/// named `p16_to_rgb_*` instead of the `p_n_to_rgb_*<BITS>` family
/// that covers 10/12. The chroma multiply-add (`c_u * u_d + c_v *
/// v_d`) overflows i32 at 16 bits for standard matrices (e.g.,
/// BT.709 `b_u = 60808` × `u_d ≈ 32768` alone is within 1 bit of
/// i32 max; summing both chroma terms exceeds it). The 16-bit path
/// runs those multiplies as i64 and shifts i64 right by 15 before
/// narrowing back. The 10/12 paths stay on the i32 pipeline
/// unchanged.
pub type P016Frame<'a> = PnFrame<'a, 16>;

/// A validated **4:2:2** semi-planar high-bit-packed frame, generic
/// over `const BITS: u32 ∈ {10, 12, 16}`.
///
/// The 4:2:2 twin of [`PnFrame`]: same Y + interleaved-UV plane shape,
/// but chroma is **full-height** (one chroma row per Y row, not one
/// per two). UV remains horizontally subsampled — each chroma row
/// holds `width / 2` interleaved `U, V` pairs = `width` `u16` elements.
/// Hardware decoders / transcode pipelines emit this layout for
/// chroma-rich pro-video sources (NVDEC / CUDA HDR 4:2:2 download
/// targets and some QSV configurations).
///
/// FFmpeg variants: `P210LE` (10-bit), `P212LE` (12-bit, FFmpeg 5.0+),
/// `P216LE` (16-bit). Each `u16` packs its `BITS` active bits in the
/// **high** `BITS` positions, matching the [`PnFrame`] convention; at
/// `BITS == 16` every bit is active.
///
/// Stride is in **`u16` samples**, not bytes (callers holding an
/// FFmpeg byte buffer must cast and divide `linesize[i]` by 2).
///
/// Two planes:
/// - `y` — full-size luma, `y_stride >= width`, length
///   `>= y_stride * height`.
/// - `uv` — interleaved chroma at **half-width × full-height**, so
///   each chroma row holds `width` `u16` elements (= `width / 2`
///   pairs); `uv_stride >= width`, length `>= uv_stride * height`.
///
/// `width` must be even (4:2:2 subsamples chroma horizontally).
/// `height` has no parity constraint.
///
/// # Input sample range and packing sanity
///
/// Same conventions and caveats as [`PnFrame`] —
/// [`Self::try_new_checked`] scans every sample and rejects any with
/// non-zero low `16 - BITS` bits. The catch rate is identical to
/// [`PnFrame`] at the same `BITS`. See [`PnFrame::try_new_checked`]
/// for the full discussion of why this is a packing sanity check, not
/// a provenance validator.
#[derive(Debug, Clone, Copy)]
pub struct PnFrame422<'a, const BITS: u32> {
  y: &'a [u16],
  uv: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a, const BITS: u32> PnFrame422<'a, BITS> {
  /// Constructs a new [`PnFrame422`], validating dimensions, plane
  /// lengths, and the `BITS` parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    if BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(PnFrameError::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(PnFrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(PnFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(PnFrameError::YStrideTooSmall { width, y_stride });
    }
    let uv_row_elems = width;
    if uv_stride < uv_row_elems {
      return Err(PnFrameError::UvStrideTooSmall {
        uv_row_elems,
        uv_stride,
      });
    }
    // Interleaved UV is consecutive `(U, V)` u16 pairs — see
    // [`PnFrame::try_new`] for the full rationale.
    if uv_stride & 1 != 0 {
      return Err(PnFrameError::UvStrideOdd { uv_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(PnFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2: chroma is full-height (height rows, not div_ceil(height/2)).
    let uv_min = match (uv_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: uv_stride,
          rows: height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(PnFrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`PnFrame422`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid PnFrame422 dimensions, plane lengths, or BITS value"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample and
  /// rejects any whose low `16 - BITS` bits are non-zero. See
  /// [`PnFrame::try_new_checked`] for the full discussion of catch
  /// rates and limitations at each `BITS`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    let frame = Self::try_new(y, uv, width, height, y_stride, uv_stride)?;
    if BITS == 16 {
      return Ok(frame);
    }
    let low_bits = 16 - BITS;
    let low_mask: u16 = ((1u32 << low_bits) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let uv_w = w; // half-width × 2 elements per pair = width u16 elements per row
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Y,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    // 4:2:2: scan every chroma row (full-height).
    for row in 0..h {
      let start = row * uv_stride as usize;
      for (col, &s) in uv[start..start + uv_w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Uv,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y (luma) plane samples (`u16` elements).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Interleaved UV plane samples — each row holds `width` `u16`
  /// elements laid out as `U0, V0, U1, V1, …, U_{w/2-1}, V_{w/2-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u16] {
    self.uv
  }
  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Sample stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Sample stride of the interleaved UV plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:2:2 semi-planar, 10-bit (`AV_PIX_FMT_P210LE`). Alias over
/// [`PnFrame422`]`<10>`.
pub type P210Frame<'a> = PnFrame422<'a, 10>;
/// 4:2:2 semi-planar, 12-bit (`AV_PIX_FMT_P212LE`, FFmpeg 5.0+).
/// Alias over [`PnFrame422`]`<12>`.
pub type P212Frame<'a> = PnFrame422<'a, 12>;
/// 4:2:2 semi-planar, 16-bit (`AV_PIX_FMT_P216LE`). Alias over
/// [`PnFrame422`]`<16>`. At 16 bits the high-vs-low packing
/// distinction degenerates — every bit is active.
pub type P216Frame<'a> = PnFrame422<'a, 16>;

/// A validated **4:4:4** semi-planar high-bit-packed frame, generic
/// over `const BITS: u32 ∈ {10, 12, 16}`.
///
/// The 4:4:4 twin of [`PnFrame`] / [`PnFrame422`]: same Y + interleaved
/// UV layout, but chroma is **full-width × full-height** (1:1 with Y,
/// no subsampling). Each chroma row holds `2 * width` `u16` elements
/// (= `width` interleaved `U, V` pairs). NVDEC / CUDA HDR 4:4:4
/// download target.
///
/// FFmpeg variants: `P410LE` (10-bit), `P412LE` (12-bit, FFmpeg 5.0+),
/// `P416LE` (16-bit). Active-bit packing identical to [`PnFrame`].
///
/// Stride is in **`u16` samples**, not bytes.
///
/// Two planes:
/// - `y` — full-size luma, `y_stride >= width`, length
///   `>= y_stride * height`.
/// - `uv` — interleaved chroma at **full-width × full-height**, so
///   each chroma row holds `2 * width` `u16` elements (= `width`
///   pairs); `uv_stride >= 2 * width`, length `>= uv_stride * height`.
///
/// No width-parity constraint (4:4:4 chroma is 1:1 with Y, not paired
/// horizontally).
#[derive(Debug, Clone, Copy)]
pub struct PnFrame444<'a, const BITS: u32> {
  y: &'a [u16],
  uv: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a, const BITS: u32> PnFrame444<'a, BITS> {
  /// Constructs a new [`PnFrame444`], validating dimensions, plane
  /// lengths, and the `BITS` parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    if BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(PnFrameError::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(PnFrameError::ZeroDimension { width, height });
    }
    // 4:4:4: no width-parity constraint.
    if y_stride < width {
      return Err(PnFrameError::YStrideTooSmall { width, y_stride });
    }
    // UV row holds 2 * width u16 elements (one pair per pixel).
    let uv_row_elems = match width.checked_mul(2) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: width,
          rows: 2,
        });
      }
    };
    if uv_stride < uv_row_elems {
      return Err(PnFrameError::UvStrideTooSmall {
        uv_row_elems,
        uv_stride,
      });
    }
    // Interleaved UV is consecutive `(U, V)` u16 pairs — see
    // [`PnFrame::try_new`] for the full rationale.
    if uv_stride & 1 != 0 {
      return Err(PnFrameError::UvStrideOdd { uv_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(PnFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let uv_min = match (uv_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(PnFrameError::GeometryOverflow {
          stride: uv_stride,
          rows: height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(PnFrameError::UvPlaneTooShort {
        expected: uv_min,
        actual: uv.len(),
      });
    }

    Ok(Self {
      y,
      uv,
      width,
      height,
      y_stride,
      uv_stride,
    })
  }

  /// Constructs a new [`PnFrame444`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Self {
    match Self::try_new(y, uv, width, height, y_stride, uv_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid PnFrame444 dimensions, plane lengths, or BITS value"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample and
  /// rejects any whose low `16 - BITS` bits are non-zero. See
  /// [`PnFrame::try_new_checked`] for the full discussion of catch
  /// rates and limitations.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, PnFrameError> {
    let frame = Self::try_new(y, uv, width, height, y_stride, uv_stride)?;
    if BITS == 16 {
      return Ok(frame);
    }
    let low_bits = 16 - BITS;
    let low_mask: u16 = ((1u32 << low_bits) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let uv_w = 2 * w; // full-width × 2 elements per pair
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Y,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * uv_stride as usize;
      for (col, &s) in uv[start..start + uv_w].iter().enumerate() {
        if s & low_mask != 0 {
          return Err(PnFrameError::SampleLowBitsSet {
            plane: PnFramePlane::Uv,
            index: start + col,
            value: s,
            low_bits,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y (luma) plane samples (`u16` elements).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// Interleaved UV plane samples — each row holds `2 * width` `u16`
  /// elements laid out as `U0, V0, U1, V1, …, U_{w-1}, V_{w-1}`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv(&self) -> &'a [u16] {
    self.uv
  }
  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Sample stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Sample stride of the interleaved UV plane (`>= 2 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn uv_stride(&self) -> u32 {
    self.uv_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:4:4 semi-planar, 10-bit (`AV_PIX_FMT_P410LE`). Alias over
/// [`PnFrame444`]`<10>`.
pub type P410Frame<'a> = PnFrame444<'a, 10>;
/// 4:4:4 semi-planar, 12-bit (`AV_PIX_FMT_P412LE`, FFmpeg 5.0+).
/// Alias over [`PnFrame444`]`<12>`.
pub type P412Frame<'a> = PnFrame444<'a, 12>;
/// 4:4:4 semi-planar, 16-bit (`AV_PIX_FMT_P416LE`). Alias over
/// [`PnFrame444`]`<16>`.
pub type P416Frame<'a> = PnFrame444<'a, 16>;

/// Identifies which plane of a [`PnFrame`] a
/// [`PnFrameError::SampleLowBitsSet`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum PnFramePlane {
  /// Luma plane.
  Y,
  /// Interleaved UV plane.
  Uv,
}

/// Back‑compat alias for the pre‑generalization plane enum name.
pub type P010FramePlane = PnFramePlane;

/// Errors returned by [`PnFrame::try_new`] and
/// [`PnFrame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum PnFrameError {
  /// `BITS` was not one of the supported high‑bit‑packed depths
  /// (10, 12, 16). 14 exists in the planar `yuv420p14le` family but
  /// not as a Pn hardware output.
  #[error("unsupported BITS ({bits}) for PnFrame; must be 10, 12, or 16")]
  UnsupportedBits {
    /// The unsupported value of the `BITS` const parameter.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. Returned by [`PnFrame::try_new`] (4:2:0) and
  /// [`PnFrame422::try_new`] (4:2:2) — both subsample chroma 2:1
  /// horizontally and pair `(U, V)` per chroma sample, so the frame
  /// width must be even. 4:4:4 ([`PnFrame444`]) has no parity
  /// constraint and never emits this variant.
  #[error("width ({width}) is odd; horizontally-subsampled chroma requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width` (in `u16` samples).
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride (samples).
    y_stride: u32,
  },
  /// `uv_stride` is smaller than the interleaved UV row payload
  /// one chroma row must hold (in `u16` elements). The required
  /// payload depends on the format: `width` for 4:2:0 / 4:2:2
  /// (half-width × 2 elements per pair) and `2 * width` for 4:4:4
  /// (full-width × 2 elements per pair).
  #[error("uv_stride ({uv_stride}) is smaller than UV row payload ({uv_row_elems} u16 elements)")]
  UvStrideTooSmall {
    /// Required minimum UV‑plane stride, in `u16` elements.
    uv_row_elems: u32,
    /// The supplied UV‑plane stride (samples).
    uv_stride: u32,
  },
  /// `uv_stride` is odd. Each interleaved chroma row is laid out as
  /// `(U, V)` pairs of `u16` elements; an odd stride starts every
  /// other row on the opposite element of the pair, swapping the U /
  /// V interpretation deterministically and producing wrong colors on
  /// alternate rows. Returned by all three `PnFrame*::try_new`
  /// constructors (`PnFrame` 4:2:0, `PnFrame422` 4:2:2,
  /// `PnFrame444` 4:4:4).
  #[error(
    "uv_stride ({uv_stride}) is odd; semi-planar interleaved UV requires an even u16-element stride"
  )]
  UvStrideOdd {
    /// The supplied UV‑plane stride (samples).
    uv_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` samples.
  #[error("Y plane has {actual} samples but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// UV plane is shorter than `uv_stride * ceil(height / 2)` samples.
  #[error("UV plane has {actual} samples but at least {expected} are required")]
  UvPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// Size arithmetic overflowed. Fires for either
  /// `stride * rows` exceeding `usize::MAX` (the usual case, only
  /// reachable on 32‑bit targets like wasm32 / i686 with extreme
  /// dimensions) **or** the `width * 2` `u32` computation for the
  /// 4:4:4 UV-row-payload length (`PnFrame444::try_new` only)
  /// exceeding `u32::MAX` at extreme widths.
  #[error("declared geometry overflows: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride (or `width`, for the `width * 2` overflow case) of
    /// the dimension whose product overflowed.
    stride: u32,
    /// Row count (or `2`, for the `width * 2` overflow case) that
    /// overflowed against the stride.
    rows: u32,
  },
  /// A sample's low `16 - BITS` bits were non‑zero — a Pn sample
  /// packs its `BITS` active bits in the high `BITS` of each `u16`,
  /// so valid samples are always multiples of `1 << (16 - BITS)`
  /// (64 for 10‑bit, 16 for 12‑bit). Only
  /// [`PnFrame::try_new_checked`] can produce this error.
  ///
  /// Note: the absence of this error does **not** prove the buffer
  /// is Pn. A low‑bit‑packed buffer of samples that all happen to be
  /// multiples of `1 << (16 - BITS)` passes the check silently. See
  /// [`PnFrame::try_new_checked`] for the full discussion.
  #[error(
    "sample {value:#06x} on plane {plane} at element {index} has non-zero low {low_bits} bits (not a valid Pn sample at the declared BITS)"
  )]
  SampleLowBitsSet {
    /// Which plane the offending sample lives on.
    plane: PnFramePlane,
    /// Element index within that plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// Number of low bits expected to be zero (`16 - BITS`).
    low_bits: u32,
  },
}

/// Back‑compat alias for the pre‑generalization error enum name.
pub type P010FrameError = PnFrameError;

/// A validated NV21 (semi‑planar 4:2:0) frame.
///
/// Structurally identical to [`Nv12Frame`] — one full-size luma plane
/// plus one interleaved chroma plane at half width and half height —
/// but the chroma bytes are **VU-ordered** instead of UV-ordered:
/// each row is `V0, U0, V1, U1, …, V_{w/2-1}, U_{w/2-1}`. This is
/// Android MediaCodec's default output for 8-bit decoded frames and
/// shows up in iOS camera capture under specific configurations.
///
/// Dimension / stride validation is identical to [`Nv12Frame`]:
/// `width` must be even, `height` may be odd (chroma row sizing uses
/// `height.div_ceil(2)`).
#[derive(Debug, Clone, Copy)]
pub struct Nv21Frame<'a> {
  y: &'a [u8],
  vu: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  vu_stride: u32,
}

impl<'a> Nv21Frame<'a> {
  /// Constructs a new [`Nv21Frame`], validating dimensions and plane
  /// lengths.
  ///
  /// Returns [`Nv21FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd (4:2:0 subsamples chroma 2:1 in width; odd
  ///   height is allowed and handled via `height.div_ceil(2)`),
  /// - `y_stride < width`,
  /// - `vu_stride < width` (the VU row holds `width / 2` interleaved
  ///   pairs = `width` bytes of payload),
  /// - either plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    y: &'a [u8],
    vu: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    vu_stride: u32,
  ) -> Result<Self, Nv21FrameError> {
    if width == 0 || height == 0 {
      return Err(Nv21FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Nv21FrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Nv21FrameError::YStrideTooSmall { width, y_stride });
    }
    let vu_row_bytes = width;
    if vu_stride < vu_row_bytes {
      return Err(Nv21FrameError::VuStrideTooSmall {
        vu_row_bytes,
        vu_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv21FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Nv21FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let vu_min = match (vu_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Nv21FrameError::GeometryOverflow {
          stride: vu_stride,
          rows: chroma_height,
        });
      }
    };
    if vu.len() < vu_min {
      return Err(Nv21FrameError::VuPlaneTooShort {
        expected: vu_min,
        actual: vu.len(),
      });
    }

    Ok(Self {
      y,
      vu,
      width,
      height,
      y_stride,
      vu_stride,
    })
  }

  /// Constructs a new [`Nv21Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(
    y: &'a [u8],
    vu: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    vu_stride: u32,
  ) -> Self {
    match Self::try_new(y, vu, width, height, y_stride, vu_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Nv21Frame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes. Row `r` starts at byte offset `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }

  /// Interleaved VU plane. Each chroma row starts at offset
  /// `chroma_row * vu_stride()` and contains `width` bytes of payload
  /// laid out as `V0, U0, V1, U1, …, V_{w/2-1}, U_{w/2-1}` — the
  /// chroma bytes are **VU-ordered**, the opposite of NV12. The
  /// chroma row index for an output row `r` is `r / 2` (4:2:0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vu(&self) -> &'a [u8] {
    self.vu
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Byte stride of the interleaved VU plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vu_stride(&self) -> u32 {
    self.vu_stride
  }
}

/// Errors returned by [`Nv21Frame::try_new`]. Variant shape is
/// identical to [`Nv12FrameError`] — only the "UV" → "VU" naming
/// changes to match the plane's byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Nv21FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. Same rationale as [`Nv12FrameError::OddWidth`].
  #[error("width ({width}) is odd; 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride.
    y_stride: u32,
  },
  /// `vu_stride` is smaller than the `width` bytes of interleaved VU
  /// payload one chroma row must hold.
  #[error("vu_stride ({vu_stride}) is smaller than VU row payload ({vu_row_bytes} bytes)")]
  VuStrideTooSmall {
    /// Required minimum VU‑plane stride (`= width`).
    vu_row_bytes: u32,
    /// The supplied VU‑plane stride.
    vu_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// VU plane is shorter than `vu_stride * ceil(height / 2)` bytes.
  #[error("VU plane has {actual} bytes but at least {expected} are required")]
  VuPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated YUV 4:2:0 planar frame at bit depths > 8 (10/12/14).
///
/// Structurally identical to [`Yuv420pFrame`] — three planes, half‑
/// size chroma — but sample storage is **`u16`** so every pixel
/// carries up to 16 bits of payload. `BITS` is the active bit depth
/// (10, 12, 14, or 16). Callers are **expected** to store each sample in
/// the **low** `BITS` bits of its `u16` (upper `16 - BITS` bits zero),
/// matching FFmpeg's little‑endian `yuv420p10le` / `yuv420p12le` /
/// `yuv420p14le` convention, where each plane is a byte buffer
/// reinterpretable as `u16` little‑endian. `try_new` validates plane
/// geometry / strides / lengths but does **not** inspect sample
/// values to verify this packing.
///
/// This is **not** the FFmpeg `p010` layout — `p010` stores samples
/// in the **high** 10 bits of each `u16` (`sample << 6`). Callers
/// holding a p010 buffer must shift right by `16 - BITS` before
/// construction.
///
/// # Input sample range
///
/// The kernels assume every input sample is in `[0, (1 << BITS) - 1]`
/// — i.e., upper `16 - BITS` bits zero. Validating this at
/// construction would require scanning every sample of every plane
/// (megabytes per frame at video rates); instead the constructor
/// validates geometry only and the contract falls on the caller.
/// Decoders and FFmpeg output satisfy this by construction.
///
/// **Output for out‑of‑range samples is equivalent to pre‑masking
/// every sample to the low `BITS` bits.** Every kernel (scalar + all
/// 5 SIMD tiers) AND‑masks each `u16` load to `(1 << BITS) - 1`
/// before the Q15 path, so a sample like `0xFFC0` (p010 white =
/// `1023 << 6`) is treated identically to `0x03C0` on every backend
/// when `BITS == 10`. This gives deterministic, backend‑independent
/// output for mispacked input — feeding `p010` data into a
/// `yuv420p10le`‑shaped frame produces severely distorted, but stable,
/// pixel values across scalar / NEON / SSE4.1 / AVX2 / AVX‑512 /
/// wasm simd128, which is an obvious signal for downstream diffing.
/// The mask is a single AND per load and a no‑op on valid input
/// (upper bits already zero).
///
/// Callers who want the mispacking to surface as a loud error
/// instead of silent color corruption should use
/// [`Self::try_new_checked`] — it scans every sample and returns
/// [`Yuv420pFrame16Error::SampleOutOfRange`] on the first violation.
///
/// All four supported depths — `BITS == 10` (HDR10 / 10‑bit SDR
/// keystone), `BITS == 12` (HEVC Main 12 / VP9 Profile 3),
/// `BITS == 14` (grading / mastering pipelines), and `BITS == 16`
/// (reference / intermediate HDR) — share this frame struct but
/// **use two kernel families**:
///
/// - 10 / 12 / 14 run on a single const-generic Q15 i32 pipeline
///   (`scalar::yuv_420p_n_to_rgb_*<BITS>` + matching SIMD kernels
///   across NEON / SSE4.1 / AVX2 / AVX-512 / wasm simd128).
/// - 16 runs on a parallel i64 kernel family
///   (`scalar::yuv_420p16_to_rgb_*` + matching SIMD) because the
///   Q15 chroma multiply-add overflows i32 at 16 bits.
///
/// The constructor validates `BITS ∈ {10, 12, 14, 16}` up front;
/// kernel selection is at the public dispatcher boundary
/// (`yuv420pNN_to_rgb_*`). The selection is free — each dispatcher
/// is a dedicated function that knows which family to call.
///
/// Stride is in **samples** (`u16` elements), not bytes. Users
/// holding a byte buffer from FFmpeg should cast via
/// [`bytemuck::cast_slice`] and divide `linesize[i]` by 2 before
/// constructing.
///
/// `width` must be even (same 4:2:0 rationale as [`Yuv420pFrame`]);
/// `height` may be odd and is handled via `height.div_ceil(2)` in
/// chroma‑row sizing.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a, const BITS: u32> Yuv420pFrame16<'a, BITS> {
  /// Constructs a new [`Yuv420pFrame16`], validating dimensions, plane
  /// lengths, and the `BITS` parameter.
  ///
  /// Returns [`Yuv420pFrame16Error`] if any of:
  /// - `BITS` is not 10, 12, 14, or 16 — use [`Yuv420p10Frame`],
  ///   [`Yuv420p12Frame`], [`Yuv420p14Frame`], or [`Yuv420p16Frame`]
  ///   at call sites for readability, all four are type aliases
  ///   over this struct,
  /// - `width` or `height` is zero,
  /// - `width` is odd,
  /// - any stride is smaller than the plane's declared pixel width,
  /// - any plane is too short to cover its declared rows, or
  /// - `stride * rows` overflows `usize` (32‑bit targets only).
  ///
  /// All strides are in **samples** (`u16` elements).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    // Guard the `BITS` parameter at the top. 10/12/14 share the Q15
    // i32 kernel family; 16 uses a parallel i64 kernel family (see
    // [`Yuv420p16Frame`] and `yuv_420p16_to_rgb_*`). 8 has its own
    // (non-generic) 8-bit kernels in [`Yuv420pFrame`].
    if BITS != 9 && BITS != 10 && BITS != 12 && BITS != 14 && BITS != 16 {
      return Err(Yuv420pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuv420pFrame16Error::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuv420pFrame16Error::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuv420pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuv420pFrame16Error::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuv420pFrame16Error::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }

    // Plane sizes are in `u16` elements, so the overflow guard runs
    // against the sample count — callers converting from byte strides
    // should have already divided by 2.
    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv420pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv420pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv420pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv420pFrame16`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv420pFrame16 dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input (e.g., a `u16` buffer of unknown provenance
  /// that might be `p010`‑packed or otherwise dirty) where accepting
  /// out-of-range samples would be unacceptable because they violate
  /// the expected bit-depth contract and can produce invalid results.
  ///
  /// Cost: one O(plane_size) linear scan per plane — a few megabytes
  /// per 1080p frame at 10 bits. The default [`Self::try_new`] skips
  /// this so the hot path (decoder output, already-conforming
  /// buffers) stays O(1).
  ///
  /// Returns [`Yuv420pFrame16Error::SampleOutOfRange`] on the first
  /// offending sample — the error carries the plane, element index
  /// within that plane's slice, offending value, and the valid
  /// maximum so the caller can pinpoint the bad sample. All of
  /// [`Self::try_new`]'s geometry errors are still possible.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    let frame = Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride)?;
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    // Scan the declared-payload region of each plane. Stride may add
    // unused padding past the declared width; we don't inspect that —
    // callers often pass buffers whose padding bytes are arbitrary,
    // and the kernels never read them.
    let w = width as usize;
    let h = height as usize;
    let chroma_w = w / 2;
    let chroma_h = height.div_ceil(2) as usize;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y (luma) plane samples. Row `r` starts at sample offset
  /// `r * y_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }

  /// U (Cb) plane samples. Row `r` starts at sample offset
  /// `r * u_stride()`. U has half the width and half the height of the
  /// frame (chroma row index for output row `r` is `r / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }

  /// V (Cr) plane samples. Row `r` starts at sample offset
  /// `r * v_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }

  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Sample stride of the Y plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }

  /// Sample stride of the U plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }

  /// Sample stride of the V plane (`>= width / 2`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }

  /// Active bit depth — 10, 12, 14, or 16. Mirrors the `BITS` const
  /// parameter so generic code can read it without naming the type.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// Type alias for a validated YUV 4:2:0 planar frame at 10 bits per
/// sample (`AV_PIX_FMT_YUV420P10LE`). Tight wrapper over
/// [`Yuv420pFrame16`] with `BITS == 10` — use this name at call sites
/// for readability.
pub type Yuv420p10Frame<'a> = Yuv420pFrame16<'a, 10>;

/// Type alias for a validated YUV 4:2:0 planar frame at 9 bits per
/// sample (`AV_PIX_FMT_YUV420P9LE`). Tight wrapper over
/// [`Yuv420pFrame16`] with `BITS == 9` — same low‑bit‑packed `u16`
/// layout as [`Yuv420p10Frame`] / [`Yuv420p12Frame`], just with 9
/// active bits in the low 9 of each element (upper 7 bits zero).
/// Niche format — AVC High 9 profile only; HEVC/VP9/AV1 don't
/// produce 9-bit. Reuses the same Q15 i32 kernel family as 10/12/14.
pub type Yuv420p9Frame<'a> = Yuv420pFrame16<'a, 9>;

/// Type alias for a validated YUV 4:2:0 planar frame at 12 bits per
/// sample (`AV_PIX_FMT_YUV420P12LE`). Tight wrapper over
/// [`Yuv420pFrame16`] with `BITS == 12` — same low‑bit‑packed `u16`
/// layout as [`Yuv420p10Frame`], just with 12 active bits in the
/// low 12 of each element (upper 4 bits zero).
pub type Yuv420p12Frame<'a> = Yuv420pFrame16<'a, 12>;

/// Type alias for a validated YUV 4:2:0 planar frame at 14 bits per
/// sample (`AV_PIX_FMT_YUV420P14LE`). Tight wrapper over
/// [`Yuv420pFrame16`] with `BITS == 14` — same low‑bit‑packed `u16`
/// layout as [`Yuv420p10Frame`], just with 14 active bits in the
/// low 14 of each element (upper 2 bits zero).
pub type Yuv420p14Frame<'a> = Yuv420pFrame16<'a, 14>;

/// Type alias for a validated YUV 4:2:0 planar frame at 16 bits per
/// sample (`AV_PIX_FMT_YUV420P16LE`). Tight wrapper over
/// [`Yuv420pFrame16`] with `BITS == 16` — the full `u16` range is
/// active (no upper-bit zero guarantee). **Uses a parallel i64
/// kernel family** because the Q15 chroma sum overflows i32 at
/// 16 bits; scalar + SIMD kernels named `yuv_420p16_to_rgb_*`
/// instead of the `yuv_420p_n_to_rgb_*<BITS>` family that covers
/// 10/12/14.
pub type Yuv420p16Frame<'a> = Yuv420pFrame16<'a, 16>;

/// Errors returned by [`Yuv420pFrame16::try_new`]. Variant shape
/// mirrors [`Yuv420pFrameError`], with `UnsupportedBits` added for
/// the new `BITS` parameter and all sizes expressed in **samples**
/// (`u16` elements) instead of bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuv420pFrame16Error {
  /// `BITS` was not one of the supported depths (10, 12, 14, 16).
  /// 8‑bit frames should use [`Yuv420pFrame`]; 16‑bit is supported,
  /// but uses a different kernel family (see [`Yuv420pFrame16`] docs).
  #[error("unsupported BITS ({bits}) for Yuv420pFrame16; must be 10, 12, 14, or 16")]
  UnsupportedBits {
    /// The unsupported value of the `BITS` const parameter.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. Same 4:2:0 rationale as
  /// [`Yuv420pFrameError::OddWidth`].
  #[error("width ({width}) is odd; YUV420p / 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width` (in samples).
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride (samples).
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)` (in samples).
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// Required minimum chroma‑plane stride.
    chroma_width: u32,
    /// The supplied U‑plane stride (samples).
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)` (in samples).
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// Required minimum chroma‑plane stride.
    chroma_width: u32,
    /// The supplied V‑plane stride (samples).
    v_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` samples.
  #[error("Y plane has {actual} samples but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * ceil(height / 2)` samples.
  #[error("U plane has {actual} samples but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * ceil(height / 2)` samples.
  #[error("V plane has {actual} samples but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A plane sample exceeds `(1 << BITS) - 1` — i.e., a bit above the
  /// declared active depth is set. Only [`Yuv420pFrame16::try_new_checked`]
  /// can produce this error; [`Yuv420pFrame16::try_new`] validates
  /// geometry only and treats the low‑bit‑packing contract as an
  /// expectation. Use the checked constructor for untrusted input
  /// (e.g., a buffer that might be `p010`‑packed instead of
  /// `yuv420p10le`‑packed).
  #[error(
    "sample {value} on plane {plane} at element {index} exceeds {max_valid} ((1 << BITS) - 1)"
  )]
  SampleOutOfRange {
    /// Which plane the offending sample lives on.
    plane: Yuv420pFrame16Plane,
    /// Element index within that plane's slice. This is the raw
    /// `&[u16]` index — it accounts for stride padding rows, so
    /// `index / stride` is the row, `index % stride` is the
    /// in‑row position.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// The maximum allowed value for this `BITS` (`(1 << BITS) - 1`).
    max_valid: u16,
  },
}

/// Identifies which plane of a [`Yuv420pFrame16`] an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Yuv420pFrame16Plane {
  /// Luma plane.
  Y,
  /// U (Cb) chroma plane.
  U,
  /// V (Cr) chroma plane.
  V,
}

/// A validated planar 4:2:2 `u16`-backed frame, generic over
/// `const BITS: u32 ∈ {10, 12, 14, 16}`. Samples are low-bit-packed
/// (the `BITS` active bits sit in the **low** bits of each `u16`).
///
/// Layout mirrors [`Yuv420pFrame16`] but with chroma half-width,
/// **full-height**: `u.len() >= u_stride * height`. The per-row
/// kernel contract is identical to the 4:2:0 family — the 4:2:2
/// difference lives in the walker (chroma row matches Y row instead
/// of `Y / 2`).
///
/// All strides are in **samples** (`u16` elements). Use the
/// [`Yuv422p10Frame`] / [`Yuv422p12Frame`] / [`Yuv422p14Frame`] /
/// [`Yuv422p16Frame`] aliases at call sites.
#[derive(Debug, Clone, Copy)]
pub struct Yuv422pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a, const BITS: u32> Yuv422pFrame16<'a, BITS> {
  /// Constructs a new [`Yuv422pFrame16`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    if BITS != 9 && BITS != 10 && BITS != 12 && BITS != 14 && BITS != 16 {
      return Err(Yuv420pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuv420pFrame16Error::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuv420pFrame16Error::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuv420pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuv420pFrame16Error::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuv420pFrame16Error::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv420pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2: chroma is **full-height** (no `div_ceil(2)`).
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv420pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv420pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv422pFrame16`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv422pFrame16 dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input where accepting out-of-range samples would
  /// silently corrupt the conversion via the kernels' bit-mask.
  ///
  /// Returns [`Yuv420pFrame16Error::SampleOutOfRange`] on the first
  /// offending sample. All of [`Self::try_new`]'s geometry errors are
  /// still possible. At `BITS == 16` the check is a no-op (every
  /// `u16` value is valid) — same convention as
  /// [`Yuv420pFrame16::try_new_checked`].
  ///
  /// Cost: one O(plane_size) linear scan per plane. The default
  /// [`Self::try_new`] skips this so the hot path (decoder output,
  /// already-conforming buffers) stays O(1).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    let frame = Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride)?;
    if BITS == 16 {
      return Ok(frame);
    }
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    // 4:2:2: chroma is half-width, FULL-height.
    let chroma_w = w / 2;
    let chroma_h = h;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y plane (`u16` elements).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U plane. Half-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V plane. Half-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:2:2 planar, 9-bit (`AV_PIX_FMT_YUV422P9LE`). Alias over
/// [`Yuv422pFrame16`]`<9>`. Niche format; reuses the same Q15 i32
/// kernel family as the 10/12/14 siblings.
pub type Yuv422p9Frame<'a> = Yuv422pFrame16<'a, 9>;
/// 4:2:2 planar, 10-bit. Alias over [`Yuv422pFrame16`]`<10>`.
pub type Yuv422p10Frame<'a> = Yuv422pFrame16<'a, 10>;
/// 4:2:2 planar, 12-bit. Alias over [`Yuv422pFrame16`]`<12>`.
pub type Yuv422p12Frame<'a> = Yuv422pFrame16<'a, 12>;
/// 4:2:2 planar, 14-bit. Alias over [`Yuv422pFrame16`]`<14>`.
pub type Yuv422p14Frame<'a> = Yuv422pFrame16<'a, 14>;
/// 4:2:2 planar, 16-bit. Alias over [`Yuv422pFrame16`]`<16>`. Uses
/// the parallel i64 kernel family (see `yuv_422p16_to_rgb_*`).
pub type Yuv422p16Frame<'a> = Yuv422pFrame16<'a, 16>;

/// A validated planar 4:4:4 `u16`-backed frame, generic over
/// `const BITS: u32 ∈ {10, 12, 14, 16}`. All three planes are
/// full-size. No width parity constraint.
#[derive(Debug, Clone, Copy)]
pub struct Yuv444pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a, const BITS: u32> Yuv444pFrame16<'a, BITS> {
  /// Constructs a new [`Yuv444pFrame16`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    if BITS != 9 && BITS != 10 && BITS != 12 && BITS != 14 && BITS != 16 {
      return Err(Yuv420pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuv420pFrame16Error::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuv420pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    // 4:4:4: chroma stride ≥ width (not width / 2).
    if u_stride < width {
      return Err(Yuv420pFrame16Error::UStrideTooSmall {
        chroma_width: width,
        u_stride,
      });
    }
    if v_stride < width {
      return Err(Yuv420pFrame16Error::VStrideTooSmall {
        chroma_width: width,
        v_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv420pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv420pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv420pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv444pFrame16`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv444pFrame16 dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input where accepting out-of-range samples would
  /// silently corrupt the conversion via the kernels' bit-mask.
  ///
  /// Returns [`Yuv420pFrame16Error::SampleOutOfRange`] on the first
  /// offending sample. All of [`Self::try_new`]'s geometry errors are
  /// still possible. At `BITS == 16` the check is a no-op (every
  /// `u16` value is valid) — same convention as
  /// [`Yuv420pFrame16::try_new_checked`].
  ///
  /// Cost: one O(plane_size) linear scan per plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv420pFrame16Error> {
    let frame = Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride)?;
    if BITS == 16 {
      return Ok(frame);
    }
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    // 4:4:4: chroma is full-width, full-height (1:1 with Y).
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U plane. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V plane. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:4:4 planar, 9-bit (`AV_PIX_FMT_YUV444P9LE`). Alias over
/// [`Yuv444pFrame16`]`<9>`. Niche; reuses the same Q15 i32 kernel
/// family as the 10/12/14 siblings.
pub type Yuv444p9Frame<'a> = Yuv444pFrame16<'a, 9>;
/// 4:4:4 planar, 10-bit. Alias over [`Yuv444pFrame16`]`<10>`.
pub type Yuv444p10Frame<'a> = Yuv444pFrame16<'a, 10>;
/// 4:4:4 planar, 12-bit. Alias over [`Yuv444pFrame16`]`<12>`.
pub type Yuv444p12Frame<'a> = Yuv444pFrame16<'a, 12>;
/// 4:4:4 planar, 14-bit. Alias over [`Yuv444pFrame16`]`<14>`.
pub type Yuv444p14Frame<'a> = Yuv444pFrame16<'a, 14>;
/// 4:4:4 planar, 16-bit. Alias over [`Yuv444pFrame16`]`<16>`. Uses
/// the parallel i64 kernel family (see `yuv_444p16_to_rgb_*`).
pub type Yuv444p16Frame<'a> = Yuv444pFrame16<'a, 16>;

/// Errors returned by [`Yuva444pFrame16::try_new`] and
/// [`Yuva444pFrame16::try_new_checked`].
///
/// Variant shape mirrors [`Yuv420pFrame16Error`] (geometry,
/// `UnsupportedBits`, `SampleOutOfRange`, plane-too-short),
/// extended with the `A`-plane variants ([`Self::AStrideTooSmall`],
/// [`Self::APlaneTooShort`]) for the 4:4:4 alpha plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva444pFrame16Error {
  /// `BITS` was not one of the supported depths. Yuva444p shipped
  /// progressively — 8b‑1 (10), 8b‑3 (9), 8b‑4 (12 / 14), 8b‑5a (16,
  /// scalar; SIMD lands in 8b‑5b/c).
  #[error("unsupported BITS ({bits}) for Yuva444pFrame16; must be 9, 10, 12, 14, or 16")]
  UnsupportedBits {
    /// The unsupported value of the `BITS` const parameter.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `y_stride < width` (in samples).
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y‑plane stride (samples).
    y_stride: u32,
  },
  /// `u_stride < width` (in samples). 4:4:4 chroma is full-width.
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// Required minimum chroma‑plane stride.
    chroma_width: u32,
    /// The supplied U‑plane stride (samples).
    u_stride: u32,
  },
  /// `v_stride < width` (in samples). 4:4:4 chroma is full-width.
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// Required minimum chroma‑plane stride.
    chroma_width: u32,
    /// The supplied V‑plane stride (samples).
    v_stride: u32,
  },
  /// `a_stride < width` (in samples). The alpha plane is full-width
  /// at the source's bit depth (1:1 with Y, like the chroma planes).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A‑plane stride (samples).
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` samples.
  #[error("Y plane has {actual} samples but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` samples.
  #[error("U plane has {actual} samples but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` samples.
  #[error("V plane has {actual} samples but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` samples.
  #[error("A plane has {actual} samples but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A plane sample exceeds `(1 << BITS) - 1`. Only
  /// [`Yuva444pFrame16::try_new_checked`] can produce this error.
  #[error(
    "sample {value} on plane {plane} at element {index} exceeds {max_valid} ((1 << BITS) - 1)"
  )]
  SampleOutOfRange {
    /// Which plane the offending sample lives on.
    plane: Yuva444pFrame16Plane,
    /// Element index within that plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// The maximum allowed value for this `BITS` (`(1 << BITS) - 1`).
    max_valid: u16,
  },
}

/// Identifies which plane of a [`Yuva444pFrame16`] an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Yuva444pFrame16Plane {
  /// Luma plane.
  Y,
  /// U (Cb) chroma plane.
  U,
  /// V (Cr) chroma plane.
  V,
  /// Alpha plane.
  A,
}

/// A validated planar 4:4:4 `u16`-backed frame **with an alpha plane**,
/// generic over `const BITS: u32`. Tranche 1 ships `BITS == 10` only
/// (`AV_PIX_FMT_YUVA444P10LE`); later tranches will admit additional
/// depths as the corresponding YUVA pixel formats land.
///
/// Four planes — Y, U, V, A — all full-width × full-height (the alpha
/// plane is at the source's bit depth, low-bit-packed in `u16`,
/// matching the Y/U/V planes).
#[derive(Debug, Clone, Copy)]
pub struct Yuva444pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  a: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a, const BITS: u32> Yuva444pFrame16<'a, BITS> {
  /// Constructs a new [`Yuva444pFrame16`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva444pFrame16Error> {
    // Ship 8b‑1 shipped 10-bit; 8b‑3 added 9; 8b‑4 added 12/14;
    // 8b‑5a opens 16. The 16-bit path uses the dedicated i64 4:4:4
    // kernel family (separate from the BITS-generic Q15 i32
    // template that covers {9,10,12,14}).
    if BITS != 9 && BITS != 10 && BITS != 12 && BITS != 14 && BITS != 16 {
      return Err(Yuva444pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuva444pFrame16Error::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuva444pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    // 4:4:4: chroma stride ≥ width.
    if u_stride < width {
      return Err(Yuva444pFrame16Error::UStrideTooSmall {
        chroma_width: width,
        u_stride,
      });
    }
    if v_stride < width {
      return Err(Yuva444pFrame16Error::VStrideTooSmall {
        chroma_width: width,
        v_stride,
      });
    }
    // Alpha is full-width (1:1 with Y).
    if a_stride < width {
      return Err(Yuva444pFrame16Error::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva444pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva444pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva444pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrame16Error::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva444pFrame16Error::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva444pFrame16`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva444pFrame16 dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input where accepting out-of-range samples would
  /// silently corrupt the conversion via the kernels' bit-mask.
  ///
  /// Returns [`Yuva444pFrame16Error::SampleOutOfRange`] on the first
  /// offending sample. All of [`Self::try_new`]'s geometry errors are
  /// still possible.
  ///
  /// Cost: one O(plane_size) linear scan per plane (Y, U, V, A — four
  /// planes total).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva444pFrame16Error> {
    let frame = Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    )?;
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva444pFrame16Error::SampleOutOfRange {
            plane: Yuva444pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva444pFrame16Error::SampleOutOfRange {
            plane: Yuva444pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva444pFrame16Error::SampleOutOfRange {
            plane: Yuva444pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * a_stride as usize;
      for (col, &s) in a[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva444pFrame16Error::SampleOutOfRange {
            plane: Yuva444pFrame16Plane::A,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U plane. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V plane. Full-width, full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// A plane. Full-width, full-height. Native bit depth, low-bit-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u16] {
    self.a
  }
  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// A‑plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:4:4 planar with alpha, 9-bit (`AV_PIX_FMT_YUVA444P9LE`). Alias
/// over [`Yuva444pFrame16`]`<9>`. Niche format (AVC High 9 + α only).
pub type Yuva444p9Frame<'a> = Yuva444pFrame16<'a, 9>;
/// 4:4:4 planar with alpha, 10-bit (`AV_PIX_FMT_YUVA444P10LE`). Alias
/// over [`Yuva444pFrame16`]`<10>`. The highest-value VFX format —
/// maps to ProRes 4444+α and similar mastering pipelines.
pub type Yuva444p10Frame<'a> = Yuva444pFrame16<'a, 10>;
/// 4:4:4 planar with alpha, 12-bit (`AV_PIX_FMT_YUVA444P12LE`). Alias
/// over [`Yuva444pFrame16`]`<12>`. Reuses the BITS-generic 4:4:4
/// kernel templates that already cover `BITS ∈ {9, 10, 12, 14}`.
pub type Yuva444p12Frame<'a> = Yuva444pFrame16<'a, 12>;
/// 4:4:4 planar with alpha, 14-bit. Alias over [`Yuva444pFrame16`]`<14>`.
/// FFmpeg does not ship this depth, but the colconv 4:4:4 BITS-generic
/// kernel templates already cover it for symmetry with [`Yuv444p14Frame`].
pub type Yuva444p14Frame<'a> = Yuva444pFrame16<'a, 14>;
/// 4:4:4 planar with alpha, 16-bit (`AV_PIX_FMT_YUVA444P16LE`). Alias
/// over [`Yuva444pFrame16`]`<16>`. Uses the dedicated i64 4:4:4
/// 16-bit kernel family (mirrors [`Yuva420p16Frame`]).
pub type Yuva444p16Frame<'a> = Yuva444pFrame16<'a, 16>;

/// Errors returned by [`Yuva444pFrame::try_new`].
///
/// Variant shape mirrors [`Yuva420pFrameError`] (geometry, plane-too-short,
/// `AStrideTooSmall` / `APlaneTooShort` for the alpha plane) but
/// without `OddWidth` because 4:4:4 has no chroma subsampling, so any
/// width is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva444pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride.
    y_stride: u32,
  },
  /// `u_stride < width`. 4:4:4 chroma is full-width.
  #[error("u_stride ({u_stride}) is smaller than width ({width})")]
  UStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied U-plane stride.
    u_stride: u32,
  },
  /// `v_stride < width`. 4:4:4 chroma is full-width.
  #[error("v_stride ({v_stride}) is smaller than width ({width})")]
  VStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied V-plane stride.
    v_stride: u32,
  },
  /// `a_stride < width`. The alpha plane is full-width × full-height
  /// (1:1 with Y).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride.
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` bytes.
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` bytes.
  #[error("A plane has {actual} bytes but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated YUVA 4:4:4 planar frame at 8 bits per sample
/// (`AV_PIX_FMT_YUVA444P`).
///
/// Four planes — all full-width × full-height (4:4:4 has no chroma
/// subsampling): Y, U, V, and A. Mirrors [`Yuv444pFrame`] plus the
/// alpha plane.
#[derive(Debug, Clone, Copy)]
pub struct Yuva444pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  a: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a> Yuva444pFrame<'a> {
  /// Constructs a new [`Yuva444pFrame`], validating dimensions and
  /// plane lengths.
  ///
  /// Returns [`Yuva444pFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - any stride is smaller than `width`,
  /// - any plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva444pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuva444pFrameError::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuva444pFrameError::YStrideTooSmall { width, y_stride });
    }
    if u_stride < width {
      return Err(Yuva444pFrameError::UStrideTooSmall { width, u_stride });
    }
    if v_stride < width {
      return Err(Yuva444pFrameError::VStrideTooSmall { width, v_stride });
    }
    if a_stride < width {
      return Err(Yuva444pFrameError::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva444pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva444pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva444pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva444pFrameError::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva444pFrameError::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva444pFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva444pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }
  /// U (Cb) plane bytes — full-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }
  /// V (Cr) plane bytes — full-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }
  /// A (alpha) plane bytes — full-width × full-height (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u8] {
    self.a
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
  /// Byte stride of the Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Byte stride of the U plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// Byte stride of the V plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// Byte stride of the A plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
}

/// Errors returned by [`Yuva420pFrame::try_new`].
///
/// Variant shape mirrors [`Yuv420pFrameError`] (geometry, plane-too-short)
/// extended with [`Self::AStrideTooSmall`] / [`Self::APlaneTooShort`]
/// for the 4:2:0 alpha plane (full-width × full-height — alpha is at
/// luma resolution, only chroma is subsampled).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva420pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. YUVA420p / 4:2:0 subsamples chroma 2:1 in width.
  #[error("width ({width}) is odd; YUVA420p / 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride.
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)`.
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied U-plane stride.
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)`.
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied V-plane stride.
    v_stride: u32,
  },
  /// `a_stride < width`. The alpha plane is full-width × full-height
  /// (1:1 with Y, like Yuv444p planes — only chroma is subsampled in
  /// 4:2:0).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride.
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height.div_ceil(2)` bytes.
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height.div_ceil(2)` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` bytes.
  #[error("A plane has {actual} bytes but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated YUVA 4:2:0 planar frame at 8 bits per sample
/// (`AV_PIX_FMT_YUVA420P`).
///
/// Four planes:
/// - `y` — full-size luma (same as [`Yuv420pFrame::y`]).
/// - `u` / `v` — half-width × half-height chroma (same as the parent
///   YUV 4:2:0 layout).
/// - `a` — **full-width × full-height** alpha (1:1 with Y; only chroma
///   is subsampled in 4:2:0).
///
/// `width` must be even (4:2:0 subsamples chroma 2:1 in width).
/// `height` may be odd (chroma row sizing uses `height.div_ceil(2)`,
/// alpha sizing uses `height` since alpha is full-resolution).
#[derive(Debug, Clone, Copy)]
pub struct Yuva420pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  a: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a> Yuva420pFrame<'a> {
  /// Constructs a new [`Yuva420pFrame`], validating dimensions and
  /// plane lengths.
  ///
  /// Returns [`Yuva420pFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd,
  /// - `y_stride < width`, `u_stride < (width + 1) / 2`,
  ///   `v_stride < (width + 1) / 2`, or `a_stride < width`,
  /// - any plane is too short to cover its declared rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva420pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuva420pFrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuva420pFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuva420pFrameError::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuva420pFrameError::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuva420pFrameError::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }
    // Alpha is full-width (1:1 with Y).
    if a_stride < width {
      return Err(Yuva420pFrameError::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva420pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva420pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva420pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrameError::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva420pFrameError::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva420pFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva420pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }
  /// U (Cb) plane bytes — half-width × half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }
  /// V (Cr) plane bytes — half-width × half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }
  /// A (alpha) plane bytes — full-width × full-height (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u8] {
    self.a
  }
  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride of the Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Byte stride of the U plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// Byte stride of the V plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// Byte stride of the A plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
}

/// Errors returned by [`Yuva420pFrame16::try_new`] and
/// [`Yuva420pFrame16::try_new_checked`].
///
/// Variant shape mirrors [`Yuv420pFrame16Error`] extended with the
/// `A`-plane variants ([`Self::AStrideTooSmall`] /
/// [`Self::APlaneTooShort`]) for the 4:2:0 alpha plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva420pFrame16Error {
  /// `BITS` was not one of the supported depths (9, 10, 16). FFmpeg
  /// only ships `yuva420p9le`, `yuva420p10le`, `yuva420p16le` — no
  /// 12/14-bit YUVA 4:2:0 pixel formats exist.
  #[error("unsupported BITS ({bits}) for Yuva420pFrame16; must be 9, 10, or 16")]
  UnsupportedBits {
    /// The unsupported value of the `BITS` const parameter.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd.
  #[error("width ({width}) is odd; YUVA420p / 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width` (in samples).
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride (samples).
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)` (in samples).
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// Required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied U-plane stride (samples).
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)` (in samples).
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// Required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied V-plane stride (samples).
    v_stride: u32,
  },
  /// `a_stride < width` (in samples).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride (samples).
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` samples.
  #[error("Y plane has {actual} samples but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * ceil(height / 2)` samples.
  #[error("U plane has {actual} samples but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * ceil(height / 2)` samples.
  #[error("V plane has {actual} samples but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` samples.
  #[error("A plane has {actual} samples but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A plane sample exceeds `(1 << BITS) - 1`. Only
  /// [`Yuva420pFrame16::try_new_checked`] can produce this error.
  #[error(
    "sample {value} on plane {plane} at element {index} exceeds {max_valid} ((1 << BITS) - 1)"
  )]
  SampleOutOfRange {
    /// Which plane the offending sample lives on.
    plane: Yuva420pFrame16Plane,
    /// Element index within that plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// The maximum allowed value for this `BITS` (`(1 << BITS) - 1`).
    max_valid: u16,
  },
}

/// Identifies which plane of a [`Yuva420pFrame16`] an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Yuva420pFrame16Plane {
  /// Luma plane.
  Y,
  /// U (Cb) chroma plane.
  U,
  /// V (Cr) chroma plane.
  V,
  /// Alpha plane.
  A,
}

/// A validated planar 4:2:0 `u16`-backed frame **with an alpha plane**,
/// generic over `const BITS: u32 ∈ {9, 10, 16}`. FFmpeg ships
/// `yuva420p9le`, `yuva420p10le`, and `yuva420p16le` — no 12/14-bit
/// YUVA 4:2:0 pixel formats exist, so [`Self::try_new`] rejects them.
///
/// Four planes — Y full-width × full-height, U/V half-width ×
/// half-height (4:2:0 chroma subsampling), A full-width × full-height
/// (alpha is at luma resolution; only chroma is subsampled).
#[derive(Debug, Clone, Copy)]
pub struct Yuva420pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  a: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a, const BITS: u32> Yuva420pFrame16<'a, BITS> {
  /// Constructs a new [`Yuva420pFrame16`], validating dimensions,
  /// plane lengths, and the `BITS` parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva420pFrame16Error> {
    // FFmpeg's only YUVA 4:2:0 high-bit pixel formats: yuva420p9le,
    // yuva420p10le, yuva420p16le. No 12/14-bit variants exist.
    if BITS != 9 && BITS != 10 && BITS != 16 {
      return Err(Yuva420pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuva420pFrame16Error::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuva420pFrame16Error::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuva420pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuva420pFrame16Error::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuva420pFrame16Error::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }
    if a_stride < width {
      return Err(Yuva420pFrame16Error::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva420pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva420pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva420pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva420pFrame16Error::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva420pFrame16Error::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva420pFrame16`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva420pFrame16 dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input.
  ///
  /// Cost: one O(plane_size) linear scan per plane (Y, U, V, A —
  /// four planes total).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva420pFrame16Error> {
    let frame = Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    )?;
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let chroma_w = w / 2;
    let chroma_h = height.div_ceil(2) as usize;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva420pFrame16Error::SampleOutOfRange {
            plane: Yuva420pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva420pFrame16Error::SampleOutOfRange {
            plane: Yuva420pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva420pFrame16Error::SampleOutOfRange {
            plane: Yuva420pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * a_stride as usize;
      for (col, &s) in a[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva420pFrame16Error::SampleOutOfRange {
            plane: Yuva420pFrame16Plane::A,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U plane samples — half-width × half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V plane samples — half-width × half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// A plane samples — full-width × full-height, native bit depth,
  /// low-bit-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u16] {
    self.a
  }
  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y-plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U-plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V-plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// A-plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:2:0 planar with alpha, 9-bit (`AV_PIX_FMT_YUVA420P9LE`). Alias
/// over [`Yuva420pFrame16`]`<9>`.
pub type Yuva420p9Frame<'a> = Yuva420pFrame16<'a, 9>;

/// 4:2:0 planar with alpha, 10-bit (`AV_PIX_FMT_YUVA420P10LE`). Alias
/// over [`Yuva420pFrame16`]`<10>`.
pub type Yuva420p10Frame<'a> = Yuva420pFrame16<'a, 10>;

/// 4:2:0 planar with alpha, 16-bit (`AV_PIX_FMT_YUVA420P16LE`). Alias
/// over [`Yuva420pFrame16`]`<16>`. Uses the parallel i64 kernel
/// family for the u16 RGBA path.
pub type Yuva420p16Frame<'a> = Yuva420pFrame16<'a, 16>;

/// Errors returned by [`Yuva422pFrame::try_new`].
///
/// Variant shape mirrors [`Yuva420pFrameError`]; the only semantic
/// difference is that 4:2:2 chroma is full-height, so plane-size
/// validation uses `u_stride * height` / `v_stride * height` rather
/// than `_stride * height.div_ceil(2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva422pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. YUVA422p / 4:2:2 subsamples chroma 2:1 in width.
  #[error("width ({width}) is odd; YUVA422p / 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride.
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)`.
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied U-plane stride.
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)`.
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied V-plane stride.
    v_stride: u32,
  },
  /// `a_stride < width`. The alpha plane is full-width × full-height
  /// (1:1 with Y, like Yuv422p planes — only chroma is subsampled in
  /// 4:2:2 horizontally, alpha is at luma resolution).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride.
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` bytes (chroma is
  /// full-height in 4:2:2).
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` bytes.
  #[error("A plane has {actual} bytes but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated planar 4:2:2 `u8`-backed frame **with an alpha plane**
/// (`AV_PIX_FMT_YUVA422P`).
///
/// Storage mirrors [`Yuv422pFrame`] (Y full-size, U / V half-width ×
/// full-height — 4:2:2 only subsamples chroma horizontally) plus a
/// fourth full-resolution alpha plane (1:1 with Y).
///
/// `width` must be even (4:2:2 subsamples chroma 2:1 in width).
/// `height` may be any positive value.
#[derive(Debug, Clone, Copy)]
pub struct Yuva422pFrame<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  a: &'a [u8],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a> Yuva422pFrame<'a> {
  /// Constructs a new [`Yuva422pFrame`], validating dimensions and
  /// plane lengths.
  ///
  /// Returns [`Yuva422pFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `width` is odd,
  /// - `y_stride < width`, `u_stride < (width + 1) / 2`,
  ///   `v_stride < (width + 1) / 2`, or `a_stride < width`,
  /// - any plane is too short to cover its declared rows
  ///   (chroma uses `_stride * height` because 4:2:2 chroma is
  ///   full-height; alpha uses `a_stride * height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva422pFrameError> {
    if width == 0 || height == 0 {
      return Err(Yuva422pFrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuva422pFrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuva422pFrameError::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuva422pFrameError::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuva422pFrameError::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }
    if a_stride < width {
      return Err(Yuva422pFrameError::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva422pFrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2: chroma is full-height (only subsamples horizontally).
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrameError::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva422pFrameError::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrameError::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva422pFrameError::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrameError::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva422pFrameError::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva422pFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva422pFrame dimensions or plane lengths"),
    }
  }

  /// Y (luma) plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u8] {
    self.y
  }
  /// U (Cb) plane bytes — half-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u8] {
    self.u
  }
  /// V (Cr) plane bytes — half-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u8] {
    self.v
  }
  /// A (alpha) plane bytes — full-width × full-height (1:1 with Y).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u8] {
    self.a
  }
  /// Frame width in pixels. Always even.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Byte stride of the Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Byte stride of the U plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// Byte stride of the V plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// Byte stride of the A plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
}

/// Errors returned by [`Yuva422pFrame16::try_new`] and
/// [`Yuva422pFrame16::try_new_checked`]. Variant shape mirrors
/// [`Yuva420pFrame16Error`] — only the semantic difference is in
/// chroma row count (4:2:2 chroma is full-height; the
/// `UPlaneTooShort` / `VPlaneTooShort` docs document
/// `_stride * height` rather than `_stride * height.div_ceil(2)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuva422pFrame16Error {
  /// `BITS` was not one of the supported depths (9, 10, 12, 16).
  /// FFmpeg ships `yuva422p9le`, `yuva422p10le`, `yuva422p12le`,
  /// `yuva422p16le`; Ship 8b‑4 wired 12-bit through the existing
  /// BITS-generic 4:2:2 row kernel templates.
  #[error("unsupported BITS ({bits}) for Yuva422pFrame16; must be 9, 10, 12, or 16")]
  UnsupportedBits {
    /// The unsupported value of the `BITS` const parameter.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd.
  #[error("width ({width}) is odd; YUVA422p / 4:2:2 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width` (in samples).
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride (samples).
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)` (in samples).
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// Required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied U-plane stride (samples).
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)` (in samples).
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// Required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied V-plane stride (samples).
    v_stride: u32,
  },
  /// `a_stride < width` (in samples).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride (samples).
    a_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` samples.
  #[error("Y plane has {actual} samples but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * height` samples (chroma is
  /// full-height in 4:2:2).
  #[error("U plane has {actual} samples but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * height` samples.
  #[error("V plane has {actual} samples but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` samples.
  #[error("A plane has {actual} samples but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// `stride * rows` overflows `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A plane sample exceeds `(1 << BITS) - 1`. Only
  /// [`Yuva422pFrame16::try_new_checked`] can produce this error.
  #[error(
    "sample {value} on plane {plane} at element {index} exceeds {max_valid} ((1 << BITS) - 1)"
  )]
  SampleOutOfRange {
    /// Which plane the offending sample lives on.
    plane: Yuva422pFrame16Plane,
    /// Element index within that plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// The maximum allowed value for this `BITS` (`(1 << BITS) - 1`).
    max_valid: u16,
  },
}

/// Identifies which plane of a [`Yuva422pFrame16`] an error refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Yuva422pFrame16Plane {
  /// Luma plane.
  Y,
  /// U (Cb) chroma plane.
  U,
  /// V (Cr) chroma plane.
  V,
  /// Alpha plane.
  A,
}

/// A validated planar 4:2:2 `u16`-backed frame **with an alpha plane**,
/// generic over `const BITS: u32 ∈ {9, 10, 12, 16}`. Matches the full
/// FFmpeg set — `yuva422p9le`, `yuva422p10le`, `yuva422p12le`,
/// `yuva422p16le`.
///
/// Four planes — Y full-width × full-height, U / V half-width ×
/// full-height (4:2:2 chroma subsamples horizontally only), A
/// full-width × full-height (alpha is at luma resolution).
#[derive(Debug, Clone, Copy)]
pub struct Yuva422pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  a: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
  a_stride: u32,
}

impl<'a, const BITS: u32> Yuva422pFrame16<'a, BITS> {
  /// Constructs a new [`Yuva422pFrame16`], validating dimensions,
  /// plane lengths, and the `BITS` parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva422pFrame16Error> {
    if BITS != 9 && BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(Yuva422pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuva422pFrame16Error::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(Yuva422pFrame16Error::OddWidth { width });
    }
    if y_stride < width {
      return Err(Yuva422pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    let chroma_width = width.div_ceil(2);
    if u_stride < chroma_width {
      return Err(Yuva422pFrame16Error::UStrideTooSmall {
        chroma_width,
        u_stride,
      });
    }
    if v_stride < chroma_width {
      return Err(Yuva422pFrame16Error::VStrideTooSmall {
        chroma_width,
        v_stride,
      });
    }
    if a_stride < width {
      return Err(Yuva422pFrame16Error::AStrideTooSmall { width, a_stride });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuva422pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:2:2: chroma full-height.
    let u_min = match (u_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuva422pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuva422pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuva422pFrame16Error::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(Yuva422pFrame16Error::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      a,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
      a_stride,
    })
  }

  /// Constructs a new [`Yuva422pFrame16`], panicking on invalid
  /// inputs. Prefer [`Self::try_new`] when inputs may be invalid at
  /// runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuva422pFrame16 dimensions, plane lengths, or BITS"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample of
  /// every plane and rejects values above `(1 << BITS) - 1`. Use this
  /// on untrusted input where accepting out-of-range samples would
  /// silently corrupt the conversion via the kernels' bit-mask.
  ///
  /// Returns [`Yuva422pFrame16Error::SampleOutOfRange`] on the first
  /// offending sample. All of [`Self::try_new`]'s geometry errors are
  /// still possible.
  ///
  /// 4:2:2 geometry: Y and A are full-width × full-height; U and V
  /// are half-width × full-height (chroma subsamples horizontally
  /// only).
  ///
  /// Cost: one O(plane_size) linear scan per plane (Y, U, V, A —
  /// four planes total). The default [`Self::try_new`] skips this so
  /// the hot path (decoder output, already-conforming buffers) stays
  /// O(1).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
    a_stride: u32,
  ) -> Result<Self, Yuva422pFrame16Error> {
    let frame = Self::try_new(
      y, u, v, a, width, height, y_stride, u_stride, v_stride, a_stride,
    )?;
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let chroma_w = w / 2;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva422pFrame16Error::SampleOutOfRange {
            plane: Yuva422pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva422pFrame16Error::SampleOutOfRange {
            plane: Yuva422pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + chroma_w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva422pFrame16Error::SampleOutOfRange {
            plane: Yuva422pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..h {
      let start = row * a_stride as usize;
      for (col, &s) in a[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuva422pFrame16Error::SampleOutOfRange {
            plane: Yuva422pFrame16Plane::A,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y (luma) plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U (Cb) plane samples — half-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V (Cr) plane samples — half-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// A (alpha) plane samples — full-width × full-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u16] {
    self.a
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
  /// Sample stride of the Y plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// Sample stride of the U plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// Sample stride of the V plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// Sample stride of the A plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:2:2 planar with alpha, 9-bit (`AV_PIX_FMT_YUVA422P9LE`). Alias
/// over [`Yuva422pFrame16`]`<9>`.
pub type Yuva422p9Frame<'a> = Yuva422pFrame16<'a, 9>;

/// 4:2:2 planar with alpha, 10-bit (`AV_PIX_FMT_YUVA422P10LE`). Alias
/// over [`Yuva422pFrame16`]`<10>`.
pub type Yuva422p10Frame<'a> = Yuva422pFrame16<'a, 10>;

/// 4:2:2 planar with alpha, 12-bit (`AV_PIX_FMT_YUVA422P12LE`). Alias
/// over [`Yuva422pFrame16`]`<12>`. Reuses the BITS-generic 4:2:2
/// kernel templates that already cover `BITS ∈ {9, 10, 12, 14}`.
pub type Yuva422p12Frame<'a> = Yuva422pFrame16<'a, 12>;

/// 4:2:2 planar with alpha, 16-bit (`AV_PIX_FMT_YUVA422P16LE`). Alias
/// over [`Yuva422pFrame16`]`<16>`. Uses the parallel i64 kernel
/// family for the u16 RGBA path (mirrors [`Yuva420p16Frame`]).
pub type Yuva422p16Frame<'a> = Yuva422pFrame16<'a, 16>;

/// Errors returned by [`Yuv440pFrame16::try_new`] and
/// [`Yuv440pFrame16::try_new_checked`]. Transparent alias of
/// [`Yuv420pFrame16Error`] — same `UnsupportedBits` /
/// `SampleOutOfRange` / geometry variants apply. The alias keeps the
/// public 4:4:0 surface self-descriptive without duplicating an
/// otherwise-identical enum.
pub type Yuv440pFrame16Error = Yuv420pFrame16Error;

/// A validated planar 4:4:0 `u16`-backed frame, generic over
/// `const BITS: u32 ∈ {10, 12}`. Samples are low-bit-packed (the
/// `BITS` active bits sit in the **low** bits of each `u16`).
///
/// Layout: Y full-size, U/V **full-width × half-height** — same
/// vertical subsampling as 4:2:0, no horizontal subsampling like
/// 4:4:4. Per-row kernel reuses the 4:4:4 family
/// (`yuv_444p_n_to_rgb_*<BITS>`) verbatim — only the walker reads
/// chroma row `r / 2` instead of `r`.
///
/// FFmpeg variants: `yuv440p10le`, `yuv440p12le`. No 9/14/16-bit
/// variants exist in FFmpeg, so [`Self::try_new`] rejects them.
#[derive(Debug, Clone, Copy)]
pub struct Yuv440pFrame16<'a, const BITS: u32> {
  y: &'a [u16],
  u: &'a [u16],
  v: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  u_stride: u32,
  v_stride: u32,
}

impl<'a, const BITS: u32> Yuv440pFrame16<'a, BITS> {
  /// Constructs a new [`Yuv440pFrame16`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv440pFrame16Error> {
    if BITS != 10 && BITS != 12 {
      return Err(Yuv420pFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Yuv420pFrame16Error::ZeroDimension { width, height });
    }
    if y_stride < width {
      return Err(Yuv420pFrame16Error::YStrideTooSmall { width, y_stride });
    }
    // 4:4:0 chroma is full-width — chroma_width == width.
    if u_stride < width {
      return Err(Yuv420pFrame16Error::UStrideTooSmall {
        chroma_width: width,
        u_stride,
      });
    }
    if v_stride < width {
      return Err(Yuv420pFrame16Error::VStrideTooSmall {
        chroma_width: width,
        v_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(Yuv420pFrame16Error::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    // 4:4:0: chroma is half-height (same axis as 4:2:0 vertical).
    let chroma_height = height.div_ceil(2);
    let u_min = match (u_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: u_stride,
          rows: chroma_height,
        });
      }
    };
    if u.len() < u_min {
      return Err(Yuv420pFrame16Error::UPlaneTooShort {
        expected: u_min,
        actual: u.len(),
      });
    }
    let v_min = match (v_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(Yuv420pFrame16Error::GeometryOverflow {
          stride: v_stride,
          rows: chroma_height,
        });
      }
    };
    if v.len() < v_min {
      return Err(Yuv420pFrame16Error::VPlaneTooShort {
        expected: v_min,
        actual: v.len(),
      });
    }

    Ok(Self {
      y,
      u,
      v,
      width,
      height,
      y_stride,
      u_stride,
      v_stride,
    })
  }

  /// Constructs a new [`Yuv440pFrame16`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Self {
    match Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Yuv440pFrame16 dimensions or plane lengths"),
    }
  }

  /// Constructs a new [`Yuv440pFrame16`] and additionally rejects any
  /// sample whose value exceeds `(1 << BITS) - 1`. Mirrors
  /// [`Yuv420pFrame16::try_new_checked`] /
  /// [`Yuv444pFrame16::try_new_checked`]; downstream row kernels mask
  /// the high bits at load time, so out-of-range samples otherwise
  /// produce silently wrong output. Use this constructor on untrusted
  /// inputs (custom decoders, unchecked FFI buffers, etc.).
  ///
  /// Cost: one O(plane_size) linear scan per plane. The chroma planes
  /// here are full-width × half-height (4:4:0 layout).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub fn try_new_checked(
    y: &'a [u16],
    u: &'a [u16],
    v: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    u_stride: u32,
    v_stride: u32,
  ) -> Result<Self, Yuv440pFrame16Error> {
    let frame = Self::try_new(y, u, v, width, height, y_stride, u_stride, v_stride)?;
    // No BITS == 16 early-return: `try_new` rejects everything outside
    // {10, 12}, so unlike Yuv420p/444p (which both accept 16) the
    // u16-saturating-noop case can't occur here.
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    let chroma_h = (height as usize).div_ceil(2);
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::Y,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * u_stride as usize;
      for (col, &s) in u[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::U,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * v_stride as usize;
      for (col, &s) in v[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(Yuv420pFrame16Error::SampleOutOfRange {
            plane: Yuv420pFrame16Plane::V,
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Y plane (`u16` elements).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y(&self) -> &'a [u16] {
    self.y
  }
  /// U plane. **Full-width, half-height.**
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u(&self) -> &'a [u16] {
    self.u
  }
  /// V plane. Full-width, half-height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v(&self) -> &'a [u16] {
    self.v
  }
  /// Frame width in pixels. No parity constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Y plane stride in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn y_stride(&self) -> u32 {
    self.y_stride
  }
  /// U plane stride in samples (full-width).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn u_stride(&self) -> u32 {
    self.u_stride
  }
  /// V plane stride in samples (full-width).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn v_stride(&self) -> u32 {
    self.v_stride
  }
  /// The `BITS` const parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// 4:4:0 planar, 10-bit (`AV_PIX_FMT_YUV440P10LE`). Alias over
/// [`Yuv440pFrame16`]`<10>`.
pub type Yuv440p10Frame<'a> = Yuv440pFrame16<'a, 10>;
/// 4:4:0 planar, 12-bit (`AV_PIX_FMT_YUV440P12LE`). Alias over
/// [`Yuv440pFrame16`]`<12>`.
pub type Yuv440p12Frame<'a> = Yuv440pFrame16<'a, 12>;

/// Errors returned by [`Yuv420pFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuv420pFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. YUV420p / 4:2:0 subsamples chroma 2:1 in width,
  /// so each chroma column pairs two Y columns — odd widths leave the
  /// last Y column without a paired chroma sample, and the SIMD
  /// kernels assume `width & 1 == 0`. Height is allowed to be odd
  /// (handled by `height.div_ceil(2)` in chroma‑row sizing).
  #[error("width ({width}) is odd; YUV420p / 4:2:0 requires even width")]
  OddWidth {
    /// The supplied width.
    width: u32,
  },
  /// `y_stride < width`.
  #[error("y_stride ({y_stride}) is smaller than width ({width})")]
  YStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied Y-plane stride.
    y_stride: u32,
  },
  /// `u_stride < ceil(width / 2)`.
  #[error("u_stride ({u_stride}) is smaller than chroma width ({chroma_width})")]
  UStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied U-plane stride.
    u_stride: u32,
  },
  /// `v_stride < ceil(width / 2)`.
  #[error("v_stride ({v_stride}) is smaller than chroma width ({chroma_width})")]
  VStrideTooSmall {
    /// The required minimum chroma-plane stride.
    chroma_width: u32,
    /// The supplied V-plane stride.
    v_stride: u32,
  },
  /// Y plane is shorter than `y_stride * height` bytes.
  #[error("Y plane has {actual} bytes but at least {expected} are required")]
  YPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// U plane is shorter than `u_stride * (height / 2)` bytes.
  #[error("U plane has {actual} bytes but at least {expected} are required")]
  UPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// V plane is shorter than `v_stride * (height / 2)` bytes.
  #[error("V plane has {actual} bytes but at least {expected} are required")]
  VPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (can only fire on 32‑bit
  /// targets — wasm32, i686 — with extreme dimensions).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated Bayer-mosaic frame at 8 bits per sample.
///
/// Single plane: each `u8` element is one sensor sample, with the
/// color (R / G / B) determined by the [`crate::raw::BayerPattern`]
/// passed at the walker boundary and the sample's `(row, column)`
/// position within the repeating 2×2 tile.
///
/// Odd `width` and `height` are accepted: a cropped Bayer plane
/// (post-production crop, sensor-specific active area) legitimately
/// exhibits a partial 2×2 tile at the right column or bottom row.
/// The walker clamps top / bottom rows and the demosaic kernel
/// clamps left / right columns, so the math is defined for every
/// site regardless of dimension parity.
///
/// `stride` is the sample stride of the plane — `>= width`,
/// permitting the upstream decoder to pad rows.
///
/// Source: FFmpeg's `bayer_bggr8` / `bayer_rggb8` / `bayer_grbg8` /
/// `bayer_gbrg8` decoders, vendor-SDK Bayer ingest paths (R3D /
/// BRAW / NRAW), and any custom RAW pipeline that has already
/// extracted a Bayer plane from the camera bitstream.
#[derive(Debug, Clone, Copy)]
pub struct BayerFrame<'a> {
  data: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> BayerFrame<'a> {
  /// Constructs a new [`BayerFrame`], validating dimensions and
  /// plane length.
  ///
  /// Returns [`BayerFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `stride < width`,
  /// - `data.len() < stride * height`, or
  /// - `stride * height` overflows `usize` (32‑bit targets only).
  ///
  /// Odd widths and heights are accepted; see the type-level docs
  /// for the rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, BayerFrameError> {
    if width == 0 || height == 0 {
      return Err(BayerFrameError::ZeroDimension { width, height });
    }
    // Odd Bayer widths and heights are accepted: a cropped Bayer
    // plane (post-production crop, sensor-specific active area)
    // legitimately exhibits a partial 2×2 tile at the right column
    // or bottom row. The walker clamps top / bottom rows and the
    // demosaic kernel clamps left / right columns, so the math is
    // defined for every site regardless of dimension parity.
    if stride < width {
      return Err(BayerFrameError::StrideTooSmall { width, stride });
    }
    let min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(BayerFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if data.len() < min {
      return Err(BayerFrameError::PlaneTooShort {
        expected: min,
        actual: data.len(),
      });
    }
    Ok(Self {
      data,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`BayerFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(data: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(data, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid BayerFrame dimensions or plane length"),
    }
  }

  /// The Bayer plane bytes. Row `r` starts at byte offset
  /// `r * stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn data(&self) -> &'a [u8] {
    self.data
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

  /// Byte stride of the Bayer plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// A validated Bayer-mosaic frame at 10 / 12 / 14 / 16 bits per
/// sample, **low-packed** in `u16` containers.
///
/// `BITS` ∈ {10, 12, 14, 16}; samples occupy the **low** `BITS`
/// bits of each `u16` (range `[0, (1 << BITS) - 1]`), with the high
/// `16 - BITS` bits zero. This matches the planar high-bit-depth
/// convention used by [`Yuv420pFrame16`], `Yuv422p*`, and
/// `Yuv444p*`. Note that this is **not** the [`PnFrame`]
/// (`P010` / `P012`) convention, which is high-bit-packed
/// (semi-planar `u16` containers carry samples in the *high* bits);
/// Bayer is single-plane and tracks the planar family instead.
///
/// **Type-level guarantee.** [`Self::try_new`] validates every
/// active sample against the low-packed range as part of
/// construction, so an existing `BayerFrame16<BITS>` value is
/// guaranteed to carry only in-range samples. Downstream
/// [`crate::raw::bayer16_to`] therefore needs no further
/// runtime validation and never panics on bad sample data —
/// any `Result::Err` from the conversion comes from the sink,
/// never from the frame's contents.
///
/// Diverges from the rest of the high-bit-depth crate
/// (`Yuv420pFrame16` / `Yuv422pFrame16` / `Yuv444pFrame16` ship a
/// cheap `try_new` + opt-in `try_new_checked`) because Bayer16
/// frames typically come from less-trusted RAW pipelines (vendor
/// SDKs, file loaders) and have no hot-path performance pressure
/// to skip the per-sample check. Mandatory validation makes the
/// `bayer16_to` walker fully fallible.
///
/// Odd widths and heights are accepted (cropped Bayer is a real
/// workflow; the kernel handles partial 2×2 tiles via edge
/// clamping).
///
/// Source: FFmpeg's `bayer_*16le` decoders, vendor-SDK
/// 10/12/14/16-bit RAW ingest paths. If your upstream provides
/// high-bit-packed Bayer (active bits in the *high* `BITS`),
/// right-shift each sample by `(16 - BITS)` before constructing
/// [`BayerFrame16`].
#[derive(Debug, Clone, Copy)]
pub struct BayerFrame16<'a, const BITS: u32> {
  data: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a, const BITS: u32> BayerFrame16<'a, BITS> {
  /// Constructs a new [`BayerFrame16`], validating dimensions,
  /// plane length, the `BITS` parameter, **and every active
  /// sample's value**.
  ///
  /// Unlike the rest of the high-bit-depth crate (`Yuv420pFrame16`,
  /// `Yuv422pFrame16`, etc.) which split the validation into
  /// `try_new` (geometry) + `try_new_checked` (samples), Bayer16
  /// always validates samples here. RAW pipelines often surface
  /// trusted-but-actually-mispacked input (MSB-aligned bytes from
  /// a sensor SDK, stale high bits from a copy that didn't mask
  /// the source), and downstream demosaic / WB / CCM math has no
  /// well-defined behavior on out-of-range samples. Catching at
  /// construction lets callers handle the failure as a normal
  /// `Result` instead of risking a panic later in
  /// [`crate::raw::bayer16_to`].
  ///
  /// `stride` is in **samples** (`u16` elements). Returns
  /// [`BayerFrame16Error`] if any of:
  /// - `BITS` is not 10, 12, 14, or 16,
  /// - `width` or `height` is zero,
  /// - `stride < width`,
  /// - `data.len() < stride * height`,
  /// - `stride * height` overflows `usize`, or
  /// - any sample's value exceeds `(1 << BITS) - 1` (returned as
  ///   [`BayerFrame16Error::SampleOutOfRange`]).
  ///
  /// Odd widths and heights are accepted; see the type-level docs
  /// for the rationale.
  ///
  /// Cost: O(width × height) sample scan in addition to the
  /// O(1) geometry checks. The scan is a tight loop over `u16`
  /// values per row and runs once per frame; downstream
  /// [`crate::raw::bayer16_to`] therefore needs no further
  /// sample validation.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new(
    data: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, BayerFrame16Error> {
    if BITS != 10 && BITS != 12 && BITS != 14 && BITS != 16 {
      return Err(BayerFrame16Error::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(BayerFrame16Error::ZeroDimension { width, height });
    }
    // Odd Bayer widths and heights are accepted; see
    // [`BayerFrame::try_new`] for the rationale (cropped Bayer is
    // a real workflow, edge clamping handles partial tiles).
    if stride < width {
      return Err(BayerFrame16Error::StrideTooSmall { width, stride });
    }
    let min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(BayerFrame16Error::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if data.len() < min {
      return Err(BayerFrame16Error::PlaneTooShort {
        expected: min,
        actual: data.len(),
      });
    }
    // Sample range scan — only the **active** per-row region
    // (`r * stride .. r * stride + width`) is checked. Row padding
    // and trailing storage are deliberately skipped because the
    // walker never reads them, matching the boundary contract of
    // the row dispatchers.
    let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
    let w = width as usize;
    let h = height as usize;
    for row in 0..h {
      let start = row * stride as usize;
      for (col, &s) in data[start..start + w].iter().enumerate() {
        if s > max_valid {
          return Err(BayerFrame16Error::SampleOutOfRange {
            index: start + col,
            value: s,
            max_valid,
          });
        }
      }
    }
    Ok(Self {
      data,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`BayerFrame16`], panicking on invalid inputs.
  /// Includes sample-range validation; see [`Self::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(data: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(data, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => {
        panic!("invalid BayerFrame16 dimensions, plane length, BITS value, or sample range")
      }
    }
  }

  /// The Bayer plane samples. Row `r` starts at sample offset
  /// `r * stride()`. Each `u16` carries the `BITS` active bits in
  /// its **low** `BITS` positions; the high `16 - BITS` bits are
  /// zero on well-formed input.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn data(&self) -> &'a [u16] {
    self.data
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

  /// Sample stride of the Bayer plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Active bit depth — 10, 12, 14, or 16.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// Type alias for a 10-bit Bayer frame — low-packed `u16` samples
/// with values in `[0, 1023]` (the high 6 bits are zero).
pub type Bayer10Frame<'a> = BayerFrame16<'a, 10>;
/// Type alias for a 12-bit Bayer frame.
pub type Bayer12Frame<'a> = BayerFrame16<'a, 12>;
/// Type alias for a 14-bit Bayer frame.
pub type Bayer14Frame<'a> = BayerFrame16<'a, 14>;
/// Type alias for a 16-bit Bayer frame.
pub type Bayer16Frame<'a> = BayerFrame16<'a, 16>;

/// Errors returned by [`BayerFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum BayerFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < width`.
  #[error("stride ({stride}) is smaller than width ({width})")]
  StrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied plane stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("Bayer plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (can only fire on
  /// 32‑bit targets — wasm32, i686 — with extreme dimensions).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// Errors returned by [`BayerFrame16::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum BayerFrame16Error {
  /// `BITS` const-generic parameter is not one of `{10, 12, 14, 16}`.
  #[error("BITS ({bits}) is not 10, 12, 14, or 16")]
  UnsupportedBits {
    /// The supplied `BITS` value.
    bits: u32,
  },
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < width` (in `u16` samples).
  #[error("stride ({stride}) is smaller than width ({width})")]
  StrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied plane stride (samples).
    stride: u32,
  },
  /// Plane is shorter than `stride * height` samples.
  #[error("Bayer plane has {actual} samples but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A sample's value exceeds `(1 << BITS) - 1` — the sample's
  /// high `16 - BITS` bits are non-zero, which is invalid under
  /// the low-packed Bayer16 convention. Returned by
  /// [`BayerFrame16::try_new`] (and [`BayerFrame16::new`] which
  /// wraps it) — sample-range validation is part of standard
  /// frame construction so the [`crate::raw::bayer16_to`] walker
  /// is fully fallible.
  #[error("sample {value} at element {index} exceeds {max_valid} ((1 << BITS) - 1)")]
  SampleOutOfRange {
    /// Element index within the plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
    /// The valid maximum at the declared `BITS` (`(1 << BITS) - 1`).
    max_valid: u16,
  },
}

// ============================================================
// Tier 6 — Packed RGB / BGR 8-bit source-side frames (Ship 9a)
// ============================================================

/// Errors returned by [`Rgb24Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Rgb24FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width`. Each row needs `3 * width` bytes for packed RGB.
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride (`3 * width`).
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("RGB plane has {actual} bytes but at least {expected} are required")]
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
  /// `3 * width` overflows `u32`.
  #[error("3 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **RGB24** frame at 8 bits per channel
/// (`AV_PIX_FMT_RGB24`). One plane, 3 bytes per pixel, byte order
/// `R, G, B`.
///
/// `stride` is in **bytes** (≥ `3 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct Rgb24Frame<'a> {
  rgb: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgb24Frame<'a> {
  /// Constructs a new [`Rgb24Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgb: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Rgb24FrameError> {
    if width == 0 || height == 0 {
      return Err(Rgb24FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Rgb24FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Rgb24FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Rgb24FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgb.len() < plane_min {
      return Err(Rgb24FrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgb.len(),
      });
    }
    Ok(Self {
      rgb,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Rgb24Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgb: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgb, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Rgb24Frame dimensions or plane length"),
    }
  }

  /// Packed RGB plane bytes (`R, G, B, R, G, B, …` per row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> &'a [u8] {
    self.rgb
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
  /// Byte stride (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Errors returned by [`Bgr24Frame::try_new`]. Variant shape mirrors
/// [`Rgb24FrameError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Bgr24FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width`.
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("BGR plane has {actual} bytes but at least {expected} are required")]
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
  /// `3 * width` overflows `u32`.
  #[error("3 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **BGR24** frame at 8 bits per channel
/// (`AV_PIX_FMT_BGR24`). One plane, 3 bytes per pixel, byte order
/// `B, G, R` — only the channel-order distinction differentiates
/// this from [`Rgb24Frame`].
///
/// `stride` is in **bytes** (≥ `3 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct Bgr24Frame<'a> {
  bgr: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Bgr24Frame<'a> {
  /// Constructs a new [`Bgr24Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    bgr: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Bgr24FrameError> {
    if width == 0 || height == 0 {
      return Err(Bgr24FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Bgr24FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Bgr24FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Bgr24FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if bgr.len() < plane_min {
      return Err(Bgr24FrameError::PlaneTooShort {
        expected: plane_min,
        actual: bgr.len(),
      });
    }
    Ok(Self {
      bgr,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Bgr24Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(bgr: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(bgr, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Bgr24Frame dimensions or plane length"),
    }
  }

  /// Packed BGR plane bytes (`B, G, R, B, G, R, …` per row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bgr(&self) -> &'a [u8] {
    self.bgr
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
  /// Byte stride (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

// ============================================================
// Tier 6 — Packed RGBA / BGRA 8-bit source-side frames (Ship 9b)
// ============================================================
//
// Both formats are single-plane 8 bits per channel, 4 bytes per
// pixel. The 4th byte is real alpha (not padding) — for the
// `0rgb` / `rgb0` / `0bgr` / `bgr0` family where the 4th byte is
// padding, the planned `RgbPaddingFrame` (Ship 9d) handles that
// case so callers can't accidentally treat padding as alpha.

/// Errors returned by [`RgbaFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum RgbaFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 4 * width`. Each row needs `4 * width` bytes for packed RGBA.
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride (`4 * width`).
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("RGBA plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **RGBA** frame at 8 bits per channel
/// (`AV_PIX_FMT_RGBA`). One plane, 4 bytes per pixel, byte order
/// `R, G, B, A`.
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint. The 4th byte is real alpha — for the `0rgb` / `rgb0`
/// / `0bgr` / `bgr0` padding-byte family (where the 4th byte is
/// ignored padding, not alpha) see the planned Ship 9d
/// `RgbPaddingFrame` type.
#[derive(Debug, Clone, Copy)]
pub struct RgbaFrame<'a> {
  rgba: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> RgbaFrame<'a> {
  /// Constructs a new [`RgbaFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgba: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, RgbaFrameError> {
    if width == 0 || height == 0 {
      return Err(RgbaFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(RgbaFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(RgbaFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(RgbaFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgba.len() < plane_min {
      return Err(RgbaFrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgba.len(),
      });
    }
    Ok(Self {
      rgba,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`RgbaFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgba: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgba, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid RgbaFrame dimensions or plane length"),
    }
  }

  /// Packed RGBA plane bytes (`R, G, B, A` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgba(&self) -> &'a [u8] {
    self.rgba
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

/// Errors returned by [`BgraFrame::try_new`]. Variant shape mirrors
/// [`RgbaFrameError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum BgraFrameError {
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
  #[error("BGRA plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **BGRA** frame at 8 bits per channel
/// (`AV_PIX_FMT_BGRA`). One plane, 4 bytes per pixel, byte order
/// `B, G, R, A` — channel-order distinction from [`RgbaFrame`]
/// is at the kernel level (sinker swaps `R↔B` while keeping `A`).
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct BgraFrame<'a> {
  bgra: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> BgraFrame<'a> {
  /// Constructs a new [`BgraFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    bgra: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, BgraFrameError> {
    if width == 0 || height == 0 {
      return Err(BgraFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(BgraFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(BgraFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(BgraFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if bgra.len() < plane_min {
      return Err(BgraFrameError::PlaneTooShort {
        expected: plane_min,
        actual: bgra.len(),
      });
    }
    Ok(Self {
      bgra,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`BgraFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(bgra: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(bgra, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid BgraFrame dimensions or plane length"),
    }
  }

  /// Packed BGRA plane bytes (`B, G, R, A` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bgra(&self) -> &'a [u8] {
    self.bgra
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

/// Errors returned by [`ArgbFrame::try_new`]. Variant shape mirrors
/// [`RgbaFrameError`] — only the channel order on the four bytes
/// per pixel differs at the kernel level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum ArgbFrameError {
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
  #[error("ARGB plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **ARGB** frame at 8 bits per channel
/// (`AV_PIX_FMT_ARGB`). One plane, 4 bytes per pixel, byte order
/// `A, R, G, B` — alpha is at the **leading** position (byte 0),
/// vs trailing for [`RgbaFrame`].
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint. The 1st byte is real alpha — for the `0rgb` / `rgb0`
/// / `0bgr` / `bgr0` padding-byte family (where the alpha-position
/// byte is ignored padding, not alpha) see the planned Ship 9d
/// `RgbPaddingFrame` type.
#[derive(Debug, Clone, Copy)]
pub struct ArgbFrame<'a> {
  argb: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> ArgbFrame<'a> {
  /// Constructs a new [`ArgbFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    argb: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, ArgbFrameError> {
    if width == 0 || height == 0 {
      return Err(ArgbFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(ArgbFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(ArgbFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(ArgbFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if argb.len() < plane_min {
      return Err(ArgbFrameError::PlaneTooShort {
        expected: plane_min,
        actual: argb.len(),
      });
    }
    Ok(Self {
      argb,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`ArgbFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(argb: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(argb, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid ArgbFrame dimensions or plane length"),
    }
  }

  /// Packed ARGB plane bytes (`A, R, G, B` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn argb(&self) -> &'a [u8] {
    self.argb
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

/// Errors returned by [`AbgrFrame::try_new`]. Variant shape mirrors
/// [`ArgbFrameError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum AbgrFrameError {
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
  #[error("ABGR plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **ABGR** frame at 8 bits per channel
/// (`AV_PIX_FMT_ABGR`). One plane, 4 bytes per pixel, byte order
/// `A, B, G, R` — leading alpha + reversed RGB order vs
/// [`ArgbFrame`].
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct AbgrFrame<'a> {
  abgr: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> AbgrFrame<'a> {
  /// Constructs a new [`AbgrFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    abgr: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, AbgrFrameError> {
    if width == 0 || height == 0 {
      return Err(AbgrFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(AbgrFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(AbgrFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(AbgrFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if abgr.len() < plane_min {
      return Err(AbgrFrameError::PlaneTooShort {
        expected: plane_min,
        actual: abgr.len(),
      });
    }
    Ok(Self {
      abgr,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`AbgrFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(abgr: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(abgr, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid AbgrFrame dimensions or plane length"),
    }
  }

  /// Packed ABGR plane bytes (`A, B, G, R` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn abgr(&self) -> &'a [u8] {
    self.abgr
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

/// Errors returned by [`XrgbFrame::try_new`]. Variant shape mirrors
/// the [`RgbaFrameError`] family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum XrgbFrameError {
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
  #[error("XRGB plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **0RGB** frame at 8 bits per channel
/// (`AV_PIX_FMT_0RGB`). One plane, 4 bytes per pixel, byte order
/// `X, R, G, B` — the leading byte is **ignored padding** (not real
/// alpha — see [`ArgbFrame`] for the alpha-bearing analogue).
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct XrgbFrame<'a> {
  xrgb: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> XrgbFrame<'a> {
  /// Constructs a new [`XrgbFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    xrgb: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, XrgbFrameError> {
    if width == 0 || height == 0 {
      return Err(XrgbFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(XrgbFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(XrgbFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(XrgbFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if xrgb.len() < plane_min {
      return Err(XrgbFrameError::PlaneTooShort {
        expected: plane_min,
        actual: xrgb.len(),
      });
    }
    Ok(Self {
      xrgb,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`XrgbFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(xrgb: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(xrgb, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid XrgbFrame dimensions or plane length"),
    }
  }

  /// Packed XRGB plane bytes (leading padding byte ignored).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn xrgb(&self) -> &'a [u8] {
    self.xrgb
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

/// Errors returned by [`RgbxFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum RgbxFrameError {
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
  #[error("RGBX plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **RGB0** frame at 8 bits per channel
/// (`AV_PIX_FMT_RGB0`). One plane, 4 bytes per pixel, byte order
/// `R, G, B, X` — the trailing byte is **ignored padding**.
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct RgbxFrame<'a> {
  rgbx: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> RgbxFrame<'a> {
  /// Constructs a new [`RgbxFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgbx: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, RgbxFrameError> {
    if width == 0 || height == 0 {
      return Err(RgbxFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(RgbxFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(RgbxFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(RgbxFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgbx.len() < plane_min {
      return Err(RgbxFrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgbx.len(),
      });
    }
    Ok(Self {
      rgbx,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`RgbxFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgbx: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgbx, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid RgbxFrame dimensions or plane length"),
    }
  }

  /// Packed RGBX plane bytes (trailing padding byte ignored).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgbx(&self) -> &'a [u8] {
    self.rgbx
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

/// Errors returned by [`XbgrFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum XbgrFrameError {
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
  #[error("XBGR plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **0BGR** frame at 8 bits per channel
/// (`AV_PIX_FMT_0BGR`). One plane, 4 bytes per pixel, byte order
/// `X, B, G, R` — leading padding + reversed RGB order vs
/// [`XrgbFrame`].
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct XbgrFrame<'a> {
  xbgr: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> XbgrFrame<'a> {
  /// Constructs a new [`XbgrFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    xbgr: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, XbgrFrameError> {
    if width == 0 || height == 0 {
      return Err(XbgrFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(XbgrFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(XbgrFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(XbgrFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if xbgr.len() < plane_min {
      return Err(XbgrFrameError::PlaneTooShort {
        expected: plane_min,
        actual: xbgr.len(),
      });
    }
    Ok(Self {
      xbgr,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`XbgrFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(xbgr: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(xbgr, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid XbgrFrame dimensions or plane length"),
    }
  }

  /// Packed XBGR plane bytes (leading padding byte ignored).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn xbgr(&self) -> &'a [u8] {
    self.xbgr
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

/// Errors returned by [`BgrxFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum BgrxFrameError {
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
  #[error("BGRX plane has {actual} bytes but at least {expected} are required")]
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

/// A validated packed **BGR0** frame at 8 bits per channel
/// (`AV_PIX_FMT_BGR0`). One plane, 4 bytes per pixel, byte order
/// `B, G, R, X` — trailing padding + reversed RGB order vs
/// [`RgbxFrame`].
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
#[derive(Debug, Clone, Copy)]
pub struct BgrxFrame<'a> {
  bgrx: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> BgrxFrame<'a> {
  /// Constructs a new [`BgrxFrame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    bgrx: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, BgrxFrameError> {
    if width == 0 || height == 0 {
      return Err(BgrxFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(BgrxFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(BgrxFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(BgrxFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if bgrx.len() < plane_min {
      return Err(BgrxFrameError::PlaneTooShort {
        expected: plane_min,
        actual: bgrx.len(),
      });
    }
    Ok(Self {
      bgrx,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`BgrxFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(bgrx: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(bgrx, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid BgrxFrame dimensions or plane length"),
    }
  }

  /// Packed BGRX plane bytes (trailing padding byte ignored).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bgrx(&self) -> &'a [u8] {
    self.bgrx
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

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
