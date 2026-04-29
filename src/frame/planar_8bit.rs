use derive_more::IsVariant;
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
