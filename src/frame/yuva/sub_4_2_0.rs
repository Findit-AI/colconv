use derive_more::{Display, IsVariant};
use thiserror::Error;

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
