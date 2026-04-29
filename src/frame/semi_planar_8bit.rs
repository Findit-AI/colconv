use derive_more::IsVariant;
use thiserror::Error;

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
