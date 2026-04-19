//! Validated source-frame types.
//!
//! Each pixel family has its own frame struct carrying the backing
//! plane slice(s), pixel dimensions, and byte strides. Construction
//! validates strides vs. widths and that each plane covers its
//! declared area.

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
/// (10, 12, or 14); samples occupy the **low** `BITS` bits of each
/// `u16` with the upper bits zero. This matches FFmpeg's little‑endian
/// `yuv420p10le` / `yuv420p12le` / `yuv420p14le` convention, where
/// each plane is a byte buffer reinterpretable as `u16` little‑endian.
///
/// Ship 2 ships `BITS == 10` only (the use‑case keystone for HDR and
/// 10‑bit SDR). 12 and 14 are mechanical follow‑ups that just relax
/// the constructor's `BITS` check and add a tiered aliases — the
/// kernel math (Q15 coefficients + i32 intermediates) works unchanged
/// across all three, derived at compile time from `BITS`.
///
/// 16‑bit input (which would overflow the i32 chroma sum in the
/// Q15 path) is **not** represented by this type — it needs a
/// separate kernel family with i64 intermediates or a lower Q
/// coefficient format. That lands in a later ship.
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
  /// - `BITS` is not 10, 12, or 14 (Ship 2 additionally rejects 12/14
  ///   at the type alias layer — see [`Yuv420p10Frame`]),
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
    // Guard the `BITS` parameter at the top so users who accidentally
    // monomorphize on e.g. `BITS == 8` (which would work numerically
    // but should go through [`Yuv420pFrame`] instead) or `BITS == 16`
    // (which would overflow the i32 chroma sum in the Q15 kernel)
    // get a clear error rather than silently wrong output.
    if BITS != 10 && BITS != 12 && BITS != 14 {
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

  /// Active bit depth — 10, 12, or 14. Mirrors the `BITS` const
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

/// Errors returned by [`Yuv420pFrame16::try_new`]. Variant shape
/// mirrors [`Yuv420pFrameError`], with `UnsupportedBits` added for
/// the new `BITS` parameter and all sizes expressed in **samples**
/// (`u16` elements) instead of bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Yuv420pFrame16Error {
  /// `BITS` was not one of the supported depths (10, 12, 14). 8‑bit
  /// frames should use [`Yuv420pFrame`]; 16‑bit needs a separate
  /// kernel family (see [`Yuv420pFrame16`] docs).
  #[error("unsupported BITS ({bits}) for Yuv420pFrame16; must be 10, 12, or 14")]
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
}

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

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests {
  use super::*;

  fn planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
    // 16×8 frame, U/V are 8×4.
    (
      std::vec![0u8; 16 * 8],
      std::vec![128u8; 8 * 4],
      std::vec![128u8; 8 * 4],
    )
  }

  #[test]
  fn try_new_accepts_valid_tight() {
    let (y, u, v) = planes();
    let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.height(), 8);
  }

  #[test]
  fn try_new_accepts_valid_padded_strides() {
    // 16×8 frame, strides padded (32 for y, 16 for u/v).
    let y = std::vec![0u8; 32 * 8];
    let u = std::vec![128u8; 16 * 4];
    let v = std::vec![128u8; 16 * 4];
    let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 32, 16, 16).expect("valid");
    assert_eq!(f.y_stride(), 32);
  }

  #[test]
  fn try_new_rejects_zero_dim() {
    let (y, u, v) = planes();
    let e = Yuv420pFrame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::ZeroDimension { .. }));
  }

  #[test]
  fn try_new_rejects_odd_width() {
    let (y, u, v) = planes();
    let e = Yuv420pFrame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::OddWidth { width: 15 }));
  }

  #[test]
  fn try_new_accepts_odd_height() {
    // 16x9 frame — chroma_height = ceil(9/2) = 5. Y plane 16*9 = 144
    // bytes, U/V plane 8*5 = 40 bytes each. Valid 4:2:0 frame;
    // height=9 must not be rejected just because it's odd.
    let y = std::vec![0u8; 16 * 9];
    let u = std::vec![128u8; 8 * 5];
    let v = std::vec![128u8; 8 * 5];
    let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
    assert_eq!(f.height(), 9);
  }

  #[test]
  fn try_new_rejects_y_stride_under_width() {
    let y = std::vec![0u8; 16 * 8];
    let u = std::vec![128u8; 8 * 4];
    let v = std::vec![128u8; 8 * 4];
    let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 8, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::YStrideTooSmall { .. }));
  }

  #[test]
  fn try_new_rejects_short_y_plane() {
    let y = std::vec![0u8; 10];
    let u = std::vec![128u8; 8 * 4];
    let v = std::vec![128u8; 8 * 4];
    let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::YPlaneTooShort { .. }));
  }

  #[test]
  fn try_new_rejects_short_u_plane() {
    let y = std::vec![0u8; 16 * 8];
    let u = std::vec![128u8; 4];
    let v = std::vec![128u8; 8 * 4];
    let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::UPlaneTooShort { .. }));
  }

  #[test]
  #[should_panic(expected = "invalid Yuv420pFrame")]
  fn new_panics_on_invalid() {
    let y = std::vec![0u8; 10];
    let u = std::vec![128u8; 8 * 4];
    let v = std::vec![128u8; 8 * 4];
    let _ = Yuv420pFrame::new(&y, &u, &v, 16, 8, 16, 8, 8);
  }

  // ---- Nv12Frame ---------------------------------------------------------

  fn nv12_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
    // 16×8 frame → UV is 8 chroma columns × 4 chroma rows = 16 bytes/row.
    (std::vec![0u8; 16 * 8], std::vec![128u8; 16 * 4])
  }

  #[test]
  fn nv12_try_new_accepts_valid_tight() {
    let (y, uv) = nv12_planes();
    let f = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.height(), 8);
    assert_eq!(f.uv_stride(), 16);
  }

  #[test]
  fn nv12_try_new_accepts_valid_padded_strides() {
    let y = std::vec![0u8; 32 * 8];
    let uv = std::vec![128u8; 32 * 4];
    let f = Nv12Frame::try_new(&y, &uv, 16, 8, 32, 32).expect("valid");
    assert_eq!(f.y_stride(), 32);
    assert_eq!(f.uv_stride(), 32);
  }

  #[test]
  fn nv12_try_new_rejects_zero_dim() {
    let (y, uv) = nv12_planes();
    let e = Nv12Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv12FrameError::ZeroDimension { .. }));
  }

  #[test]
  fn nv12_try_new_rejects_odd_width() {
    let (y, uv) = nv12_planes();
    let e = Nv12Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv12FrameError::OddWidth { width: 15 }));
  }

  #[test]
  fn nv12_try_new_accepts_odd_height() {
    // 640x481 — concrete case flagged by adversarial review. chroma_height =
    // ceil(481/2) = 241, so UV plane is 640*241 bytes. Constructor must
    // accept this.
    let y = std::vec![0u8; 640 * 481];
    let uv = std::vec![128u8; 640 * 241];
    let f = Nv12Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
    assert_eq!(f.height(), 481);
    assert_eq!(f.width(), 640);
  }

  #[test]
  fn nv12_try_new_rejects_y_stride_under_width() {
    let (y, uv) = nv12_planes();
    let e = Nv12Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
    assert!(matches!(e, Nv12FrameError::YStrideTooSmall { .. }));
  }

  #[test]
  fn nv12_try_new_rejects_uv_stride_under_width() {
    let (y, uv) = nv12_planes();
    let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
    assert!(matches!(e, Nv12FrameError::UvStrideTooSmall { .. }));
  }

  #[test]
  fn nv12_try_new_rejects_short_y_plane() {
    let y = std::vec![0u8; 10];
    let uv = std::vec![128u8; 16 * 4];
    let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv12FrameError::YPlaneTooShort { .. }));
  }

  #[test]
  fn nv12_try_new_rejects_short_uv_plane() {
    let y = std::vec![0u8; 16 * 8];
    let uv = std::vec![128u8; 8];
    let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv12FrameError::UvPlaneTooShort { .. }));
  }

  #[test]
  #[should_panic(expected = "invalid Nv12Frame")]
  fn nv12_new_panics_on_invalid() {
    let y = std::vec![0u8; 10];
    let uv = std::vec![128u8; 16 * 4];
    let _ = Nv12Frame::new(&y, &uv, 16, 8, 16, 16);
  }

  // ---- 32-bit overflow regressions --------------------------------------
  //
  // `u32 * u32` can exceed `usize::MAX` only on 32-bit targets (wasm32,
  // i686). Gate the tests so they actually run on those hosts under CI
  // cross builds; on 64-bit they're trivially uninteresting (the
  // product always fits).

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn yuv420p_try_new_rejects_y_geometry_overflow() {
    // 0x1_0000 * 0x1_0000 = 2^32, which overflows a 32-bit `usize`
    // (max = 2^32 − 1). Even so the odd-width check passes, so we
    // actually reach `checked_mul` and hit `GeometryOverflow`.
    let big: u32 = 0x1_0000;
    let y: [u8; 0] = [];
    let u: [u8; 0] = [];
    let v: [u8; 0] = [];
    let e = Yuv420pFrame::try_new(&y, &u, &v, big, big, big, big / 2, big / 2).unwrap_err();
    assert!(matches!(e, Yuv420pFrameError::GeometryOverflow { .. }));
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn nv12_try_new_rejects_geometry_overflow() {
    let big: u32 = 0x1_0000;
    let y: [u8; 0] = [];
    let uv: [u8; 0] = [];
    let e = Nv12Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
    assert!(matches!(e, Nv12FrameError::GeometryOverflow { .. }));
  }

  // ---- Nv21Frame ---------------------------------------------------------
  //
  // NV21 is structurally identical to NV12 (same plane count, same
  // stride/size math) — only the byte order within the chroma plane
  // differs. Validation tests mirror the NV12 set. Kernel-level
  // equivalence with NV12-swapped-UV is tested in `src/row/arch/*`.

  fn nv21_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
    // 16×8 frame → VU is 16 bytes × 4 chroma rows.
    (std::vec![0u8; 16 * 8], std::vec![128u8; 16 * 4])
  }

  #[test]
  fn nv21_try_new_accepts_valid_tight() {
    let (y, vu) = nv21_planes();
    let f = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 16).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.height(), 8);
    assert_eq!(f.vu_stride(), 16);
  }

  #[test]
  fn nv21_try_new_accepts_odd_height() {
    // Same concrete case as NV12 — 640x481.
    let y = std::vec![0u8; 640 * 481];
    let vu = std::vec![128u8; 640 * 241];
    let f = Nv21Frame::try_new(&y, &vu, 640, 481, 640, 640).expect("odd height valid");
    assert_eq!(f.height(), 481);
  }

  #[test]
  fn nv21_try_new_rejects_odd_width() {
    let (y, vu) = nv21_planes();
    let e = Nv21Frame::try_new(&y, &vu, 15, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv21FrameError::OddWidth { width: 15 }));
  }

  #[test]
  fn nv21_try_new_rejects_zero_dim() {
    let (y, vu) = nv21_planes();
    let e = Nv21Frame::try_new(&y, &vu, 0, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv21FrameError::ZeroDimension { .. }));
  }

  #[test]
  fn nv21_try_new_rejects_vu_stride_under_width() {
    let (y, vu) = nv21_planes();
    let e = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 8).unwrap_err();
    assert!(matches!(e, Nv21FrameError::VuStrideTooSmall { .. }));
  }

  #[test]
  fn nv21_try_new_rejects_short_vu_plane() {
    let y = std::vec![0u8; 16 * 8];
    let vu = std::vec![128u8; 8];
    let e = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, Nv21FrameError::VuPlaneTooShort { .. }));
  }

  #[test]
  #[should_panic(expected = "invalid Nv21Frame")]
  fn nv21_new_panics_on_invalid() {
    let y = std::vec![0u8; 10];
    let vu = std::vec![128u8; 16 * 4];
    let _ = Nv21Frame::new(&y, &vu, 16, 8, 16, 16);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn nv21_try_new_rejects_geometry_overflow() {
    let big: u32 = 0x1_0000;
    let y: [u8; 0] = [];
    let vu: [u8; 0] = [];
    let e = Nv21Frame::try_new(&y, &vu, big, big, big, big).unwrap_err();
    assert!(matches!(e, Nv21FrameError::GeometryOverflow { .. }));
  }

  // ---- Yuv420pFrame16 / Yuv420p10Frame ----------------------------------
  //
  // Storage is `&[u16]` with sample-indexed strides. Validation mirrors
  // the 8-bit [`Yuv420pFrame`] with the addition of the `BITS` guard.

  fn p10_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
    // 16×8 frame, chroma 8×4. Neutral 10-bit mid-gray (Y=512, UV=512).
    (
      std::vec![0u16; 16 * 8],
      std::vec![512u16; 8 * 4],
      std::vec![512u16; 8 * 4],
    )
  }

  #[test]
  fn yuv420p10_try_new_accepts_valid_tight() {
    let (y, u, v) = p10_planes();
    let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.height(), 8);
    assert_eq!(f.bits(), 10);
  }

  #[test]
  fn yuv420p10_try_new_accepts_odd_height() {
    // 16x9 → chroma_height = 5. Y plane 16*9 = 144 samples, U/V 8*5 = 40.
    let y = std::vec![0u16; 16 * 9];
    let u = std::vec![512u16; 8 * 5];
    let v = std::vec![512u16; 8 * 5];
    let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
    assert_eq!(f.height(), 9);
  }

  #[test]
  fn yuv420p10_try_new_rejects_odd_width() {
    let (y, u, v) = p10_planes();
    let e = Yuv420p10Frame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::OddWidth { width: 15 }));
  }

  #[test]
  fn yuv420p10_try_new_rejects_zero_dim() {
    let (y, u, v) = p10_planes();
    let e = Yuv420p10Frame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::ZeroDimension { .. }));
  }

  #[test]
  fn yuv420p10_try_new_rejects_short_y_plane() {
    let y = std::vec![0u16; 10];
    let u = std::vec![512u16; 8 * 4];
    let v = std::vec![512u16; 8 * 4];
    let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
  }

  #[test]
  fn yuv420p10_try_new_rejects_short_u_plane() {
    let y = std::vec![0u16; 16 * 8];
    let u = std::vec![512u16; 4];
    let v = std::vec![512u16; 8 * 4];
    let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::UPlaneTooShort { .. }));
  }

  #[test]
  fn yuv420p16_try_new_rejects_unsupported_bits() {
    // BITS == 9 is not in {10, 12, 14}; the constructor must reject it
    // before any plane math runs. This also exercises the 12/14 path
    // at the validation layer even though Ship 2 only ships a 10-bit
    // alias.
    let y = std::vec![0u16; 16 * 8];
    let u = std::vec![128u16; 8 * 4];
    let v = std::vec![128u16; 8 * 4];
    let e = Yuv420pFrame16::<9>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(
      e,
      Yuv420pFrame16Error::UnsupportedBits { bits: 9 }
    ));

    let e16 = Yuv420pFrame16::<16>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(
      e16,
      Yuv420pFrame16Error::UnsupportedBits { bits: 16 }
    ));
  }

  #[test]
  fn yuv420p16_try_new_accepts_12_and_14() {
    // The constructor admits 12 and 14 — Ship 2 doesn't ship kernels
    // for them but the geometry validator shouldn't block the types.
    let y = std::vec![0u16; 16 * 8];
    let u = std::vec![2048u16; 8 * 4];
    let v = std::vec![2048u16; 8 * 4];
    let f12 = Yuv420pFrame16::<12>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("12-bit valid");
    assert_eq!(f12.bits(), 12);
    let f14 = Yuv420pFrame16::<14>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("14-bit valid");
    assert_eq!(f14.bits(), 14);
  }

  #[test]
  #[should_panic(expected = "invalid Yuv420pFrame16")]
  fn yuv420p10_new_panics_on_invalid() {
    let y = std::vec![0u16; 10];
    let u = std::vec![512u16; 8 * 4];
    let v = std::vec![512u16; 8 * 4];
    let _ = Yuv420p10Frame::new(&y, &u, &v, 16, 8, 16, 8, 8);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn yuv420p10_try_new_rejects_geometry_overflow() {
    // Sample count overflow on 32-bit. Same rationale as the 8-bit
    // version — strides are in `u16` elements here, so the same
    // `0x1_0000 * 0x1_0000` product overflows `usize`.
    let big: u32 = 0x1_0000;
    let y: [u16; 0] = [];
    let u: [u16; 0] = [];
    let v: [u16; 0] = [];
    let e = Yuv420p10Frame::try_new(&y, &u, &v, big, big, big, big / 2, big / 2).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::GeometryOverflow { .. }));
  }
}
