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
  /// - `width` or `height` is zero or odd,
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
  /// - `width` or `height` is zero or odd,
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
}
