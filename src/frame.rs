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
/// Each `u16` sample's 10 active bits live in the high 10 positions;
/// the low 6 bits are expected to be zero. [`Self::try_new`] validates
/// geometry only.
///
/// [`Self::try_new_checked`] additionally scans every sample and
/// rejects any with non‑zero low 6 bits — a **necessary but not
/// sufficient** packing sanity check. It catches mispacked
/// `yuv420p10le` buffers as long as **at least one** sample has
/// low‑bit content (the usual case for noisy real‑world image data),
/// but it **cannot distinguish** P010 from a `yuv420p10le` buffer
/// whose samples all happen to be multiples of 64. Values like
/// `Y = 64` (limited‑range black) and `UV = 512` (neutral chroma)
/// both have low 6 bits zero and so pass the check, even though the
/// buffer layout is wrong. For strict provenance, callers must rely
/// on their source format metadata and pick the right frame type
/// ([`P010Frame`] vs [`Yuv420p10Frame`]) at construction.
///
/// Kernels shift each load right by 6 to extract the 10‑bit value,
/// so mispacked input (e.g. a `yuv420p10le` buffer handed to the
/// P010 kernel) produces deterministic, backend‑independent output
/// — wrong colors, but consistently wrong across scalar + every
/// SIMD backend, which is visible in any output diff.
#[derive(Debug, Clone, Copy)]
pub struct P010Frame<'a> {
  y: &'a [u16],
  uv: &'a [u16],
  width: u32,
  height: u32,
  y_stride: u32,
  uv_stride: u32,
}

impl<'a> P010Frame<'a> {
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
  ) -> Result<Self, P010FrameError> {
    if width == 0 || height == 0 {
      return Err(P010FrameError::ZeroDimension { width, height });
    }
    if width & 1 != 0 {
      return Err(P010FrameError::OddWidth { width });
    }
    if y_stride < width {
      return Err(P010FrameError::YStrideTooSmall { width, y_stride });
    }
    let uv_row_elems = width;
    if uv_stride < uv_row_elems {
      return Err(P010FrameError::UvStrideTooSmall {
        uv_row_elems,
        uv_stride,
      });
    }

    let y_min = match (y_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(P010FrameError::GeometryOverflow {
          stride: y_stride,
          rows: height,
        });
      }
    };
    if y.len() < y_min {
      return Err(P010FrameError::YPlaneTooShort {
        expected: y_min,
        actual: y.len(),
      });
    }
    let chroma_height = height.div_ceil(2);
    let uv_min = match (uv_stride as usize).checked_mul(chroma_height as usize) {
      Some(v) => v,
      None => {
        return Err(P010FrameError::GeometryOverflow {
          stride: uv_stride,
          rows: chroma_height,
        });
      }
    };
    if uv.len() < uv_min {
      return Err(P010FrameError::UvPlaneTooShort {
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
      Err(_) => panic!("invalid P010Frame dimensions or plane lengths"),
    }
  }

  /// Like [`Self::try_new`] but additionally scans every sample and
  /// rejects any whose **low 6 bits** are non‑zero. A valid P010
  /// sample has its 10 active bits in the high 10 positions and zero
  /// below, so non‑zero low bits is evidence the buffer isn't P010.
  ///
  /// **This is a packing sanity check, not a provenance validator.**
  /// The check catches noisy `yuv420p10le` data (where most samples
  /// have low‑bit content), but it **cannot** distinguish P010 from
  /// a `yuv420p10le` buffer whose samples all happen to be multiples
  /// of 64. Common flat‑region values like `Y = 64` (limited‑range
  /// black) or `UV = 512` (neutral chroma) are multiples of 64 in
  /// both layouts, so a yuv420p10le buffer of flat content will
  /// silently pass this check. Callers who need strict provenance
  /// must rely on their source format metadata and pick the right
  /// frame type at construction ([`P010Frame`] vs [`Yuv420p10Frame`]);
  /// no runtime check on opaque `u16` data can reliably tell the two
  /// layouts apart.
  ///
  /// Cost: one O(plane_size) scan per plane. The default
  /// [`Self::try_new`] skips this so the hot path stays O(1).
  ///
  /// Returns [`P010FrameError::SampleLowBitsSet`] on the first
  /// offending sample — carries the plane, element index, and
  /// offending value.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    y: &'a [u16],
    uv: &'a [u16],
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
  ) -> Result<Self, P010FrameError> {
    let frame = Self::try_new(y, uv, width, height, y_stride, uv_stride)?;
    let w = width as usize;
    let h = height as usize;
    let uv_w = w; // interleaved: `width / 2` pairs × 2 elements
    let chroma_h = height.div_ceil(2) as usize;
    for row in 0..h {
      let start = row * y_stride as usize;
      for (col, &s) in y[start..start + w].iter().enumerate() {
        if s & 0x3F != 0 {
          return Err(P010FrameError::SampleLowBitsSet {
            plane: P010FramePlane::Y,
            index: start + col,
            value: s,
          });
        }
      }
    }
    for row in 0..chroma_h {
      let start = row * uv_stride as usize;
      for (col, &s) in uv[start..start + uv_w].iter().enumerate() {
        if s & 0x3F != 0 {
          return Err(P010FrameError::SampleLowBitsSet {
            plane: P010FramePlane::Uv,
            index: start + col,
            value: s,
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
}

/// Identifies which plane of a [`P010Frame`] a
/// [`P010FrameError::SampleLowBitsSet`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum P010FramePlane {
  /// Luma plane.
  Y,
  /// Interleaved UV plane.
  Uv,
}

/// Errors returned by [`P010Frame::try_new`] and
/// [`P010Frame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum P010FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `width` was odd. Same 4:2:0 rationale as the other semi‑planar
  /// formats.
  #[error("width ({width}) is odd; 4:2:0 requires even width")]
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
  /// `uv_stride` is smaller than the `width` `u16` elements of
  /// interleaved UV payload one chroma row must hold.
  #[error("uv_stride ({uv_stride}) is smaller than UV row payload ({uv_row_elems} u16 elements)")]
  UvStrideTooSmall {
    /// Required minimum UV‑plane stride (`= width`).
    uv_row_elems: u32,
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
  /// `stride * rows` overflows `usize` (32‑bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// A sample's low 6 bits were non‑zero — P010 packs its 10 active
  /// bits in the high 10 of each `u16`, so valid samples are always
  /// multiples of 64 (`value << 6`). Only
  /// [`P010Frame::try_new_checked`] can produce this error.
  ///
  /// Note: the absence of this error does **not** prove the buffer
  /// is P010. A `yuv420p10le` buffer of samples that all happen to
  /// be multiples of 64 (e.g. `Y = 64`, `UV = 512`) passes the
  /// check silently. See [`P010Frame::try_new_checked`] for the
  /// full discussion.
  #[error(
    "sample {value:#06x} on plane {plane} at element {index} has non-zero low 6 bits (not a valid P010 sample)"
  )]
  SampleLowBitsSet {
    /// Which plane the offending sample lives on.
    plane: P010FramePlane,
    /// Element index within that plane's slice.
    index: usize,
    /// The offending sample value.
    value: u16,
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
/// (10, 12, or 14). Callers are **expected** to store each sample in
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
/// colconv v0.2 ships `BITS == 10` only (the use‑case keystone for
/// HDR and 10‑bit SDR). 12 and 14 are mechanical follow‑ups that
/// just relax the constructor's `BITS` check and add tiered aliases
/// — the kernel math (Q15 coefficients + i32 intermediates) works
/// unchanged across all three, derived at compile time from `BITS`.
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
  /// - `BITS` is not 10, 12, or 14 (colconv v0.2 additionally rejects
  ///   12/14 at the type alias layer — see [`Yuv420p10Frame`]),
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
    // 16×8 frame, chroma 8×4. Y plane solid black (Y=0); UV planes
    // neutral (UV=512 = 10‑bit chroma center). Exact sample values
    // don't matter for the constructor tests that use this helper —
    // they only look at shape, geometry errors, and the reported
    // bits.
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

  #[test]
  fn yuv420p10_try_new_checked_accepts_in_range_samples() {
    // Same valid frame as `yuv420p10_try_new_accepts_valid_tight`,
    // but run through the checked constructor. All samples live in
    // the 10‑bit range.
    let (y, u, v) = p10_planes();
    let f = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.bits(), 10);
  }

  #[test]
  fn yuv420p10_try_new_checked_rejects_y_high_bit_set() {
    // A Y sample with bit 15 set — typical of `p010` packing where
    // the 10 active bits sit in the high bits. `try_new` would
    // accept this and let the SIMD kernels produce arch‑dependent
    // garbage; `try_new_checked` catches it up front.
    let mut y = std::vec![0u16; 16 * 8];
    y[3 * 16 + 5] = 0x8000; // bit 15 set → way above 1023
    let u = std::vec![512u16; 8 * 4];
    let v = std::vec![512u16; 8 * 4];
    let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    match e {
      Yuv420pFrame16Error::SampleOutOfRange {
        plane,
        value,
        max_valid,
        ..
      } => {
        assert_eq!(plane, Yuv420pFrame16Plane::Y);
        assert_eq!(value, 0x8000);
        assert_eq!(max_valid, 1023);
      }
      other => panic!("expected SampleOutOfRange, got {other:?}"),
    }
  }

  #[test]
  fn yuv420p10_try_new_checked_rejects_u_plane_sample() {
    // Offending sample in the U plane — error must name U, not Y or V.
    let y = std::vec![0u16; 16 * 8];
    let mut u = std::vec![512u16; 8 * 4];
    u[2 * 8 + 3] = 1024; // just above max
    let v = std::vec![512u16; 8 * 4];
    let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(
      e,
      Yuv420pFrame16Error::SampleOutOfRange {
        plane: Yuv420pFrame16Plane::U,
        value: 1024,
        max_valid: 1023,
        ..
      }
    ));
  }

  #[test]
  fn yuv420p10_try_new_checked_rejects_v_plane_sample() {
    let y = std::vec![0u16; 16 * 8];
    let u = std::vec![512u16; 8 * 4];
    let mut v = std::vec![512u16; 8 * 4];
    v[1 * 8 + 7] = 0xFFFF; // all bits set
    let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(
      e,
      Yuv420pFrame16Error::SampleOutOfRange {
        plane: Yuv420pFrame16Plane::V,
        max_valid: 1023,
        ..
      }
    ));
  }

  #[test]
  fn yuv420p10_try_new_checked_accepts_exact_max_sample() {
    // Boundary: sample value == (1 << BITS) - 1 is valid.
    let mut y = std::vec![0u16; 16 * 8];
    y[0] = 1023;
    let u = std::vec![512u16; 8 * 4];
    let v = std::vec![512u16; 8 * 4];
    Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("1023 is in range");
  }

  #[test]
  fn yuv420p10_try_new_checked_reports_geometry_errors_first() {
    // If geometry is invalid, we never get to the sample scan — the
    // same errors as `try_new` surface first. Prevents the checked
    // path from doing unnecessary O(N) work on inputs that would
    // fail for a simpler reason.
    let y = std::vec![0u16; 10]; // Too small.
    let u = std::vec![512u16; 8 * 4];
    let v = std::vec![512u16; 8 * 4];
    let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
    assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
  }

  // ---- P010Frame ---------------------------------------------------------
  //
  // Semi‑planar 10‑bit. Plane shape mirrors Nv12Frame (Y + interleaved
  // UV) but sample width is `u16` with the 10 active bits in the
  // **high** 10 of each element (`value << 6`). Strides are in
  // samples, not bytes.

  fn p010_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
    // 16×8 frame — UV plane carries 16 u16 × 4 chroma rows = 64 u16.
    // P010 white Y = 1023 << 6 = 0xFFC0; neutral UV = 512 << 6 = 0x8000.
    (std::vec![0xFFC0u16; 16 * 8], std::vec![0x8000u16; 16 * 4])
  }

  #[test]
  fn p010_try_new_accepts_valid_tight() {
    let (y, uv) = p010_planes();
    let f = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
    assert_eq!(f.width(), 16);
    assert_eq!(f.height(), 8);
    assert_eq!(f.uv_stride(), 16);
  }

  #[test]
  fn p010_try_new_accepts_odd_height() {
    // 640×481 — same concrete odd‑height case covered by NV12 / NV21.
    let y = std::vec![0u16; 640 * 481];
    let uv = std::vec![0x8000u16; 640 * 241];
    let f = P010Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
    assert_eq!(f.height(), 481);
  }

  #[test]
  fn p010_try_new_rejects_odd_width() {
    let (y, uv) = p010_planes();
    let e = P010Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::OddWidth { width: 15 }));
  }

  #[test]
  fn p010_try_new_rejects_zero_dim() {
    let (y, uv) = p010_planes();
    let e = P010Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::ZeroDimension { .. }));
  }

  #[test]
  fn p010_try_new_rejects_y_stride_under_width() {
    let (y, uv) = p010_planes();
    let e = P010Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::YStrideTooSmall { .. }));
  }

  #[test]
  fn p010_try_new_rejects_uv_stride_under_width() {
    let (y, uv) = p010_planes();
    let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
    assert!(matches!(e, P010FrameError::UvStrideTooSmall { .. }));
  }

  #[test]
  fn p010_try_new_rejects_short_y_plane() {
    let y = std::vec![0u16; 10];
    let uv = std::vec![0x8000u16; 16 * 4];
    let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::YPlaneTooShort { .. }));
  }

  #[test]
  fn p010_try_new_rejects_short_uv_plane() {
    let y = std::vec![0u16; 16 * 8];
    let uv = std::vec![0x8000u16; 8];
    let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::UvPlaneTooShort { .. }));
  }

  #[test]
  #[should_panic(expected = "invalid P010Frame")]
  fn p010_new_panics_on_invalid() {
    let y = std::vec![0u16; 10];
    let uv = std::vec![0x8000u16; 16 * 4];
    let _ = P010Frame::new(&y, &uv, 16, 8, 16, 16);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn p010_try_new_rejects_geometry_overflow() {
    let big: u32 = 0x1_0000;
    let y: [u16; 0] = [];
    let uv: [u16; 0] = [];
    let e = P010Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
    assert!(matches!(e, P010FrameError::GeometryOverflow { .. }));
  }

  #[test]
  fn p010_try_new_checked_accepts_shifted_samples() {
    // Valid P010 samples: low 6 bits zero.
    let (y, uv) = p010_planes();
    P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).expect("shifted samples valid");
  }

  #[test]
  fn p010_try_new_checked_rejects_y_low_bits_set() {
    // A Y sample with low 6 bits set — characteristic of yuv420p10le
    // packing (value in low 10 bits) accidentally handed to the P010
    // constructor. `try_new_checked` catches this; plain `try_new`
    // would let the kernel mask it down and produce wrong colors.
    let mut y = std::vec![0xFFC0u16; 16 * 8];
    y[3 * 16 + 5] = 0x03FF; // 10-bit value in low bits — wrong packing
    let uv = std::vec![0x8000u16; 16 * 4];
    let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
    match e {
      P010FrameError::SampleLowBitsSet { plane, value, .. } => {
        assert_eq!(plane, P010FramePlane::Y);
        assert_eq!(value, 0x03FF);
      }
      other => panic!("expected SampleLowBitsSet, got {other:?}"),
    }
  }

  #[test]
  fn p010_try_new_checked_rejects_uv_plane_sample() {
    let y = std::vec![0xFFC0u16; 16 * 8];
    let mut uv = std::vec![0x8000u16; 16 * 4];
    uv[2 * 16 + 3] = 0x0001; // low bit set
    let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(
      e,
      P010FrameError::SampleLowBitsSet {
        plane: P010FramePlane::Uv,
        value: 0x0001,
        ..
      }
    ));
  }

  #[test]
  fn p010_try_new_checked_reports_geometry_errors_first() {
    let y = std::vec![0u16; 10]; // Too small.
    let uv = std::vec![0x8000u16; 16 * 4];
    let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
    assert!(matches!(e, P010FrameError::YPlaneTooShort { .. }));
  }

  /// Regression documenting a **known limitation** of
  /// [`P010Frame::try_new_checked`]: the low‑6‑bits‑zero check is a
  /// packing sanity check, not a provenance validator. A
  /// `yuv420p10le` buffer whose samples all happen to be multiples
  /// of 64 — e.g. `Y = 64` (limited‑range black, `0x0040`) and
  /// `UV = 512` (neutral chroma, `0x0200`) — passes the check
  /// silently, even though the layout is wrong and downstream P010
  /// kernels will produce incorrect output.
  ///
  /// The test asserts the check accepts these values so the limit
  /// is visible in the test log; any future attempt to tighten the
  /// constructor into a real provenance validator will need to
  /// update or replace this test.
  #[test]
  fn p010_try_new_checked_accepts_ambiguous_yuv420p10le_samples() {
    // `yuv420p10le`-style samples, all multiples of 64: low 6 bits
    // are zero, so they pass the P010 sanity check even though this
    // is wrong data for a P010 frame.
    let y = std::vec![0x0040u16; 16 * 8]; // limited-range black in 10-bit low-packed
    let uv = std::vec![0x0200u16; 16 * 4]; // neutral chroma in 10-bit low-packed
    let f = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
      .expect("known limitation: low-6-bits-zero check cannot tell yuv420p10le from P010");
    assert_eq!(f.width(), 16);
    // Downstream decoding of this frame would produce wrong colors
    // (every `>> 6` extracts 1 from Y=0x0040 and 8 from UV=0x0200,
    // which P010 kernels then bias/scale as if those were the 10-bit
    // source values). That's accepted behavior — the type system,
    // not `try_new_checked`, is what keeps yuv420p10le out of P010.
  }
}
