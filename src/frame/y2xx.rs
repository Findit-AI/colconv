//! Packed YUV 4:2:2 high-bit-depth source family `Y2xx` — common
//! frame template + format aliases.
//!
//! Each row contains `width × 4` u8 bytes laid out as YUYV-shaped
//! u16 quadruples (`Y₀, U, Y₁, V`). Active bits are MSB-aligned;
//! low `(16 - BITS)` bits are zero.
//!
//! | Format | BITS | Active bit width | Low bits |
//! |--------|------|------------------|----------|
//! | Y210   | 10   | bits[15:6]       | bits[5:0] = 0 |
//! | Y212   | 12   | bits[15:4]       | bits[3:0] = 0 |
//! | Y216   | 16   | bits[15:0]       | n/a (full range) |
//!
//! Width must be even (4:2:2 chroma subsampling).
//!
//! Used by Ship 11b (Y210), Ship 11c (Y212 — wiring-only), and
//! Ship 11d (Y216 — separate kernel family with i64 chroma path).

use derive_more::IsVariant;
use thiserror::Error;

/// Validated wrapper around a packed YUV 4:2:2 high-bit-depth plane
/// for the `Y210` / `Y212` / `Y216` family.
///
/// `BITS` selects the active sample width: 10, 12, or 16. Construct
/// via [`Self::try_new`] (fallible) or [`Self::new`] (panics on
/// invalid input). For `BITS ∈ {10, 12}` the optional
/// [`Self::try_new_checked`] additionally verifies that every
/// sample's low `(16 - BITS)` bits are zero (matches the
/// `P010::try_new_checked` pattern).
#[derive(Debug, Clone, Copy)]
pub struct Y2xxFrame<'a, const BITS: u32> {
  packed: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

/// Y210 alias — 10-bit MSB-aligned packed YUV 4:2:2.
pub type Y210Frame<'a> = Y2xxFrame<'a, 10>;

/// Errors returned by [`Y2xxFrame::try_new`] and
/// [`Y2xxFrame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Y2xxFrameError {
  /// `BITS ∉ {10, 12, 16}`.
  #[error("Y2xxFrame: unsupported BITS {bits}; must be 10, 12, or 16")]
  UnsupportedBits {
    /// `BITS` const-generic value.
    bits: u32,
  },
  /// `width == 0` or `height == 0`.
  #[error("Y2xxFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width % 2 != 0`. 4:2:2 subsampling requires even width.
  #[error("Y2xxFrame: width {width} is odd; 4:2:2 chroma subsampling requires even width")]
  OddWidth {
    /// Configured width.
    width: u32,
  },
  /// `stride < width * 2` (u16 elements). Each row needs at least
  /// `width × 2` u16 elements (= `width × 4` bytes) to hold all
  /// pixels.
  #[error("Y2xxFrame: stride {stride} u16 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in u16 elements (`width × 2`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry (in u16 elements).
  #[error("Y2xxFrame: plane too short: expected >= {expected} u16 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u16 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u16 elements.
    actual: usize,
  },
  /// `stride * height` overflows `u32`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("Y2xxFrame: stride × height overflows u32 (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
  /// `width × 2` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("Y2xxFrame: width {width} × 2 overflows u32 (per-row u16 element count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `try_new_checked` only: a sample's low `(16 - BITS)` bits are
  /// non-zero. Diagnoses callers feeding non-MSB-aligned data
  /// (e.g. low-bit-packed yuv422p10le mistakenly handed to a Y210
  /// path). Y216 doesn't emit this since all 16 bits are active.
  #[error("Y2xxFrame: sample with non-zero low bits found; expected MSB-aligned data")]
  SampleLowBitsSet,
}

impl<'a, const BITS: u32> Y2xxFrame<'a, BITS> {
  /// Validates and constructs a [`Y2xxFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Y2xxFrameError> {
    if BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(Y2xxFrameError::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Y2xxFrameError::ZeroDimension { width, height });
    }
    if !width.is_multiple_of(2) {
      return Err(Y2xxFrameError::OddWidth { width });
    }
    let min_stride = match width.checked_mul(2) {
      Some(n) => n,
      None => return Err(Y2xxFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Y2xxFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(Y2xxFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(Y2xxFrameError::PlaneTooShort {
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
  /// low `(16 - BITS)` bits are non-zero. Only meaningful for
  /// `BITS ∈ {10, 12}`; for `BITS = 16` this delegates to
  /// [`Self::try_new`] (no low bits to check).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Y2xxFrameError> {
    let frame = Self::try_new(packed, width, height, stride)?;
    if BITS < 16 {
      let low_mask: u16 = (1u16 << (16 - BITS)) - 1;
      let row_elems = (width * 2) as usize;
      let h = height as usize;
      let stride_us = stride as usize;
      for row in 0..h {
        let start = row * stride_us;
        for &sample in &packed[start..start + row_elems] {
          if sample & low_mask != 0 {
            return Err(Y2xxFrameError::SampleLowBitsSet);
          }
        }
      }
    }
    Ok(frame)
  }

  /// Panicking convenience over [`Self::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(_) => panic!("invalid Y2xxFrame: validation failed"),
    }
  }

  /// Packed plane (u16 elements; each pair of samples = 1 pixel).
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
  /// Stride in u16 elements (NOT bytes — this is the number of
  /// u16 slots per row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
