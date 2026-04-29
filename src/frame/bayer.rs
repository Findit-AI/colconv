use derive_more::IsVariant;
use thiserror::Error;

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
