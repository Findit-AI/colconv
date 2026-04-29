use derive_more::{Display, IsVariant};
use thiserror::Error;

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
