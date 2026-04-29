use derive_more::{Display, IsVariant};
use thiserror::Error;

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
