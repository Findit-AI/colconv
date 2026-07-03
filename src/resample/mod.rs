//! Resampling strategies for [`MixedSinker`](crate::sinker::MixedSinker).
//!
//! A [`Resampler`] is injected at sinker construction and decides the
//! sinker's **output geometry** once, before any output buffer
//! attaches: [`Resampler::plan`] returns `Ok(None)` for the identity
//! (output geometry == source geometry — the sinker takes the direct
//! conversion path) or `Ok(Some(plan))` carrying the output geometry
//! that every output buffer is then validated against. The walker-side
//! contract is unchanged either way:
//! [`PixelSink::begin_frame`](crate::PixelSink::begin_frame) keeps
//! validating frames against the **source** geometry.
//!
//! The trait is sealed — resampling strategies ship with this crate.
//! [`NoopResampler`], the default `R` of
//! [`MixedSinker`](crate::sinker::MixedSinker), is the identity
//! strategy; [`AreaResampler`] plans area (box-coverage) downscales.
//! The streaming engine that executes non-identity plans lands with
//! the per-format route dispatch.

// The streaming engine compiles for every source family whose gate is
// widened (so its formats can route in a later PR) but has no consumer
// until that family is wired up. In a family-only build that has not
// yet routed — anything outside `yuv-planar` / `rgb` — the engine is
// legitimately present-but-unused; allow it rather than gate each item
// on the moving target of "which families route today".
#![cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]

use std::vec::Vec;

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

mod filter;
use filter::FilterAxis;
pub use filter::{
  BlackmanSinc, CatmullRom, CubicBSpline, FilterKernel, Gaussian, InvalidFilterSupport, Lanczos3,
  Lanczos4, Mitchell, OpenCvCubic, Spline16, Spline36, Spline64, SwscaleBicubic, Triangle,
};
// Re-exported for the sinker's `*_filter_stream` fields and tails, the
// filter twin of the `AreaStream` / `AreaSample` pair. Compiled wherever
// the `stream` submodule is (the 14-feature engine cascade); only the
// routed families (`rgb` / `gray` in this stage) actually reference it.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(unused_imports))]
pub(crate) use filter::{FilterSample, FilterStream};

mod strategy;
/// Crate-internal re-export of the verified ITU-R BT.2020 OETF / inverse-OETF
/// math (the SDR camera gamma). The constant-luminance non-affine decode
/// ([`crate::row::scalar::cl`], H.273 `MatrixCoefficients = 13`, #303) reaches
/// its transfer stage through here. Gated identically to [`pq_hlg`] — its sole
/// consumer (the `cl` module, `all(yuv-planar, any(std, alloc))`).
#[cfg(feature = "yuv-planar")]
pub(crate) use strategy::transfer::bt2020_oetf;
/// Crate-internal re-export of the verified BT.2100 PQ / HLG inverse-EOTF /
/// OETF math (the #313 foundation, parked in [`strategy::transfer`] until a
/// consumer landed). The ICtCp non-affine decode
/// ([`crate::row::scalar::ictcp`], H.273 `MatrixCoefficients = 14`, #303)
/// reaches the transfer stage through here. Gated on `yuv-planar` — its sole
/// consumer (the `ictcp` module, itself `all(yuv-planar, any(std, alloc))`);
/// `resample` already carries the `any(std, alloc)` gate.
#[cfg(feature = "yuv-planar")]
pub(crate) use strategy::transfer::pq_hlg;
pub use strategy::{AveragingDomain, FilterSpec, LinearMode, ResampleStrategy, TransferFunction};
// Phase-0-internal: the splice-stage selector consumed by the per-format
// route dispatch. `InsertionPoint` / its context stay crate-private until
// later phases widen the splice surface; only `yuv-planar` (the routed
// `Yuv420p`) references them today.
#[cfg_attr(not(feature = "yuv-planar"), allow(unused_imports))]
pub(crate) use strategy::{InsertionContext, InsertionPoint, select_insertion_point};

/// Decides a [`MixedSinker`](crate::sinker::MixedSinker)'s output
/// geometry from its source geometry, once, at construction.
///
/// Sealed — only strategies shipped by this crate implement it.
pub trait Resampler: sealed::Sealed {
  /// Builds the resampling plan for a `src_w x src_h` source frame.
  ///
  /// `Ok(None)` means the resampling is the identity: output geometry
  /// equals source geometry and the sinker takes the direct conversion
  /// path with no resampling state at all.
  ///
  /// # Errors
  ///
  /// Strategy-specific validation of the requested output geometry —
  /// see [`ResampleError`] for the variants.
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError>;
}

mod sealed {
  pub trait Sealed {}
}

/// Identity strategy and the default `R` of
/// [`MixedSinker`](crate::sinker::MixedSinker): [`Resampler::plan`]
/// always returns `Ok(None)`, so output geometry equals source
/// geometry and the sinker behaves exactly like a non-resampling sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopResampler;

impl sealed::Sealed for NoopResampler {}

impl Resampler for NoopResampler {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn plan(&self, _src_w: usize, _src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    Ok(None)
  }
}

/// Area (box-coverage) downscale strategy — the `cv2.INTER_AREA`
/// convention that analysis pipelines are calibrated against. Plans
/// exact integer coverage spans on both axes, fractional ratios
/// included (1920 -> 336 is a x40/7 scale). Requesting the source
/// geometry plans the identity (`Ok(None)`); upscaling on either axis
/// is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AreaResampler {
  out_w: usize,
  out_h: usize,
}

impl AreaResampler {
  /// Strategy producing an `out_w x out_h` output frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn to(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

impl sealed::Sealed for AreaResampler {}

impl Resampler for AreaResampler {
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    if self.out_w == 0 || self.out_h == 0 {
      return Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(self.out_w, self.out_h),
      ));
    }
    if self.out_w > src_w || self.out_h > src_h {
      return Err(ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(
        src_w, src_h, self.out_w, self.out_h,
      )));
    }
    if self.out_w == src_w && self.out_h == src_h {
      return Ok(None);
    }
    ResamplePlan::area(src_w, src_h, self.out_w, self.out_h).map(Some)
  }
}

/// Separable **windowed-filter** resampling strategy — the Pillow (PIL)
/// `Image.resize` convention, reproduced within +-1 LSB. Plans signed
/// reconstruction-filter coefficients on both axes under the kernel `K`:
/// [`Triangle`] (PIL `BILINEAR`), [`CatmullRom`] (PIL `BICUBIC`), or
/// [`Lanczos3`] (PIL `LANCZOS`), or a custom [`FilterKernel`].
///
/// Downscales, upscales, and mixed per-axis ratios are all supported (PIL
/// widens the support when reducing and keeps it native when enlarging).
/// Requesting the source geometry plans the identity (`Ok(None)`). The
/// kernel is plain data — construct via [`Self::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilteredResampler<K: FilterKernel> {
  out_w: usize,
  out_h: usize,
  kernel: K,
}

impl<K: FilterKernel> FilteredResampler<K> {
  /// Strategy producing an `out_w x out_h` output frame under `kernel`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(out_w: usize, out_h: usize, kernel: K) -> Self {
    Self {
      out_w,
      out_h,
      kernel,
    }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }

  /// The reconstruction-filter kernel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kernel(&self) -> &K {
    &self.kernel
  }
}

impl<K: FilterKernel> sealed::Sealed for FilteredResampler<K> {}

impl<K: FilterKernel> Resampler for FilteredResampler<K> {
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    if self.out_w == 0 || self.out_h == 0 {
      return Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(self.out_w, self.out_h),
      ));
    }
    if self.out_w == src_w && self.out_h == src_h {
      return Ok(None);
    }
    // Upscale, downscale, or a mixed per-axis ratio: the filter engine plans
    // each axis independently (PIL's `precompute_coeffs` widens the support on
    // a downscaling axis and leaves it native when enlarging), so no axis-
    // direction guard is needed here.
    ResamplePlan::filter(src_w, src_h, self.out_w, self.out_h, &self.kernel).map(Some)
  }
}

/// FFmpeg `swscale` **`SWS_BICUBLIN`** — the concrete per-plane reconstruction
/// filter that resamples the **luma** plane with a cubic kernel
/// ([`SwscaleBicubic`], swscale's default `a = -0.6` bicubic) and the
/// **chroma** planes with a linear kernel ([`Triangle`], the bilinear tent).
///
/// This is fundamentally different from a single-kernel [`FilteredResampler`]:
/// that path converts the (chroma-upsampled) YUV to RGB and applies one kernel
/// to the RGB, so it cannot express a luma-vs-chroma kernel split. BICUBLIN
/// filters the three planes **separately in YUV-plane space** — Y at luma
/// resolution with the cubic, U / V at chroma resolution with the linear — and
/// converts the filtered planes to RGB at the output grid. It is the filter
/// twin of the native area tier (which *bins* the planes, then converts).
///
/// Wired for [`Yuv420p`](crate::source::Yuv420p) (4:2:0): the chroma planes are
/// half-width, ceil-half-height, so the chroma windows are planned over
/// `(src_w / 2) x ceil(src_h / 2)`. Upscale, downscale, and mixed ratios are
/// all supported (each plane's axes plan independently). There is **no**
/// same-size identity short-circuit (unlike the single-kernel strategies):
/// even at an unchanged luma size the half-resolution chroma plane must be
/// upsampled with the linear kernel — so a BICUBLIN plan is always built and
/// always carries all four per-plane windows. Constructed with no parameters
/// — the two kernels are fixed by the swscale BICUBLIN definition.
#[cfg(feature = "yuv-planar")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bicublin {
  out_w: usize,
  out_h: usize,
}

#[cfg(feature = "yuv-planar")]
impl Bicublin {
  /// Strategy producing an `out_w x out_h` output frame under the swscale
  /// BICUBLIN per-plane kernels (cubic luma, linear chroma).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn to(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

#[cfg(feature = "yuv-planar")]
impl sealed::Sealed for Bicublin {}

#[cfg(feature = "yuv-planar")]
impl Resampler for Bicublin {
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    if self.out_w == 0 || self.out_h == 0 {
      return Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(self.out_w, self.out_h),
      ));
    }
    // No same-size identity short-circuit (unlike the single-kernel
    // strategies): even when the LUMA size is unchanged (`out == src`), the
    // 4:2:0 chroma plane is half-resolution and so must be UPSAMPLED to the
    // output grid with the linear kernel — a real convolution, not a
    // pass-through. The luma plane is likewise a same-size cubic convolution
    // (not the identity for a non-box kernel). So a BICUBLIN plan is built and
    // applied on all four axes regardless of `in == out`; `FilterAxis::build`
    // evaluates a same-size axis as the convolution it is.
    //
    // 4:2:0 chroma grid: half width, ceil-half height. The chroma plane is
    // filtered as its own `chroma_w x chroma_h -> out` image (swscale's
    // per-plane behaviour), so the windows are planned at that resolution.
    let chroma_w = src_w / 2;
    let chroma_h = src_h.div_ceil(2);
    ResamplePlan::bicublin(
      src_w,
      src_h,
      chroma_w,
      chroma_h,
      self.out_w,
      self.out_h,
      &SwscaleBicubic,
      &Triangle,
      0.0,
      0.0,
    )
    .map(Some)
  }
}

/// Zero-filled buffer via fallible reservation: `resize` after an
/// exact reserve cannot reallocate, so refusal is the only failure
/// and it surfaces as the error instead of aborting.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
pub(crate) fn try_zeroed<T: Clone + Default>(
  n: usize,
) -> Result<Vec<T>, std::collections::TryReserveError> {
  let mut buf = Vec::new();
  buf.try_reserve_exact(n)?;
  buf.resize(n, T::default());
  Ok(buf)
}

/// Greatest common divisor (Euclid); both inputs are nonzero axis
/// dimensions by the time the planner runs.
fn gcd(mut a: usize, mut b: usize) -> usize {
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a
}

/// Why one axis of span planning failed; [`ResamplePlan::area`] maps
/// these onto [`ResampleError`] with the full two-axis geometry.
#[derive(Debug)]
enum AxisError {
  /// A grid product or arena length is unrepresentable.
  Overflow,
  /// An arena reservation was refused by the allocator.
  Alloc,
}

/// Per-axis area-coverage spans of a [`ResamplePlan`]: for each output
/// index, the first contributing source cell plus the integer overlap
/// weight of every contributing cell.
///
/// Geometry lives on the axis's `x out` integer grid — output pixel
/// `j` covers `[j * src, (j + 1) * src)` and source cell `i` covers
/// `[i * out, (i + 1) * out)` — so weights are exact for fractional
/// ratios and every span sums to `src`, the normalization denominator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisSpans {
  /// First contributing source cell per output index.
  starts: Vec<usize>,
  /// Prefix offsets into `weights`; `out_len() + 1` entries.
  offsets: Vec<usize>,
  /// Concatenated per-span overlap weights.
  weights: Vec<usize>,
}

impl AxisSpans {
  /// Exact weight-arena length for a `src -> out` area axis:
  /// `src + out - gcd(src, out)` — every source cell contributes to
  /// exactly one span, plus one shared straddle cell for each of the
  /// `out - 1` interior output boundaries that does NOT fall on a
  /// cell boundary (the aligned ones number `gcd - 1`).
  fn area_taps(src: usize, out: usize) -> Option<usize> {
    // `gcd <= min(src, out)`, so subtracting first cannot underflow
    // and keeps a representable count (e.g. usize::MAX source, one
    // output) from being misreported as overflow.
    (src - gcd(src, out)).checked_add(out)
  }

  /// Empty placeholder spans — carried by a [`SpanKind::Filter`]
  /// [`ResamplePlan`] in the area-span fields the filter path never reads.
  /// Allocation-free (`Vec::new`).
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  fn empty() -> Self {
    Self {
      starts: Vec::new(),
      offsets: Vec::new(),
      weights: Vec::new(),
    }
  }

  /// Builds the exact area spans for one `src -> out` axis.
  ///
  /// Coordinate arithmetic runs in `u64` so 32-bit targets plan any
  /// geometry whose buffers fit — `usize`-sized `src` and `out` can
  /// only overflow the `src * out` grid product on 64-bit hosts. The
  /// up-front check bounds every loop term, so the loop body stays in
  /// plain arithmetic; per-cell weights are at most `out` and starts
  /// at most `src`, so the casts back to `usize` are lossless.
  ///
  /// Arena sizes are precomputed exactly ([`Self::area_taps`]) and
  /// reserved fallibly, so a hostile source dimension from untrusted
  /// metadata surfaces [`AxisError::Alloc`] instead of aborting
  /// inside infallible allocation; the pushes below never reallocate.
  fn area(src: usize, out: usize) -> Result<Self, AxisError> {
    let src64 = src as u64;
    let out64 = out as u64;
    src64.checked_mul(out64).ok_or(AxisError::Overflow)?;
    let taps = Self::area_taps(src, out).ok_or(AxisError::Overflow)?;
    let offsets_len = out.checked_add(1).ok_or(AxisError::Overflow)?;
    let mut starts = Vec::new();
    starts
      .try_reserve_exact(out)
      .map_err(|_| AxisError::Alloc)?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| AxisError::Alloc)?;
    let mut weights = Vec::new();
    weights
      .try_reserve_exact(taps)
      .map_err(|_| AxisError::Alloc)?;
    offsets.push(0);
    for j in 0..out64 {
      let lo = j * src64;
      let hi = lo + src64;
      let start = lo / out64;
      starts.push(start as usize);
      for i in start..hi.div_ceil(out64) {
        let cell_lo = i * out64;
        let cell_hi = cell_lo + out64;
        weights.push((cell_hi.min(hi) - cell_lo.max(lo)) as usize);
      }
      offsets.push(weights.len());
    }
    debug_assert_eq!(weights.len(), taps);
    Ok(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// Builds spans for a vertically subsampled axis at ratio `factor`
  /// (`2` for 4:2:0 / 4:4:0, `4` for 4:1:0): cell `c` is the group of
  /// full-grid rows `[factor*c, factor*c + factor)` clipped to
  /// `src_full`, so a partial trailing group of `1..factor` rows forms a
  /// short tail cell weighted by its true luma coverage. Weights live on
  /// the `x out` grid against the FULL-resolution axis — every span sums
  /// to `src_full`, which is therefore the normalization denominator (for
  /// `src_full` an exact multiple of `factor` this is the uniform
  /// chroma-grid weighting with numerator and denominator scaled by
  /// `factor`, which round-half-up preserves exactly).
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  fn area_subsampled(src_full: usize, out: usize, factor: usize) -> Result<Self, AxisError> {
    let src64 = src_full as u64;
    let out64 = out as u64;
    let f = factor as u64;
    src64.checked_mul(out64).ok_or(AxisError::Overflow)?;
    let cells = src_full.div_ceil(factor);
    // Upper bound; the exact reservation below never reallocates.
    let taps = Self::area_taps(src_full, out).ok_or(AxisError::Overflow)?;
    let offsets_len = out.checked_add(1).ok_or(AxisError::Overflow)?;
    let mut starts = Vec::new();
    starts
      .try_reserve_exact(out)
      .map_err(|_| AxisError::Alloc)?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| AxisError::Alloc)?;
    let mut weights = Vec::new();
    weights
      .try_reserve_exact(taps)
      .map_err(|_| AxisError::Alloc)?;
    offsets.push(0);
    for j in 0..out64 {
      let lo = j * src64;
      let hi = lo + src64;
      // First full-grid row touched, mapped to its subsample cell.
      let start = ((lo / out64) / f) as usize;
      starts.push(start);
      let mut c = start as u64;
      loop {
        let cell_lo = (f * c) * out64;
        let cell_hi = ((f * c + f).min(src64)) * out64;
        if cell_lo >= hi {
          break;
        }
        let w = cell_hi.min(hi) - cell_lo.max(lo);
        if w == 0 {
          break;
        }
        weights.push(w as usize);
        if cell_hi >= hi || c as usize + 1 >= cells {
          break;
        }
        c += 1;
      }
      offsets.push(weights.len());
    }
    Ok(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// 4:2:0 / 4:4:0 vertical pairing — [`Self::area_subsampled`] at factor 2.
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  fn area_halved(src_full: usize, out: usize) -> Result<Self, AxisError> {
    Self::area_subsampled(src_full, out, 2)
  }

  /// Horizontal chroma spans for the RFC #238 **centered** 4:2:x siting — the
  /// folded triangle⊗box weights (design §3). A centered chroma sample sits at
  /// `+0.5` luma between the two columns it covers, so reconstructing full
  /// width is the #302 `1/4`–`3/4` MPEG-1 / JPEG triangle
  /// ([`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)):
  /// even full column `2j → (c[j-1] + 3·c[j]) / 4`, odd `2j+1 → (3·c[j] +
  /// c[j+1]) / 4`, with boundary replication (`c[-1] → c[0]`,
  /// `c[chroma_w] → c[chroma_w-1]`). This folds that reconstruction INTO the
  /// box bin so ONE phased weighted area pass on the SUBSAMPLED chroma grid
  /// reproduces "reconstruct to full width, then box-average" with a SINGLE
  /// rounding: `W[o,j] = Σ_X w_bin[o,X]·r[X,j]`, where `w_bin` are the
  /// full-grid box overlaps ([`Self::area`] over `2·chroma_w → out`) and `r`
  /// is the triangle. The triangle is scaled ×4 to stay integral (`r ∈ {1, 3,
  /// 4}`), so each output span sums to `4·(2·chroma_w)` — the H half of the
  /// caller's `src_w·src_h` normalization denominator.
  ///
  /// At phase 0 the triangle degenerates to nearest-replicate and this equals
  /// [`Self::area`] over `chroma_w → out` (scaled by the same factor); the
  /// caller keeps the unscaled [`Self::area`] for the co-sited path, so this
  /// builder is invoked only for the centered group. `chroma_w ≥ 1` (a real
  /// 4:2:x frame), so `chroma_w - 1` below never underflows.
  #[cfg(feature = "yuv-planar")]
  fn area_chroma_phased_h_centered(chroma_w: usize, out: usize) -> Result<Self, AxisError> {
    let full_w = chroma_w.checked_mul(2).ok_or(AxisError::Overflow)?;
    // Each folded span sums to `4·full_w`; bound it so the `u64`-accumulated
    // weights cast back to `usize` losslessly below.
    full_w.checked_mul(4).ok_or(AxisError::Overflow)?;
    let box_spans = Self::area(full_w, out)?;
    // First / last chroma cell the centered triangle scatters full cell `x` to.
    // An even `x = 2j` spreads to `{j-1, j}` (`{0}` clamped at the left edge);
    // an odd `x = 2j+1` spreads to `{j, j+1}` (`{chroma_w-1}` clamped at the
    // right edge). Consecutive full cells share a chroma cell, so a box span's
    // chroma reach is contiguous.
    let first_chroma = |x: usize| -> usize {
      if x.is_multiple_of(2) {
        (x / 2).saturating_sub(1)
      } else {
        (x - 1) / 2
      }
    };
    let last_chroma = |x: usize| -> usize {
      if x.is_multiple_of(2) {
        x / 2
      } else {
        ((x - 1) / 2 + 1).min(chroma_w - 1)
      }
    };
    let offsets_len = out.checked_add(1).ok_or(AxisError::Overflow)?;
    let mut starts = Vec::new();
    starts
      .try_reserve_exact(out)
      .map_err(|_| AxisError::Alloc)?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| AxisError::Alloc)?;
    // Pass 1: each output's chroma span `[j_first, j_last]`, the total tap
    // count, and the widest span (the reused scatter accumulator's size).
    let mut total = 0usize;
    let mut max_span = 0usize;
    for o in 0..out {
      let (xstart, wbins) = box_spans.span(o);
      let jf = first_chroma(xstart);
      let jl = last_chroma(xstart + wbins.len() - 1);
      starts.push(jf);
      let span = jl - jf + 1;
      max_span = max_span.max(span);
      total = total.checked_add(span).ok_or(AxisError::Overflow)?;
    }
    let mut weights = Vec::new();
    weights
      .try_reserve_exact(total)
      .map_err(|_| AxisError::Alloc)?;
    let mut local: Vec<u64> = Vec::new();
    local
      .try_reserve_exact(max_span)
      .map_err(|_| AxisError::Alloc)?;
    offsets.push(0);
    // Pass 2: scatter each box overlap through the ×4 triangle into the local
    // chroma-cell accumulator, then flush the contiguous run.
    for (o, &jf) in starts.iter().enumerate() {
      let (xstart, wbins) = box_spans.span(o);
      let jl = last_chroma(xstart + wbins.len() - 1);
      local.clear();
      local.resize(jl - jf + 1, 0);
      for (k, &wb) in wbins.iter().enumerate() {
        let x = xstart + k;
        let wb = wb as u64;
        if x.is_multiple_of(2) {
          let j = x / 2;
          if j == 0 {
            local[0] += 4 * wb; // left neighbor clamps to c[0]
          } else {
            local[j - 1 - jf] += wb; // 1/4 → c[j-1]
            local[j - jf] += 3 * wb; // 3/4 → c[j]
          }
        } else {
          let j = (x - 1) / 2;
          if j == chroma_w - 1 {
            local[j - jf] += 4 * wb; // right neighbor clamps to c[chroma_w-1]
          } else {
            local[j - jf] += 3 * wb; // 3/4 → c[j]
            local[j + 1 - jf] += wb; // 1/4 → c[j+1]
          }
        }
      }
      for &wv in &local {
        weights.push(wv as usize);
      }
      offsets.push(weights.len());
    }
    debug_assert_eq!(weights.len(), total);
    Ok(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// Vertical chroma spans for the RFC #238 **Bottom** 4:2:0 siting — the
  /// folded triangle⊗box weights on the V axis (design §3), the `v = 1`
  /// analog of [`Self::area_chroma_phased_h_centered`]. A Bottom chroma
  /// sample sits on the ODD luma row it covers, so reconstructing full
  /// height is the co-sited-on-odd triangle: odd luma row `2i + 1 → c[i]`
  /// (the sample lies exactly on it), even luma row `2i → (c[i-1] + c[i]) /
  /// 2` (midway between the sample above and the one it sits on), with
  /// top-edge replication (`c[-1] → c[0]`). Unlike the centered H triangle
  /// this reaches only BACKWARD — a Bottom sample never spreads to `i + 1` —
  /// so the sole clamp is the top edge. This folds that reconstruction INTO
  /// the box bin so ONE phased weighted area pass on the SUBSAMPLED chroma
  /// grid reproduces "reconstruct to full height, then box-average" with a
  /// SINGLE rounding: `W[o,i] = Σ_R w_bin[o,R]·t[R,i]`, where `w_bin` are the
  /// full-grid box overlaps ([`Self::area`] over `luma_h → out`) and `t` is
  /// the triangle. The triangle is scaled ×2 to stay integral (`t ∈ {1,
  /// 2}`), so each output span sums to `2·luma_h` — the V half of the
  /// caller's `src_w·src_h` normalization denominator.
  ///
  /// `luma_h` may be ODD (4:2:0 permits an odd frame height, `chroma_h =
  /// ⌈luma_h/2⌉`): [`Self::area`] already clamps the partial trailing box
  /// cell, and the last luma row is then even, folding to `{chroma_h - 2,
  /// chroma_h - 1}` — in range, so no tail special-case is needed beyond the
  /// top clamp. `chroma_h ≥ 1`, so the `i == 0` branch absorbs the only
  /// possible top-edge underflow and every other `i - 1` stays a live cell.
  #[cfg(feature = "yuv-planar")]
  fn area_chroma_phased_v(luma_h: usize, out: usize) -> Result<Self, AxisError> {
    // Each folded span sums to `2·luma_h`; bound it so the `u64`-accumulated
    // weights cast back to `usize` losslessly below.
    luma_h.checked_mul(2).ok_or(AxisError::Overflow)?;
    let box_spans = Self::area(luma_h, out)?;
    // First / last chroma cell a full luma row folds to. Even `r = 2i`
    // spreads to `{i-1, i}` (`{0}` clamped at the top edge); odd `r = 2i+1`
    // sits on `{i}`. Both end at `i = r/2`; the first cell steps back one
    // only for an even row above the top. Consecutive luma rows share a
    // chroma cell, so a box span's chroma reach is contiguous.
    let first_chroma = |r: usize| -> usize {
      if r.is_multiple_of(2) {
        (r / 2).saturating_sub(1)
      } else {
        r / 2
      }
    };
    let last_chroma = |r: usize| -> usize { r / 2 };
    let offsets_len = out.checked_add(1).ok_or(AxisError::Overflow)?;
    let mut starts = Vec::new();
    starts
      .try_reserve_exact(out)
      .map_err(|_| AxisError::Alloc)?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| AxisError::Alloc)?;
    // Pass 1: each output's chroma span `[i_first, i_last]`, the total tap
    // count, and the widest span (the reused scatter accumulator's size).
    let mut total = 0usize;
    let mut max_span = 0usize;
    for o in 0..out {
      let (rstart, wbins) = box_spans.span(o);
      let i_f = first_chroma(rstart);
      let i_l = last_chroma(rstart + wbins.len() - 1);
      starts.push(i_f);
      let span = i_l - i_f + 1;
      max_span = max_span.max(span);
      total = total.checked_add(span).ok_or(AxisError::Overflow)?;
    }
    let mut weights = Vec::new();
    weights
      .try_reserve_exact(total)
      .map_err(|_| AxisError::Alloc)?;
    let mut local: Vec<u64> = Vec::new();
    local
      .try_reserve_exact(max_span)
      .map_err(|_| AxisError::Alloc)?;
    offsets.push(0);
    // Pass 2: scatter each box overlap through the ×2 triangle into the local
    // chroma-cell accumulator, then flush the contiguous run.
    for (o, &i_f) in starts.iter().enumerate() {
      let (rstart, wbins) = box_spans.span(o);
      let i_l = last_chroma(rstart + wbins.len() - 1);
      local.clear();
      local.resize(i_l - i_f + 1, 0);
      for (k, &wb) in wbins.iter().enumerate() {
        let r = rstart + k;
        let wb = wb as u64;
        let i = r / 2;
        if r.is_multiple_of(2) {
          if i == 0 {
            local[0] += 2 * wb; // both taps clamp to c[0] at the top edge
          } else {
            local[i - 1 - i_f] += wb; // 1/2 → c[i-1]
            local[i - i_f] += wb; // 1/2 → c[i]
          }
        } else {
          local[i - i_f] += 2 * wb; // co-sited on c[i]
        }
      }
      for &wv in &local {
        weights.push(wv as usize);
      }
      offsets.push(weights.len());
    }
    debug_assert_eq!(weights.len(), total);
    Ok(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// Number of output samples on this axis.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn out_len(&self) -> usize {
    self.starts.len()
  }

  /// `(first source cell, overlap weights)` for output index `j`;
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  /// `j` must be below [`Self::out_len`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span(&self, j: usize) -> (usize, &[usize]) {
    (
      self.starts[j],
      &self.weights[self.offsets[j]..self.offsets[j + 1]],
    )
  }

  /// The maximum number of output rows whose spans are SIMULTANEOUSLY open
  /// (started, not yet finalized) at any source row — the depth of the
  /// streaming accumulator window [`AreaStream`] must hold for this axis.
  ///
  /// Two adjacent outputs that share ONLY their boundary row (`start(o+1) ==
  /// last(o)`) do not count as simultaneously open: the boundary row
  /// finalizes `o` as it opens `o+1`, which the single accumulator already
  /// handles. Only an output that STARTS STRICTLY BEFORE the current one's
  /// last row (`start(o2) < last(o)`) needs its own live slot. So a co-sited
  /// box plan returns 1; a `v = 1` Bottom fold on a downscale returns 2 (one
  /// backward reach); an EXPANDING (upscale) phased plan can return 3+, which
  /// the bounded 2-row window cannot stream — the caller rejects it.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  pub(crate) fn window_depth(&self) -> usize {
    let mut depth = 0;
    for o in 0..self.out_len() {
      let (s, w) = self.span(o);
      let last = s + w.len() - 1;
      // Starts are non-decreasing, so the outputs that open before `o`
      // finalizes form a contiguous run right after `o`.
      let mut open = 1;
      for o2 in (o + 1)..self.out_len() {
        if self.span(o2).0 < last {
          open += 1;
        } else {
          break;
        }
      }
      depth = depth.max(open);
    }
    depth
  }

  /// Fallible deep copy following the planner's recoverable-allocation
  /// contract — used by [`AreaStream`] to own its geometry for the
  /// frame, so scalar and SIMD passes cannot be fed mismatched spans.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  fn try_clone(&self) -> Result<Self, AxisError> {
    fn copy<T: Copy>(src: &[T]) -> Result<Vec<T>, AxisError> {
      let mut v = Vec::new();
      v.try_reserve_exact(src.len())
        .map_err(|_| AxisError::Alloc)?;
      v.extend_from_slice(src);
      Ok(v)
    }
    Ok(Self {
      starts: copy(&self.starts)?,
      offsets: copy(&self.offsets)?,
      weights: copy(&self.weights)?,
    })
  }
}

/// Which streaming engine a [`ResamplePlan`] drives: the integer
/// box-coverage [`Area`](SpanKind::Area) engine, or the
/// signed-coefficient [`Filter`](SpanKind::Filter) engine. A sinker's
/// resample tail matches on [`ResamplePlan::kind`] to feed the right
/// stream — the two engines never share a span arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant)]
pub enum SpanKind {
  /// Exact integer area (box-coverage) spans — see [`AreaResampler`].
  Area,
  /// Signed windowed-filter coefficients — see
  /// [`FilteredResampler`](crate::resample::FilteredResampler).
  Filter,
}

impl SpanKind {
  /// Lowercase identifier for diagnostics.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Area => "area",
      Self::Filter => "filter",
    }
  }
}

/// Output-geometry product of [`Resampler::plan`], built once at
/// sinker construction. Carries the per-axis spans the streaming engine
/// consumes; for [`SpanKind::Area`] the source dimensions double as the
/// spans' normalization denominators.
///
/// The area spans `h`/`v` are always present (an [`AreaResampler`] plan,
/// the native-tier chroma grids); a [`SpanKind::Filter`] plan additionally
/// carries the signed [`FilterAxis`] windows in `filter_h`/`filter_v` and
/// leaves the area spans empty (the area path never reads them — the
/// sink branches on [`Self::kind`] first). `PartialEq` only (the filter
/// coefficients are `f32`, hence no `Eq`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResamplePlan {
  src_w: usize,
  src_h: usize,
  out_w: usize,
  out_h: usize,
  kind: SpanKind,
  h: AxisSpans,
  v: AxisSpans,
  /// Horizontal filter windows; `Some` iff `kind == Filter`. For a BICUBLIN
  /// plan ([`Self::bicublin`]) this carries the **luma** plane's horizontal
  /// windows (the cubic kernel at luma resolution).
  filter_h: Option<FilterAxis>,
  /// Vertical filter windows; `Some` iff `kind == Filter`. For a BICUBLIN
  /// plan this carries the **luma** plane's vertical windows.
  filter_v: Option<FilterAxis>,
  /// Horizontal **chroma**-plane filter windows — `Some` only for a BICUBLIN
  /// plan ([`Self::bicublin`]), where the chroma plane is filtered at its own
  /// `chroma_w -> out_w` resolution with a second (linear) kernel. `None` for
  /// every single-kernel [`Self::filter`] / area plan, so those plans are
  /// byte-identical to before BICUBLIN existed and the existing filter path
  /// never reads them.
  filter_h_chroma: Option<FilterAxis>,
  /// Vertical **chroma**-plane filter windows — the `chroma_h -> out_h`
  /// twin of [`Self::filter_h_chroma`]. `Some` only for a BICUBLIN plan.
  filter_v_chroma: Option<FilterAxis>,
  /// Horizontal chroma **sampling phase** (RFC #238 chroma siting), in
  /// chroma-sample units: a sub-sample additive shift of the chroma
  /// resample's window centers (`center = (xx + 0.5) * scale + h_phase`).
  /// `0.0` is co-sited — today's reconstruction — so every plan built in this
  /// foundation carries `0.0` and is byte-identical to before the field
  /// existed. A [`Self::bicublin`] plan bakes it into the chroma
  /// [`FilterAxis`] centers at build time; an area chroma plan
  /// ([`Self::area_chroma_420`] and siblings) carries it for the phase-aware
  /// folded-weight series, leaving the integer cell-overlap weights unchanged
  /// while it is `0.0`.
  h_phase: f64,
  /// Vertical chroma sampling phase — the `h_phase` twin on the V axis
  /// (e.g. 4:2:0 Bottom siting). `0.0` = co-sited.
  v_phase: f64,
}

impl ResamplePlan {
  /// Builds the exact area plan for `src -> out`. The strategy has
  /// already validated zero, upscale, and identity geometry. Also the
  /// constructor for auxiliary plane grids: the native tier plans a
  /// subsampled format's chroma grid against the same output geometry,
  /// where the coverage may run in the upsample direction.
  pub(crate) fn area(
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
  ) -> Result<Self, ResampleError> {
    let fail = |e: AxisError| match e {
      AxisError::Overflow => ResampleError::Overflow(PlanGeometry::new(src_w, src_h, out_w, out_h)),
      AxisError::Alloc => {
        ResampleError::AllocationFailed(PlanGeometry::new(src_w, src_h, out_w, out_h))
      }
    };
    // Sequential on purpose: the second axis is not built when the
    // first has already failed.
    let h = AxisSpans::area(src_w, out_w).map_err(fail)?;
    let v = AxisSpans::area(src_h, out_h).map_err(fail)?;
    Ok(Self {
      src_w,
      src_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase: 0.0,
      v_phase: 0.0,
    })
  }

  /// Builds a signed-coefficient **filter** plan for `src -> out` under
  /// `kernel`. The strategy ([`FilteredResampler`]) has already validated
  /// zero, upscale, and identity geometry. The area spans are left empty
  /// (this plan's [`Self::kind`] is [`SpanKind::Filter`], so they are
  /// never read); the per-axis windows live in `filter_h`/`filter_v`.
  pub(crate) fn filter(
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
    kernel: &dyn FilterKernel,
  ) -> Result<Self, ResampleError> {
    // Sequential on purpose: the second axis is not built when the first
    // has already failed (and a hostile-support rejection short-circuits).
    let filter_h = FilterAxis::build(src_w, out_w, kernel, 0.0)?;
    let filter_v = FilterAxis::build(src_h, out_h, kernel, 0.0)?;
    Ok(Self {
      src_w,
      src_h,
      out_w,
      out_h,
      kind: SpanKind::Filter,
      h: AxisSpans::empty(),
      v: AxisSpans::empty(),
      filter_h: Some(filter_h),
      filter_v: Some(filter_v),
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase: 0.0,
      v_phase: 0.0,
    })
  }

  /// Builds a **BICUBLIN** plan: swscale's per-plane filter, where the luma
  /// plane is filtered with `luma_kernel` (the cubic) and the chroma planes
  /// with `chroma_kernel` (the linear / tent), each at its own native
  /// resolution into the shared output grid.
  ///
  /// The luma windows (`src_w x src_h -> out_w x out_h`) land in
  /// `filter_h`/`filter_v`, the chroma windows
  /// (`chroma_w x chroma_h -> out_w x out_h`) in
  /// `filter_h_chroma`/`filter_v_chroma`. The plan's [`Self::kind`] is
  /// [`SpanKind::Filter`] so every area-only consumer's existing filter guard
  /// rejects it unchanged; only the `Yuv420p` BICUBLIN route reads the chroma
  /// windows ([`Self::is_bicublin`]). Both kernels are evaluated into
  /// coefficients here, once, at plan time.
  ///
  /// All four axes are built sequentially under the recoverable-allocation
  /// contract — a hostile-support rejection on any axis short-circuits before
  /// the next is sized, so an invalid kernel never allocates past the failing
  /// axis.
  ///
  /// # Errors
  ///
  /// As [`Self::filter`], for any of the four axes:
  /// [`ResampleError::InvalidFilterSupport`] / [`ResampleError::Overflow`] /
  /// [`ResampleError::AllocationFailed`].
  #[cfg(feature = "yuv-planar")]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn bicublin(
    src_w: usize,
    src_h: usize,
    chroma_w: usize,
    chroma_h: usize,
    out_w: usize,
    out_h: usize,
    luma_kernel: &dyn FilterKernel,
    chroma_kernel: &dyn FilterKernel,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    // Sequential on purpose (matching [`Self::filter`]): a failing axis
    // short-circuits before the next is built, so a hostile support never
    // sizes a later axis's table.
    let filter_h = FilterAxis::build(src_w, out_w, luma_kernel, 0.0)?;
    let filter_v = FilterAxis::build(src_h, out_h, luma_kernel, 0.0)?;
    let filter_h_chroma = FilterAxis::build(chroma_w, out_w, chroma_kernel, h_phase)?;
    let filter_v_chroma = FilterAxis::build(chroma_h, out_h, chroma_kernel, v_phase)?;
    Ok(Self {
      src_w,
      src_h,
      out_w,
      out_h,
      kind: SpanKind::Filter,
      h: AxisSpans::empty(),
      v: AxisSpans::empty(),
      filter_h: Some(filter_h),
      filter_v: Some(filter_v),
      filter_h_chroma: Some(filter_h_chroma),
      filter_v_chroma: Some(filter_v_chroma),
      h_phase,
      v_phase,
    })
  }

  /// Builds the 4:2:0 chroma plan for the native tier: horizontal
  /// spans over the (uniform, exact — frame widths are even) chroma
  /// width, vertical spans over the LUMA height with paired cells
  /// ([`AxisSpans::area_halved`]) so an odd trailing luma row weights
  /// its chroma row by half. The stored source dims are
  /// `(chroma_w, luma_h)` — the per-plane normalization denominators.
  ///
  /// `h_phase` carries the RFC #238 horizontal chroma siting. At phase 0
  /// (co-sited / unspecified) this is **byte-identical** to the plain
  /// [`AxisSpans::area`]-over-`chroma_w` H axis the native 4:2:0 chroma
  /// resample built before siting existed. For the centered group
  /// (`h_phase ≠ 0`) the horizontal spans are the folded triangle⊗box weights
  /// ([`AxisSpans::area_chroma_phased_h_centered`], shared with
  /// [`Self::area_chroma_422`]): one phased area pass on the subsampled grid
  /// reproducing "reconstruct full width with the #302 `1/4`–`3/4` triangle,
  /// then box-average" with a SINGLE rounding. Its output spans sum to
  /// `4·(2·chroma_w) = 8·chroma_w`, so the stored `src_w` becomes that scaled
  /// H denominator and the existing [`AreaStream`] normalization
  /// (`src_w·src_h`) and exactness guards bound the scaled weights with no
  /// change (the `4·` only tightens the maximum representable width, which
  /// stays absurd).
  ///
  /// The vertical axis mirrors the horizontal: at `v_phase == 0` it is the
  /// co-sited [`AxisSpans::area_halved`] luma→chroma pairing (`src_h` stays
  /// `luma_h`), byte-identical to before vertical siting existed; for 4:2:0
  /// **Bottom** siting (`v_phase ≠ 0`) it is the folded `v = 1` triangle
  /// ([`AxisSpans::area_chroma_phased_v`]), whose spans sum to `2·luma_h`, so
  /// the stored `src_h` becomes that scaled V denominator. The two folds are
  /// independent — a centered-H + Bottom-V plan scales both dimensions, and
  /// the [`AreaStream`] `src_w·src_h` denominator and exactness guards bound
  /// the combined weights unchanged.
  #[cfg(feature = "yuv-planar")]
  pub(crate) fn area_chroma_420(
    chroma_w: usize,
    luma_h: usize,
    out_w: usize,
    out_h: usize,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    let fail_overflow =
      || ResampleError::Overflow(PlanGeometry::new(chroma_w, luma_h, out_w, out_h));
    let fail_alloc =
      || ResampleError::AllocationFailed(PlanGeometry::new(chroma_w, luma_h, out_w, out_h));
    let fail = |e: AxisError| match e {
      AxisError::Overflow => fail_overflow(),
      AxisError::Alloc => fail_alloc(),
    };
    // Vertical axis first (matching the pre-siting build order so an
    // allocator refusal on the paired axis still surfaces before the
    // horizontal one): the co-sited luma→chroma pairing, or — for 4:2:0
    // Bottom siting — the folded `v = 1` triangle whose spans sum to
    // `2·luma_h`, making that the scaled V denominator. At `v_phase == 0`
    // the spans and the `luma_h` denominator are byte-identical to the plain
    // pairing every co-sited caller built before vertical siting existed.
    let (v, denom_h) = if v_phase == 0.0 {
      (AxisSpans::area_halved(luma_h, out_h).map_err(fail)?, luma_h)
    } else {
      let denom_h = luma_h.checked_mul(2).ok_or_else(fail_overflow)?;
      (
        AxisSpans::area_chroma_phased_v(luma_h, out_h).map_err(fail)?,
        denom_h,
      )
    };
    // Horizontal axis: the co-sited chroma-grid box, or the centered folded
    // triangle whose output spans sum to `4·(2·chroma_w) = 8·chroma_w`, the
    // scaled H denominator. At `h_phase == 0` both are byte-identical to the
    // pre-siting area plan.
    let (h, denom_w) = if h_phase == 0.0 {
      (AxisSpans::area(chroma_w, out_w).map_err(fail)?, chroma_w)
    } else {
      let denom_w = chroma_w.checked_mul(8).ok_or_else(fail_overflow)?;
      (
        AxisSpans::area_chroma_phased_h_centered(chroma_w, out_w).map_err(fail)?,
        denom_w,
      )
    };
    Ok(Self {
      src_w: denom_w,
      src_h: denom_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase,
      v_phase,
    })
  }

  /// Builds the 4:4:0 chroma plan for the native tier: horizontal spans over
  /// the FULL frame width (4:4:0 chroma is full-width — never subsampled
  /// horizontally, so `h_phase` folds no triangle here), vertical spans over
  /// the LUMA height.
  ///
  /// The vertical axis carries the RFC #238 vertical chroma siting. At
  /// `v_phase == 0` it is the co-sited [`AxisSpans::area_halved`] luma→chroma
  /// pairing (an odd trailing luma row weights its chroma row by half — the
  /// same luma-domain vertical weighting as 4:2:0) and the stored `src_h`
  /// stays `luma_h`, byte-identical to before vertical siting existed; for
  /// **Bottom** siting (`v_phase ≠ 0`) it is the folded `v = 1` triangle
  /// ([`AxisSpans::area_chroma_phased_v`]), whose spans sum to `2·luma_h`, so
  /// the stored `src_h` becomes that scaled V denominator. The stored source
  /// width is always `frame_w`.
  #[cfg(feature = "yuv-planar")]
  pub(crate) fn area_chroma_440(
    frame_w: usize,
    luma_h: usize,
    out_w: usize,
    out_h: usize,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    let fail_overflow =
      || ResampleError::Overflow(PlanGeometry::new(frame_w, luma_h, out_w, out_h));
    let fail_alloc =
      || ResampleError::AllocationFailed(PlanGeometry::new(frame_w, luma_h, out_w, out_h));
    let fail = |e: AxisError| match e {
      AxisError::Overflow => fail_overflow(),
      AxisError::Alloc => fail_alloc(),
    };
    // Horizontal axis: the full-width chroma box — 4:4:0 keeps full-width
    // chroma, so there is no folded H triangle and the denominator stays
    // `frame_w`.
    let h = AxisSpans::area(frame_w, out_w).map_err(fail)?;
    // Vertical axis: the co-sited luma→chroma pairing, or — for Bottom siting
    // — the folded `v = 1` triangle whose spans sum to `2·luma_h`, making that
    // the scaled V denominator. At `v_phase == 0` the spans and the `luma_h`
    // denominator are byte-identical to the plain pairing every co-sited
    // caller built before vertical siting existed.
    let (v, denom_h) = if v_phase == 0.0 {
      (AxisSpans::area_halved(luma_h, out_h).map_err(fail)?, luma_h)
    } else {
      let denom_h = luma_h.checked_mul(2).ok_or_else(fail_overflow)?;
      (
        AxisSpans::area_chroma_phased_v(luma_h, out_h).map_err(fail)?,
        denom_h,
      )
    };
    Ok(Self {
      src_w: frame_w,
      src_h: denom_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase,
      v_phase,
    })
  }

  /// Builds the 4:1:0 chroma plan for the row-stage HSV-only resample:
  /// horizontal spans over the (uniform, exact — 4:1:0 width is a multiple
  /// of 4) quarter-width chroma, vertical spans over the LUMA height with
  /// quartered cells ([`AxisSpans::area_subsampled`] at factor 4) so a
  /// partial trailing group of `1..=3` luma rows weights its chroma row by
  /// its true coverage — the same luma-domain vertical weighting as 4:2:0
  /// ([`Self::area_chroma_420`]), only quartered. The stored source dims
  /// are `(chroma_w, luma_h)`.
  #[cfg(feature = "yuv-planar")]
  pub(crate) fn area_chroma_410(
    chroma_w: usize,
    luma_h: usize,
    out_w: usize,
    out_h: usize,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    let fail = |e: AxisError| match e {
      AxisError::Overflow => {
        ResampleError::Overflow(PlanGeometry::new(chroma_w, luma_h, out_w, out_h))
      }
      AxisError::Alloc => {
        ResampleError::AllocationFailed(PlanGeometry::new(chroma_w, luma_h, out_w, out_h))
      }
    };
    let h = AxisSpans::area(chroma_w, out_w).map_err(fail)?;
    let v = AxisSpans::area_subsampled(luma_h, out_h, 4).map_err(fail)?;
    Ok(Self {
      src_w: chroma_w,
      src_h: luma_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase,
      v_phase,
    })
  }

  /// Builds the 4:1:1 chroma plan for the row-stage HSV-only resample:
  /// horizontal spans over the LUMA width with quartered cells
  /// ([`AxisSpans::area_subsampled`] at factor 4) so a partial trailing
  /// group of `1..=3` luma columns weights its chroma sample by its true
  /// coverage (4:1:1 width may be non-multiple-of-4 — the last chroma
  /// sample is shared by the trailing `1..=3` luma columns), vertical spans
  /// over the FULL frame height (4:1:1 chroma is full-height). The stored
  /// source dims are `(luma_w, frame_h)`.
  #[cfg(feature = "yuv-planar")]
  pub(crate) fn area_chroma_411(
    luma_w: usize,
    frame_h: usize,
    out_w: usize,
    out_h: usize,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    let fail = |e: AxisError| match e {
      AxisError::Overflow => {
        ResampleError::Overflow(PlanGeometry::new(luma_w, frame_h, out_w, out_h))
      }
      AxisError::Alloc => {
        ResampleError::AllocationFailed(PlanGeometry::new(luma_w, frame_h, out_w, out_h))
      }
    };
    let h = AxisSpans::area_subsampled(luma_w, out_w, 4).map_err(fail)?;
    let v = AxisSpans::area(frame_h, out_h).map_err(fail)?;
    Ok(Self {
      src_w: luma_w,
      src_h: frame_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase,
      v_phase,
    })
  }

  /// Builds the 4:2:2 chroma plan for the native tier: half-width chroma
  /// (`chroma_w = frame_w / 2`), full height (`chroma_vsub = 1`), against the
  /// shared output geometry. Vertical spans bin the full-height chroma to
  /// `out_h` ([`AxisSpans::area`] — 4:2:2 has no vertical chroma subsampling),
  /// so `v_phase` must be `0`.
  ///
  /// `h_phase` carries the RFC #238 horizontal chroma siting. At phase 0
  /// (co-sited / unspecified) this is **byte-identical** to [`Self::area`] over
  /// `(chroma_w, luma_h)` — the plan the native 4:2:2 chroma resample built
  /// before siting existed. For the centered group (`h_phase ≠ 0`) the
  /// horizontal spans are the folded triangle⊗box weights
  /// ([`AxisSpans::area_chroma_phased_h_centered`]): one phased area pass on
  /// the subsampled grid reproducing "reconstruct full width with the #302
  /// `1/4`–`3/4` triangle, then box-average" with a SINGLE rounding. The stored
  /// `src_w` is then the scaled H denominator `4·(2·chroma_w) = 8·chroma_w`, so
  /// the existing [`AreaStream`] normalization (`src_w·src_h`) and exactness
  /// guards ([`AreaSample::h_sum_fits`] / [`AreaSample::denom_fits`], evaluated
  /// against the stored `src_w`) bound the scaled weights with no change — the
  /// `4·` only tightens the maximum representable width, which stays absurd.
  #[cfg(feature = "yuv-planar")]
  pub(crate) fn area_chroma_422(
    chroma_w: usize,
    luma_h: usize,
    out_w: usize,
    out_h: usize,
    h_phase: f64,
    v_phase: f64,
  ) -> Result<Self, ResampleError> {
    debug_assert_eq!(v_phase, 0.0, "4:2:2 carries no vertical chroma phase");
    if h_phase == 0.0 {
      // Co-sited / unspecified — byte-identical to the pre-siting area plan.
      return Self::area(chroma_w, luma_h, out_w, out_h);
    }
    let geometry = || PlanGeometry::new(chroma_w, luma_h, out_w, out_h);
    let fail = |e: AxisError| match e {
      AxisError::Overflow => ResampleError::Overflow(geometry()),
      AxisError::Alloc => ResampleError::AllocationFailed(geometry()),
    };
    // Scaled H denominator: each folded output span sums to `4·(2·chroma_w)`.
    let denom_w = chroma_w
      .checked_mul(8)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    let h = AxisSpans::area_chroma_phased_h_centered(chroma_w, out_w).map_err(fail)?;
    let v = AxisSpans::area(luma_h, out_h).map_err(fail)?;
    Ok(Self {
      src_w: denom_w,
      src_h: luma_h,
      out_w,
      out_h,
      kind: SpanKind::Area,
      h,
      v,
      filter_h: None,
      filter_v: None,
      filter_h_chroma: None,
      filter_v_chroma: None,
      h_phase,
      v_phase,
    })
  }

  /// Source width in pixels — the horizontal spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height in pixels — the vertical spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Which streaming engine this plan drives.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kind(&self) -> SpanKind {
    self.kind
  }

  /// Whether this plan carries a non-zero chroma sampling phase (RFC #238) —
  /// `true` for a centered / phased chroma resample, `false` for co-sited.
  /// The native and HSV-only joins cache a chroma plan built for one frame's
  /// siting; a reused sink whose `chroma_location` moved to a different phase
  /// compares this flag to REBUILD the join rather than reuse a stale plan.
  /// (Comparison against `0.0` is exact and exempt from `clippy::float_cmp`.)
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn has_chroma_phase(&self) -> bool {
    self.h_phase != 0.0 || self.v_phase != 0.0
  }

  /// Whether this plan carries a non-zero **vertical** chroma phase (RFC #238
  /// 4:2:0 `Bottom` siting, `v = 1`). [`Self::has_chroma_phase`] cannot tell
  /// `Center` (`h ≠ 0`, `v = 0`) from `Bottom` (`h ≠ 0`, `v = 1`), so the
  /// native / HSV-only joins compare THIS flag alongside it to rebuild a cached
  /// chroma plan whose vertical siting moved. `false` for co-sited and every
  /// horizontal-only siting. (Comparison against `0.0` is exact and exempt from
  /// `clippy::float_cmp`.)
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn has_chroma_v_phase(&self) -> bool {
    self.v_phase != 0.0
  }

  /// [`ResampleError::UnsupportedFilter`] carrying this plan's geometry —
  /// returned by an area-only format's resample tail when handed a
  /// [`SpanKind::Filter`] plan, so the empty area spans never reach an
  /// area stream. The area helpers are shared across formats, so the guard
  /// lives at each plan-consumption point that builds an area stream.
  // Idle in feature combos that compile no plan-consuming sink.
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn unsupported_filter(&self) -> ResampleError {
    ResampleError::UnsupportedFilter(PlanGeometry::new(
      self.src_w, self.src_h, self.out_w, self.out_h,
    ))
  }

  /// [`ResampleError::LinearDomainUnsupported`] carrying this plan's geometry
  /// — returned by the planar 8-bit YUV dispatch when a sink requests the
  /// [`AveragingDomain::Linear`] domain on a build without the `rgb` feature,
  /// so the request fails with a typed error rather than silently falling
  /// through to the encoded average.
  // Idle in feature combos that compile the linear tail (`rgb` on) or no
  // planar 8-bit YUV sink at all.
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn linear_domain_unsupported(&self) -> ResampleError {
    ResampleError::LinearDomainUnsupported(PlanGeometry::new(
      self.src_w, self.src_h, self.out_w, self.out_h,
    ))
  }

  /// [`ResampleError::PremultipliedDomainUnsupported`] carrying this plan's
  /// geometry — returned by the planar 8-bit YUV dispatch when a sink requests
  /// the [`AveragingDomain::Premultiplied`] domain. Premultiplied weighting is
  /// only meaningful for a format with an alpha channel; these YUV formats
  /// carry no alpha, so the request is rejected with a typed error rather than
  /// silently downgrading to the encoded average.
  // Idle in feature combos that compile no planar 8-bit YUV sink.
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn premultiplied_domain_unsupported(&self) -> ResampleError {
    ResampleError::PremultipliedDomainUnsupported(PlanGeometry::new(
      self.src_w, self.src_h, self.out_w, self.out_h,
    ))
  }

  /// Horizontal filter windows — `Some` iff [`Self::kind`] is
  /// [`SpanKind::Filter`]. Consumed by the filter streaming engine.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn filter_h(&self) -> Option<&FilterAxis> {
    self.filter_h.as_ref()
  }

  /// Vertical filter windows — `Some` iff [`Self::kind`] is
  /// [`SpanKind::Filter`]. Consumed by the filter streaming engine.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn filter_v(&self) -> Option<&FilterAxis> {
    self.filter_v.as_ref()
  }

  /// Horizontal **chroma**-plane filter windows — `Some` only for a BICUBLIN
  /// plan ([`Self::bicublin`]), `None` otherwise (including a single-kernel
  /// [`Self::filter`] plan). Read by the `Yuv420p` BICUBLIN route to filter
  /// the U / V planes with the chroma (linear) kernel.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn filter_h_chroma(&self) -> Option<&FilterAxis> {
    self.filter_h_chroma.as_ref()
  }

  /// Vertical **chroma**-plane filter windows — the twin of
  /// [`Self::filter_h_chroma`]. `Some` only for a BICUBLIN plan.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn filter_v_chroma(&self) -> Option<&FilterAxis> {
    self.filter_v_chroma.as_ref()
  }

  /// Whether this is a **BICUBLIN** per-plane filter plan ([`Self::bicublin`])
  /// — a [`SpanKind::Filter`] plan that *also* carries the chroma-plane
  /// windows. A single-kernel [`Self::filter`] plan reports `false` (it has no
  /// chroma windows), so the `Yuv420p` dispatch routes BICUBLIN to the
  /// per-plane path and leaves every other filter plan on the unchanged
  /// single-kernel path.
  ///
  /// Always compiled (the backing `filter_h_chroma` field is not feature-gated)
  /// so [`Self::ensure_single_kernel_filter`] can fence a BICUBLIN plan out of
  /// every single-kernel consumer regardless of which families a build enables.
  // Idle in feature combos that compile no filter consumer.
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn is_bicublin(&self) -> bool {
    self.filter_h_chroma.is_some()
  }

  /// Fences a **BICUBLIN** plan out of a single-kernel filter consumer:
  /// `Err(`[`ResampleError::UnsupportedFilter`]`)` if this is a BICUBLIN plan
  /// ([`Self::is_bicublin`]), `Ok(())` otherwise.
  ///
  /// A BICUBLIN plan keeps [`Self::kind`] `==` [`SpanKind::Filter`] (no
  /// dedicated span-kind variant), and it carries a SECOND (chroma) window set
  /// that only the `Yuv420p` BICUBLIN route reads. Every OTHER filter consumer
  /// builds a SINGLE-kernel [`FilterStream`] from the luma windows
  /// ([`Self::filter_h`]/[`Self::filter_v`]) and would silently ignore the
  /// chroma windows — filtering all planes with the luma cubic kernel — which
  /// is wrong output. So each single-kernel filter consumer calls this at the
  /// top of its filter path: the `Yuv420p` per-plane route is the ONLY consumer
  /// that may accept a BICUBLIN plan; every other rejects it with the typed
  /// error rather than mis-filtering it.
  // Idle in feature combos that compile no filter consumer.
  #[allow(dead_code)]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn ensure_single_kernel_filter(&self) -> Result<(), ResampleError> {
    if self.is_bicublin() {
      return Err(self.unsupported_filter());
    }
    Ok(())
  }

  /// Horizontal-axis spans.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn h(&self) -> &AxisSpans {
    &self.h
  }

  /// Vertical-axis spans.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it.
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &AxisSpans {
    &self.v
  }

  /// Output width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Output height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }

  /// Output `(width, height)` pair.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_dims(&self) -> (usize, usize) {
    (self.out_w, self.out_h)
  }
}

/// Source vs requested-output geometry payload for
/// [`ResampleError::UpscaleUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscaleUnsupported {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl UpscaleUnsupported {
  /// Constructs a new `UpscaleUnsupported` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Requested-output geometry payload for
/// [`ResampleError::ZeroOutputDimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroOutputDimension {
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl ZeroOutputDimension {
  /// Constructs a new `ZeroOutputDimension` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// `round(a / d)` with ties rounding up, overflow-free for every
/// `u64` pair (`d > 0`): the naive `(a + d / 2) / d` wraps when `a`
/// sits near `u64::MAX`, while `r >= d - d / 2` compares the remainder
/// against `ceil(d / 2)` without any widening arithmetic.
// Consumed by the area streaming engine (gated to routed families)
// and its std-gated tests; allowed to idle in the remaining combos.
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
fn round_div_half_up(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// `u128` twin of [`round_div_half_up`] for the `u32` area stream, whose
/// numerator/denominator (`denom * u32::MAX`-bounded V-accumulation over a
/// `src_w * src_h` denominator) overflows `u64`. Same overflow-free
/// ties-up rounding: `r >= d - d / 2` compares the remainder against
/// `ceil(d / 2)` without widening past `u128`.
// Consumed only by `AreaSample<u32>::finalize`, instantiated only by `Gray32`
// (`gray`) for now; allowed to idle in the combos with no `u32` area router.
#[cfg_attr(not(feature = "gray"), allow(dead_code))]
fn round_div_half_up_u128(a: u128, d: u128) -> u128 {
  let q = a / d;
  let r = a % d;
  q + u128::from(r >= d - d / 2)
}

/// The sample element an [`AreaStream`] resamples — abstracts the
/// element width, the H-pass accumulator ([`Self::HSum`]), the V-pass
/// accumulator ([`Self::VAcc`]), the per-axis kernels, and the
/// finalize. `u8` and `u16` route the SIMD dispatchers and finalize
/// round-half-up in `u64`; `f32` accumulates in float (scalar; SIMD
/// follows) and finalizes with a plain divide.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
pub(crate) trait AreaSample: Copy + Default {
  /// Whether this element routes a SIMD H-pass that consumes the plan-time
  /// [`PaddedSpans`](crate::row::PaddedSpans) staging arena. `true` for the
  /// SIMD-backed tiers (`u8` / `u16` / `f32` — the default); `false` for the
  /// scalar-only `u32` tier, whose [`Self::h_reduce`] always uses the
  /// unpadded `u128` reducer and ignores `padded`. [`AreaStream::new`] skips
  /// building the arena entirely when this is `false`, so the scalar-only
  /// path carries no dead staging allocation.
  const NEEDS_SIMD_STAGING: bool = true;
  /// H-pass accumulator element. Integer samples sum exactly in a wide
  /// integer (`u32` for `u8` — an H-sum reaches `src_w * 255` and the
  /// narrow lanes drive its SIMD kernel; `u64` for `u16`); `f32` sums
  /// in `f32`.
  type HSum: Copy + Default;
  /// V-pass accumulator element: `u64` for the integer streams (exact
  /// associative adds, hence 0-ULP SIMD parity), `f32` for the float
  /// stream (non-associative, hence parity is a small tolerance).
  type VAcc: Copy + Default;
  /// `true` iff an H-sum for a `src_w`-wide plane fits [`Self::HSum`]
  /// without overflow (always `true` for the float stream).
  fn h_sum_fits(src_w: u64) -> bool;
  /// `true` iff `denom` and the V-accumulation it bounds stay exact
  /// (always `true` for the float stream, which cannot integer-overflow).
  fn denom_fits(denom: u64) -> bool;
  /// Reduces one source row into `h_tmp` (per-span weighted sums).
  fn h_reduce(
    row: &[Self],
    channels: usize,
    h: &AxisSpans,
    padded: Option<&crate::row::PaddedSpans>,
    h_tmp: &mut [Self::HSum],
    use_simd: bool,
  );
  /// Accumulates `acc[i] += w * h_tmp[i]` into the V-accumulators.
  fn v_accumulate(acc: &mut [Self::VAcc], h_tmp: &[Self::HSum], w: u64, use_simd: bool);
  /// Finalizes one accumulator — divide by `denom` (round-half-up for
  /// the integer streams, plain divide for the float stream).
  fn finalize(acc: Self::VAcc, denom: u64) -> Self;
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
impl AreaSample for u8 {
  type HSum = u32;
  type VAcc = u64;
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_sum_fits(src_w: u64) -> bool {
    src_w <= u64::from(u32::MAX) / 255
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn denom_fits(denom: u64) -> bool {
    denom <= u64::MAX / 255
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_reduce(
    row: &[u8],
    channels: usize,
    h: &AxisSpans,
    padded: Option<&crate::row::PaddedSpans>,
    h_tmp: &mut [u32],
    use_simd: bool,
  ) {
    crate::row::area_h_reduce_row(
      row, channels, &h.starts, &h.offsets, &h.weights, padded, h_tmp, use_simd,
    );
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u64, use_simd: bool) {
    crate::row::area_v_accumulate(acc, h_tmp, w, use_simd);
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: u64, denom: u64) -> u8 {
    round_div_half_up(acc, denom) as u8
  }
}

/// 16-bit element path: routes the SIMD dispatchers, consumed in
/// production by the high-bit packed-RGB sinkers (`Rgb48` / `Bgr48`).
/// Still unreachable under a `yuv-planar`-only build (no high-bit
/// planar format routes through it yet), hence the `rgb`-gated allow.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[cfg_attr(not(feature = "rgb"), allow(dead_code))]
impl AreaSample for u16 {
  type HSum = u64;
  type VAcc = u64;
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_sum_fits(src_w: u64) -> bool {
    src_w <= u64::MAX / 65535
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn denom_fits(denom: u64) -> bool {
    denom <= u64::MAX / 65535
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_reduce(
    row: &[u16],
    channels: usize,
    h: &AxisSpans,
    padded: Option<&crate::row::PaddedSpans>,
    h_tmp: &mut [u64],
    use_simd: bool,
  ) {
    crate::row::area_h_reduce_row_u16(
      row, channels, &h.starts, &h.offsets, &h.weights, padded, h_tmp, use_simd,
    );
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_accumulate(acc: &mut [u64], h_tmp: &[u64], w: u64, use_simd: bool) {
    crate::row::area_v_accumulate_u16(acc, h_tmp, w, use_simd);
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: u64, denom: u64) -> u16 {
    round_div_half_up(acc, denom) as u16
  }
}

/// 32-bit float element path: area-resamples float samples (linear-light
/// RGB / XYZ). Samples and the emitted output are `f32`, but **both
/// accumulators are `f64`** ([`Self::HSum`] / [`Self::VAcc`]): the
/// unnormalized area numerator reaches `denom * sample`, and an `f32`
/// numerator silently stops summing unit contributions past `2^24` (so
/// a wide constant image would not round-trip) and can overflow to
/// infinity for large finite samples. `f64`'s 53-bit mantissa and
/// vast range keep the numerator exact-enough for any realistic
/// geometry, so `h_sum_fits` / `denom_fits` need no overflow bound.
/// The single `f32` rounding is the final divide-and-cast. Scalar for
/// now; the SIMD kernels follow (their float adds reorder, so that
/// parity is a small tolerance, not 0-ULP). Gated like the integer
/// paths; reachable only from the parity tests until a float source
/// format routes through it (the dead-code allow drops then, as it did
/// for `u16` once `Rgb48` landed).
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[allow(dead_code)]
impl AreaSample for f32 {
  type HSum = f64;
  type VAcc = f64;
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_sum_fits(_src_w: u64) -> bool {
    true
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn denom_fits(_denom: u64) -> bool {
    true
  }
  fn h_reduce(
    row: &[f32],
    channels: usize,
    h: &AxisSpans,
    padded: Option<&crate::row::PaddedSpans>,
    h_tmp: &mut [f64],
    use_simd: bool,
  ) {
    crate::row::area_h_reduce_row_f32(
      row, channels, &h.starts, &h.offsets, &h.weights, padded, h_tmp, use_simd,
    );
  }
  fn v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: u64, use_simd: bool) {
    crate::row::area_v_accumulate_f32(acc, h_tmp, w as f64, use_simd);
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: f64, denom: u64) -> f32 {
    (acc / denom as f64) as f32
  }
}

/// 32-bit integer element path: area-resamples `u32` samples at **native
/// `u32` precision** for the `u32` source formats (`Gray32` / `Rgb96` /
/// `Rgba128` / `Gbrap32`). Both accumulators are **`u128`** ([`Self::HSum`] /
/// [`Self::VAcc`]): a single H-term is `weight * sample` (a `u16`-bounded
/// coverage weight times a `u32` sample, `~2^48`), a span sums many of them,
/// and the V-pass scales by the vertical coverage up to a `denom * u32::MAX`
/// numerator (`~2^96` worst case) — all far past `u64`. The integer adds are
/// exact, so the single round-half-up divide at finalize makes the resample
/// 0-ULP versus narrowing each `u32` *after* binning. **Scalar-only**: no
/// `u128` SIMD area kernel exists (the lane widths the integer dispatchers
/// drive top out at `u64`), and these formats are rare downscale paths, so
/// the H/V passes route the scalar references directly (ignoring `padded` /
/// `use_simd`). Gated like the other integer paths; the dead-code allow on
/// the scalar references drops once a `u32` format routes.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
// Instantiated only by `Gray32` (`gray`) for now — the impl is compiled under
// the shared area gate, so allow it to idle when no `u32` area router is on.
#[cfg_attr(not(feature = "gray"), allow(dead_code))]
impl AreaSample for u32 {
  // Scalar-only `u128` tier: no SIMD H-pass, so the stream skips the
  // `PaddedSpans` staging arena (`h_reduce` always uses the unpadded reducer).
  const NEEDS_SIMD_STAGING: bool = false;
  type HSum = u128;
  type VAcc = u128;
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_sum_fits(src_w: u64) -> bool {
    // An H-sum is bounded by `src_w * u32::MAX`; `u128` clears that for any
    // representable `src_w` (a `usize` is at most `2^64 - 1`, so the product
    // is below `2^96`). Expressed as the honest division bound.
    u128::from(src_w) <= u128::MAX / u128::from(u32::MAX)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn denom_fits(denom: u64) -> bool {
    // The V-accumulation is bounded by `denom * u32::MAX`; `u128` clears that
    // for any `u64` denom (`< 2^96`). Expressed as the honest bound so the
    // predicate documents the exactness contract rather than asserting `true`.
    u128::from(denom) <= u128::MAX / u128::from(u32::MAX)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_reduce(
    row: &[u32],
    channels: usize,
    h: &AxisSpans,
    _padded: Option<&crate::row::PaddedSpans>,
    h_tmp: &mut [u128],
    _use_simd: bool,
  ) {
    // Scalar-only: there is no `u128` SIMD area kernel, so `padded` /
    // `use_simd` are unused and the scalar reference is the production path.
    crate::row::area_h_reduce_row_u32(row, channels, &h.starts, &h.offsets, &h.weights, h_tmp);
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_accumulate(acc: &mut [u128], h_tmp: &[u128], w: u64, _use_simd: bool) {
    crate::row::area_v_accumulate_u32(acc, h_tmp, u128::from(w));
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: u128, denom: u64) -> u32 {
    round_div_half_up_u128(acc, u128::from(denom)) as u32
  }
}

/// Streaming separable area accumulator over [`AreaSample`] elements:
/// H-reduces each source row through the plan's horizontal spans,
/// accumulates it under the vertical span weights, and finalizes an
/// output row the moment its last contributing source row arrives —
/// the walker hands rows in order and [`PixelSink`](crate::PixelSink)
/// has no end-of-frame hook, so emission must ride the last
/// contribution.
///
/// Arithmetic is exact: weights are the plan's integer coverage
/// lengths, accumulation is `u64`, and the single divide per output
/// sample rounds half-up by `src_w * src_h`. Exactness makes the math
/// order-independent, which is what lets the SIMD tiers match the
/// scalar reference bit-for-bit.
///
/// Source rows must arrive strictly in order from row 0 each frame —
/// the accumulator state is meaningless otherwise — and
/// [`Self::feed_row`] enforces it, so a direct
/// [`process`](crate::PixelSink::process) caller replaying or
/// reordering rows gets an error instead of silently corrupted output.
///
/// Gated to the families that route through it; the gate widens as
/// formats wire in.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[derive(Debug)]
pub(crate) struct AreaStream<S: AreaSample> {
  /// Owned horizontal spans — both the scalar reference and the SIMD
  /// arena consume exactly this geometry; a caller cannot supply a
  /// divergent plan per row.
  h: AxisSpans,
  /// Owned vertical spans.
  v: AxisSpans,
  channels: usize,
  /// `src_w * src_h` — the exact normalization denominator.
  denom: u64,
  /// H-reduced current source row, `out_w * channels`. The element is
  /// [`AreaSample::HSum`] — exact for the sample width (`u32` for `u8`,
  /// an H-sum reaching `src_w * 255`; `u64` for `u16`), and creation
  /// bounds `src_w` accordingly via [`AreaSample::h_sum_fits`].
  h_tmp: Vec<S::HSum>,
  /// In-flight accumulator for the primary open output row (`cur_out`),
  /// `out_w * channels`. The element is [`AreaSample::VAcc`] — `u64` for
  /// the integer streams, `f32` for the float stream.
  acc: Vec<S::VAcc>,
  /// Second slot of the bounded 2-row window: the look-ahead accumulator
  /// for `cur_out + 1`, engaged only when a phased-V ([`v = 1`
  /// Bottom](AxisSpans::area_chroma_phased_v)) span reaches back into
  /// `cur_out`'s rows before it finalizes. Dormant (never written) for
  /// every co-sited plan, whose output spans overlap by at most one source
  /// row — see [`Self::feed_row`].
  acc_next: Vec<S::VAcc>,
  /// Finalized staging row handed to `emit`, `out_w * channels`.
  out_tmp: Vec<S>,
  /// Plan-time SIMD staging for the H-pass
  /// ([`crate::row::PaddedSpans`]); `None` routes the dispatcher to
  /// scalar.
  h_padded: Option<crate::row::PaddedSpans>,
  /// Next output row to finalize (the primary accumulator's row).
  cur_out: usize,
  /// Whether `acc_next` holds `cur_out + 1`'s partial rows — set when the
  /// look-ahead engages, cleared when it is promoted or the stream resets.
  next_open: bool,
  /// Next source row the frame expects; rows are strictly sequential.
  next_y: usize,
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
impl<S: AreaSample> AreaStream<S> {
  /// Creates a stream for `channels` interleaved channels of the
  /// plan's geometry. Fails with [`ResampleError::Overflow`] when the
  /// normalization denominator (or the per-element accumulator bound it
  /// must satisfy via [`AreaSample::denom_fits`]) is unrepresentable —
  /// vacuous for the float stream. `h`/`v` are the plane's own span
  /// sets and `src_w`/`src_h` its own grid (the chroma planes of a
  /// subsampled format run smaller grids — and possibly the upsample
  /// direction — against the same output geometry).
  pub(crate) fn new(
    h: &AxisSpans,
    v: &AxisSpans,
    src_w: usize,
    src_h: usize,
    channels: usize,
  ) -> Result<Self, ResampleError> {
    let geometry = || PlanGeometry::new(src_w, src_h, h.out_len(), v.out_len());
    // The bounded 2-row accumulator window streams at most two simultaneously
    // open output rows. A co-sited plan needs one; a `v = 1` Bottom fold on a
    // downscale needs two; an expanding (upscale) phased plan can need three or
    // more — which this engine cannot stream, so reject it up front rather than
    // silently drop the contributions of the third open row. `window` also
    // decides whether the look-ahead accumulator is allocated at all: a co-sited
    // stream (`window == 1`) never opens a second row, so it must NOT pay for
    // `acc_next` (keeping the pre-Bottom allocation envelope unchanged).
    let window = v.window_depth();
    if window > 2 {
      return Err(ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(
        src_w,
        src_h,
        h.out_len(),
        v.out_len(),
      )));
    }
    // Exactness bounds for the integer streams: an H-sum must fit
    // S::HSum and the V-accumulation must stay exact in u64. Both reject
    // only absurd magnitudes (a >16.8-million-pixel-wide plane for the
    // u8 H-sum); the float stream cannot integer-overflow, so its
    // predicates are vacuously true.
    if !S::h_sum_fits(src_w as u64) {
      return Err(ResampleError::Overflow(geometry()));
    }
    let denom = (src_w as u64)
      .checked_mul(src_h as u64)
      .filter(|d| S::denom_fits(*d))
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    let n = h
      .out_len()
      .checked_mul(channels)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    // Row buffers follow the planner's recoverable-allocation
    // contract: output-width rows are caller-proportional, not
    // small-constant, so refusal surfaces as an error rather than an
    // abort on the first processed row.
    let alloc = |_| ResampleError::AllocationFailed(geometry());
    let h = h
      .try_clone()
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    let v = v
      .try_clone()
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    // The SIMD H-pass staging arena is an accelerator only the SIMD-backed
    // tiers consume; the scalar-only `u32` tier always uses the unpadded
    // reducer, so skip building (and retaining) the arena for it entirely.
    let h_padded = if S::NEEDS_SIMD_STAGING {
      crate::row::PaddedSpans::build(&h.starts, &h.offsets, &h.weights)
    } else {
      None
    };
    Ok(Self {
      h,
      v,
      channels,
      denom,
      h_tmp: try_zeroed(n).map_err(alloc)?,
      acc: try_zeroed(n).map_err(alloc)?,
      // Allocated ONLY when the plan can open two rows at once (`window == 2`);
      // a co-sited stream leaves this empty and never touches it (the look-ahead
      // path is unreachable when spans overlap by at most their boundary row).
      acc_next: if window >= 2 {
        try_zeroed(n).map_err(alloc)?
      } else {
        Vec::new()
      },
      out_tmp: try_zeroed(n).map_err(alloc)?,
      h_padded,
      cur_out: 0,
      next_open: false,
      next_y: 0,
    })
  }

  /// Next source row this stream expects — the sinker-level preflight
  /// checks every requested stream against the incoming row index
  /// before any stream is fed, keeping a multi-stream `process` call
  /// atomic.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn next_y(&self) -> usize {
    self.next_y
  }

  /// Restarts the stream for a new frame.
  pub(crate) fn reset(&mut self) {
    self.acc.fill(S::VAcc::default());
    self.acc_next.fill(S::VAcc::default());
    self.cur_out = 0;
    self.next_open = false;
    self.next_y = 0;
  }

  /// Feeds source row `y` (`channels`-interleaved, source width) and
  /// invokes `emit(out_y, finalized_row)` for every output row this
  /// source row completes. Rows beyond the plan's coverage are
  /// accepted and ignored.
  ///
  /// # Errors
  ///
  /// [`ResampleError::OutOfSequenceRow`] when `y` is not the next
  /// expected source row; the stream state is untouched so the caller
  /// can resume with the expected row.
  pub(crate) fn feed_row(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    mut emit: impl FnMut(usize, &[S]),
  ) -> Result<(), ResampleError> {
    if y != self.next_y {
      return Err(ResampleError::OutOfSequenceRow(OutOfSequenceRow::new(
        self.next_y,
        y,
      )));
    }
    self.next_y += 1;
    if self.cur_out >= self.v.out_len() {
      return Ok(());
    }
    S::h_reduce(
      row,
      self.channels,
      &self.h,
      self.h_padded.as_ref(),
      &mut self.h_tmp,
      use_simd,
    );
    // Fold this source row into the open output rows, finalizing each as
    // its last contributing row arrives. `cur_out` is the primary in-flight
    // row (`acc`); a phased-V ([`v = 1`](AxisSpans::area_chroma_phased_v))
    // fold can additionally open ONE look-ahead row (`cur_out + 1`, in
    // `acc_next`) whose span reaches back into `cur_out`'s rows before it
    // finalizes — the bounded 2-row window. A co-sited plan's output spans
    // overlap by at most one source row, so the look-ahead never engages and
    // this is exactly the original single-accumulator path.
    loop {
      // With rows strictly sequential, `y` always lies in the current
      // span; the two defensive exits keep the no-panic contract if
      // that invariant is ever broken by a future edit.
      let (start, weights) = self.v.span(self.cur_out);
      let Some(idx) = y.checked_sub(start) else {
        return Ok(());
      };
      let Some(&w) = weights.get(idx) else {
        return Ok(());
      };
      let last = idx + 1 == weights.len();
      S::v_accumulate(&mut self.acc, &self.h_tmp, w as u64, use_simd);
      if !last {
        // `cur_out` stays open. Fold `y` into the look-ahead row too when a
        // phased-V span has already started under it — a no-op, and thus
        // byte-identical, for every co-sited plan.
        self.fold_lookahead(y, use_simd);
        return Ok(());
      }
      for (o, a) in self.out_tmp.iter_mut().zip(self.acc.iter_mut()) {
        *o = S::finalize(*a, self.denom);
        *a = S::VAcc::default();
      }
      emit(self.cur_out, &self.out_tmp);
      self.cur_out += 1;
      if self.next_open {
        // The look-ahead row already carries its earlier rows; promote it to
        // the primary accumulator (the just-finalized `acc` was zeroed, so
        // the swap leaves `acc_next` clear) and loop to fold `y` into it —
        // its own span may also end at `y`, cascading the finalize.
        core::mem::swap(&mut self.acc, &mut self.acc_next);
        self.next_open = false;
        if self.cur_out >= self.v.out_len() {
          return Ok(());
        }
        continue;
      }
      if self.cur_out >= self.v.out_len() || self.v.span(self.cur_out).0 != y {
        return Ok(());
      }
    }
  }

  /// Folds source row `y` into the look-ahead output row (`cur_out + 1`) —
  /// the second slot of the bounded 2-row window — when a phased-V span has
  /// reached back into `cur_out`'s still-open rows. A no-op for every
  /// co-sited plan (the next output span starts at or after `cur_out`'s last
  /// row, so `y` never lands in it while `cur_out` is open), which keeps the
  /// single-row path byte-identical.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn fold_lookahead(&mut self, y: usize, use_simd: bool) {
    let nxt = self.cur_out + 1;
    if nxt >= self.v.out_len() {
      return;
    }
    let (start, weights) = self.v.span(nxt);
    let Some(idx) = y.checked_sub(start) else {
      return;
    };
    let Some(&w) = weights.get(idx) else {
      return;
    };
    // Bounded to two open rows: a `v = 1` fold reaches back by exactly one
    // output row, so the row AFTER the look-ahead must not have started at or
    // before `y` (that would put `y` inside three open output spans). The
    // `window_depth <= 2` check in `new` rejects any plan that could reach
    // here, so this is a strict backstop — `start(nxt+1) == y` is already an
    // overflow, not a boundary.
    debug_assert!(
      nxt + 1 >= self.v.out_len() || self.v.span(nxt + 1).0 > y,
      "phased-V window exceeded two open output rows"
    );
    S::v_accumulate(&mut self.acc_next, &self.h_tmp, w as u64, use_simd);
    self.next_open = true;
  }
}

/// The streaming-engine surface a resample sink tail drives, shared by the
/// integer [`AreaStream`] and the signed [`FilterStream`] so one emit
/// helper feeds either: the sink branches on [`ResamplePlan::kind`] to
/// pick the concrete stream, then drives it through this trait. The sink's
/// per-kind sequencing reads `next_y` on the concrete stream; this trait
/// only abstracts the shared `feed_row` fan-out.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
pub(crate) trait RowResampler<S> {
  /// Feeds source row `y`, invoking `emit(out_y, finalized_row)` per
  /// completed output row. Errors with [`ResampleError::OutOfSequenceRow`]
  /// on a non-sequential row, leaving the stream untouched.
  fn feed_row<F: FnMut(usize, &[S])>(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    emit: F,
  ) -> Result<(), ResampleError>;
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
impl<S: AreaSample> RowResampler<S> for AreaStream<S> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn feed_row<F: FnMut(usize, &[S])>(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    emit: F,
  ) -> Result<(), ResampleError> {
    AreaStream::feed_row(self, y, row, use_simd, emit)
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
impl<S: FilterSample> RowResampler<S> for FilterStream<S> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn feed_row<F: FnMut(usize, &[S])>(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    emit: F,
  ) -> Result<(), ResampleError> {
    FilterStream::feed_row(self, y, row, use_simd, emit)
  }
}

// The `MixedSinker` holds its lazily-created area / filter streams behind a
// `Box` to keep its inline stack footprint small (the streams own several
// `Vec`s plus the span geometry). Forwarding the trait through the box lets the
// generic row-stage helpers take `&mut Box<Stream>` with no per-call deref.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
impl<S, R: RowResampler<S> + ?Sized> RowResampler<S> for std::boxed::Box<R> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn feed_row<F: FnMut(usize, &[S])>(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    emit: F,
  ) -> Result<(), ResampleError> {
    (**self).feed_row(y, row, use_simd, emit)
  }
}

// Single-shot box-allocation failpoint for the recoverably-boxed native joins
// (the straight-alpha `Yuva420p` tier and the `Yuv420p` BICUBLIN tier). Gated
// on `yuv-planar` (which `yuva` implies) because those are its only arming
// tests; `try_box` itself is gated wider (every boxed-stream consumer) but only
// consults the failpoint under this same gate.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_BOX_FAILURE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Arms a single-shot failpoint that makes the next [`try_box`] refuse, exactly
/// as a host OOM would. Used to prove the outer box allocation is recoverable
/// (typed `AllocationFailed`, not an abort) and transactional (the join field
/// is left `None`, so the call is retryable). Shared by the straight-alpha
/// `Yuva420p` and the BICUBLIN box tests. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_box_failure() {
  FORCE_BOX_FAILURE.with(|f| f.set(true));
}

/// Recoverable heap-box of a sized value: the stable analogue of the
/// nightly-only `Box::try_new`. The single backing allocation is taken
/// through a one-element [`Vec`] reservation
/// ([`try_reserve_exact`](Vec::try_reserve_exact)) so an allocator refusal
/// surfaces as `Err(TryReserveError)` instead of aborting the process the way
/// `Box::new` does. `into_boxed_slice` then hands back the exact-capacity
/// allocation with no copy; reinterpreting that single-element `Box<[T]>` as
/// `Box<T>` is the layout no-op below.
///
/// Shared by every lazily-boxed `MixedSinker` field — the per-plane area /
/// filter streams and the native-tier decimator joins — so a box-alloc refusal
/// surfaces the same recoverable `AllocationFailed` the inner stream / join
/// build already does. Gated like the blanket `Box<R>` impl above (the union of
/// every family whose sink boxes a stream).
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
pub(crate) fn try_box<T>(
  value: T,
) -> Result<std::boxed::Box<T>, std::collections::TryReserveError> {
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if FORCE_BOX_FAILURE.with(|f| f.take()) {
    // Reproduce the exact error `try_reserve_exact` yields on refusal so the
    // failpoint is indistinguishable from a real OOM.
    let mut probe = std::vec::Vec::<T>::new();
    probe.try_reserve_exact(usize::MAX)?;
  }
  let mut backing = std::vec::Vec::with_capacity(0);
  backing.try_reserve_exact(1)?;
  backing.push(value);
  // `backing` now holds exactly one initialized element at capacity one, so
  // `into_boxed_slice` returns that same allocation without reallocating.
  let boxed_slice: std::boxed::Box<[T]> = backing.into_boxed_slice();
  // SAFETY: `boxed_slice` owns a single element. `Box<[T; 1]>`, `Box<[T]>`
  // (over one element), and `Box<T>` all point at one `T` with identical size
  // and alignment, so reinterpreting the raw pointer transfers ownership of
  // the same allocation unchanged. The `Box` is rebuilt from the same pointer
  // exactly once, so the allocation is neither leaked nor double-freed.
  Ok(unsafe { std::boxed::Box::from_raw(std::boxed::Box::into_raw(boxed_slice).cast::<T>()) })
}

/// Row-sequencing payload for [`ResampleError::OutOfSequenceRow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfSequenceRow {
  /// Source row the stream expected next.
  expected: usize,
  /// Source row that was fed.
  got: usize,
}

impl OutOfSequenceRow {
  /// Constructs a new `OutOfSequenceRow` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(expected: usize, got: usize) -> Self {
    Self { expected, got }
  }

  /// Source row the stream expected next.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Source row that was fed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn got(&self) -> usize {
    self.got
  }
}

/// Geometry payload shared by [`ResampleError::Overflow`] and
/// [`ResampleError::AllocationFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanGeometry {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl PlanGeometry {
  /// Constructs a new `PlanGeometry` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Errors returned by [`Resampler::plan`] while validating the
/// requested output geometry at sinker construction, and by the
/// streaming engine when a direct caller violates the row-sequencing
/// contract.
///
/// The plan-validation variants surface before the sinker exists and
/// before any output buffer attaches; the sequencing variant leaves
/// the stream untouched. All are recoverable.
// No `Eq`: `InvalidFilterSupport` carries an `f64` support value (for
// diagnostics), and `f64` is not `Eq`. `PartialEq` covers every existing
// `assert_eq!` on this type; nothing requires `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum ResampleError {
  /// The requested output geometry exceeds the source geometry on at
  /// least one axis and the strategy only downscales.
  #[error(
    "resample output {}x{} exceeds source {}x{} (upscaling is unsupported)",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  UpscaleUnsupported(UpscaleUnsupported),

  /// The requested output geometry has a zero side.
  #[error(
    "resample output dimensions must be nonzero, got {}x{}",
    .0.out_w(), .0.out_h()
  )]
  ZeroOutputDimension(ZeroOutputDimension),

  /// Building the span tables would overflow `usize`: a per-axis
  /// `source x output` product is unrepresentable. Only reachable
  /// with extreme dimensions (32-bit targets foremost).
  #[error(
    "resample plan geometry overflows usize: source {}x{}, output {}x{}",
    .0.src_w(), .0.src_h(), .0.out_w(), .0.out_h()
  )]
  Overflow(PlanGeometry),

  /// A span-table arena reservation was refused by the allocator. The
  /// per-axis tables are `O(source + output)` entries, so a hostile
  /// source dimension from untrusted metadata lands here as a
  /// recoverable error instead of aborting inside infallible
  /// allocation.
  #[error(
    "resample plan allocation failed: source {}x{}, output {}x{}",
    .0.src_w(), .0.src_h(), .0.out_w(), .0.out_h()
  )]
  AllocationFailed(PlanGeometry),

  /// A streaming sinker was fed a source row out of order — rows must
  /// arrive strictly sequentially from row 0 each frame. Walker-driven
  /// processing never trips this; it guards direct
  /// [`process`](crate::PixelSink::process) callers replaying or
  /// reordering rows.
  #[error(
    "resample stream fed source row {}, expected row {}",
    .0.got(), .0.expected()
  )]
  OutOfSequenceRow(OutOfSequenceRow),

  /// A [`FilterKernel`]'s reported support radius was not finite,
  /// not strictly positive, or too large to size a window safely — a
  /// custom kernel cannot make the reconstruction window unsafe; it is
  /// rejected at plan build before any allocation.
  #[error(
    "filter kernel support {} is invalid for source extent {} (must be finite, > 0, and bounded)",
    .0.support(), .0.in_size()
  )]
  InvalidFilterSupport(InvalidFilterSupport),

  /// A [`SpanKind::Filter`] plan (from a
  /// [`FilteredResampler`]) was handed to a source format whose
  /// streaming sink only implements the integer area engine. Only
  /// `Rgb24` / `Rgb48` / `Grayf32` route the filter path in this
  /// release; every other format rejects a filter plan here rather
  /// than feeding the plan's empty area spans to its area stream
  /// (which would emit no output rows and leave attached outputs
  /// stale). Surfaces at the first processed row, before any output
  /// buffer is written.
  #[error(
    "filter resampling is unsupported for this source format (output {}x{} from source {}x{}); \
     route it through Rgb24, Rgb48, or Grayf32",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  UnsupportedFilter(PlanGeometry),

  /// The [`AveragingDomain::Linear`] area downscale was requested on a build
  /// whose feature set omits the RGB decode the linear-light tail needs. The
  /// domain is settable whenever `yuv-planar` is on (it gates the
  /// configuration), but the linear-light resample decodes every source pixel
  /// to RGB and so is only compiled under `rgb`; without it the domain cannot
  /// be honoured. Rather than silently downgrade to the encoded average
  /// (resampling in the wrong colour domain behind the caller's back), the sink
  /// rejects here at the first processed row, before any output buffer is
  /// written. Enable the `rgb` feature, or leave the domain at
  /// [`AveragingDomain::Encoded`].
  #[error(
    "the Linear averaging domain needs the `rgb` feature for its RGB decode \
     (output {}x{} from source {}x{}); enable `rgb` or use AveragingDomain::Encoded",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  LinearDomainUnsupported(PlanGeometry),

  /// The [`AveragingDomain::Premultiplied`] area downscale was requested on a
  /// format with no alpha channel. Premultiplied weighting scales each colour
  /// sample by its own alpha before averaging, so it is only meaningful for an
  /// alpha-bearing format; these YUV formats carry no alpha, making the domain
  /// a category error here. Rather than silently downgrade to the encoded
  /// average (resampling in a different domain than the caller asked for
  /// behind their back), the sink rejects here at the first processed row,
  /// before any output buffer is written. Use [`AveragingDomain::Encoded`] (or
  /// [`AveragingDomain::Linear`] under the `rgb` feature) on these formats.
  #[error(
    "the Premultiplied averaging domain is unsupported for this format \
     (output {}x{} from source {}x{}): it has no alpha channel; \
     use AveragingDomain::Encoded",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  PremultipliedDomainUnsupported(PlanGeometry),
}

#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
mod cv2_goldens;
#[cfg(all(
  test,
  feature = "std",
  any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  )
))]
mod pil_goldens;
#[cfg(all(test, feature = "std"))]
mod tests;
