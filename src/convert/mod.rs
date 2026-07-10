//! Tier-0 [`Convert`] — the golden one-call decode entry point.
//!
//! [`Convert`] is the top of colconv's tier stack (issue
//! [#392](https://github.com/findit-studio/colconv/issues/392)): a single
//! generic builder that decodes a validated source frame to one or more
//! output buffers with zero redundant parameters. Dimensions come from the
//! frame, colorimetry comes from a single [`ColorSpec`], and the per-format
//! walk knobs are *derived* from that spec via [`FromSpec`] (overridable).
//! Every setter is infallible; [`Convert::run`] is the only fallible call.
//!
//! ```text
//! Convert::from(&frame).spec(spec).resize(w, h).rgb(&mut rgb).run()?  // Tier 0 — this module
//! MixedSinker::new(w, h).with_rgb(&mut rgb)?; yuv420p_to(&frame, …)?  // Tier 1 — streaming / custom
//! ```
//!
//! Tier 0 is purely additive: it assembles the mid-tier
//! [`MixedSinker`] internally and forwards to the
//! same per-format walkers, so its output is byte-identical to hand-assembling
//! the sink. The mid tier stays the supported home for custom sinks, partial
//! feeding, and bespoke geometry.
//!
//! The [`Source`] contract between frames and [`Convert`] is *sealed*: it is
//! implemented, per source format, by the same `walker!` table that drives the
//! [`Walker`](crate::Walker) tier, so the two can never drift.

#[cfg(all(test, feature = "yuv-planar"))]
mod tests;

use crate::{
  SourceFormat,
  resample::{AreaResampler, FilterKernel, FilteredResampler, NoopResampler, Resampler},
  sinker::{MixedSinker, MixedSinkerError},
  walker::{ColorSpec, Xyz12Options, YuvOptions},
};

// `ChromaLocation` and `St428Interpretation` feed only the `yuv-planar`
// sink-knob setters and their fields, so their imports must ride the same gate
// or they warn `unused_imports` under the other feature powersets (CI builds
// those with `-Dwarnings`).
#[cfg(feature = "yuv-planar")]
use crate::{ChromaLocation, sinker::St428Interpretation};

/// Seals [`Source`] and [`FromSpec`]: only the crate's own types implement
/// them, so neither the frame ↔ [`Convert`] contract nor the set of
/// spec-derivable walk options is user-extensible.
pub(crate) mod sealed {
  /// The sealing supertrait of [`Source`](super::Source).
  pub trait Sealed {}

  /// The sealing supertrait of [`FromSpec`](super::FromSpec).
  pub trait SealedOptions {}
}

/// Derives a source family's walk [`Options`](Source::Options) from the single
/// colorimetric truth of a [`ColorSpec`].
///
/// This is what lets [`Convert::spec`] seed the per-format knobs from the same
/// spec that configures the sink, so the matrix and range are never passed
/// twice. The [`Default`] supertrait supplies the no-spec fallback (the
/// unspecified-colorimetry defaults each `Options` type already defines).
///
/// Not every `Options` type can implement this: a value that is *frame
/// intrinsic* rather than colorimetric — the Bayer mosaic pattern of
/// [`BayerOptions`](crate::walker::BayerOptions), which has no [`Default`]
/// because it cannot be guessed — is excluded by construction.
///
/// Sealed: only the crate's own walk-option types
/// ([`YuvOptions`], [`Xyz12Options`], and the knob-free `()`) implement it, so
/// the set of spec-derivable options is not user-extensible.
pub trait FromSpec: Default + sealed::SealedOptions {
  /// Builds the walk options carried by `spec`, defaulting any knob the spec
  /// does not describe.
  fn from_spec(spec: &ColorSpec) -> Self;
}

impl sealed::SealedOptions for YuvOptions {}
impl FromSpec for YuvOptions {
  /// The spec's resolved range + matrix become the YUV walk options; the rest
  /// of the spec is sink-consumed, not walker-consumed.
  #[inline]
  fn from_spec(spec: &ColorSpec) -> Self {
    Self::from_color_spec(*spec)
  }
}

impl sealed::SealedOptions for Xyz12Options {}
impl FromSpec for Xyz12Options {
  /// A [`ColorSpec`] carries no target RGB gamut, so the XYZ decode keeps its
  /// default target; the spec's other fields are not consumed by the XYZ path.
  #[inline]
  fn from_spec(_spec: &ColorSpec) -> Self {
    Self::new()
  }
}

impl sealed::SealedOptions for () {}
impl FromSpec for () {
  /// The knob-free sources (a frame-intrinsic palette / conversion) have no
  /// spec-derived state.
  #[inline]
  fn from_spec(_spec: &ColorSpec) -> Self {}
}

/// A validated source-frame borrow that [`Convert`] can decode.
///
/// Implemented, per source format, by the crate's `walker!` table — mapping the
/// frame borrow (`Self`) to its zero-sized [`Marker`](Self::Marker), its walk
/// [`Options`](Self::Options), and the underlying `{fmt}_to` walker. Sealed: the
/// contract is not user-implementable.
pub trait Source: sealed::Sealed {
  /// The zero-sized source-format marker (the `F` on
  /// [`MixedSinker`]).
  type Marker: SourceFormat;

  /// The per-format walk options
  /// ([`YuvOptions`], …), derivable from a
  /// [`ColorSpec`] via [`FromSpec`].
  type Options: FromSpec;

  /// The frame's logical `(width, height)` in pixels.
  fn dimensions(&self) -> (usize, usize);

  /// Walks the frame into an assembled `sink`. Called by [`Convert::run`]
  /// after the sink's outputs and tweaks are attached; delegates to the
  /// format's `{fmt}_to` walker.
  #[doc(hidden)]
  fn walk_mixed<R>(
    &self,
    opts: &Self::Options,
    sink: &mut MixedSinker<'_, Self::Marker, R>,
  ) -> Result<(), MixedSinkerError>
  where
    R: Resampler;

  /// Attaches a packed RGBA output buffer to `sink`. Separate from the
  /// generic-over-format `set_rgb` / `set_luma` because RGBA is a per-format
  /// sink capability; called by [`Convert::run`] only when
  /// [`Convert::rgba`] was set.
  #[doc(hidden)]
  fn attach_rgba<'sink, R>(
    sink: &mut MixedSinker<'sink, Self::Marker, R>,
    buf: &'sink mut [u8],
  ) -> Result<(), MixedSinkerError>
  where
    R: Resampler;
}

/// The golden one-call decode builder — Tier 0 of colconv's API.
///
/// Construct with [`Convert::from`]`(&frame)`, attach outputs and (optional)
/// colorimetry / resize / tuning with the infallible consuming setters, then
/// [`run`](Self::run) exactly once. `R` is the resampling strategy, defaulting
/// to the identity ([`NoopResampler`]); [`resize`](Self::resize) /
/// [`resize_with`](Self::resize_with) transition it.
///
/// ```
/// use colconv::{Convert, ColorMatrix, ColorSpec, DynamicRange, PixelFormat};
/// use colconv::frame::Yuv420pFrame;
///
/// let (w, h) = (8u32, 8u32);
/// let y = [16u8; 64];
/// let (u, v) = ([128u8; 16], [128u8; 16]);
/// let frame = Yuv420pFrame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
/// let spec = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, ColorMatrix::Bt709);
///
/// let mut rgb = [0u8; 4 * 4 * 3];
/// let mut luma = [0u8; 4 * 4];
/// Convert::from(&frame)
///     .spec(spec)
///     .resize(4, 4)
///     .rgb(&mut rgb)
///     .luma(&mut luma)
///     .run()?;
/// # Ok::<(), colconv::Error>(())
/// ```
pub struct Convert<'a, Fr: Source, R = NoopResampler> {
  frame: &'a Fr,
  resampler: R,
  rgb: Option<&'a mut [u8]>,
  rgba: Option<&'a mut [u8]>,
  luma: Option<&'a mut [u8]>,
  // HSV, the siting-aware chroma tweaks, the ST 428 toggle, and the native
  // fast-tier flag are all sink knobs gated to the source families whose sink
  // exposes them. PR-A implements [`Source`] only for the planar-YUV proof
  // formats, so these ride the matching family gates; a later PR widens them
  // alongside the table.
  #[cfg(feature = "yuv-planar")]
  hsv: Option<(&'a mut [u8], &'a mut [u8], &'a mut [u8])>,
  spec: Option<ColorSpec>,
  format_options: Option<Fr::Options>,
  simd: bool,
  #[cfg(feature = "yuv-planar")]
  chroma_location: Option<ChromaLocation>,
  #[cfg(feature = "yuv-planar")]
  st428: Option<St428Interpretation>,
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  native: Option<bool>,
}

impl<'a, Fr: Source> From<&'a Fr> for Convert<'a, Fr, NoopResampler> {
  /// Captures the frame; dimensions are read from it at [`run`](Convert::run).
  #[inline]
  fn from(frame: &'a Fr) -> Self {
    Convert {
      frame,
      resampler: NoopResampler,
      rgb: None,
      rgba: None,
      luma: None,
      #[cfg(feature = "yuv-planar")]
      hsv: None,
      spec: None,
      format_options: None,
      simd: true,
      #[cfg(feature = "yuv-planar")]
      chroma_location: None,
      #[cfg(feature = "yuv-planar")]
      st428: None,
      #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
      native: None,
    }
  }
}

impl<'a, Fr: Source, R> Convert<'a, Fr, R> {
  /// Moves every non-resampler field onto a builder carrying a new resampler
  /// strategy — the shared body of [`resize`](Self::resize) /
  /// [`resize_with`](Self::resize_with).
  #[inline]
  fn rebind_resampler<R2>(self, resampler: R2) -> Convert<'a, Fr, R2> {
    Convert {
      frame: self.frame,
      resampler,
      rgb: self.rgb,
      rgba: self.rgba,
      luma: self.luma,
      #[cfg(feature = "yuv-planar")]
      hsv: self.hsv,
      spec: self.spec,
      format_options: self.format_options,
      simd: self.simd,
      #[cfg(feature = "yuv-planar")]
      chroma_location: self.chroma_location,
      #[cfg(feature = "yuv-planar")]
      st428: self.st428,
      #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
      native: self.native,
    }
  }

  /// Sets the single colorimetric truth. Seeds the walk
  /// [`Options`](Source::Options) via [`FromSpec`] (unless overridden by
  /// [`format_options`](Self::format_options)) and configures the sink's
  /// siting-aware decode.
  #[must_use]
  #[inline]
  pub fn spec(mut self, spec: ColorSpec) -> Self {
    self.spec = Some(spec);
    self
  }

  /// Overrides the per-format walk options that [`spec`](Self::spec) would
  /// otherwise derive — the rare case where a knob is known out of band.
  #[must_use]
  #[inline]
  pub fn format_options(mut self, options: Fr::Options) -> Self {
    self.format_options = Some(options);
    self
  }

  /// Attaches a packed 24-bit RGB output buffer
  /// (`out_width * out_height * 3` bytes).
  #[must_use]
  #[inline]
  pub fn rgb(mut self, buf: &'a mut [u8]) -> Self {
    self.rgb = Some(buf);
    self
  }

  /// Attaches a packed 32-bit RGBA output buffer
  /// (`out_width * out_height * 4` bytes). Alpha is opaque when the source
  /// carries none.
  #[must_use]
  #[inline]
  pub fn rgba(mut self, buf: &'a mut [u8]) -> Self {
    self.rgba = Some(buf);
    self
  }

  /// Attaches a single-plane luma output buffer
  /// (`out_width * out_height` bytes).
  #[must_use]
  #[inline]
  pub fn luma(mut self, buf: &'a mut [u8]) -> Self {
    self.luma = Some(buf);
    self
  }

  /// Attaches three HSV output planes (`out_width * out_height` bytes each).
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn hsv(mut self, h: &'a mut [u8], s: &'a mut [u8], v: &'a mut [u8]) -> Self {
    self.hsv = Some((h, s, v));
    self
  }

  /// Sets whether row primitives dispatch to their SIMD backend (default
  /// `true`); pass `false` to force the scalar reference path.
  #[must_use]
  #[inline]
  pub fn simd(mut self, simd: bool) -> Self {
    self.simd = simd;
    self
  }

  /// Pins the source [`ChromaLocation`] for siting-aware 4:2:0 chroma
  /// reconstruction, overriding the value [`spec`](Self::spec) carries.
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn chroma_location(mut self, loc: ChromaLocation) -> Self {
    self.chroma_location = Some(loc);
    self
  }

  /// Selects how [`Primaries::SmpteSt428`](crate::Primaries::SmpteSt428) is
  /// interpreted for the `ChromaDerivedNcl` decode (see
  /// [`St428Interpretation`]).
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn st428_interpretation(mut self, interp: St428Interpretation) -> Self {
    self.st428 = Some(interp);
    self
  }

  /// Toggles the native (source-domain averaging) fast tier for area
  /// downscales (default `true` on the sink).
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[must_use]
  #[inline]
  pub fn native(mut self, native: bool) -> Self {
    self.native = Some(native);
    self
  }
}

impl<'a, Fr: Source> Convert<'a, Fr, NoopResampler> {
  /// Area (box-coverage) downscale to `out_width * out_height` — the
  /// `cv2.INTER_AREA` convention. [`run`](Convert::run) errors on upscale,
  /// matching [`AreaResampler`].
  #[must_use]
  #[inline]
  pub fn resize(self, out_width: usize, out_height: usize) -> Convert<'a, Fr, AreaResampler> {
    self.rebind_resampler(AreaResampler::to(out_width, out_height))
  }

  /// Filtered resample to `out_width * out_height` under `kernel` (any ratio) —
  /// the windowed-filter (PIL) convention. See [`FilteredResampler`].
  #[must_use]
  #[inline]
  pub fn resize_with<K: FilterKernel>(
    self,
    kernel: K,
    out_width: usize,
    out_height: usize,
  ) -> Convert<'a, Fr, FilteredResampler<K>> {
    self.rebind_resampler(FilteredResampler::new(out_width, out_height, kernel))
  }
}

impl<Fr: Source, R: Resampler> Convert<'_, Fr, R> {
  /// Assembles the sink, attaches the requested outputs and tweaks, resolves
  /// the walk options, and decodes the frame. The only fallible call.
  ///
  /// Options resolve as [`format_options`](Self::format_options), else
  /// [`FromSpec::from_spec`] of the [`spec`](Self::spec), else
  /// [`Default`].
  ///
  /// # Errors
  ///
  /// [`MixedSinkerError`] — a rejected resize plan (e.g. an upscale requested
  /// through [`resize`](Self::resize)), a short output buffer, or a walk-time
  /// failure.
  pub fn run(self) -> Result<(), MixedSinkerError> {
    let (w, h) = self.frame.dimensions();
    let mut sink = MixedSinker::<Fr::Marker, R>::with_resampler(w, h, self.resampler)?;

    if let Some(buf) = self.rgb {
      sink.set_rgb(buf)?;
    }
    if let Some(buf) = self.luma {
      sink.set_luma(buf)?;
    }
    #[cfg(feature = "yuv-planar")]
    if let Some((h_plane, s_plane, v_plane)) = self.hsv {
      sink.set_hsv(h_plane, s_plane, v_plane)?;
    }
    if let Some(buf) = self.rgba {
      Fr::attach_rgba(&mut sink, buf)?;
    }

    sink.set_simd(self.simd);
    #[cfg(feature = "yuv-planar")]
    {
      if let Some(spec) = self.spec {
        sink.set_color_spec(spec);
      }
      if let Some(loc) = self.chroma_location {
        sink.set_chroma_location(loc);
      }
      if let Some(interp) = self.st428 {
        sink.set_st428_interpretation(interp);
      }
    }
    #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
    if let Some(native) = self.native {
      sink.set_native(native);
    }

    let options = self
      .format_options
      .or_else(|| self.spec.map(|spec| Fr::Options::from_spec(&spec)))
      .unwrap_or_default();
    self.frame.walk_mixed(&options, &mut sink)
  }
}
