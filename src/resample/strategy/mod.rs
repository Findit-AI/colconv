//! RFC #238 staged-resampling-pipeline framework — Phase 0.
//!
//! The RFC models the convert→resample path as a *staged pipeline* and
//! treats a resample as a **splice** into that pipeline at a chosen
//! stage. The colour domain an area downscale averages in then falls out
//! of *where* the splice lands: averaging the native codes before the
//! convert is a different result from converting first and averaging the
//! output. This module is the production foundation those splice stages
//! and the domain choice are expressed through; the design was validated
//! end-to-end against one format by the held PoC (see the
//! `AveragingDomain` history note below).
//!
//! Phase 0 introduces the framework **types** plus an insertion-point
//! [`select_insertion_point`] **selector**, and re-expresses the existing
//! native-vs-row-stage dispatch of `Yuv420p` (4:2:0 planar) through that
//! selector with **zero behaviour change** — the selector reproduces
//! today's tier choice bit-for-bit. Only [`AveragingDomain::Encoded`] and
//! its two splice stages ([`InsertionPoint::NativeCodes`] /
//! [`InsertionPoint::EncodedOutput`]) are exercised; the [`Linear`] and
//! [`Premultiplied`] domains, the full per-plane filter spec, and the
//! support policy are later phases.
//!
//! [`Linear`]: AveragingDomain::Linear
//! [`Premultiplied`]: AveragingDomain::Premultiplied

#[cfg(test)]
mod tests;

pub(crate) mod transfer;
pub use transfer::TransferFunction;

/// The colour domain an RFC #238 area downscale averages in.
///
/// Each variant names the colour space the box-average is taken in,
/// which is equivalently the pipeline stage the resample is spliced at
/// (see the [module docs](self)). Because a YUV→RGB convert is affine,
/// the domains land at materially different RGB, so the choice is
/// observable — that is the reason it is offered.
///
/// Phase 0 only exercises [`Self::Encoded`]; [`Self::Linear`] and
/// [`Self::Premultiplied`] are reserved for later phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AveragingDomain {
  /// Average the **encoded codes** before conversion: bin the native
  /// (e.g. Y / U / V) samples, then run the convert once per output
  /// pixel. The fused, libswscale-class semantics and the default.
  #[default]
  Encoded,
  /// Average in **linear light**: decode to RGB, linearise via the
  /// inverse transfer function ([`TransferFunction::eotf`]), bin the
  /// linear RGB, then re-encode ([`TransferFunction::oetf`]) per output
  /// pixel. The physically-correct light-mixing domain; splices at the
  /// crate-internal `InsertionPoint::LinearLight` stage. Wired for the
  /// planar 8-bit YUV family (Yuv420p / Yuv422p / Yuv444p / Yuv440p) as of
  /// Phase 2.
  Linear,
  /// Average **premultiplied** RGBA: convert at source resolution,
  /// premultiply by α, bin, then un-premultiply per output row. Reserved
  /// for a later phase.
  Premultiplied,
}

impl AveragingDomain {
  /// Lowercase identifier for the domain (`"encoded"` / `"linear"` /
  /// `"premultiplied"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Encoded => "encoded",
      Self::Linear => "linear",
      Self::Premultiplied => "premultiplied",
    }
  }
}

/// How the [`AveragingDomain::Linear`] tail decodes `YUV→RGB` before it
/// lifts the result to linear light — the *referent* of the linear-light
/// average (RFC #238 #244).
///
/// Both modes share the rest of the linear pipeline (decode → EOTF → area
/// bin → OETF → clamp at output); they differ only in *which* `YUV→RGB`
/// decode fills the buffer the EOTF lifts:
///
/// - [`Self::DisplayReferred`] (the **default**) decodes through the
///   production Q15 `yuv_*_to_rgb_row` kernel, which **clamps and quantizes**
///   the result to 8-bit `[0, 255]` before the EOTF. The average is then
///   *display-referred*: it mixes the converted in-gamut 8-bit RGB in linear
///   light (a gamma-correct resize). Out-of-gamut excursions — super-black /
///   super-white luma, or chroma that drives a channel past the `[0, 1]`
///   cube — are discarded at the convert clamp.
/// - [`Self::SceneReferred`] decodes the **same affine matrix** in unclamped
///   `f32`, preserving those out-of-gamut excursions, lifts *that* to linear
///   light via the EOTF (whose odd-symmetric extrapolation handles
///   out-of-`[0, 1]` inputs), averages in linear light, and clamps **only**
///   at the re-encoded output. This is the physically faithful average for
///   content with super-black / super-white or saturated chroma.
///
/// The two coincide (modulo `f32` rounding) on content that stays in gamut
/// through the decode — they diverge exactly where the 8-bit convert clamp
/// would otherwise have thrown information away.
///
/// The mode is consulted only on the [`AveragingDomain::Linear`] path of the
/// planar 8-bit YUV family; the encoded and direct paths ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LinearMode {
  /// Decode through the clamped/quantized 8-bit convert, then average the
  /// in-gamut display-referred RGB in linear light. The default; the
  /// behaviour RFC #238 Phase 2 shipped.
  #[default]
  DisplayReferred,
  /// Decode the same affine matrix in unclamped `f32` (preserving
  /// out-of-gamut excursions), average in linear light, clamp only at the
  /// output. The scene-referred upgrade (RFC #238 #244).
  SceneReferred,
}

impl LinearMode {
  /// Lowercase identifier for the mode
  /// (`"display-referred"` / `"scene-referred"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::DisplayReferred => "display-referred",
      Self::SceneReferred => "scene-referred",
    }
  }
}

/// A pipeline splice stage — *where* in the convert pipeline an RFC #238
/// resample is inserted.
///
/// Phase 0 enumerates the two stages the [`AveragingDomain::Encoded`]
/// domain splices at; the linear-light and premultiplied stages arrive
/// with their domains in later phases. The [`select_insertion_point`]
/// selector maps a resample's eligibility and plan onto one of these.
#[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum InsertionPoint {
  /// Splice at the **native codes**, before the convert: bin the source
  /// codes at output resolution, then convert once per output row. The
  /// native fast tier (e.g. `yuv420p_process_native`).
  NativeCodes,
  /// Splice at the **encoded output**, after the convert: convert each
  /// source row, then area-stream the encoded output rows. The row-stage
  /// tier (e.g. `yuv420p_process_resampled`).
  EncodedOutput,
  /// Splice at **linear light**, between the convert and the re-encode:
  /// decode each source pixel to RGB, linearise via the
  /// [`TransferFunction`] EOTF, area-stream the linear RGB, then re-encode
  /// per output pixel via the OETF. The [`AveragingDomain::Linear`] stage
  /// (e.g. the planar 8-bit YUV linear-light tail).
  LinearLight,
}

/// Inputs to [`select_insertion_point`] — the facts that determine which
/// pipeline stage a resample splices at.
///
/// These mirror exactly the values today's per-format dispatch already
/// branches on, so the selector can reproduce the current choice without
/// any new information: a format's static eligibility for a native tier,
/// the resample plan's [area-vs-filter](crate::resample::SpanKind) kind,
/// and the sink's `with_native` request.
#[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InsertionContext {
  /// Whether this format ships a native fast tier at all. Formats with
  /// no averageable native path (e.g. palette / Bayer) are never
  /// eligible and always splice at the output.
  pub native_eligible: bool,
  /// Whether the sink was built with the native fast tier enabled
  /// (`with_native`). `false` forces the output splice.
  pub with_native: bool,
  /// Whether the resample plan carries an area (box-coverage) span. The
  /// native tier is an area-only optimization; a filter plan never
  /// splices at the native codes.
  pub area_plan: bool,
}

/// Selects the pipeline splice stage for a resample, given its
/// [domain](AveragingDomain) and [context](InsertionContext).
///
/// Phase 0 resolves only [`AveragingDomain::Encoded`]: the resample
/// splices at the [native codes](InsertionPoint::NativeCodes) when the
/// format is native-eligible, the sink enabled the native tier, and the
/// plan is an area downscale; otherwise it splices at the
/// [encoded output](InsertionPoint::EncodedOutput). This reproduces the
/// existing `Yuv420p` native-vs-row-stage decision exactly — see
/// [`crate::sinker::MixedSinker`]'s `Yuv420p` `process` dispatch.
///
/// As of Phase 2 the [`Linear`] domain resolves to the
/// [linear-light](InsertionPoint::LinearLight) stage (decode → linearise →
/// bin → re-encode), independent of the native-tier inputs (the linear
/// average is its own splice, not a native fast-tier variant).
///
/// # Premultiplied is rejected upstream, never resolved here
///
/// [`Premultiplied`] is a **reserved future-phase** domain with *no valid
/// insertion point yet*, so the selector deliberately does **not** map it to
/// any splice — in particular it must not silently resolve to the
/// [encoded output](InsertionPoint::EncodedOutput), which is a different
/// domain (`Premultiplied` is not `Encoded`; for the Phase 5 alpha formats
/// they diverge). Every caller rejects `Premultiplied` with a typed error in
/// its own exhaustive `match *averaging_domain` *before* reaching this
/// selector (see `MixedSinker`'s per-format `process` dispatch, which only
/// ever calls the selector with [`AveragingDomain::Encoded`]). Reaching the
/// `Premultiplied` arm therefore signals a routing bug — a dispatch that
/// failed to reject it — so the arm is an [`unreachable!`] rather than a
/// silent legitimate route. Phase 5 will give `Premultiplied` its real
/// insertion point and replace the guard.
///
/// [`Linear`]: AveragingDomain::Linear
/// [`Premultiplied`]: AveragingDomain::Premultiplied
#[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn select_insertion_point(
  domain: AveragingDomain,
  ctx: InsertionContext,
) -> InsertionPoint {
  match domain {
    AveragingDomain::Encoded => {
      if ctx.native_eligible && ctx.with_native && ctx.area_plan {
        InsertionPoint::NativeCodes
      } else {
        InsertionPoint::EncodedOutput
      }
    }
    // The linear-light average is its own splice stage, not a native
    // fast-tier variant; it ignores the native-tier inputs.
    AveragingDomain::Linear => InsertionPoint::LinearLight,
    // Reserved for a later phase with no insertion point yet; callers reject
    // it before the selector (see the doc above), so reaching here is a
    // routing bug — NOT a silent downgrade to the encoded output. Phase 5
    // adds its splice. (`panic!` with a `&'static str`, not `unreachable!`,
    // because this is a `const fn` and the latter formats its message.)
    AveragingDomain::Premultiplied => {
      panic!("Premultiplied is rejected at dispatch; the selector has no splice for it yet")
    }
  }
}

/// An RFC #238 resampling strategy: the [averaging domain](AveragingDomain)
/// and (in later phases) the filter specification it resamples with.
///
/// Phase 0 carries only the domain plus a minimal filter placeholder; the
/// real per-plane filter spec and support policy are later phases. The
/// [default](Default) is [`AveragingDomain::Encoded`] with the
/// [area](FilterSpec::Area) filter — the current behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ResampleStrategy {
  domain: AveragingDomain,
  filter: FilterSpec,
}

/// Minimal filter placeholder for [`ResampleStrategy`] — Phase 0 scope.
///
/// Distinguishes only the box-average area path from a (to be specified)
/// windowed-filter kernel; the real per-plane kernel parameters and
/// support policy are later phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterSpec {
  /// Exact integer area (box-coverage) averaging — the Phase 0 default.
  #[default]
  Area,
  /// A windowed-filter resample. The concrete kernel spec is a later
  /// phase; this variant only reserves the distinction.
  Kernel,
}

impl ResampleStrategy {
  /// The averaging domain this strategy resamples in.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn domain(&self) -> AveragingDomain {
    self.domain
  }

  /// The filter specification this strategy resamples with.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn filter(&self) -> FilterSpec {
    self.filter
  }
}
