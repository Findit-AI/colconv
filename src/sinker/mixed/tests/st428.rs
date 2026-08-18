//! SMPTE ST 428-1 CIE-XYZ interpretation toggle (#310).
//!
//! Covers the [`St428Interpretation`] selector (default, predicates, `as_str`,
//! builder / setter roundtrip), the [`st428_chroma_derived_guard`] branch
//! (default derives, CIE-XYZ rejects `ChromaDerivedNcl` over `SmpteSt428`,
//! other primaries / matrices unaffected), and the end-to-end `Yuv420p` sink
//! wiring (rejection surfaces as [`MixedSinkerError::St428CieXyzUnsupported`];
//! the default mode is byte-identical to a sink built without the toggle).

use super::*;
use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer,
};

// ---- the selector enum -----------------------------------------------------

/// The toggle defaults to the FFmpeg-tabulated interpretation, and its
/// predicates / `as_str` agree with the variant.
#[test]
fn interpretation_default_and_projections() {
  assert_eq!(
    St428Interpretation::default(),
    St428Interpretation::FfmpegTabulated
  );

  assert!(St428Interpretation::FfmpegTabulated.is_ffmpeg_tabulated());
  assert!(!St428Interpretation::FfmpegTabulated.is_cie_xyz());
  assert_eq!(
    St428Interpretation::FfmpegTabulated.as_str(),
    "ffmpeg-tabulated"
  );

  assert!(St428Interpretation::CieXyz.is_cie_xyz());
  assert!(!St428Interpretation::CieXyz.is_ffmpeg_tabulated());
  assert_eq!(St428Interpretation::CieXyz.as_str(), "cie-xyz");
}

/// A freshly built sink starts FFmpeg-tabulated; `with_*` / `set_*` roundtrip
/// the getter both ways.
#[test]
fn sink_toggle_roundtrips() {
  let mut rgb = std::vec![0u8; 4 * 2 * 3];
  let sink = MixedSinker::<Yuv420p>::new(4, 2)
    .with_rgb(&mut rgb)
    .unwrap();
  assert_eq!(
    sink.st428_interpretation(),
    St428Interpretation::FfmpegTabulated
  );

  let sink = sink.with_st428_interpretation(St428Interpretation::CieXyz);
  assert_eq!(sink.st428_interpretation(), St428Interpretation::CieXyz);

  let mut sink = sink;
  sink.set_st428_interpretation(St428Interpretation::FfmpegTabulated);
  assert_eq!(
    sink.st428_interpretation(),
    St428Interpretation::FfmpegTabulated
  );
}

// ---- the derivation-gating guard -------------------------------------------

/// The default FFmpeg-tabulated mode never rejects — every matrix / primary
/// combination is permitted, so the tabulated derivation runs unchanged.
#[test]
fn guard_ffmpeg_tabulated_permits_everything() {
  for matrix in [
    KernelMatrix::ChromaDerivedNcl,
    KernelMatrix::Bt709,
    KernelMatrix::Bt601,
  ] {
    for primaries in [Primaries::SmpteSt428, Primaries::Bt2020, Primaries::Bt709] {
      assert!(
        st428_chroma_derived_guard(matrix, &primaries, St428Interpretation::FfmpegTabulated)
          .is_ok(),
        "FfmpegTabulated must permit ({matrix:?}, {primaries:?})"
      );
    }
  }
}

/// CIE-XYZ mode rejects exactly the meaningless combination:
/// `ChromaDerivedNcl` over the CIE-XYZ `SmpteSt428` primaries.
#[test]
fn guard_cie_xyz_rejects_chroma_derived_over_smpte_st428() {
  let err = st428_chroma_derived_guard(
    KernelMatrix::ChromaDerivedNcl,
    &Primaries::SmpteSt428,
    St428Interpretation::CieXyz,
  )
  .expect_err("ChromaDerivedNcl over SmpteSt428 must be rejected in CIE-XYZ mode");
  // Round-trips through the sink error variant.
  let mapped: MixedSinkerError = MixedSinkerError::St428CieXyzUnsupported(err);
  assert!(mapped.is_st_428_cie_xyz_unsupported());
}

/// CIE-XYZ mode leaves every other combination alone: non-ST 428-1 primaries
/// (even with `ChromaDerivedNcl`) and non-`ChromaDerivedNcl` matrices (even
/// over `SmpteSt428`) still derive as tabulated.
#[test]
fn guard_cie_xyz_ignores_non_st428_and_fixed_matrices() {
  // Non-ST 428-1 primaries under ChromaDerivedNcl: unaffected.
  for primaries in [Primaries::Bt2020, Primaries::Bt709, Primaries::Smpte170M] {
    assert!(
      st428_chroma_derived_guard(
        KernelMatrix::ChromaDerivedNcl,
        &primaries,
        St428Interpretation::CieXyz
      )
      .is_ok(),
      "CIE-XYZ must not affect ChromaDerivedNcl over {primaries:?}"
    );
  }
  // A fixed matrix over SmpteSt428 does not derive from the primaries at all.
  for matrix in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
  ] {
    assert!(
      st428_chroma_derived_guard(matrix, &Primaries::SmpteSt428, St428Interpretation::CieXyz)
        .is_ok(),
      "CIE-XYZ must not affect the fixed {matrix:?} over SmpteSt428"
    );
  }
}

/// Only `SmpteSt428` is treated as CIE XYZ (mediaframe's `is_cie_xyz`, which
/// the ST 428-1 guard consumes).
#[test]
fn cie_xyz_predicate_is_smpte_st428_only() {
  assert!(Primaries::SmpteSt428.is_cie_xyz());
  for primaries in [
    Primaries::Bt709,
    Primaries::Bt2020,
    Primaries::Smpte170M,
    Primaries::SmpteRp431,
    Primaries::Unspecified,
  ] {
    assert!(!primaries.is_cie_xyz(), "{primaries:?} is not CIE XYZ");
  }
}

// ---- end-to-end sink wiring ------------------------------------------------

/// Decodes a solid 4×2 `Yuv420p` frame to packed RGB with the given matrix /
/// primaries / interpretation, returning the sink's `process` result.
fn decode(
  matrix: KernelMatrix,
  primaries: Primaries,
  interp: St428Interpretation,
) -> Result<std::vec::Vec<u8>, MixedSinkerError> {
  let (w, h) = (4u32, 2u32);
  let n = (w * h) as usize;
  let cn = ((w / 2) * (h / 2)) as usize;
  let y = std::vec![128u8; n];
  let u = std::vec![110u8; cn];
  let v = std::vec![140u8; cn];
  let src = Yuv420pFrame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
  let mut rgb = std::vec![0u8; n * 3];
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv420p,
    ColorInfo::new(
      primaries,
      Transfer::Unspecified,
      crate::ColorMatrix::from(matrix),
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  );
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(&spec)
      .with_st428_interpretation(interp);
    yuv420p_to(&src, false, matrix, &mut sink)?;
  }
  Ok(rgb)
}

/// CIE-XYZ mode: a `ChromaDerivedNcl` + `SmpteSt428` decode is rejected
/// end-to-end with the typed error.
#[test]
fn cie_xyz_rejects_chroma_derived_smpte_st428_end_to_end() {
  let err = decode(
    KernelMatrix::ChromaDerivedNcl,
    Primaries::SmpteSt428,
    St428Interpretation::CieXyz,
  )
  .expect_err("CIE-XYZ ChromaDerivedNcl over SmpteSt428 must be rejected");
  assert!(
    matches!(err, MixedSinkerError::St428CieXyzUnsupported(_)),
    "unexpected error variant: {err:?}"
  );
}

/// FFmpeg-tabulated (default) mode: the same `ChromaDerivedNcl` + `SmpteSt428`
/// decode succeeds and is byte-identical to a sink built without ever touching
/// the toggle — the default is a no-op on existing callers.
#[test]
fn ffmpeg_tabulated_is_byte_identical_to_untoggled() {
  let toggled = decode(
    KernelMatrix::ChromaDerivedNcl,
    Primaries::SmpteSt428,
    St428Interpretation::FfmpegTabulated,
  )
  .expect("FFmpeg-tabulated ChromaDerivedNcl over SmpteSt428 decodes");

  // A sink that never calls `with_st428_interpretation` — the pre-#310 path.
  let untoggled = {
    let (w, h) = (4u32, 2u32);
    let n = (w * h) as usize;
    let cn = ((w / 2) * (h / 2)) as usize;
    let y = std::vec![128u8; n];
    let u = std::vec![110u8; cn];
    let v = std::vec![140u8; cn];
    let src = Yuv420pFrame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
    let mut rgb = std::vec![0u8; n * 3];
    let spec = ColorSpec::from_info(
      PixelFormat::Yuv420p,
      ColorInfo::new(
        Primaries::SmpteSt428,
        Transfer::Unspecified,
        ColorMatrix::ChromaDerivedNcl,
        DynamicRange::Limited,
        ChromaLocation::Left,
      ),
    );
    {
      let mut sink = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_color_spec(&spec);
      yuv420p_to(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
    }
    rgb
  };

  assert_eq!(
    toggled, untoggled,
    "the default toggle must not perturb output"
  );
}

/// The toggle has no effect on non-ST 428-1 primaries: a `ChromaDerivedNcl`
/// decode over BT.2020 is byte-identical in both modes.
#[test]
fn cie_xyz_is_a_no_op_for_non_st428_primaries() {
  let tabulated = decode(
    KernelMatrix::ChromaDerivedNcl,
    Primaries::Bt2020,
    St428Interpretation::FfmpegTabulated,
  )
  .expect("BT.2020 ChromaDerivedNcl decodes");
  let cie_xyz = decode(
    KernelMatrix::ChromaDerivedNcl,
    Primaries::Bt2020,
    St428Interpretation::CieXyz,
  )
  .expect("BT.2020 is unaffected by the CIE-XYZ interpretation");
  assert_eq!(
    tabulated, cie_xyz,
    "non-ST 428-1 primaries derive identically in both modes"
  );
}

// ---- Top forward-delay + mid-frame colorimetry switch (#310) ----------------
//
// The 4:2:0 `Top` (`v = 0`) identity decode HOLDS each odd colour row until the
// following even row (the forward one-row delay), decoding it against the
// primaries frozen when it was SUBMITTED — not the sink's current state. These
// regressions drive that path directly through `PixelSink::process` so a
// mid-frame `set_color_spec` cannot slip a `ChromaDerivedNcl` derivation onto
// the switched colorimetry (the guard-bypass the forward delay would otherwise
// open).

const TOP_W: u32 = 4;
const TOP_H: u32 = 4;

/// A flat-luma `Yuv420p` frame with a strong per-chroma-row step, so BOTH the
/// `Top` vertical blend and the `ChromaDerivedNcl` primaries derivation are
/// observable (a neutral `128` chroma would decode identically under every
/// primary set).
fn top_frame() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let (w, h) = (TOP_W as usize, TOP_H as usize);
  let (cw, ch) = (w / 2, h / 2);
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (24 + r * 48).min(240) as u8;
      v[r * cw + c] = (216 - r * 48).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Builds `Yuv420p` row `r` of [`top_frame`] with the given per-row `matrix`
/// for direct `process` feeding (chroma row `r / 2`, the walker's vertical
/// replication).
fn top_row<'a>(
  yp: &'a [u8],
  up: &'a [u8],
  vp: &'a [u8],
  r: usize,
  matrix: KernelMatrix,
) -> Yuv420pRow<'a> {
  let w = TOP_W as usize;
  let cw = w / 2;
  let cr = r / 2;
  Yuv420pRow::new(
    &yp[r * w..r * w + w],
    &up[cr * cw..cr * cw + cw],
    &vp[cr * cw..cr * cw + cw],
    r,
    matrix,
    false,
  )
}

/// A `Top`-sited `ChromaDerivedNcl` `ColorSpec` over `primaries`.
fn top_spec(primaries: Primaries) -> ColorSpec {
  ColorSpec::from_info(
    PixelFormat::Yuv420p,
    ColorInfo::new(
      primaries,
      Transfer::Unspecified,
      ColorMatrix::ChromaDerivedNcl,
      DynamicRange::Limited,
      ChromaLocation::Top,
    ),
  )
}

/// A full in-order `Top` `ChromaDerivedNcl` decode over `primaries` (via the
/// walker), the byte-exact reference an unswitched forward-delay decode must
/// reproduce.
fn top_walker_decode(primaries: Primaries, interp: St428Interpretation) -> std::vec::Vec<u8> {
  let (yp, up, vp) = top_frame();
  let (w, h) = (TOP_W as usize, TOP_H as usize);
  let src = Yuv420pFrame::new(&yp, &up, &vp, TOP_W, TOP_H, TOP_W, TOP_W / 2, TOP_W / 2);
  let mut rgb = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(&top_spec(primaries))
      .with_st428_interpretation(interp);
    yuv420p_to(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
  }
  rgb
}

/// The default (FFmpeg-tabulated) `Top` forward-delay path is unperturbed by the
/// frozen-colorimetry bookkeeping: a direct `process` decode of an all-
/// `ChromaDerivedNcl` frame (holding + flushing the odd rows) is byte-identical
/// to the walker, for both a non-ST 428-1 primary (`Bt709`) and `SmpteSt428`.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_delay_ncl_direct_matches_walker() {
  let (yp, up, vp) = top_frame();
  let (w, h) = (TOP_W as usize, TOP_H as usize);
  for primaries in [Primaries::Bt709, Primaries::SmpteSt428] {
    let reference = top_walker_decode(primaries.clone(), St428Interpretation::FfmpegTabulated);
    let mut rgb = std::vec![0u8; w * h * 3];
    {
      let mut sink = MixedSinker::<Yuv420p>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_color_spec(&top_spec(primaries.clone()));
      crate::PixelSink::begin_frame(&mut sink, TOP_W, TOP_H).unwrap();
      for r in 0..h {
        crate::PixelSink::process(
          &mut sink,
          top_row(&yp, &up, &vp, r, KernelMatrix::ChromaDerivedNcl),
        )
        .unwrap();
      }
    }
    assert_eq!(
      rgb, reference,
      "{primaries:?}: the Top forward-delay decode must match the in-order walker"
    );
  }
}

/// The forward delay HOLDS an odd `ChromaDerivedNcl` row against the colorimetry
/// active at SUBMIT. Switching the sink to `SmpteSt428` + `CieXyz` mid-frame
/// (the combination whose tabulated derivation the guard forbids) must NOT
/// change how the already-accepted row decodes: it flushes through its FROZEN
/// (`Bt709`, non-XYZ) primaries, so no silent ST 428 derivation runs and the
/// output matches a clean unswitched `Bt709` decode.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_delay_freezes_submit_colorimetry_across_cie_xyz_switch() {
  let (yp, up, vp) = top_frame();
  let (w, h) = (TOP_W as usize, TOP_H as usize);

  // The frozen (Bt709) and switched (SmpteSt428) primaries must derive
  // observably-different RGB, else the assertion below would pass vacuously.
  let bt709_ref = top_walker_decode(Primaries::Bt709, St428Interpretation::CieXyz);
  let smpte_ref = top_walker_decode(Primaries::SmpteSt428, St428Interpretation::FfmpegTabulated);
  assert_ne!(
    bt709_ref, smpte_ref,
    "Bt709 vs SmpteSt428 ChromaDerivedNcl must differ for this chroma"
  );

  // Subject: rows 0/1 submitted with Bt709 + CieXyz (permitted — Bt709 is not
  // CIE XYZ). Row 1 (odd) is DEFERRED, freezing (Bt709, CieXyz).
  let mut rgb = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(&top_spec(Primaries::Bt709))
      .with_st428_interpretation(St428Interpretation::CieXyz);
    crate::PixelSink::begin_frame(&mut sink, TOP_W, TOP_H).unwrap();
    crate::PixelSink::process(
      &mut sink,
      top_row(&yp, &up, &vp, 0, KernelMatrix::ChromaDerivedNcl),
    )
    .unwrap();
    crate::PixelSink::process(
      &mut sink,
      top_row(&yp, &up, &vp, 1, KernelMatrix::ChromaDerivedNcl),
    )
    .unwrap();
    assert!(
      sink.chroma_top_pending.is_some(),
      "the odd ChromaDerivedNcl row must be held pending its even-row flush"
    );

    // Flip to SmpteSt428 (CIE-XYZ interpretation unchanged) — the SAME Top siting,
    // so the phase freeze holds. The now-active (SmpteSt428, CieXyz) would reject
    // a fresh ChromaDerivedNcl row, but must not reach back to the held one.
    sink.set_color_spec(&top_spec(Primaries::SmpteSt428));
    assert_eq!(sink.primaries(), Primaries::SmpteSt428);

    // The even flush row carries a FIXED matrix, which the current-row guard
    // permits even under (SmpteSt428, CieXyz); it flushes the deferred odd row.
    crate::PixelSink::process(&mut sink, top_row(&yp, &up, &vp, 2, KernelMatrix::Bt601))
      .expect("the fixed-matrix even row is permitted and flushes the held row");
    crate::PixelSink::process(&mut sink, top_row(&yp, &up, &vp, 3, KernelMatrix::Bt601))
      .expect("the trailing odd row clamps co-sited");
  }

  // Reference: the identical row program with Bt709 held throughout (no switch).
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap()
      .with_color_spec(&top_spec(Primaries::Bt709))
      .with_st428_interpretation(St428Interpretation::CieXyz);
    crate::PixelSink::begin_frame(&mut sink, TOP_W, TOP_H).unwrap();
    crate::PixelSink::process(
      &mut sink,
      top_row(&yp, &up, &vp, 0, KernelMatrix::ChromaDerivedNcl),
    )
    .unwrap();
    crate::PixelSink::process(
      &mut sink,
      top_row(&yp, &up, &vp, 1, KernelMatrix::ChromaDerivedNcl),
    )
    .unwrap();
    crate::PixelSink::process(&mut sink, top_row(&yp, &up, &vp, 2, KernelMatrix::Bt601)).unwrap();
    crate::PixelSink::process(&mut sink, top_row(&yp, &up, &vp, 3, KernelMatrix::Bt601)).unwrap();
  }

  assert_eq!(
    rgb, rgb_ref,
    "the held odd row must decode through its FROZEN Bt709 primaries, not the switched SmpteSt428"
  );
}
