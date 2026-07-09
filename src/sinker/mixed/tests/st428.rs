//! SMPTE ST 428-1 CIE-XYZ interpretation toggle (#310).
//!
//! Covers the [`St428Interpretation`] selector (default, predicates, `as_str`,
//! builder / setter roundtrip), the [`st428_chroma_derived_guard`] branch
//! (default derives, CIE-XYZ rejects `ChromaDerivedNcl` over `SmpteSt428`,
//! other primaries / matrices unaffected), and the end-to-end `Yuv420p` sink
//! wiring (rejection surfaces as [`MixedSinkerError::St428CieXyzUnsupported`];
//! the default mode is byte-identical to a sink built without the toggle).

use super::*;
use crate::{ChromaLocation, ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

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
    ColorMatrix::ChromaDerivedNcl,
    ColorMatrix::Bt709,
    ColorMatrix::Bt601,
  ] {
    for primaries in [Primaries::SmpteSt428, Primaries::Bt2020, Primaries::Bt709] {
      assert!(
        st428_chroma_derived_guard(matrix, primaries, St428Interpretation::FfmpegTabulated).is_ok(),
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
    ColorMatrix::ChromaDerivedNcl,
    Primaries::SmpteSt428,
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
        ColorMatrix::ChromaDerivedNcl,
        primaries,
        St428Interpretation::CieXyz
      )
      .is_ok(),
      "CIE-XYZ must not affect ChromaDerivedNcl over {primaries:?}"
    );
  }
  // A fixed matrix over SmpteSt428 does not derive from the primaries at all.
  for matrix in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
  ] {
    assert!(
      st428_chroma_derived_guard(matrix, Primaries::SmpteSt428, St428Interpretation::CieXyz)
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
  matrix: ColorMatrix,
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
      matrix,
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  );
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(spec)
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
    ColorMatrix::ChromaDerivedNcl,
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
    ColorMatrix::ChromaDerivedNcl,
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
        .with_color_spec(spec);
      yuv420p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
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
    ColorMatrix::ChromaDerivedNcl,
    Primaries::Bt2020,
    St428Interpretation::FfmpegTabulated,
  )
  .expect("BT.2020 ChromaDerivedNcl decodes");
  let cie_xyz = decode(
    ColorMatrix::ChromaDerivedNcl,
    Primaries::Bt2020,
    St428Interpretation::CieXyz,
  )
  .expect("BT.2020 is unaffected by the CIE-XYZ interpretation");
  assert_eq!(
    tabulated, cie_xyz,
    "non-ST 428-1 primaries derive identically in both modes"
  );
}
