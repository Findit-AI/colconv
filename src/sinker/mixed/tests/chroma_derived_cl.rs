//! End-to-end constant-luminance `YcCbcCrc` (BT.2020 CL, H.273
//! `MatrixCoefficients = 13`, #303) wiring through the `Yuv444p12`
//! `MixedSinker` identity path.
//!
//! Proves the full routing the row-kernel tests cannot: a
//! `ColorMatrix::ChromaDerivedCl` source with **BT.2020 primaries** (delivered
//! via [`MixedSinker::with_color_spec`]) decodes through the non-affine CL
//! kernel; the 10-/12-bit OETF selection (from the transfer) is honoured
//! end-to-end; and non-BT.2020 primaries — or a non-`ChromaDerivedCl` matrix —
//! fall back to the affine path. The expected values are the `colour-science`
//! 0.4.7 references the `row::scalar::cl` tests pin.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer, sinker::MixedSinker,
};

/// Decodes a `w×h` solid-`(yc,cbc,crc)` 12-bit 4:4:4 frame to packed u8 RGB
/// through the `MixedSinker`, with the sink's primaries + transfer set from a
/// `ColorSpec`.
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

fn decode_rgb(
  yc: u16,
  cbc: u16,
  crc: u16,
  full_range: bool,
  matrix: ColorMatrix,
  primaries: Primaries,
  transfer: Transfer,
) -> std::vec::Vec<u8> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (std::vec![le_wire_u16(yc); n], std::vec![le_wire_u16(cbc); n], std::vec![le_wire_u16(crc); n]);
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
  );
  let mut rgb = std::vec![0u8; n * 3];
  let range = if full_range {
    DynamicRange::Full
  } else {
    DynamicRange::Limited
  };
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(primaries, transfer, matrix, range, ChromaLocation::Left),
  );
  {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(spec);
    crate::source::yuv444p12_to(&src, full_range, matrix, &mut sink).unwrap();
  }
  rgb
}

/// `decode_rgb`'s native-depth `u16` sibling.
fn decode_rgb_u16(
  yc: u16,
  cbc: u16,
  crc: u16,
  full_range: bool,
  matrix: ColorMatrix,
  primaries: Primaries,
  transfer: Transfer,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (std::vec![le_wire_u16(yc); n], std::vec![le_wire_u16(cbc); n], std::vec![le_wire_u16(crc); n]);
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
  );
  let mut rgb = std::vec![0u16; n * 3];
  let range = if full_range {
    DynamicRange::Full
  } else {
    DynamicRange::Limited
  };
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(primaries, transfer, matrix, range, ChromaLocation::Left),
  );
  {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_color_spec(spec);
    crate::source::yuv444p12_to(&src, full_range, matrix, &mut sink).unwrap();
  }
  rgb
}

/// Decodes a solid CL frame to native-depth `rgba_u16`. When `also_rgb_u16`
/// the sink **also** attaches `rgb_u16`, forcing the
/// convert-rgb-then-`expand_rgb_u16_to_rgba_u16_row` route instead of the
/// direct RGBA kernel — the two must yield identical `rgba_u16`.
fn decode_rgba_u16(
  yc: u16,
  cbc: u16,
  crc: u16,
  full_range: bool,
  transfer: Transfer,
  also_rgb_u16: bool,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (std::vec![le_wire_u16(yc); n], std::vec![le_wire_u16(cbc); n], std::vec![le_wire_u16(crc); n]);
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
  );
  let range = if full_range {
    DynamicRange::Full
  } else {
    DynamicRange::Limited
  };
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      ColorMatrix::ChromaDerivedCl,
      range,
      ChromaLocation::Left,
    ),
  );
  let mut rgba = std::vec![0u16; n * 4];
  let mut rgb = std::vec![0u16; n * 3];
  if also_rgb_u16 {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgba_u16(&mut rgba)
      .unwrap()
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_color_spec(spec);
    crate::source::yuv444p12_to(&src, full_range, ColorMatrix::ChromaDerivedCl, &mut sink).unwrap();
  } else {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgba_u16(&mut rgba)
      .unwrap()
      .with_color_spec(spec);
    crate::source::yuv444p12_to(&src, full_range, ColorMatrix::ChromaDerivedCl, &mut sink).unwrap();
  }
  rgba
}

fn assert_all_pixels_u8(rgb: &[u8], want: [u8; 3], tol: i32, what: &str) {
  for (px, chunk) in rgb.chunks_exact(3).enumerate() {
    for c in 0..3 {
      assert!(
        (chunk[c] as i32 - want[c] as i32).abs() <= tol,
        "{what}: px{px} ch{c} = {} (want {})",
        chunk[c],
        want[c]
      );
    }
  }
}

fn assert_all_pixels_u16(rgb: &[u16], want: [u16; 3], tol: i32, what: &str) {
  for (px, chunk) in rgb.chunks_exact(3).enumerate() {
    for c in 0..3 {
      assert!(
        chunk[c] <= 4095,
        "{what}: px{px} ch{c} over native 12-bit range"
      );
      assert!(
        (chunk[c] as i32 - want[c] as i32).abs() <= tol,
        "{what}: px{px} ch{c} = {} (want {})",
        chunk[c],
        want[c]
      );
    }
  }
}

/// A `ChromaDerivedCl` source with BT.2020 primaries decodes through the
/// non-affine CL kernel: studio-range 12-bit `(2048, 2148, 1948)` →
/// `colour-science` `R'G'B'` narrow `[118, 134, 142]` (u8), `[1898, 2149,
/// 2275]` (u16, native depth).
#[test]
fn sink_routes_bt2020_chroma_derived_cl_through_non_affine_decode() {
  let rgb = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_all_pixels_u8(&rgb, [118, 134, 142], 1, "CL sink RGB (studio 12-bit)");

  let rgb_u16 = decode_rgb_u16(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_all_pixels_u16(
    &rgb_u16,
    [1898, 2149, 2275],
    2,
    "CL sink u16 (studio 12-bit)",
  );
}

/// Full-range routing differs from studio (the dequant range changes the
/// decode): full-range 12-bit `(2048, 2148, 1948)` → `[117, 131, 137]` (u8).
#[test]
fn sink_full_range_chroma_derived_cl_matches_reference() {
  let rgb = decode_rgb(
    2048,
    2148,
    1948,
    true,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_all_pixels_u8(&rgb, [117, 131, 137], 1, "CL sink RGB (full 12-bit)");
}

/// Gray axis end-to-end: zero chroma with a neutral `Y'c` decodes to a neutral
/// grey that round-trips its native code (`2048 → 2048` full range).
#[test]
fn sink_gray_axis_is_neutral_and_round_trips() {
  let rgb = decode_rgb(
    2048,
    2048,
    2048,
    true,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_all_pixels_u8(&rgb, [128, 128, 128], 0, "CL gray u8");
  let rgb_u16 = decode_rgb_u16(
    2048,
    2048,
    2048,
    true,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_all_pixels_u16(&rgb_u16, [2048, 2048, 2048], 0, "CL gray u16 round-trip");
}

/// The 10-bit and 12-bit OETF constant sets (`Bt2020_10Bit` vs
/// `Bt2020_12Bit`) produce a genuinely different CL decode — the transfer
/// selects `(α, β)`. The difference is sub-LSB at most code points (the
/// `α/β` gap is in the 4th decimal); studio `(293, 2600, 1500)` is a code
/// point where the green channel narrows to a different native code (`482`
/// vs `483`), pinned against the `colour-science` reference.
#[test]
fn ten_bit_and_twelve_bit_oetf_differ_end_to_end() {
  let ten = decode_rgb_u16(
    293,
    2600,
    1500,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_10Bit,
  );
  let twelve = decode_rgb_u16(
    293,
    2600,
    1500,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_ne!(ten, twelve, "10-bit vs 12-bit OETF must decode differently");
}

/// `ChromaDerivedCl` with **non-BT.2020** primaries has no published CL
/// derivation, so it must fall back to the affine path — its output must NOT
/// equal the BT.2020 CL decode.
#[test]
fn chroma_derived_cl_without_bt2020_primaries_falls_back_to_affine() {
  let bt709 = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt709,
    Transfer::Bt2020_12Bit,
  );
  let bt2020 = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_ne!(
    bt709, bt2020,
    "ChromaDerivedCl + non-BT.2020 primaries must fall back to affine (≠ CL decode)"
  );
}

/// BT.2020 primaries but a non-BT.2020 transfer (here PQ) is *not* the CL
/// camera gamma, so the row must NOT decode through CL: it takes the affine
/// (BT.709) fallback — byte-identical to the output a non-BT.2020-primaries
/// source takes, and distinct from the resolved CL decode. Guards the
/// `ClSystem::resolve` allow-list against silently routing an HDR transfer
/// through the BT.2020 camera inverse-OETF.
#[test]
fn chroma_derived_cl_bt2020_pq_transfer_falls_back_to_affine() {
  let pq = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::SmpteSt2084Pq,
  );
  // The affine fallback for ChromaDerivedCl is primaries-independent (BT.709
  // coefficients), so an unresolved BT.709-primaries source is the reference
  // affine output.
  let affine_ref = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt709,
    Transfer::Bt2020_12Bit,
  );
  let cl = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::ChromaDerivedCl,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  assert_eq!(
    pq, affine_ref,
    "BT.2020 + PQ ChromaDerivedCl must take the affine fallback, like a non-BT.2020 source"
  );
  assert_ne!(
    pq, cl,
    "BT.2020 + PQ must NOT decode through the CL camera inverse-OETF"
  );
}

/// A non-`ChromaDerivedCl` matrix must ignore the primaries entirely (no CL
/// routing): BT.709 decodes identically whether the primaries are BT.2020 or
/// not.
#[test]
fn non_chroma_derived_cl_matrix_ignores_primaries() {
  let bt709_with_2020 = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::Bt709,
    Primaries::Bt2020,
    Transfer::Bt2020_12Bit,
  );
  let bt709_with_709 = decode_rgb(
    2048,
    2148,
    1948,
    false,
    ColorMatrix::Bt709,
    Primaries::Bt709,
    Transfer::Bt2020_12Bit,
  );
  assert_eq!(
    bt709_with_2020, bt709_with_709,
    "non-CL matrix must not route on primaries"
  );
}

/// The SAME CL sample decoded to `rgba_u16` two ways must be identical:
///   (a) `rgba_u16`-only      → the direct native-depth RGBA kernel
///   (b) `rgb_u16` + `rgba_u16` → convert `rgb_u16`, then expand to `rgba_u16`
/// Both the RGB and the opaque alpha must match, every value native 12-bit
/// `[0, 4095]`. Guards the over-scaled-RGB + route-dependent-alpha defect.
#[test]
fn chroma_derived_cl_u16_rgba_route_consistent_and_native_depth() {
  for &full in &[true, false] {
    let only = decode_rgba_u16(2048, 2148, 1948, full, Transfer::Bt2020_12Bit, false);
    let with_rgb = decode_rgba_u16(2048, 2148, 1948, full, Transfer::Bt2020_12Bit, true);
    assert_eq!(
      only, with_rgb,
      "rgba_u16-only must equal rgb_u16+rgba_u16 (full={full})"
    );
    for px in only.chunks_exact(4) {
      assert!(
        px[..3].iter().all(|&c| c <= 4095),
        "rgba_u16 RGB over native 12-bit range: {px:?}"
      );
      assert_eq!(
        px[3], 4095,
        "native 12-bit opaque alpha must be 4095, got {}",
        px[3]
      );
    }
  }
}

// ---- Resample tier: non-affine matrices are rejected, not silently affine --
//
// The resample tail decodes colour through the affine `yuv444p12_to_rgb*`
// kernels (matrix + range only), so a resolved non-affine CL frame must NOT be
// silently decoded affine when a resize plan is active — it returns the typed
// `UnsupportedMatrixResample` error (#303). An *unresolved* CL tag
// (non-BT.2020 primaries) already falls back to affine on the identity route,
// so it resamples affinely without error.

/// Drives a solid `(yc,cbc,crc)` 12-bit 4:4:4 frame through a **resampling**
/// (downscale) `MixedSinker` to packed u8 RGB, returning the `process` result.
fn resample_rgb(
  yc: u16,
  cbc: u16,
  crc: u16,
  primaries: Primaries,
  transfer: Transfer,
) -> Result<(), crate::sinker::MixedSinkerError> {
  use crate::resample::AreaResampler;
  const SRC: usize = 4;
  const OUT: usize = 2;
  let n = SRC * SRC;
  let (y, u, v) = (std::vec![le_wire_u16(yc); n], std::vec![le_wire_u16(cbc); n], std::vec![le_wire_u16(crc); n]);
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(
      primaries,
      transfer,
      ColorMatrix::ChromaDerivedCl,
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  );
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut sink = MixedSinker::<crate::source::Yuv444p12, AreaResampler>::with_resampler(
    SRC,
    SRC,
    AreaResampler::to(OUT, OUT),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap()
  .with_color_spec(spec);
  crate::source::yuv444p12_to(&src, false, ColorMatrix::ChromaDerivedCl, &mut sink)
}

/// A resolved CL frame (BT.2020 primaries) + a resize plan must return the
/// typed `UnsupportedMatrixResample` error — NOT silent affine output, NOT a
/// panic.
#[test]
fn chroma_derived_cl_bt2020_resample_returns_typed_error() {
  let err = resample_rgb(2048, 2148, 1948, Primaries::Bt2020, Transfer::Bt2020_12Bit)
    .expect_err("CL + BT.2020 + resample must be rejected");
  match err {
    crate::sinker::MixedSinkerError::UnsupportedMatrixResample(e) => {
      assert_eq!(
        e.matrix(),
        "ChromaDerivedCl",
        "error names the offending matrix"
      );
    }
    other => panic!("expected UnsupportedMatrixResample, got {other:?}"),
  }
}

/// An UNRESOLVED CL tag (non-BT.2020 primaries) falls back to the affine path,
/// so a resize plan is accepted and resamples affinely — no error.
#[test]
fn chroma_derived_cl_unresolved_resample_is_affine_ok() {
  resample_rgb(2048, 2148, 1948, Primaries::Bt709, Transfer::Bt2020_12Bit)
    .expect("unresolved CL (non-BT.2020) must resample affinely, no error");
}

/// BT.2020 primaries but a non-BT.2020 transfer (PQ) is *unresolved* CL — it
/// takes the affine fallback, so a resize plan must be accepted and resample
/// affinely, NOT trip the `UnsupportedMatrixResample` reject (which fires only
/// for a *resolved* non-affine decode).
#[test]
fn chroma_derived_cl_bt2020_unsupported_transfer_resample_is_affine_ok() {
  resample_rgb(2048, 2148, 1948, Primaries::Bt2020, Transfer::SmpteSt2084Pq)
    .expect("BT.2020 + PQ (unresolved CL) must resample affinely, no error");
}
