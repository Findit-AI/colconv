//! End-to-end ICtCp (BT.2100, H.273 `MatrixCoefficients = 14`, #303) wiring
//! through the `Yuv444p12` `MixedSinker` identity path.
//!
//! Proves the full routing the row-kernel tests cannot: a
//! `ColorMatrix::Ictcp` source carrying a PQ/HLG transfer (delivered via
//! [`MixedSinker::with_color_spec`]) decodes through the non-affine ICtCp
//! kernel, the PQ-vs-HLG selection is honoured end-to-end, and a missing /
//! non-PQ-HLG transfer — or a non-`Ictcp` matrix — falls back to the affine
//! path. The expected values are the `colour-science` 0.4.7 references the
//! `row::scalar::ictcp` tests pin.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer, sinker::MixedSinker,
};

/// Decodes a `w×h` solid-`(i,ct,cp)` 12-bit 4:4:4 frame to packed u8 RGB
/// through the `MixedSinker`, with the sink's transfer set from a
/// `ColorSpec`.
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

fn decode_rgb(
  i: u16,
  ct: u16,
  cp: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u8> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(ct); n],
    std::vec![le_wire_u16(cp); n],
  );
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
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      matrix,
      range,
      ChromaLocation::Left,
    ),
  );
  {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(&spec);
    crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
      .unwrap();
  }
  rgb
}

fn decode_rgb_u16(
  i: u16,
  ct: u16,
  cp: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(ct); n],
    std::vec![le_wire_u16(cp); n],
  );
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
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      matrix,
      range,
      ChromaLocation::Left,
    ),
  );
  {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_color_spec(&spec);
    crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
      .unwrap();
  }
  rgb
}

/// Decodes a solid ICtCp frame to native-depth `rgba_u16`. When
/// `also_rgb_u16` the sink **also** attaches `rgb_u16`, which makes the sink
/// produce `rgba_u16` via the convert-rgb-then-`expand_rgb_u16_to_rgba_u16_row`
/// route instead of the direct RGBA kernel — the two routes must yield an
/// identical `rgba_u16` (same RGB, same opaque alpha).
fn decode_rgba_u16(
  i: u16,
  ct: u16,
  cp: u16,
  full_range: bool,
  transfer: Transfer,
  also_rgb_u16: bool,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(ct); n],
    std::vec![le_wire_u16(cp); n],
  );
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
      ColorMatrix::Ictcp,
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
      .with_color_spec(&spec);
    crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
      .unwrap();
  } else {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgba_u16(&mut rgba)
      .unwrap()
      .with_color_spec(&spec);
    crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
      .unwrap();
  }
  rgba
}

fn assert_all_pixels(rgb: &[u8], want: [u8; 3], tol: i32, what: &str) {
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

#[test]
fn sink_routes_pq_ictcp_through_non_affine_decode() {
  // I=2048, Ct=2148, Cp=2248, full range, PQ → colour-science [135,123,127].
  let rgb = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::SmpteSt2084Pq,
  );
  assert_all_pixels(&rgb, [135, 123, 127], 1, "PQ ICtCp sink RGB");
}

#[test]
fn sink_routes_hlg_ictcp_through_non_affine_decode() {
  // Same samples, HLG → colour-science [141,120,126]; differs from PQ.
  let rgb = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::AribStdB67Hlg,
  );
  assert_all_pixels(&rgb, [141, 120, 126], 1, "HLG ICtCp sink RGB");
}

#[test]
fn sink_routes_pq_ictcp_u16() {
  // Native 12-bit output (× 4095, NOT full-16-bit): every value in [0, 4095].
  let rgb = decode_rgb_u16(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::SmpteSt2084Pq,
  );
  for (px, chunk) in rgb.chunks_exact(3).enumerate() {
    for (c, &want) in [2167u16, 1981, 2040].iter().enumerate() {
      assert!(
        chunk[c] <= 4095,
        "PQ ICtCp sink u16 px{px} ch{c} = {} over native 12-bit range",
        chunk[c]
      );
      assert!(
        (chunk[c] as i32 - want as i32).abs() <= 2,
        "PQ ICtCp sink u16: px{px} ch{c} = {} (want {want})",
        chunk[c]
      );
    }
  }
}

#[test]
fn pq_and_hlg_routing_differ_end_to_end() {
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::SmpteSt2084Pq,
  );
  let hlg = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::AribStdB67Hlg,
  );
  assert_ne!(
    pq, hlg,
    "PQ and HLG ICtCp must decode differently end-to-end"
  );
}

#[test]
fn ictcp_without_pq_hlg_transfer_falls_back_to_affine() {
  // `Ictcp` matrix but `Unspecified` transfer → no ICtCp derivation defined;
  // routes to the affine fallback, so the output must NOT equal the PQ decode.
  let unspecified = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::Unspecified,
  );
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Ictcp,
    Transfer::SmpteSt2084Pq,
  );
  assert_ne!(
    unspecified, pq,
    "ICtCp + Unspecified transfer must fall back to affine (≠ PQ decode)"
  );
}

#[test]
fn non_ictcp_matrix_ignores_transfer() {
  // A non-`Ictcp` matrix must ignore the PQ transfer entirely (no ICtCp
  // routing): BT.709 decodes identically whether the transfer is PQ or not.
  let bt709_pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Bt709,
    Transfer::SmpteSt2084Pq,
  );
  let bt709_unspec = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Bt709,
    Transfer::Unspecified,
  );
  assert_eq!(
    bt709_pq, bt709_unspec,
    "non-ICtCp matrix must not route on transfer"
  );
}

// miri's interpreted floating-point diverges from hardware for the ICtCp
// (LMS + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn ictcp_u16_rgba_route_consistent_and_native_depth() {
  // The SAME ICtCp sample decoded to rgba_u16 two ways must be identical:
  //   (a) rgba_u16-only  -> the direct native-depth RGBA kernel
  //   (b) rgb_u16 + rgba_u16 -> convert rgb_u16, then expand to rgba_u16
  // Both RGB and the opaque alpha must match, and every value must be native
  // 12-bit [0, 4095] (not full-16-bit). Guards the over-scaled-RGB +
  // route-dependent-alpha defect. PQ (full) and HLG (studio) both covered.
  for (tf, full) in [
    (Transfer::SmpteSt2084Pq, true),
    (Transfer::AribStdB67Hlg, false),
  ] {
    let only = decode_rgba_u16(2048, 2148, 2248, full, tf.clone(), false);
    let with_rgb = decode_rgba_u16(2048, 2148, 2248, full, tf.clone(), true);
    assert_eq!(
      only, with_rgb,
      "rgba_u16-only must equal rgb_u16+rgba_u16 ({tf:?}, full={full})"
    );
    for px in only.chunks_exact(4) {
      assert!(
        px[..3].iter().all(|&c| c <= 4095),
        "rgba_u16 RGB over native 12-bit range ({tf:?}): {px:?}"
      );
      assert_eq!(
        px[3], 4095,
        "native 12-bit opaque alpha must be (1<<12)-1 = 4095 ({tf:?}), got {}",
        px[3]
      );
    }
  }
}

/// What output is co-attached alongside `with_hsv` when decoding HSV.
#[derive(Clone, Copy)]
enum HsvCo {
  /// HSV only (no RGB/RGBA) — the `want_hsv_direct` fast path for non-ICtCp.
  None,
  /// u8 RGB also attached — forces the convert-RGB-then-derive-HSV route.
  RgbU8,
  /// native u16 RGB also attached — the other route Codex named.
  RgbU16,
}

fn decode_hsv(
  i: u16,
  ct: u16,
  cp: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
  co: HsvCo,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(ct); n],
    std::vec![le_wire_u16(cp); n],
  );
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
      matrix,
      range,
      ChromaLocation::Left,
    ),
  );
  let (mut hh, mut ss, mut vv) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  let mut rgb = std::vec![0u8; n * 3];
  let mut rgb16 = std::vec![0u16; n * 3];
  match co {
    HsvCo::None => {
      let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_color_spec(&spec);
      crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
        .unwrap();
    }
    HsvCo::RgbU8 => {
      let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_color_spec(&spec);
      crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
        .unwrap();
    }
    HsvCo::RgbU16 => {
      let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
        .with_rgb_u16(&mut rgb16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_color_spec(&spec);
      crate::source::yuv444p12_to(&src, full_range, spec.kernel_matrix().unwrap(), &mut sink)
        .unwrap();
    }
  }
  (hh, ss, vv)
}

#[test]
fn ictcp_hsv_only_uses_non_affine_decode() {
  // The HSV-only (want_hsv_direct) route must decode ICtCp through the
  // non-affine kernel, NOT the affine yuv444p12_to_hsv_row_endian. Proven by:
  //   * HSV-only == RGB+HSV == rgb_u16+HSV (route-consistent — all use the
  //     same non-affine RGB then rgb_to_hsv_row), and
  //   * HSV-only (ICtCp) != HSV via the affine fallback (Ictcp + Unspecified
  //     transfer, which is NOT a defined ICtCp transfer) — so the output is
  //     genuinely ICtCp-derived, not YCbCr-derived.
  for tf in [Transfer::SmpteSt2084Pq, Transfer::AribStdB67Hlg] {
    let only = decode_hsv(
      2048,
      2148,
      2248,
      true,
      ColorMatrix::Ictcp,
      tf.clone(),
      HsvCo::None,
    );
    let via_rgb = decode_hsv(
      2048,
      2148,
      2248,
      true,
      ColorMatrix::Ictcp,
      tf.clone(),
      HsvCo::RgbU8,
    );
    let via_rgb16 = decode_hsv(
      2048,
      2148,
      2248,
      true,
      ColorMatrix::Ictcp,
      tf.clone(),
      HsvCo::RgbU16,
    );
    let affine = decode_hsv(
      2048,
      2148,
      2248,
      true,
      ColorMatrix::Ictcp,
      Transfer::Unspecified,
      HsvCo::None,
    );
    assert_eq!(
      only, via_rgb,
      "{tf:?}: ICtCp HSV-only must equal the RGB+HSV route"
    );
    assert_eq!(
      only, via_rgb16,
      "{tf:?}: ICtCp HSV-only must equal the rgb_u16+HSV route"
    );
    assert_ne!(
      only, affine,
      "{tf:?}: ICtCp HSV must differ from the affine-fallback HSV"
    );
  }
}

#[test]
fn non_ictcp_hsv_only_unchanged() {
  // Sanity: a non-ICtCp matrix keeps the affine want_hsv_direct fast path —
  // HSV-only equals the RGB+HSV route (the existing kernel contract), and a
  // PQ transfer on a BT.709 matrix does not perturb it.
  let only = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Bt709,
    Transfer::SmpteSt2084Pq,
    HsvCo::None,
  );
  let via_rgb = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Bt709,
    Transfer::SmpteSt2084Pq,
    HsvCo::RgbU8,
  );
  assert_eq!(only, via_rgb, "affine HSV-only must equal affine RGB+HSV");
}

// ---- Resample tier: ICtCp is rejected, not silently affine (#303) -------
//
// Folded-in fix for the same silent bug on the merged ICtCp decode (#324): its
// resample tail also routes through the affine kernels. A resolved ICtCp frame
// (PQ/HLG transfer) + a resize plan must return the typed
// `UnsupportedMatrixResample` error, not silent affine output.

/// Drives a solid `(i,ct,cp)` 12-bit 4:4:4 frame through a **resampling**
/// `MixedSinker` to packed u8 RGB, returning the `process` result.
fn resample_rgb(
  i: u16,
  ct: u16,
  cp: u16,
  transfer: Transfer,
) -> Result<(), crate::sinker::MixedSinkerError> {
  use crate::resample::AreaResampler;
  const SRC: usize = 4;
  const OUT: usize = 2;
  let n = SRC * SRC;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(ct); n],
    std::vec![le_wire_u16(cp); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      ColorMatrix::Ictcp,
      DynamicRange::Full,
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
  .with_color_spec(&spec);
  crate::source::yuv444p12_to(&src, true, spec.kernel_matrix().unwrap(), &mut sink)
}

/// A resolved ICtCp frame (PQ transfer) + a resize plan must return the typed
/// `UnsupportedMatrixResample` error — NOT silent affine output, NOT a panic.
#[test]
fn ictcp_pq_resample_returns_typed_error() {
  let err = resample_rgb(2048, 2148, 2248, Transfer::SmpteSt2084Pq)
    .expect_err("ICtCp + PQ + resample must be rejected");
  match err {
    crate::sinker::MixedSinkerError::UnsupportedMatrixResample(e) => {
      assert_eq!(e.matrix(), "Ictcp", "error names the offending matrix");
    }
    other => panic!("expected UnsupportedMatrixResample, got {other:?}"),
  }
}

/// An UNRESOLVED ICtCp tag (no PQ/HLG transfer) falls back to affine, so a
/// resize plan is accepted and resamples affinely — no error.
#[test]
fn ictcp_unresolved_resample_is_affine_ok() {
  resample_rgb(2048, 2148, 2248, Transfer::Unspecified)
    .expect("unresolved ICtCp (no PQ/HLG) must resample affinely, no error");
}
