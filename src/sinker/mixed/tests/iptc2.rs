//! End-to-end IPT-C2 (H.273 `MatrixCoefficients = 15`, the Dolby Vision
//! Profile 5 base colour space, #303) wiring through the `Yuv444p12`
//! `MixedSinker` identity path.
//!
//! Proves the full routing the row-kernel tests cannot: a `ColorMatrix::IptC2`
//! source carrying a PQ transfer (delivered via
//! [`MixedSinker::with_color_spec`]) decodes through the non-affine IPT-C2
//! kernel, and a missing / non-PQ transfer — or a non-`IptC2` matrix — falls
//! back to the affine path. The expected values are the spec-integer decode
//! references the `row::scalar::iptc2` tests pin.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer, sinker::MixedSinker,
};

/// Encode a logical `u16` as host-independent **LE-wire** byte storage so a
/// `Le` sink decodes it identically on any host.
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// Decodes a `w×h` solid-`(i,p,t)` 12-bit 4:4:4 frame to packed u8 RGB through
/// the `MixedSinker`, with the sink's transfer set from a `ColorSpec`.
fn decode_rgb(
  i: u16,
  p: u16,
  t: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u8> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(p); n],
    std::vec![le_wire_u16(t); n],
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
      .with_color_spec(&spec)
      .unwrap();
    crate::source::yuv444p12_to(
      &src,
      full_range,
      sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
    )
    .unwrap();
  }
  rgb
}

fn decode_rgb_u16(
  i: u16,
  p: u16,
  t: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(p); n],
    std::vec![le_wire_u16(t); n],
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
      .with_color_spec(&spec)
      .unwrap();
    crate::source::yuv444p12_to(
      &src,
      full_range,
      sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
    )
    .unwrap();
  }
  rgb
}

/// Decodes a solid IPT-C2 frame to native-depth `rgba_u16`. When `also_rgb_u16`
/// the sink **also** attaches `rgb_u16`, which makes the sink produce
/// `rgba_u16` via the convert-rgb-then-`expand_rgb_u16_to_rgba_u16_row` route
/// instead of the direct RGBA kernel — the two routes must yield an identical
/// `rgba_u16` (same RGB, same opaque alpha).
fn decode_rgba_u16(
  i: u16,
  p: u16,
  t: u16,
  full_range: bool,
  transfer: Transfer,
  also_rgb_u16: bool,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(p); n],
    std::vec![le_wire_u16(t); n],
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
      ColorMatrix::IptC2,
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
      .with_color_spec(&spec)
      .unwrap();
    crate::source::yuv444p12_to(
      &src,
      full_range,
      sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
    )
    .unwrap();
  } else {
    let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
      .with_rgba_u16(&mut rgba)
      .unwrap()
      .with_color_spec(&spec)
      .unwrap();
    crate::source::yuv444p12_to(
      &src,
      full_range,
      sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
    )
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
fn sink_routes_pq_iptc2_through_non_affine_decode() {
  // I=2048, P=2148, T=2248, full range, PQ → spec-integer decode [135,128,118].
  let rgb = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::SmpteSt2084Pq,
  );
  assert_all_pixels(&rgb, [135, 128, 118], 1, "PQ IPT-C2 sink RGB");
}

#[test]
fn sink_routes_pq_iptc2_u16() {
  // Native 12-bit output (× 4095, NOT full-16-bit): every value in [0, 4095].
  let rgb = decode_rgb_u16(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::SmpteSt2084Pq,
  );
  for (px, chunk) in rgb.chunks_exact(3).enumerate() {
    for (c, &want) in [2161u16, 2057, 1902].iter().enumerate() {
      assert!(
        chunk[c] <= 4095,
        "PQ IPT-C2 sink u16 px{px} ch{c} = {} over native 12-bit range",
        chunk[c]
      );
      assert!(
        (chunk[c] as i32 - want as i32).abs() <= 2,
        "PQ IPT-C2 sink u16: px{px} ch{c} = {} (want {want})",
        chunk[c]
      );
    }
  }
}

#[test]
fn iptc2_without_pq_transfer_falls_back_to_affine() {
  // `IptC2` matrix but `Unspecified` transfer → no IPT-C2 derivation defined;
  // routes to the affine fallback, so the output must NOT equal the PQ decode.
  let unspecified = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::Unspecified,
  );
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::SmpteSt2084Pq,
  );
  assert_ne!(
    unspecified, pq,
    "IPT-C2 + Unspecified transfer must fall back to affine (≠ PQ decode)"
  );
}

#[test]
fn iptc2_hlg_transfer_falls_back_to_affine() {
  // HLG is NOT an IPT-C2 transfer (PQ-only), so an `IptC2` + HLG source must
  // route to the affine fallback — identical to the `Unspecified` fallback and
  // distinct from the PQ non-affine decode.
  let hlg = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::AribStdB67Hlg,
  );
  let unspecified = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::Unspecified,
  );
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::SmpteSt2084Pq,
  );
  assert_eq!(hlg, unspecified, "IPT-C2 + HLG must fall back to affine");
  assert_ne!(
    hlg, pq,
    "IPT-C2 + HLG must NOT take the PQ non-affine decode"
  );
}

#[test]
fn non_iptc2_matrix_ignores_transfer() {
  // A non-`IptC2` matrix must ignore the PQ transfer entirely (no IPT-C2
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
    "non-IPT-C2 matrix must not route on transfer"
  );
}

// miri's interpreted floating-point diverges from hardware for the IPT-C2
// (LMS + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn iptc2_u16_rgba_route_consistent_and_native_depth() {
  // The SAME IPT-C2 sample decoded to rgba_u16 two ways must be identical:
  //   (a) rgba_u16-only  -> the direct native-depth RGBA kernel
  //   (b) rgb_u16 + rgba_u16 -> convert rgb_u16, then expand to rgba_u16
  // Both RGB and the opaque alpha must match, and every value must be native
  // 12-bit [0, 4095] (not full-16-bit). Full and studio range both covered.
  for &full in &[true, false] {
    let only = decode_rgba_u16(2048, 2148, 2248, full, Transfer::SmpteSt2084Pq, false);
    let with_rgb = decode_rgba_u16(2048, 2148, 2248, full, Transfer::SmpteSt2084Pq, true);
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
        "native 12-bit opaque alpha must be (1<<12)-1 = 4095, got {}",
        px[3]
      );
    }
  }
}

/// What output is co-attached alongside `with_hsv` when decoding HSV.
#[derive(Clone, Copy)]
enum HsvCo {
  /// HSV only (no RGB/RGBA) — the `want_hsv_direct` fast path for non-IPT-C2.
  None,
  /// u8 RGB also attached — forces the convert-RGB-then-derive-HSV route.
  RgbU8,
  /// native u16 RGB also attached — the other convert-then-derive route.
  RgbU16,
}

fn decode_hsv(
  i: u16,
  p: u16,
  t: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
  co: HsvCo,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(p); n],
    std::vec![le_wire_u16(t); n],
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
        .with_color_spec(&spec)
        .unwrap();
      crate::source::yuv444p12_to(
        &src,
        full_range,
        sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
      )
      .unwrap();
    }
    HsvCo::RgbU8 => {
      let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_color_spec(&spec)
        .unwrap();
      crate::source::yuv444p12_to(
        &src,
        full_range,
        sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
      )
      .unwrap();
    }
    HsvCo::RgbU16 => {
      let mut sink = MixedSinker::<crate::source::Yuv444p12>::new(w, h)
        .with_rgb_u16(&mut rgb16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap()
        .with_color_spec(&spec)
        .unwrap();
      crate::source::yuv444p12_to(
        &src,
        full_range,
        sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
      )
      .unwrap();
    }
  }
  (hh, ss, vv)
}

#[test]
fn iptc2_hsv_only_uses_non_affine_decode() {
  // The HSV-only (want_hsv_direct) route must decode IPT-C2 through the
  // non-affine kernel, NOT the affine yuv444p12_to_hsv_row_endian. Proven by:
  //   * HSV-only == RGB+HSV == rgb_u16+HSV (route-consistent — all use the
  //     same non-affine RGB then rgb_to_hsv_row), and
  //   * HSV-only (IPT-C2 PQ) != HSV via the affine fallback (IptC2 + Unspecified
  //     transfer, which is NOT a defined IPT-C2 transfer) — so the output is
  //     genuinely IPT-C2-derived, not YCbCr-derived.
  let tf = Transfer::SmpteSt2084Pq;
  let only = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    tf.clone(),
    HsvCo::None,
  );
  let via_rgb = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    tf.clone(),
    HsvCo::RgbU8,
  );
  let via_rgb16 = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    tf,
    HsvCo::RgbU16,
  );
  let affine = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::IptC2,
    Transfer::Unspecified,
    HsvCo::None,
  );
  assert_eq!(
    only, via_rgb,
    "IPT-C2 HSV-only must equal the RGB+HSV route"
  );
  assert_eq!(
    only, via_rgb16,
    "IPT-C2 HSV-only must equal the rgb_u16+HSV route"
  );
  assert_ne!(
    only, affine,
    "IPT-C2 HSV must differ from the affine-fallback HSV"
  );
}

#[test]
fn non_iptc2_hsv_only_unchanged() {
  // Sanity: a non-IPT-C2 matrix keeps the affine want_hsv_direct fast path —
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

// ---- Resample tier: IPT-C2 is rejected, not silently affine (#303) ------
//
// The resample tail routes through the affine kernels. A resolved IPT-C2 frame
// (PQ transfer) + a resize plan must return the typed
// `UnsupportedMatrixResample` error, not silent affine output.

/// Drives a solid `(i,p,t)` 12-bit 4:4:4 frame through a **resampling**
/// `MixedSinker` to packed u8 RGB, returning the `process` result.
fn resample_rgb(
  i: u16,
  p: u16,
  t: u16,
  transfer: Transfer,
) -> Result<(), crate::sinker::MixedSinkerError> {
  use crate::resample::AreaResampler;
  const SRC: usize = 4;
  const OUT: usize = 2;
  let n = SRC * SRC;
  let (y, u, v) = (
    std::vec![le_wire_u16(i); n],
    std::vec![le_wire_u16(p); n],
    std::vec![le_wire_u16(t); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      ColorMatrix::IptC2,
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
  .with_color_spec(&spec)
  .unwrap();
  crate::source::yuv444p12_to(
    &src,
    true,
    sink.set_kernel_matrix(spec.kernel_matrix().unwrap()),
  )
}

/// A resolved IPT-C2 frame (PQ transfer) + a resize plan must return the typed
/// `UnsupportedMatrixResample` error — NOT silent affine output, NOT a panic.
#[test]
fn iptc2_pq_resample_returns_typed_error() {
  let err = resample_rgb(2048, 2148, 2248, Transfer::SmpteSt2084Pq)
    .expect_err("IPT-C2 + PQ + resample must be rejected");
  match err {
    crate::sinker::MixedSinkerError::UnsupportedMatrixResample(e) => {
      assert_eq!(e.matrix(), "IptC2", "error names the offending matrix");
    }
    other => panic!("expected UnsupportedMatrixResample, got {other:?}"),
  }
}

/// An UNRESOLVED IPT-C2 tag (no PQ transfer) falls back to affine, so a resize
/// plan is accepted and resamples affinely — no error.
#[test]
fn iptc2_unresolved_resample_is_affine_ok() {
  resample_rgb(2048, 2148, 2248, Transfer::Unspecified)
    .expect("unresolved IPT-C2 (no PQ) must resample affinely, no error");
}
