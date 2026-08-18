//! End-to-end SMPTE ST 2085 (H.273 `MatrixCoefficients = 11`, "Y'D'zD'x", the
//! PQ-only non-affine X'Y'Z' colour-difference model, #303) wiring through the
//! `Yuv444p12` `MixedSinker` identity path.
//!
//! Proves the full routing the row-kernel tests cannot: a
//! `ColorMatrix::Smpte2085` source carrying a PQ transfer (delivered via
//! [`MixedSinker::with_color_spec`]) decodes through the non-affine SMPTE 2085
//! kernel, and a missing / non-PQ transfer — or a non-`Smpte2085` matrix —
//! falls back to the affine path. The expected values are computed directly
//! from the `row::scalar::smpte2085` reference kernel the row tests pin, so the
//! sink output must be bit-identical to it.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer, sinker::MixedSinker,
};

/// Encode a logical `u16` as host-independent **LE-wire** byte storage so a
/// `Le` sink decodes it identically on any host.
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// The `row::scalar::smpte2085` reference decode of a solid `(y,dz,dx)` sample
/// to packed u8 RGB — the exact value the sink must reproduce when it routes
/// through the non-affine kernel.
fn reference_rgb_u8(y: u16, dz: u16, dx: u16, full: bool) -> [u8; 3] {
  let (yy, u, v) = ([le_wire_u16(y)], [le_wire_u16(dz)], [le_wire_u16(dx)]);
  let mut out = [0u8; 3];
  crate::row::scalar::smpte2085::smpte2085_444p_n_to_rgb_row::<12, false>(
    &yy, &u, &v, &mut out, 1, full,
  );
  out
}

/// The `row::scalar::smpte2085` reference decode to native-depth u16 RGB.
fn reference_rgb_u16(y: u16, dz: u16, dx: u16, full: bool) -> [u16; 3] {
  let (yy, u, v) = ([le_wire_u16(y)], [le_wire_u16(dz)], [le_wire_u16(dx)]);
  let mut out = [0u16; 3];
  crate::row::scalar::smpte2085::smpte2085_444p_n_to_rgb_u16_row::<12, false>(
    &yy, &u, &v, &mut out, 1, full,
  );
  out
}

/// Decodes a `w×h` solid-`(y,dz,dx)` 12-bit 4:4:4 frame to packed u8 RGB
/// through the `MixedSinker`, with the sink's transfer set from a `ColorSpec`.
fn decode_rgb(
  y: u16,
  dz: u16,
  dx: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u8> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (yy, u, v) = (
    std::vec![le_wire_u16(y); n],
    std::vec![le_wire_u16(dz); n],
    std::vec![le_wire_u16(dx); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &yy, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
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
  y: u16,
  dz: u16,
  dx: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (yy, u, v) = (
    std::vec![le_wire_u16(y); n],
    std::vec![le_wire_u16(dz); n],
    std::vec![le_wire_u16(dx); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &yy, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
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

/// Decodes a solid SMPTE 2085 frame to native-depth `rgba_u16`. When
/// `also_rgb_u16` the sink **also** attaches `rgb_u16`, which makes the sink
/// produce `rgba_u16` via the convert-rgb-then-`expand_rgb_u16_to_rgba_u16_row`
/// route instead of the direct RGBA kernel — the two routes must yield an
/// identical `rgba_u16` (same RGB, same opaque alpha).
fn decode_rgba_u16(
  y: u16,
  dz: u16,
  dx: u16,
  full_range: bool,
  transfer: Transfer,
  also_rgb_u16: bool,
) -> std::vec::Vec<u16> {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (yy, u, v) = (
    std::vec![le_wire_u16(y); n],
    std::vec![le_wire_u16(dz); n],
    std::vec![le_wire_u16(dx); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &yy, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
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
      ColorMatrix::Smpte2085,
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

#[test]
fn sink_routes_pq_smpte2085_through_non_affine_decode() {
  // A Smpte2085 + PQ source must decode through the non-affine kernel: the sink
  // output equals the `row::scalar::smpte2085` reference for the same sample.
  for &full in &[true, false] {
    let want = reference_rgb_u8(2048, 2148, 2248, full);
    let rgb = decode_rgb(
      2048,
      2148,
      2248,
      full,
      ColorMatrix::Smpte2085,
      Transfer::SmpteSt2084Pq,
    );
    for (px, chunk) in rgb.chunks_exact(3).enumerate() {
      assert_eq!(
        chunk,
        &want[..],
        "PQ SMPTE 2085 sink RGB px{px} (full={full})"
      );
    }
  }
}

#[test]
fn sink_routes_pq_smpte2085_u16() {
  // Native 12-bit output (× 4095, NOT full-16-bit): every value in [0, 4095]
  // and bit-identical to the reference u16 kernel.
  let want = reference_rgb_u16(2048, 2148, 2248, true);
  let rgb = decode_rgb_u16(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::SmpteSt2084Pq,
  );
  for (px, chunk) in rgb.chunks_exact(3).enumerate() {
    assert!(
      chunk.iter().all(|&c| c <= 4095),
      "PQ SMPTE 2085 sink u16 px{px} over native 12-bit range: {chunk:?}"
    );
    assert_eq!(chunk, &want[..], "PQ SMPTE 2085 sink u16 px{px}");
  }
}

#[test]
fn smpte2085_without_pq_transfer_falls_back_to_affine() {
  // `Smpte2085` matrix but `Unspecified` transfer → no SMPTE 2085 derivation
  // defined; routes to the affine fallback, so the output must NOT equal the PQ
  // decode.
  let unspecified = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::Unspecified,
  );
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::SmpteSt2084Pq,
  );
  assert_ne!(
    unspecified, pq,
    "SMPTE 2085 + Unspecified transfer must fall back to affine (≠ PQ decode)"
  );
}

#[test]
fn smpte2085_hlg_transfer_falls_back_to_affine() {
  // HLG is NOT a SMPTE 2085 transfer (PQ-only), so a `Smpte2085` + HLG source
  // must route to the affine fallback — identical to the `Unspecified` fallback
  // and distinct from the PQ non-affine decode.
  let hlg = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::AribStdB67Hlg,
  );
  let unspecified = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::Unspecified,
  );
  let pq = decode_rgb(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::SmpteSt2084Pq,
  );
  assert_eq!(
    hlg, unspecified,
    "SMPTE 2085 + HLG must fall back to affine"
  );
  assert_ne!(
    hlg, pq,
    "SMPTE 2085 + HLG must NOT take the PQ non-affine decode"
  );
}

#[test]
fn non_smpte2085_matrix_ignores_transfer() {
  // A non-`Smpte2085` matrix must ignore the PQ transfer entirely (no SMPTE
  // 2085 routing): BT.709 decodes identically whether the transfer is PQ or
  // not.
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
    "non-SMPTE 2085 matrix must not route on transfer"
  );
}

// miri's interpreted floating-point diverges from hardware for the SMPTE 2085
// (XYZ + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn smpte2085_u16_rgba_route_consistent_and_native_depth() {
  // The SAME SMPTE 2085 sample decoded to rgba_u16 two ways must be identical:
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
  /// HSV only (no RGB/RGBA) — the `want_hsv_direct` fast path for non-affine.
  None,
  /// u8 RGB also attached — forces the convert-RGB-then-derive-HSV route.
  RgbU8,
  /// native u16 RGB also attached — the other convert-then-derive route.
  RgbU16,
}

fn decode_hsv(
  y: u16,
  dz: u16,
  dx: u16,
  full_range: bool,
  matrix: ColorMatrix,
  transfer: Transfer,
  co: HsvCo,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let (w, h) = (4usize, 2usize);
  let n = w * h;
  let (yy, u, v) = (
    std::vec![le_wire_u16(y); n],
    std::vec![le_wire_u16(dz); n],
    std::vec![le_wire_u16(dx); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &yy, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32,
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
fn smpte2085_hsv_only_uses_non_affine_decode() {
  // The HSV-only (want_hsv_direct) route must decode SMPTE 2085 through the
  // non-affine kernel, NOT the affine yuv444p12_to_hsv_row_endian. Proven by:
  //   * HSV-only == RGB+HSV == rgb_u16+HSV (route-consistent — all use the
  //     same non-affine RGB then rgb_to_hsv_row), and
  //   * HSV-only (SMPTE 2085 PQ) != HSV via the affine fallback (Smpte2085 +
  //     Unspecified transfer, which is NOT a defined SMPTE 2085 transfer) — so
  //     the output is genuinely SMPTE-2085-derived, not YCbCr-derived.
  let tf = Transfer::SmpteSt2084Pq;
  let only = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    tf.clone(),
    HsvCo::None,
  );
  let via_rgb = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    tf.clone(),
    HsvCo::RgbU8,
  );
  let via_rgb16 = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    tf,
    HsvCo::RgbU16,
  );
  let affine = decode_hsv(
    2048,
    2148,
    2248,
    true,
    ColorMatrix::Smpte2085,
    Transfer::Unspecified,
    HsvCo::None,
  );
  assert_eq!(
    only, via_rgb,
    "SMPTE 2085 HSV-only must equal the RGB+HSV route"
  );
  assert_eq!(
    only, via_rgb16,
    "SMPTE 2085 HSV-only must equal the rgb_u16+HSV route"
  );
  assert_ne!(
    only, affine,
    "SMPTE 2085 HSV must differ from the affine-fallback HSV"
  );
}

// ---- Resample tier: SMPTE 2085 is rejected, not silently affine (#303) ---
//
// The resample tail routes through the affine kernels. A resolved SMPTE 2085
// frame (PQ transfer) + a resize plan must return the typed
// `UnsupportedMatrixResample` error, not silent affine output.

/// Drives a solid `(y,dz,dx)` 12-bit 4:4:4 frame through a **resampling**
/// `MixedSinker` to packed u8 RGB, returning the `process` result.
fn resample_rgb(
  y: u16,
  dz: u16,
  dx: u16,
  transfer: Transfer,
) -> Result<(), crate::sinker::MixedSinkerError> {
  use crate::resample::AreaResampler;
  const SRC: usize = 4;
  const OUT: usize = 2;
  let n = SRC * SRC;
  let (yy, u, v) = (
    std::vec![le_wire_u16(y); n],
    std::vec![le_wire_u16(dz); n],
    std::vec![le_wire_u16(dx); n],
  );
  let src = crate::frame::Yuv444pFrame16::<12>::new(
    &yy, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p12Le,
    ColorInfo::new(
      Primaries::Bt2020,
      transfer,
      ColorMatrix::Smpte2085,
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

/// A resolved SMPTE 2085 frame (PQ transfer) + a resize plan must return the
/// typed `UnsupportedMatrixResample` error — NOT silent affine output, NOT a
/// panic.
#[test]
fn smpte2085_pq_resample_returns_typed_error() {
  let err = resample_rgb(2048, 2148, 2248, Transfer::SmpteSt2084Pq)
    .expect_err("SMPTE 2085 + PQ + resample must be rejected");
  match err {
    crate::sinker::MixedSinkerError::UnsupportedMatrixResample(e) => {
      assert_eq!(e.matrix(), "Smpte2085", "error names the offending matrix");
    }
    other => panic!("expected UnsupportedMatrixResample, got {other:?}"),
  }
}

/// An UNRESOLVED SMPTE 2085 tag (no PQ transfer) falls back to affine, so a
/// resize plan is accepted and resamples affinely — no error.
#[test]
fn smpte2085_unresolved_resample_is_affine_ok() {
  resample_rgb(2048, 2148, 2248, Transfer::Unspecified)
    .expect("unresolved SMPTE 2085 (no PQ) must resample affinely, no error");
}
