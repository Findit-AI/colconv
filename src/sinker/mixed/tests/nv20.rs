//! NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) `MixedSinker`
//! coverage — the low-bit twin of P210.
//!
//! NV20 packs its 10 active bits in the **low** 10 of each `u16`
//! (`& 0x03FF`); P210 packs them in the high 10 (`>> 6`). The YUV→RGB
//! math is otherwise byte-identical. These tests therefore focus on:
//!
//! - **extraction correctness on REAL low-bit data** (not zeros): a
//!   low-bit-packed gray frame round-trips to the expected RGB, and a
//!   handcrafted low-bit pixel decodes to the same RGB as the equivalent
//!   P210 (high-bit-packed) pixel carrying the SAME logical sample;
//! - **the low-vs-high distinction is load-bearing**: feeding the SAME
//!   wire bytes to an NV20 sink vs a P210 sink yields DIFFERENT output
//!   (a wrong shift would silently agree);
//! - **per-SIMD-tier equivalence** (every backend == scalar) over
//!   full/limited × Bt709/Bt601 × LE/BE with a tail width;
//! - **all output paths** P210 supports (rgb / rgba / rgb_u16 /
//!   rgba_u16 / luma / hsv);
//! - **Walker parity** (`Walker::walk` == `nv20_to` directly), LE + BE.

use super::*;
use crate::{
  Walker,
  frame::Nv20Frame,
  row::{nv20_to_hsv_row_endian, nv20_to_rgb_row_endian, rgb_to_hsv_row},
  source::{Nv20, P210, nv20_to, nv20_to_endian, p210_to},
  walker::YuvOptions,
};

/// Builds an NV20 4:2:2 frame (full-width Y + half-width interleaved UV at
/// full height) from **logical** sample values, low-bit-packed (active
/// bits in the low 10, high 6 zero — NV20's `& 0x03FF` contract), encoded
/// in the wire byte order selected by `BE`.
fn nv20_solid(
  width: u32,
  height: u32,
  y_value: u16,
  u_value: u16,
  v_value: u16,
  be: bool,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let enc = |v: u16| -> u16 {
    // Low-bit-packed: the logical value is already in the low 10 bits.
    let logical = v & 0x03FF;
    if be {
      u16::from_ne_bytes(logical.to_be_bytes())
    } else {
      u16::from_ne_bytes(logical.to_le_bytes())
    }
  };
  let y = std::vec![enc(y_value); w * h];
  let uv: Vec<u16> = (0..cw * h)
    .flat_map(|_| [enc(u_value), enc(v_value)])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_gray_to_gray_real_low_bits() {
  // Mid-gray: logical Y/U/V = 512 in the LOW 10 bits (the literal stored
  // u16 is 0x0200, high 6 zero — genuine low-bit data, NOT zeros). The
  // `& 0x03FF` de-pack must recover 512 → 8-bit gray ≈ 128.
  let (yp, uvp) = nv20_solid(16, 8, 512, 512, 512, false);
  // Sanity: the stored words really do carry the value in the low bits.
  assert_eq!(yp[0], 0x0200, "low-bit-packed Y word must be 0x0200, not 0");
  let src = Nv20Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv20>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv20_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_low_bit_decodes_same_as_p210_with_equal_logical_sample() {
  // The CRUX: an NV20 pixel whose logical 10-bit sample is L (stored in
  // the low 10) must decode to the SAME RGB as a P210 pixel whose logical
  // sample is also L (stored in the high 10, i.e. `L << 6`). This proves
  // NV20's `& 0x03FF` extraction recovers the intended value — a wrong
  // shift (e.g. `>> 6` on low-packed data) would yield ~0.
  let w = 64u32;
  let h = 4u32;
  // Non-neutral logical samples, all within 10-bit range.
  let (yl, ul, vl) = (650u16, 300u16, 720u16);

  // NV20: low-bit-packed (logical value in the low 10).
  let (nv_y, nv_uv) = nv20_solid(w, h, yl, ul, vl, false);
  let nv_src = Nv20Frame::new(&nv_y, &nv_uv, w, h, w, w);

  // P210: high-bit-packed (logical value << 6). Build directly.
  let cw = (w / 2) as usize;
  let p_y = std::vec![yl << 6; (w * h) as usize];
  let p_uv: Vec<u16> = (0..cw * h as usize)
    .flat_map(|_| [ul << 6, vl << 6])
    .collect();
  let p_src = crate::frame::P210Frame::new(&p_y, &p_uv, w, h, w, w);

  for &(fr, m) in &[
    (true, KernelMatrix::Bt709),
    (false, KernelMatrix::Bt709),
    (true, KernelMatrix::Bt601),
    (false, KernelMatrix::Bt601),
  ] {
    let mut nv_rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut nv_sink = MixedSinker::<Nv20>::new(w as usize, h as usize)
      .with_rgb(&mut nv_rgb)
      .unwrap();
    nv20_to(&nv_src, fr, m, &mut nv_sink).unwrap();

    let mut p_rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut p_sink = MixedSinker::<P210>::new(w as usize, h as usize)
      .with_rgb(&mut p_rgb)
      .unwrap();
    p210_to(&p_src, fr, m, &mut p_sink).unwrap();

    assert_eq!(
      nv_rgb, p_rgb,
      "NV20(low-packed L) must decode == P210(high-packed L) for fr={fr} {m:?}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_and_p210_disagree_on_identical_wire_bytes() {
  // The same raw wire buffer fed to NV20 (low de-pack) vs P210 (high
  // de-pack) MUST produce different RGB — otherwise the extraction is not
  // actually reading different bits. Use bytes with signal in BOTH the
  // low and high halves so neither interpretation collapses to gray.
  let w = 16u32;
  let h = 4u32;
  // Stored word 0x1234: low 10 = 0x234 (564), high 6 over the top → as
  // P210 `>> 6` = 0x48 (72). Distinct logical samples ⇒ distinct RGB.
  let word = 0x1234u16;
  let cw = (w / 2) as usize;
  let y = std::vec![word; (w * h) as usize];
  let uv: Vec<u16> = (0..cw * h as usize).flat_map(|_| [word, word]).collect();

  let nv_src = Nv20Frame::new(&y, &uv, w, h, w, w);
  let p_src = crate::frame::P210Frame::new(&y, &uv, w, h, w, w);

  let mut nv_rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut nv_sink = MixedSinker::<Nv20>::new(w as usize, h as usize)
    .with_rgb(&mut nv_rgb)
    .unwrap();
  nv20_to(&nv_src, true, KernelMatrix::Bt709, &mut nv_sink).unwrap();

  let mut p_rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut p_sink = MixedSinker::<P210>::new(w as usize, h as usize)
    .with_rgb(&mut p_rgb)
    .unwrap();
  p210_to(&p_src, true, KernelMatrix::Bt709, &mut p_sink).unwrap();

  assert_ne!(
    nv_rgb, p_rgb,
    "NV20 (low de-pack) and P210 (high de-pack) must differ on identical wire bytes"
  );
}

/// Pseudo-random low-bit-packed NV20 planes for parity testing — every
/// stored word has its 10 active bits in the low 10 (high 6 zero),
/// encoded in the `BE` wire order. Returns `(y, uv)` for a 4:2:2 frame
/// (`uv` holds `w` u16 per row).
fn nv20_random<const BE: bool>(w: u32, h: u32, seed: u32) -> (Vec<u16>, Vec<u16>) {
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (w * h) as usize];
  let mut state = seed;
  let enc = |logical: u16| -> u16 {
    if BE {
      u16::from_ne_bytes(logical.to_be_bytes())
    } else {
      u16::from_ne_bytes(logical.to_le_bytes())
    }
  };
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = enc((state >> 13) as u16 & 0x03FF);
  }
  (yp, uvp)
}

/// Drives every output path on an NV20 sink (SIMD or scalar per `simd`)
/// and returns the six output buffers. Const-generic over `BE` so the
/// `Nv20Frame<'_, BE>` / `MixedSinker<Nv20<BE>>` types line up without
/// runtime endian juggling.
#[allow(clippy::type_complexity)]
fn nv20_run_all<const BE: bool>(
  yp: &[u16],
  uvp: &[u16],
  w: u32,
  h: u32,
  matrix: KernelMatrix,
  full_range: bool,
  simd: bool,
) -> (
  Vec<u8>,
  Vec<u16>,
  Vec<u8>,
  Vec<u16>,
  Vec<u8>,
  (Vec<u8>, Vec<u8>, Vec<u8>),
) {
  let wz = w as usize;
  let hz = h as usize;
  let src = Nv20Frame::<BE>::new(yp, uvp, w, h, w, w);

  let mut rgb = std::vec![0u8; wz * hz * 3];
  let mut rgb_u16 = std::vec![0u16; wz * hz * 3];
  let mut rgba = std::vec![0u8; wz * hz * 4];
  let mut rgba_u16 = std::vec![0u16; wz * hz * 4];
  let mut luma = std::vec![0u8; wz * hz];
  let mut hh = std::vec![0u8; wz * hz];
  let mut ss = std::vec![0u8; wz * hz];
  let mut vv = std::vec![0u8; wz * hz];

  // rgb + rgb_u16 + luma + hsv together (Strategy A keeps rgb the kernel
  // output; hsv derives from the same rgb row).
  {
    let mut sink = MixedSinker::<Nv20<BE>>::new(wz, hz)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    sink.set_simd(simd);
    nv20_to_endian::<_, BE>(&src, full_range, matrix, &mut sink).unwrap();
  }
  // rgba (u8) standalone — exercises the rgba-only kernel path.
  {
    let mut sink = MixedSinker::<Nv20<BE>>::new(wz, hz)
      .with_rgba(&mut rgba)
      .unwrap();
    sink.set_simd(simd);
    nv20_to_endian::<_, BE>(&src, full_range, matrix, &mut sink).unwrap();
  }
  // rgba_u16 standalone.
  {
    let mut sink = MixedSinker::<Nv20<BE>>::new(wz, hz)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
    sink.set_simd(simd);
    nv20_to_endian::<_, BE>(&src, full_range, matrix, &mut sink).unwrap();
  }

  (rgb, rgb_u16, rgba, rgba_u16, luma, (hh, ss, vv))
}

/// Per-tier equivalence body: SIMD output == scalar output for the
/// numeric RGB / RGBA / luma paths, at one `(BE, matrix, full_range)`
/// point. HSV is intentionally excluded: the crate's HSV contract is
/// `*_to_hsv_row == rgb_to_hsv_row(*_to_rgb_row)` **within a tier**, and
/// `rgb_to_hsv_row` itself is not byte-stable SIMD-vs-scalar (a
/// pre-existing property exercised by the HSV-direct suites); it is
/// covered by [`nv20_hsv_within_tier_equals_rgb_then_hsv`] instead.
fn nv20_assert_simd_eq_scalar<const BE: bool>(w: u32, h: u32, matrix: KernelMatrix, fr: bool) {
  let (yp, uvp) = nv20_random::<BE>(w, h, 0x9E37_79B9 ^ ((BE as u32) << 16) ^ (fr as u32));
  let simd = nv20_run_all::<BE>(&yp, &uvp, w, h, matrix, fr, true);
  let scal = nv20_run_all::<BE>(&yp, &uvp, w, h, matrix, fr, false);
  assert_eq!(
    simd.0, scal.0,
    "rgb SIMD != scalar (BE={BE} fr={fr} {matrix:?})"
  );
  assert_eq!(simd.1, scal.1, "rgb_u16 SIMD != scalar (BE={BE} fr={fr})");
  assert_eq!(simd.2, scal.2, "rgba SIMD != scalar (BE={BE} fr={fr})");
  assert_eq!(simd.3, scal.3, "rgba_u16 SIMD != scalar (BE={BE} fr={fr})");
  assert_eq!(simd.4, scal.4, "luma SIMD != scalar (BE={BE} fr={fr})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_simd_matches_scalar_all_outputs() {
  // Width 1922 forces a scalar tail on every backend
  // (1922 % {16, 32, 64} != 0). Cover full/limited × Bt709/Bt601 × LE/BE.
  let (w, h) = (1922u32, 3u32);
  for &(fr, m) in &[
    (true, KernelMatrix::Bt709),
    (false, KernelMatrix::Bt709),
    (true, KernelMatrix::Bt601),
    (false, KernelMatrix::Bt601),
  ] {
    nv20_assert_simd_eq_scalar::<false>(w, h, m, fr);
    nv20_assert_simd_eq_scalar::<true>(w, h, m, fr);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_hsv_within_tier_equals_rgb_then_hsv() {
  // The crate's HSV contract: WITHIN a tier, NV20 HSV (the sink's
  // `with_hsv` output, the direct `nv20_to_hsv_row_endian` dispatcher, and
  // `rgb_to_hsv_row(nv20_to_rgb_row_endian(...))`) all agree byte-for-byte.
  // Checked for both the scalar (`use_simd = false`) and host SIMD
  // (`use_simd = true`) tiers, LE + BE, full/limited × Bt709/Bt601, with a
  // tail width.
  let (w, h) = (66u32, 3u32);

  fn check<const BE: bool>(w: u32, h: u32, m: KernelMatrix, fr: bool, use_simd: bool) {
    let (yp, uvp) = nv20_random::<BE>(w, h, 0x0BAD_F00D ^ (use_simd as u32));
    let wz = w as usize;
    let hz = h as usize;
    let src = Nv20Frame::<BE>::new(&yp, &uvp, w, h, w, w);

    // Sink with_hsv output.
    let mut sh = std::vec![0u8; wz * hz];
    let mut ss = std::vec![0u8; wz * hz];
    let mut sv = std::vec![0u8; wz * hz];
    {
      let mut sink = MixedSinker::<Nv20<BE>>::new(wz, hz)
        .with_hsv(&mut sh, &mut ss, &mut sv)
        .unwrap();
      sink.set_simd(use_simd);
      nv20_to_endian::<_, BE>(&src, fr, m, &mut sink).unwrap();
    }

    // Row-direct: nv20_to_hsv_row_endian, and rgb_to_hsv_row(nv20 rgb).
    let mut dh = std::vec![0u8; wz * hz];
    let mut ds = std::vec![0u8; wz * hz];
    let mut dv = std::vec![0u8; wz * hz];
    let mut rh = std::vec![0u8; wz * hz];
    let mut rs = std::vec![0u8; wz * hz];
    let mut rv = std::vec![0u8; wz * hz];
    for r in 0..hz {
      let yr = &yp[r * wz..r * wz + wz];
      let uvr = &uvp[r * wz..r * wz + wz];
      nv20_to_hsv_row_endian(
        yr,
        uvr,
        &mut dh[r * wz..r * wz + wz],
        &mut ds[r * wz..r * wz + wz],
        &mut dv[r * wz..r * wz + wz],
        wz,
        m,
        fr,
        use_simd,
        BE,
      );
      let mut rgb = std::vec![0u8; wz * 3];
      nv20_to_rgb_row_endian(yr, uvr, &mut rgb, wz, m, fr, use_simd, BE);
      rgb_to_hsv_row(
        &rgb,
        &mut rh[r * wz..r * wz + wz],
        &mut rs[r * wz..r * wz + wz],
        &mut rv[r * wz..r * wz + wz],
        wz,
        use_simd,
      );
    }

    assert_eq!(
      sh, dh,
      "sink H != direct H (BE={BE} simd={use_simd} fr={fr} {m:?})"
    );
    assert_eq!(ss, ds, "sink S != direct S");
    assert_eq!(sv, dv, "sink V != direct V");
    assert_eq!(dh, rh, "direct H != rgb_to_hsv H");
    assert_eq!(ds, rs, "direct S != rgb_to_hsv S");
    assert_eq!(dv, rv, "direct V != rgb_to_hsv V");
  }

  for &(fr, m) in &[(true, KernelMatrix::Bt709), (false, KernelMatrix::Bt601)] {
    for &simd in &[false, true] {
      check::<false>(w, h, m, fr, simd);
      check::<true>(w, h, m, fr, simd);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_be_equals_le_on_matching_logical_samples() {
  // An NV20 BE frame and an NV20 LE frame carrying the SAME logical
  // samples must decode identically (the kernel normalizes wire order
  // before the low-bit mask). Proves the BE path is wired + correct.
  let (w, h) = (40u32, 4u32);
  let (yl, uvl) = nv20_random::<false>(w, h, 0xABCD_1234);
  let (yb, uvb) = nv20_random::<true>(w, h, 0xABCD_1234);

  for &(fr, m) in &[(true, KernelMatrix::Bt601), (false, KernelMatrix::Bt709)] {
    let le = nv20_run_all::<false>(&yl, &uvl, w, h, m, fr, true);
    let be = nv20_run_all::<true>(&yb, &uvb, w, h, m, fr, true);
    assert_eq!(le.0, be.0, "rgb LE != BE (fr={fr} {m:?})");
    assert_eq!(le.1, be.1, "rgb_u16 LE != BE");
    assert_eq!(le.4, be.4, "luma LE != BE");
    assert_eq!(le.5, be.5, "hsv LE != BE");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // Mid-gray → u16 RGBA: each colour element ≈ 512 (low-bit-packed),
  // alpha = (1 << 10) - 1 = 1023.
  let (yp, uvp) = nv20_solid(16, 8, 512, 512, 512, false);
  let src = Nv20Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv20>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  nv20_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_luma_is_depacked_y_narrowed_to_8bit() {
  // NV20 native luma = (logical_Y & 0x03FF) >> 2. For logical Y = 700,
  // luma = 700 >> 2 = 175 (independent of chroma).
  let (yp, uvp) = nv20_solid(16, 8, 700, 512, 512, false);
  let src = Nv20Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv20>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv20_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 175), "luma must be 700>>2 = 175");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_with_rgb_and_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A: with both rgb + rgba attached, the rgb buffer is the
  // kernel output and rgba is a cheap expand. RGB triples must match the
  // standalone rgb-only run.
  let (yp, uvp) = nv20_random::<false>(48, 8, 0x55AA_33CC);
  let src = Nv20Frame::new(&yp, &uvp, 48, 8, 48, 48);

  let mut rgb_solo = std::vec![0u8; 48 * 8 * 3];
  let mut s_solo = MixedSinker::<Nv20>::new(48, 8)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  nv20_to(&src, true, KernelMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u8; 48 * 8 * 3];
  let mut rgba = std::vec![0u8; 48 * 8 * 4];
  let mut s_combined = MixedSinker::<Nv20>::new(48, 8)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv20_to(&src, true, KernelMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[test]
fn nv20_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Nv20>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected InsufficientRgbaBuffer");
  assert!(matches!(err, MixedSinkerError::InsufficientRgbaBuffer(_)));
}

// ---- Walker parity: `Walker::walk` byte-identical to `nv20_to` --------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
// Sweeps a fixed full-range NV20 frame for kernel parity; the deprecated
// raw setter is intentional here (NV20 pins no range).
#[allow(deprecated)]
fn nv20_walker_matches_direct_le_and_be() {
  let (w, h) = (40u32, 4u32);
  let opts = YuvOptions::from_color_spec(&crate::ColorSpec::of_matrix(KernelMatrix::Bt601))
    .unwrap()
    .with_full_range();

  // LE.
  {
    let (yp, uvp) = nv20_random::<false>(w, h, 0x1357_9BDF);
    let src = Nv20Frame::<false>::new(&yp, &uvp, w, h, w, w);

    let mut rgb_w = std::vec![0u8; (w * h * 3) as usize];
    let mut sink_w = MixedSinker::<Nv20<false>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_w)
      .unwrap();
    <Nv20<false> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut rgb_d = std::vec![0u8; (w * h * 3) as usize];
    let mut sink_d = MixedSinker::<Nv20<false>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_d)
      .unwrap();
    nv20_to(&src, opts.full_range(), opts.matrix(), &mut sink_d).unwrap();

    assert_eq!(rgb_w, rgb_d, "LE Walker::walk != nv20_to");
  }

  // BE.
  {
    let (yp, uvp) = nv20_random::<true>(w, h, 0x2468_ACE0);
    let src = Nv20Frame::<true>::new(&yp, &uvp, w, h, w, w);

    let mut rgb_w = std::vec![0u8; (w * h * 3) as usize];
    let mut sink_w = MixedSinker::<Nv20<true>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_w)
      .unwrap();
    <Nv20<true> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut rgb_d = std::vec![0u8; (w * h * 3) as usize];
    let mut sink_d = MixedSinker::<Nv20<true>>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_d)
      .unwrap();
    nv20_to_endian::<_, true>(&src, opts.full_range(), opts.matrix(), &mut sink_d).unwrap();

    assert_eq!(rgb_w, rgb_d, "BE Walker::walk != nv20_to_endian");
  }
}

// ---- Atomicity (#308): NV20 (broader gate — no direct HSV kernel) ------
//
// NV20's identity `process` hoists an up-front RGB-scratch preflight, like the
// high-bit P-formats. But UNLIKE them NV20 has NO direct YUV→HSV kernel, so HSV
// is ALWAYS derived from the RGB row: the allocating (rgb=None) arm of
// `rgb_row_buf_or_scratch` is reached at the BROADER `want_hsv && !want_rgb`,
// independent of rgba. This test therefore arms the failpoint with luma + HSV
// and NO rgb / NO rgba — a set under which the P-formats route HSV-direct and
// never allocate, but NV20 must preflight. The refusal must surface as
// `AllocationFailed` BEFORE the luma plane is written. `yuva`-gated (shares the
// crate's RGB-scratch failpoint).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv20_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = nv20_solid(16, 8, 512, 512, 512, false);
  let src = Nv20Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Nv20>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = nv20_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}
