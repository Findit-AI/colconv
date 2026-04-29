use super::*;

// ---- Tier 6 — Rgb24 / Bgr24 (Ship 9a) ----------------------------

fn solid_rgb24_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
  }
  buf
}

#[test]
fn rgb24_with_rgb_passes_through_identity() {
  let pix = solid_rgb24_frame(16, 4, 200, 100, 50);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4).with_rgb(&mut out).unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  assert_eq!(out, pix);
}

#[test]
fn rgb24_with_rgba_appends_opaque_alpha() {
  let pix = solid_rgb24_frame(16, 4, 200, 100, 50);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut rgba = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
fn rgb24_with_luma_derives_bt709_full_range() {
  // Pure red full-range BT.709: Y = 0.2126 * 255 ≈ 54.21 → rounded
  // to 54 by Q15 math.
  let pix = solid_rgb24_frame(16, 4, 255, 0, 0);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut luma = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut luma)
    .unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // 0.2126 × 255 = 54.213 → 54 after rounding.
  for &y in &luma {
    assert!(y.abs_diff(54) <= 1, "got Y={y}");
  }
}

#[test]
fn rgb24_with_luma_derives_bt601_full_range() {
  // Pure green BT.601: Y = 0.587 * 255 ≈ 149.685 → 150 rounded.
  let pix = solid_rgb24_frame(16, 4, 0, 255, 0);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut luma = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut luma)
    .unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for &y in &luma {
    assert!(y.abs_diff(150) <= 1, "got Y={y}");
  }
}

#[test]
fn rgb24_with_luma_limited_range_falls_in_studio_band() {
  // Pure white full-range maps to Y=255 full → 235 limited.
  let pix = solid_rgb24_frame(16, 4, 255, 255, 255);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut luma = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut luma)
    .unwrap();
  rgb24_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma {
    assert!((234..=236).contains(&y), "got Y={y}");
  }

  // Pure black → Y=16 (limited-range black floor).
  let black = solid_rgb24_frame(16, 4, 0, 0, 0);
  let src = Rgb24Frame::try_new(&black, 16, 4, 48).unwrap();
  let mut luma = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut luma)
    .unwrap();
  rgb24_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma {
    assert_eq!(y, 16);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb24_with_hsv_matches_existing_kernel() {
  // Pure red → H=0, S=255, V=255 (OpenCV 8-bit HSV uses H ∈ [0, 179]).
  let pix = solid_rgb24_frame(16, 4, 255, 0, 0);
  let src = Rgb24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut h = std::vec![0u8; 16 * 4];
  let mut s = std::vec![0u8; 16 * 4];
  let mut v = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgb24>::new(16, 4)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for i in 0..16 * 4 {
    assert_eq!(h[i], 0, "px {i}");
    assert_eq!(s[i], 255, "px {i}");
    assert_eq!(v[i], 255, "px {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb24_random_input_produces_stable_output() {
  // Smoke test that non-uniform input doesn't crash and produces
  // well-formed output across all sinks.
  let w = 31usize;
  let h = 5usize;
  let mut pix = std::vec![0u8; w * h * 3];
  pseudo_random_u8(&mut pix, 0xC001_C0DE);
  let src = Rgb24Frame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let mut sink = MixedSinker::<Rgb24>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
  rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // RGB is identity-copied.
  assert_eq!(rgb, pix);
  // RGBA's RGB channels match RGB output, and alpha is 0xFF.
  for (i, px) in rgba.chunks(4).enumerate() {
    assert_eq!(px[0], pix[i * 3]);
    assert_eq!(px[1], pix[i * 3 + 1]);
    assert_eq!(px[2], pix[i * 3 + 2]);
    assert_eq!(px[3], 0xFF);
  }
}

// ---- Bgr24 ---------------------------------------------------------

fn solid_bgr24_frame(width: u32, height: u32, b: u8, g: u8, r: u8) -> Vec<u8> {
  // Stored byte order is B, G, R per pixel.
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = b;
    px[1] = g;
    px[2] = r;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr24_with_rgb_swaps_channel_order() {
  // Source byte order B=50, G=100, R=200 → expected RGB output:
  // R=200, G=100, B=50.
  let pix = solid_bgr24_frame(16, 4, 50, 100, 200);
  let src = Bgr24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Bgr24>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [200, 100, 50]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr24_with_rgba_swaps_then_appends_opaque_alpha() {
  let pix = solid_bgr24_frame(16, 4, 50, 100, 200);
  let src = Bgr24Frame::try_new(&pix, 16, 4, 48).unwrap();

  let mut rgba = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Bgr24>::new(16, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr24_luma_matches_rgb24_after_swap() {
  // Same pixel data interpreted as Rgb24 vs Bgr24 should produce the
  // same luma values (the BGR sinker swaps to RGB before deriving Y).
  let bgr_pix = solid_bgr24_frame(16, 4, 50, 100, 200); // B, G, R
  let rgb_pix = solid_rgb24_frame(16, 4, 200, 100, 50); // R, G, B
  let bgr_src = Bgr24Frame::try_new(&bgr_pix, 16, 4, 48).unwrap();
  let rgb_src = Rgb24Frame::try_new(&rgb_pix, 16, 4, 48).unwrap();

  let mut bgr_luma = std::vec![0u8; 16 * 4];
  let mut s_bgr = MixedSinker::<Bgr24>::new(16, 4)
    .with_luma(&mut bgr_luma)
    .unwrap();
  bgr24_to(&bgr_src, true, ColorMatrix::Bt709, &mut s_bgr).unwrap();

  let mut rgb_luma = std::vec![0u8; 16 * 4];
  let mut s_rgb = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut rgb_luma)
    .unwrap();
  rgb24_to(&rgb_src, true, ColorMatrix::Bt709, &mut s_rgb).unwrap();

  assert_eq!(bgr_luma, rgb_luma);
}

// ---- Tier 6 — Rgba / Bgra (Ship 9b) -------------------------------

fn solid_rgba_frame(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
    px[3] = a;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_with_rgb_drops_alpha() {
  // Distinct alpha (0x80) verifies the alpha byte is dropped, not
  // copied into one of the RGB channels.
  let pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80);
  let src = RgbaFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgba>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgba_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [200, 100, 50]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_with_rgba_passes_through_alpha() {
  // Source alpha is 0x80, NOT 0xFF — verifies the sinker preserves
  // caller alpha rather than overwriting with opaque.
  let pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80);
  let src = RgbaFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgba>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgba_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0x80]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_luma_matches_rgb24_after_alpha_drop() {
  // RGBA luma must equal Rgb24 luma on the same RGB triple — alpha
  // never participates in Y'.
  let rgba_pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80);
  let rgb_pix = solid_rgb24_frame(16, 4, 200, 100, 50);
  let rgba_src = RgbaFrame::try_new(&rgba_pix, 16, 4, 64).unwrap();
  let rgb_src = Rgb24Frame::try_new(&rgb_pix, 16, 4, 48).unwrap();

  let mut rgba_luma = std::vec![0u8; 16 * 4];
  let mut s_rgba = MixedSinker::<Rgba>::new(16, 4)
    .with_luma(&mut rgba_luma)
    .unwrap();
  rgba_to(&rgba_src, true, ColorMatrix::Bt709, &mut s_rgba).unwrap();

  let mut rgb_luma = std::vec![0u8; 16 * 4];
  let mut s_rgb = MixedSinker::<Rgb24>::new(16, 4)
    .with_luma(&mut rgb_luma)
    .unwrap();
  rgb24_to(&rgb_src, true, ColorMatrix::Bt709, &mut s_rgb).unwrap();

  assert_eq!(rgba_luma, rgb_luma);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_with_hsv_matches_rgb24_kernel() {
  // HSV from RGBA must equal HSV from the matching RGB triple.
  let rgba_pix = solid_rgba_frame(16, 4, 255, 0, 0, 0x40);
  let rgb_pix = solid_rgb24_frame(16, 4, 255, 0, 0);
  let rgba_src = RgbaFrame::try_new(&rgba_pix, 16, 4, 64).unwrap();
  let rgb_src = Rgb24Frame::try_new(&rgb_pix, 16, 4, 48).unwrap();

  let (mut h1, mut s1, mut v1) = (
    std::vec![0u8; 16 * 4],
    std::vec![0u8; 16 * 4],
    std::vec![0u8; 16 * 4],
  );
  let mut sink_rgba = MixedSinker::<Rgba>::new(16, 4)
    .with_hsv(&mut h1, &mut s1, &mut v1)
    .unwrap();
  rgba_to(&rgba_src, true, ColorMatrix::Bt709, &mut sink_rgba).unwrap();

  let (mut h2, mut s2, mut v2) = (
    std::vec![0u8; 16 * 4],
    std::vec![0u8; 16 * 4],
    std::vec![0u8; 16 * 4],
  );
  let mut sink_rgb = MixedSinker::<Rgb24>::new(16, 4)
    .with_hsv(&mut h2, &mut s2, &mut v2)
    .unwrap();
  rgb24_to(&rgb_src, true, ColorMatrix::Bt709, &mut sink_rgb).unwrap();

  assert_eq!(h1, h2);
  assert_eq!(s1, s2);
  assert_eq!(v1, v2);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_random_input_produces_stable_output() {
  let w = 31usize;
  let h = 5usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xBEEF_F00D);
  let src = RgbaFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let mut sink = MixedSinker::<Rgba>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
  rgba_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // RGB drops alpha (per-pixel R, G, B copied verbatim).
  for (i, px) in rgb.chunks(3).enumerate() {
    assert_eq!(px[0], pix[i * 4]);
    assert_eq!(px[1], pix[i * 4 + 1]);
    assert_eq!(px[2], pix[i * 4 + 2]);
  }
  // RGBA is an identity copy (alpha pass-through).
  assert_eq!(rgba, pix);
}

// ---- Bgra ----------------------------------------------------------

fn solid_bgra_frame(width: u32, height: u32, b: u8, g: u8, r: u8, a: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = b;
    px[1] = g;
    px[2] = r;
    px[3] = a;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgra_with_rgb_swaps_and_drops_alpha() {
  // Source byte order B=50, G=100, R=200, A=0x80 → expected RGB:
  // R=200, G=100, B=50 (alpha dropped).
  let pix = solid_bgra_frame(16, 4, 50, 100, 200, 0x80);
  let src = BgraFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Bgra>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  bgra_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [200, 100, 50]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgra_with_rgba_swaps_with_alpha_passthrough() {
  let pix = solid_bgra_frame(16, 4, 50, 100, 200, 0x80);
  let src = BgraFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Bgra>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgra_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0x80]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgra_luma_matches_rgba_after_swap() {
  // Same RGB triple, opposite channel order — luma must agree.
  let bgra_pix = solid_bgra_frame(16, 4, 50, 100, 200, 0x80); // B, G, R, A
  let rgba_pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80); // R, G, B, A
  let bgra_src = BgraFrame::try_new(&bgra_pix, 16, 4, 64).unwrap();
  let rgba_src = RgbaFrame::try_new(&rgba_pix, 16, 4, 64).unwrap();

  let mut bgra_luma = std::vec![0u8; 16 * 4];
  let mut s_bgra = MixedSinker::<Bgra>::new(16, 4)
    .with_luma(&mut bgra_luma)
    .unwrap();
  bgra_to(&bgra_src, true, ColorMatrix::Bt709, &mut s_bgra).unwrap();

  let mut rgba_luma = std::vec![0u8; 16 * 4];
  let mut s_rgba = MixedSinker::<Rgba>::new(16, 4)
    .with_luma(&mut rgba_luma)
    .unwrap();
  rgba_to(&rgba_src, true, ColorMatrix::Bt709, &mut s_rgba).unwrap();

  assert_eq!(bgra_luma, rgba_luma);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgra_random_input_produces_stable_output() {
  let w = 31usize;
  let h = 5usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xFEED_FACE);
  let src = BgraFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let mut sink = MixedSinker::<Bgra>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
  bgra_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // RGB output: R/G/B come from BGRA byte-2 / byte-1 / byte-0
  // (R↔B swap); alpha dropped.
  for (i, px) in rgb.chunks(3).enumerate() {
    assert_eq!(px[0], pix[i * 4 + 2]);
    assert_eq!(px[1], pix[i * 4 + 1]);
    assert_eq!(px[2], pix[i * 4]);
  }
  // RGBA output: R↔B swap, alpha pass-through.
  for (i, px) in rgba.chunks(4).enumerate() {
    assert_eq!(px[0], pix[i * 4 + 2]);
    assert_eq!(px[1], pix[i * 4 + 1]);
    assert_eq!(px[2], pix[i * 4]);
    assert_eq!(px[3], pix[i * 4 + 3]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both the SIMD main loop AND a scalar tail
  // across every backend block size (16 / 32 / 64). Per-pixel random
  // R/G/B/A means a bad shuffle in any backend produces a measurable
  // diff vs scalar.
  //
  // HSV is intentionally **not** part of the parity check —
  // `rgb_to_hsv_row` uses `_mm_rcp_ps` + Newton-Raphson refinement
  // and is documented to differ ±1 LSB from scalar. The new Ship 9b
  // RGBA shuffle kernels are byte-exact, so RGB / RGBA / luma cover
  // their parity.
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xBEEF_F00D);
  let src = RgbaFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Rgba>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  rgba_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgba>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgba_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "RGB output diverges between SIMD and scalar"
  );
  assert_eq!(
    rgba_simd, rgba_scalar,
    "RGBA output diverges between SIMD and scalar"
  );
  assert_eq!(
    luma_simd, luma_scalar,
    "Luma output diverges between SIMD and scalar"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgra_simd_matches_scalar_with_random_input() {
  // See `rgba_simd_matches_scalar_with_random_input` for rationale.
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xFEED_FACE);
  let src = BgraFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Bgra>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  bgra_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Bgra>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  bgra_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "RGB output diverges between SIMD and scalar"
  );
  assert_eq!(
    rgba_simd, rgba_scalar,
    "RGBA output diverges between SIMD and scalar"
  );
  assert_eq!(
    luma_simd, luma_scalar,
    "Luma output diverges between SIMD and scalar"
  );
}

// ---- Tier 6 — Argb / Abgr (Ship 9c) -------------------------------

fn solid_argb_frame(width: u32, height: u32, a: u8, r: u8, g: u8, b: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = a;
    px[1] = r;
    px[2] = g;
    px[3] = b;
  }
  buf
}

fn solid_abgr_frame(width: u32, height: u32, a: u8, b: u8, g: u8, r: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = a;
    px[1] = b;
    px[2] = g;
    px[3] = r;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn argb_with_rgb_drops_leading_alpha() {
  let pix = solid_argb_frame(16, 4, 0x80, 200, 100, 50);
  let src = ArgbFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Argb>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  argb_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [200, 100, 50]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn argb_with_rgba_rotates_alpha_to_trailing() {
  let pix = solid_argb_frame(16, 4, 0x80, 200, 100, 50);
  let src = ArgbFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Argb>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  argb_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0x80]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn argb_luma_matches_rgba_after_alpha_drop() {
  let argb_pix = solid_argb_frame(16, 4, 0x80, 200, 100, 50);
  let rgba_pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80);
  let argb_src = ArgbFrame::try_new(&argb_pix, 16, 4, 64).unwrap();
  let rgba_src = RgbaFrame::try_new(&rgba_pix, 16, 4, 64).unwrap();

  let mut argb_luma = std::vec![0u8; 16 * 4];
  let mut s_argb = MixedSinker::<Argb>::new(16, 4)
    .with_luma(&mut argb_luma)
    .unwrap();
  argb_to(&argb_src, true, ColorMatrix::Bt709, &mut s_argb).unwrap();

  let mut rgba_luma = std::vec![0u8; 16 * 4];
  let mut s_rgba = MixedSinker::<Rgba>::new(16, 4)
    .with_luma(&mut rgba_luma)
    .unwrap();
  rgba_to(&rgba_src, true, ColorMatrix::Bt709, &mut s_rgba).unwrap();

  assert_eq!(argb_luma, rgba_luma);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn abgr_with_rgb_swaps_and_drops_leading_alpha() {
  let pix = solid_abgr_frame(16, 4, 0x80, 50, 100, 200);
  let src = AbgrFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Abgr>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  abgr_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [200, 100, 50]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn abgr_with_rgba_full_byte_reverse() {
  let pix = solid_abgr_frame(16, 4, 0x80, 50, 100, 200);
  let src = AbgrFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Abgr>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  abgr_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0x80]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn abgr_luma_matches_rgba_after_swap() {
  let abgr_pix = solid_abgr_frame(16, 4, 0x80, 50, 100, 200);
  let rgba_pix = solid_rgba_frame(16, 4, 200, 100, 50, 0x80);
  let abgr_src = AbgrFrame::try_new(&abgr_pix, 16, 4, 64).unwrap();
  let rgba_src = RgbaFrame::try_new(&rgba_pix, 16, 4, 64).unwrap();

  let mut abgr_luma = std::vec![0u8; 16 * 4];
  let mut s_abgr = MixedSinker::<Abgr>::new(16, 4)
    .with_luma(&mut abgr_luma)
    .unwrap();
  abgr_to(&abgr_src, true, ColorMatrix::Bt709, &mut s_abgr).unwrap();

  let mut rgba_luma = std::vec![0u8; 16 * 4];
  let mut s_rgba = MixedSinker::<Rgba>::new(16, 4)
    .with_luma(&mut rgba_luma)
    .unwrap();
  rgba_to(&rgba_src, true, ColorMatrix::Bt709, &mut s_rgba).unwrap();

  assert_eq!(abgr_luma, rgba_luma);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn argb_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop AND scalar tail across
  // every backend block size (16 / 32 / 64). HSV omitted — see
  // `rgba_simd_matches_scalar_with_random_input` for rationale.
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xA53A_F00D);
  let src = ArgbFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Argb>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  argb_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Argb>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  argb_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges (SIMD vs scalar)");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn abgr_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0x5A5A_BABE);
  let src = AbgrFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Abgr>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  abgr_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Abgr>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  abgr_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges (SIMD vs scalar)");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

// ---- Tier 6 — Padding-byte family (Ship 9d) ------------------------

fn solid_padded_frame(width: u32, height: u32, b0: u8, b1: u8, b2: u8, b3: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = b0;
    px[1] = b1;
    px[2] = b2;
    px[3] = b3;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xrgb_with_rgba_forces_alpha_to_ff() {
  // Padding byte is 0x42 (anything non-FF); verify alpha is forced.
  let pix = solid_padded_frame(16, 4, 0x42, 200, 100, 50);
  let src = XrgbFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Xrgb>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  xrgb_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbx_with_rgba_forces_alpha_to_ff() {
  let pix = solid_padded_frame(16, 4, 200, 100, 50, 0x42);
  let src = RgbxFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbx>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgbx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xbgr_with_rgba_swaps_and_forces_alpha() {
  let pix = solid_padded_frame(16, 4, 0x42, 50, 100, 200); // X,B,G,R
  let src = XbgrFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Xbgr>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  xbgr_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgrx_with_rgba_swaps_and_forces_alpha() {
  let pix = solid_padded_frame(16, 4, 50, 100, 200, 0x42); // B,G,R,X
  let src = BgrxFrame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Bgrx>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgrx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [200, 100, 50, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xrgb_luma_matches_argb_after_alpha_drop() {
  // Same RGB triple, padding byte vs alpha byte — luma must agree
  // because both ignore the leading byte.
  let xrgb_pix = solid_padded_frame(16, 4, 0x42, 200, 100, 50);
  let argb_pix = solid_argb_frame(16, 4, 0x42, 200, 100, 50);
  let xrgb_src = XrgbFrame::try_new(&xrgb_pix, 16, 4, 64).unwrap();
  let argb_src = ArgbFrame::try_new(&argb_pix, 16, 4, 64).unwrap();

  let mut xrgb_luma = std::vec![0u8; 16 * 4];
  let mut s_xrgb = MixedSinker::<Xrgb>::new(16, 4)
    .with_luma(&mut xrgb_luma)
    .unwrap();
  xrgb_to(&xrgb_src, true, ColorMatrix::Bt709, &mut s_xrgb).unwrap();

  let mut argb_luma = std::vec![0u8; 16 * 4];
  let mut s_argb = MixedSinker::<Argb>::new(16, 4)
    .with_luma(&mut argb_luma)
    .unwrap();
  argb_to(&argb_src, true, ColorMatrix::Bt709, &mut s_argb).unwrap();

  assert_eq!(xrgb_luma, argb_luma);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xrgb_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop AND scalar tail across
  // every backend block size. HSV omitted — see Ship 9b for the
  // ±1 LSB rationale.
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xFADE_F00D);
  let src = XrgbFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Xrgb>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  xrgb_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Xrgb>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  xrgb_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbx_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xC0DE_BABE);
  let src = RgbxFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Rgbx>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  rgbx_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbx>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbx_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xbgr_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xDEAD_C0DE);
  let src = XbgrFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Xbgr>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  xbgr_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Xbgr>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  xbgr_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgrx_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xCAFE_BEEF);
  let src = BgrxFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Bgrx>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  bgrx_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Bgrx>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  bgrx_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

