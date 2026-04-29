use super::*;

pub(super) fn solid_yuv420p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_copies_y_plane() {
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_only_converts_gray_to_gray() {
  // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_only_allocates_scratch_and_produces_gray_hsv() {
  // Neutral gray → H=0, S=0, V=~128. No RGB buffer provided.
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mixed_all_three_outputs_populated() {
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 200, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  // Luma = Y plane verbatim.
  assert!(luma.iter().all(|&y| y == 200));
  // RGB gray.
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
  // HSV of gray.
  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
}

// ---- RGBA (Ship 8) tests ------------------------------------------------
//
// Yuv420p is the template format for the const-generic-ALPHA
// refactor — proves the kernel writes 4 bytes per pixel correctly,
// alpha defaults to 0xFF (sources with no alpha plane), the RGB
// bytes match what `with_rgb` would have written, and SIMD ≡
// scalar bit-for-bit. Future formats inherit the pattern.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_alpha_is_opaque_for_arbitrary_color() {
  // Non-gray content. The RGB three bytes will vary by pixel; alpha
  // must stay 0xFF because Yuv420p has no alpha plane.
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 180, 60, 200);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for (i, px) in rgba.chunks(4).enumerate() {
    assert_eq!(px[3], 0xFF, "alpha must be opaque (px {i})");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Cross-format invariant: alpha is the *only* difference between
  // the two output buffers. RGBA bytes 0..3 of each pixel must
  // equal the corresponding RGB pixel exactly.
  let w = 32usize;
  let h = 16usize;
  let (yp, up, vp) = solid_yuv420p_frame(w as u32, h as u32, 180, 60, 200);
  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_with_simd_false_matches_with_simd_true() {
  // SIMD ≡ scalar parity for the RGBA path. Widths chosen to force
  // both the SIMD main loop and the scalar tail across every
  // backend block size (16 / 32 / 64). 4:2:0 requires even width,
  // so the tail is exercised via `block + 2/4/6` rather than odd
  // widths.
  for &w in &[16usize, 18, 32, 34, 64, 66, 128, 130] {
    let h = 8usize;
    let (yp, up, vp) = solid_yuv420p_frame(w as u32, h as u32, 180, 60, 200);
    let src = Yuv420pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
    );

    let mut rgba_simd = std::vec![0u8; w * h * 4];
    let mut rgba_scalar = std::vec![0u8; w * h * 4];

    let mut sink_simd = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgba(&mut rgba_simd)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink_simd).unwrap();

    let mut sink_scalar = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgba(&mut rgba_scalar)
      .unwrap();
    sink_scalar.set_simd(false);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink_scalar).unwrap();

    assert_eq!(
      rgba_simd, rgba_scalar,
      "SIMD vs scalar diverged at width {w}"
    );
  }
}

#[test]
fn rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv420p>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv_420_to_rgba_simd_matches_scalar_with_random_yuv() {
  // The earlier `rgba_with_simd_false_matches_with_simd_true` test
  // uses solid Y/U/V, so every pixel collapses to the same RGBA
  // quad and the new RGBA shuffle masks could permute / duplicate
  // lanes within a SIMD block undetected. This test uses
  // **pseudo-random per-pixel Y/U/V** so a bad shuffle in any of
  // `write_rgba_16` (SSE4.1 / AVX2 / AVX-512 / wasm), `vst4q_u8`
  // (NEON), or the scalar-tail handoff produces a measurable
  // diff against the scalar reference. Width 1922 forces both
  // the SIMD main loop AND a scalar tail across every backend
  // block size (16 / 32 / 64). All four `ColorMatrix` variants
  // exercise different `(r_u, r_v, g_u, g_v, b_u, b_v)`
  // coefficient sets, and both ranges exercise the `y_off` /
  // `y_scale` / `c_scale` parameter shape.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; (w / 2) * (h / 2)];
  let mut vp = std::vec![0u8; (w / 2) * (h / 2)];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuv420p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv420p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv420p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv420p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      // Locate the first divergence to make backend-bug
      // diagnosis tractable instead of dumping ~30 KB of bytes.
      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) \
             for matrix={matrix:?} full_range={full_range}: \
             simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_with_hsv_uses_user_buffer_not_scratch() {
  // When caller provides RGB, the scratch should remain empty (Vec len 0).
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 100, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert_eq!(
    sink.rgb_scratch.len(),
    0,
    "scratch should stay unallocated when RGB buffer is provided"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn with_simd_false_matches_with_simd_true() {
  // A/B test: same frame, one sinker forces scalar, the other uses
  // SIMD. NEON is bit‑exact to scalar so outputs must match.
  let w = 32usize;
  let h = 16usize;
  let (yp, up, vp) = solid_yuv420p_frame(w as u32, h as u32, 180, 60, 200);
  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];

  let mut sink_simd = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  let mut sink_scalar = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  assert!(sink_simd.simd());
  assert!(!sink_scalar.simd());

  yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
  yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn stride_padded_source_reads_correct_pixels() {
  // 16×8 frame, Y stride 32 (padding), chroma stride 16.
  let w = 16usize;
  let h = 8usize;
  let y_stride = 32usize;
  let c_stride = 16usize;
  let mut yp = std::vec![0xFFu8; y_stride * h]; // padding = 0xFF
  let mut up = std::vec![0xFFu8; c_stride * h / 2];
  let mut vp = std::vec![0xFFu8; c_stride * h / 2];
  // Write actual pixel data in non-padding bytes.
  for row in 0..h {
    for x in 0..w {
      yp[row * y_stride + x] = 50;
    }
  }
  for row in 0..h / 2 {
    for x in 0..w / 2 {
      up[row * c_stride + x] = 128;
      vp[row * c_stride + x] = 128;
    }
  }

  let src = Yuv420pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    y_stride as u32,
    c_stride as u32,
    c_stride as u32,
  );

  let mut luma = std::vec![0u8; w * h];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(
    luma.iter().all(|&y| y == 50),
    "padding bytes leaked into output"
  );
}
