use super::*;

// ---- Yuva422p family (Ship 8b‑3) -----------------------------------

fn solid_yuva422p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
  a: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  // 4:2:2: chroma full-height (only horizontal subsampling).
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
    std::vec![a; w * h],
  )
}

pub(super) fn solid_yuva422p_frame_u16(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva422p_frame(16, 8, 128, 128, 128, 128);
  let src = Yuva422pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 128, "alpha pass-through");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_with_rgb_alpha_drop_matches_yuv422p() {
  let (yp_a, up_a, vp_a, ap) = solid_yuva422p_frame(16, 8, 180, 60, 200, 200);
  let yuv = Yuv422pFrame::try_new(&yp_a, &up_a, &vp_a, 16, 8, 16, 8, 8).unwrap();
  let yuva = Yuva422pFrame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv422p_to(&yuv, true, s_yuv.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva422p>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva422p_to(&yuva, true, s_yuva.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p9_rgba_u8_with_source_alpha_passes_through() {
  // 9-bit mid-gray (Y=U=V=256) and mid-alpha (A=128 → u8 alpha = 128 >> 1 = 64).
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 256, 256, 256, 128);
  let src = Yuva422p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p9_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 64, "alpha = 128 >> (9-8) = 64");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p9_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 256, 256, 256, 128);
  let src = Yuva422p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p9_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 512, 512, 512, 512);
  let src = Yuva422p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 512 >> (10-8) = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 512, 512, 512, 512);
  let src = Yuva422p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 512, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_rgba_u8_with_source_alpha_passes_through() {
  // 16-bit full-range Y=U=V=32768 (mid-gray) + alpha=32768 → u8 alpha = 32768 >> 8 = 128.
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva422p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 32768 >> 8 = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_rgba_u16_native_depth_full_range() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva422p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 32768, "alpha at native depth");
  }
}

// ---- Yuva422p12 SIMD-vs-scalar parity (Ship 8b‑4) -----------------
//
// Yuva422p12 routes through the BITS-generic `yuv_420p_n_*<12>` row
// kernels via the new yuva420p12 dispatchers. Width 1922 enters and
// exits the main SIMD loop on every backend block size (NEON 16,
// AVX2 32, AVX-512 64) so a bad 12-bit alpha shift, chroma
// duplication, or RGBA interleave on any tier shows up here.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_rgba_u8_simd_matches_scalar_with_random_yuva() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; (w / 2) * h];
  let mut vp = std::vec![0u16; (w / 2) * h];
  let mut ap = std::vec![0u16; w * h];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 12);
  let src = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
    w as u32,
  )
  .unwrap();

  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuva422p12_to(&src, full_range, s_simd.set_kernel_matrix(matrix)).unwrap();

      let mut s_scalar = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva422p12_to(&src, full_range, s_scalar.set_kernel_matrix(matrix)).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva422p12 RGBA u8 SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn yuva422p12_rgba_u16_simd_matches_scalar_with_random_yuva() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; (w / 2) * h];
  let mut vp = std::vec![0u16; (w / 2) * h];
  let mut ap = std::vec![0u16; w * h];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 12);
  let src = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
    w as u32,
  )
  .unwrap();

  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u16; w * h * 4];
      let mut rgba_scalar = std::vec![0u16; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba_u16(&mut rgba_simd)
        .unwrap();
      yuva422p12_to(&src, full_range, s_simd.set_kernel_matrix(matrix)).unwrap();

      let mut s_scalar = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba_u16(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva422p12_to(&src, full_range, s_scalar.set_kernel_matrix(matrix)).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva422p12 RGBA u16 SIMD ≠ scalar at element {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

// ---- Yuva422p Strategy A+ correctness (spec § 6.1) ----------------------

/// Strategy A+ correctness: combo path output == scalar inline-α kernel output
/// at all (range, matrix) combinations. See spec § 6.1.
///
/// Yuva422p uses the same per-row chroma layout as Yuva420p (half-width U/V)
/// but chroma is full-height. The A+ path uses expand_rgb_to_rgba_row +
/// copy_alpha_plane_u8 which must be byte-identical to yuv_420_to_rgba_with_alpha_src_row.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;

  let mut yp = std::vec![0u8; width * height];
  let mut up = std::vec![0u8; cw * height];
  let mut vp = std::vec![0u8; cw * height];
  let mut ap = std::vec![0u8; width * height];
  pseudo_random_u8(&mut yp, 0xC0FFEE_u32);
  pseudo_random_u8(&mut up, 0xBADF00D_u32);
  pseudo_random_u8(&mut vp, 0xFEEDFACE_u32);
  pseudo_random_u8(&mut ap, 0xA1FA5EED_u32);

  let frame = Yuva422pFrame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();

  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      // Sinker combo path (A+).
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        yuva422p_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }

      // Reference: scalar inline-α kernel per row.
      // Yuva422p chroma is full-height (row r uses u_row = &up[r*cw..]).
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420_to_rgb_row(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420_to_rgba_with_alpha_src_row(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }

      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p A+ RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p A+ RGBA diverges from scalar inline-α (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

// ---- Yuva422p9/10/12/16 Strategy A+ correctness (spec § 6.1) ------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p9_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE_u32, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D_u32, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE_u32, 9);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA5EED_u32, 9);
  let frame = Yuva422p9Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p9>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        yuva422p9_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_row::<9, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<9, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p9 A+ u8 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p9 A+ u8 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p9_strategy_a_plus_u16_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xDEADBEEF_u32, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0xBAADC0DE_u32, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0xCAFEBABE_u32, 9);
  pseudo_random_u16_low_n_bits(&mut ap, 0x1337C0DE_u32, 9);
  let frame = Yuva422p9Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u16; width * height * 3];
      let mut sinker_rgba = std::vec![0u16; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p9>::new(width, height)
          .with_rgb_u16(&mut sinker_rgb)
          .unwrap()
          .with_rgba_u16(&mut sinker_rgba)
          .unwrap();
        yuva422p9_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u16; width * height * 3];
      let mut inline_rgba = std::vec![0u16; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_u16_row::<9, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<9, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p9 A+ u16 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p9 A+ u16 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE_u32, 10);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D_u32, 10);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE_u32, 10);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA5EED_u32, 10);
  let frame = Yuva422p10Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p10>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        yuva422p10_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_row::<10, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<10, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p10 A+ u8 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p10 A+ u8 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_strategy_a_plus_u16_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xDEADBEEF_u32, 10);
  pseudo_random_u16_low_n_bits(&mut up, 0xBAADC0DE_u32, 10);
  pseudo_random_u16_low_n_bits(&mut vp, 0xCAFEBABE_u32, 10);
  pseudo_random_u16_low_n_bits(&mut ap, 0x1337C0DE_u32, 10);
  let frame = Yuva422p10Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u16; width * height * 3];
      let mut sinker_rgba = std::vec![0u16; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p10>::new(width, height)
          .with_rgb_u16(&mut sinker_rgb)
          .unwrap()
          .with_rgba_u16(&mut sinker_rgba)
          .unwrap();
        yuva422p10_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u16; width * height * 3];
      let mut inline_rgba = std::vec![0u16; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_u16_row::<10, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<10, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p10 A+ u16 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p10 A+ u16 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE_u32, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D_u32, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE_u32, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA5EED_u32, 12);
  let frame = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p12>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        yuva422p12_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_row::<12, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<12, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p12 A+ u8 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p12 A+ u8 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_strategy_a_plus_u16_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xDEADBEEF_u32, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xBAADC0DE_u32, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xCAFEBABE_u32, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0x1337C0DE_u32, 12);
  let frame = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u16; width * height * 3];
      let mut sinker_rgba = std::vec![0u16; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p12>::new(width, height)
          .with_rgb_u16(&mut sinker_rgb)
          .unwrap()
          .with_rgba_u16(&mut sinker_rgba)
          .unwrap();
        yuva422p12_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u16; width * height * 3];
      let mut inline_rgba = std::vec![0u16; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p_n_to_rgb_u16_row::<12, false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<12, false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p12 A+ u16 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p12 A+ u16 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

// Yuva422p16 uses the dedicated 16-bit scalar family.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D_u32, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA5EED_u32, 16);
  let frame = Yuva422p16Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p16>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        yuva422p16_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p16_to_rgb_row::<false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p16_to_rgba_with_alpha_src_row::<false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p16 A+ u8 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p16 A+ u8 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_strategy_a_plus_u16_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;
  let cw = width / 2;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; cw * height];
  let mut vp = std::vec![0u16; cw * height];
  let mut ap = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xDEADBEEF_u32, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xBAADC0DE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xCAFEBABE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut ap, 0x1337C0DE_u32, 16);
  let frame = Yuva422p16Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    cw as u32,
    cw as u32,
    width as u32,
  )
  .unwrap();
  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      let mut sinker_rgb = std::vec![0u16; width * height * 3];
      let mut sinker_rgba = std::vec![0u16; width * height * 4];
      {
        let mut sink = MixedSinker::<Yuva422p16>::new(width, height)
          .with_rgb_u16(&mut sinker_rgb)
          .unwrap()
          .with_rgba_u16(&mut sinker_rgba)
          .unwrap();
        yuva422p16_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
      }
      let mut inline_rgb = std::vec![0u16; width * height * 3];
      let mut inline_rgba = std::vec![0u16; width * height * 4];
      for r in 0..height {
        let y_row = &yp[r * width..(r + 1) * width];
        let u_row = &up[r * cw..(r + 1) * cw];
        let v_row = &vp[r * cw..(r + 1) * cw];
        let a_row = &ap[r * width..(r + 1) * width];
        crate::row::scalar::yuv_420p16_to_rgb_u16_row::<false>(
          y_row,
          u_row,
          v_row,
          &mut inline_rgb[r * width * 3..(r + 1) * width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row::<false>(
          y_row,
          u_row,
          v_row,
          a_row,
          &mut inline_rgba[r * width * 4..(r + 1) * width * 4],
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(
        sinker_rgb, inline_rgb,
        "Yuva422p16 A+ u16 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "Yuva422p16 A+ u16 RGBA diverges (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_direct_luma_u16_with_hsv_no_rgb_buffer_writes_both() {
  // #263 PR 8: `with_luma_u16` + `with_hsv` with NO rgb / rgba plane
  // attached routes HSV through the direct `yuv_420_to_hsv_row` kernel
  // (4:2:2 reuses the half-chroma 4:2:0 kernel) — RGB-free (no rgb
  // scratch). Both outputs must be produced: luma_u16 is the zero-extended
  // Y; HSV must match the RGB-attached oracle (same kernel — direct vs
  // derived-from-RGB is the only difference).
  let w = 16usize;
  let h = 8usize;
  // 4:2:2: chroma full-height (only horizontal subsampling).
  let cw = w / 2;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; cw * h];
  let mut vp = std::vec![0u8; cw * h];
  let mut ap = std::vec![0u8; w * h];
  pseudo_random_u8(&mut yp, 0x7E57_C0DE);
  pseudo_random_u8(&mut up, 0x7E57_CAFE);
  pseudo_random_u8(&mut vp, 0x7E57_BEEF);
  pseudo_random_u8(&mut ap, 0x7E57_5EED);
  let src = Yuva422pFrame::try_new(
    &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, cw as u32, cw as u32, w as u32,
  )
  .unwrap();

  // No-rgb scratch path: luma_u16 + hsv only.
  let mut lu16 = std::vec![0u16; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Yuva422p>::new(w, h)
      .with_luma_u16(&mut lu16)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    yuva422p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    // White-box: the direct HSV path is RGB-free — the rgb scratch is
    // never grown.
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "HSV-only direct path must not allocate the rgb scratch"
    );
  }
  let lu16_ref: std::vec::Vec<u16> = yp.iter().map(|&b| b as u16).collect();
  assert_eq!(lu16, lu16_ref, "no-rgb direct luma_u16 == zero-extended Y");

  // Oracle: same source with rgb attached (HSV derives from the caller
  // RGB buffer) — HSV must be identical.
  let mut rgb = std::vec![0u8; w * h * 3];
  let mut oh = std::vec![0u8; w * h];
  let mut os = std::vec![0u8; w * h];
  let mut ov = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Yuva422p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut oh, &mut os, &mut ov)
      .unwrap();
    yuva422p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(hh, oh, "direct H == rgb-attached H");
  assert_eq!(ss, os, "direct S == rgb-attached S");
  assert_eq!(vv, ov, "direct V == rgb-attached V");
  assert!(
    hh.iter().chain(ss.iter()).chain(vv.iter()).any(|&b| b != 0),
    "HSV direct path produced no output"
  );
}
