use super::*;

// ---- NV21 MixedSinker ---------------------------------------------------

pub(super) fn solid_nv21_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let ch = h / 2;
  // VU row payload = `width` bytes = `width/2` interleaved V/U pairs
  // (V first).
  let mut vu = std::vec![0u8; w * ch];
  for row in 0..ch {
    for i in 0..w / 2 {
      vu[row * w + i * 2] = v;
      vu[row * w + i * 2 + 1] = u;
    }
  }
  (std::vec![y; w * h], vu)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_luma_only_copies_y_plane() {
  let (yp, vup) = solid_nv21_frame(16, 8, 42, 128, 128);
  let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv21>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_rgb_only_converts_gray_to_gray() {
  let (yp, vup) = solid_nv21_frame(16, 8, 128, 128, 128);
  let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv21>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv21_mixed_all_three_outputs_populated() {
  let (yp, vup) = solid_nv21_frame(16, 8, 200, 128, 128);
  let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv21>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 200));
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_matches_nv12_mixed_sinker_with_swapped_chroma() {
  // Cross-format guarantee: an NV21 frame built from the same U / V
  // bytes as an NV12 frame (just byte-swapped in the chroma plane)
  // must produce identical RGB output via MixedSinker.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;

  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = ((c + r * 53) & 0xFF) as u8; // U
      uvp[r * ws + 2 * c + 1] = ((c + r * 71) & 0xFF) as u8; // V
    }
  }
  // Byte-swap each chroma pair to get the VU-ordered stream.
  let mut vup: Vec<u8> = uvp.clone();
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      vup[r * ws + 2 * c] = uvp[r * ws + 2 * c + 1];
      vup[r * ws + 2 * c + 1] = uvp[r * ws + 2 * c];
    }
  }

  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);
  let nv21_src = Nv21Frame::new(&yp, &vup, w, h, w, w);

  let mut rgb_nv12 = std::vec![0u8; ws * hs * 3];
  let mut rgb_nv21 = std::vec![0u8; ws * hs * 3];
  let mut s_nv12 = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgb(&mut rgb_nv12)
    .unwrap();
  let mut s_nv21 = MixedSinker::<Nv21>::new(ws, hs)
    .with_rgb(&mut rgb_nv21)
    .unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();
  nv21_to(&nv21_src, false, ColorMatrix::Bt709, &mut s_nv21).unwrap();

  assert_eq!(rgb_nv12, rgb_nv21);
}

// ---- NV21 RGBA (Ship 8 PR 2) tests --------------------------------------
//
// Mirrors the NV12 RGBA tests. The cross-format invariant against
// NV12 RGBA (with byte-swapped chroma) catches the case where
// SWAP_UV is wired through correctly for the RGB path but not the
// RGBA path.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, vup) = solid_nv21_frame(16, 8, 128, 128, 128);
  let src = Nv21Frame::new(&yp, &vup, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv21>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv21_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32usize;
  let h = 16usize;
  let (yp, vup) = solid_nv21_frame(w as u32, h as u32, 180, 60, 200);
  let src = Nv21Frame::new(&yp, &vup, w as u32, h as u32, w as u32, w as u32);

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Nv21>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv21_rgba_simd_matches_scalar_with_random_yuv() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut vup = std::vec![0u8; w * (h / 2)];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut vup, 0xCAFE_F00D);
  let src = Nv21Frame::new(&yp, &vup, w as u32, h as u32, w as u32, w as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv21>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv21_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv21>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv21_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "NV21 RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn nv21_rgba_matches_nv12_rgba_with_swapped_chroma() {
  // Cross-format invariant on the RGBA path. Same shape as
  // `nv21_matches_nv12_mixed_sinker_with_swapped_chroma` for RGB:
  // building NV21 from NV12's bytes with the chroma pairs swapped
  // must produce byte-identical RGBA. Catches cases where SWAP_UV
  // is honored for RGB but not RGBA.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;

  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = ((c + r * 53) & 0xFF) as u8;
      uvp[r * ws + 2 * c + 1] = ((c + r * 71) & 0xFF) as u8;
    }
  }
  let mut vup: Vec<u8> = uvp.clone();
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      vup[r * ws + 2 * c] = uvp[r * ws + 2 * c + 1];
      vup[r * ws + 2 * c + 1] = uvp[r * ws + 2 * c];
    }
  }

  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);
  let nv21_src = Nv21Frame::new(&yp, &vup, w, h, w, w);

  let mut rgba_nv12 = std::vec![0u8; ws * hs * 4];
  let mut rgba_nv21 = std::vec![0u8; ws * hs * 4];
  let mut s_nv12 = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgba(&mut rgba_nv12)
    .unwrap();
  let mut s_nv21 = MixedSinker::<Nv21>::new(ws, hs)
    .with_rgba(&mut rgba_nv21)
    .unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();
  nv21_to(&nv21_src, false, ColorMatrix::Bt709, &mut s_nv21).unwrap();

  assert_eq!(rgba_nv12, rgba_nv21);
}
