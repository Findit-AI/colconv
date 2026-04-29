use super::*;
use crate::{ColorMatrix, frame::*, raw::*, yuv::*};

// Per-format-family submodules. Each houses tests + format-local
// helpers (`solid_*_frame` builders); cross-cutting helpers
// (`pseudo_random_u8`, `pseudo_random_u16_low_n_bits`) live at
// module scope below and are re-exported as `pub(super)` so the
// submodules can pull them via `use super::*;`.
mod bayer;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod yuva;

fn solid_yuv420p_frame(
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

// ---- NV12 ---------------------------------------------------------------

fn solid_nv12_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let ch = h / 2;
  // UV row payload = `width` bytes = `width/2` interleaved UV pairs.
  let mut uv = std::vec![0u8; w * ch];
  for row in 0..ch {
    for i in 0..w / 2 {
      uv[row * w + i * 2] = u;
      uv[row * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 128, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv12_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 200, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv12_with_simd_false_matches_with_simd_true() {
  // 32×16 pseudo-random frame so the SIMD path exercises its main
  // loop and the scalar path processes the full width too.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uvp: Vec<u8> = (0..w * h / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut sink_simd = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  let mut sink_scalar = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
  nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar);
}

// ---- preflight buffer-size errors ------------------------------------
//
// Undersized RGB / luma / HSV buffers must be rejected at attachment
// time, not part-way through processing. Catching the mistake before
// any rows are written avoids partially-mutated caller buffers
// flagged by the adversarial review. With the fallible API these
// surface as `Err(MixedSinkerError::*BufferTooShort)` / `HsvPlaneTooShort`.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_rgb_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3 - 1]; // 1 byte short
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RgbBufferTooShort {
      expected: 16 * 8 * 3,
      actual: 16 * 8 * 3 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_luma_returns_err() {
  let mut luma = std::vec![0u8; 16 * 8 - 1];
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::LumaBufferTooShort {
      expected: 16 * 8,
      actual: 16 * 8 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_hsv_returns_err() {
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8 - 1]; // V plane short
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::HsvPlaneTooShort {
      which: HsvPlane::V,
      expected: 16 * 8,
      actual: 16 * 8 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn taller_frame_returns_err_before_any_row_written() {
  // Sink sized for 16×8, feed a 16×10 frame. `begin_frame` returns
  // `Err(DimensionMismatch)` before row 0 — no partial writes.
  let (yp, up, vp) = solid_yuv420p_frame(16, 10, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 10, 16, 8, 8);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::DimensionMismatch {
      configured_w: 16,
      configured_h: 8,
      frame_w: 16,
      frame_h: 10,
    }
  );
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn shorter_frame_returns_err_before_any_row_written() {
  // Sink sized 16×8, frame is 16×4. Without the `begin_frame`
  // preflight, the walker would silently process 4 rows and leave
  // rows 4..7 stale from the previous frame. Preflight returns
  // `Err(DimensionMismatch)` with no side effects.
  let (yp, up, vp) = solid_yuv420p_frame(16, 4, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 4, 16, 8, 8);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::DimensionMismatch {
      configured_w: 16,
      configured_h: 8,
      frame_w: 16,
      frame_h: 4,
    }
  );
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_width_mismatch_returns_err() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 32 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(32, 8).with_rgb(&mut rgb).unwrap();
  let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(
    matches!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 32,
        frame_w: 16,
        ..
      }
    ),
    "unexpected error variant: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_width_mismatch_returns_err() {
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 32 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(32, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(
    matches!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 32,
        frame_w: 16,
        ..
      }
    ),
    "unexpected error variant: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_shorter_frame_returns_err_before_any_row_written() {
  let (yp, uvp) = solid_nv12_frame(16, 4, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 4, 16, 16);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

/// Sanity check that an Infallible sink (compile-time proof of
/// no-error) compiles and runs. Mirrors the trait-docs pattern.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn infallible_sink_compiles_and_runs() {
  use core::convert::Infallible;

  struct RowCounter(usize);
  impl PixelSink for RowCounter {
    type Input<'a> = Yuv420pRow<'a>;
    type Error = Infallible;
    fn process(&mut self, _row: Yuv420pRow<'_>) -> Result<(), Infallible> {
      self.0 += 1;
      Ok(())
    }
  }
  impl Yuv420pSink for RowCounter {}

  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let mut counter = RowCounter(0);
  // `Result<(), Infallible>` — the compiler knows Err is
  // uninhabited, so `.unwrap()` here is free and infallible.
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut counter).unwrap();
  assert_eq!(counter.0, 8);
}

// ---- direct process() bypass paths ----------------------------------
//
// The walker normally guarantees (a) begin_frame runs first and
// validates frame dimensions, (b) row.y()/u/v/uv slices have the
// right length, (c) `idx < height`. A direct `process` call can
// break any of these. The defense-in-depth checks in `process`
// must return a specific error variant, not panic — verified here
// by constructing rows manually and calling `process`.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_short_y_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  // Build a row with a 15-byte Y slice (wrong — sink configured for 16).
  let y = [0u8; 15];
  let u = [128u8; 8];
  let v = [128u8; 8];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Y,
      row: 0,
      expected: 16,
      actual: 15,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_short_u_half() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 16];
  let u = [128u8; 7]; // expected 8
  let v = [128u8; 8];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::UHalf,
      row: 0,
      expected: 8,
      actual: 7,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 16];
  let u = [128u8; 8];
  let v = [128u8; 8];
  // idx = 8 exceeds configured height 8 — would otherwise panic on
  // `rgb[idx * w * 3 ..]` indexing.
  let row = Yuv420pRow::new(&y, &u, &v, 8, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_odd_width_sink_returns_err_at_begin_frame() {
  // A sink configured with an odd width would later panic inside
  // `yuv_420_to_rgb_row` (which asserts `width & 1 == 0`). The
  // fallible API surfaces this as `OddWidth` at frame start — no
  // rows are processed, no panic. Width=15, height=8 — matching
  // frame so `DimensionMismatch` can't fire first.
  let w = 15usize;
  let h = 8usize;
  let y = std::vec![0u8; w * h];
  let u = std::vec![128u8; w.div_ceil(2) * h / 2 + 8]; // any valid size
  let v = std::vec![128u8; w.div_ceil(2) * h / 2 + 8];
  // Build the Frame separately — Yuv420pFrame rejects odd width
  // too, so we can't construct a 15-wide frame. That's fine: we
  // only need to hit `begin_frame`, which takes (width, height)
  // parameters directly. Call it manually.
  let mut rgb = std::vec![0u8; 16 * 8 * 3]; // Dummy; not touched.
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  // Silence unused-vec warnings — these would have been the plane data.
  let _ = (y, u, v);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_odd_width_sink_returns_err_at_direct_process() {
  // Direct `process` caller bypassing `begin_frame`. Process must
  // still reject odd width before calling the kernel.
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(15, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 15];
  let u = [128u8; 7]; // ceil(15/2) = 8; 7 triggers the width check first
  let v = [128u8; 7];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_odd_width_sink_returns_err_at_begin_frame() {
  let w = 15usize;
  let h = 8usize;
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(w, h).with_rgb(&mut rgb).unwrap();
  let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_odd_width_sink_returns_err_at_direct_process() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(15, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 15];
  let uv = [128u8; 15];
  let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_process_rejects_short_uv_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 15]; // expected 16
  let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::UvHalf,
      row: 0,
      expected: 16,
      actual: 15,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 16];
  let row = Nv12Row::new(&y, &uv, 8, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_matches_yuv420p_mixed_sinker() {
  // Cross-format guarantee: an NV12 frame built from the same U / V
  // bytes as a Yuv420p frame produces byte-identical RGB output via
  // MixedSinker on both families.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let up: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let vp: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  // Build NV12 UV plane: chroma row r, column c → uv[r * w + 2*c] = U,
  // uv[r * w + 2*c + 1] = V, where U / V come from the same (r, c)
  // sample of the planar fixture above.
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = up[r * (ws / 2) + c];
      uvp[r * ws + 2 * c + 1] = vp[r * (ws / 2) + c];
    }
  }

  let yuv420p_src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);

  let mut rgb_yuv420p = std::vec![0u8; ws * hs * 3];
  let mut rgb_nv12 = std::vec![0u8; ws * hs * 3];
  let mut s_yuv = MixedSinker::<Yuv420p>::new(ws, hs)
    .with_rgb(&mut rgb_yuv420p)
    .unwrap();
  let mut s_nv = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgb(&mut rgb_nv12)
    .unwrap();
  yuv420p_to(&yuv420p_src, false, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv).unwrap();

  assert_eq!(rgb_yuv420p, rgb_nv12);
}

// ---- NV12 RGBA (Ship 8 PR 2) tests --------------------------------------
//
// Mirrors the Yuv420p RGBA test set. Adds a cross-format invariant
// proving NV12 RGBA is byte-identical to Yuv420p RGBA when fed the
// same pixels — catches U/V swap bugs in the new RGBA path that
// a pure RGB-path test would miss.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 128, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv12_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32usize;
  let h = 16usize;
  let (yp, uvp) = solid_nv12_frame(w as u32, h as u32, 180, 60, 200);
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv12_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv12>::new(16, 8).with_rgba(&mut rgba_short);
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
fn nv12_rgba_simd_matches_scalar_with_random_yuv() {
  // Pseudo-random per-pixel YUV across all 4 matrices × both
  // ranges. Width 1922 forces both the SIMD main loop AND a scalar
  // tail across every backend block size (16 / 32 / 64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; w * (h / 2)];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv12>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv12_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv12>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv12_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "NV12 RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn nv12_rgba_matches_yuv420p_rgba_with_same_pixels() {
  // Cross-format invariant: NV12 RGBA byte-identical to Yuv420p
  // RGBA when the chroma is the same. Mirrors the existing
  // `nv12_matches_yuv420p_mixed_sinker` RGB-path test for the new
  // RGBA path. Catches U/V swap bugs in the NV12 RGBA kernel that
  // would silently differ from the planar reference.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let up: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let vp: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = up[r * (ws / 2) + c];
      uvp[r * ws + 2 * c + 1] = vp[r * (ws / 2) + c];
    }
  }

  let yuv420p_src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);

  let mut rgba_yuv420p = std::vec![0u8; ws * hs * 4];
  let mut sink_yuv420p = MixedSinker::<Yuv420p>::new(ws, hs)
    .with_rgba(&mut rgba_yuv420p)
    .unwrap();
  yuv420p_to(&yuv420p_src, true, ColorMatrix::Bt709, &mut sink_yuv420p).unwrap();

  let mut rgba_nv12 = std::vec![0u8; ws * hs * 4];
  let mut sink_nv12 = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgba(&mut rgba_nv12)
    .unwrap();
  nv12_to(&nv12_src, true, ColorMatrix::Bt709, &mut sink_nv12).unwrap();

  assert_eq!(rgba_yuv420p, rgba_nv12);
}

// ---- NV16 MixedSinker ---------------------------------------------------
//
// 4:2:2: chroma is half-width, full-height. Per-row math is
// identical to NV12 (the impl calls `nv12_to_rgb_row`), so the
// tests mirror the NV12 set and add a cross-layout parity check
// against an NV12-shaped frame whose chroma rows are each
// duplicated (simulating 4:2:0 from 4:2:2 by vertical downsampling).

fn solid_nv16_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // NV16 UV is full-height (h rows, not h/2).
  let mut uv = std::vec![0u8; w * h];
  for row in 0..h {
    for i in 0..w / 2 {
      uv[row * w + i * 2] = u;
      uv[row * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 42, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 128, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv16>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 200, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_with_simd_false_matches_with_simd_true() {
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uvp: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut sink_simd = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  let mut sink_scalar = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
  nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_matches_nv12_mixed_sinker_with_duplicated_chroma() {
  // Cross-layout parity: if we build an NV12 frame whose `uv_half`
  // plane contains only the even NV16 chroma rows (row 0, 2, 4, …),
  // the two frames must produce identical RGB output at every Y
  // row. This validates that NV16's walker + NV12's row primitive
  // yield the right 4:2:2 semantics (one UV row per Y row) on a
  // 4:2:0 reference that shares chroma across row pairs.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv16: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  // Build NV12 chroma by sampling only even NV16 chroma rows.
  let mut uv_nv12 = std::vec![0u8; w * h / 2];
  for c_row in 0..h / 2 {
    let src_row = c_row * 2; // even NV16 chroma rows
    uv_nv12[c_row * w..(c_row + 1) * w].copy_from_slice(&uv_nv16[src_row * w..(src_row + 1) * w]);
  }
  // …and make the NV16 odd chroma rows match their even neighbors so
  // the 4:2:0 vertical upsample (same chroma for row pairs) matches
  // what NV16 carries through.
  let mut uv_nv16_aligned = uv_nv16.clone();
  for c_row in 0..h / 2 {
    let even_row = c_row * 2;
    let odd_row = even_row + 1;
    let (even, odd) = uv_nv16_aligned.split_at_mut(odd_row * w);
    odd[..w].copy_from_slice(&even[even_row * w..even_row * w + w]);
  }
  let nv16_src = Nv16Frame::new(
    &yp,
    &uv_nv16_aligned,
    w as u32,
    h as u32,
    w as u32,
    w as u32,
  );
  let nv12_src = Nv12Frame::new(&yp, &uv_nv12, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_nv16 = std::vec![0u8; w * h * 3];
  let mut rgb_nv12 = std::vec![0u8; w * h * 3];
  let mut s_nv16 = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_nv16)
    .unwrap();
  let mut s_nv12 = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_nv12)
    .unwrap();
  nv16_to(&nv16_src, false, ColorMatrix::Bt709, &mut s_nv16).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();

  assert_eq!(rgb_nv16, rgb_nv12);
}

// ---- NV16 RGBA (Ship 8 PR 3) tests --------------------------------------
//
// NV16 reuses the NV12 `_to_rgba_row` dispatcher (4:2:2's row
// contract is identical to NV12's). Tests mirror the NV12 set;
// the cross-format invariant against NV12 (with duplicated
// chroma rows so 4:2:0 vertical upsample matches NV16's per-row
// chroma) catches any wiring regression specific to the NV16
// walker that the kernel-level tests don't cover.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 128, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32usize;
  let h = 16usize;
  let (yp, uvp) = solid_nv16_frame(w as u32, h as u32, 180, 60, 200);
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv16_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv16>::new(16, 8).with_rgba(&mut rgba_short);
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
fn nv16_rgba_simd_matches_scalar_with_random_yuv() {
  // NV16 reuses the NV12 RGBA kernel; this test pins the wiring
  // regardless of which tier the dispatcher picks. Width 1922 +
  // height 4 to exercise both main loop and tail per backend.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; w * h]; // NV16 UV is full-height
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv16>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv16_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv16>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv16_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "NV16 RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn nv16_rgba_matches_nv12_rgba_with_duplicated_chroma() {
  // Cross-format invariant on the RGBA path. Mirrors the existing
  // `nv16_matches_nv12_mixed_sinker_with_duplicated_chroma` for
  // RGB: duplicating NV16 chroma rows pairwise so the 4:2:0
  // vertical upsample matches NV16's per-row chroma must yield
  // byte-identical RGBA. Catches NV16-vs-NV12 wiring regressions
  // specific to the new RGBA path.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv16: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let mut uv_nv12 = std::vec![0u8; w * h / 2];
  for c_row in 0..h / 2 {
    let src_row = c_row * 2;
    uv_nv12[c_row * w..(c_row + 1) * w].copy_from_slice(&uv_nv16[src_row * w..(src_row + 1) * w]);
  }
  let mut uv_nv16_aligned = uv_nv16.clone();
  for c_row in 0..h / 2 {
    let even_row = c_row * 2;
    let odd_row = even_row + 1;
    let (even, odd) = uv_nv16_aligned.split_at_mut(odd_row * w);
    odd[..w].copy_from_slice(&even[even_row * w..even_row * w + w]);
  }
  let nv16_src = Nv16Frame::new(
    &yp,
    &uv_nv16_aligned,
    w as u32,
    h as u32,
    w as u32,
    w as u32,
  );
  let nv12_src = Nv12Frame::new(&yp, &uv_nv12, w as u32, h as u32, w as u32, w as u32);

  let mut rgba_nv16 = std::vec![0u8; w * h * 4];
  let mut rgba_nv12 = std::vec![0u8; w * h * 4];
  let mut s_nv16 = MixedSinker::<Nv16>::new(w, h)
    .with_rgba(&mut rgba_nv16)
    .unwrap();
  let mut s_nv12 = MixedSinker::<Nv12>::new(w, h)
    .with_rgba(&mut rgba_nv12)
    .unwrap();
  nv16_to(&nv16_src, false, ColorMatrix::Bt709, &mut s_nv16).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();

  assert_eq!(rgba_nv16, rgba_nv12);
}

#[test]
fn nv16_odd_width_sink_returns_err_at_begin_frame() {
  let mut rgb = std::vec![0u8; 15 * 8 * 3];
  let mut sink = MixedSinker::<Nv16>::new(15, 8).with_rgb(&mut rgb).unwrap();
  let (yp, uvp) = solid_nv16_frame(16, 8, 0, 0, 0); // dummy 16-wide frame
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let err = nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 15 }));
}

// ---- NV21 MixedSinker ---------------------------------------------------

fn solid_nv21_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
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

// ---- NV24 MixedSinker ---------------------------------------------------
//
// 4:4:4 semi-planar: UV row is `2 * width` bytes (one UV pair per
// Y pixel). Tests mirror the NV12 set plus one cross-format parity
// check against a synthetic NV42 frame (byte-swap the interleaved
// chroma → identical RGB output).

fn solid_nv24_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // UV row payload = `2 * width` bytes = `width` interleaved U/V pairs.
  let mut uv = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for i in 0..w {
      uv[row * 2 * w + i * 2] = u;
      uv[row * 2 * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
fn nv24_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 42, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 128, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv24_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 200, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv24_accepts_odd_width() {
  // 4:4:4 removes the width parity constraint. A 17-wide frame
  // should round-trip cleanly.
  let (yp, uvp) = solid_nv24_frame(17, 8, 200, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);

  let mut rgb = std::vec![0u8; 17 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(17, 8).with_rgb(&mut rgb).unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
}

// ---- NV42 MixedSinker ---------------------------------------------------

fn solid_nv42_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // VU row payload = `2 * width` bytes = `width` interleaved V/U pairs
  // (byte-swapped relative to NV24).
  let mut vu = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for i in 0..w {
      vu[row * 2 * w + i * 2] = v;
      vu[row * 2 * w + i * 2 + 1] = u;
    }
  }
  (std::vec![y; w * h], vu)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgb_only_converts_gray_to_gray() {
  let (yp, vup) = solid_nv42_frame(16, 8, 128, 128, 128);
  let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv42>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv42_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv42_matches_nv24_mixed_sinker_with_swapped_chroma() {
  // Cross-format parity: for the same Y plane and byte-swapped
  // interleaved chroma, NV24 and NV42 must produce identical RGB
  // output. Mirrors the NV21↔NV12 test.
  let w = 33usize; // deliberately odd to exercise the no-parity-constraint path
  let h = 8usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv24: Vec<u8> = (0..2 * w * h)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  // Build NV42 chroma by swapping each (U, V) pair.
  let mut vu_nv42 = std::vec![0u8; 2 * w * h];
  for i in 0..w * h {
    vu_nv42[i * 2] = uv_nv24[i * 2 + 1];
    vu_nv42[i * 2 + 1] = uv_nv24[i * 2];
  }
  let nv24_src = Nv24Frame::new(&yp, &uv_nv24, w as u32, h as u32, w as u32, (2 * w) as u32);
  let nv42_src = Nv42Frame::new(&yp, &vu_nv42, w as u32, h as u32, w as u32, (2 * w) as u32);

  let mut rgb_nv24 = std::vec![0u8; w * h * 3];
  let mut rgb_nv42 = std::vec![0u8; w * h * 3];
  let mut s_nv24 = MixedSinker::<Nv24>::new(w, h)
    .with_rgb(&mut rgb_nv24)
    .unwrap();
  let mut s_nv42 = MixedSinker::<Nv42>::new(w, h)
    .with_rgb(&mut rgb_nv42)
    .unwrap();
  nv24_to(&nv24_src, false, ColorMatrix::Bt709, &mut s_nv24).unwrap();
  nv42_to(&nv42_src, false, ColorMatrix::Bt709, &mut s_nv42).unwrap();

  assert_eq!(rgb_nv24, rgb_nv42);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_with_simd_false_matches_with_simd_true() {
  // Widths chosen to force each backend's main loop AND its
  // scalar-tail path:
  // - 16, 17 → NEON/SSE4.1/wasm main (16-Y block), AVX2 + AVX-512 no main.
  // - 32, 33 → AVX2 main (32-Y block), AVX-512 no main.
  // - 64, 65 → AVX-512 main (64-Y block) once + optional 1-px tail.
  // - 127, 128 → AVX-512 main twice, 127 also forces a 63-px tail.
  // - 1920 → wide real-world baseline.
  for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
    let h = 4usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uvp: Vec<u8> = (0..2 * w * h)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let src = Nv24Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv24>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv24>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv24_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    nv24_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "NV24 SIMD≠scalar at width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_with_simd_false_matches_with_simd_true() {
  // Same width coverage as the NV24 variant — exercises every
  // backend's main loop + scalar tail for the `SWAP_UV = true`
  // monomorphization.
  for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
    let h = 4usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vup: Vec<u8> = (0..2 * w * h)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let src = Nv42Frame::new(&yp, &vup, w as u32, h as u32, w as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv42>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv42>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv42_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    nv42_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "NV42 SIMD≠scalar at width {w}");
  }
}

#[test]
fn nv24_width_mismatch_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  // 8-tall src matches the sink; width 17 vs sink's 16 triggers the
  // mismatch in `begin_frame`.
  let (yp, uvp) = solid_nv24_frame(17, 8, 0, 0, 0);
  let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);
  let err = nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
}

#[test]
fn nv24_process_rejects_short_uv_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 31]; // expected 2 * 16 = 32
  let row = Nv24Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::UvFull,
      row: 0,
      expected: 32,
      actual: 31,
    }
  );
}

#[test]
fn nv24_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 32];
  let row = Nv24Row::new(&y, &uv, 8, ColorMatrix::Bt601, true); // row 8 == height
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
fn nv42_process_rejects_short_vu_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv42>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let vu = [128u8; 31]; // expected 32
  let row = Nv42Row::new(&y, &vu, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::VuFull,
      row: 0,
      expected: 32,
      actual: 31,
    }
  );
}

// ---- Nv24/Nv42 RGBA (Ship 8 PR 4b) tests --------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 128, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv24_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A invariant: when both RGB and RGBA are attached, the
  // RGBA bytes must be byte-for-byte identical to the RGB row +
  // 0xFF alpha. This is the cross-format guarantee that holds even
  // after we replaced the dual-kernel path with the
  // expand_rgb_to_rgba_row fan-out.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, uvp) = solid_nv24_frame(w, h, 180, 60, 200);
  let src = Nv24Frame::new(&yp, &uvp, w, h, w, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Nv24>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv24_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv24>::new(16, 8).with_rgba(&mut rgba_short);
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
fn nv24_rgba_simd_matches_scalar_with_random_yuv() {
  // Width 1922 forces both the SIMD main loop AND scalar tail across
  // every backend block size (16/32/64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv24Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, (2 * w) as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv24>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv24_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv24>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv24_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Nv24 RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, vup) = solid_nv42_frame(16, 8, 128, 128, 128);
  let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv42>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv42_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, vup) = solid_nv42_frame(w, h, 180, 60, 200);
  let src = Nv42Frame::new(&yp, &vup, w, h, w, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Nv42>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv42_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFF);
  }
}

#[test]
fn nv42_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv42>::new(16, 8).with_rgba(&mut rgba_short);
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
fn nv42_rgba_simd_matches_scalar_with_random_yuv() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut vup = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut vup, 0xCAFE_F00D);
  let src = Nv42Frame::new(&yp, &vup, w as u32, h as u32, w as u32, (2 * w) as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv42>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv42_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv42>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv42_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Nv42 RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

// Cross-format Strategy A invariant: when both RGB+RGBA are
// attached, all 9 wired families derive RGBA from the RGB row via
// expand_rgb_to_rgba_row. This test runs all 9 process methods with
// the same gray input and asserts every RGBA sample equals the RGB
// sample with alpha = 0xFF — proving the fan-out shape never
// diverges from the kernel output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn strategy_a_rgb_and_rgba_byte_identical_for_all_wired_families() {
  let w: u32 = 32;
  let h: u32 = 8;
  let ws = w as usize;
  let hs = h as usize;

  let assert_match = |rgb: &[u8], rgba: &[u8], who: &str| {
    for i in 0..(ws * hs) {
      assert_eq!(rgba[i * 4], rgb[i * 3], "{who}: R differs at px {i}");
      assert_eq!(
        rgba[i * 4 + 1],
        rgb[i * 3 + 1],
        "{who}: G differs at px {i}"
      );
      assert_eq!(
        rgba[i * 4 + 2],
        rgb[i * 3 + 2],
        "{who}: B differs at px {i}"
      );
      assert_eq!(rgba[i * 4 + 3], 0xFF, "{who}: alpha not opaque at px {i}");
    }
  };

  {
    let (yp, up, vp) = solid_yuv420p_frame(w, h, 200, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv420p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Yuv420p");
  }

  {
    let (yp, up, vp) = solid_yuv422p_frame(w, h, 200, 128, 128);
    let src = Yuv422pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv422p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Yuv422p");
  }

  {
    let (yp, up, vp) = solid_yuv444p_frame(w, h, 200, 128, 128);
    let src = Yuv444pFrame::new(&yp, &up, &vp, w, h, w, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv444p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Yuv444p");
  }

  {
    let (yp, uvp) = solid_nv12_frame(w, h, 200, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv12>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Nv12");
  }

  {
    let (yp, vup) = solid_nv21_frame(w, h, 200, 128, 128);
    let src = Nv21Frame::new(&yp, &vup, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv21>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv21_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Nv21");
  }

  {
    let (yp, uvp) = solid_nv16_frame(w, h, 200, 128, 128);
    let src = Nv16Frame::new(&yp, &uvp, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv16>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Nv16");
  }

  {
    let (yp, uvp) = solid_nv24_frame(w, h, 200, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, w, h, w, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv24>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Nv24");
  }

  {
    let (yp, vup) = solid_nv42_frame(w, h, 200, 128, 128);
    let src = Nv42Frame::new(&yp, &vup, w, h, w, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv42>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv42_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Nv42");
  }

  {
    let (yp, up, vp) = solid_yuv440p_frame(w, h, 200, 128, 128);
    let src = Yuv440pFrame::new(&yp, &up, &vp, w, h, w, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv440p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
    assert_match(&rgb, &rgba, "Yuv440p");
  }
}

// ---- Yuv420p10 --------------------------------------------------------

fn solid_yuv420p10_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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
fn yuv420p10_rgb_u8_only_gray_is_gray() {
  // 10-bit mid-gray: Y=512, UV=512 → 8-bit RGB ≈ 128 on every channel.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p10_rgb_u16_only_native_depth_gray() {
  // Same mid-gray frame → u16 RGB output in native 10-bit depth, so
  // each channel should be ≈ 512 (the 10-bit mid).
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    // Upper 6 bits of each u16 must be zero — 10-bit convention.
    assert!(px[0] <= 1023);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgb_u8_and_u16_both_populated() {
  // 10-bit full-range white: Y=1023, UV=512. Both buffers should
  // fill with their respective "white" values (255 for u8, 1023 for
  // u16) in the same call.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 1023, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 1023));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_luma_downshifts_to_8bit() {
  // Y=512 at 10 bits → 512 >> 2 = 128 at 8 bits.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_hsv_from_gray_is_zero_hue_zero_sat() {
  // HSV derived from the internal u8 RGB scratch: neutral gray →
  // H=0, S=0, V≈128. Exercises the "HSV without RGB" scratch path
  // on the 10-bit source.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10]; // Way too small.
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_with_simd_false_matches_with_simd_true() {
  // The SIMD toggle exercises scalar-vs-SIMD dispatch. Both paths
  // must produce byte-identical results on both outputs.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- Yuv420p10 RGBA (Ship 8 Tranche 5b) -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255) per pixel.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A: when both rgb and rgba are attached, the rgb buffer is
  // populated by the RGB kernel and the rgba buffer is populated via a
  // cheap expand pass. RGB triples must be byte-identical to the
  // standalone RGB-only run.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(
    rgb_solo, rgb_combined,
    "RGB u16 elements must match across runs"
  );
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 1023, "alpha = (1 << 10) - 1");
  }
}

#[test]
fn yuv420p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuv420p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

// ---- P010 --------------------------------------------------------------
//
// Semi-planar 10-bit, high-bit-packed (samples in high 10 of each
// u16). Mirrors the Yuv420p10 test shape but with UV interleaved.

fn solid_p010_frame(
  width: u32,
  height: u32,
  y_10bit: u16,
  u_10bit: u16,
  v_10bit: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // Shift into the high 10 bits (P010 packing).
  let y = std::vec![y_10bit << 6; w * h];
  let uv: Vec<u16> = (0..cw * ch)
    .flat_map(|_| [u_10bit << 6, v_10bit << 6])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u8_only_gray_is_gray() {
  // 10-bit mid-gray Y=512, UV=512 → ~128 u8 RGB across the frame.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn p010_rgb_u16_only_native_depth_gray() {
  // Output u16 is yuv420p10le-packed (10-bit in low 10) even though
  // the input is P010-packed.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(
      px[0] <= 1023,
      "output must stay within 10-bit low-packed range"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u8_and_u16_both_populated() {
  // 10-bit full-range white: Y=1023, UV=512. Both buffers fill in
  // one call.
  let (yp, uvp) = solid_p010_frame(16, 8, 1023, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 1023));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_luma_downshifts_to_8bit() {
  // Y=512 at 10 bits, P010-packed (0x8000). After >> 8, the 8-bit
  // luma is 0x80 = 128.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_matches_yuv420p10_mixed_sinker_with_shifted_samples() {
  // Logical equivalence: same samples fed through the two formats
  // (low-packed as yuv420p10, high-packed as P010) must produce
  // byte-identical u8 RGB.
  let w = 16u32;
  let h = 8u32;
  let y = 600u16;
  let u = 400u16;
  let v = 700u16;

  let (yp_p10, up_p10, vp_p10) = solid_yuv420p10_frame(w, h, y, u, v);
  let src_p10 = Yuv420p10Frame::new(&yp_p10, &up_p10, &vp_p10, w, h, w, w / 2, w / 2);

  let (yp_p010, uvp_p010) = solid_p010_frame(w, h, y, u, v);
  let src_p010 = P010Frame::new(&yp_p010, &uvp_p010, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p010 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p10>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p010 = MixedSinker::<P010>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p010)
    .unwrap();
  yuv420p10_to(&src_p10, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p010_to(&src_p010, true, ColorMatrix::Bt709, &mut s_p010).unwrap();
  assert_eq!(rgb_yuv, rgb_p010);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P010>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_with_simd_false_matches_with_simd_true() {
  // Stubs delegate to scalar so simd=true and simd=false produce
  // byte-identical output for now. Real SIMD backends will replace
  // the stubs — equivalence is preserved by design.
  let (yp, uvp) = solid_p010_frame(64, 16, 600, 400, 700);
  let src = P010Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P010>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p010_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P010>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p010_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- P010 RGBA (Ship 8 Tranche 5b) ------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P010 mid-gray (10-bit values shifted into the high 10): Y/U/V = 512 << 6.
  // Output u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha = (1 << 10) - 1");
  }
}

// ---- Yuv420p12 ---------------------------------------------------------
//
// Planar 12-bit, low-bit-packed. Mirrors the Yuv420p10 shape — same
// planar layout, wider sample range. `mid-gray` for 12-bit is
// Y=UV=2048; native-depth white (full-range) is 4095.

fn solid_yuv420p12_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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
fn yuv420p12_rgb_u8_only_gray_is_gray() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p12_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    // Upper 4 bits must be zero — 12-bit low-packed convention.
    assert!(px[0] <= 4095);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u8_and_u16_both_populated() {
  // Full-range white: Y=4095, UV=2048.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 4095, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 4095));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_luma_downshifts_to_8bit() {
  // Y=2048 at 12 bits → 2048 >> (12 - 8) = 128 at 8 bits.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_hsv_from_gray_is_zero_hue_zero_sat() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p12_frame(64, 16, 2400, 1600, 2800);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p12>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p12>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray (Y=U=V=2048) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

// ---- Yuv420p14 ---------------------------------------------------------

fn solid_yuv420p14_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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
fn yuv420p14_rgb_u8_only_gray_is_gray() {
  // 14-bit mid-gray: Y=UV=8192.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p14_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(px[0] <= 16383);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_luma_downshifts_to_8bit() {
  // Y=8192 at 14 bits → 8192 >> (14 - 8) = 128.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgb_u8_and_u16_both_populated() {
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 16383, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 16383));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p14_frame(64, 16, 9600, 6400, 11200);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p14>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p14>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray (Y=U=V=8192) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

// ---- P012 --------------------------------------------------------------
//
// Semi-planar 12-bit, high-bit-packed (samples in high 12 of each
// u16). Mirrors the P010 test shape — UV interleaved, `value << 4`.

fn solid_p012_frame(
  width: u32,
  height: u32,
  y_12bit: u16,
  u_12bit: u16,
  v_12bit: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // Shift into the high 12 bits (P012 packing).
  let y = std::vec![y_12bit << 4; w * h];
  let uv: Vec<u16> = (0..cw * ch)
    .flat_map(|_| [u_12bit << 4, v_12bit << 4])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u8_only_gray_is_gray() {
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn p012_rgb_u16_only_native_depth_gray() {
  // Output is low-bit-packed 12-bit (yuv420p12le convention).
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(
      px[0] <= 4095,
      "output must stay within 12-bit low-packed range"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u8_and_u16_both_populated() {
  let (yp, uvp) = solid_p012_frame(16, 8, 4095, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 4095));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_luma_downshifts_to_8bit() {
  // Y=2048 at 12 bits, P012-packed (2048 << 4 = 0x8000). After >> 8,
  // the 8-bit luma is 0x80 = 128 — same accessor as P010 since both
  // store active bits in the high positions.
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_matches_yuv420p12_mixed_sinker_with_shifted_samples() {
  // Logical equivalence — same 12-bit samples fed through both
  // layouts must produce byte-identical u8 RGB.
  let w = 16u32;
  let h = 8u32;
  let y = 2400u16;
  let u = 1600u16;
  let v = 2800u16;

  let (yp_p12, up_p12, vp_p12) = solid_yuv420p12_frame(w, h, y, u, v);
  let src_p12 = Yuv420p12Frame::new(&yp_p12, &up_p12, &vp_p12, w, h, w, w / 2, w / 2);

  let (yp_p012, uvp_p012) = solid_p012_frame(w, h, y, u, v);
  let src_p012 = P012Frame::new(&yp_p012, &uvp_p012, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p012 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p12>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p012 = MixedSinker::<P012>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p012)
    .unwrap();
  yuv420p12_to(&src_p12, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p012_to(&src_p012, true, ColorMatrix::Bt709, &mut s_p012).unwrap();
  assert_eq!(rgb_yuv, rgb_p012);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P012>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_with_simd_false_matches_with_simd_true() {
  let (yp, uvp) = solid_p012_frame(64, 16, 2400, 1600, 2800);
  let src = P012Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P012>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p012_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P012>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p012_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgba_u8_only_gray_with_opaque_alpha() {
  // P012 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P012 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

// ---- Yuv420p16 ---------------------------------------------------------
//
// Planar 16-bit, full u16 range. Mid-gray is Y=UV=32768; full-range
// white luma is 65535.

fn solid_yuv420p16_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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
fn yuv420p16_rgb_u8_only_gray_is_gray() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p16_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgb_u8_and_u16_both_populated() {
  // Full-range white: Y=65535, UV=32768.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 65535, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 65535));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_luma_downshifts_to_8bit() {
  // Y=32768 at 16 bits → 32768 >> (16 - 8) = 128.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_hsv_from_gray_is_zero_hue_zero_sat() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p16_frame(64, 16, 40000, 20000, 45000);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p16>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p16>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- Yuv420p16 RGBA (Ship 8 Tranche 5b) -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray: Y=UV=32768. Output u16 RGBA: each color element ≈
  // 32768, alpha = 0xFFFF.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 8, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// ---- P016 --------------------------------------------------------------

fn solid_p016_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // At 16 bits there's no shift — samples go in raw.
  let y_plane = std::vec![y; w * h];
  let uv: Vec<u16> = (0..cw * ch).flat_map(|_| [u, v]).collect();
  (y_plane, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u8_only_gray_is_gray() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn p016_rgb_u16_only_native_depth_gray() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u8_and_u16_both_populated() {
  let (yp, uvp) = solid_p016_frame(16, 8, 65535, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 65535));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_luma_downshifts_to_8bit() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_matches_yuv420p16_mixed_sinker() {
  // At 16 bits P016 and yuv420p16 are numerically identical —
  // the packing distinction degenerates when every bit is active.
  // Only the plane count / interleave layout differs.
  let w = 16u32;
  let h = 8u32;
  let y = 40000u16;
  let u = 20000u16;
  let v = 45000u16;

  let (yp_p16, up_p16, vp_p16) = solid_yuv420p16_frame(w, h, y, u, v);
  let src_p16 = Yuv420p16Frame::new(&yp_p16, &up_p16, &vp_p16, w, h, w, w / 2, w / 2);

  let (yp_p016, uvp_p016) = solid_p016_frame(w, h, y, u, v);
  let src_p016 = P016Frame::new(&yp_p016, &uvp_p016, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p016 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p16>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p016 = MixedSinker::<P016>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p016)
    .unwrap();
  yuv420p16_to(&src_p16, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p016_to(&src_p016, true, ColorMatrix::Bt709, &mut s_p016).unwrap();
  assert_eq!(rgb_yuv, rgb_p016);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P016>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_with_simd_false_matches_with_simd_true() {
  let (yp, uvp) = solid_p016_frame(64, 16, 40000, 20000, 45000);
  let src = P016Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P016>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p016_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P016>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p016_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- Ship 6: sanity tests for new 4:2:2 / 4:4:4 formats ---------------

fn solid_yuv422p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  // 4:2:2: chroma is half-width, FULL-height.
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

fn solid_yuv444p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
  )
}

fn solid_yuv422p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

fn solid_yuv444p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_frame(16, 8, 128, 128, 128);
  let src = Yuv422pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_frame(16, 8, 128, 128, 128);
  let src = Yuv444pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

// ---- Yuv444p RGBA (Ship 8 PR 4a) tests ----------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv444p_frame(16, 8, 128, 128, 128);
  let src = Yuv444pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv444p_frame(w, h, 180, 60, 200);
  let src = Yuv444pFrame::new(&yp, &up, &vp, w, h, w, w, w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv444p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv444p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv444p>::new(16, 8).with_rgba(&mut rgba_short);
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
fn yuv444p_rgba_simd_matches_scalar_with_random_yuv() {
  // 4:4:4 has full-width chroma — U / V are width-sized per row.
  // Width 1922 forces both the SIMD main loop AND scalar tail
  // across every backend block size (16/32/64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; w * h];
  let mut vp = std::vec![0u8; w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv444pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
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

      let mut s_simd = MixedSinker::<Yuv444p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv444p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv444p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv444p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuv444p RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn yuv422p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p14_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p16_gray_to_gray_u16() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv422p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv422p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p14_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p16_gray_to_gray_u16() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv444p16Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv444p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_matches_yuv420p_luma_when_chroma_matches() {
  // 4:2:2 and 4:2:0 differ only in vertical chroma walk. With solid
  // chroma planes they must produce identical RGB output — this is
  // the whole reason Yuv422p reuses the yuv_420 row kernel.
  let w = 32u32;
  let h = 8u32;
  let (yp, up422, vp422) = solid_yuv422p_frame(w, h, 140, 100, 160);
  let src422 = Yuv422pFrame::new(&yp, &up422, &vp422, w, h, w, w / 2, w / 2);

  let (yp420, up420, vp420) = solid_yuv420p_frame(w, h, 140, 100, 160);
  let src420 = Yuv420pFrame::new(&yp420, &up420, &vp420, w, h, w, w / 2, w / 2);

  let mut rgb422 = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb420 = std::vec![0u8; (w * h * 3) as usize];
  let mut s422 = MixedSinker::<Yuv422p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb422)
    .unwrap();
  let mut s420 = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb420)
    .unwrap();
  yuv422p_to(&src422, true, ColorMatrix::Bt709, &mut s422).unwrap();
  yuv420p_to(&src420, true, ColorMatrix::Bt709, &mut s420).unwrap();
  assert_eq!(rgb422, rgb420);
}

// ---- Yuv422p RGBA (Ship 8 PR 3) tests -----------------------------------
//
// Yuv422p reuses the Yuv420p `_to_rgba_row` dispatcher (same row
// contract). Tests mirror the Yuv420p RGBA set; the cross-format
// invariant against Yuv420p (with solid chroma so 4:2:0 vertical
// upsample matches Yuv422p's per-row chroma) catches walker
// regressions specific to the Yuv422p RGBA wiring.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv422p_frame(16, 8, 128, 128, 128);
  let src = Yuv422pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv422p_frame(w, h, 180, 60, 200);
  let src = Yuv422pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv422p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv422p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv422p>::new(16, 8).with_rgba(&mut rgba_short);
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
fn yuv422p_rgba_simd_matches_scalar_with_random_yuv() {
  // Random per-pixel YUV across all matrices × both ranges. Width
  // 1922 forces both the SIMD main loop AND a scalar tail across
  // every backend block size (16/32/64). 4:2:2 chroma is full-
  // height, so up/vp use `w/2 × h` instead of `w/2 × h/2`.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; (w / 2) * h];
  let mut vp = std::vec![0u8; (w / 2) * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv422pFrame::new(
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

      let mut s_simd = MixedSinker::<Yuv422p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv422p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv422p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv422p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuv422p RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn yuv422p_rgba_matches_yuv420p_rgba_when_chroma_matches() {
  // 4:2:2 and 4:2:0 differ only in vertical chroma walk. With
  // solid chroma planes they must produce identical RGBA — same
  // shape as the existing `yuv422p_matches_yuv420p_luma_when_chroma_matches`
  // RGB-path test for the new RGBA path.
  let w = 32u32;
  let h = 8u32;
  let (yp, up422, vp422) = solid_yuv422p_frame(w, h, 140, 100, 160);
  let src422 = Yuv422pFrame::new(&yp, &up422, &vp422, w, h, w, w / 2, w / 2);

  let (yp420, up420, vp420) = solid_yuv420p_frame(w, h, 140, 100, 160);
  let src420 = Yuv420pFrame::new(&yp420, &up420, &vp420, w, h, w, w / 2, w / 2);

  let mut rgba422 = std::vec![0u8; (w * h * 4) as usize];
  let mut rgba420 = std::vec![0u8; (w * h * 4) as usize];
  let mut s422 = MixedSinker::<Yuv422p>::new(w as usize, h as usize)
    .with_rgba(&mut rgba422)
    .unwrap();
  let mut s420 = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
    .with_rgba(&mut rgba420)
    .unwrap();
  yuv422p_to(&src422, true, ColorMatrix::Bt709, &mut s422).unwrap();
  yuv420p_to(&src420, true, ColorMatrix::Bt709, &mut s420).unwrap();
  assert_eq!(rgba422, rgba420);
}

// ---- 9-bit family + 4:4:0 family sanity tests ------------------------

fn solid_yuv440p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let ch = (height as usize).div_ceil(2);
  (
    std::vec![y; w * h],
    std::vec![u; w * ch],
    std::vec![v; w * ch],
  )
}

fn solid_yuv440p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let ch = (height as usize).div_ceil(2);
  (
    std::vec![y; w * h],
    std::vec![u; w * ch],
    std::vec![v; w * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  // 4:2:0 chroma is w/2 × h/2; reuse the 4:2:2 helper's `cw * h` and
  // truncate to the 4:2:0 layout (cw = 8, ch = 4).
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray (Y=U=V=256) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_frame(16, 8, 128, 128, 128);
  let src = Yuv440pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

// ---- Yuv440p RGBA (Ship 8 PR 4c) tests --------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv440p_frame(16, 8, 128, 128, 128);
  let src = Yuv440pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv440p_frame(w, h, 180, 60, 200);
  let src = Yuv440pFrame::new(&yp, &up, &vp, w, h, w, w, w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv440p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv440p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv440p>::new(16, 8).with_rgba(&mut rgba_short);
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
fn yuv440p_rgba_simd_matches_scalar_with_random_yuv() {
  // Width 1922 forces both the SIMD main loop AND scalar tail across
  // every backend block size (16/32/64). 4:4:0 chroma is full-width
  // but half-height, so chroma plane is `w * h/2`.
  let w = 1922usize;
  let h = 4usize;
  let ch = h / 2;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; w * ch];
  let mut vp = std::vec![0u8; w * ch];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv440pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
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

      let mut s_simd = MixedSinker::<Yuv440p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv440p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv440p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv440p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Yuv440p RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv440p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 4:4:0 reuses the 4:4:4 dispatcher. 12-bit mid-gray → 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_matches_yuv444p_when_chroma_constant_per_pair() {
  // 4:4:0 reuses the 4:4:4 row math; the only difference is the
  // walker reads chroma row r/2. With the same chroma value at every
  // (r, c), Yuv440p must produce identical RGB to Yuv444p with
  // duplicated chroma rows.
  let w = 32u32;
  let h = 8u32;
  let (yp, up440, vp440) = solid_yuv440p_frame(w, h, 140, 100, 160);
  let src440 = Yuv440pFrame::new(&yp, &up440, &vp440, w, h, w, w, w);

  // Yuv444p needs full-height chroma, so duplicate each of the 4 4:4:0
  // chroma rows into 2 rows.
  let mut up444 = std::vec::Vec::with_capacity((w * h) as usize);
  let mut vp444 = std::vec::Vec::with_capacity((w * h) as usize);
  for r in 0..h {
    let cr = (r / 2) as usize;
    let row_start = cr * w as usize;
    let row_end = row_start + w as usize;
    up444.extend_from_slice(&up440[row_start..row_end]);
    vp444.extend_from_slice(&vp440[row_start..row_end]);
  }
  let src444 = Yuv444pFrame::new(&yp, &up444, &vp444, w, h, w, w, w);

  let mut rgb440 = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb444 = std::vec![0u8; (w * h * 3) as usize];
  let mut s440 = MixedSinker::<Yuv440p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb440)
    .unwrap();
  let mut s444 = MixedSinker::<Yuv444p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb444)
    .unwrap();
  yuv440p_to(&src440, true, ColorMatrix::Bt709, &mut s440).unwrap();
  yuv444p_to(&src444, true, ColorMatrix::Bt709, &mut s444).unwrap();
  assert_eq!(rgb440, rgb444);
}

// ---- Walker-level SIMD-vs-scalar equivalence for 9-bit 4:2:x --------
//
// Per-arch row-kernel tests cover the BITS=9 path with non-neutral
// chroma directly. These walker-level tests additionally pin the
// public dispatcher behavior — Yuv420p9 / Yuv422p9 read through the
// same `yuv_420p_n_to_rgb_*<9>` half-width kernels, so a backend
// bug here would silently corrupt user output. Width 1922 forces
// both the SIMD main loop and a scalar tail; chroma is non-neutral
// and limited-range parameters are exercised below.

pub(super) fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
  let mask = ((1u32 << bits) - 1) as u16;
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) as u16) & mask;
  }
}

pub(super) fn pseudo_random_u8(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p9_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32; // forces tail handling on every backend
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut up = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
  let mut vp = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
  pseudo_random_u16_low_n_bits(&mut yp, 0x1111, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0x2222, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0x3333, 9);
  let src = Yuv420p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yuv420p9 SIMD u8 ≠ scalar u8");
    assert_eq!(
      rgb_u16_simd, rgb_u16_scalar,
      "Yuv420p9 SIMD u16 ≠ scalar u16"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p9_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut up = std::vec![0u16; ((w / 2) * h) as usize];
  let mut vp = std::vec![0u16; ((w / 2) * h) as usize];
  pseudo_random_u16_low_n_bits(&mut yp, 0x4444, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0x5555, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0x6666, 9);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yuv422p9 SIMD u8 ≠ scalar u8");
    assert_eq!(
      rgb_u16_simd, rgb_u16_scalar,
      "Yuv422p9 SIMD u16 ≠ scalar u16"
    );
  }
}

// ---- P210 / P212 / P216 / P410 / P412 / P416 sanity tests --------------

/// 4:2:2 P-family solid frame helper. UV is `width` u16 elements per
/// row, **full-height** chroma. All samples are high-bit-packed
/// (shifted left by `16 - bits`).
fn solid_p2x0_frame(
  width: u32,
  height: u32,
  bits: u32,
  y_value: u16,
  u_value: u16,
  v_value: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let shift = 16 - bits;
  let y = std::vec![y_value << shift; w * h];
  // 4:2:2: full-height chroma, half-width × 2 elements per pair.
  let uv: Vec<u16> = (0..cw * h)
    .flat_map(|_| [u_value << shift, v_value << shift])
    .collect();
  (y, uv)
}

/// 4:4:4 P-family solid frame helper. UV is `2 * width` u16 elements
/// per row, **full-height** chroma (one pair per pixel).
fn solid_p4x0_frame(
  width: u32,
  height: u32,
  bits: u32,
  y_value: u16,
  u_value: u16,
  v_value: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let shift = 16 - bits;
  let y = std::vec![y_value << shift; w * h];
  // 4:4:4: full-height × full-width × 2 elements per pair.
  let uv: Vec<u16> = (0..w * h)
    .flat_map(|_| [u_value << shift, v_value << shift])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p210_gray_to_gray() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
  let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P210>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p210_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn p212_gray_to_gray() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P212>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p212_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn p212_rgba_u8_only_gray_with_opaque_alpha() {
  // P212 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P212>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p212_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p212_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P212 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P212>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p212_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p216_gray_to_gray_u16() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p216_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p216_rgba_u8_only_gray_with_opaque_alpha() {
  // P216 mid-gray (16-bit, no shift): Y/U/V = 32768. Output 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p216_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p216_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p216_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_gray_to_gray() {
  // 4:4:4: uv_stride = 2 * width = 32 (16 pairs × 2 elements).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
  let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P410>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p410_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn p412_gray_to_gray() {
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P412>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p412_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn p412_rgba_u8_only_gray_with_opaque_alpha() {
  // P412 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P412>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p412_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p412_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P412 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P412>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p412_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_gray_to_gray_u16() {
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p416_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_rgba_u8_only_gray_with_opaque_alpha() {
  // P416 mid-gray (16-bit, no shift): Y/U/V = 32768. Output 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p416_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p416_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// ---- Walker-level SIMD-vs-scalar equivalence for P410 (4:4:4 Pn) ------
//
// P410 is the only new format in Ship 7 that ships a genuinely new
// SIMD kernel family (`p_n_444_to_rgb_*<BITS>`). Validate the
// walker against scalar with non-neutral chroma and tail widths.
// P210/P212/P216 reuse 4:2:0 P-family kernels (already covered by
// earlier ships' tests).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32; // forces tail handling on every backend
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  // Seed pseudo-random samples in the high 10 bits.
  let mut state: u32 = 0x1111_2222;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = ((state & 0x3FF) as u16) << 6;
  }

  let src = P410Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P410>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p410_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<P410>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p410_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P410 SIMD u8 ≠ scalar u8");
    assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P410 SIMD u16 ≠ scalar u16");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p412_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  let mut state: u32 = 0x3333_4444;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = ((state & 0xFFF) as u16) << 4;
  }

  let src = P412Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P412>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    p412_to(&src, full_range, ColorMatrix::Bt709, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<P412>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p412_to(&src, full_range, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P412 SIMD u8 ≠ scalar u8");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  let mut state: u32 = 0x5555_6666;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = state as u16;
  }

  let src = P416Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P416>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p416_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<P416>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p416_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P416 SIMD u8 ≠ scalar u8");
    assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P416 SIMD u16 ≠ scalar u16");
  }
}


// ---- Ship 8 PR 5d: high-bit 4:2:2 RGBA wiring -------------------------
//
// Strategy A combine for the eight 4:2:2 high-bit sinker formats wired
// in the 4:2:2 high-bit file. Mirrors the 4:2:0 PR #26 test suite;
// covers Yuv422p10 (planar BITS-generic), Yuv422p16 (planar 16-bit
// dedicated kernel), and P210 (semi-planar BITS-generic) — the row
// layer is exhaustively tested elsewhere.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255) per pixel.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A: when both rgb and rgba are attached, the rgb buffer is
  // populated by the RGB kernel and the rgba buffer is populated via a
  // cheap expand pass. RGB triples must be byte-identical to the
  // standalone RGB-only run.
  let (yp, up, vp) = solid_yuv422p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv422p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(
    rgb_solo, rgb_combined,
    "RGB u16 elements must match across runs"
  );
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 1023, "alpha = (1 << 10) - 1");
  }
}

#[test]
fn yuv422p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuv422p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p210_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P210 stores 10-bit samples high-bit-packed (`<< 6`). Mid-gray u16
  // RGBA elements ≈ 512 (low-bit-packed, yuv420p10le convention) and
  // alpha = (1 << 10) - 1 = 1023.
  let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
  let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P210>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p210_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn yuv422p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv422p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// ===== Ship 8 Tranche 7c — high-bit 4:4:4 RGBA sinker tests ==========
//
// Mirrors PR #26's 4:2:0 coverage scope: representative formats only,
// not exhaustive per-format. Yuv444p10 covers the BITS-generic planar
// path; P410 covers the Pn semi-planar path; Yuv444p16 covers the
// 16-bit dedicated kernel; Yuv440p10 covers the 4:4:0 kernel-reuse
// path. The remaining 4:4:4 high-bit formats (9/12/14, P412/P416,
// Yuv440p12) are exercised by row-layer tests + the compile-time
// guarantee that the new sinker builders typecheck.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray (Y=512, U=512, V=512) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A on the u8 path: rgb buffer populated by the RGB kernel,
  // rgba buffer populated via the cheap expand_rgb_to_rgba_row pass.
  // RGB triples must be byte-identical to the standalone RGB-only run.
  let (yp, up, vp) = solid_yuv444p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 64, 16, 64, 64, 64);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv444p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 64, 16, 64, 64, 64);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(
    rgb_solo, rgb_combined,
    "RGB u16 elements must match across runs"
  );
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 1023, "alpha = (1 << 10) - 1");
  }
}

#[test]
fn yuv444p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuv444p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P410 (semi-planar 10-bit): mid-gray (high-bit-packed = 512 << 6).
  // u16 RGBA output ≈ 512, alpha = 1023.
  let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
  let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P410>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p410_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn yuv444p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv444p16Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 4:4:0 reuses the 4:4:4 dispatcher. Confirms the kernel-reuse path
  // wires through correctly at the sinker boundary.
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv440p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv440p10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

