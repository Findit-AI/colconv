//! Fused-downscale + filter coverage for the packed-half-float-RGBA
//! family (`Rgbaf16`): binning runs in f32 (there is no `AreaStream<f16>`),
//! and every output is the direct `Rgbaf16` conversion of the f32
//! block-mean **rounded to f16** — the parity oracle. The `rgba_f16` output
//! is exactly that rounded binned row, so re-feeding it through a direct
//! `Rgbaf16` sink must reproduce every derived output bit-for-bit.

use crate::{
  KernelMatrix,
  frame::Rgbaf16LeFrame,
  resample::{AreaResampler, CatmullRom, FilteredResampler, Triangle},
  sinker::MixedSinker,
  source::{Rgbaf16, rgbaf16_to},
};

fn as_le(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

fn ramp_frame(w: usize, h: usize) -> Vec<half::f16> {
  let mut buf = vec![half::f16::ZERO; w * h * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    let i = i as f32;
    px[0] = half::f16::from_f32(0.1 + i * 0.05);
    px[1] = half::f16::from_f32(1.5 - i * 0.03);
    px[2] = half::f16::from_f32(-0.4 + i * 0.02);
    px[3] = half::f16::from_f32(0.2 + i * 0.015); // alpha — real channel
  }
  buf
}

/// Resamples every output with the given resampler, then re-feeds the
/// produced `rgba_f16` (the rounded binned row) through a direct full-res
/// `Rgbaf16` sink and asserts every derived output matches bit-for-bit.
macro_rules! assert_refeed {
  ($sw:expr, $sh:expr, $ow:expr, $oh:expr, $resampler:expr, $ctx:expr) => {{
    let (sw, sh, ow, oh) = ($sw, $sh, $ow, $oh);
    let host = ramp_frame(sw, sh);
    let wire = as_le(&host);
    let src = Rgbaf16LeFrame::try_new(&wire, sw as u32, sh as u32, (sw * 4) as u32).unwrap();

    let mut rgb = vec![0u8; ow * oh * 3];
    let mut rgba = vec![0u8; ow * oh * 4];
    let mut rgba_u16 = vec![0u16; ow * oh * 4];
    let mut rgba_f16 = vec![half::f16::ZERO; ow * oh * 4];
    let mut rgba_f32 = vec![0.0f32; ow * oh * 4];
    let mut luma = vec![0u8; ow * oh];
    let mut hh = vec![0u8; ow * oh];
    let mut ss = vec![0u8; ow * oh];
    let mut vv = vec![0u8; ow * oh];
    {
      let mut sink = MixedSinker::<Rgbaf16, _>::with_resampler(sw, sh, $resampler)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      rgbaf16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }

    let binned_wire = as_le(&rgba_f16);
    let mut ref_rgb = vec![0u8; ow * oh * 3];
    let mut ref_rgba = vec![0u8; ow * oh * 4];
    let mut ref_rgba_u16 = vec![0u16; ow * oh * 4];
    let mut ref_rgba_f32 = vec![0.0f32; ow * oh * 4];
    let mut ref_luma = vec![0u8; ow * oh];
    let mut ref_h = vec![0u8; ow * oh];
    let mut ref_s = vec![0u8; ow * oh];
    let mut ref_v = vec![0u8; ow * oh];
    {
      let binned =
        Rgbaf16LeFrame::try_new(&binned_wire, ow as u32, oh as u32, (ow * 4) as u32).unwrap();
      let mut sink = MixedSinker::<Rgbaf16>::new(ow, oh)
        .with_rgb(&mut ref_rgb)
        .unwrap()
        .with_rgba(&mut ref_rgba)
        .unwrap()
        .with_rgba_u16(&mut ref_rgba_u16)
        .unwrap()
        .with_rgba_f32(&mut ref_rgba_f32)
        .unwrap()
        .with_luma(&mut ref_luma)
        .unwrap()
        .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
        .unwrap();
      rgbaf16_to(&binned, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }
    assert_eq!(rgb, ref_rgb, "{} rgb", $ctx);
    assert_eq!(rgba, ref_rgba, "{} rgba", $ctx);
    assert_eq!(rgba_u16, ref_rgba_u16, "{} rgba_u16", $ctx);
    assert_eq!(rgba_f32, ref_rgba_f32, "{} rgba_f32", $ctx);
    assert_eq!(luma, ref_luma, "{} luma", $ctx);
    assert_eq!(hh, ref_h, "{} h", $ctx);
    assert_eq!(ss, ref_s, "{} s", $ctx);
    assert_eq!(vv, ref_v, "{} v", $ctx);
  }};
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_area_outputs_match_refeed() {
  assert_refeed!(8, 8, 4, 4, AreaResampler::to(4, 4), "area down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_filter_outputs_match_refeed() {
  assert_refeed!(
    8,
    8,
    4,
    4,
    FilteredResampler::new(4, 4, Triangle),
    "triangle down"
  );
  assert_refeed!(
    8,
    8,
    4,
    4,
    FilteredResampler::new(4, 4, CatmullRom),
    "catmullrom down"
  );
  assert_refeed!(
    4,
    4,
    7,
    7,
    FilteredResampler::new(7, 7, Triangle),
    "triangle up"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_identity_plan_matches_new_sink() {
  let host = ramp_frame(8, 8);
  let wire = as_le(&host);
  let src = Rgbaf16LeFrame::try_new(&wire, 8, 8, 8 * 4).unwrap();

  let mut direct = vec![half::f16::ZERO; 8 * 8 * 4];
  {
    let mut sink = MixedSinker::<Rgbaf16>::new(8, 8)
      .with_rgba_f16(&mut direct)
      .unwrap();
    rgbaf16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let mut via_area = vec![half::f16::ZERO; 8 * 8 * 4];
  {
    let mut sink =
      MixedSinker::<Rgbaf16, AreaResampler>::with_resampler(8, 8, AreaResampler::to(8, 8))
        .unwrap()
        .with_rgba_f16(&mut via_area)
        .unwrap();
    rgbaf16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(direct, via_area);
}
