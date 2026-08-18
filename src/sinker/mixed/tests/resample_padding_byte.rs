//! Fused-downscale coverage for the padding-byte packed family
//! (`Xrgb` / `Rgbx` / `Xbgr` / `Bgrx`). Each drops its padding byte to
//! the same canonical RGB and routes the shared packed-RGB resample
//! tail, so all four must match an `Rgb24` resample of that RGB —
//! including the RGBA output, whose alpha is forced to `0xFF` (the
//! padding byte is never real alpha). The source padding byte is a
//! non-trivial `0x77` so a leak would change the result.

use crate::{
  KernelMatrix,
  resample::{AreaResampler, CatmullRom, FilterKernel, FilteredResampler, Lanczos3, Triangle},
  sinker::MixedSinker,
  source::{Bgrx, Rgb24, Rgbx, Xbgr, Xrgb, bgrx_to, rgb24_to, rgbx_to, xbgr_to, xrgb_to},
};
use mediaframe::frame::{BgrxFrame, Rgb24Frame, RgbxFrame, XbgrFrame, XrgbFrame};

const SRC: usize = 8;
const OUT: usize = 4;
const PAD: u8 = 0x77;

/// `(r, g, b)` ramp for source pixel `i` — interior values so derived
/// kernels see real math.
fn rgb_px(i: usize) -> [u8; 3] {
  [
    40 + (i as u8) * 2,
    200 - (i as u8) * 2,
    60 + ((i % 8) as u8) * 10,
  ]
}

/// Canonical RGB frame the four padding sources all decode to.
fn rgb_ramp() -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px.copy_from_slice(&rgb_px(i));
  }
  buf
}

/// 4-byte source frame; `pack` maps a pixel's RGB to its 4 wire bytes.
fn frame4(pack: impl Fn([u8; 3]) -> [u8; 4]) -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    px.copy_from_slice(&pack(rgb_px(i)));
  }
  buf
}

/// `Rgb24` resample of the canonical RGB with rgb + rgba + luma
/// attached — the reference every padding format must match.
fn rgb24_reference() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ramp = rgb_ramp();
  let src = Rgb24Frame::new(&ramp, SRC as u32, SRC as u32, (SRC * 3) as u32);
  let (mut rgb, mut rgba, mut luma) = (
    vec![0u8; OUT * OUT * 3],
    vec![0u8; OUT * OUT * 4],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    rgb24_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  (rgb, rgba, luma)
}

macro_rules! padding_format_matches_rgb24 {
  ($name:ident, $marker:ty, $frame:ident, $walk:ident, $pack:expr) => {
    #[test]
    fn $name() {
      let pix = frame4($pack);
      let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
      let (mut rgb, mut rgba, mut luma) = (
        vec![0u8; OUT * OUT * 3],
        vec![0u8; OUT * OUT * 4],
        vec![0u8; OUT * OUT],
      );
      {
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
        $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
      }
      let (rgb_ref, rgba_ref, luma_ref) = rgb24_reference();
      assert_eq!(rgb, rgb_ref, "rgb");
      assert_eq!(rgba, rgba_ref, "rgba (alpha forced 0xFF)");
      assert_eq!(luma, luma_ref, "luma");
    }
  };
}

padding_format_matches_rgb24!(xrgb_resample_matches_rgb24, Xrgb, XrgbFrame, xrgb_to, |[
  r,
  g,
  b,
]| [
  PAD, r, g, b
]);
padding_format_matches_rgb24!(rgbx_resample_matches_rgb24, Rgbx, RgbxFrame, rgbx_to, |[
  r,
  g,
  b,
]| [
  r, g, b, PAD
]);
padding_format_matches_rgb24!(xbgr_resample_matches_rgb24, Xbgr, XbgrFrame, xbgr_to, |[
  r,
  g,
  b,
]| [
  PAD, b, g, r
]);
padding_format_matches_rgb24!(bgrx_resample_matches_rgb24, Bgrx, BgrxFrame, bgrx_to, |[
  r,
  g,
  b,
]| [
  b, g, r, PAD
]);

/// `Rgb24` **filter** resample of the canonical RGB (rgb + rgba + luma)
/// for a given kernel/geometry — the reference every padding format's
/// filter route must match. The X byte is padding, so the padding
/// format's filtered RGB must equal `Rgb24`'s (the X byte is never
/// filtered) and its RGBA alpha must be a forced `0xFF`.
fn rgb24_filter_reference<K: FilterKernel + Copy>(
  kernel: K,
  ow: usize,
  oh: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ramp = rgb_ramp();
  let src = Rgb24Frame::new(&ramp, SRC as u32, SRC as u32, (SRC * 3) as u32);
  let (mut rgb, mut rgba, mut luma) = (
    vec![0u8; ow * oh * 3],
    vec![0u8; ow * oh * 4],
    vec![0u8; ow * oh],
  );
  {
    let mut sink = MixedSinker::<Rgb24, FilteredResampler<K>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    rgb24_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  (rgb, rgba, luma)
}

macro_rules! padding_format_filter_matches_rgb24 {
  ($name:ident, $marker:ty, $frame:ident, $walk:ident, $pack:expr) => {
    #[test]
    fn $name() {
      let pix = frame4($pack);
      // Downscale and enlarge, each over all three kernels.
      let geoms: &[(usize, usize)] = &[(OUT, OUT), (SRC * 2, SRC * 2)];
      for &(ow, oh) in geoms {
        for kind in 0u8..3 {
          let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
          let (mut rgb, mut rgba, mut luma) = (
            vec![0u8; ow * oh * 3],
            vec![0u8; ow * oh * 4],
            vec![0u8; ow * oh],
          );
          macro_rules! run_kernel {
            ($k:expr) => {{
              let mut sink = MixedSinker::<$marker, FilteredResampler<_>>::with_resampler(
                SRC,
                SRC,
                FilteredResampler::new(ow, oh, $k),
              )
              .unwrap()
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap()
              .with_luma(&mut luma)
              .unwrap();
              $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
            }};
          }
          let (rgb_ref, rgba_ref, luma_ref) = match kind {
            0 => {
              run_kernel!(Triangle);
              rgb24_filter_reference(Triangle, ow, oh)
            }
            1 => {
              run_kernel!(CatmullRom);
              rgb24_filter_reference(CatmullRom, ow, oh)
            }
            _ => {
              run_kernel!(Lanczos3);
              rgb24_filter_reference(Lanczos3, ow, oh)
            }
          };
          assert_eq!(rgb, rgb_ref, "rgb {ow}x{oh} kind {kind}");
          assert_eq!(rgba, rgba_ref, "rgba (X forced 0xFF) {ow}x{oh} kind {kind}");
          assert_eq!(luma, luma_ref, "luma {ow}x{oh} kind {kind}");
          // Belt and braces: every output alpha is exactly 0xFF (the
          // padding contract — X is never the filtered source byte).
          assert!(
            rgba.chunks_exact(4).all(|px| px[3] == 0xFF),
            "padding alpha must be forced 0xFF, {ow}x{oh} kind {kind}"
          );
        }
      }
    }
  };
}

padding_format_filter_matches_rgb24!(xrgb_filter_matches_rgb24, Xrgb, XrgbFrame, xrgb_to, |[
  r,
  g,
  b,
]| [
  PAD, r, g, b
]);
padding_format_filter_matches_rgb24!(rgbx_filter_matches_rgb24, Rgbx, RgbxFrame, rgbx_to, |[
  r,
  g,
  b,
]| [
  r, g, b, PAD
]);
padding_format_filter_matches_rgb24!(xbgr_filter_matches_rgb24, Xbgr, XbgrFrame, xbgr_to, |[
  r,
  g,
  b,
]| [
  PAD, b, g, r
]);
padding_format_filter_matches_rgb24!(bgrx_filter_matches_rgb24, Bgrx, BgrxFrame, bgrx_to, |[
  r,
  g,
  b,
]| [
  b, g, r, PAD
]);

#[test]
fn xrgb_identity_plan_matches_new_sink() {
  let pix = frame4(|[r, g, b]| [PAD, r, g, b]);
  let src = XrgbFrame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Xrgb>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    xrgb_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Xrgb, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    xrgb_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}
