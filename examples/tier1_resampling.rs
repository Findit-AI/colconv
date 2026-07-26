//! Tier-1 resampling knobs: assemble the [`MixedSinker`] yourself to reach
//! the strategies `Convert` doesn't surface — the averaging domain
//! (gamma-correct linear-light binning, scene-referred mode, caller-chosen
//! transfer function) and the swscale `BICUBLIN` resampler.
//!
//! ```sh
//! cargo run --example tier1_resampling
//! ```

use pixon::{
  ColorMatrix, ColorSpec, DynamicRange, PixelFormat,
  frame::Yuv420pFrame,
  resample::{AreaResampler, AveragingDomain, Bicublin, LinearMode, TransferFunction},
  sinker::MixedSinker,
  source::{Yuv420p, yuv420p_to},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // A high-contrast 8x8 checker: encoded-domain and linear-light averages of
  // black-and-white disagree the most (the classic gamma-blur artifact).
  let (sw, sh) = (8usize, 8usize);
  let y: Vec<u8> = (0..sw * sh)
    .map(|i| if (i % sw + i / sw) % 2 == 0 { 235 } else { 16 })
    .collect();
  let u = vec![128u8; (sw / 2) * (sh / 2)];
  let v = vec![128u8; (sw / 2) * (sh / 2)];
  let frame = Yuv420pFrame::new(&y, &u, &v, sw as u32, sh as u32, 8, 4, 4);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );
  let (ow, oh) = (2usize, 2usize);

  let decode = |domain: AveragingDomain,
                tf: Option<TransferFunction>,
                mode: LinearMode|
   -> Result<Vec<u8>, pixon::Error> {
    let mut rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink =
        MixedSinker::<Yuv420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))?
          .with_color_spec(spec)
          .with_averaging_domain(domain)
          .with_linear_mode(mode)
          .with_rgb(&mut rgb)?;
      if let Some(tf) = tf {
        sink.set_transfer_function(tf);
      }
      yuv420p_to(&frame, spec.full_range(), spec.matrix(), &mut sink)?;
    }
    Ok(rgb)
  };

  // Encoded (the default, the cv2/swscale convention) averages the gamma
  // codes; Linear decodes to linear light, averages, re-encodes. The
  // transfer function is caller-configurable and defaults per the matrix.
  let encoded = decode(AveragingDomain::Encoded, None, LinearMode::DisplayReferred)?;
  let linear = decode(AveragingDomain::Linear, None, LinearMode::DisplayReferred)?;
  let gamma22 = decode(
    AveragingDomain::Linear,
    Some(TransferFunction::Gamma22),
    LinearMode::DisplayReferred,
  )?;
  let scene = decode(AveragingDomain::Linear, None, LinearMode::SceneReferred)?;
  println!("8x8 checker -> {ow}x{oh}, first pixel:");
  println!(
    "  Encoded domain             rgb({:3}, {:3}, {:3})",
    encoded[0], encoded[1], encoded[2]
  );
  println!(
    "  Linear domain (default tf) rgb({:3}, {:3}, {:3})",
    linear[0], linear[1], linear[2]
  );
  println!(
    "  Linear domain (Gamma22)    rgb({:3}, {:3}, {:3})",
    gamma22[0], gamma22[1], gamma22[2]
  );
  println!(
    "  Linear, scene-referred     rgb({:3}, {:3}, {:3})",
    scene[0], scene[1], scene[2]
  );

  // The swscale `BICUBLIN` convention (cubic luma, bilinear chroma) is its
  // own resampler type — pass it to `with_resampler` like any other. The
  // sink can also emit 16-bit luma alongside.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut luma16 = vec![0u16; 4 * 4];
  {
    let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(sw, sh, Bicublin::to(4, 4))?
      .with_color_spec(spec)
      .with_rgb(&mut rgb)?
      .with_luma_u16(&mut luma16)?;
    yuv420p_to(&frame, spec.full_range(), spec.matrix(), &mut sink)?;
  }
  println!(
    "8x8 -> 4x4 BICUBLIN, first pixel: rgb({}, {}, {}), luma16={}",
    rgb[0], rgb[1], rgb[2], luma16[0]
  );

  Ok(())
}
