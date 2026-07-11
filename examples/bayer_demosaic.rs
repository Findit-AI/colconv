//! Bayer mosaics decode through Tier 1: the raw pipeline takes the CFA
//! pattern, the demosaic algorithm, white balance and a color-correction
//! matrix as explicit walk arguments — deliberately not derivable from a
//! `ColorSpec`, which is why Bayer is the one family without a `Convert`
//! entry.
//!
//! ```sh
//! cargo run --example bayer_demosaic
//! ```

use colconv::{
  frame::BayerFrame,
  raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  sinker::{LumaCoefficients, MixedSinker},
  source::Bayer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // A 4x4 RGGB mosaic: each byte is one photosite behind the color filter.
  //   R G R G
  //   G B G B
  //   R G R G
  //   G B G B
  let raw = [
    200u8, 120, 180, 110, //
    130, 60, 125, 70, //
    190, 115, 170, 105, //
    128, 55, 120, 65,
  ];
  let (w, h) = (4u32, 4u32);
  let frame = BayerFrame::try_new(&raw, w, h, w)?;

  // Neutral white balance + identity CCM: the demosaic alone.
  let mut rgb = vec![0u8; (w * h) as usize * 3];
  {
    let mut sink = MixedSinker::<Bayer>::new(w as usize, h as usize).with_rgb(&mut rgb)?;
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )?;
  }
  println!(
    "bilinear demosaic, neutral WB, identity CCM — first pixel: rgb({}, {}, {})",
    rgb[0], rgb[1], rgb[2]
  );

  // Real camera processing: per-channel WB gains plus a CCM. Both are
  // validated constructors — finite-but-extreme values that would overflow
  // the per-pixel matmul are rejected up front.
  let wb = WhiteBalance::try_new(1.9, 1.0, 1.6)?;
  let ccm =
    ColorCorrectionMatrix::try_new([[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]])?;
  let mut graded = vec![0u8; (w * h) as usize * 3];
  {
    let mut sink = MixedSinker::<Bayer>::new(w as usize, h as usize).with_rgb(&mut graded)?;
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      wb,
      ccm,
      &mut sink,
    )?;
  }
  println!(
    "with WB(1.9, 1.0, 1.6) + CCM          — first pixel: rgb({}, {}, {})",
    graded[0], graded[1], graded[2]
  );

  // Luma from a Bayer source picks its weighting: BT.709 (default), BT.601,
  // BT.2020, DCI-P3, ACES AP1, or validated custom coefficients.
  for (label, coeffs) in [
    ("Bt709 ", LumaCoefficients::Bt709),
    ("Bt601 ", LumaCoefficients::Bt601),
    ("custom", LumaCoefficients::try_custom(0.4, 0.4, 0.2)?),
  ] {
    let mut luma = vec![0u8; (w * h) as usize];
    let mut sink = MixedSinker::<Bayer>::new(w as usize, h as usize)
      .with_luma(&mut luma)?
      .with_luma_coefficients(coeffs);
    bayer_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )?;
    println!("luma[{label}] first row: {:?}", &luma[..w as usize]);
  }

  Ok(())
}
