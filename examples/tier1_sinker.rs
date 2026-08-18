//! Tier 1 spelled out: assemble a [`MixedSinker`], drive it with the
//! matching `{format}_to` walker, and reach the sink-only knobs — geometry
//! accessors and the alpha resampling mode. `Convert` runs on exactly this
//! machinery, byte-for-byte.
//!
//! ```sh
//! cargo run --example tier1_sinker
//! ```

use pixon::{
  ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat,
  frame::{Yuv420pFrame, Yuva420pFrame},
  resample::AreaResampler,
  sinker::{AlphaMode, MixedSinker},
  source::{Yuv420p, Yuva420p, yuv420p_to, yuva420p_to},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let y = [40u8, 80, 120, 160, 200, 240, 30, 70];
  let u = [100u8, 150];
  let v = [150u8, 100];
  let (w, h) = (4usize, 2usize);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt601,
  );

  // The manual path: sink + walker. `Convert` assembles exactly this.
  let mut manual = vec![0u8; w * h * 3];
  {
    let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 4, 2, 2);
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut manual)?
      .with_color_spec(&spec);
    println!(
      "sink geometry: {}x{} -> {}x{}, produces_rgb={} produces_luma={}",
      sink.width(),
      sink.height(),
      sink.out_width(),
      sink.out_height(),
      sink.produces_rgb(),
      sink.produces_luma(),
    );
    yuv420p_to(&frame, spec.full_range(), spec.kernel_matrix()?, &mut sink)?;
  }
  let mut golden = vec![0u8; w * h * 3];
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 4, 2, 2);
  Convert::from(&frame).spec(spec).rgb(&mut golden).run()?;
  assert_eq!(manual, golden);
  println!("manual MixedSinker + yuv420p_to == Convert, byte-for-byte");

  // Alpha under resampling is a semantic choice. `Straight` (the default)
  // averages RGB and coverage independently; `Premultiplied` bins
  // alpha-weighted color and un-premultiplies per output pixel — a
  // transparent black pixel then no longer darkens its neighborhood.
  let (sw, sh) = (4usize, 4usize);
  let ya: Vec<u8> = (0..sw * sh).map(|i| (i * 16) as u8).collect();
  let ua = [64u8, 192, 128, 30];
  let va = [200u8, 90, 60, 250];
  let alpha: Vec<u8> = (0..sw * sh)
    .map(|i| if i % 3 == 0 { 0 } else { 255 })
    .collect();
  let (ow, oh) = (2usize, 2usize);
  // A fresh spec resolved against the alpha format — never reuse a spec
  // resolved for a different `PixelFormat`.
  let aspec = ColorSpec::resolve(
    PixelFormat::Yuva420p,
    DynamicRange::Limited,
    ColorMatrix::Bt601,
  );
  let mut per_mode = Vec::new();
  for mode in [AlphaMode::Straight, AlphaMode::Premultiplied] {
    let aframe = Yuva420pFrame::new(&ya, &ua, &va, &alpha, sw as u32, sh as u32, 4, 2, 2, 4);
    let mut rgba = vec![0u8; ow * oh * 4];
    {
      let mut sink =
        MixedSinker::<Yuva420p, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))?
          .with_color_spec(&aspec)
          .with_alpha_mode(mode)
          .with_rgba(&mut rgba)?;
      yuva420p_to(
        &aframe,
        aspec.full_range(),
        aspec.kernel_matrix()?,
        &mut sink,
      )?;
    }
    println!(
      "Yuva420p {sw}x{sh} -> {ow}x{oh}, {mode:?}: first pixel rgba({}, {}, {}, {})",
      rgba[0], rgba[1], rgba[2], rgba[3]
    );
    per_mode.push(rgba);
  }
  assert_ne!(
    per_mode[0], per_mode[1],
    "the two alpha modes bin differently"
  );

  Ok(())
}
