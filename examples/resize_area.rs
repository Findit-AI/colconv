//! Fused area downscale: `.resize(w, h)` splices a `cv2.INTER_AREA`-convention
//! box average into the convert walk — the frame is binned and converted in
//! one pass, never materialized at full resolution.
//!
//! ```sh
//! cargo run --example resize_area
//! ```

use colconv::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, frame::Yuv420pFrame};

fn main() -> Result<(), colconv::Error> {
  // An 8x8 4:2:0 frame with a left-to-right luma ramp and saturated chroma.
  let (w, h) = (8usize, 8usize);
  let y: Vec<u8> = (0..w * h).map(|i| ((i % w) * 32) as u8).collect();
  let u = vec![32u8; (w / 2) * (h / 2)];
  let v = vec![224u8; (w / 2) * (h / 2)];
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 8, 4, 4);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );

  // 8x8 -> 4x4: every source pixel contributes exactly once, with
  // box-coverage weights (fractional ratios work too).
  let (ow, oh) = (4usize, 4usize);
  let mut rgb = vec![0u8; ow * oh * 3];
  Convert::from(&frame)
    .spec(spec)
    .resize(ow, oh)
    .rgb(&mut rgb)
    .run()?;
  println!("8x8 -> {ow}x{oh} area downscale, first row:");
  for px in rgb[..ow * 3].chunks_exact(3) {
    println!("  rgb({:3}, {:3}, {:3})", px[0], px[1], px[2]);
  }

  // Two averaging semantics for YUV sources:
  //   native (default) — bin the Y/U/V codes, convert once per output row
  //                      (libswscale-class fused semantics, fastest);
  //   .native(false)   — convert first, average in RGB
  //                      (strict `cv2.INTER_AREA` RGB-domain semantics).
  // Luma is bit-identical either way; saturated (out-of-gamut) chroma may
  // differ slightly.
  let mut rgb_domain = vec![0u8; ow * oh * 3];
  Convert::from(&frame)
    .spec(spec)
    .resize(ow, oh)
    .native(false)
    .rgb(&mut rgb_domain)
    .run()?;
  println!(
    "native YUV-domain vs RGB-domain first pixel: rgb({}, {}, {}) vs rgb({}, {}, {})",
    rgb[0], rgb[1], rgb[2], rgb_domain[0], rgb_domain[1], rgb_domain[2]
  );

  // Area resampling is downscale-only by construction; an upscale request is
  // rejected with a typed error (use `resize_with` + a filter kernel to
  // upscale — see the resize_filtered example).
  let mut too_big = vec![0u8; 16 * 16 * 3];
  let err = Convert::from(&frame)
    .spec(spec)
    .resize(16, 16)
    .rgb(&mut too_big)
    .run()
    .unwrap_err();
  println!("area upscale rejected: {err}");

  Ok(())
}
