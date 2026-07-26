//! One decode, every output: attach RGB, RGBA, luma and HSV buffers to the
//! same [`Convert`] and fill them in a single pass over the source.
//!
//! A YUV source with only HSV (and/or luma) attached converts straight from
//! YUV — no RGB intermediate row is ever staged.
//!
//! ```sh
//! cargo run --example multi_output
//! ```

use pixon::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, frame::Yuv420pFrame};

fn main() -> Result<(), pixon::Error> {
  let y = [40u8, 80, 120, 160, 200, 240, 30, 70];
  let u = [100u8, 150];
  let v = [150u8, 100];
  let (w, h) = (4usize, 2usize);
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 4, 2, 2);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt601,
  );

  let mut rgb = vec![0u8; w * h * 3];
  let mut rgba = vec![0u8; w * h * 4]; // opaque alpha: the source carries none
  let mut luma = vec![0u8; w * h];
  let (mut hue, mut sat, mut val) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);

  Convert::from(&frame)
    .spec(spec)
    .rgb(&mut rgb)
    .rgba(&mut rgba)
    .luma(&mut luma)
    .hsv(&mut hue, &mut sat, &mut val)
    .run()?;

  println!("{w}x{h} Yuv420p -> rgb + rgba + luma + hsv in one pass");
  for i in 0..w * h {
    let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    let a = rgba[i * 4 + 3];
    println!(
      "  pixel {i}: rgb({r:3}, {g:3}, {b:3})  a={a}  luma={:3}  hsv({:3}, {:3}, {:3})",
      luma[i], hue[i], sat[i], val[i]
    );
  }

  // HSV-only (no RGB/RGBA attached): the direct YUV -> HSV kernels run and
  // produce the same planes.
  let (mut h2, mut s2, mut v2) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
  Convert::from(&frame)
    .spec(spec)
    .hsv(&mut h2, &mut s2, &mut v2)
    .run()?;
  assert_eq!((h2, s2, v2), (hue, sat, val));
  println!("HSV-only decode (RGB-free fast path) matches the combined pass");

  Ok(())
}
