//! Filtered resampling: `.resize_with(kernel, w, h)` runs a separable
//! windowed-filter resample (the PIL `Image.resize` convention — byte-exact
//! to Pillow on the `u8` path) fused into the convert walk. Any ratio works,
//! including upscale.
//!
//! ```sh
//! cargo run --example resize_filtered
//! ```

use pixon::{
  ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat,
  frame::Yuv420pFrame,
  resample::{CatmullRom, FilterKernel, Lanczos3, Mitchell, Triangle},
};

/// Downscales `frame` to 3x3 under `kernel` and prints the first pixel.
fn downscale<K: FilterKernel>(
  label: &str,
  frame: &Yuv420pFrame<'_>,
  spec: ColorSpec,
  kernel: K,
) -> Result<(), pixon::Error> {
  let mut rgb = vec![0u8; 3 * 3 * 3];
  Convert::from(frame)
    .spec(spec)
    .resize_with(kernel, 3, 3)
    .rgb(&mut rgb)
    .run()?;
  println!("  {label:10} rgb({:3}, {:3}, {:3})", rgb[0], rgb[1], rgb[2]);
  Ok(())
}

fn main() -> Result<(), pixon::Error> {
  let (w, h) = (8usize, 8usize);
  let y: Vec<u8> = (0..w * h).map(|i| ((i % w) * 32) as u8).collect();
  let u: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| (i * 14) as u8).collect();
  let v = vec![160u8; (w / 2) * (h / 2)];
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 8, 4, 4);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );

  // Downscale 8x8 -> 3x3 under different kernels: sharper kernels ring,
  // softer kernels blur — the choice is visible in the bytes.
  println!("8x8 -> 3x3 first pixel per kernel:");
  downscale("Triangle", &frame, spec, Triangle)?;
  downscale("CatmullRom", &frame, spec, CatmullRom)?;
  downscale("Mitchell", &frame, spec, Mitchell)?;
  downscale("Lanczos3", &frame, spec, Lanczos3)?;

  // Upscale is a first-class filtered resample (area `.resize` would reject
  // it): 8x8 -> 12x12 under Lanczos3.
  let (uw, uh) = (12usize, 12usize);
  let mut up = vec![0u8; uw * uh * 3];
  Convert::from(&frame)
    .spec(spec)
    .resize_with(Lanczos3, uw, uh)
    .rgb(&mut up)
    .run()?;
  println!(
    "8x8 -> {uw}x{uh} Lanczos3 upscale, corners: rgb({}, {}, {}) .. rgb({}, {}, {})",
    up[0],
    up[1],
    up[2],
    up[(uw * uh - 1) * 3],
    up[(uw * uh - 1) * 3 + 1],
    up[(uw * uh - 1) * 3 + 2]
  );

  Ok(())
}
