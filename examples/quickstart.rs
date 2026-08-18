//! The Tier-0 golden path: decode a validated source frame to RGB in one
//! call with [`Convert`].
//!
//! ```sh
//! cargo run --example quickstart
//! ```

use pixon::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, frame::Yuv420pFrame};

fn main() -> Result<(), pixon::Error> {
  // A 4x2 YUV 4:2:0 frame: three borrowed planes plus per-plane strides.
  let y = [16u8, 60, 110, 160, 200, 240, 32, 90];
  let u = [128u8, 96];
  let v = [128u8, 176];
  let (w, h) = (4usize, 2usize);
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 4, 2, 2);

  // One colorimetric truth. `resolve` pins the canonical format, dynamic
  // range and matrix; carried metadata (primaries, transfer, chroma siting)
  // stays unspecified — use `ColorSpec::from_info` when the stream provides
  // those (see the colorimetry example). Every per-format walk knob derives
  // from the spec inside `run`.
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );

  let mut rgb = vec![0u8; w * h * 3];
  Convert::from(&frame) // dimensions + format come from the frame
    .spec(spec.clone()) //          colorimetry comes from the spec
    .rgb(&mut rgb) //       attach any subset of outputs
    .run()?; //             the only fallible call

  println!("{w}x{h} Yuv420p -> RGB:");
  for (i, px) in rgb.chunks_exact(3).enumerate() {
    println!("  pixel {i}: rgb({}, {}, {})", px[0], px[1], px[2]);
  }

  // Every SIMD backend is bit-identical to the scalar reference; force the
  // scalar path with `.simd(false)` and the bytes cannot change.
  let mut scalar = vec![0u8; w * h * 3];
  Convert::from(&frame)
    .spec(spec)
    .simd(false)
    .rgb(&mut scalar)
    .run()?;
  assert_eq!(rgb, scalar, "SIMD and scalar tiers are byte-identical");
  println!("scalar reference (.simd(false)) matches byte-for-byte");

  Ok(())
}
