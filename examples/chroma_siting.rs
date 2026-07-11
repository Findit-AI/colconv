//! Chroma siting: where the subsampled chroma samples actually sit relative
//! to the luma grid. Every [`ChromaLocation`] is honored — the reconstruction
//! kernels change with the siting, so the decoded RGB changes too.
//!
//! ```sh
//! cargo run --example chroma_siting
//! ```

use colconv::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, Primaries,
  Transfer, frame::Yuv420pFrame,
};

fn main() -> Result<(), colconv::Error> {
  // 4x4 YUV 4:2:0 with a strong chroma gradient, so the siting phase is
  // visible in the reconstructed pixels.
  let y = [128u8; 16];
  let u = [16u8, 240, 128, 64];
  let v = [240u8, 16, 64, 200];
  let (w, h) = (4usize, 4usize);
  let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, 4, 2, 2);
  let spec = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );

  // `.chroma_location(..)` pins the siting, overriding whatever the spec
  // carries. Horizontal (Left vs Center) and vertical (Top / Bottom
  // variants) reconstruction all differ.
  println!("first row of RGB under each siting:");
  for loc in [
    ChromaLocation::Left,
    ChromaLocation::Center,
    ChromaLocation::TopLeft,
    ChromaLocation::Top,
    ChromaLocation::BottomLeft,
    ChromaLocation::Bottom,
  ] {
    let mut rgb = vec![0u8; w * h * 3];
    Convert::from(&frame)
      .spec(spec)
      .chroma_location(loc)
      .rgb(&mut rgb)
      .run()?;
    println!("  {loc:?}: {:?}", &rgb[..w * 3]);
  }

  // The siting can also travel inside the spec itself: `from_info` carries
  // the stream's `ChromaLocation`, and the decode honors it with no
  // per-call override.
  let info = ColorInfo::new(
    Primaries::Bt709,
    Transfer::Bt709,
    ColorMatrix::Bt709,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec_center = ColorSpec::from_info(PixelFormat::Yuv420p, info);
  let mut via_spec = vec![0u8; w * h * 3];
  Convert::from(&frame)
    .spec(spec_center)
    .rgb(&mut via_spec)
    .run()?;
  let mut via_override = vec![0u8; w * h * 3];
  Convert::from(&frame)
    .spec(spec)
    .chroma_location(ChromaLocation::Center)
    .rgb(&mut via_override)
    .run()?;
  assert_eq!(via_spec, via_override);
  println!("spec-carried siting == explicit .chroma_location(Center) override");

  Ok(())
}
