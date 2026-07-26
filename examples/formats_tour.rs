//! One golden call, every family: the same `Convert::from(&frame)` decodes
//! any source format whose frame type validates — planar, semi-planar,
//! packed, high-bit, float, palette or bit-packed. The `Source` impls are
//! emitted by the same table that drives the walkers, so nothing here is
//! hand-wired.
//!
//! (The one deliberate exception is Bayer, whose walk needs camera
//! parameters no `ColorSpec` can supply — see the bayer_demosaic example.)
//!
//! ```sh
//! cargo run --example formats_tour
//! ```

use pixon::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, Source};

/// Decodes `frame` to RGB and prints the first pixel. The spec is resolved
/// against each source's **actual** pixel format — never reuse a spec
/// resolved for a different format, or the format-pinned range handling
/// can silently derive the wrong walk options.
fn show<Fr: Source>(label: &str, frame: &Fr, fmt: PixelFormat, w: usize, h: usize) {
  let spec = ColorSpec::resolve(fmt, DynamicRange::Limited, ColorMatrix::Bt601);
  let mut rgb = vec![0u8; w * h * 3];
  Convert::from(frame).spec(spec).rgb(&mut rgb).run().unwrap();
  println!(
    "  {label:34} first pixel rgb({:3}, {:3}, {:3})",
    rgb[0], rgb[1], rgb[2]
  );
}

/// Encodes a logical `u16` as **LE-wire** storage for the `*LeFrame` types
/// (identity on little-endian hosts).
fn le16(v: u16) -> u16 {
  v.to_le()
}

/// The `f32` twin of [`le16`] for the float `*LeFrame` types.
fn le_f32(v: f32) -> f32 {
  f32::from_bits(v.to_bits().to_le())
}

/// Packs six 10-bit 4:2:2 pixels into one 16-byte V210 word group.
fn pack_v210_row(samples: [u16; 12]) -> [u8; 16] {
  let mut out = [0u8; 16];
  let pack = |a: u16, b: u16, c: u16| -> u32 {
    (a as u32 & 0x3FF) | ((b as u32 & 0x3FF) << 10) | ((c as u32 & 0x3FF) << 20)
  };
  out[0..4].copy_from_slice(&pack(samples[0], samples[1], samples[2]).to_le_bytes());
  out[4..8].copy_from_slice(&pack(samples[3], samples[4], samples[5]).to_le_bytes());
  out[8..12].copy_from_slice(&pack(samples[6], samples[7], samples[8]).to_le_bytes());
  out[12..16].copy_from_slice(&pack(samples[9], samples[10], samples[11]).to_le_bytes());
  out
}

fn main() {
  use pixon::frame::{
    Ayuv64LeFrame, GbrpFrame, Gray8Frame, MonoblackFrame, Nv12Frame, Pal8Frame, Rgb24Frame,
    Rgb48LeFrame, Rgb565Frame, Rgbaf32LeFrame, V210LeFrame, VuyxFrame, Xyz12LeFrame, Y210LeFrame,
    Ya8Frame, Yuv420p10Frame, Yuv444pFrame, Yuva420pFrame, Yuyv422Frame,
  };

  println!("planar / semi-planar YUV:");
  let (y, u, v) = (
    [81u8, 145, 41, 210],
    [90u8, 54, 240, 16],
    [240u8, 34, 110, 146],
  );
  show(
    "Yuv444p (8-bit planar)",
    &Yuv444pFrame::new(&y, &u, &v, 2, 2, 2, 2, 2),
    PixelFormat::Yuv444p,
    2,
    2,
  );

  // `Yuv420p10Frame` is the default-LE alias: its planes are LE-wire u16
  // storage, so the fixture encodes through `le16` like every `*Le` type.
  let (y10, u10, v10) = (
    [100u16, 200, 300, 400, 500, 600, 700, 800].map(le16),
    [512u16, 300].map(le16),
    [400u16, 600].map(le16),
  );
  show(
    "Yuv420p10 (10-bit planar)",
    &Yuv420p10Frame::new(&y10, &u10, &v10, 4, 2, 4, 2, 2),
    PixelFormat::Yuv420p10Le,
    4,
    2,
  );

  let (ny, nuv) = (
    [16u8, 60, 110, 160, 200, 240, 32, 90],
    [128u8, 110, 140, 120],
  );
  show(
    "Nv12 (semi-planar 4:2:0)",
    &Nv12Frame::new(&ny, &nuv, 4, 2, 4, 4),
    PixelFormat::Nv12,
    4,
    2,
  );

  println!("alpha-carrying sources (through .rgba):");
  let (ay, au, av, aa) = (
    [40u8, 80, 120, 160, 200, 240, 30, 70],
    [100u8, 150],
    [150u8, 100],
    [10u8, 20, 30, 40, 50, 60, 70, 80],
  );
  let aframe = Yuva420pFrame::new(&ay, &au, &av, &aa, 4, 2, 4, 2, 2, 4);
  let aspec = ColorSpec::resolve(
    PixelFormat::Yuva420p,
    DynamicRange::Limited,
    ColorMatrix::Bt601,
  );
  let mut rgba = vec![0u8; 4 * 2 * 4];
  Convert::from(&aframe)
    .spec(aspec)
    .rgba(&mut rgba)
    .run()
    .unwrap();
  println!(
    "  {:34} first pixel rgba({:3}, {:3}, {:3}, {:3})",
    "Yuva420p (planar + alpha plane)", rgba[0], rgba[1], rgba[2], rgba[3]
  );
  let ya = [100u8, 200, 50, 150];
  let yaframe = Ya8Frame::new(&ya, 2, 1, 4);
  let yaspec = ColorSpec::resolve(PixelFormat::Ya8, DynamicRange::Limited, ColorMatrix::Bt601);
  let mut rgba = vec![0u8; 2 * 4];
  Convert::from(&yaframe)
    .spec(yaspec)
    .rgba(&mut rgba)
    .run()
    .unwrap();
  println!(
    "  {:34} first pixel rgba({:3}, {:3}, {:3}, {:3})",
    "Ya8 (gray + alpha, interleaved)", rgba[0], rgba[1], rgba[2], rgba[3]
  );

  println!("packed YUV:");
  let yuyv = [
    40u8, 128, 80, 128, 120, 110, 160, 140, 200, 120, 30, 130, 70, 150, 240, 100,
  ];
  show(
    "Yuyv422 (packed 4:2:2)",
    &Yuyv422Frame::new(&yuyv, 4, 2, 8),
    PixelFormat::Yuyv422,
    4,
    2,
  );

  let vuyx = [
    0x80u8, 0x80, 0xA0, 0xFF, 0x70, 0x60, 0xC0, 0xFF, 0x90, 0x40, 0xB0, 0xFF, 0x50, 0x30, 0xD0,
    0xFF,
  ];
  show(
    "Vuyx (packed 4:4:4)",
    &VuyxFrame::new(&vuyx, 2, 2, 8),
    PixelFormat::Vuyx,
    2,
    2,
  );

  let ayuv: Vec<u16> = [
    [0xFFFFu16, 60160, 32768, 32768],
    [0xFFFF, 20000, 40000, 28000],
  ]
  .concat()
  .into_iter()
  .map(le16)
  .collect();
  show(
    "Ayuv64 (16-bit packed 4:4:4, LE)",
    &Ayuv64LeFrame::try_new(&ayuv, 2, 1, 8).unwrap(),
    PixelFormat::Ayuv64Le,
    2,
    1,
  );

  // Y210 stores 10-bit samples MSB-aligned in u16 lanes, [Y0 U Y1 V].
  let y210: Vec<u16> = [700u16, 512, 300, 480]
    .iter()
    .map(|s| le16(s << 6))
    .collect();
  show(
    "Y210 (10-bit packed 4:2:2, LE)",
    &Y210LeFrame::new(&y210, 2, 1, 4),
    PixelFormat::Y210Le,
    2,
    1,
  );

  // V210 packs six 4:2:2 pixels into four little-endian 10:10:10 words.
  let word = pack_v210_row([
    // [U0, Y0, V0][Y1, U1, Y2][V1, Y3, U2][Y4, V2, Y5]
    512, 100, 512, 250, 300, 400, 700, 550, 512, 700, 512, 850,
  ]);
  show(
    "V210 (10-bit packed, 6-px words)",
    &V210LeFrame::new(&word, 6, 1, 16),
    PixelFormat::V210,
    6,
    1,
  );

  println!("RGB-family sources:");
  let px = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
  show(
    "Rgb24 (packed 8-bit)",
    &Rgb24Frame::new(&px, 2, 2, 6),
    PixelFormat::Rgb24,
    2,
    2,
  );

  let px48 = [1000u16, 2000, 3000, 40000, 50000, 60000].map(le16);
  show(
    "Rgb48 (packed 16-bit, LE)",
    &Rgb48LeFrame::new(&px48, 2, 1, 6),
    PixelFormat::Rgb48Le,
    2,
    1,
  );

  let pxf = [0.1f32, 0.5, 0.9, 1.0, 0.8, 0.4, 0.2, 0.5].map(le_f32);
  show(
    "Rgbaf32 (packed float RGBA, LE)",
    &Rgbaf32LeFrame::new(&pxf, 2, 1, 8),
    PixelFormat::Rgbaf32Le,
    2,
    1,
  );

  // RGB565: one little-endian u16 per pixel (0xF800 = pure red).
  let legacy: Vec<u8> = [0xF800u16, 0x07E0, 0x001F, 0xFFFF]
    .iter()
    .flat_map(|px| px.to_le_bytes())
    .collect();
  show(
    "Rgb565 (legacy 16-bit)",
    &Rgb565Frame::try_new(&legacy, 2, 2, 4).unwrap(),
    PixelFormat::Rgb565Le,
    2,
    2,
  );

  let (g, b, r) = (
    [120u8, 200, 40, 90],
    [30u8, 60, 220, 10],
    [240u8, 20, 130, 170],
  );
  show(
    "Gbrp (planar GBR)",
    &GbrpFrame::try_new(&g, &b, &r, 2, 2, 2, 2, 2).unwrap(),
    PixelFormat::Gbrp,
    2,
    2,
  );

  println!("single-channel and indexed sources:");
  let gray = [30u8, 90, 150, 210];
  show(
    "Gray8",
    &Gray8Frame::new(&gray, 2, 2, 2),
    PixelFormat::Gray8,
    2,
    2,
  );

  // XYZ12: D-Cinema tristimulus values. The 12-bit codes sit in the top
  // bits (15:4) of each LE u16 lane.
  let xyz = [1600u16, 1800, 1500, 3000, 3200, 2800].map(|code| le16(code << 4));
  show(
    "Xyz12 (DCDM tristimulus, LE)",
    &Xyz12LeFrame::new(&xyz, 2, 1, 6),
    PixelFormat::Xyz12Le,
    2,
    1,
  );

  // 1-bit mono, MSB-first; monoblack means a set bit is white:
  // 0b1010_0000 = white, black, white, black.
  let bits = [0b1010_0000u8];
  show(
    "Monoblack (1-bit, bit-packed)",
    &MonoblackFrame::try_new(&bits, 4, 1, 1).unwrap(),
    PixelFormat::Monoblack,
    4,
    1,
  );

  let idx = [0u8, 1, 2, 3];
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 255];
  palette[1] = [40, 50, 60, 255];
  palette[2] = [70, 80, 90, 255];
  palette[3] = [100, 110, 120, 255];
  show(
    "Pal8 (palette-indexed)",
    &Pal8Frame::new(&idx, &palette, 2, 2, 2),
    PixelFormat::Pal8,
    2,
    2,
  );
}
