//! Colorimetry drives the decode: the matrix, dynamic range, primaries and
//! transfer all live in one [`ColorSpec`].
//!
//! Shows `resolve` (format defaults) vs `from_info` (explicit stream
//! metadata), matrix and range effects on the same bytes, the non-affine
//! IPT-C2 / SMPTE ST 2085 matrices, and the SMPTE ST 428-1 interpretation
//! toggle.
//!
//! ```sh
//! cargo run --example colorimetry
//! ```

use colconv::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, Primaries,
  Transfer,
  frame::{Yuv420pFrame, Yuv444p12LeFrame, Yuv444pFrame},
  sinker::St428Interpretation,
};

fn decode(frame: &Yuv444pFrame<'_>, spec: ColorSpec) -> Result<Vec<u8>, colconv::Error> {
  let mut rgb = vec![0u8; 2 * 2 * 3];
  Convert::from(frame).spec(spec).rgb(&mut rgb).run()?;
  Ok(rgb)
}

fn main() -> Result<(), colconv::Error> {
  // 2x2 YUV 4:4:4 (full-resolution chroma, so the matrix is the whole story).
  let y = [81u8, 145, 41, 210];
  let u = [90u8, 54, 240, 16];
  let v = [240u8, 34, 110, 146];
  let frame = Yuv444pFrame::new(&y, &u, &v, 2, 2, 2, 2, 2);

  // The same code words decode to different colors under different matrices
  // and ranges — colorimetry is not optional metadata.
  println!("same bytes, different colorimetry (first pixel):");
  for (label, matrix, range) in [
    (
      "Bt601  / limited",
      ColorMatrix::Bt601,
      DynamicRange::Limited,
    ),
    (
      "Bt709  / limited",
      ColorMatrix::Bt709,
      DynamicRange::Limited,
    ),
    (
      "Bt2020Ncl / limited",
      ColorMatrix::Bt2020Ncl,
      DynamicRange::Limited,
    ),
    ("Bt709  / full", ColorMatrix::Bt709, DynamicRange::Full),
  ] {
    let spec = ColorSpec::resolve(PixelFormat::Yuv444p, range, matrix);
    let rgb = decode(&frame, spec)?;
    println!(
      "  {label:20} -> rgb({:3}, {:3}, {:3})",
      rgb[0], rgb[1], rgb[2]
    );
  }

  // The non-affine H.273 matrices (the ICtCp family) engage on the
  // high-bit 4:4:4 path when the spec carries a PQ transfer — the Dolby
  // Vision Profile 5 shape (IPT-C2) and the SMPTE ST 2085 Y'D'zD'x shape.
  // Without the PQ transfer the decode deliberately falls back to the
  // affine path, so carrying the real stream transfer matters.
  // `*LeFrame` planes are LE-wire storage; `to_le` keeps the fixture
  // host-independent (identity on little-endian hosts).
  let ich = [2300u16, 2048, 1800, 2560].map(u16::to_le);
  let pch = [2500u16, 2048, 1900, 2100].map(u16::to_le);
  let tch = [1700u16, 2048, 2200, 2000].map(u16::to_le);
  let dolby = Yuv444p12LeFrame::new(&ich, &pch, &tch, 2, 2, 2, 2, 2);
  println!("non-affine matrices (12-bit 4:4:4, first pixel):");
  for (label, matrix, transfer) in [
    ("IptC2 + PQ", ColorMatrix::IptC2, Transfer::SmpteSt2084Pq),
    (
      "Smpte2085 + PQ",
      ColorMatrix::Smpte2085,
      Transfer::SmpteSt2084Pq,
    ),
    (
      "IptC2, no PQ (affine fallback)",
      ColorMatrix::IptC2,
      Transfer::Unspecified,
    ),
  ] {
    let spec = ColorSpec::from_info(
      PixelFormat::Yuv444p12Le,
      ColorInfo::new(
        Primaries::Bt2020,
        transfer,
        matrix,
        DynamicRange::Limited,
        ChromaLocation::Left,
      ),
    );
    let mut rgb = vec![0u8; 2 * 2 * 3];
    Convert::from(&dolby).spec(spec).rgb(&mut rgb).run()?;
    println!(
      "  {label:30} -> rgb({:3}, {:3}, {:3})",
      rgb[0], rgb[1], rgb[2]
    );
  }

  // `from_info` carries the full stream description — primaries, transfer
  // and chroma siting included — when the container provides one.
  let info = ColorInfo::new(
    Primaries::Bt2020,
    Transfer::SmpteSt2084Pq,
    ColorMatrix::Bt2020Ncl,
    DynamicRange::Limited,
    ChromaLocation::TopLeft,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv444p, info);
  println!(
    "from_info spec: matrix={:?} full_range={} primaries={:?} transfer={:?} siting={:?}",
    spec.matrix(),
    spec.full_range(),
    spec.primaries(),
    spec.transfer(),
    spec.chroma_location(),
  );

  // SMPTE ST 428-1 primaries under a chroma-derived matrix (a 4:2:0
  // D-Cinema stream): FFmpeg's tabulated D-Cinema values decode (the
  // default); the true CIE-XYZ interpretation rejects the derivation with a
  // typed error instead of inventing meaningless YCbCr weights.
  let (dy, du, dv) = ([128u8; 8], [110u8; 2], [140u8; 2]);
  let dcp = Yuv420pFrame::new(&dy, &du, &dv, 4, 2, 4, 2, 2);
  let st428 = ColorSpec::from_info(
    PixelFormat::Yuv420p,
    ColorInfo::new(
      Primaries::SmpteSt428,
      Transfer::SmpteSt428,
      ColorMatrix::ChromaDerivedNcl,
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  );
  let mut rgb = vec![0u8; 4 * 2 * 3];
  Convert::from(&dcp).spec(st428).rgb(&mut rgb).run()?;
  println!(
    "ST 428-1, FfmpegTabulated (default) -> rgb({:3}, {:3}, {:3})",
    rgb[0], rgb[1], rgb[2]
  );
  let rejected = Convert::from(&dcp)
    .spec(st428)
    .st428_interpretation(St428Interpretation::CieXyz)
    .rgb(&mut rgb)
    .run()
    .unwrap_err();
  println!("ST 428-1, CieXyz interpretation      -> rejected: {rejected}");

  Ok(())
}
