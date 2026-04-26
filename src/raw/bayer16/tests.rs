use super::*;
use crate::{
  frame::Bayer12Frame,
  row::{bayer16_to_rgb_row, bayer16_to_rgb_u16_row},
};
use core::convert::Infallible;

/// Sink that walks each `BayerRow16<BITS>` through the public
/// u8-output dispatcher with SIMD off.
struct CaptureRgbU8<'a, const BITS: u32> {
  out: &'a mut [u8],
  width: u32,
}

impl<const BITS: u32> PixelSink for CaptureRgbU8<'_, BITS> {
  type Input<'b> = BayerRow16<'b, BITS>;
  type Error = Infallible;

  fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
    self.width = width;
    Ok(())
  }

  fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
    let r = row.row();
    let w = self.width as usize;
    let off = r * w * 3;
    let dst = &mut self.out[off..off + 3 * w];
    bayer16_to_rgb_row::<BITS>(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      dst,
      false,
    );
    Ok(())
  }
}

impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU8<'_, BITS> {}

/// Sink that walks each `BayerRow16<BITS>` through the public
/// u16-output dispatcher with SIMD off.
struct CaptureRgbU16<'a, const BITS: u32> {
  out: &'a mut [u16],
  width: u32,
}

impl<const BITS: u32> PixelSink for CaptureRgbU16<'_, BITS> {
  type Input<'b> = BayerRow16<'b, BITS>;
  type Error = Infallible;

  fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
    self.width = width;
    Ok(())
  }

  fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
    let r = row.row();
    let w = self.width as usize;
    let off = r * w * 3;
    let dst = &mut self.out[off..off + 3 * w];
    bayer16_to_rgb_u16_row::<BITS>(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      dst,
      false,
    );
    Ok(())
  }
}

impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU16<'_, BITS> {}

/// Build a 12-bit low-packed RGGB Bayer plane from per-channel
/// nominal values (each 0..=4095). Bayer16 is low-packed: samples
/// occupy the low 12 bits of each `u16`, no shift required.
fn solid_rggb_12bit(width: u32, height: u32, r: u16, g: u16, b: u16) -> std::vec::Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut data = std::vec![0u16; w * h];
  for y in 0..h {
    for x in 0..w {
      let v = match (y & 1, x & 1) {
        (0, 0) => r,
        (0, 1) => g,
        (1, 0) => g,
        (1, 1) => b,
        _ => unreachable!(),
      };
      data[y * w + x] = v;
    }
  }
  data
}

fn assert_full_frame_u8(rgb: &[u8], w: u32, h: u32, expect: (u8, u8, u8)) {
  let w = w as usize;
  let h = h as usize;
  for y in 0..h {
    for x in 0..w {
      let i = (y * w + x) * 3;
      assert_eq!(rgb[i], expect.0, "u8 px ({x},{y}) R");
      assert_eq!(rgb[i + 1], expect.1, "u8 px ({x},{y}) G");
      assert_eq!(rgb[i + 2], expect.2, "u8 px ({x},{y}) B");
    }
  }
}

fn assert_full_frame_u16(rgb: &[u16], w: u32, h: u32, expect: (u16, u16, u16)) {
  let w = w as usize;
  let h = h as usize;
  for y in 0..h {
    for x in 0..w {
      let i = (y * w + x) * 3;
      assert_eq!(rgb[i], expect.0, "u16 px ({x},{y}) R");
      assert_eq!(rgb[i + 1], expect.1, "u16 px ({x},{y}) G");
      assert_eq!(rgb[i + 2], expect.2, "u16 px ({x},{y}) B");
    }
  }
}

#[test]
fn bayer12_solid_red_rggb_yields_u8_red_full_frame() {
  // 12-bit max = 4095. R = 4095 (white at this channel), G = B = 0.
  // Mirror-by-2 boundary handling means every output pixel
  // matches, including borders and corners.
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  assert_full_frame_u8(&rgb, w, h, (255, 0, 0));
}

#[test]
fn bayer12_solid_red_rggb_yields_u16_red_full_frame() {
  let (w, h) = (8u32, 6u32);
  let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut sink = CaptureRgbU16::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  assert_full_frame_u16(&rgb, w, h, (4095, 0, 0));
}

#[test]
fn bayer12_uniform_value_yields_uniform_u8_output() {
  // Every sample = 2048 (low-packed 12-bit midgray). u8 output:
  // 2048 / 4095 * 255 ≈ 127.53 → 128 everywhere (uniform input
  // so edge clamping doesn't shift the value).
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![2048u16; (w * h) as usize];
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  for &c in &rgb {
    assert!((c as i32 - 128).abs() <= 1, "got {c}");
  }
}

#[test]
fn bayer12_uniform_value_yields_uniform_u16_output() {
  // Every sample = 4095 (max 12-bit low-packed). u16 output
  // should be 4095 (low-packed full white).
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![4095u16; (w * h) as usize];
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut sink = CaptureRgbU16::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  for &c in &rgb {
    assert_eq!(c, 4095);
  }
}

#[test]
fn bayer10_low_packed_white_yields_full_scale_u8() {
  // 10-bit low-packed white (1023). u8 output should be 255.
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![1023u16; (w * h) as usize];
  let frame = crate::frame::Bayer10Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<10> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  for &c in &rgb {
    assert_eq!(c, 255, "10-bit low-packed white must scale to u8 255");
  }
}

#[test]
fn bayer14_low_packed_white_yields_full_scale_u16() {
  // 14-bit low-packed white (16383). u16 output should be 16383.
  let (w, h) = (8u32, 6u32);
  let raw = std::vec![16383u16; (w * h) as usize];
  let frame = crate::frame::Bayer14Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u16; (w * h * 3) as usize];
  let mut sink = CaptureRgbU16::<14> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  for &c in &rgb {
    assert_eq!(c, 16383, "14-bit low-packed white must stay 16383");
  }
}

/// `Bayer12Frame::try_new` rejects out-of-range samples as
/// `BayerFrame16Error::SampleOutOfRange` — a recoverable
/// `Result::Err`, not a panic. Sample-range validation is now
/// part of standard frame construction so the walker is fully
/// fallible.
#[test]
fn bayer12_try_new_rejects_sample_above_max() {
  let (w, h) = (4u32, 2u32);
  let mut raw = std::vec![100u16; (w * h) as usize];
  raw[3] = 4096; // just above 12-bit max
  let e = Bayer12Frame::try_new(&raw, w, h, w).unwrap_err();
  assert!(matches!(
    e,
    crate::frame::BayerFrame16Error::SampleOutOfRange {
      index: 3,
      value: 4096,
      max_valid: 4095,
    }
  ));
}

/// Codex-recommended regression: MSB-aligned 12-bit midgray
/// (e.g., `2048 << 4 = 0x8000`) is exactly the common
/// packing-mismatch bug, where a caller forgot to right-shift
/// before constructing the `Bayer12Frame`. Now caught at
/// construction as `Result::Err` instead of a runtime panic.
#[test]
fn bayer12_try_new_rejects_msb_aligned_input() {
  let (w, h) = (4u32, 2u32);
  let raw = std::vec![0x8000u16; (w * h) as usize]; // MSB-aligned 12-bit midgray
  let e = Bayer12Frame::try_new(&raw, w, h, w).unwrap_err();
  assert!(matches!(
    e,
    crate::frame::BayerFrame16Error::SampleOutOfRange {
      value: 0x8000,
      max_valid: 4095,
      ..
    }
  ));
}

/// Codex-recommended partial-output regression: a Bayer12 frame
/// with a bad sample in a *later* row used to trigger a runtime
/// panic mid-walk; now `try_new` catches the bad sample upfront
/// and returns `Err`, so the user's output buffer is never
/// touched. (The `bayer16_to` walker can no longer be reached
/// with bad sample data because no `BayerFrame16<BITS>` value
/// can exist with out-of-range samples.)
#[test]
fn bayer12_try_new_rejects_bad_sample_in_later_row() {
  let (w, h) = (4u32, 8u32);
  let mut raw = std::vec![100u16; (w * h) as usize];
  let off = (6 * w) as usize + 2;
  raw[off] = 4096; // exceeds 12-bit max
  let e = Bayer12Frame::try_new(&raw, w, h, w).unwrap_err();
  assert!(matches!(
    e,
    crate::frame::BayerFrame16Error::SampleOutOfRange {
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

/// Codex-recommended regression: a valid padded RAW buffer
/// (`stride > width`) with **stale high bits in the row
/// padding** must NOT trip the upfront pre-pass. The walker
/// only reads the active per-row region (`r * stride .. r *
/// stride + width`) so padding bytes are out of scope.
#[test]
fn bayer12_walker_accepts_padded_stride_with_dirty_padding() {
  let w: u32 = 4;
  let h: u32 = 4;
  let stride: u32 = 8; // padding = 4 samples per row
  let mut raw = std::vec![100u16; (stride * h) as usize];
  // Fill the padding with stale high bits (would trigger the
  // upfront panic if validated). Active region (cols 0..4) is
  // valid 12-bit data.
  for r in 0..(h as usize) {
    for c in 4..(stride as usize) {
      raw[r * stride as usize + c] = 0xFFFF;
    }
  }
  let frame = Bayer12Frame::try_new(&raw, w, h, stride).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  // Sanity: kernel ran without panicking. Output content is
  // not the focus; the test is the absence of panic.
}

/// Companion regression: trailing backing storage past
/// `(h - 1) * stride + width` with junk must NOT trip the
/// pre-pass either — the walker doesn't read past the last
/// active row.
#[test]
fn bayer12_walker_accepts_overlong_slice_with_trailing_junk() {
  let w: u32 = 4;
  let h: u32 = 2;
  let stride: u32 = 4;
  // Backing storage is twice the declared geometry; trailing
  // half is filled with values that would trip a wholesale
  // scan.
  let mut raw = std::vec![100u16; (stride * h * 2) as usize];
  for v in raw.iter_mut().skip((stride * h) as usize) {
    *v = 0xFFFF;
  }
  let frame = Bayer12Frame::try_new(&raw, w, h, stride).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<12> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
}

/// At BITS=16 every `u16` is valid; the dispatcher's bad-bit
/// mask is zero so the check is a no-op and 0xFFFF passes.
#[test]
fn bayer16bit_dispatcher_accepts_full_u16_range() {
  let (w, h) = (4u32, 2u32);
  let raw = std::vec![0xFFFFu16; (w * h) as usize];
  let frame = crate::frame::Bayer16Frame::try_new(&raw, w, h, w).unwrap();
  let mut rgb = std::vec![0u8; (w * h * 3) as usize];
  let mut sink = CaptureRgbU8::<16> {
    out: &mut rgb,
    width: 0,
  };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  // Solid 0xFFFF saturates to 255 on every channel.
  for &c in &rgb {
    assert_eq!(c, 255);
  }
}

#[test]
fn bayer12_walker_calls_sink_once_per_row() {
  struct CountSink<const BITS: u32> {
    rows: u32,
  }
  impl<const BITS: u32> PixelSink for CountSink<BITS> {
    type Input<'a> = BayerRow16<'a, BITS>;
    type Error = Infallible;
    fn process(&mut self, _row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
      self.rows += 1;
      Ok(())
    }
  }
  impl<const BITS: u32> BayerSink16<BITS> for CountSink<BITS> {}

  let (w, h) = (8u32, 6u32);
  let raw = std::vec![0u16; (w * h) as usize];
  let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
  let mut sink = CountSink::<12> { rows: 0 };
  bayer16_to(
    &frame,
    BayerPattern::Rggb,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  assert_eq!(sink.rows, h);
}
