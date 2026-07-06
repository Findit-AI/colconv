use crate::{
  ColorMatrix,
  frame::{
    Gray8Frame, Gray16Frame, Gray32Frame, GrayNFrame, Grayf16Frame, Grayf32Frame, Ya8Frame,
    Ya16Frame,
  },
  sinker::MixedSinker,
  source::{
    gray8_to, gray9_to, gray9_to_endian, gray10_to, gray10_to_endian, gray12_to, gray12_to_endian,
    gray14_to, gray14_to_endian, gray16_to, gray16_to_endian, gray32_to, gray32_to_endian,
    grayf16_to, grayf16_to_endian, grayf32_to, grayf32_to_endian, ya8_to, ya16_to, ya16_to_endian,
  },
};
use half::f16;

/// Re-encode a host-native f16 slice as LE-encoded byte storage. The
/// `Grayf16Frame` plane contract is FFmpeg `grayf16le` — f16 bit patterns
/// whose underlying bytes are little-endian; the sinker applies `u16::from_le`
/// (via `<BE = false>`) to recover host-native f16 (no-op on LE, swap on BE).
fn as_le_f16(host: &[f16]) -> std::vec::Vec<f16> {
  host
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Sink kernels
/// recover the intended logical values via `u16::from_le` on both LE (no-op)
/// and BE (byte-swap) hosts.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native f32 slice as LE-encoded byte storage. The
/// `Grayf32Frame` plane contract is FFmpeg `grayf32le` — i.e. f32 bit
/// patterns whose underlying bytes are little-endian. Sink kernels apply
/// `u32::from_le` (via `<BE = false>`) to recover the host-native f32:
/// no-op on LE hosts, byte-swap on BE.
fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

// Gray formats are luma-only; full_range and matrix are unused by the kernels
// but are required by the walker signature. Use full_range=true, Bt709.
const FR: bool = true;
const M: ColorMatrix = ColorMatrix::Bt709;

fn make_gray8_frame(data: &[u8], w: u32, h: u32) -> Gray8Frame<'_> {
  Gray8Frame::new(data, w, h, w)
}
fn make_gray10_frame(data: &[u16], w: u32, h: u32) -> GrayNFrame<'_, 10> {
  GrayNFrame::new(data, w, h, w)
}
fn make_gray16_frame(data: &[u16], w: u32, h: u32) -> Gray16Frame<'_> {
  Gray16Frame::new(data, w, h, w)
}
/// Re-encode a host-native u32 slice as LE-encoded byte storage. Sink kernels
/// recover the intended logical values via `u32::from_le` on both LE (no-op)
/// and BE (byte-swap) hosts.
fn as_le_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}
fn as_be_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}
fn make_gray32_frame(data: &[u32], w: u32, h: u32) -> Gray32Frame<'_> {
  Gray32Frame::new(data, w, h, w)
}

#[test]
fn gray8_with_rgb_broadcasts_to_packed() {
  // 4x1 frame: [0, 64, 128, 255]
  let plane = [0u8, 64, 128, 255];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  // Each pixel should be [Y, Y, Y]
  assert_eq!(rgb[0..3], [0, 0, 0]);
  assert_eq!(rgb[3..6], [64, 64, 64]);
  assert_eq!(rgb[6..9], [128, 128, 128]);
  assert_eq!(rgb[9..12], [255, 255, 255]);
}

#[test]
fn gray8_with_rgba_alpha_is_0xff() {
  let plane = [100u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgba = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  // Alpha byte (index 3, 7, 11, 15) should be 0xFF.
  for i in 0..4 {
    assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
    assert_eq!(rgba[i * 4], 100, "pixel {i} R");
  }
}

#[test]
fn gray8_with_luma_copies_plane() {
  let plane: Vec<u8> = (0..16u8).collect();
  let frame = make_gray8_frame(&plane, 4, 4);
  let mut luma = std::vec![0u8; 16];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 4)
    .with_luma(&mut luma)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(luma, plane);
}

#[test]
fn gray8_with_luma_u16_zero_extends() {
  let plane = [0u8, 64, 128, 255];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [0, 64, 128, 255]);
}

#[test]
fn gray8_with_hsv_h_s_zero_v_equals_y() {
  let plane = [50u8, 100, 200, 0];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, plane.as_slice(), "V must equal Y");
}

#[test]
fn gray10_with_rgb_masks_and_shifts() {
  // 10-bit sample: value 512 = 0b10_0000_0000, masked = 512, >> 2 = 128
  let plane = as_le_u16(&[512u16; 4]);
  let frame = make_gray10_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray10>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray10_to(&frame, FR, M, &mut sink).unwrap();
  // 512 & 0x3FF = 512, >> 2 = 128. All channels should be 128.
  assert_eq!(rgb[0..3], [128, 128, 128]);
  assert_eq!(rgb[3..6], [128, 128, 128]);
}

#[test]
fn gray10_with_luma_u16_masks_only() {
  // 10-bit, over-range sample: 0x0800 (bit 11 set) masked → 0.
  let plane = as_le_u16(&[0x0800u16, 0x03FFu16, 0x0200u16, 0x0001u16]);
  let frame = make_gray10_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::source::Gray10>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray10_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [0x0000, 0x03FF, 0x0200, 0x0001]);
}

#[test]
fn gray16_with_rgb_shifts_to_u8() {
  // Gray16 sample 0x8000 → >> 8 = 0x80 = 128.
  let plane = as_le_u16(&[0x8000u16, 0xFFFFu16, 0x0000u16, 0x0100u16]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  // Each pixel [Y>>8, Y>>8, Y>>8]
  assert_eq!(rgb[0..3], [0x80, 0x80, 0x80]);
  assert_eq!(rgb[3..6], [0xFF, 0xFF, 0xFF]);
  assert_eq!(rgb[6..9], [0x00, 0x00, 0x00]);
  assert_eq!(rgb[9..12], [0x01, 0x01, 0x01]);
}

#[test]
fn gray16_with_luma_u16_copies_plane() {
  // Host-native intended luma values; the plane bytes must be LE-encoded
  // so the `<BE = false>` Gray16 kernel decodes them to `intended` on
  // every host (no-op LE / byte-swap BE).
  let intended: Vec<u16> = (0u16..16).map(|x| x * 4096).collect();
  let plane = as_le_u16(&intended);
  let frame = make_gray16_frame(&plane, 4, 4);
  let mut lu16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 4)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, intended);
}

#[test]
fn gray16_with_rgba_u16_alpha_is_0xffff() {
  let plane = as_le_u16(&[0x1234u16; 4]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
    assert_eq!(rgba_u16[i * 4], 0x1234, "pixel {i} R");
  }
}

#[test]
fn gray9_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = as_le_u16(&[100u16; 4]);
  let frame: GrayNFrame<'_, 9> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray9>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray9_to(&frame, FR, M, &mut sink).unwrap();
  // 100 & 0x1FF = 100, >> 1 = 50.
  assert_eq!(luma, [50, 50, 50, 50]);
}

#[test]
fn gray12_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = as_le_u16(&[0x0FFFu16; 4]);
  let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray12>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray12_to(&frame, FR, M, &mut sink).unwrap();
  // 0x0FFF & 0x0FFF = 0x0FFF = 4095. >> 4 = 255.
  assert_eq!(luma, [255, 255, 255, 255]);
}

#[test]
fn gray14_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = as_le_u16(&[0x3FFFu16; 4]);
  let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray14>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray14_to(&frame, FR, M, &mut sink).unwrap();
  // 0x3FFF & 0x3FFF = 0x3FFF = 16383. >> 6 = 255.
  assert_eq!(luma, [255, 255, 255, 255]);
}

// ---- Limited-range integration tests ----------------------------------------
//
// For 8-bit limited-range: black=16, white=235, range=219.
//   rescale(y) = clamp_u8(((y - 16) * 255 + 109) / 219)
// For N-bit limited-range: black = 16 << (N-8), range = 219 << (N-8).
//   rescale(y) = clamp_u8(((y - black) * 255 + range/2) / range)
// Luma outputs always pass raw Y through (no rescaling regardless of
// full_range).

#[test]
fn gray8_limited_range_black_maps_to_zero() {
  // Y=16 (limited-range black) → RGB(0, 0, 0).
  let plane = [16u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0xFFu8; 12];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3..i * 3 + 3], [0, 0, 0], "pixel {i}");
  }
}

#[test]
fn gray8_limited_range_white_maps_to_255() {
  // Y=235 (limited-range white) → RGB(255, 255, 255).
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3..i * 3 + 3], [255, 255, 255], "pixel {i}");
  }
}

#[test]
fn gray8_limited_range_midpoint() {
  // Y=125 → ((125-16)*255+109)/219 = 27904/219 = 127.
  let plane = [125u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3], 127, "pixel {i} R");
  }
}

#[test]
fn gray8_limited_range_luma_passthrough_unchanged() {
  // Luma output must pass raw Y through even for limited-range; no rescaling.
  let plane = [16u8, 235u8, 125u8, 0u8];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut luma = std::vec![0xAAu8; 4];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(luma, [16, 235, 125, 0]);
}

#[test]
fn gray8_limited_range_rgba_alpha_is_0xff() {
  // Verify limited-range RGBA: alpha=0xFF, channels rescaled.
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgba = std::vec![0u8; 16];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba[i * 4], 255, "pixel {i} R");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
  }
}

#[test]
fn gray8_limited_range_hsv_v_is_rescaled() {
  // HSV V channel must use rescaled Y in limited-range mode.
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray8>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
}

#[test]
fn gray10_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 10-bit: black=64, white=940, range=876.
  let plane = as_le_u16(&[64u16, 940, 64, 940]);
  let frame: GrayNFrame<'_, 10> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray10>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray10_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=64 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=940 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=64 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=940 → white");
}

#[test]
fn gray12_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 12-bit: black=256, white=3760, range=3504.
  let plane = as_le_u16(&[256u16, 3760, 256, 3760]);
  let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray12>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray12_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=256 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=3760 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=256 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=3760 → white");
}

#[test]
fn gray14_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 14-bit: black=1024, white=15040, range=14016.
  let plane = as_le_u16(&[1024u16, 15040, 1024, 15040]);
  let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray14>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray14_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=1024 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=15040 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=1024 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=15040 → white");
}

#[test]
fn gray16_limited_range_black_and_white() {
  // 16-bit: black=4096, white=60160, range=56064.
  let plane = as_le_u16(&[4096u16, 60160, 4096, 60160]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=4096 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=60160 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=4096 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=60160 → white");
}

#[test]
fn gray16_limited_range_luma_passthrough_unchanged() {
  // Luma u16 must copy raw Y regardless of full_range.
  let plane = as_le_u16(&[4096u16, 60160, 32768, 0]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(lu16, [4096, 60160, 32768, 0]);
}

#[test]
fn gray16_limited_range_rgba_u16_alpha_is_0xffff() {
  // RGBA u16 — alpha=0xFFFF; channels hold the native Y broadcast.
  // In limited-range the u16 RGB path passes native Y through (no >>8).
  let plane = as_le_u16(&[4096u16; 4]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
  }
}

#[test]
fn gray16_limited_range_rgba_u16_channels_rescale_at_boundaries() {
  // Regression for the i32-overflow bug at BITS=16: limited-range white
  // 60160 x max_native 65535 ≈ 3.67e9 overflows i32. Math runs in i64;
  // assert that RGB channels reach black=0 and white=65535 at the
  // limited-range boundaries (pinning u16-channel values, not just alpha).
  let plane = as_le_u16(&[4096u16, 60160u16, 65535u16, 0u16]);
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  // pixel 0: limited black 4096 → 0
  assert_eq!(&rgba_u16[0..3], &[0, 0, 0]);
  // pixel 1: limited white 60160 → 65535 (over-i32 path)
  assert_eq!(&rgba_u16[4..7], &[65535, 65535, 65535]);
  // pixel 2: over-white 65535 → clamped to 65535
  assert_eq!(&rgba_u16[8..11], &[65535, 65535, 65535]);
  // pixel 3: below-black 0 → clamped to 0
  assert_eq!(&rgba_u16[12..15], &[0, 0, 0]);
  // alpha unchanged
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF);
  }
}

#[test]
fn gray16_limited_range_rgb_u16_channels_rescale_at_boundaries() {
  // Same i32-overflow regression on the with_rgb_u16 path.
  let plane = as_le_u16(&[4096u16, 60160u16]);
  let frame = make_gray16_frame(&plane, 2, 1);
  let mut rgb_u16 = std::vec![0u16; 6];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(&rgb_u16[0..3], &[0, 0, 0]);
  assert_eq!(&rgb_u16[3..6], &[65535, 65535, 65535]);
}

#[test]
fn gray16_limited_range_hsv_v_is_rescaled() {
  // HSV V must reflect limited-range rescaling.
  let plane = as_le_u16(&[60160u16; 4]); // white
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray16>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
}

// ---- Grayf32 integration tests ----------------------------------------------

#[test]
fn grayf32_with_luma_f32_passthrough() {
  // NaN, out-of-range, and normal values all pass through unchanged.
  let intended: std::vec::Vec<f32> = std::vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5, f32::NAN];
  let plane = as_le_f32(&intended);
  let frame = Grayf32Frame::new(&plane, 8, 1, 8);
  let mut out = std::vec![0.0f32; 8];
  let mut sink = MixedSinker::<crate::source::Grayf32>::new(8, 1)
    .with_luma_f32(&mut out)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  for (i, (&a, &b)) in intended.iter().zip(out.iter()).enumerate() {
    if a.is_nan() {
      assert!(b.is_nan(), "pixel {i}: expected NaN");
    } else {
      assert_eq!(a, b, "pixel {i}");
    }
  }
}

#[test]
fn grayf32_with_rgb_f32_replicates_losslessly() {
  let intended: std::vec::Vec<f32> = std::vec![0.25, 0.75, 1.5, -0.5];
  let plane = as_le_f32(&intended);
  let frame = Grayf32Frame::new(&plane, 4, 1, 4);
  let mut out = std::vec![0.0f32; 4 * 3];
  let mut sink = MixedSinker::<crate::source::Grayf32>::new(4, 1)
    .with_rgb_f32(&mut out)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  for (x, &y) in intended.iter().enumerate() {
    assert_eq!(out[x * 3], y, "pixel {x} R");
    assert_eq!(out[x * 3 + 1], y, "pixel {x} G");
    assert_eq!(out[x * 3 + 2], y, "pixel {x} B");
  }
}

#[test]
fn grayf32_with_rgb_saturates() {
  // -0.5 → 0, 0.5 → 128, 1.0 → 255, 1.5 → 255
  let intended: std::vec::Vec<f32> = std::vec![-0.5, 0.0, 0.5, 1.0, 1.5];
  let plane = as_le_f32(&intended);
  let frame = Grayf32Frame::new(&plane, 5, 1, 5);
  let mut rgb = std::vec![0u8; 5 * 3];
  let mut sink = MixedSinker::<crate::source::Grayf32>::new(5, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgb[0..3], &[0, 0, 0]); // -0.5 clamps to 0
  assert_eq!(&rgb[3..6], &[0, 0, 0]); // 0.0
  assert_eq!(&rgb[6..9], &[128, 128, 128]); // 0.5 x 255 + 0.5 = 128
  assert_eq!(&rgb[9..12], &[255, 255, 255]); // 1.0
  assert_eq!(&rgb[12..15], &[255, 255, 255]); // 1.5 clamps to 255
}

#[test]
fn grayf32_with_hsv_h0_s0_v_saturated() {
  let intended: std::vec::Vec<f32> = std::vec![0.0, 0.5, 1.0];
  let plane = as_le_f32(&intended);
  let frame = Grayf32Frame::new(&plane, 3, 1, 3);
  let mut h = std::vec![0xFFu8; 3];
  let mut s = std::vec![0xFFu8; 3];
  let mut v = std::vec![0u8; 3];
  let mut sink = MixedSinker::<crate::source::Grayf32>::new(3, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0]);
  assert_eq!(s, [0, 0, 0]);
  assert_eq!(v, [0, 128, 255]);
}

#[test]
fn grayf32_with_luma_u16_and_rgb_u16() {
  // 1x1 frame: Y = 0.5 → luma_u16 ≈ 32768, rgb_u16 ≈ [32768, 32768, 32768]
  let intended = std::vec![0.5f32];
  let plane = as_le_f32(&intended);
  let frame = Grayf32Frame::new(&plane, 1, 1, 1);
  let mut lu16 = std::vec![0u16; 1];
  let mut rgb_u16 = std::vec![0u16; 3];
  let mut sink = MixedSinker::<crate::source::Grayf32>::new(1, 1)
    .with_luma_u16(&mut lu16)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  // (0.5 * 65535 + 0.5) as u16 = 32768
  assert_eq!(lu16[0], 32768);
  assert_eq!(rgb_u16, [32768, 32768, 32768]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let intended: std::vec::Vec<f32> = (0..w).map(|i| i as f32 / w as f32).collect();
    let plane = as_le_f32(&intended);
    let frame = Grayf32Frame::new(&plane, w as u32, 1, w as u32);
    let mut rgb = std::vec![0u8; w * 3];
    let mut luma_f32 = std::vec![0.0f32; w];
    let mut sink = MixedSinker::<crate::source::Grayf32>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
    // Verify first and last pixel.
    assert_eq!(rgb[0], 0, "w={w} first R");
    assert_eq!(luma_f32[0], 0.0, "w={w} first luma_f32");
    assert!(luma_f32[w - 1] > 0.9, "w={w} last luma_f32");
  }
}

/// Sinker-layer Frame-contract regression for `Grayf32`.
///
/// [`Grayf32Frame`] documents its `&[f32]` plane as **FFmpeg `grayf32le`**
/// (see `src/frame/gray.rs`): the byte layout is little-endian-encoded f32,
/// produced by FFmpeg and reinterpreted as `&[f32]` via
/// `bytemuck::cast_slice`. This is **not** host-native f32 on a BE host —
/// the bytes are byte-swapped from the intended values until the loader
/// applies `u32::from_le`.
///
/// The `Grayf32` sinker therefore correctly hardcodes `::<false>` (i.e.
/// "input is LE-encoded") on every host:
///
///   • LE host: `from_le` is a no-op → LE bytes read as LE-interpreted f32
///     → correct host-native value.
///   • BE host: `from_le` is a byte-swap → restores LE-encoded bytes to
///     host-native f32 → correct host-native value.
///
/// This test constructs an explicitly LE-encoded f32 fixture (mirroring
/// `bytemuck::cast_slice` over `f32::to_le_bytes` output) and feeds it
/// through the sinker. On a LE host the assertion is vacuous (LE bytes
/// already are host-native), but it pins the contract; on a BE host it
/// catches any regression that drops the `::<false>` routing.
///
/// Replaces the two earlier (incorrectly-typed) regressions that assumed
/// `Grayf32Frame` was host-native f32.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_sinker_le_encoded_frame_decodes_correctly() {
  use crate::source::Grayf32;

  // Host-native intended values (mix in-range, HDR, negative).
  let w = 16usize;
  let h = 4usize;
  let mut intended = std::vec![0.0f32; w * h];
  for (i, v) in intended.iter_mut().enumerate() {
    *v = match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    };
  }

  // Build an `&[f32]` whose bit pattern, when read as raw bytes, is
  // little-endian-encoded — i.e. the layout an FFmpeg `grayf32le` plane
  // hands to `bytemuck::cast_slice`. We do this without a `&[u8]` →
  // `&[f32]` cast (which would need 4-byte alignment) by storing the
  // LE-encoded `u32` bit pattern directly into an aligned `Vec<f32>`:
  //
  //   `f32::from_bits(intended.to_bits().to_le())`
  //
  //   • LE host: `to_le` is a no-op → element bits = intended bits → the
  //     in-memory bytes are LE-encoded (which on a LE host is also the
  //     host-native f32 = intended).
  //   • BE host: `to_le` byte-swaps → element bits = byte-swapped intended
  //     → the in-memory bytes are LE-encoded; reinterpreted as host-native
  //     (BE) f32 they are *not* the intended value. The sinker's
  //     `from_le` swap then restores the intended bits.
  let le_plane: std::vec::Vec<f32> = intended
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let frame = Grayf32Frame::new(&le_plane, w as u32, h as u32, w as u32);

  // luma_f32 pass-through must restore host-native intended values.
  let mut luma_f32_out = std::vec![0.0f32; w * h];
  {
    let mut sink = MixedSinker::<Grayf32>::new(w, h)
      .with_luma_f32(&mut luma_f32_out)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_f32_out, intended,
    "Grayf32 sinker failed to decode LE-encoded plane to host-native"
  );

  // rgb_f32 lossless replicate (R = G = B = host-native Y, bit-exact).
  let mut rgb_f32_out = std::vec![0.0f32; w * h * 3];
  {
    let mut sink = MixedSinker::<Grayf32>::new(w, h)
      .with_rgb_f32(&mut rgb_f32_out)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
  }
  for (x, &y) in intended.iter().enumerate() {
    assert_eq!(rgb_f32_out[x * 3], y, "pixel {x} R diverges");
    assert_eq!(rgb_f32_out[x * 3 + 1], y, "pixel {x} G diverges");
    assert_eq!(rgb_f32_out[x * 3 + 2], y, "pixel {x} B diverges");
  }
}

// ---- Grayf16 integration tests ----------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_luma_f32_widens() {
  // NaN, out-of-range, and normal f16 values all widen to f32 unchanged.
  let intended: std::vec::Vec<f16> = std::vec![
    f16::from_f32(0.0),
    f16::from_f32(0.25),
    f16::from_f32(0.5),
    f16::from_f32(0.75),
    f16::from_f32(1.0),
    f16::from_f32(1.5),
    f16::from_f32(-0.5),
    f16::NAN,
  ];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 8, 1, 8);
  let mut out = std::vec![0.0f32; 8];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(8, 1)
    .with_luma_f32(&mut out)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  for (i, (&a, &b)) in intended.iter().zip(out.iter()).enumerate() {
    if a.is_nan() {
      assert!(b.is_nan(), "pixel {i}: expected NaN");
    } else {
      assert_eq!(a.to_f32(), b, "pixel {i}");
    }
  }
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_rgb_f32_replicates_losslessly() {
  let intended: std::vec::Vec<f16> = std::vec![
    f16::from_f32(0.25),
    f16::from_f32(0.75),
    f16::from_f32(1.5),
    f16::from_f32(-0.5),
  ];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 4, 1, 4);
  let mut out = std::vec![0.0f32; 4 * 3];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(4, 1)
    .with_rgb_f32(&mut out)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  for (x, &y) in intended.iter().enumerate() {
    let yf = y.to_f32();
    assert_eq!(out[x * 3], yf, "pixel {x} R");
    assert_eq!(out[x * 3 + 1], yf, "pixel {x} G");
    assert_eq!(out[x * 3 + 2], yf, "pixel {x} B");
  }
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_rgb_saturates() {
  // -0.5 → 0, 0.5 → 128, 1.0 → 255, 1.5 → 255
  let intended: std::vec::Vec<f16> = std::vec![
    f16::from_f32(-0.5),
    f16::from_f32(0.0),
    f16::from_f32(0.5),
    f16::from_f32(1.0),
    f16::from_f32(1.5),
  ];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 5, 1, 5);
  let mut rgb = std::vec![0u8; 5 * 3];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(5, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgb[0..3], &[0, 0, 0]); // -0.5 clamps to 0
  assert_eq!(&rgb[3..6], &[0, 0, 0]); // 0.0
  assert_eq!(&rgb[6..9], &[128, 128, 128]); // 0.5 x 255 + 0.5 = 128
  assert_eq!(&rgb[9..12], &[255, 255, 255]); // 1.0
  assert_eq!(&rgb[12..15], &[255, 255, 255]); // 1.5 clamps to 255
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_hsv_h0_s0_v_saturated() {
  let intended: std::vec::Vec<f16> =
    std::vec![f16::from_f32(0.0), f16::from_f32(0.5), f16::from_f32(1.0)];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 3, 1, 3);
  let mut h = std::vec![0xFFu8; 3];
  let mut s = std::vec![0xFFu8; 3];
  let mut v = std::vec![0u8; 3];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(3, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0]);
  assert_eq!(s, [0, 0, 0]);
  assert_eq!(v, [0, 128, 255]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_luma_u16_and_rgb_u16() {
  // 1x1 frame: Y = 0.5 → luma_u16 = 32768, rgb_u16 = [32768; 3]
  let intended = std::vec![f16::from_f32(0.5)];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 1, 1, 1);
  let mut lu16 = std::vec![0u16; 1];
  let mut rgb_u16 = std::vec![0u16; 3];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(1, 1)
    .with_luma_u16(&mut lu16)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  // (0.5 * 65535 + 0.5) as u16 = 32768
  assert_eq!(lu16[0], 32768);
  assert_eq!(rgb_u16, [32768, 32768, 32768]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_with_rgba_alpha_opaque() {
  let intended = std::vec![f16::from_f32(1.0), f16::from_f32(0.0)];
  let plane = as_le_f16(&intended);
  let frame = Grayf16Frame::new(&plane, 2, 1, 2);
  let mut rgba = std::vec![0u8; 2 * 4];
  let mut sink = MixedSinker::<crate::source::Grayf16>::new(2, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  grayf16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgba[0..4], &[255, 255, 255, 0xFF]);
  assert_eq!(&rgba[4..8], &[0, 0, 0, 0xFF]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let intended: std::vec::Vec<f16> = (0..w).map(|i| f16::from_f32(i as f32 / w as f32)).collect();
    let plane = as_le_f16(&intended);
    let frame = Grayf16Frame::new(&plane, w as u32, 1, w as u32);
    let mut rgb = std::vec![0u8; w * 3];
    let mut luma_f32 = std::vec![0.0f32; w];
    let mut sink = MixedSinker::<crate::source::Grayf16>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
    grayf16_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(rgb[0], 0, "w={w} first R");
    assert_eq!(luma_f32[0], 0.0, "w={w} first luma_f32");
    assert!(luma_f32[w - 1] > 0.9, "w={w} last luma_f32");
  }
}

/// BE/LE parity at the sinker layer: a single host-native `intended` fixture is
/// materialised as LE-encoded bytes (fed to `grayf16_to` = `BE = false`) and
/// BE-encoded bytes (fed to `grayf16_to_endian::<_, true>`); both must decode
/// to the same host-native widened f32 luma.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_sinker_be_le_parity() {
  use crate::source::Grayf16;
  let w = 16usize;
  let h = 4usize;
  let intended: std::vec::Vec<f16> = (0..w * h)
    .map(|i| match i % 4 {
      0 => f16::from_f32(0.5),
      1 => f16::from_f32(1.5),
      2 => f16::from_f32(-0.25),
      _ => f16::from_f32(0.125),
    })
    .collect();

  let le_plane = as_le_f16(&intended);
  let be_plane: std::vec::Vec<f16> = intended
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect();

  let mut out_le = std::vec![0.0f32; w * h];
  {
    let le_frame = Grayf16Frame::new(&le_plane, w as u32, h as u32, w as u32);
    let mut sink = MixedSinker::<Grayf16>::new(w, h)
      .with_luma_f32(&mut out_le)
      .unwrap();
    grayf16_to(&le_frame, FR, M, &mut sink).unwrap();
  }

  let mut out_be = std::vec![0.0f32; w * h];
  {
    let be_frame = crate::frame::Grayf16Frame::<true>::new(&be_plane, w as u32, h as u32, w as u32);
    let mut sink = MixedSinker::<crate::source::Grayf16<true>>::new(w, h)
      .with_luma_f32(&mut out_be)
      .unwrap();
    grayf16_to_endian::<_, true>(&be_frame, FR, M, &mut sink).unwrap();
  }

  let expected: std::vec::Vec<f32> = intended.iter().map(|v| v.to_f32()).collect();
  assert_eq!(
    out_le, expected,
    "LE sinker decode must widen to host-native"
  );
  assert_eq!(
    out_be, expected,
    "BE sinker decode must widen to host-native"
  );
}

// ---- Ya8 integration tests --------------------------------------------------

#[test]
fn ya8_with_rgb_and_rgba_strategy_a_plus() {
  // 2-pixel Ya8: [Y=100, A=200], [Y=50, A=150]
  let packed: std::vec::Vec<u8> = std::vec![100, 200, 50, 150];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut rgb = std::vec![0u8; 6];
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::source::Ya8>::new(2, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  // RGB: Y broadcast, α dropped.
  assert_eq!(&rgb[0..3], &[100, 100, 100]);
  assert_eq!(&rgb[3..6], &[50, 50, 50]);
  // RGBA: Y broadcast, α from source.
  assert_eq!(&rgba[0..4], &[100, 100, 100, 200]);
  assert_eq!(&rgba[4..8], &[50, 50, 50, 150]);
}

#[test]
fn ya8_standalone_rgba_source_alpha() {
  // Standalone RGBA path (no RGB requested).
  let packed: std::vec::Vec<u8> = std::vec![77, 11, 88, 22];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::source::Ya8>::new(2, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgba[0..4], &[77, 77, 77, 11]);
  assert_eq!(&rgba[4..8], &[88, 88, 88, 22]);
}

#[test]
fn ya8_with_luma_u16_zero_extends() {
  let packed: std::vec::Vec<u8> = std::vec![200, 50, 100, 25];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut lu16 = std::vec![0u16; 2];
  let mut sink = MixedSinker::<crate::source::Ya8>::new(2, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [200, 100]);
}

#[test]
fn ya8_with_hsv_h0_s0_v_y() {
  let packed: std::vec::Vec<u8> = std::vec![200, 50, 100, 25];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut h = std::vec![0xFFu8; 2];
  let mut s = std::vec![0xFFu8; 2];
  let mut v = std::vec![0u8; 2];
  let mut sink = MixedSinker::<crate::source::Ya8>::new(2, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [200, 100]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya8_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let packed: std::vec::Vec<u8> = (0..w).flat_map(|i| [i as u8, (255 - i as u8)]).collect();
    let frame = Ya8Frame::new(&packed, w as u32, 1, (w * 2) as u32);
    let mut rgb = std::vec![0u8; w * 3];
    let mut rgba = std::vec![0u8; w * 4];
    let mut sink = MixedSinker::<crate::source::Ya8>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    ya8_to(&frame, FR, M, &mut sink).unwrap();
    // Spot-check pixel 0: Y=0, A=255
    assert_eq!(&rgb[0..3], &[0, 0, 0], "w={w}");
    assert_eq!(&rgba[0..4], &[0, 0, 0, 255], "w={w}");
  }
}

// ---- Ya16 integration tests -------------------------------------------------

#[test]
fn ya16_with_rgba_u16_source_alpha() {
  // 1-pixel: Y=0x8000, A=0x4000. The Ya16 plane is FFmpeg
  // `AV_PIX_FMT_YA16LE`-byte-encoded; wrap host-native samples via
  // `as_le_u16` so kernels recover host-native values on every host.
  let packed = as_le_u16(&[0x8000u16, 0x4000]);
  let frame = Ya16Frame::new(&packed, 1, 1, 2);
  let mut rgba_u16 = std::vec![0u16; 4];
  let mut luma_u16 = std::vec![0u16; 1];
  let mut sink = MixedSinker::<crate::source::Ya16>::new(1, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgba_u16, &[0x8000, 0x8000, 0x8000, 0x4000]);
  assert_eq!(luma_u16[0], 0x8000);
}

#[test]
fn ya16_with_rgba_u8_source_alpha_shifted() {
  // 2-pixel: [Y=0x8000, A=0x4000], [Y=0xFFFF, A=0x8000]
  let packed = as_le_u16(&[0x8000u16, 0x4000, 0xFFFF, 0x8000]);
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::source::Ya16>::new(2, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  // Y=0x8000>>8=0x80=128, A=0x4000>>8=0x40=64
  assert_eq!(&rgba[0..4], &[0x80, 0x80, 0x80, 0x40]);
  // Y=0xFFFF>>8=0xFF=255, A=0x8000>>8=0x80=128
  assert_eq!(&rgba[4..8], &[0xFF, 0xFF, 0xFF, 0x80]);
}

#[test]
fn ya16_with_rgb_and_rgba_strategy_a_plus() {
  let packed = as_le_u16(&[0x8000u16, 0x4000, 0x2000, 0xC000]);
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut rgb = std::vec![0u8; 6];
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::source::Ya16>::new(2, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  // Y=0x8000>>8=0x80, A dropped for rgb.
  assert_eq!(&rgb[0..3], &[0x80, 0x80, 0x80]);
  // RGBA: Y broadcast, A=0x4000>>8=0x40
  assert_eq!(&rgba[0..4], &[0x80, 0x80, 0x80, 0x40]);
  // Pixel 1: Y=0x2000>>8=0x20, A=0xC000>>8=0xC0
  assert_eq!(&rgb[3..6], &[0x20, 0x20, 0x20]);
  assert_eq!(&rgba[4..8], &[0x20, 0x20, 0x20, 0xC0]);
}

#[test]
fn ya16_with_hsv_h0_s0_v_shifted() {
  let packed = as_le_u16(&[0x8000u16, 0x4000, 0xFFFF, 0x0000]);
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut h = std::vec![0xFFu8; 2];
  let mut s = std::vec![0xFFu8; 2];
  let mut v = std::vec![0u8; 2];
  let mut sink = MixedSinker::<crate::source::Ya16>::new(2, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [0x80, 0xFF]);
}

/// Strategy A+ (combined `with_rgb` + `with_rgba`) must produce α bytes
/// byte-identical to the standalone `with_rgba` path. Locks down the
/// corruption where a BE host processing the LE-encoded `Ya16Frame`
/// would otherwise diverge between the two paths: standalone
/// uses the endian-aware `ya16_to_rgba_row::<false>` kernel; combined
/// expanded RGB → RGBA then patched α via `copy_alpha_ya_u16_to_u8` which
/// previously read raw `packed[n*2+1]` host-native and so emitted a
/// byte-reversed α byte on BE. After the fix, `copy_alpha_ya_u16_to_u8`
/// is target-endian-aware (`<false>` for the LE Frame contract) and the
/// two paths agree on every host.
///
/// To exercise the LE-encoded byte contract on every host we build the
/// `&[u16]` plane by bit-casting LE bytes — `u16::from_le_bytes` per
/// sample. On LE hosts that's a no-op; on BE hosts it byte-swaps so the
/// in-memory bytes match the FFmpeg `AV_PIX_FMT_YA16LE` layout.
#[test]
fn ya16_combined_rgb_and_rgba_alpha_matches_standalone_le_encoded() {
  let w: u32 = 8;
  let h: u32 = 1;
  // Logical samples (Y, A) per pixel.
  let samples: [(u16, u16); 8] = [
    (0x0000, 0xFFFF),
    (0x8000, 0x4000),
    (0xFFFF, 0x0000),
    (0x1234, 0xABCD),
    (0x00FF, 0xFF00),
    (0x5A5A, 0xA5A5),
    (0x7FFF, 0x8000),
    (0xC000, 0x3FFF),
  ];
  // Build the `&[u16]` plane such that its in-memory bytes match the
  // FFmpeg `AV_PIX_FMT_YA16LE` byte layout on every host. We want a
  // host-native u16 whose underlying bytes spell `[low, high]` (LE):
  // `u16::from_ne_bytes(x.to_le_bytes())` is `x` on LE and `x.swap_bytes()`
  // on BE — the right value to store in either case.
  let le_encoded = |x: u16| -> u16 { u16::from_ne_bytes(x.to_le_bytes()) };
  let packed: std::vec::Vec<u16> = samples
    .iter()
    .flat_map(|&(y, a)| [le_encoded(y), le_encoded(a)])
    .collect();
  let frame = Ya16Frame::new(&packed, w, h, w * 2);

  // Run combined (with_rgb + with_rgba) — exercises Strategy A+ with the
  // newly endian-aware `copy_alpha_ya_u16_to_u8::<false>`. Forces
  // `with_simd(false)` so the test runs purely scalar — no SIMD intrinsics
  // — which lets it execute under `cargo miri test`. BE CI is driven by
  // miri on s390x / powerpc64; gating it out of miri would skip exactly
  // the host where BE corruption would surface.
  let mut rgb_combined = std::vec![0u8; (w * h * 3) as usize];
  let mut rgba_combined = std::vec![0u8; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::source::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgb(&mut rgb_combined)
      .unwrap()
      .with_rgba(&mut rgba_combined)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  // Run standalone (with_rgba only) — exercises the endian-aware
  // `ya16_to_rgba_row::<false>` kernel. Same scalar-only rationale.
  let mut rgba_standalone = std::vec![0u8; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::source::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgba(&mut rgba_standalone)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combined, rgba_standalone,
    "combined (with_rgb+with_rgba) RGBA must equal standalone with_rgba"
  );
}

/// u16 RGBA variant of the combined-vs-standalone parity check. Locks
/// down `copy_alpha_ya_u16::<false>` (the u16 alpha-patch helper for
/// 16-bit RGBA outputs).
#[test]
fn ya16_combined_rgb_u16_and_rgba_u16_alpha_matches_standalone_le_encoded() {
  let w: u32 = 8;
  let h: u32 = 1;
  let samples: [(u16, u16); 8] = [
    (0x0000, 0xFFFF),
    (0x8000, 0x4000),
    (0xFFFF, 0x0000),
    (0x1234, 0xABCD),
    (0x00FF, 0xFF00),
    (0x5A5A, 0xA5A5),
    (0x7FFF, 0x8000),
    (0xC000, 0x3FFF),
  ];
  // See sibling test for the `le_encoded` rationale.
  let le_encoded = |x: u16| -> u16 { u16::from_ne_bytes(x.to_le_bytes()) };
  let packed: std::vec::Vec<u16> = samples
    .iter()
    .flat_map(|&(y, a)| [le_encoded(y), le_encoded(a)])
    .collect();
  let frame = Ya16Frame::new(&packed, w, h, w * 2);

  // Forces `with_simd(false)` so this test runs purely scalar — no SIMD
  // intrinsics — which lets it execute under `cargo miri test`. BE CI is
  // driven by miri on s390x / powerpc64; gating it out of miri would skip
  // exactly the host where BE corruption would surface.
  let mut rgb_combined = std::vec![0u16; (w * h * 3) as usize];
  let mut rgba_combined = std::vec![0u16; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::source::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgb_u16(&mut rgb_combined)
      .unwrap()
      .with_rgba_u16(&mut rgba_combined)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut rgba_standalone = std::vec![0u16; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::source::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgba_u16(&mut rgba_standalone)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combined, rgba_standalone,
    "combined (with_rgb_u16+with_rgba_u16) RGBA u16 must equal standalone"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya16_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let intended: std::vec::Vec<u16> = (0..w)
      .flat_map(|i| [(i as u16) << 8, (255u16 - i as u16) << 8])
      .collect();
    let packed = as_le_u16(&intended);
    let frame = Ya16Frame::new(&packed, w as u32, 1, (w * 2) as u32);
    let mut rgba = std::vec![0u8; w * 4];
    let mut luma_u16 = std::vec![0u16; w];
    let mut sink = MixedSinker::<crate::source::Ya16>::new(w, 1)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
    // Pixel 0: Y=0, A=255<<8=0xFF00 → a8=0xFF
    assert_eq!(&rgba[0..4], &[0, 0, 0, 0xFF], "w={w} px0");
  }
}
// Phase 4 — Frame BE flag, Tier 11. LE+BE round-trip parity tests.
//
// Pattern (per format):
//   1. Build a host-native `intended` plane.
//   2. Encode the plane as LE bytes → `pix_le`. Build `*LeFrame` +
//      `MixedSinker<Marker<false>>`. Walk; collect output A.
//   3. Encode the same plane as BE bytes → `pix_be`. Build `*BeFrame` +
//      `MixedSinker<Marker<true>>`. Walk; collect output B.
//   4. Assert `A == B` byte-identical.
//
// Output A and B must be byte-identical because the kernel byte-swaps under
// the hood — the same logical samples should yield the same RGBA bytes
// regardless of input byte order.
use crate::{
  frame::{
    Gray9BeFrame, Gray9LeFrame, Gray10BeFrame, Gray10LeFrame, Gray12BeFrame, Gray12LeFrame,
    Gray14BeFrame, Gray14LeFrame, Gray16BeFrame, Gray16LeFrame, Grayf32BeFrame, Grayf32LeFrame,
    Ya16BeFrame, Ya16LeFrame,
  },
  source::{Gray9, Gray10, Gray12, Gray14, Gray16, Grayf32, Ya16},
};

/// Re-encode a host-native u16 slice as **BE-encoded** byte storage. Used to
/// build `*BeFrame` planes whose bytes are big-endian; the kernel swaps them
/// back to host-native via `from_be`.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Re-encode a host-native f32 slice as **BE-encoded** byte storage.
fn as_be_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

// LE/BE parity helper macros. Each format's intent + frame ctor + sink shape
// varies, so we factor the boilerplate (the `for use_simd in [false, true]`
// loop, walker invocations, and the divergence assert) and let each call
// site supply only what's unique. Mirrors the
// `planar3_be_roundtrip_test! / planar4_be_roundtrip_test! / pn_be_roundtrip_test!`
// shape used elsewhere.

/// Single-output u16 parity test (Gray9/10/12/14): one `with_rgba` sink and a
/// masked-u16 `intended` pattern.
macro_rules! gray_planar_u16_le_be_roundtrip_test {
  (
    label: $label:literal,
    marker: $marker:ident,
    le_frame: $le_frame:ident,
    be_frame: $be_frame:ident,
    walker_le: $walker_le:ident,
    walker_be: $walker_be:ident,
    intended: $intended:expr,
  ) => {{
    let intended: std::vec::Vec<u16> = $intended;
    let pix_le = as_le_u16(&intended);
    let pix_be = as_be_u16(&intended);

    // Cover both scalar (`with_simd(false)`) and SIMD (`with_simd(true)`)
    // dispatch paths so neither alone can mask a BE-parity divergence.
    // Under Miri, drop the SIMD iteration only — Miri can't
    // run the NEON/x86 intrinsics, but the scalar BE/LE path is exactly what
    // Miri exists to verify, so we keep that arm.
    #[cfg(miri)]
    let modes: &[bool] = &[false];
    #[cfg(not(miri))]
    let modes: &[bool] = &[false, true];
    for &use_simd in modes {
      let frame_le = $le_frame::try_new(&pix_le, 16, 4, 16).unwrap();
      let mut out_le = std::vec![0u8; 16 * 4 * 4];
      let mut sink_le = MixedSinker::<$marker>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_le)
        .unwrap();
      $walker_le(&frame_le, true, M, &mut sink_le).unwrap();

      let frame_be = $be_frame::try_new(&pix_be, 16, 4, 16).unwrap();
      let mut out_be = std::vec![0u8; 16 * 4 * 4];
      let mut sink_be = MixedSinker::<$marker<true>>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_be)
        .unwrap();
      $walker_be(&frame_be, true, M, &mut sink_be).unwrap();

      assert_eq!(
        out_le, out_be,
        "{} LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
    }
  }};
}

#[test]
fn gray9_le_be_roundtrip_byte_identical() {
  gray_planar_u16_le_be_roundtrip_test! {
    label: "Gray9",
    marker: Gray9,
    le_frame: Gray9LeFrame,
    be_frame: Gray9BeFrame,
    walker_le: gray9_to,
    walker_be: gray9_to_endian,
    intended: (0..16 * 4).map(|i| ((i * 7) as u16) & 0x01FF).collect(),
  }
}

#[test]
fn gray10_le_be_roundtrip_byte_identical() {
  gray_planar_u16_le_be_roundtrip_test! {
    label: "Gray10",
    marker: Gray10,
    le_frame: Gray10LeFrame,
    be_frame: Gray10BeFrame,
    walker_le: gray10_to,
    walker_be: gray10_to_endian,
    intended: (0..16 * 4).map(|i| ((i * 11) as u16) & 0x03FF).collect(),
  }
}

#[test]
fn gray12_le_be_roundtrip_byte_identical() {
  gray_planar_u16_le_be_roundtrip_test! {
    label: "Gray12",
    marker: Gray12,
    le_frame: Gray12LeFrame,
    be_frame: Gray12BeFrame,
    walker_le: gray12_to,
    walker_be: gray12_to_endian,
    intended: (0..16 * 4).map(|i| ((i * 17) as u16) & 0x0FFF).collect(),
  }
}

#[test]
fn gray14_le_be_roundtrip_byte_identical() {
  gray_planar_u16_le_be_roundtrip_test! {
    label: "Gray14",
    marker: Gray14,
    le_frame: Gray14LeFrame,
    be_frame: Gray14BeFrame,
    walker_le: gray14_to,
    walker_be: gray14_to_endian,
    intended: (0..16 * 4).map(|i| ((i * 23) as u16) & 0x3FFF).collect(),
  }
}

/// Two-output u16 parity test (Gray16): `with_rgba` + `with_luma_u16`.
macro_rules! gray16_dual_output_le_be_roundtrip_test {
  (
    label: $label:literal,
    marker: $marker:ident,
    le_frame: $le_frame:ident,
    be_frame: $be_frame:ident,
    walker_le: $walker_le:ident,
    walker_be: $walker_be:ident,
    intended: $intended:expr,
  ) => {{
    let intended: std::vec::Vec<u16> = $intended;
    let pix_le = as_le_u16(&intended);
    let pix_be = as_be_u16(&intended);

    // Drop SIMD iteration under Miri (NEON/x86 intrinsics unsupported);
    // keep scalar arm so Miri still verifies the BE/LE byte path.
    #[cfg(miri)]
    let modes: &[bool] = &[false];
    #[cfg(not(miri))]
    let modes: &[bool] = &[false, true];
    for &use_simd in modes {
      // Exercise both u8 RGBA and u16 luma paths.
      let frame_le = $le_frame::try_new(&pix_le, 16, 4, 16).unwrap();
      let mut out_le_rgba = std::vec![0u8; 16 * 4 * 4];
      let mut out_le_luma_u16 = std::vec![0u16; 16 * 4];
      let mut sink_le = MixedSinker::<$marker>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_le_rgba)
        .unwrap()
        .with_luma_u16(&mut out_le_luma_u16)
        .unwrap();
      $walker_le(&frame_le, true, M, &mut sink_le).unwrap();

      let frame_be = $be_frame::try_new(&pix_be, 16, 4, 16).unwrap();
      let mut out_be_rgba = std::vec![0u8; 16 * 4 * 4];
      let mut out_be_luma_u16 = std::vec![0u16; 16 * 4];
      let mut sink_be = MixedSinker::<$marker<true>>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_be_rgba)
        .unwrap()
        .with_luma_u16(&mut out_be_luma_u16)
        .unwrap();
      $walker_be(&frame_be, true, M, &mut sink_be).unwrap();

      assert_eq!(
        out_le_rgba, out_be_rgba,
        "{} RGBA u8 LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
      assert_eq!(
        out_le_luma_u16, out_be_luma_u16,
        "{} luma u16 LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
    }
  }};
}

#[test]
fn gray16_le_be_roundtrip_byte_identical() {
  gray16_dual_output_le_be_roundtrip_test! {
    label: "Gray16",
    marker: Gray16,
    le_frame: Gray16LeFrame,
    be_frame: Gray16BeFrame,
    walker_le: gray16_to,
    walker_be: gray16_to_endian,
    intended: (0..16 * 4)
      .map(|i| match i % 4 {
        0 => 0x1234,
        1 => 0xABCD,
        2 => 0x00FF,
        _ => 0xFF00,
      })
      .collect(),
  }
}

/// Single-output f32 parity test (Grayf32): one `with_rgba` sink and an
/// f32-encoded plane.
macro_rules! grayf32_le_be_roundtrip_test {
  (
    label: $label:literal,
    marker: $marker:ident,
    le_frame: $le_frame:ident,
    be_frame: $be_frame:ident,
    walker_le: $walker_le:ident,
    walker_be: $walker_be:ident,
    intended: $intended:expr,
  ) => {{
    let intended: std::vec::Vec<f32> = $intended;
    let pix_le = as_le_f32(&intended);
    let pix_be = as_be_f32(&intended);

    // Drop SIMD iteration under Miri (NEON/x86 intrinsics unsupported);
    // keep scalar arm so Miri still verifies the BE/LE byte path.
    #[cfg(miri)]
    let modes: &[bool] = &[false];
    #[cfg(not(miri))]
    let modes: &[bool] = &[false, true];
    for &use_simd in modes {
      let frame_le = $le_frame::try_new(&pix_le, 16, 4, 16).unwrap();
      let mut out_le = std::vec![0u8; 16 * 4 * 4];
      let mut sink_le = MixedSinker::<$marker>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_le)
        .unwrap();
      $walker_le(&frame_le, true, M, &mut sink_le).unwrap();

      let frame_be = $be_frame::try_new(&pix_be, 16, 4, 16).unwrap();
      let mut out_be = std::vec![0u8; 16 * 4 * 4];
      let mut sink_be = MixedSinker::<$marker<true>>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_be)
        .unwrap();
      $walker_be(&frame_be, true, M, &mut sink_be).unwrap();

      assert_eq!(
        out_le, out_be,
        "{} LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
    }
  }};
}

#[test]
fn grayf32_le_be_roundtrip_byte_identical() {
  // Mix of values to surface any byte-swap regression. Includes typical
  // luma values (0.0, 0.25, 0.5, 0.75, 1.0) plus HDR (>1) and sub-zero.
  grayf32_le_be_roundtrip_test! {
    label: "Grayf32",
    marker: Grayf32,
    le_frame: Grayf32LeFrame,
    be_frame: Grayf32BeFrame,
    walker_le: grayf32_to,
    walker_be: grayf32_to_endian,
    intended: (0..16 * 4)
      .map(|i| match i % 5 {
        0 => 0.0f32,
        1 => 0.25,
        2 => 0.5,
        3 => 0.75,
        _ => 1.0,
      })
      .collect(),
  }
}

/// Two-output u16 parity test for the packed Y+A format (Ya16):
/// `with_rgba` + `with_rgba_u16`. Uses 2 u16 per pixel (Y, A), so input
/// + stride are double the planar variant.
macro_rules! ya16_le_be_roundtrip_test {
  (
    label: $label:literal,
    marker: $marker:ident,
    le_frame: $le_frame:ident,
    be_frame: $be_frame:ident,
    walker_le: $walker_le:ident,
    walker_be: $walker_be:ident,
    intended: $intended:expr,
  ) => {{
    let intended: std::vec::Vec<u16> = $intended;
    let pix_le = as_le_u16(&intended);
    let pix_be = as_be_u16(&intended);

    // Drop SIMD iteration under Miri (NEON/x86 intrinsics unsupported);
    // keep scalar arm so Miri still verifies the BE/LE byte path.
    #[cfg(miri)]
    let modes: &[bool] = &[false];
    #[cfg(not(miri))]
    let modes: &[bool] = &[false, true];
    for &use_simd in modes {
      // Exercise both u8 RGBA (with α copy) and u16 RGBA (with α copy) paths.
      let frame_le = $le_frame::try_new(&pix_le, 16, 4, 16 * 2).unwrap();
      let mut out_le_rgba = std::vec![0u8; 16 * 4 * 4];
      let mut out_le_rgba_u16 = std::vec![0u16; 16 * 4 * 4];
      let mut sink_le = MixedSinker::<$marker>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_le_rgba)
        .unwrap()
        .with_rgba_u16(&mut out_le_rgba_u16)
        .unwrap();
      $walker_le(&frame_le, true, M, &mut sink_le).unwrap();

      let frame_be = $be_frame::try_new(&pix_be, 16, 4, 16 * 2).unwrap();
      let mut out_be_rgba = std::vec![0u8; 16 * 4 * 4];
      let mut out_be_rgba_u16 = std::vec![0u16; 16 * 4 * 4];
      let mut sink_be = MixedSinker::<$marker<true>>::new(16, 4)
        .with_simd(use_simd)
        .with_rgba(&mut out_be_rgba)
        .unwrap()
        .with_rgba_u16(&mut out_be_rgba_u16)
        .unwrap();
      $walker_be(&frame_be, true, M, &mut sink_be).unwrap();

      assert_eq!(
        out_le_rgba, out_be_rgba,
        "{} RGBA u8 LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
      assert_eq!(
        out_le_rgba_u16, out_be_rgba_u16,
        "{} RGBA u16 LE/BE outputs diverge (simd={use_simd}) — `<const BE>` propagation broken",
        $label,
      );
    }
  }};
}

#[test]
fn ya16_le_be_roundtrip_byte_identical() {
  ya16_le_be_roundtrip_test! {
    label: "Ya16",
    marker: Ya16,
    le_frame: Ya16LeFrame,
    be_frame: Ya16BeFrame,
    walker_le: ya16_to,
    walker_be: ya16_to_endian,
    // 16 px x 4 rows x 2 u16 (Y,A) per pixel = 128 u16 elements.
    intended: (0..16 * 4 * 2)
      .map(|i| match i % 4 {
        0 => 0x1234, // Y
        1 => 0xABCD, // A
        2 => 0x00FF, // Y
        _ => 0xFF00, // A
      })
      .collect(),
  }
}

// ---- Gray32 (u32 luma; u8 = >> 24, native u16 = >> 16) ---------------------

#[test]
fn gray32_with_rgb_shifts_to_u8() {
  // raw >> 24 broadcast.
  let plane = as_le_u32(&[0x8000_0000, 0xFFFF_FFFF, 0x0000_0000, 0x0100_0000]);
  let frame = make_gray32_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0x80, 0x80, 0x80]);
  assert_eq!(rgb[3..6], [0xFF, 0xFF, 0xFF]);
  assert_eq!(rgb[6..9], [0x00, 0x00, 0x00]);
  assert_eq!(rgb[9..12], [0x01, 0x01, 0x01]);
}

#[test]
fn gray32_with_rgba_alpha_is_0xff() {
  let plane = as_le_u32(&[0xC000_0000u32; 4]);
  let frame = make_gray32_frame(&plane, 4, 1);
  let mut rgba = std::vec![0u8; 16];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(
      rgba[i * 4..i * 4 + 4],
      [0xC0, 0xC0, 0xC0, 0xFF],
      "pixel {i}"
    );
  }
}

#[test]
fn gray32_with_rgb_u16_shifts_to_native() {
  // raw >> 16 broadcast.
  let plane = as_le_u32(&[0x8000_0000, 0x1234_5678, 0xFFFF_FFFF, 0x0000_0000]);
  let frame = make_gray32_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u16; 12];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0x8000, 0x8000, 0x8000]);
  assert_eq!(rgb[3..6], [0x1234, 0x1234, 0x1234]);
  assert_eq!(rgb[6..9], [0xFFFF, 0xFFFF, 0xFFFF]);
  assert_eq!(rgb[9..12], [0x0000, 0x0000, 0x0000]);
}

#[test]
fn gray32_with_rgba_u16_alpha_is_0xffff() {
  let plane = as_le_u32(&[0x1234_5678u32; 4]);
  let frame = make_gray32_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
    assert_eq!(rgba_u16[i * 4], 0x1234, "pixel {i} R");
  }
}

#[test]
fn gray32_with_luma_shifts_to_u8() {
  let plane = as_le_u32(&[0, 0x4000_0000, 0x8000_0000, 0xFFFF_FFFF]);
  let frame = make_gray32_frame(&plane, 4, 1);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(luma, [0x00, 0x40, 0x80, 0xFF]);
}

#[test]
fn gray32_with_luma_u16_shifts_to_native() {
  let intended: Vec<u32> = (0u32..16).map(|x| x << 24 | x << 12).collect();
  let plane = as_le_u32(&intended);
  let frame = make_gray32_frame(&plane, 4, 4);
  let mut lu16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(4, 4)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  let expect: Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  assert_eq!(lu16, expect);
}

#[test]
fn gray32_with_hsv_h0_s0_v_shift24() {
  let plane = as_le_u32(&[0x8000_0000u32, 0x4000_0000u32]);
  let frame = make_gray32_frame(&plane, 2, 1);
  let mut h = std::vec![0xFFu8; 2];
  let mut s = std::vec![0xFFu8; 2];
  let mut v = std::vec![0u8; 2];
  let mut sink = MixedSinker::<crate::source::Gray32>::new(2, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0], "H must be 0");
  assert_eq!(s, [0, 0], "S must be 0");
  assert_eq!(v, [0x80, 0x40], "V = Y >> 24");
}

#[test]
fn gray32_be_endian_matches_le() {
  // BE-encoded bytes decoded by the `<BE = true>` kernel must equal the same
  // logical values decoded by the LE kernel.
  let intended = [0x8012_3456u32, 0x00FF_8800, 0xDEAD_BEEF, 0x1357_9BDF];
  let le_plane = as_le_u32(&intended);
  let be_plane = as_be_u32(&intended);
  let le_frame = make_gray32_frame(&le_plane, 4, 1);
  let be_frame = Gray32Frame::<true>::new(&be_plane, 4, 1, 4);

  let mut rgb_le = std::vec![0u8; 12];
  let mut sink_le = MixedSinker::<crate::source::Gray32>::new(4, 1)
    .with_rgb(&mut rgb_le)
    .unwrap();
  gray32_to(&le_frame, FR, M, &mut sink_le).unwrap();

  let mut rgb_be = std::vec![0u8; 12];
  let mut sink_be = MixedSinker::<crate::source::Gray32<true>>::new(4, 1)
    .with_rgb(&mut rgb_be)
    .unwrap();
  gray32_to_endian::<_, true>(&be_frame, FR, M, &mut sink_be).unwrap();

  assert_eq!(rgb_le, rgb_be, "BE and LE Gray32 RGB must match");
}

// ---- Atomicity (#308) -------------------------------------------------------
//
// The gray identity-path `process` impls that route the `want_hsv && want_rgba
// && !want_rgb` combo through the RGB row scratch (Gray8, Ya8, GrayN, Ya16 —
// the narrow `want_hsv && !want_rgb && !want_rgba` HSV intercept lets that one
// shape fall through, unlike the broad-intercept Gray16/Gray32/Grayf* which
// never reach the scratch's allocating arm) must run the RGB-scratch preflight
// BEFORE any output row (luma included) is written, so an allocator refusal
// returns a recoverable `AllocationFailed` leaving the output frame untouched
// rather than partially mutated. Each test attaches `luma + RGBA + HSV` (no
// RGB) — the lone identity shape that reaches `rgb_row_buf_or_scratch`'s scratch
// arm — and proves the source-copied luma plane stays at its sentinel. Reuses
// the crate's RGB-scratch failpoint (`yuva`-gated, so these tests are too).

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let (w, h) = (16usize, 8usize);
  let n = w * h;
  let plane = std::vec![0x12u8; n];
  let frame = make_gray8_frame(&plane, w as u32, h as u32);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Gray8>::new(w, h)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = gray8_to(&frame, FR, M, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma (Gray8: Y copied before the scratch) must be untouched on the alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya8_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let (w, h) = (16usize, 8usize);
  let n = w * h;
  let packed: std::vec::Vec<u8> = (0..n).flat_map(|_| [0x12u8, 0x80u8]).collect();
  let frame = Ya8Frame::new(&packed, w as u32, h as u32, (w * 2) as u32);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Ya8>::new(w, h)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = ya8_to(&frame, FR, M, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma (Ya8: native Y written before the scratch) must be untouched on the alloc-failure path"
  );
}

// GrayN (high-bit gray, u16 plane) shares the narrow-intercept structure of the
// 8-bit gray paths, so it has the identical latent bug despite being >8-bit.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray_n_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let (w, h) = (16usize, 8usize);
  let n = w * h;
  let plane = std::vec![0x0012u16; n];
  let frame: GrayNFrame<'_, 10> = GrayNFrame::new(&plane, w as u32, h as u32, w as u32);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Gray10>::new(w, h)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = gray10_to(&frame, FR, M, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma (GrayN: raw Y written before the scratch) must be untouched on the alloc-failure path"
  );
}

// Ya16 (16-bit gray+alpha) shares the narrow-intercept structure too.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya16_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use super::super::MixedSinkerError;
  use crate::resample::ResampleError;

  let (w, h) = (16usize, 8usize);
  let n = w * h;
  let host: std::vec::Vec<u16> = (0..n).flat_map(|_| [0x1234u16, 0x8000u16]).collect();
  let packed = as_le_u16(&host);
  let frame = Ya16Frame::new(&packed, w as u32, h as u32, (w * 2) as u32);
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<crate::source::Ya16>::new(w, h)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = ya16_to(&frame, FR, M, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma (Ya16: Y>>8 written before the scratch) must be untouched on the alloc-failure path"
  );
}
