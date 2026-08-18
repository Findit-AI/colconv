//! Integration tests for Monoblack / Monowhite sources through MixedSinker.

use super::*;

#[test]
fn monoblack_walker_to_rgb() {
  let data = [0b11110000u8, 0b11110000u8];
  let frame = MonoblackFrame::try_new(&data, 8, 2, 1).expect("valid frame");

  let mut rgb = vec![0u8; 8 * 2 * 3];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 2)
      .with_rgb(&mut rgb)
      .expect("attach rgb");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(rgb[0], 255);
  assert_eq!(rgb[1], 255);
  assert_eq!(rgb[2], 255);
  assert_eq!(rgb[3 * 4], 0);
  assert_eq!(rgb[3 * 4 + 1], 0);
  assert_eq!(rgb[3 * 4 + 2], 0);
  let row2_base = 8 * 3;
  assert_eq!(rgb[row2_base], 255);
  assert_eq!(rgb[row2_base + 1], 255);
  assert_eq!(rgb[row2_base + 2], 255);
}

#[test]
fn monoblack_walker_to_rgba() {
  let data = [0xFFu8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgba = vec![0u8; 8 * 4];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_rgba(&mut rgba)
      .expect("attach rgba");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgba.chunks_exact(4) {
    assert_eq!(chunk[0], 255);
    assert_eq!(chunk[1], 255);
    assert_eq!(chunk[2], 255);
    assert_eq!(chunk[3], 0xFF);
  }
}

#[test]
fn monoblack_walker_to_luma() {
  let data = [0b10101010u8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut luma = vec![0u8; 8];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_luma(&mut luma)
      .expect("attach luma");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(luma, vec![255, 0, 255, 0, 255, 0, 255, 0]);
}

#[test]
fn monoblack_walker_to_luma_u16() {
  let data = [0x80u8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut luma_u16 = vec![0u16; 8];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_luma_u16(&mut luma_u16)
      .expect("attach luma_u16");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(luma_u16[0], 0x00FF);
  for val in &luma_u16[1..8] {
    assert_eq!(val, &0);
  }
}

#[test]
fn monoblack_walker_to_rgb_u16() {
  let data = [0xFFu8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgb_u16 = vec![0u16; 8 * 3];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_rgb_u16(&mut rgb_u16)
      .expect("attach rgb_u16");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgb_u16.chunks_exact(3) {
    assert_eq!(chunk[0], 0x00FF);
    assert_eq!(chunk[1], 0x00FF);
    assert_eq!(chunk[2], 0x00FF);
  }
}

#[test]
fn monoblack_walker_to_rgba_u16() {
  let data = [0x00u8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgba_u16 = vec![0u16; 8 * 4];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_rgba_u16(&mut rgba_u16)
      .expect("attach rgba_u16");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgba_u16.chunks_exact(4) {
    assert_eq!(chunk[0], 0);
    assert_eq!(chunk[1], 0);
    assert_eq!(chunk[2], 0);
    assert_eq!(chunk[3], 0x00FF);
  }
}

#[test]
fn monoblack_walker_to_hsv() {
  let data = [0b11110000u8];
  let frame = MonoblackFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut h = vec![0u8; 8];
  let mut s = vec![0u8; 8];
  let mut v = vec![0u8; 8];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 1)
      .with_hsv(&mut h, &mut s, &mut v)
      .expect("attach hsv");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(h, vec![0; 8]);
  assert_eq!(s, vec![0; 8]);
  assert_eq!(v, vec![255, 255, 255, 255, 0, 0, 0, 0]);
}

#[test]
fn monowhite_walker_to_luma() {
  let data = [0b10101010u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut luma = vec![0u8; 8];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_luma(&mut luma)
      .expect("attach luma");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(luma, vec![0, 255, 0, 255, 0, 255, 0, 255]);
}

#[test]
fn monowhite_walker_to_rgb() {
  let data = [0b11110000u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgb = vec![0u8; 8 * 3];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_rgb(&mut rgb)
      .expect("attach rgb");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgb[0..12].chunks_exact(3) {
    assert_eq!(chunk[0], 0);
    assert_eq!(chunk[1], 0);
    assert_eq!(chunk[2], 0);
  }
  for chunk in rgb[12..24].chunks_exact(3) {
    assert_eq!(chunk[0], 255);
    assert_eq!(chunk[1], 255);
    assert_eq!(chunk[2], 255);
  }
}

#[test]
fn monowhite_walker_to_rgba() {
  let data = [0x00u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgba = vec![0u8; 8 * 4];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_rgba(&mut rgba)
      .expect("attach rgba");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgba.chunks_exact(4) {
    assert_eq!(chunk[0], 255);
    assert_eq!(chunk[1], 255);
    assert_eq!(chunk[2], 255);
    assert_eq!(chunk[3], 0xFF);
  }
}

#[test]
fn monowhite_walker_to_luma_u16() {
  let data = [0xFFu8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut luma_u16 = vec![0u16; 8];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_luma_u16(&mut luma_u16)
      .expect("attach luma_u16");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for val in &luma_u16 {
    assert_eq!(val, &0);
  }
}

#[test]
fn monowhite_walker_to_rgb_u16() {
  let data = [0x80u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgb_u16 = vec![0u16; 8 * 3];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_rgb_u16(&mut rgb_u16)
      .expect("attach rgb_u16");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(rgb_u16[0], 0);
  assert_eq!(rgb_u16[1], 0);
  assert_eq!(rgb_u16[2], 0);
  for chunk in rgb_u16[3..].chunks_exact(3) {
    assert_eq!(chunk[0], 0x00FF);
    assert_eq!(chunk[1], 0x00FF);
    assert_eq!(chunk[2], 0x00FF);
  }
}

#[test]
fn monowhite_walker_to_rgba_u16() {
  let data = [0x00u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut rgba_u16 = vec![0u16; 8 * 4];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_rgba_u16(&mut rgba_u16)
      .expect("attach rgba_u16");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  for chunk in rgba_u16.chunks_exact(4) {
    assert_eq!(chunk[0], 0x00FF);
    assert_eq!(chunk[1], 0x00FF);
    assert_eq!(chunk[2], 0x00FF);
    assert_eq!(chunk[3], 0x00FF);
  }
}

#[test]
fn monowhite_walker_to_hsv() {
  let data = [0xFFu8];
  let frame = MonowhiteFrame::try_new(&data, 8, 1, 1).expect("valid frame");

  let mut h = vec![0u8; 8];
  let mut s = vec![0u8; 8];
  let mut v = vec![0u8; 8];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 1)
      .with_hsv(&mut h, &mut s, &mut v)
      .expect("attach hsv");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(h, vec![0; 8]);
  assert_eq!(s, vec![0; 8]);
  assert_eq!(v, vec![0; 8]);
}

#[test]
fn monoblack_multiple_rows() {
  let data = [0b10101010u8, 0b01010101u8, 0b11110000u8];
  let frame = MonoblackFrame::try_new(&data, 8, 3, 1).expect("valid frame");

  let mut luma = vec![0u8; 8 * 3];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 3)
      .with_luma(&mut luma)
      .expect("attach luma");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(&luma[0..8], vec![255, 0, 255, 0, 255, 0, 255, 0].as_slice());
  assert_eq!(
    &luma[8..16],
    vec![0, 255, 0, 255, 0, 255, 0, 255].as_slice()
  );
  assert_eq!(
    &luma[16..24],
    vec![255, 255, 255, 255, 0, 0, 0, 0].as_slice()
  );
}

#[test]
fn monoblack_partial_byte_width() {
  let data = [0b11000000u8, 0b10000000u8];
  let frame = MonoblackFrame::try_new(&data, 5, 2, 1).expect("valid frame");

  let mut luma = vec![0u8; 5 * 2];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(5, 2)
      .with_luma(&mut luma)
      .expect("attach luma");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(&luma[0..5], vec![255, 255, 0, 0, 0].as_slice());
  assert_eq!(&luma[5..10], vec![255, 0, 0, 0, 0].as_slice());
}

#[test]
fn monowhite_partial_byte_width() {
  let data = [0b11000000u8, 0b10000000u8];
  let frame = MonowhiteFrame::try_new(&data, 5, 2, 1).expect("valid frame");

  let mut luma = vec![0u8; 5 * 2];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(5, 2)
      .with_luma(&mut luma)
      .expect("attach luma");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(&luma[0..5], vec![0, 0, 255, 255, 255].as_slice());
  assert_eq!(&luma[5..10], vec![0, 255, 255, 255, 255].as_slice());
}

#[test]
fn monoblack_both_polarities_in_frame() {
  let data = [0xFFu8, 0x00u8];
  let frame = MonoblackFrame::try_new(&data, 8, 2, 1).expect("valid frame");

  let mut luma = vec![0u8; 8 * 2];
  {
    let mut sinker = MixedSinker::<Monoblack>::new(8, 2)
      .with_luma(&mut luma)
      .expect("attach luma");
    monoblack_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(&luma[0..8], vec![255; 8].as_slice());
  assert_eq!(&luma[8..16], vec![0; 8].as_slice());
}

#[test]
fn monowhite_both_polarities_in_frame() {
  let data = [0xFFu8, 0x00u8];
  let frame = MonowhiteFrame::try_new(&data, 8, 2, 1).expect("valid frame");

  let mut luma = vec![0u8; 8 * 2];
  {
    let mut sinker = MixedSinker::<Monowhite>::new(8, 2)
      .with_luma(&mut luma)
      .expect("attach luma");
    monowhite_to(&frame, true, KernelMatrix::Bt709, &mut sinker).expect("walk ok");
  }

  assert_eq!(&luma[0..8], vec![0; 8].as_slice());
  assert_eq!(&luma[8..16], vec![255; 8].as_slice());
}
