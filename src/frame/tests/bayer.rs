use super::*;

// ----- BayerFrame (8-bit) -----

#[test]
fn bayer_try_new_accepts_valid_tight() {
  let data = std::vec![0u8; 16 * 8];
  let f = BayerFrame::try_new(&data, 16, 8, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.stride(), 16);
}

#[test]
fn bayer_try_new_accepts_padded_stride() {
  let data = std::vec![0u8; 24 * 8];
  let f = BayerFrame::try_new(&data, 16, 8, 24).expect("padded stride valid");
  assert_eq!(f.stride(), 24);
}

#[test]
fn bayer_try_new_rejects_zero_dim() {
  let data = std::vec![0u8; 16 * 8];
  let e = BayerFrame::try_new(&data, 0, 8, 16).unwrap_err();
  assert!(matches!(e, BayerFrameError::ZeroDimension { .. }));
  let e = BayerFrame::try_new(&data, 16, 0, 16).unwrap_err();
  assert!(matches!(e, BayerFrameError::ZeroDimension { .. }));
}

#[test]
fn bayer_try_new_accepts_odd_width() {
  // Cropped Bayer planes can have odd dimensions; the kernel
  // handles partial 2×2 tiles via edge clamping.
  let data = std::vec![0u8; 15 * 8];
  let f = BayerFrame::try_new(&data, 15, 8, 15).expect("odd width valid");
  assert_eq!(f.width(), 15);
}

#[test]
fn bayer_try_new_accepts_odd_height() {
  let data = std::vec![0u8; 16 * 7];
  let f = BayerFrame::try_new(&data, 16, 7, 16).expect("odd height valid");
  assert_eq!(f.height(), 7);
}

#[test]
fn bayer_try_new_accepts_odd_width_and_height() {
  let data = std::vec![0u8; 15 * 7];
  let f = BayerFrame::try_new(&data, 15, 7, 15).expect("odd both valid");
  assert_eq!(f.width(), 15);
  assert_eq!(f.height(), 7);
}

#[test]
fn bayer_try_new_accepts_1x1() {
  let data = std::vec![42u8];
  let f = BayerFrame::try_new(&data, 1, 1, 1).expect("1x1 valid");
  assert_eq!(f.width(), 1);
  assert_eq!(f.height(), 1);
}

#[test]
fn bayer_try_new_rejects_stride_under_width() {
  let data = std::vec![0u8; 16 * 8];
  let e = BayerFrame::try_new(&data, 16, 8, 8).unwrap_err();
  assert!(matches!(e, BayerFrameError::StrideTooSmall { .. }));
}

#[test]
fn bayer_try_new_rejects_short_plane() {
  let data = std::vec![0u8; 10];
  let e = BayerFrame::try_new(&data, 16, 8, 16).unwrap_err();
  assert!(matches!(e, BayerFrameError::PlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid BayerFrame")]
fn bayer_new_panics_on_invalid() {
  let data = std::vec![0u8; 10];
  let _ = BayerFrame::new(&data, 16, 8, 16);
}

// ----- BayerFrame16 (high-bit-depth) -----

#[test]
fn bayer16_try_new_rejects_unsupported_bits() {
  let data = std::vec![0u16; 16 * 8];
  let e = BayerFrame16::<11>::try_new(&data, 16, 8, 16).unwrap_err();
  assert!(matches!(e, BayerFrame16Error::UnsupportedBits { bits: 11 }));
  let e = BayerFrame16::<8>::try_new(&data, 16, 8, 16).unwrap_err();
  assert!(matches!(e, BayerFrame16Error::UnsupportedBits { bits: 8 }));
}

#[test]
fn bayer16_try_new_accepts_each_supported_bits() {
  let data = std::vec![0u16; 16 * 8];
  Bayer10Frame::try_new(&data, 16, 8, 16).expect("10");
  Bayer12Frame::try_new(&data, 16, 8, 16).expect("12");
  Bayer14Frame::try_new(&data, 16, 8, 16).expect("14");
  Bayer16Frame::try_new(&data, 16, 8, 16).expect("16");
}

#[test]
fn bayer16_try_new_accepts_odd_dims() {
  let data = std::vec![0u16; 15 * 7];
  let f = Bayer12Frame::try_new(&data, 15, 7, 15).expect("odd both valid");
  assert_eq!(f.width(), 15);
  assert_eq!(f.height(), 7);
}

#[test]
fn bayer16_try_new_accepts_low_packed_12bit() {
  // 12-bit low-packed: every value ≤ 4095 is valid.
  let mut data = std::vec![2048u16; 16 * 8];
  data[7] = 4095; // max valid 12-bit
  data[42] = 0; // black
  Bayer12Frame::try_new(&data, 16, 8, 16).expect("12-bit low-packed");
}

#[test]
fn bayer16_try_new_rejects_above_max_at_12bit() {
  let mut data = std::vec![2048u16; 16 * 8];
  data[42] = 4096; // just above 12-bit max
  let e = Bayer12Frame::try_new(&data, 16, 8, 16).unwrap_err();
  assert!(matches!(
    e,
    BayerFrame16Error::SampleOutOfRange {
      index: 42,
      value: 4096,
      max_valid: 4095,
    }
  ));
}

#[test]
fn bayer16_try_new_rejects_above_max_at_10bit() {
  let mut data = std::vec![512u16; 16 * 8];
  data[3] = 1024; // just above 10-bit max
  let e = Bayer10Frame::try_new(&data, 16, 8, 16).unwrap_err();
  assert!(matches!(
    e,
    BayerFrame16Error::SampleOutOfRange {
      index: 3,
      value: 1024,
      max_valid: 1023,
    }
  ));
}

#[test]
fn bayer16_try_new_accepts_full_u16_range_at_16bit() {
  // At BITS=16 every u16 is valid.
  let mut data = std::vec![0u16; 16 * 8];
  data[7] = 0xFFFF;
  data[42] = 0x1234;
  Bayer16Frame::try_new(&data, 16, 8, 16).expect("any u16 valid at 16-bit");
}

#[test]
#[should_panic(expected = "invalid BayerFrame16")]
fn bayer16_new_panics_on_invalid() {
  let data = std::vec![0u16; 10];
  let _ = Bayer12Frame::new(&data, 16, 8, 16);
}

