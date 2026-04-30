use super::super::{Y2xxFrame, Y2xxFrameError, Y210Frame, Y212Frame};

#[test]
fn y210_frame_try_new_accepts_valid_tight() {
  // Width 4, height 2, stride = 4 * 2 = 8 u16 elements per row.
  let buf = std::vec![0u16; 8 * 2];
  let frame = Y210Frame::try_new(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 8);
}

#[test]
fn y210_frame_try_new_accepts_oversized_stride() {
  // Padded-row case: caller may supply a larger stride.
  let buf = std::vec![0u16; 16 * 4];
  Y210Frame::try_new(&buf, 4, 4, 16).unwrap();
}

#[test]
fn y210_frame_try_new_rejects_zero_dimension() {
  let buf: [u16; 0] = [];
  // Frame structs don't derive `PartialEq` (matching V210Frame), so
  // we extract the error before comparing.
  let err = Y210Frame::try_new(&buf, 0, 1, 0).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 0,
      height: 1
    }
  );
  let err = Y210Frame::try_new(&buf, 4, 0, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 4,
      height: 0
    }
  );
}

#[test]
fn y210_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u16; 64];
  for w in [1u32, 3, 5, 7, 9, 11, 13] {
    let stride = (w as usize) * 2;
    let err = Y210Frame::try_new(&buf, w, 1, stride as u32).unwrap_err();
    assert_eq!(err, Y2xxFrameError::OddWidth { width: w });
  }
  // 2, 4, 6, 8 must succeed.
  for w in [2u32, 4, 6, 8] {
    let stride = w * 2;
    let buf = std::vec![0u16; stride as usize];
    Y210Frame::try_new(&buf, w, 1, stride).unwrap();
  }
}

#[test]
fn y210_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u16; 16];
  // For width=4, min_stride = 8 u16 elements.
  let err = Y210Frame::try_new(&buf, 4, 1, 7).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::StrideTooSmall {
      min_stride: 8,
      stride: 7
    }
  );
}

#[test]
fn y210_frame_try_new_rejects_short_plane() {
  let buf = std::vec![0u16; 7]; // need 8 for width=4 height=1
  let err = Y210Frame::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::PlaneTooShort {
      expected: 8,
      actual: 7
    }
  );
}

#[test]
fn y210_frame_accessors_round_trip() {
  let buf = std::vec![0u16; 16 * 4];
  let frame = Y210Frame::try_new(&buf, 8, 4, 16).unwrap();
  assert_eq!(frame.packed().len(), 16 * 4);
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 4);
  assert_eq!(frame.stride(), 16);
}

#[test]
fn y2xx_frame_try_new_rejects_unsupported_bits() {
  // BITS must be {10, 12, 16}. The compile-time-asserted dimensions
  // 8 are valid but BITS=11 is not.
  let buf = std::vec![0u16; 16];
  let err = Y2xxFrame::<11>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 11 });
  let err = Y2xxFrame::<8>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 8 });
  // 14 is NOT in the supported set for Y2xx (no FFmpeg y214 format exists).
  let err = Y2xxFrame::<14>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 14 });
}

#[test]
fn y210_frame_try_new_checked_rejects_low_bit_violations() {
  // Y210 = MSB-aligned 10-bit; low 6 bits must be zero.
  let mut buf = std::vec![0u16; 8]; // width=4, height=1
  buf[0] = 0xFFC0; // valid: 10-bit value 0x3FF in high 10
  buf[1] = 0xFFC1; // INVALID: low 6 bits = 0x01 (non-zero)
  let err = Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::SampleLowBitsSet);
}

#[test]
fn y210_frame_try_new_checked_accepts_valid_msb_aligned_data() {
  // All samples have low 6 bits == 0.
  let buf: std::vec::Vec<u16> = (0..8).map(|i| ((i as u16) << 6) & 0xFFC0).collect();
  Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap();
}

#[test]
#[should_panic(expected = "invalid Y2xxFrame:")]
fn y210_frame_new_panics_on_invalid() {
  let buf: [u16; 0] = [];
  let _ = Y210Frame::new(&buf, 0, 0, 0);
}

#[test]
fn y210_frame_try_new_checked_ignores_stride_padding_bytes() {
  // Width=4 → row_elems = 8 u16; stride = 12 u16 (4 u16 padding per row).
  // All declared-payload samples have low 6 bits == 0 (valid 10-bit MSB-aligned).
  // Padding samples have arbitrary low bits set — must not trigger
  // SampleLowBitsSet (matches PnFrame::try_new_checked behavior).
  let mut buf = std::vec![0u16; 12 * 2]; // height=2
  for row in 0..2 {
    // Declared payload (first 8 u16 of each row) — clean MSB-aligned.
    for i in 0..8 {
      buf[row * 12 + i] = ((i as u16) << 6) & 0xFFC0;
    }
    // Stride padding (last 4 u16 of each row) — arbitrary low bits.
    for i in 8..12 {
      buf[row * 12 + i] = 0xFFFF; // every bit set, including low 6
    }
  }
  // try_new_checked must accept this — it scans only the declared payload.
  Y210Frame::try_new_checked(&buf, 4, 2, 12).unwrap();
}

#[test]
fn y212_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u16; 8 * 2];
  let frame = Y212Frame::try_new(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
}

#[test]
fn y212_frame_try_new_checked_rejects_low_bit_violations() {
  // Y212 = MSB-aligned 12-bit; low 4 bits must be zero.
  let mut buf = std::vec![0u16; 8]; // width=4, height=1
  buf[0] = 0xFFF0; // valid: 12-bit value 0xFFF in high 12, low 4 = 0
  buf[1] = 0xFFF1; // INVALID: low 4 bits = 0x1
  let err = Y212Frame::try_new_checked(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::SampleLowBitsSet);
}
