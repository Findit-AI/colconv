use super::*;

// ---- Yuv422pFrame16::try_new_checked ---------------------------------

fn p422_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // Width 16, height 8 — 4:2:2 chroma is half-width, FULL-height.
  let y = std::vec![64u16; 16 * 8];
  let u = std::vec![512u16; 8 * 8];
  let v = std::vec![512u16; 8 * 8];
  (y, u, v)
}

#[test]
fn yuv422p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p422_planes_10bit();
  let f = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv422p10_try_new_checked_accepts_max_valid_value() {
  // Exactly `(1 << 10) - 1 = 1023` must pass.
  let y = std::vec![1023u16; 16 * 8];
  let u = std::vec![1023u16; 8 * 8];
  let v = std::vec![1023u16; 8 * 8];
  Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("max valid passes");
}

#[test]
fn yuv422p10_try_new_checked_rejects_y_high_bit_set() {
  let mut y = std::vec![0u16; 16 * 8];
  y[3 * 16 + 5] = 0x8000;
  let u = std::vec![512u16; 8 * 8];
  let v = std::vec![512u16; 8 * 8];
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  match e {
    Yuv420pFrame16Error::SampleOutOfRange {
      plane,
      value,
      max_valid,
      ..
    } => {
      assert_eq!(plane, Yuv420pFrame16Plane::Y);
      assert_eq!(value, 0x8000);
      assert_eq!(max_valid, 1023);
    }
    other => panic!("expected SampleOutOfRange, got {other:?}"),
  }
}

#[test]
fn yuv422p10_try_new_checked_rejects_u_plane_sample_in_full_height_chroma() {
  // Crucial 4:2:2-specific test: the offending sample is on the
  // last chroma row (row 7), which only exists because 4:2:2
  // chroma is full-height (8 rows). The 4:2:0 scan would stop at
  // row 3.
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 8 * 8];
  u[7 * 8 + 3] = 1024; // last chroma row, just above max
  let v = std::vec![512u16; 8 * 8];
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::U,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv422p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 8 * 8];
  let mut v = std::vec![512u16; 8 * 8];
  v[5 * 8 + 6] = 0xFFFF;
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv422p12_try_new_checked_rejects_above_4095() {
  let mut y = std::vec![2048u16; 16 * 8];
  y[0] = 4096; // just above 12-bit max
  let u = std::vec![2048u16; 8 * 8];
  let v = std::vec![2048u16; 8 * 8];
  let e = Yuv422p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

#[test]
fn yuv422p16_try_new_checked_accepts_full_u16_range() {
  // At 16 bits the full u16 range is valid — no scan needed.
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 8 * 8];
  let v = std::vec![32768u16; 8 * 8];
  Yuv422p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8)
    .expect("every u16 value is in range at 16 bits");
}

