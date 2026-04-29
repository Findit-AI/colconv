use super::*;

// ---- Yuv444pFrame16::try_new_checked ---------------------------------

fn p444_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 4:4:4: chroma is FULL-width, full-height (1:1 with Y).
  let y = std::vec![64u16; 16 * 8];
  let u = std::vec![512u16; 16 * 8];
  let v = std::vec![512u16; 16 * 8];
  (y, u, v)
}

#[test]
fn yuv444p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p444_planes_10bit();
  let f = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv444p10_try_new_checked_accepts_max_valid_value() {
  let y = std::vec![1023u16; 16 * 8];
  let u = std::vec![1023u16; 16 * 8];
  let v = std::vec![1023u16; 16 * 8];
  Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("max valid passes");
}

#[test]
fn yuv444p10_try_new_checked_rejects_y_high_bit_set() {
  let mut y = std::vec![0u16; 16 * 8];
  y[2 * 16 + 9] = 0x8000;
  let u = std::vec![512u16; 16 * 8];
  let v = std::vec![512u16; 16 * 8];
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::Y,
      value: 0x8000,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv444p10_try_new_checked_rejects_u_plane_sample_in_full_width_chroma() {
  // 4:4:4-specific: the offending sample is in the FULL-WIDTH
  // chroma plane, at column 13 (which doesn't exist in 4:2:0/4:2:2
  // half-width chroma). Forces the scan to extend across the full
  // chroma width.
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 16 * 8];
  u[3 * 16 + 13] = 1024;
  let v = std::vec![512u16; 16 * 8];
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
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
fn yuv444p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 16 * 8];
  let mut v = std::vec![512u16; 16 * 8];
  v[7 * 16 + 15] = 0xFFFF; // last chroma sample
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv444p14_try_new_checked_rejects_above_16383() {
  let mut y = std::vec![8192u16; 16 * 8];
  y[42] = 16384; // just above 14-bit max
  let u = std::vec![8192u16; 16 * 8];
  let v = std::vec![8192u16; 16 * 8];
  let e = Yuv444p14Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 16384,
      max_valid: 16383,
      ..
    }
  ));
}

#[test]
fn yuv444p16_try_new_checked_accepts_full_u16_range() {
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 16 * 8];
  let v = std::vec![32768u16; 16 * 8];
  Yuv444p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16)
    .expect("every u16 value is in range at 16 bits");
}

// ---- Yuv440p10/12 checked-constructor tests ---------------------------

fn p440_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 4:4:0: chroma is FULL-width × HALF-height (8 / 2 = 4 chroma rows).
  let y = std::vec![64u16; 16 * 8];
  let u = std::vec![512u16; 16 * 4];
  let v = std::vec![512u16; 16 * 4];
  (y, u, v)
}

#[test]
fn yuv440p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p440_planes_10bit();
  let f = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv440p10_try_new_checked_rejects_y_high_bit_set() {
  let mut y = std::vec![0u16; 16 * 8];
  y[2 * 16 + 9] = 0x8000;
  let u = std::vec![512u16; 16 * 4];
  let v = std::vec![512u16; 16 * 4];
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::Y,
      value: 0x8000,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv440p10_try_new_checked_rejects_u_plane_sample_in_full_width_chroma() {
  // 4:4:0-specific: chroma is full-width × half-height. Plant the
  // bad sample at column 13 (would be out of range for half-width
  // 4:2:0/4:2:2 chroma) on the last chroma row (index 3 for height
  // 8 ⇒ 4 chroma rows).
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 16 * 4];
  u[3 * 16 + 13] = 1024;
  let v = std::vec![512u16; 16 * 4];
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
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
fn yuv440p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 16 * 4];
  let mut v = std::vec![512u16; 16 * 4];
  v[3 * 16 + 15] = 0xFFFF; // last chroma sample of the last chroma row
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv440p12_try_new_checked_rejects_above_4095() {
  let mut y = std::vec![2048u16; 16 * 8];
  y[42] = 4096; // just above 12-bit max
  let u = std::vec![2048u16; 16 * 4];
  let v = std::vec![2048u16; 16 * 4];
  let e = Yuv440p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

