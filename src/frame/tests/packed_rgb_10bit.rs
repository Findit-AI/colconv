use super::*;

// ---- 10-bit packed RGB family (Ship 9e) ------------------------------

#[test]
fn x2rgb10_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  X2Rgb10Frame::try_new(&buf, 16, 4, 64).expect("valid");
}

#[test]
fn x2rgb10_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 4];
  assert!(matches!(
    X2Rgb10Frame::try_new(&small, 16, 4, 64),
    Err(X2Rgb10FrameError::PlaneTooShort { .. })
  ));
}

#[test]
fn x2rgb10_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    X2Rgb10Frame::try_new(&buf, 0, 4, 64),
    Err(X2Rgb10FrameError::ZeroDimension { .. })
  ));
}

#[test]
fn x2rgb10_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    X2Rgb10Frame::try_new(&buf, 16, 4, 63),
    Err(X2Rgb10FrameError::StrideTooSmall { .. })
  ));
}

#[test]
#[should_panic(expected = "invalid X2Rgb10Frame")]
fn x2rgb10_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = X2Rgb10Frame::new(&buf, 16, 4, 64);
}

#[test]
fn x2bgr10_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  X2Bgr10Frame::try_new(&buf, 16, 4, 64).expect("valid");
}

#[test]
fn x2bgr10_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 4];
  assert!(matches!(
    X2Bgr10Frame::try_new(&small, 16, 4, 64),
    Err(X2Bgr10FrameError::PlaneTooShort { .. })
  ));
}

#[test]
#[should_panic(expected = "invalid X2Bgr10Frame")]
fn x2bgr10_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = X2Bgr10Frame::new(&buf, 16, 4, 64);
}
