use super::super::{V210Frame, V210FrameError};

#[test]
fn v210_frame_try_new_accepts_valid_tight() {
  // Width 6, height 2, stride = 6 * 8 / 3 = 16 bytes per row.
  let buf = std::vec![0u8; 16 * 2];
  let frame = V210Frame::try_new(&buf, 6, 2, 16).unwrap();
  assert_eq!(frame.width(), 6);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 16);
}

#[test]
fn v210_frame_try_new_accepts_oversized_stride() {
  // FFmpeg pads v210 rows to a multiple of 128 bytes; we accept that.
  let buf = std::vec![0u8; 128 * 4];
  V210Frame::try_new(&buf, 6, 4, 128).unwrap();
}

#[test]
fn v210_frame_try_new_rejects_zero_dimension() {
  let buf = [];
  assert!(matches!(
    V210Frame::try_new(&buf, 0, 1, 0),
    Err(V210FrameError::ZeroDimension {
      width: 0,
      height: 1
    })
  ));
  assert!(matches!(
    V210Frame::try_new(&buf, 6, 0, 16),
    Err(V210FrameError::ZeroDimension {
      width: 6,
      height: 0
    })
  ));
}

#[test]
fn v210_frame_try_new_rejects_width_not_multiple_of_6() {
  let buf = std::vec![0u8; 256];
  for w in [1u32, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13] {
    let stride = ((w as usize) * 8 / 3).div_ceil(16) * 16;
    let err = V210Frame::try_new(&buf, w, 1, stride as u32).unwrap_err();
    assert_eq!(err, V210FrameError::WidthNotMultipleOf6 { width: w });
  }
  // 6, 12, 18, 24 must succeed (when buffer is large enough).
  for w in [6u32, 12, 18, 24] {
    let stride = ((w as usize) * 8 / 3) as u32;
    let buf = std::vec![0u8; stride as usize];
    V210Frame::try_new(&buf, w, 1, stride).unwrap();
  }
}

#[test]
fn v210_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 32];
  // For width=6, min_stride = 16.
  assert!(matches!(
    V210Frame::try_new(&buf, 6, 1, 15),
    Err(V210FrameError::StrideTooSmall {
      min_stride: 16,
      stride: 15
    })
  ));
}

#[test]
fn v210_frame_try_new_rejects_short_plane() {
  let buf = std::vec![0u8; 15]; // need 16 for width=6 height=1
  assert!(matches!(
    V210Frame::try_new(&buf, 6, 1, 16),
    Err(V210FrameError::PlaneTooShort {
      expected: 16,
      actual: 15
    })
  ));
}

#[test]
fn v210_frame_accessors_round_trip() {
  let buf = std::vec![0u8; 32 * 4];
  let frame = V210Frame::try_new(&buf, 12, 4, 32).unwrap();
  assert_eq!(frame.v210().len(), 32 * 4);
  assert_eq!(frame.width(), 12);
  assert_eq!(frame.height(), 4);
  assert_eq!(frame.stride(), 32);
}

#[test]
#[should_panic(expected = "invalid V210Frame:")]
fn v210_frame_new_panics_on_invalid() {
  let buf = [];
  // `new` is the panicking convenience.
  let _ = V210Frame::new(&buf, 0, 0, 0);
}
