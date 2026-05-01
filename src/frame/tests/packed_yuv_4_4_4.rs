use crate::frame::{V30XFrame, V30XFrameError, V410Frame, V410FrameError};

const fn zero_buf<const N: usize>() -> [u32; N] {
  [0u32; N]
}

#[test]
fn v410_frame_try_new_accepts_valid_tight() {
  // Tight stride: stride == width.
  let buf = zero_buf::<16>();
  let f = V410Frame::try_new(&buf, 4, 4, 4).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed().len(), 16);
}

#[test]
fn v410_frame_try_new_accepts_oversized_stride() {
  let buf = zero_buf::<32>();
  V410Frame::try_new(&buf, 4, 4, 8).unwrap();
}

#[test]
fn v410_frame_try_new_rejects_zero_dimension() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V410Frame::try_new(&buf, 0, 4, 4),
    Err(V410FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    V410Frame::try_new(&buf, 4, 0, 4),
    Err(V410FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn v410_frame_try_new_rejects_stride_too_small() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V410Frame::try_new(&buf, 4, 4, 3),
    Err(V410FrameError::StrideTooSmall {
      min_stride: 4,
      stride: 3
    })
  ));
}

#[test]
fn v410_frame_try_new_rejects_short_plane() {
  let buf = zero_buf::<8>();
  assert!(matches!(
    V410Frame::try_new(&buf, 4, 4, 4),
    Err(V410FrameError::PlaneTooShort {
      expected: 16,
      actual: 8
    })
  ));
}

#[test]
fn v410_frame_accessors_round_trip() {
  let buf = zero_buf::<32>();
  let f = V410Frame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.packed().len(), 32);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
}

#[test]
#[should_panic(expected = "invalid V410Frame:")]
fn v410_frame_new_panics_on_invalid() {
  let buf = zero_buf::<8>();
  let _ = V410Frame::new(&buf, 4, 4, 4); // PlaneTooShort
}

#[test]
fn v30x_frame_try_new_accepts_valid_tight() {
  // Tight stride: stride == width.
  let buf = zero_buf::<16>();
  let f = V30XFrame::try_new(&buf, 4, 4, 4).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed().len(), 16);
}

#[test]
fn v30x_frame_try_new_accepts_oversized_stride() {
  let buf = zero_buf::<32>();
  V30XFrame::try_new(&buf, 4, 4, 8).unwrap();
}

#[test]
fn v30x_frame_try_new_rejects_zero_dimension() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 0, 4, 4),
    Err(V30XFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 0, 4),
    Err(V30XFrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn v30x_frame_try_new_rejects_stride_too_small() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 4, 3),
    Err(V30XFrameError::StrideTooSmall {
      min_stride: 4,
      stride: 3
    })
  ));
}

#[test]
fn v30x_frame_try_new_rejects_short_plane() {
  let buf = zero_buf::<8>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 4, 4),
    Err(V30XFrameError::PlaneTooShort {
      expected: 16,
      actual: 8
    })
  ));
}

#[test]
fn v30x_frame_accessors_round_trip() {
  let buf = zero_buf::<32>();
  let f = V30XFrame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.packed().len(), 32);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
}

#[test]
#[should_panic(expected = "invalid V30XFrame:")]
fn v30x_frame_new_panics_on_invalid() {
  let buf = zero_buf::<8>();
  let _ = V30XFrame::new(&buf, 4, 4, 4); // PlaneTooShort
}
