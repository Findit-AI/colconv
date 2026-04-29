use super::*;

// ---- Yuyv422Frame ------------------------------------------------------
//
// Single-plane 8-bit packed YUV 4:2:2. 4 bytes per 2-pixel block;
// `stride >= 2 * width`. Width must be even (4:2:2 chroma pair).

#[test]
fn yuyv422_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  Yuyv422Frame::try_new(&buf, 16, 4, 32).expect("valid");
}

#[test]
fn yuyv422_frame_try_new_accepts_oversized_stride() {
  // Padded rows are allowed.
  let buf = std::vec![0u8; 48 * 4];
  Yuyv422Frame::try_new(&buf, 16, 4, 48).expect("padded stride is valid");
}

#[test]
fn yuyv422_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Yuyv422Frame::try_new(&buf, 0, 4, 32),
    Err(Yuyv422FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Yuyv422Frame::try_new(&buf, 16, 0, 32),
    Err(Yuyv422FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn yuyv422_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u8; 17 * 4 * 2];
  assert!(matches!(
    Yuyv422Frame::try_new(&buf, 17, 4, 34),
    Err(Yuyv422FrameError::OddWidth { width: 17 })
  ));
}

#[test]
fn yuyv422_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Yuyv422Frame::try_new(&buf, 16, 4, 31),
    Err(Yuyv422FrameError::StrideTooSmall {
      min_stride: 32,
      stride: 31,
    })
  ));
}

#[test]
fn yuyv422_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 2];
  assert!(matches!(
    Yuyv422Frame::try_new(&small, 16, 4, 32),
    Err(Yuyv422FrameError::PlaneTooShort {
      expected: 128,
      actual: 32,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid Yuyv422Frame")]
fn yuyv422_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Yuyv422Frame::new(&buf, 16, 4, 32);
}

#[test]
fn yuyv422_frame_accessors_round_trip() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  let f = Yuyv422Frame::try_new(&buf, 16, 4, 32).unwrap();
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
  assert_eq!(f.yuyv().len(), 128);
}

// ---- Uyvy422Frame ------------------------------------------------------
//
// Mirrors Yuyv422Frame layout — only byte ordering inside each
// 4-byte block differs (UYVY vs YUYV). Re-tests catch parallel-
// implementation typos.

#[test]
fn uyvy422_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  Uyvy422Frame::try_new(&buf, 16, 4, 32).expect("valid");
}

#[test]
fn uyvy422_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Uyvy422Frame::try_new(&buf, 0, 4, 32),
    Err(Uyvy422FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn uyvy422_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u8; 17 * 4 * 2];
  assert!(matches!(
    Uyvy422Frame::try_new(&buf, 17, 4, 34),
    Err(Uyvy422FrameError::OddWidth { width: 17 })
  ));
}

#[test]
fn uyvy422_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Uyvy422Frame::try_new(&buf, 16, 4, 31),
    Err(Uyvy422FrameError::StrideTooSmall {
      min_stride: 32,
      stride: 31,
    })
  ));
}

#[test]
fn uyvy422_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 2];
  assert!(matches!(
    Uyvy422Frame::try_new(&small, 16, 4, 32),
    Err(Uyvy422FrameError::PlaneTooShort {
      expected: 128,
      actual: 32,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid Uyvy422Frame")]
fn uyvy422_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Uyvy422Frame::new(&buf, 16, 4, 32);
}

#[test]
fn uyvy422_frame_accessors_round_trip() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  let f = Uyvy422Frame::try_new(&buf, 16, 4, 32).unwrap();
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
  assert_eq!(f.uyvy().len(), 128);
}

// ---- Yvyu422Frame ------------------------------------------------------
//
// Same shape as Yuyv422Frame; differs only in UV byte order.

#[test]
fn yvyu422_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  Yvyu422Frame::try_new(&buf, 16, 4, 32).expect("valid");
}

#[test]
fn yvyu422_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Yvyu422Frame::try_new(&buf, 16, 0, 32),
    Err(Yvyu422FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn yvyu422_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u8; 17 * 4 * 2];
  assert!(matches!(
    Yvyu422Frame::try_new(&buf, 17, 4, 34),
    Err(Yvyu422FrameError::OddWidth { width: 17 })
  ));
}

#[test]
fn yvyu422_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  assert!(matches!(
    Yvyu422Frame::try_new(&buf, 16, 4, 31),
    Err(Yvyu422FrameError::StrideTooSmall {
      min_stride: 32,
      stride: 31,
    })
  ));
}

#[test]
fn yvyu422_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 2];
  assert!(matches!(
    Yvyu422Frame::try_new(&small, 16, 4, 32),
    Err(Yvyu422FrameError::PlaneTooShort {
      expected: 128,
      actual: 32,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid Yvyu422Frame")]
fn yvyu422_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Yvyu422Frame::new(&buf, 16, 4, 32);
}

#[test]
fn yvyu422_frame_accessors_round_trip() {
  let buf = std::vec![0u8; 16 * 4 * 2];
  let f = Yvyu422Frame::try_new(&buf, 16, 4, 32).unwrap();
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
  assert_eq!(f.yvyu().len(), 128);
}
