use super::*;

fn planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame, U/V are 8×4.
  (
    std::vec![0u8; 16 * 8],
    std::vec![128u8; 8 * 4],
    std::vec![128u8; 8 * 4],
  )
}

#[test]
fn try_new_accepts_valid_tight() {
  let (y, u, v) = planes();
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
}

#[test]
fn try_new_accepts_valid_padded_strides() {
  // 16×8 frame, strides padded (32 for y, 16 for u/v).
  let y = std::vec![0u8; 32 * 8];
  let u = std::vec![128u8; 16 * 4];
  let v = std::vec![128u8; 16 * 4];
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 32, 16, 16).expect("valid");
  assert_eq!(f.y_stride(), 32);
}

#[test]
fn try_new_rejects_zero_dim() {
  let (y, u, v) = planes();
  let e = Yuv420pFrame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::ZeroDimension { .. }));
}

#[test]
fn try_new_rejects_odd_width() {
  let (y, u, v) = planes();
  let e = Yuv420pFrame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::OddWidth { width: 15 }));
}

#[test]
fn try_new_accepts_odd_height() {
  // 16x9 frame — chroma_height = ceil(9/2) = 5. Y plane 16*9 = 144
  // bytes, U/V plane 8*5 = 40 bytes each. Valid 4:2:0 frame;
  // height=9 must not be rejected just because it's odd.
  let y = std::vec![0u8; 16 * 9];
  let u = std::vec![128u8; 8 * 5];
  let v = std::vec![128u8; 8 * 5];
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
  assert_eq!(f.height(), 9);
}

#[test]
fn try_new_rejects_y_stride_under_width() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 8, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::YStrideTooSmall { .. }));
}

#[test]
fn try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::YPlaneTooShort { .. }));
}

#[test]
fn try_new_rejects_short_u_plane() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::UPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Yuv420pFrame")]
fn new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let _ = Yuv420pFrame::new(&y, &u, &v, 16, 8, 16, 8, 8);
}
