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
fn v210_frame_try_new_rejects_odd_width() {
  // Odd widths violate the 4:2:2 chroma-pair constraint. Even widths
  // that aren't multiples of 6 (e.g. 1280 = 720p) are now supported
  // via partial-word handling — see the dedicated accept tests.
  let buf = std::vec![0u8; 4096];
  for w in [1u32, 3, 5, 7, 9, 11, 13, 15] {
    let stride = ((w as usize) * 8 / 3 + 16) as u32;
    let err = V210Frame::try_new(&buf, w, 1, stride).unwrap_err();
    assert_eq!(err, V210FrameError::OddWidth { width: w });
  }
}

#[test]
fn v210_frame_try_new_accepts_partial_word_width() {
  // 720p HD: width = 1280 ⇒ 1280 / 6 = 213 rem 2 ⇒ ceil(1280/6) = 214
  // words = 214 * 16 = 3424 bytes per row. The last word holds 2 valid
  // pixels (Cb, Y, Cr, Y) and 12 unused/undefined bytes.
  let stride = 1280u32.div_ceil(6) * 16;
  assert_eq!(stride, 3424);
  let buf = std::vec![0u8; stride as usize];
  let frame = V210Frame::try_new(&buf, 1280, 1, stride).unwrap();
  assert_eq!(frame.width(), 1280);
  assert_eq!(frame.stride(), 3424);

  // Width = 1922 (1920 + 2) — forces the partial-word tail right after
  // a long sequence of complete words.
  let stride = 1922u32.div_ceil(6) * 16;
  let buf = std::vec![0u8; stride as usize];
  V210Frame::try_new(&buf, 1922, 1, stride).unwrap();

  // Smaller partial-word widths.
  for w in [2u32, 4, 8, 10, 14, 16, 20] {
    let stride = w.div_ceil(6) * 16;
    let buf = std::vec![0u8; stride as usize];
    V210Frame::try_new(&buf, w, 1, stride).unwrap();
  }
}

#[test]
fn v210_frame_accessors_round_trip_partial_word() {
  // Round-trip on a 720p-sized frame (partial-word at end of row).
  let stride = 1280u32.div_ceil(6) * 16;
  let buf = std::vec![0u8; stride as usize * 720];
  let frame = V210Frame::try_new(&buf, 1280, 720, stride).unwrap();
  assert_eq!(frame.v210().len(), stride as usize * 720);
  assert_eq!(frame.width(), 1280);
  assert_eq!(frame.height(), 720);
  assert_eq!(frame.stride(), stride);
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
  // For width=8 (partial-word, 2 words), min_stride = 32.
  assert!(matches!(
    V210Frame::try_new(&buf, 8, 1, 31),
    Err(V210FrameError::StrideTooSmall {
      min_stride: 32,
      stride: 31
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
  // `new` is the panicking convenience; zero dimension is a guaranteed
  // invariant violation regardless of the partial-word width relaxation.
  let _ = V210Frame::new(&buf, 0, 0, 0);
}
