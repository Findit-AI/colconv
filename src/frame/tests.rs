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

// ---- Nv12Frame ---------------------------------------------------------

fn nv12_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame → UV is 8 chroma columns × 4 chroma rows = 16 bytes/row.
  (std::vec![0u8; 16 * 8], std::vec![128u8; 16 * 4])
}

#[test]
fn nv12_try_new_accepts_valid_tight() {
  let (y, uv) = nv12_planes();
  let f = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.uv_stride(), 16);
}

#[test]
fn nv12_try_new_accepts_valid_padded_strides() {
  let y = std::vec![0u8; 32 * 8];
  let uv = std::vec![128u8; 32 * 4];
  let f = Nv12Frame::try_new(&y, &uv, 16, 8, 32, 32).expect("valid");
  assert_eq!(f.y_stride(), 32);
  assert_eq!(f.uv_stride(), 32);
}

#[test]
fn nv12_try_new_rejects_zero_dim() {
  let (y, uv) = nv12_planes();
  let e = Nv12Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv12FrameError::ZeroDimension { .. }));
}

#[test]
fn nv12_try_new_rejects_odd_width() {
  let (y, uv) = nv12_planes();
  let e = Nv12Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv12FrameError::OddWidth { width: 15 }));
}

#[test]
fn nv12_try_new_accepts_odd_height() {
  // 640x481 — concrete case flagged by adversarial review. chroma_height =
  // ceil(481/2) = 241, so UV plane is 640*241 bytes. Constructor must
  // accept this.
  let y = std::vec![0u8; 640 * 481];
  let uv = std::vec![128u8; 640 * 241];
  let f = Nv12Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
  assert_eq!(f.height(), 481);
  assert_eq!(f.width(), 640);
}

#[test]
fn nv12_try_new_rejects_y_stride_under_width() {
  let (y, uv) = nv12_planes();
  let e = Nv12Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(e, Nv12FrameError::YStrideTooSmall { .. }));
}

#[test]
fn nv12_try_new_rejects_uv_stride_under_width() {
  let (y, uv) = nv12_planes();
  let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
  assert!(matches!(e, Nv12FrameError::UvStrideTooSmall { .. }));
}

#[test]
fn nv12_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 16 * 4];
  let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv12FrameError::YPlaneTooShort { .. }));
}

#[test]
fn nv12_try_new_rejects_short_uv_plane() {
  let y = std::vec![0u8; 16 * 8];
  let uv = std::vec![128u8; 8];
  let e = Nv12Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv12FrameError::UvPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Nv12Frame")]
fn nv12_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 16 * 4];
  let _ = Nv12Frame::new(&y, &uv, 16, 8, 16, 16);
}

// ---- 32-bit overflow regressions --------------------------------------
//
// `u32 * u32` can exceed `usize::MAX` only on 32-bit targets (wasm32,
// i686). Gate the tests so they actually run on those hosts under CI
// cross builds; on 64-bit they're trivially uninteresting (the
// product always fits).

#[cfg(target_pointer_width = "32")]
#[test]
fn yuv420p_try_new_rejects_y_geometry_overflow() {
  // 0x1_0000 * 0x1_0000 = 2^32, which overflows a 32-bit `usize`
  // (max = 2^32 − 1). Even so the odd-width check passes, so we
  // actually reach `checked_mul` and hit `GeometryOverflow`.
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let u: [u8; 0] = [];
  let v: [u8; 0] = [];
  let e = Yuv420pFrame::try_new(&y, &u, &v, big, big, big, big / 2, big / 2).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::GeometryOverflow { .. }));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn nv12_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let uv: [u8; 0] = [];
  let e = Nv12Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
  assert!(matches!(e, Nv12FrameError::GeometryOverflow { .. }));
}

// ---- Nv16Frame ---------------------------------------------------------
//
// 4:2:2: chroma is half-width, **full-height**. UV plane is `width *
// height` bytes (vs. NV12's `width * height / 2`). No height parity
// constraint.

fn nv16_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame → UV is 8 chroma columns × 8 chroma rows = 16 bytes/row
  // × 8 rows (not 4 — full height).
  (std::vec![0u8; 16 * 8], std::vec![128u8; 16 * 8])
}

#[test]
fn nv16_try_new_accepts_valid_tight() {
  let (y, uv) = nv16_planes();
  let f = Nv16Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.uv_stride(), 16);
}

#[test]
fn nv16_try_new_accepts_valid_padded_strides() {
  let y = std::vec![0u8; 32 * 8];
  let uv = std::vec![128u8; 32 * 8];
  let f = Nv16Frame::try_new(&y, &uv, 16, 8, 32, 32).expect("valid");
  assert_eq!(f.y_stride(), 32);
  assert_eq!(f.uv_stride(), 32);
}

#[test]
fn nv16_try_new_rejects_zero_dim() {
  let (y, uv) = nv16_planes();
  let e = Nv16Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv16FrameError::ZeroDimension { .. }));
}

#[test]
fn nv16_try_new_rejects_odd_width() {
  let (y, uv) = nv16_planes();
  let e = Nv16Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv16FrameError::OddWidth { width: 15 }));
}

#[test]
fn nv16_try_new_accepts_odd_height() {
  // 4:2:2 has no height parity restriction (chroma is full-height,
  // 1:1 per Y row). A 640x481 NV16 frame should construct fine.
  let y = std::vec![0u8; 640 * 481];
  let uv = std::vec![128u8; 640 * 481];
  let f = Nv16Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
  assert_eq!(f.height(), 481);
  assert_eq!(f.width(), 640);
}

#[test]
fn nv16_try_new_rejects_y_stride_under_width() {
  let (y, uv) = nv16_planes();
  let e = Nv16Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(e, Nv16FrameError::YStrideTooSmall { .. }));
}

#[test]
fn nv16_try_new_rejects_uv_stride_under_width() {
  let (y, uv) = nv16_planes();
  let e = Nv16Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
  assert!(matches!(e, Nv16FrameError::UvStrideTooSmall { .. }));
}

#[test]
fn nv16_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 16 * 8];
  let e = Nv16Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv16FrameError::YPlaneTooShort { .. }));
}

#[test]
fn nv16_try_new_rejects_short_uv_plane() {
  let y = std::vec![0u8; 16 * 8];
  // NV12 would accept `16 * 4 = 64` bytes here; NV16 needs full
  // height → this must fail.
  let uv = std::vec![128u8; 16 * 4];
  let e = Nv16Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv16FrameError::UvPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Nv16Frame")]
fn nv16_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 16 * 8];
  let _ = Nv16Frame::new(&y, &uv, 16, 8, 16, 16);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn nv16_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let uv: [u8; 0] = [];
  let e = Nv16Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
  assert!(matches!(e, Nv16FrameError::GeometryOverflow { .. }));
}

// ---- Nv24Frame ---------------------------------------------------------
//
// 4:4:4: chroma is full-width and full-height. UV plane is
// `2 * width * height` bytes. No width parity constraint.

fn nv24_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame → UV is 16 chroma columns × 8 chroma rows = 32 bytes/row
  // × 8 rows = 256 bytes.
  (std::vec![0u8; 16 * 8], std::vec![128u8; 32 * 8])
}

#[test]
fn nv24_try_new_accepts_valid_tight() {
  let (y, uv) = nv24_planes();
  let f = Nv24Frame::try_new(&y, &uv, 16, 8, 16, 32).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.uv_stride(), 32);
}

#[test]
fn nv24_try_new_accepts_odd_width() {
  // 4:4:4 has no width parity constraint. 17×8 → UV plane = 34 * 8.
  let y = std::vec![0u8; 17 * 8];
  let uv = std::vec![128u8; 34 * 8];
  let f = Nv24Frame::try_new(&y, &uv, 17, 8, 17, 34).expect("odd width valid");
  assert_eq!(f.width(), 17);
}

#[test]
fn nv24_try_new_accepts_odd_height() {
  let y = std::vec![0u8; 16 * 7];
  let uv = std::vec![128u8; 32 * 7];
  let f = Nv24Frame::try_new(&y, &uv, 16, 7, 16, 32).expect("odd height valid");
  assert_eq!(f.height(), 7);
}

#[test]
fn nv24_try_new_rejects_zero_dim() {
  let (y, uv) = nv24_planes();
  let e = Nv24Frame::try_new(&y, &uv, 0, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv24FrameError::ZeroDimension { .. }));
}

#[test]
fn nv24_try_new_rejects_y_stride_under_width() {
  let (y, uv) = nv24_planes();
  let e = Nv24Frame::try_new(&y, &uv, 16, 8, 8, 32).unwrap_err();
  assert!(matches!(e, Nv24FrameError::YStrideTooSmall { .. }));
}

#[test]
fn nv24_try_new_rejects_uv_stride_under_double_width() {
  let (y, uv) = nv24_planes();
  // 4:4:4 requires uv_stride >= 2 * width (= 32). 16 is insufficient.
  let e = Nv24Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv24FrameError::UvStrideTooSmall { .. }));
}

#[test]
fn nv24_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 32 * 8];
  let e = Nv24Frame::try_new(&y, &uv, 16, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv24FrameError::YPlaneTooShort { .. }));
}

#[test]
fn nv24_try_new_rejects_short_uv_plane() {
  let y = std::vec![0u8; 16 * 8];
  let uv = std::vec![128u8; 32]; // one row instead of 8
  let e = Nv24Frame::try_new(&y, &uv, 16, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv24FrameError::UvPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Nv24Frame")]
fn nv24_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let uv = std::vec![128u8; 32 * 8];
  let _ = Nv24Frame::new(&y, &uv, 16, 8, 16, 32);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn nv24_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let uv: [u8; 0] = [];
  // stride * height overflow path
  let e = Nv24Frame::try_new(&y, &uv, big, big, big, big * 2).unwrap_err();
  assert!(matches!(e, Nv24FrameError::GeometryOverflow { .. }));
}

#[test]
fn nv24_try_new_rejects_uv_width_overflow_u32() {
  // `width * 2` overflows u32 → we report GeometryOverflow before
  // even looking at uv_stride.
  let y: [u8; 0] = [];
  let uv: [u8; 0] = [];
  // width >= 2^31 makes `width * 2` overflow u32.
  let w: u32 = 0x8000_0000;
  let e = Nv24Frame::try_new(&y, &uv, w, 1, w, 0).unwrap_err();
  assert!(matches!(e, Nv24FrameError::GeometryOverflow { .. }));
}

// ---- Nv42Frame ---------------------------------------------------------
//
// Structurally identical to Nv24. Tests mirror the Nv24 set.

fn nv42_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  (std::vec![0u8; 16 * 8], std::vec![128u8; 32 * 8])
}

#[test]
fn nv42_try_new_accepts_valid_tight() {
  let (y, vu) = nv42_planes();
  let f = Nv42Frame::try_new(&y, &vu, 16, 8, 16, 32).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.vu_stride(), 32);
}

#[test]
fn nv42_try_new_accepts_odd_width() {
  let y = std::vec![0u8; 17 * 8];
  let vu = std::vec![128u8; 34 * 8];
  let f = Nv42Frame::try_new(&y, &vu, 17, 8, 17, 34).expect("odd width valid");
  assert_eq!(f.width(), 17);
}

#[test]
fn nv42_try_new_rejects_zero_dim() {
  let (y, vu) = nv42_planes();
  let e = Nv42Frame::try_new(&y, &vu, 0, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv42FrameError::ZeroDimension { .. }));
}

#[test]
fn nv42_try_new_rejects_vu_stride_under_double_width() {
  let (y, vu) = nv42_planes();
  let e = Nv42Frame::try_new(&y, &vu, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv42FrameError::VuStrideTooSmall { .. }));
}

#[test]
fn nv42_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let vu = std::vec![128u8; 32 * 8];
  let e = Nv42Frame::try_new(&y, &vu, 16, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv42FrameError::YPlaneTooShort { .. }));
}

#[test]
fn nv42_try_new_rejects_short_vu_plane() {
  let y = std::vec![0u8; 16 * 8];
  let vu = std::vec![128u8; 32];
  let e = Nv42Frame::try_new(&y, &vu, 16, 8, 16, 32).unwrap_err();
  assert!(matches!(e, Nv42FrameError::VuPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Nv42Frame")]
fn nv42_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let vu = std::vec![128u8; 32 * 8];
  let _ = Nv42Frame::new(&y, &vu, 16, 8, 16, 32);
}

// ---- Nv21Frame ---------------------------------------------------------
//
// NV21 is structurally identical to NV12 (same plane count, same
// stride/size math) — only the byte order within the chroma plane
// differs. Validation tests mirror the NV12 set. Kernel-level
// equivalence with NV12-swapped-UV is tested in `src/row/arch/*`.

fn nv21_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame → VU is 16 bytes × 4 chroma rows.
  (std::vec![0u8; 16 * 8], std::vec![128u8; 16 * 4])
}

#[test]
fn nv21_try_new_accepts_valid_tight() {
  let (y, vu) = nv21_planes();
  let f = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.vu_stride(), 16);
}

#[test]
fn nv21_try_new_accepts_odd_height() {
  // Same concrete case as NV12 — 640x481.
  let y = std::vec![0u8; 640 * 481];
  let vu = std::vec![128u8; 640 * 241];
  let f = Nv21Frame::try_new(&y, &vu, 640, 481, 640, 640).expect("odd height valid");
  assert_eq!(f.height(), 481);
}

#[test]
fn nv21_try_new_rejects_odd_width() {
  let (y, vu) = nv21_planes();
  let e = Nv21Frame::try_new(&y, &vu, 15, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv21FrameError::OddWidth { width: 15 }));
}

#[test]
fn nv21_try_new_rejects_zero_dim() {
  let (y, vu) = nv21_planes();
  let e = Nv21Frame::try_new(&y, &vu, 0, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv21FrameError::ZeroDimension { .. }));
}

#[test]
fn nv21_try_new_rejects_vu_stride_under_width() {
  let (y, vu) = nv21_planes();
  let e = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 8).unwrap_err();
  assert!(matches!(e, Nv21FrameError::VuStrideTooSmall { .. }));
}

#[test]
fn nv21_try_new_rejects_short_vu_plane() {
  let y = std::vec![0u8; 16 * 8];
  let vu = std::vec![128u8; 8];
  let e = Nv21Frame::try_new(&y, &vu, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, Nv21FrameError::VuPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Nv21Frame")]
fn nv21_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let vu = std::vec![128u8; 16 * 4];
  let _ = Nv21Frame::new(&y, &vu, 16, 8, 16, 16);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn nv21_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let vu: [u8; 0] = [];
  let e = Nv21Frame::try_new(&y, &vu, big, big, big, big).unwrap_err();
  assert!(matches!(e, Nv21FrameError::GeometryOverflow { .. }));
}

// ---- Yuv420pFrame16 / Yuv420p10Frame ----------------------------------
//
// Storage is `&[u16]` with sample-indexed strides. Validation mirrors
// the 8-bit [`Yuv420pFrame`] with the addition of the `BITS` guard.

fn p10_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 16×8 frame, chroma 8×4. Y plane solid black (Y=0); UV planes
  // neutral (UV=512 = 10‑bit chroma center). Exact sample values
  // don't matter for the constructor tests that use this helper —
  // they only look at shape, geometry errors, and the reported
  // bits.
  (
    std::vec![0u16; 16 * 8],
    std::vec![512u16; 8 * 4],
    std::vec![512u16; 8 * 4],
  )
}

#[test]
fn yuv420p10_try_new_accepts_valid_tight() {
  let (y, u, v) = p10_planes();
  let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv420p10_try_new_accepts_odd_height() {
  // 16x9 → chroma_height = 5. Y plane 16*9 = 144 samples, U/V 8*5 = 40.
  let y = std::vec![0u16; 16 * 9];
  let u = std::vec![512u16; 8 * 5];
  let v = std::vec![512u16; 8 * 5];
  let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
  assert_eq!(f.height(), 9);
}

#[test]
fn yuv420p10_try_new_rejects_odd_width() {
  let (y, u, v) = p10_planes();
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::OddWidth { width: 15 }));
}

#[test]
fn yuv420p10_try_new_rejects_zero_dim() {
  let (y, u, v) = p10_planes();
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::ZeroDimension { .. }));
}

#[test]
fn yuv420p10_try_new_rejects_short_y_plane() {
  let y = std::vec![0u16; 10];
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
}

#[test]
fn yuv420p10_try_new_rejects_short_u_plane() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::UPlaneTooShort { .. }));
}

#[test]
fn yuv420p_frame16_try_new_rejects_unsupported_bits() {
  // BITS must be in {9, 10, 12, 14, 16}. 11, 15, etc. are rejected
  // before any plane math runs.
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![128u16; 8 * 4];
  let v = std::vec![128u16; 8 * 4];
  let e = Yuv420pFrame16::<11>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::UnsupportedBits { bits: 11 }
  ));
  let e15 = Yuv420pFrame16::<15>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e15,
    Yuv420pFrame16Error::UnsupportedBits { bits: 15 }
  ));
}

#[test]
fn yuv420p16_try_new_accepts_12_14_and_16() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![2048u16; 8 * 4];
  let v = std::vec![2048u16; 8 * 4];
  let f12 = Yuv420pFrame16::<12>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("12-bit valid");
  assert_eq!(f12.bits(), 12);
  let f14 = Yuv420pFrame16::<14>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("14-bit valid");
  assert_eq!(f14.bits(), 14);
  let f16 = Yuv420p16Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("16-bit valid");
  assert_eq!(f16.bits(), 16);
}

#[test]
fn yuv420p16_try_new_checked_accepts_full_u16_range() {
  // At 16 bits the full u16 range is valid — max sample = 65535.
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 8 * 4];
  let v = std::vec![32768u16; 8 * 4];
  Yuv420p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8)
    .expect("every u16 value is in range at 16 bits");
}

#[test]
fn p016_try_new_accepts_16bit() {
  let y = std::vec![0xFFFFu16; 16 * 8];
  let uv = std::vec![0x8000u16; 16 * 4];
  let f = P016Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("P016 valid");
  assert_eq!(f.bits(), 16);
}

#[test]
fn p016_try_new_checked_is_a_noop() {
  // At BITS == 16 there are zero "low" bits to check — every u16
  // value is a valid P016 sample because `16 - BITS == 0`. The
  // checked constructor therefore accepts everything. This pins
  // that behavior in a test: at 16 bits the semantic distinction
  // between P016 and yuv420p16le **cannot be detected** from
  // sample values at all (no bit pattern is packing-specific).
  let y = std::vec![0x1234u16; 16 * 8];
  let uv = std::vec![0x5678u16; 16 * 4];
  P016Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("every u16 passes the low-bits check at BITS == 16");
}

#[test]
fn pn_try_new_rejects_bits_other_than_10_12_16() {
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0u16; 16 * 4];
  let e14 = PnFrame::<14>::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e14, PnFrameError::UnsupportedBits { bits: 14 }));
  let e11 = PnFrame::<11>::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e11, PnFrameError::UnsupportedBits { bits: 11 }));
}

#[test]
#[should_panic(expected = "invalid Yuv420pFrame16")]
fn yuv420p10_new_panics_on_invalid() {
  let y = std::vec![0u16; 10];
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let _ = Yuv420p10Frame::new(&y, &u, &v, 16, 8, 16, 8, 8);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn yuv420p10_try_new_rejects_geometry_overflow() {
  // Sample count overflow on 32-bit. Same rationale as the 8-bit
  // version — strides are in `u16` elements here, so the same
  // `0x1_0000 * 0x1_0000` product overflows `usize`.
  let big: u32 = 0x1_0000;
  let y: [u16; 0] = [];
  let u: [u16; 0] = [];
  let v: [u16; 0] = [];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, big, big, big, big / 2, big / 2).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::GeometryOverflow { .. }));
}

#[test]
fn yuv420p10_try_new_checked_accepts_in_range_samples() {
  // Same valid frame as `yuv420p10_try_new_accepts_valid_tight`,
  // but run through the checked constructor. All samples live in
  // the 10‑bit range.
  let (y, u, v) = p10_planes();
  let f = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv420p10_try_new_checked_rejects_y_high_bit_set() {
  // A Y sample with bit 15 set — typical of `p010` packing where
  // the 10 active bits sit in the high bits. `try_new` would
  // accept this and let the SIMD kernels produce arch‑dependent
  // garbage; `try_new_checked` catches it up front.
  let mut y = std::vec![0u16; 16 * 8];
  y[3 * 16 + 5] = 0x8000; // bit 15 set → way above 1023
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
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
fn yuv420p10_try_new_checked_rejects_u_plane_sample() {
  // Offending sample in the U plane — error must name U, not Y or V.
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 8 * 4];
  u[2 * 8 + 3] = 1024; // just above max
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
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
fn yuv420p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 8 * 4];
  let mut v = std::vec![512u16; 8 * 4];
  v[8 + 7] = 0xFFFF; // all bits set
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv420p10_try_new_checked_accepts_exact_max_sample() {
  // Boundary: sample value == (1 << BITS) - 1 is valid.
  let mut y = std::vec![0u16; 16 * 8];
  y[0] = 1023;
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("1023 is in range");
}

#[test]
fn yuv420p10_try_new_checked_reports_geometry_errors_first() {
  // If geometry is invalid, we never get to the sample scan — the
  // same errors as `try_new` surface first. Prevents the checked
  // path from doing unnecessary O(N) work on inputs that would
  // fail for a simpler reason.
  let y = std::vec![0u16; 10]; // Too small.
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
}

// ---- P010Frame ---------------------------------------------------------
//
// Semi‑planar 10‑bit. Plane shape mirrors Nv12Frame (Y + interleaved
// UV) but sample width is `u16` with the 10 active bits in the
// **high** 10 of each element (`value << 6`). Strides are in
// samples, not bytes.

fn p010_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 16×8 frame — UV plane carries 16 u16 × 4 chroma rows = 64 u16.
  // P010 white Y = 1023 << 6 = 0xFFC0; neutral UV = 512 << 6 = 0x8000.
  (std::vec![0xFFC0u16; 16 * 8], std::vec![0x8000u16; 16 * 4])
}

#[test]
fn p010_try_new_accepts_valid_tight() {
  let (y, uv) = p010_planes();
  let f = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.uv_stride(), 16);
}

#[test]
fn p010_try_new_accepts_odd_height() {
  // 640×481 — same concrete odd‑height case covered by NV12 / NV21.
  let y = std::vec![0u16; 640 * 481];
  let uv = std::vec![0x8000u16; 640 * 241];
  let f = P010Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
  assert_eq!(f.height(), 481);
}

#[test]
fn p010_try_new_rejects_odd_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::OddWidth { width: 15 }));
}

#[test]
fn p010_try_new_rejects_zero_dim() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::ZeroDimension { .. }));
}

#[test]
fn p010_try_new_rejects_y_stride_under_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YStrideTooSmall { .. }));
}

#[test]
fn p010_try_new_rejects_uv_stride_under_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideTooSmall { .. }));
}

#[test]
fn p010_try_new_rejects_odd_uv_stride() {
  // uv_stride = 17 passes the size check (>= width = 16) but is
  // odd, which would mis-align the (U, V) pair on every other row.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 17 * 4];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 17).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 17 }));
}

#[test]
fn p210_try_new_rejects_odd_uv_stride() {
  // PnFrame422 chroma is half-width × full-height with 2 u16 per
  // pair → uv_row_elems = width. Same odd-stride bug as P010.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 17 * 8];
  let e = P210Frame::try_new(&y, &uv, 16, 8, 16, 17).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 17 }));
}

#[test]
fn p410_try_new_rejects_odd_uv_stride() {
  // PnFrame444 chroma is full-width × full-height with 2 u16 per
  // pair → uv_row_elems = 2 * width = 32. uv_stride = 33 passes
  // the size check but is odd.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 33 * 8];
  let e = P410Frame::try_new(&y, &uv, 16, 8, 16, 33).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 33 }));
}

#[test]
fn p010_try_new_rejects_short_y_plane() {
  let y = std::vec![0u16; 10];
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YPlaneTooShort { .. }));
}

#[test]
fn p010_try_new_rejects_short_uv_plane() {
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 8];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::UvPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid PnFrame")]
fn p010_new_panics_on_invalid() {
  let y = std::vec![0u16; 10];
  let uv = std::vec![0x8000u16; 16 * 4];
  let _ = P010Frame::new(&y, &uv, 16, 8, 16, 16);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn p010_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u16; 0] = [];
  let uv: [u16; 0] = [];
  let e = P010Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
  assert!(matches!(e, PnFrameError::GeometryOverflow { .. }));
}

#[test]
fn p010_try_new_checked_accepts_shifted_samples() {
  // Valid P010 samples: low 6 bits zero.
  let (y, uv) = p010_planes();
  P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).expect("shifted samples valid");
}

#[test]
fn p010_try_new_checked_rejects_y_low_bits_set() {
  // A Y sample with low 6 bits set — characteristic of yuv420p10le
  // packing (value in low 10 bits) accidentally handed to the P010
  // constructor. `try_new_checked` catches this; plain `try_new`
  // would let the kernel mask it down and produce wrong colors.
  let mut y = std::vec![0xFFC0u16; 16 * 8];
  y[3 * 16 + 5] = 0x03FF; // 10-bit value in low bits — wrong packing
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  match e {
    PnFrameError::SampleLowBitsSet { plane, value, .. } => {
      assert_eq!(plane, P010FramePlane::Y);
      assert_eq!(value, 0x03FF);
    }
    other => panic!("expected SampleLowBitsSet, got {other:?}"),
  }
}

#[test]
fn p010_try_new_checked_rejects_uv_plane_sample() {
  let y = std::vec![0xFFC0u16; 16 * 8];
  let mut uv = std::vec![0x8000u16; 16 * 4];
  uv[2 * 16 + 3] = 0x0001; // low bit set
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    PnFrameError::SampleLowBitsSet {
      plane: P010FramePlane::Uv,
      value: 0x0001,
      ..
    }
  ));
}

#[test]
fn p010_try_new_checked_reports_geometry_errors_first() {
  let y = std::vec![0u16; 10]; // Too small.
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YPlaneTooShort { .. }));
}

/// Regression documenting a **known limitation** of
/// [`P010Frame::try_new_checked`]: the low‑6‑bits‑zero check is a
/// packing sanity check, not a provenance validator. A
/// `yuv420p10le` buffer whose samples all happen to be multiples
/// of 64 — e.g. `Y = 64` (limited‑range black, `0x0040`) and
/// `UV = 512` (neutral chroma, `0x0200`) — passes the check
/// silently, even though the layout is wrong and downstream P010
/// kernels will produce incorrect output.
///
/// The test asserts the check accepts these values so the limit
/// is visible in the test log; any future attempt to tighten the
/// constructor into a real provenance validator will need to
/// update or replace this test.
#[test]
fn p010_try_new_checked_accepts_ambiguous_yuv420p10le_samples() {
  // `yuv420p10le`-style samples, all multiples of 64: low 6 bits
  // are zero, so they pass the P010 sanity check even though this
  // is wrong data for a P010 frame.
  let y = std::vec![0x0040u16; 16 * 8]; // limited-range black in 10-bit low-packed
  let uv = std::vec![0x0200u16; 16 * 4]; // neutral chroma in 10-bit low-packed
  let f = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("known limitation: low-6-bits-zero check cannot tell yuv420p10le from P010");
  assert_eq!(f.width(), 16);
  // Downstream decoding of this frame would produce wrong colors
  // (every `>> 6` extracts 1 from Y=0x0040 and 8 from UV=0x0200,
  // which P010 kernels then bias/scale as if those were the 10-bit
  // source values). That's accepted behavior — the type system,
  // not `try_new_checked`, is what keeps yuv420p10le out of P010.
}

#[test]
fn p012_try_new_checked_accepts_shifted_samples() {
  // Valid P012 samples: low 4 bits zero (12-bit value << 4).
  let y = std::vec![(2048u16) << 4; 16 * 8]; // 12-bit mid-gray shifted up
  let uv = std::vec![(2048u16) << 4; 16 * 4];
  P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).expect("shifted samples valid");
}

#[test]
fn p012_try_new_checked_rejects_low_bits_set() {
  // A Y sample with any of the low 4 bits set — e.g. yuv420p12le
  // value 0x0ABC landing where P012 expects `value << 4`. The check
  // catches samples like this that are obviously mispacked.
  let mut y = std::vec![(2048u16) << 4; 16 * 8];
  y[3 * 16 + 5] = 0x0ABC; // low 4 bits = 0xC ≠ 0
  let uv = std::vec![(2048u16) << 4; 16 * 4];
  let e = P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  match e {
    PnFrameError::SampleLowBitsSet {
      plane,
      value,
      low_bits,
      ..
    } => {
      assert_eq!(plane, PnFramePlane::Y);
      assert_eq!(value, 0x0ABC);
      assert_eq!(low_bits, 4);
    }
    other => panic!("expected SampleLowBitsSet, got {other:?}"),
  }
}

/// Regression documenting a **worse known limitation** of
/// [`P012Frame::try_new_checked`] compared to P010: because the
/// low‑bits check only has 4 bits to work with at `BITS == 12`,
/// every multiple‑of‑16 `yuv420p12le` value passes silently. The
/// practical impact is that common limited‑range flat‑region
/// content in real decoder output — `Y = 256` (limited‑range
/// black), `UV = 2048` (neutral chroma), `Y = 1024` (full black)
/// — is entirely invisible to this check.
///
/// This test pins the limitation with a reproducible input so
/// that:
/// 1. Users reading the test suite can see the exact failure
///    mode for `try_new_checked` on 12‑bit data.
/// 2. Any future attempt to strengthen `try_new_checked` (e.g.,
///    into a statistical provenance heuristic) has a concrete
///    input to validate against.
/// 3. The `PnFrame` docs' warning about this limitation has a
///    named test to point to.
///
/// For P012, the type system (choosing [`P012Frame`] vs
/// [`Yuv420p12Frame`] at construction based on decoder metadata)
/// is the only reliable provenance guarantee.
#[test]
fn p012_try_new_checked_accepts_low_packed_flat_content_by_design() {
  // All values are multiples of 16 — exactly the set that slips
  // through a 4-low-bits-zero check. `yuv420p12le` limited-range
  // black and neutral chroma both satisfy this.
  let y = std::vec![0x0100u16; 16 * 8]; // Y = 256 (limited-range black), multiple of 16
  let uv = std::vec![0x0800u16; 16 * 4]; // UV = 2048 (neutral chroma), multiple of 16
  let f = P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("known limitation: 4-low-bits-zero check cannot tell yuv420p12le from P012");
  assert_eq!(f.width(), 16);
  // Downstream P012 kernels would extract `>> 4` — giving Y=16 and
  // UV=128 instead of the intended Y=256 and UV=2048. Silent color
  // corruption. The type system, not `try_new_checked`, must
  // guarantee provenance for 12-bit.
}

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

// ---- Rgb24Frame --------------------------------------------------------
//
// Single-plane 8-bit packed RGB. `stride` is in bytes (≥ 3 * width);
// `plane.len() >= stride * height`. No width parity constraint.

#[test]
fn rgb24_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  Rgb24Frame::try_new(&buf, 16, 4, 48).expect("valid");
}

#[test]
fn rgb24_frame_try_new_accepts_oversized_stride() {
  // stride > 3 * width (row padding) is allowed.
  let buf = std::vec![0u8; 64 * 4];
  Rgb24Frame::try_new(&buf, 16, 4, 64).expect("padded stride is valid");
}

#[test]
fn rgb24_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  assert!(matches!(
    Rgb24Frame::try_new(&buf, 0, 4, 48),
    Err(Rgb24FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Rgb24Frame::try_new(&buf, 16, 0, 48),
    Err(Rgb24FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn rgb24_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  assert!(matches!(
    Rgb24Frame::try_new(&buf, 16, 4, 47),
    Err(Rgb24FrameError::StrideTooSmall {
      min_stride: 48,
      stride: 47,
    })
  ));
}

#[test]
fn rgb24_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 3];
  assert!(matches!(
    Rgb24Frame::try_new(&small, 16, 4, 48),
    Err(Rgb24FrameError::PlaneTooShort {
      expected: 192,
      actual: 48,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid Rgb24Frame")]
fn rgb24_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Rgb24Frame::new(&buf, 16, 4, 48);
}

// ---- Bgr24Frame --------------------------------------------------------
//
// Mirrors Rgb24Frame: same single-plane layout, channel order is
// purely a marker / accessor distinction. Validation is identical in
// shape so we re-test the variants to catch typos in the parallel
// implementation.

#[test]
fn bgr24_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  Bgr24Frame::try_new(&buf, 16, 4, 48).expect("valid");
}

#[test]
fn bgr24_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  assert!(matches!(
    Bgr24Frame::try_new(&buf, 0, 4, 48),
    Err(Bgr24FrameError::ZeroDimension { .. })
  ));
}

#[test]
fn bgr24_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 3];
  assert!(matches!(
    Bgr24Frame::try_new(&buf, 16, 4, 47),
    Err(Bgr24FrameError::StrideTooSmall {
      min_stride: 48,
      stride: 47,
    })
  ));
}

#[test]
fn bgr24_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 3];
  assert!(matches!(
    Bgr24Frame::try_new(&small, 16, 4, 48),
    Err(Bgr24FrameError::PlaneTooShort { .. })
  ));
}

#[test]
#[should_panic(expected = "invalid Bgr24Frame")]
fn bgr24_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Bgr24Frame::new(&buf, 16, 4, 48);
}

// ---- RgbaFrame --------------------------------------------------------
//
// Single-plane 8-bit packed RGBA. `stride` is in bytes (≥ 4 * width);
// `plane.len() >= stride * height`. No width parity constraint.

#[test]
fn rgba_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  RgbaFrame::try_new(&buf, 16, 4, 64).expect("valid");
}

#[test]
fn rgba_frame_try_new_accepts_oversized_stride() {
  // stride > 4 * width (row padding) is allowed.
  let buf = std::vec![0u8; 96 * 4];
  RgbaFrame::try_new(&buf, 16, 4, 96).expect("padded stride is valid");
}

#[test]
fn rgba_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    RgbaFrame::try_new(&buf, 0, 4, 64),
    Err(RgbaFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    RgbaFrame::try_new(&buf, 16, 0, 64),
    Err(RgbaFrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn rgba_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    RgbaFrame::try_new(&buf, 16, 4, 63),
    Err(RgbaFrameError::StrideTooSmall {
      min_stride: 64,
      stride: 63,
    })
  ));
}

#[test]
fn rgba_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 4];
  assert!(matches!(
    RgbaFrame::try_new(&small, 16, 4, 64),
    Err(RgbaFrameError::PlaneTooShort {
      expected: 256,
      actual: 64,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid RgbaFrame")]
fn rgba_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = RgbaFrame::new(&buf, 16, 4, 64);
}

// ---- BgraFrame --------------------------------------------------------
//
// Mirrors RgbaFrame: same single-plane layout, channel order is
// purely a marker / accessor distinction. Validation is identical in
// shape so we re-test the variants to catch typos in the parallel
// implementation.

#[test]
fn bgra_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  BgraFrame::try_new(&buf, 16, 4, 64).expect("valid");
}

#[test]
fn bgra_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    BgraFrame::try_new(&buf, 0, 4, 64),
    Err(BgraFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn bgra_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 4 * 4];
  assert!(matches!(
    BgraFrame::try_new(&buf, 16, 4, 63),
    Err(BgraFrameError::StrideTooSmall {
      min_stride: 64,
      stride: 63,
    })
  ));
}

#[test]
fn bgra_frame_try_new_rejects_short_plane() {
  let small = std::vec![0u8; 16 * 4];
  assert!(matches!(
    BgraFrame::try_new(&small, 16, 4, 64),
    Err(BgraFrameError::PlaneTooShort { .. })
  ));
}

#[test]
#[should_panic(expected = "invalid BgraFrame")]
fn bgra_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = BgraFrame::new(&buf, 16, 4, 64);
}
