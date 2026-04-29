use super::*;

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

