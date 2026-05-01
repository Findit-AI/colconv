use super::*;

mod bayer;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod packed_yuv_4_4_4;
mod packed_yuv_8bit;
mod planar_8bit;
mod semi_planar_8bit;
mod subsampled_4_2_0_high_bit;
mod subsampled_4_2_2_high_bit;
mod subsampled_4_4_4_high_bit;
mod v210;
mod y2xx;

// ---- 32-bit overflow regressions --------------------------------------
//
// `u32 * u32` can exceed `usize::MAX` only on 32-bit targets (wasm32,
// i686). Gate the tests so they actually run on those hosts under CI
// cross builds; on 64-bit they're trivially uninteresting (the
// product always fits). These tests stay in `tests/mod.rs` because
// they exercise both `Yuv420pFrame` (planar 8-bit family) and
// `Nv12Frame` (semi-planar 8-bit family) — cross-cutting between
// the per-family submodules above.

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
