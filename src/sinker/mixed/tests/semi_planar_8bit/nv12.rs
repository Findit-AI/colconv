use super::*;

// ---- NV12 ---------------------------------------------------------------

pub(super) fn solid_nv12_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let ch = h / 2;
  // UV row payload = `width` bytes = `width/2` interleaved UV pairs.
  let mut uv = std::vec![0u8; w * ch];
  for row in 0..ch {
    for i in 0..w / 2 {
      uv[row * w + i * 2] = u;
      uv[row * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 128, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 200, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 200));
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_with_simd_false_matches_with_simd_true() {
  // 32×16 pseudo-random frame so the SIMD path exercises its main
  // loop and the scalar path processes the full width too.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uvp: Vec<u8> = (0..w * h / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut sink_simd = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  let mut sink_scalar = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
  nv12_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar);
}

// ---- preflight buffer-size errors ------------------------------------
//
// Undersized RGB / luma / HSV buffers must be rejected at attachment
// time, not part-way through processing. Catching the mistake before
// any rows are written avoids partially-mutated caller buffers
// flagged by the adversarial review. With the fallible API these
// surface as `Err(MixedSinkerError::*BufferTooShort)` / `HsvPlaneTooShort`.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_rgb_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3 - 1]; // 1 byte short
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RgbBufferTooShort {
      expected: 16 * 8 * 3,
      actual: 16 * 8 * 3 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_luma_returns_err() {
  let mut luma = std::vec![0u8; 16 * 8 - 1];
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::LumaBufferTooShort {
      expected: 16 * 8,
      actual: 16 * 8 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn attach_short_hsv_returns_err() {
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8 - 1]; // V plane short
  let err = MixedSinker::<Yuv420p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::HsvPlaneTooShort {
      which: HsvPlane::V,
      expected: 16 * 8,
      actual: 16 * 8 - 1,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn taller_frame_returns_err_before_any_row_written() {
  // Sink sized for 16×8, feed a 16×10 frame. `begin_frame` returns
  // `Err(DimensionMismatch)` before row 0 — no partial writes.
  let (yp, up, vp) = solid_yuv420p_frame(16, 10, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 10, 16, 8, 8);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::DimensionMismatch {
      configured_w: 16,
      configured_h: 8,
      frame_w: 16,
      frame_h: 10,
    }
  );
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn shorter_frame_returns_err_before_any_row_written() {
  // Sink sized 16×8, frame is 16×4. Without the `begin_frame`
  // preflight, the walker would silently process 4 rows and leave
  // rows 4..7 stale from the previous frame. Preflight returns
  // `Err(DimensionMismatch)` with no side effects.
  let (yp, up, vp) = solid_yuv420p_frame(16, 4, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 4, 16, 8, 8);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::DimensionMismatch {
      configured_w: 16,
      configured_h: 8,
      frame_w: 16,
      frame_h: 4,
    }
  );
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_width_mismatch_returns_err() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 32 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(32, 8).with_rgb(&mut rgb).unwrap();
  let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(
    matches!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 32,
        frame_w: 16,
        ..
      }
    ),
    "unexpected error variant: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_width_mismatch_returns_err() {
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 32 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(32, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(
    matches!(
      err,
      MixedSinkerError::DimensionMismatch {
        configured_w: 32,
        frame_w: 16,
        ..
      }
    ),
    "unexpected error variant: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_shorter_frame_returns_err_before_any_row_written() {
  let (yp, uvp) = solid_nv12_frame(16, 4, 42, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 4, 16, 16);

  const SENTINEL: u8 = 0xEE;
  let mut luma = std::vec![SENTINEL; 16 * 8];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  let err = nv12_to(&src, true, ColorMatrix::Bt601, &mut sink)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "no rows should have been written before the Err"
  );
}

/// Sanity check that an Infallible sink (compile-time proof of
/// no-error) compiles and runs. Mirrors the trait-docs pattern.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn infallible_sink_compiles_and_runs() {
  use core::convert::Infallible;

  struct RowCounter(usize);
  impl PixelSink for RowCounter {
    type Input<'a> = Yuv420pRow<'a>;
    type Error = Infallible;
    fn process(&mut self, _row: Yuv420pRow<'_>) -> Result<(), Infallible> {
      self.0 += 1;
      Ok(())
    }
  }
  impl Yuv420pSink for RowCounter {}

  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
  let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let mut counter = RowCounter(0);
  // `Result<(), Infallible>` — the compiler knows Err is
  // uninhabited, so `.unwrap()` here is free and infallible.
  yuv420p_to(&src, true, ColorMatrix::Bt601, &mut counter).unwrap();
  assert_eq!(counter.0, 8);
}

// ---- direct process() bypass paths ----------------------------------
//
// The walker normally guarantees (a) begin_frame runs first and
// validates frame dimensions, (b) row.y()/u/v/uv slices have the
// right length, (c) `idx < height`. A direct `process` call can
// break any of these. The defense-in-depth checks in `process`
// must return a specific error variant, not panic — verified here
// by constructing rows manually and calling `process`.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_short_y_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  // Build a row with a 15-byte Y slice (wrong — sink configured for 16).
  let y = [0u8; 15];
  let u = [128u8; 8];
  let v = [128u8; 8];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Y,
      row: 0,
      expected: 16,
      actual: 15,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_short_u_half() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 16];
  let u = [128u8; 7]; // expected 8
  let v = [128u8; 8];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::UHalf,
      row: 0,
      expected: 8,
      actual: 7,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 16];
  let u = [128u8; 8];
  let v = [128u8; 8];
  // idx = 8 exceeds configured height 8 — would otherwise panic on
  // `rgb[idx * w * 3 ..]` indexing.
  let row = Yuv420pRow::new(&y, &u, &v, 8, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_odd_width_sink_returns_err_at_begin_frame() {
  // A sink configured with an odd width would later panic inside
  // `yuv_420_to_rgb_row` (which asserts `width & 1 == 0`). The
  // fallible API surfaces this as `OddWidth` at frame start — no
  // rows are processed, no panic. Width=15, height=8 — matching
  // frame so `DimensionMismatch` can't fire first.
  let w = 15usize;
  let h = 8usize;
  let y = std::vec![0u8; w * h];
  let u = std::vec![128u8; w.div_ceil(2) * h / 2 + 8]; // any valid size
  let v = std::vec![128u8; w.div_ceil(2) * h / 2 + 8];
  // Build the Frame separately — Yuv420pFrame rejects odd width
  // too, so we can't construct a 15-wide frame. That's fine: we
  // only need to hit `begin_frame`, which takes (width, height)
  // parameters directly. Call it manually.
  let mut rgb = std::vec![0u8; 16 * 8 * 3]; // Dummy; not touched.
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
  // Silence unused-vec warnings — these would have been the plane data.
  let _ = (y, u, v);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_odd_width_sink_returns_err_at_direct_process() {
  // Direct `process` caller bypassing `begin_frame`. Process must
  // still reject odd width before calling the kernel.
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(15, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let y = [0u8; 15];
  let u = [128u8; 7]; // ceil(15/2) = 8; 7 triggers the width check first
  let v = [128u8; 7];
  let row = Yuv420pRow::new(&y, &u, &v, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_odd_width_sink_returns_err_at_begin_frame() {
  let w = 15usize;
  let h = 8usize;
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(w, h).with_rgb(&mut rgb).unwrap();
  let err = sink.begin_frame(w as u32, h as u32).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_odd_width_sink_returns_err_at_direct_process() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(15, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 15];
  let uv = [128u8; 15];
  let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 15 });
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_process_rejects_short_uv_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 15]; // expected 16
  let row = Nv12Row::new(&y, &uv, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::UvHalf,
      row: 0,
      expected: 16,
      actual: 15,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv12>::new(16, 8).with_rgb(&mut rgb).unwrap();
  let y = [0u8; 16];
  let uv = [128u8; 16];
  let row = Nv12Row::new(&y, &uv, 8, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_matches_yuv420p_mixed_sinker() {
  // Cross-format guarantee: an NV12 frame built from the same U / V
  // bytes as a Yuv420p frame produces byte-identical RGB output via
  // MixedSinker on both families.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let up: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let vp: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  // Build NV12 UV plane: chroma row r, column c → uv[r * w + 2*c] = U,
  // uv[r * w + 2*c + 1] = V, where U / V come from the same (r, c)
  // sample of the planar fixture above.
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = up[r * (ws / 2) + c];
      uvp[r * ws + 2 * c + 1] = vp[r * (ws / 2) + c];
    }
  }

  let yuv420p_src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);

  let mut rgb_yuv420p = std::vec![0u8; ws * hs * 3];
  let mut rgb_nv12 = std::vec![0u8; ws * hs * 3];
  let mut s_yuv = MixedSinker::<Yuv420p>::new(ws, hs)
    .with_rgb(&mut rgb_yuv420p)
    .unwrap();
  let mut s_nv = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgb(&mut rgb_nv12)
    .unwrap();
  yuv420p_to(&yuv420p_src, false, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv).unwrap();

  assert_eq!(rgb_yuv420p, rgb_nv12);
}

// ---- NV12 RGBA (Ship 8 PR 2) tests --------------------------------------
//
// Mirrors the Yuv420p RGBA test set. Adds a cross-format invariant
// proving NV12 RGBA is byte-identical to Yuv420p RGBA when fed the
// same pixels — catches U/V swap bugs in the new RGBA path that
// a pure RGB-path test would miss.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv12_frame(16, 8, 128, 128, 128);
  let src = Nv12Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32usize;
  let h = 16usize;
  let (yp, uvp) = solid_nv12_frame(w as u32, h as u32, 180, 60, 200);
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv12_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv12>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgba_simd_matches_scalar_with_random_yuv() {
  // Pseudo-random per-pixel YUV across all 4 matrices × both
  // ranges. Width 1922 forces both the SIMD main loop AND a scalar
  // tail across every backend block size (16 / 32 / 64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; w * (h / 2)];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv12Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv12>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv12_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv12>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv12_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "NV12 RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rgba_matches_yuv420p_rgba_with_same_pixels() {
  // Cross-format invariant: NV12 RGBA byte-identical to Yuv420p
  // RGBA when the chroma is the same. Mirrors the existing
  // `nv12_matches_yuv420p_mixed_sinker` RGB-path test for the new
  // RGBA path. Catches U/V swap bugs in the NV12 RGBA kernel that
  // would silently differ from the planar reference.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let yp: Vec<u8> = (0..ws * hs).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let up: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let vp: Vec<u8> = (0..(ws / 2) * (hs / 2))
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut uvp: Vec<u8> = std::vec![0u8; ws * (hs / 2)];
  for r in 0..hs / 2 {
    for c in 0..ws / 2 {
      uvp[r * ws + 2 * c] = up[r * (ws / 2) + c];
      uvp[r * ws + 2 * c + 1] = vp[r * (ws / 2) + c];
    }
  }

  let yuv420p_src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
  let nv12_src = Nv12Frame::new(&yp, &uvp, w, h, w, w);

  let mut rgba_yuv420p = std::vec![0u8; ws * hs * 4];
  let mut sink_yuv420p = MixedSinker::<Yuv420p>::new(ws, hs)
    .with_rgba(&mut rgba_yuv420p)
    .unwrap();
  yuv420p_to(&yuv420p_src, true, ColorMatrix::Bt709, &mut sink_yuv420p).unwrap();

  let mut rgba_nv12 = std::vec![0u8; ws * hs * 4];
  let mut sink_nv12 = MixedSinker::<Nv12>::new(ws, hs)
    .with_rgba(&mut rgba_nv12)
    .unwrap();
  nv12_to(&nv12_src, true, ColorMatrix::Bt709, &mut sink_nv12).unwrap();

  assert_eq!(rgba_yuv420p, rgba_nv12);
}
