//! Fused-downscale coverage for packed YUV 4:1:1 (`Uyyvyy411`,
//! `AV_PIX_FMT_UYYVYY411`, DV legacy) — routed through the packed
//! dual-stream resample: the **Y samples** are de-interleaved out of the
//! packed plane and area-resampled directly for luma (the YUV luma
//! contract), while RGB / RGBA / HSV bin a source-width RGB row produced
//! by the format's own fused `uyyvyy411_to_rgb_row` kernel (chroma
//! de-interleave + 4:1:1 horizontal upsample in registers). So RGB
//! equals an area-resample of the identity-converted frame, and luma
//! equals the area-downscaled Y plane — *not* RGB-derived luma. The
//! latter is pinned under saturated chroma, where converting Y/U/V to
//! RGB and back to luma would clip far away from the true Y.
//!
//! The colour oracle deliberately avoids importing `Rgb24` (the engine
//! is gated to `yuv-packed`, which does not imply `rgb`): the expected
//! RGB is built by running the *direct* `Uyyvyy411` sink to a
//! source-width RGB frame and 2x2-block-mean-ing it here, which — for an
//! exact 2:1 integer ratio — is byte-identical to the engine's
//! convert-then-bin output.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  row::rgb_to_hsv_row,
  sinker::{MixedSinker, MixedSinkerError},
  source::{Uyyvyy411, Uyyvyy411Row, uyyvyy411_to},
};
use mediaframe::frame::Uyyvyy411Frame;

use super::force_row_stage;

const SRC_W: usize = 8;
const SRC_H: usize = 8;
const OUT_W: usize = 4;
const OUT_H: usize = 4;

/// Builds a UYYVYY411 packed plane from per-pixel Y and per-block (U, V)
/// closures. Layout per 6-byte / 4-pixel block: `U0, Y0, Y1, V0, Y2, Y3`
/// (4:1:1 — one chroma pair per 4 luma). Stride equals `width * 3 / 2`
/// (no padding). Width must be a multiple of 4.
fn build_uyyvyy411(
  width: usize,
  height: usize,
  y_at: impl Fn(usize, usize) -> u8,
  u_at: impl Fn(usize, usize) -> u8,
  v_at: impl Fn(usize, usize) -> u8,
) -> Vec<u8> {
  assert_eq!(width & 3, 0, "uyyvyy411 width must be multiple of 4");
  let mut buf = std::vec![0u8; width * 3 / 2 * height];
  for row in 0..height {
    let base = row * width * 3 / 2;
    for col in (0..width).step_by(4) {
      let blk = base + (col / 4) * 6;
      buf[blk] = u_at(col, row);
      buf[blk + 1] = y_at(col, row);
      buf[blk + 2] = y_at(col + 1, row);
      buf[blk + 3] = v_at(col, row);
      buf[blk + 4] = y_at(col + 2, row);
      buf[blk + 5] = y_at(col + 3, row);
    }
  }
  buf
}

/// De-interleaves the Y plane out of a UYYVYY411 packed frame (the
/// luma-from-Y reference, independent of the kernel under test).
fn deinterleave_y(packed: &[u8], width: usize, height: usize) -> Vec<u8> {
  let mut y = std::vec![0u8; width * height];
  for row in 0..height {
    let base = row * width * 3 / 2;
    for col in (0..width).step_by(4) {
      let blk = base + (col / 4) * 6;
      let o = row * width + col;
      y[o] = packed[blk + 1];
      y[o + 1] = packed[blk + 2];
      y[o + 2] = packed[blk + 4];
      y[o + 3] = packed[blk + 5];
    }
  }
  y
}

/// Exact 2x2-block area mean (round-half-up) of a single-channel
/// `SRC`-grid plane to the `OUT` grid — the integer-ratio (2:1)
/// area-downscale reference.
fn block_mean_2x2_plane(plane: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT_W * OUT_H];
  for oy in 0..OUT_H {
    for ox in 0..OUT_W {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC_W + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT_W + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

/// Exact 2x2-block area mean (round-half-up) of a 3-channel interleaved
/// `SRC`-grid RGB image to the `OUT` grid — the colour reference that
/// mirrors the engine's convert-then-bin for an exact 2:1 ratio.
fn block_mean_2x2_rgb(rgb: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT_W * OUT_H * 3];
  for oy in 0..OUT_H {
    for ox in 0..OUT_W {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC_W + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT_W + ox) * 3 + c] = ((s + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Interior ramps across Y / U / V so the chroma upsample and the area
/// mean both see real variation.
fn ramp_frame() -> Vec<u8> {
  build_uyyvyy411(
    SRC_W,
    SRC_H,
    |x, y| 32 + ((x + y * SRC_W) % 64) as u8,
    |x, y| 100 + ((x / 4 + y) % 40) as u8,
    |x, y| 140 - ((x / 4 + y) % 40) as u8,
  )
}

/// Direct conversion of a UYYVYY411 packed frame to a source-width RGB
/// frame via the *direct* `Uyyvyy411` sink (no resampler) — the
/// pre-binning input for the colour oracle.
fn direct_rgb(packed: &[u8]) -> Vec<u8> {
  let src = Uyyvyy411Frame::new(packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);
  let mut rgb = std::vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Uyyvyy411>::new(SRC_W, SRC_H)
      .with_rgb(&mut rgb)
      .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_resample_rgb_matches_binned_direct_conversion() {
  let packed = ramp_frame();
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);

  let mut rgb_a = std::vec![0u8; OUT_W * OUT_H * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
        SRC_W,
        SRC_H,
        AreaResampler::to(OUT_W, OUT_H),
      )
      .unwrap(),
    )
    .with_rgb(&mut rgb_a)
    .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }

  let expected = block_mean_2x2_rgb(&direct_rgb(&packed));
  assert_eq!(
    rgb_a, expected,
    "rgb: row-stage resample must equal block-mean of the direct conversion"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_resample_luma_is_area_downscaled_y_plane() {
  let packed = ramp_frame();
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);

  let (mut luma, mut luma_u16) = (
    std::vec![0u8; OUT_W * OUT_H],
    std::vec![0u16; OUT_W * OUT_H],
  );
  {
    let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let y_ref = block_mean_2x2_plane(&deinterleave_y(&packed, SRC_W, SRC_H));
  assert_eq!(luma, y_ref, "luma must be the area-downscaled Y plane");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(
    luma_u16, y_ref_u16,
    "luma_u16 must be the area-downscaled Y, zero-extended"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_resample_luma_comes_from_y_not_rgb_under_saturated_chroma() {
  // Uniform low Y with saturated chroma: converting to RGB and back to
  // luma would clip far above the true Y. The resampled luma must be
  // the (uniform) Y mean, proving it area-resamples the Y plane.
  let packed = build_uyyvyy411(SRC_W, SRC_H, |_, _| 16, |_, _| 240, |_, _| 16);
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);

  let mut luma = std::vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must area-resample the Y plane (16), not RGB-derived luma; got {luma:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_resample_all_outputs_combo() {
  // RGB, RGBA, luma, luma_u16 and HSV all attached at once: each must be
  // populated consistently (RGBA = RGB + 0xFF alpha; luma from Y; HSV
  // derived from the binned RGB).
  let packed = ramp_frame();
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);

  let mut rgb = std::vec![0u8; OUT_W * OUT_H * 3];
  let mut rgba = std::vec![0u8; OUT_W * OUT_H * 4];
  let mut luma = std::vec![0u8; OUT_W * OUT_H];
  let mut luma_u16 = std::vec![0u16; OUT_W * OUT_H];
  let mut hh = std::vec![0u8; OUT_W * OUT_H];
  let mut ss = std::vec![0u8; OUT_W * OUT_H];
  let mut vv = std::vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
        SRC_W,
        SRC_H,
        AreaResampler::to(OUT_W, OUT_H),
      )
      .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }

  let expected_rgb = block_mean_2x2_rgb(&direct_rgb(&packed));
  assert_eq!(rgb, expected_rgb, "rgb");
  for i in 0..(OUT_W * OUT_H) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "RGBA R at {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "RGBA G at {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "RGBA B at {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "RGBA alpha at {i}");
  }
  let y_ref = block_mean_2x2_plane(&deinterleave_y(&packed, SRC_W, SRC_H));
  assert_eq!(luma, y_ref, "luma from Y");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(luma_u16, y_ref_u16, "luma_u16 from Y");
  // HSV must match an independent rgb_to_hsv of the binned RGB row.
  let mut h_ref = std::vec![0u8; OUT_W * OUT_H];
  let mut s_ref = std::vec![0u8; OUT_W * OUT_H];
  let mut v_ref = std::vec![0u8; OUT_W * OUT_H];
  for oy in 0..OUT_H {
    rgb_to_hsv_row(
      &expected_rgb[oy * OUT_W * 3..(oy + 1) * OUT_W * 3],
      &mut h_ref[oy * OUT_W..(oy + 1) * OUT_W],
      &mut s_ref[oy * OUT_W..(oy + 1) * OUT_W],
      &mut v_ref[oy * OUT_W..(oy + 1) * OUT_W],
      OUT_W,
      true,
    );
  }
  assert_eq!(hh, h_ref, "hsv H");
  assert_eq!(ss, s_ref, "hsv S");
  assert_eq!(vv, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_identity_plan_matches_new_sink() {
  let packed = ramp_frame();
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);

  let direct = direct_rgb(&packed);
  let mut via_area = std::vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(SRC_W, SRC_H),
    )
    .unwrap()
    .with_rgb(&mut via_area)
    .unwrap();
    uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn uyyvyy411_resample_no_outputs_is_a_no_op() {
  let packed = ramp_frame();
  let src = Uyyvyy411Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);
  let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_resample_reuses_streams_across_frames() {
  // A reused sink must reset both the Y-plane luma stream and the RGB
  // stream each frame; without the reset, frame 2's row 0 is rejected as
  // out-of-sequence and the outputs never reflect frame 2.
  let p1 = ramp_frame();
  let mut p2 = p1.clone();
  for b in p2.iter_mut() {
    *b = 255 - *b;
  }
  let (mut luma, mut rgb) = (
    std::vec![0u8; OUT_W * OUT_H],
    std::vec![0u8; OUT_W * OUT_H * 3],
  );
  {
    let mut sink = force_row_stage(
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
        SRC_W,
        SRC_H,
        AreaResampler::to(OUT_W, OUT_H),
      )
      .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap();
    let f1 = Uyyvyy411Frame::new(&p1, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);
    let f2 = Uyyvyy411Frame::new(&p2, SRC_W as u32, SRC_H as u32, (SRC_W * 3 / 2) as u32);
    uyyvyy411_to(&f1, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    uyyvyy411_to(&f2, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let y2_ref = block_mean_2x2_plane(&deinterleave_y(&p2, SRC_W, SRC_H));
  assert_eq!(luma, y2_ref, "frame 2 luma must area-downscale frame 2's Y");
  let rgb2_ref = block_mean_2x2_rgb(&direct_rgb(&p2));
  assert_eq!(rgb, rgb2_ref, "frame 2 rgb must bin frame 2's conversion");
}

#[test]
fn uyyvyy411_resample_rejects_out_of_sequence_rows() {
  let packed = ramp_frame();
  let mut rgb = std::vec![0u8; OUT_W * OUT_H * 3];
  let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  // Feed row 2 first — the stream expects strict sequencing from 0.
  let row_bytes = SRC_W * 3 / 2;
  let row2 = Uyyvyy411Row::for_tests(
    &packed[row_bytes * 2..row_bytes * 3],
    2,
    KernelMatrix::Bt601,
    true,
  );
  let err = sink.process(row2).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}

#[test]
fn uyyvyy411_resample_rejects_changed_output_set_midframe() {
  // FrozenOutputs: the output configuration is frozen on row 0; a
  // mid-frame change (here: detaching RGB by swapping in a fresh sink is
  // not possible, so instead attach RGB then feed a row, then a second
  // process with a different output footprint via direct row calls).
  let packed = ramp_frame();
  let mut rgb = std::vec![0u8; OUT_W * OUT_H * 3];
  let mut luma = std::vec![0u8; OUT_W * OUT_H];
  let row_bytes = SRC_W * 3 / 2;
  let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  // Row 0 freezes the output set (RGB only).
  sink
    .process(Uyyvyy411Row::for_tests(
      &packed[0..row_bytes],
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap();
  // Attach an extra output mid-frame, then feed row 1 — the snapshot
  // must differ and be rejected.
  sink.set_luma(&mut luma).unwrap();
  let err = sink
    .process(Uyyvyy411Row::for_tests(
      &packed[row_bytes..row_bytes * 2],
      1,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "got {err:?}"
  );
}

#[test]
fn uyyvyy411_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after attaching a NEW output succeeds
  // instead of tripping ResampleOutputsChanged.
  let packed = ramp_frame();
  let row_bytes = SRC_W * 3 / 2;
  let mut luma = std::vec![0u8; OUT_W * OUT_H];
  let mut sink = MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_luma(&mut luma)
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  let err = sink
    .process(Uyyvyy411Row::for_tests(
      &packed[row_bytes * 3..row_bytes * 4],
      3,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb = std::vec![0u8; OUT_W * OUT_H * 3];
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(Uyyvyy411Row::for_tests(
      &packed[0..row_bytes],
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}
