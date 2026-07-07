use super::*;

#[test]
fn noop_plan_is_identity() {
  assert_eq!(NoopResampler.plan(1920, 1080), Ok(None));
}

#[test]
fn area_identity_plans_none() {
  assert_eq!(AreaResampler::to(8, 8).plan(8, 8), Ok(None));
}

#[test]
fn area_zero_output_rejected() {
  let err = AreaResampler::to(4, 0).plan(8, 8).unwrap_err();
  match err {
    ResampleError::ZeroOutputDimension(e) => {
      assert_eq!(e.out_w(), 4);
      assert_eq!(e.out_h(), 0);
    }
    other => panic!("expected ZeroOutputDimension, got {other:?}"),
  }
}

#[test]
fn area_upscale_rejected_per_axis() {
  // Width within bounds, height exceeding: either axis alone trips it.
  let err = AreaResampler::to(4, 16).plan(8, 8).unwrap_err();
  match err {
    ResampleError::UpscaleUnsupported(e) => {
      assert_eq!((e.src_w(), e.src_h()), (8, 8));
      assert_eq!((e.out_w(), e.out_h()), (4, 16));
    }
    other => panic!("expected UpscaleUnsupported, got {other:?}"),
  }
}

#[test]
fn area_plan_reports_geometry() {
  let plan = AreaResampler::to(336, 189)
    .plan(1920, 1080)
    .expect("valid downscale")
    .expect("non-identity");
  assert_eq!(plan.out_w(), 336);
  assert_eq!(plan.out_h(), 189);
  assert_eq!(plan.out_dims(), (336, 189));
  assert_eq!((plan.src_w(), plan.src_h()), (1920, 1080));
}

#[test]
fn area_integer_ratio_spans() {
  // 8 -> 4 on both axes: every output pixel covers exactly two source
  // cells with equal weight (cell width = out on the scaled grid).
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  for axis in [plan.h(), plan.v()] {
    assert_eq!(axis.out_len(), 4);
    for j in 0..4 {
      let (start, weights) = axis.span(j);
      assert_eq!(start, 2 * j);
      assert_eq!(weights, &[4, 4]);
    }
  }
}

#[test]
fn area_fractional_spans_8_to_3() {
  // Scale 8/3: out pixel j covers [8j, 8j+8) on the x3 grid, source
  // cell i covers [3i, 3i+3). Overlaps hand-derived; each span sums to
  // the denominator (the source dimension, 8).
  let plan = AreaResampler::to(3, 8)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");

  let h = plan.h();
  assert_eq!(h.out_len(), 3);
  assert_eq!(h.span(0), (0, &[3usize, 3, 2][..]));
  assert_eq!(h.span(1), (2, &[1usize, 3, 3, 1][..]));
  assert_eq!(h.span(2), (5, &[2usize, 3, 3][..]));

  // Vertical axis is identity-sized (8 -> 8): unit spans, full weight.
  let v = plan.v();
  assert_eq!(v.out_len(), 8);
  for j in 0..8 {
    assert_eq!(v.span(j), (j, &[8usize][..]));
  }
}

#[test]
fn area_spans_handle_the_upsample_direction() {
  // The coverage math is direction-agnostic; the native tier feeds
  // chroma grids through it (8x8 -> 6x6 frame means 4 -> 6 chroma).
  // Hand-derived: out pixel intervals of length 4 on the x6 grid.
  let spans = AxisSpans::area(4, 6).expect("upsample coverage is valid");
  assert_eq!(spans.out_len(), 6);
  let expected: [(usize, &[usize]); 6] = [
    (0, &[4]),
    (0, &[2, 2]),
    (1, &[4]),
    (2, &[4]),
    (2, &[2, 2]),
    (3, &[4]),
  ];
  for (j, &(start, weights)) in expected.iter().enumerate() {
    assert_eq!(spans.span(j), (start, weights), "span {j}");
  }
}

#[test]
fn area_spans_partition_the_source() {
  // The NaFlex case: 1920x1080 -> 336x189 (x40/7 per axis). Every span
  // sums to the source dimension, starts strictly increase, tap counts
  // stay within ceil(scale) + 1, and coverage ends exactly at the last
  // source cell.
  let plan = AreaResampler::to(336, 189)
    .plan(1920, 1080)
    .expect("valid")
    .expect("non-identity");

  for (axis, src, out) in [(plan.h(), 1920, 336), (plan.v(), 1080, 189)] {
    assert_eq!(axis.out_len(), out);
    let mut prev_start = None;
    for j in 0..out {
      let (start, weights) = axis.span(j);
      assert_eq!(weights.iter().sum::<usize>(), src, "span {j} sum");
      assert!(weights.iter().all(|&w| w > 0), "span {j} zero tap");
      assert!(weights.len() == 6 || weights.len() == 7, "span {j} taps");
      if let Some(p) = prev_start {
        assert!(start > p, "span {j} start not increasing");
      }
      prev_start = Some(start);
    }
    let (last_start, last_weights) = axis.span(out - 1);
    assert_eq!(last_start + last_weights.len(), src);
    assert_eq!(axis.span(0).0, 0);
  }
}

/// Direct (non-separable) 2-D area reference: per output pixel,
/// sum `w_y * w_x * sample` over the full source block via the plan's
/// own spans, then round-half-up by the `src_w * src_h` denominator.
/// The streaming engine must reproduce this exactly.
#[cfg(feature = "yuv-planar")]
fn direct_area_2d(plan: &ResamplePlan, src: &[u8], channels: usize) -> std::vec::Vec<u8> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let denom = (src_w as u64) * (plan.src_h() as u64);
  let mut out = std::vec![0u8; out_w * out_h * channels];
  for oy in 0..out_h {
    let (vy, vw) = plan.v().span(oy);
    for ox in 0..out_w {
      let (hx, hw) = plan.h().span(ox);
      for c in 0..channels {
        let mut acc = 0u64;
        for (dy, &wy) in vw.iter().enumerate() {
          for (dx, &wx) in hw.iter().enumerate() {
            let s = src[((vy + dy) * src_w + hx + dx) * channels + c] as u64;
            acc += (wy as u64) * (wx as u64) * s;
          }
        }
        out[(oy * out_w + ox) * channels + c] = ((acc + denom / 2) / denom) as u8;
      }
    }
  }
  out
}

#[cfg(feature = "yuv-planar")]
fn stream_collect(plan: &ResamplePlan, src: &[u8], channels: usize) -> std::vec::Vec<u8> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let mut stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("realistic geometry");
  let mut out = std::vec![0u8; out_w * out_h * channels];
  let mut emitted = std::vec::Vec::new();
  for y in 0..plan.src_h() {
    let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
    stream
      .feed_row(y, row, true, |oy, finalized| {
        emitted.push(oy);
        out[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
      })
      .expect("rows arrive in order");
  }
  assert_eq!(emitted, (0..out_h).collect::<std::vec::Vec<_>>());
  out
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_matches_direct_2d_reference_fractional() {
  let plan = AreaResampler::to(3, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<u8> = (0..64u8).collect();
  assert_eq!(
    stream_collect(&plan, &src, 1),
    direct_area_2d(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_matches_direct_2d_reference_multichannel() {
  let plan = AreaResampler::to(4, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // 3 interleaved channels with distinct ramps.
  let mut src = std::vec![0u8; 8 * 8 * 3];
  for (i, px) in src.chunks_exact_mut(3).enumerate() {
    px[0] = i as u8;
    px[1] = (3 * i % 251) as u8;
    px[2] = 255 - i as u8;
  }
  assert_eq!(
    stream_collect(&plan, &src, 3),
    direct_area_2d(&plan, &src, 3)
  );
}

/// `u16` analogue of [`direct_area_2d`]: exact 2D area mean over the
/// full 16-bit sample range, so the reference exercises the wider
/// (`u64`) horizontal accumulator the `u8` path never reaches.
#[cfg(feature = "yuv-planar")]
fn direct_area_2d_u16(plan: &ResamplePlan, src: &[u16], channels: usize) -> std::vec::Vec<u16> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let denom = (src_w as u64) * (plan.src_h() as u64);
  let mut out = std::vec![0u16; out_w * out_h * channels];
  for oy in 0..out_h {
    let (vy, vw) = plan.v().span(oy);
    for ox in 0..out_w {
      let (hx, hw) = plan.h().span(ox);
      for c in 0..channels {
        let mut acc = 0u64;
        for (dy, &wy) in vw.iter().enumerate() {
          for (dx, &wx) in hw.iter().enumerate() {
            let s = src[((vy + dy) * src_w + hx + dx) * channels + c] as u64;
            acc += (wy as u64) * (wx as u64) * s;
          }
        }
        out[(oy * out_w + ox) * channels + c] = ((acc + denom / 2) / denom) as u16;
      }
    }
  }
  out
}

#[cfg(feature = "yuv-planar")]
fn stream_collect_u16(plan: &ResamplePlan, src: &[u16], channels: usize) -> std::vec::Vec<u16> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let mut stream = AreaStream::<u16>::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("realistic geometry");
  let mut out = std::vec![0u16; out_w * out_h * channels];
  let mut emitted = std::vec::Vec::new();
  for y in 0..plan.src_h() {
    let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
    stream
      .feed_row(y, row, true, |oy, finalized| {
        emitted.push(oy);
        out[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
      })
      .expect("rows arrive in order");
  }
  assert_eq!(emitted, (0..out_h).collect::<std::vec::Vec<_>>());
  out
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_matches_direct_2d_reference_fractional() {
  let plan = AreaResampler::to(3, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // Full-range ramp 0, 1040, …, 65520 — samples above 255 prove the
  // u16 horizontal accumulator carries the high bits a u8 path drops.
  let src: std::vec::Vec<u16> = (0..64u16).map(|i| i * 1040).collect();
  assert_eq!(
    stream_collect_u16(&plan, &src, 1),
    direct_area_2d_u16(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_matches_direct_2d_reference_multichannel() {
  let plan = AreaResampler::to(4, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // 3 interleaved channels with distinct full-range ramps.
  let mut src = std::vec![0u16; 8 * 8 * 3];
  for (i, px) in src.chunks_exact_mut(3).enumerate() {
    px[0] = (i as u16) * 1000;
    px[1] = ((7 * i) % 211) as u16 * 300;
    px[2] = 65535 - (i as u16) * 1000;
  }
  assert_eq!(
    stream_collect_u16(&plan, &src, 3),
    direct_area_2d_u16(&plan, &src, 3)
  );
}

/// Integer-ratio 2× downscale: every output pixel is an exact
/// round-half-up mean of its 2×2 source block. Hand-computed
/// expectations pin the u16 finalize independent of the reference.
#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_exact_2x2_block_mean() {
  let plan = AreaResampler::to(2, 2)
    .plan(4, 4)
    .expect("valid")
    .expect("non-identity");
  // 4×4 source; each 2×2 quadrant chosen so its mean rounds half-up.
  let src: std::vec::Vec<u16> = std::vec![
    60000, 60002, 10, 11, //
    60004, 60006, 12, 13, //
    1, 2, 65535, 65533, //
    3, 5, 65531, 65529, //
  ];
  // Quadrant means: (60000+60002+60004+60006)/4 = 60003;
  // (10+11+12+13)/4 = 11.5 -> 12; (1+2+3+5)/4 = 2.75 -> 3;
  // (65535+65533+65531+65529)/4 = 65532.
  assert_eq!(
    stream_collect_u16(&plan, &src, 1),
    std::vec![60003u16, 12, 3, 65532]
  );
}

/// Exact f64 2-D area mean cast to f32 — the ground-truth float
/// reference. The streaming f32 engine accumulates in f32 (H then V),
/// whose adds reorder vs this 2-D sum, so it matches only within a
/// small tolerance, not bit-for-bit.
#[cfg(feature = "yuv-planar")]
fn direct_area_2d_f32(plan: &ResamplePlan, src: &[f32], channels: usize) -> std::vec::Vec<f32> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let denom = (src_w as f64) * (plan.src_h() as f64);
  let mut out = std::vec![0f32; out_w * out_h * channels];
  for oy in 0..out_h {
    let (vy, vw) = plan.v().span(oy);
    for ox in 0..out_w {
      let (hx, hw) = plan.h().span(ox);
      for c in 0..channels {
        let mut acc = 0f64;
        for (dy, &wy) in vw.iter().enumerate() {
          for (dx, &wx) in hw.iter().enumerate() {
            let s = src[((vy + dy) * src_w + hx + dx) * channels + c] as f64;
            acc += (wy as f64) * (wx as f64) * s;
          }
        }
        out[(oy * out_w + ox) * channels + c] = (acc / denom) as f32;
      }
    }
  }
  out
}

#[cfg(feature = "yuv-planar")]
fn stream_collect_f32(plan: &ResamplePlan, src: &[f32], channels: usize) -> std::vec::Vec<f32> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let mut stream = AreaStream::<f32>::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("realistic geometry");
  let mut out = std::vec![0f32; out_w * out_h * channels];
  let mut emitted = std::vec::Vec::new();
  for y in 0..plan.src_h() {
    let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
    stream
      .feed_row(y, row, false, |oy, finalized| {
        emitted.push(oy);
        out[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
      })
      .expect("rows arrive in order");
  }
  assert_eq!(emitted, (0..out_h).collect::<std::vec::Vec<_>>());
  out
}

#[cfg(feature = "yuv-planar")]
fn assert_f32_close(a: &[f32], b: &[f32], tol: f32) {
  assert_eq!(a.len(), b.len(), "length");
  for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
    assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (tol {tol})");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_f32_matches_direct_2d_reference_fractional() {
  let plan = AreaResampler::to(3, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
  assert_f32_close(
    &stream_collect_f32(&plan, &src, 1),
    &direct_area_2d_f32(&plan, &src, 1),
    1e-4,
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_f32_matches_direct_2d_reference_multichannel() {
  let plan = AreaResampler::to(4, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let mut src = std::vec![0f32; 8 * 8 * 3];
  for (i, px) in src.chunks_exact_mut(3).enumerate() {
    px[0] = i as f32 * 0.5;
    px[1] = 100.0 - i as f32 * 0.25;
    px[2] = ((7 * i) % 211) as f32;
  }
  assert_f32_close(
    &stream_collect_f32(&plan, &src, 3),
    &direct_area_2d_f32(&plan, &src, 3),
    1e-2,
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_f32_exact_2x2_block_mean() {
  // Integer-ratio 2x downscale: each output is the plain mean of its
  // 2x2 block (no round-half-up for float). All values and the divide
  // by a power-of-two denominator are exact in f32.
  let plan = AreaResampler::to(2, 2)
    .plan(4, 4)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<f32> = std::vec![
    1.0, 2.0, 10.0, 12.0, //
    3.0, 4.0, 14.0, 16.0, //
    0.5, 1.5, 100.0, 102.0, //
    2.5, 3.5, 104.0, 106.0, //
  ];
  // Quadrant means: (1+2+3+4)/4=2.5; (10+12+14+16)/4=13;
  // (0.5+1.5+2.5+3.5)/4=2.0; (100+102+104+106)/4=103.
  assert_f32_close(
    &stream_collect_f32(&plan, &src, 1),
    &std::vec![2.5f32, 13.0, 2.0, 103.0],
    1e-3,
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_f32_preserves_large_finite_values_without_overflow() {
  // The stream accumulates the *unnormalized* area numerator
  // (`denom * sample`) before the final divide. Here `denom = 16` and
  // `sample = 1e38`, so the numerator is `1.6e39` — past `f32::MAX`
  // (~3.4e38). An f32 accumulator would overflow to `+inf` and emit
  // `inf`; the f64 accumulators keep it finite, so the constant
  // round-trips. (The same f64 accumulators fix the dual failure — a
  // constant image wider than `2^24` losing unit contributions in an
  // f32 sum — which a direct test would need a >16.7M-element buffer
  // to exercise.)
  let plan = AreaResampler::to(2, 2)
    .plan(4, 4)
    .expect("valid")
    .expect("non-identity");
  let big = 1.0e38f32;
  let src = std::vec![big; 4 * 4];
  for (i, &v) in stream_collect_f32(&plan, &src, 1).iter().enumerate() {
    assert!(v.is_finite(), "output {i} overflowed to {v}");
    assert!((v - big).abs() <= big * 1e-5, "output {i}: {v} vs {big}");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_identity_vertical_axis_emits_every_row() {
  // 8x8 -> 3x8: vertical axis is identity-sized, so every source row
  // finalizes exactly one output row.
  let plan = AreaResampler::to(3, 8)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<u8> = (0..64u8).collect();
  assert_eq!(
    stream_collect(&plan, &src, 1),
    direct_area_2d(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_creation_fails_recoverably_on_huge_row_buffers() {
  // Row buffers that can never be reserved must surface
  // AllocationFailed via capacity overflow — the magnitude has to sit
  // in the CAPACITY-OVERFLOW zone for the SMALLEST arena element
  // (h_tmp is u32, so entries x 4 bytes must exceed isize::MAX);
  // anything smaller is a real near-exabyte allocator request that
  // hosts refuse but sanitizers and miri abort on. Real spans,
  // magnitude driven through the channel count: 4 outputs x
  // usize::MAX / 16 channels is representable as a length but puts
  // every per-element arena past capacity.
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let err = AreaStream::<u8>::new(
    plan.h(),
    plan.v(),
    plan.src_w(),
    plan.src_h(),
    usize::MAX / 16,
  )
  .unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_chroma_420_reports_allocation_failure_as_such() {
  // Allocator refusal on the paired vertical axis must surface as
  // AllocationFailed, not be misclassified as geometry Overflow. The
  // magnitude must sit in the CAPACITY-OVERFLOW zone (entries x 8
  // bytes above isize::MAX), where try_reserve fails deterministically
  // WITHOUT touching the allocator — a smaller huge request would be
  // a real multi-exabyte allocation that hosts refuse but miri aborts
  // on. usize::MAX/8 starts entries overflow capacity while every
  // arithmetic check still passes.
  let err = ResamplePlan::area_chroma_420(4, 8, 2, usize::MAX / 8, 0.0, 0.0, false).unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_halved_weights_the_odd_tail_row_by_its_luma_coverage() {
  // 4:2:0 vertical pairing over luma height 9 -> 3 outputs: chroma
  // cells span luma-row pairs except the single-row tail. On the x3
  // grid: out0 = (0, [6, 3]), out1 = (1, [3, 6]), out2 = (3, [6, 3])
  // — the tail cell carries HALF a full cell's weight, and every span
  // sums to the luma height (the denominator).
  let spans = AxisSpans::area_halved(9, 3).expect("valid");
  assert_eq!(spans.out_len(), 3);
  assert_eq!(spans.span(0), (0, &[6usize, 3][..]));
  assert_eq!(spans.span(1), (1, &[3usize, 6][..]));
  assert_eq!(spans.span(2), (3, &[6usize, 3][..]));

  // Even luma heights reduce to uniform double-width cells: identical
  // normalized weighting to the plain chroma-grid spans, scaled x2.
  let even = AxisSpans::area_halved(8, 4).expect("valid");
  for j in 0..4 {
    assert_eq!(even.span(j), (j, &[8usize][..]));
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_chroma_phased_v_folds_the_bottom_triangle() {
  // 4:2:0 Bottom siting on the V axis: box over the FULL luma grid folded
  // through the x2 `v = 1` triangle — odd luma row 2i+1 sits on c[i]
  // (weight 2), even 2i averages c[i-1] and c[i] (weights 1, 1), top edge
  // clamps c[-1] -> c[0]. Every output span sums to 2*luma_h.

  // EVEN luma_h = 8 -> 3 outputs (fractional, spans straddle chroma pairs).
  // Box area(8,3) = {0:[3,3,2], 2:[1,3,3,1], 5:[2,3,3]} folds to:
  //   out0 luma{0,1,2}: r0(even i0)+r1(odd i0) -> c0*(6+6), r2(even i1)
  //        -> c0*2,c1*2                                    => [14, 2] @ c0.
  //   out1 luma{2,3,4,5}                                   => [1, 10, 5] @ c0.
  //   out2 luma{5,6,7}                                     => [7, 9] @ c2.
  let even = AxisSpans::area_chroma_phased_v(8, 3).expect("valid");
  assert_eq!(even.out_len(), 3);
  assert_eq!(even.span(0), (0, &[14usize, 2][..]));
  assert_eq!(even.span(1), (0, &[1usize, 10, 5][..]));
  assert_eq!(even.span(2), (2, &[7usize, 9][..]));
  for o in 0..3 {
    assert_eq!(even.span(o).1.iter().sum::<usize>(), 2 * 8, "span {o} sum");
  }

  // ODD luma_h = 5 (chroma_h = 3) -> 2 outputs: the trailing luma row is
  // EVEN (row 4 -> {c1, c2}), exercising AxisSpans::area's odd-height tail
  // clamp — every span still sums to 2*luma_h with no cell past chroma_h-1.
  // Box area(5,2) = {0:[2,2,1], 2:[1,2,2]} folds to out0 = [9, 1] @ c0 and
  // out1 = [1, 7, 2] @ c0.
  let odd = AxisSpans::area_chroma_phased_v(5, 2).expect("valid");
  assert_eq!(odd.out_len(), 2);
  assert_eq!(odd.span(0), (0, &[9usize, 1][..]));
  assert_eq!(odd.span(1), (0, &[1usize, 7, 2][..]));
  for o in 0..2 {
    assert_eq!(odd.span(o).1.iter().sum::<usize>(), 2 * 5, "span {o} sum");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_two_row_window_matches_blend_then_bin_oracle() {
  // A FRACTIONAL vertical downscale under Bottom siting makes adjacent output
  // V-spans overlap by TWO chroma rows: area_chroma_phased_v(8,3) gives
  // out0 = chroma{0,1} and out1 = chroma{0,1,2}, so chroma rows 0 AND 1 feed
  // BOTH out0 and out1 while out0 is still open. This drives the AreaStream
  // 2-row accumulator window; the single-accumulator engine dropped rows 0/1
  // from out1 and produced the wrong value here.
  //
  // Hand oracle: reconstruct luma-resolution chroma from the Bottom triangle
  // (even 2i = (c[i-1]+c[i]+1)>>1, odd 2i+1 = c[i], c[-1] = c[0]), then
  // box-average luma_h = 8 -> 3 via area(8,3). The chroma values are chosen
  // so every blend is EXACT (even sums), collapsing the oracle's two
  // roundings onto the fold's single rounding.
  //   chroma = [10, 20, 40, 80]                    (chroma_h = 4, one column)
  //   recon  = [10, 10, 15, 20, 30, 40, 60, 80]    (blend, all exact)
  //   out0 = round((3*10 + 3*10 + 2*15) / 8) = round(90/8)  = 11
  //   out1 = round((15 + 3*20 + 3*30 + 40) / 8) = round(205/8) = 26
  //   out2 = round((2*40 + 3*60 + 3*80) / 8) = round(500/8) = 63
  let plan = ResamplePlan::area_chroma_420(1, 8, 1, 3, 0.0, 1.0, false).expect("valid Bottom plan");
  assert_eq!(
    plan.src_h(),
    16,
    "Bottom V spans sum to 2*luma_h -> scaled denom"
  );
  let chroma = [10u8, 20, 40, 80]; // one chroma column, chroma_h = 4 rows
  let mut stream =
    AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1).expect("realistic geometry");
  let mut out = [0u8; 3];
  let mut emitted = std::vec::Vec::new();
  for (y, &c) in chroma.iter().enumerate() {
    stream
      .feed_row(y, &[c], true, |oy, finalized| {
        emitted.push(oy);
        out[oy] = finalized[0];
      })
      .expect("chroma rows arrive in order");
  }
  assert_eq!(
    emitted,
    std::vec![0, 1, 2],
    "each output finalizes once, in order"
  );
  assert_eq!(out, [11, 26, 63], "2-row-window blend-then-bin oracle");
  // The direct span-sum reference reads each output's FULL folded span, so it
  // is independent of the streaming window — the stream must reproduce it.
  assert_eq!(out.to_vec(), direct_area_2d(&plan, &chroma, 1));

  // Wider genuinely-2-overlapping Bottom plans (multi-column / multi-channel,
  // and an H downscale) all reproduce the direct span-sum reference exactly.
  for &(cw, luma_h, ow, oh, channels) in &[(2usize, 8, 2, 3, 2), (4, 12, 3, 5, 3)] {
    let plan = ResamplePlan::area_chroma_420(cw, luma_h, ow, oh, 0.0, 1.0, false).expect("valid");
    let chroma_h = luma_h.div_ceil(2);
    let mut src = std::vec![0u8; cw * chroma_h * channels];
    lcg_fill(&mut src, 0x51 ^ luma_h as u32);
    let mut streamed = std::vec![0u8; ow * oh * channels];
    let mut stream =
      AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels).expect("geometry");
    for y in 0..chroma_h {
      let row = &src[y * cw * channels..(y + 1) * cw * channels];
      stream
        .feed_row(y, row, true, |oy, fin| {
          streamed[oy * ow * channels..(oy + 1) * ow * channels].copy_from_slice(fin);
        })
        .expect("rows in order");
    }
    assert_eq!(
      streamed,
      direct_area_2d(&plan, &src, channels),
      "{cw}x{chroma_h}->{ow}x{oh} c{channels} Bottom stream vs direct span-sum"
    );
  }

  // Combined centered-H (x4) + Bottom-V (x2): BOTH source denominators scale
  // (src_w = 8*chroma_w, src_h = 2*luma_h), so the AreaStream exactness guards
  // (h_sum_fits / denom_fits over src_w*src_h) see the PRODUCT of both scales.
  // The plan must build (each checked_mul holds) and stream the whole frame —
  // every output finalizes under the doubly-scaled denominator without
  // tripping an overflow guard.
  let (cw, luma_h, ow, oh, channels) = (4usize, 12, 3, 5, 2);
  let plan =
    ResamplePlan::area_chroma_420(cw, luma_h, ow, oh, 0.5, 1.0, false).expect("combined plan");
  assert_eq!(
    (plan.src_w(), plan.src_h()),
    (8 * cw, 2 * luma_h),
    "both denominators scale"
  );
  let chroma_h = luma_h.div_ceil(2);
  let mut src = std::vec![0u8; cw * chroma_h * channels];
  lcg_fill(&mut src, 0x9e);
  let mut stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("combined guards admit realistic geometry");
  let mut emitted = std::vec::Vec::new();
  for y in 0..chroma_h {
    let row = &src[y * cw * channels..(y + 1) * cw * channels];
    stream
      .feed_row(y, row, true, |oy, _| emitted.push(oy))
      .expect("rows in order");
  }
  assert_eq!(
    emitted,
    (0..oh).collect::<std::vec::Vec<_>>(),
    "every output finalizes under both scales"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_chroma_phased_v_top_folds_the_forward_triangle() {
  // 4:2:0 Top siting on the V axis: box over the FULL luma grid folded through
  // the x2 `v = 0` FORWARD triangle — even luma row 2i sits on c[i] (weight 2),
  // odd 2i+1 averages c[i] and c[i+1] (weights 1, 1), bottom edge clamps
  // c[chroma_h] -> c[chroma_h-1]. Every output span sums to 2*luma_h. This is
  // the exact vertical MIRROR of `area_chroma_phased_v` (Bottom).

  // EVEN luma_h = 8 -> 3 outputs (fractional, spans straddle chroma pairs).
  // Box area(8,3) = {0:[3,3,2], 2:[1,3,3,1], 5:[2,3,3]} folds forward to:
  //   out0 luma{0,1,2}: r0(even i0)->c0*2, r1(odd i0)->c0,c1, r2(even i1)->c1*2
  //        => [9, 7] @ c0.
  //   out1 luma{2,3,4,5}                                   => [5, 10, 1] @ c1.
  //   out2 luma{5,6,7}: r5(odd i2)->c2,c3, r6(even i3)->c3*2,
  //        r7(odd i3, bottom clamp)->c3*2                  => [2, 14] @ c2.
  let even = AxisSpans::area_chroma_phased_v_top(8, 3).expect("valid");
  assert_eq!(even.out_len(), 3);
  assert_eq!(even.span(0), (0, &[9usize, 7][..]));
  assert_eq!(even.span(1), (1, &[5usize, 10, 1][..]));
  assert_eq!(even.span(2), (2, &[2usize, 14][..]));
  for o in 0..3 {
    assert_eq!(even.span(o).1.iter().sum::<usize>(), 2 * 8, "span {o} sum");
  }

  // ODD luma_h = 5 (chroma_h = 3) -> 2 outputs: the trailing luma row is EVEN
  // (row 4 -> co-sited c2), so no odd row reaches the bottom edge and the clamp
  // is inert. Box area(5,2) = {0:[2,2,1], 2:[1,2,2]} folds to out0 = [6, 4] @ c0
  // and out1 = [4, 6] @ c1.
  let odd = AxisSpans::area_chroma_phased_v_top(5, 2).expect("valid");
  assert_eq!(odd.out_len(), 2);
  assert_eq!(odd.span(0), (0, &[6usize, 4][..]));
  assert_eq!(odd.span(1), (1, &[4usize, 6][..]));
  for o in 0..2 {
    assert_eq!(odd.span(o).1.iter().sum::<usize>(), 2 * 5, "span {o} sum");
  }

  // Top is NOT Bottom: the same geometry folds to different weights.
  let bottom = AxisSpans::area_chroma_phased_v(8, 3).expect("valid");
  assert_ne!(even.span(0), bottom.span(0), "Top out0 != Bottom out0");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn top_forward_stream_matches_reconstruct_then_bin_oracle() {
  // A FRACTIONAL vertical downscale under Top siting makes adjacent output
  // V-spans overlap so the AreaStream 2-row window (which absorbs the FORWARD
  // reach with NO sink-side delay) must reproduce "reconstruct luma-resolution
  // chroma from the forward triangle, then box-average".
  //
  // Hand oracle: reconstruct (even 2i = c[i], odd 2i+1 = (c[i]+c[i+1]+1)>>1,
  // c[chroma_h] = c[chroma_h-1]), then box-average luma_h = 8 -> 3 via area(8,3).
  // Chroma chosen so every blend is EXACT (even sums), collapsing the oracle's
  // two roundings onto the fold's single rounding.
  //   chroma = [10, 20, 40, 80]                    (chroma_h = 4, one column)
  //   recon  = [10, 15, 20, 30, 40, 60, 80, 80]    (forward blend, all exact)
  //   out0 = round((3*10 + 3*15 + 2*20) / 8) = round(115/8) = 14
  //   out1 = round((20 + 3*30 + 3*40 + 60) / 8) = round(290/8) = 36
  //   out2 = round((2*60 + 3*80 + 3*80) / 8) = round(600/8) = 75
  let plan = ResamplePlan::area_chroma_420(1, 8, 1, 3, 0.0, 0.0, true).expect("valid Top plan");
  assert_eq!(
    plan.src_h(),
    16,
    "Top V spans sum to 2*luma_h -> scaled denom"
  );
  assert!(
    !plan.has_chroma_v_phase(),
    "Top is v = 0, not Bottom's v = 1"
  );
  let chroma = [10u8, 20, 40, 80];
  let mut stream =
    AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1).expect("realistic geometry");
  let mut out = [0u8; 3];
  let mut emitted = std::vec::Vec::new();
  for (y, &c) in chroma.iter().enumerate() {
    stream
      .feed_row(y, &[c], true, |oy, finalized| {
        emitted.push(oy);
        out[oy] = finalized[0];
      })
      .expect("chroma rows arrive in order");
  }
  assert_eq!(emitted, std::vec![0, 1, 2], "each output finalizes once");
  assert_eq!(out, [14, 36, 75], "forward reconstruct-then-bin oracle");
  assert_eq!(out.to_vec(), direct_area_2d(&plan, &chroma, 1));

  // Top differs from Bottom on the SAME geometry (the earlier Bottom oracle gave
  // [11, 26, 63]).
  let bottom = ResamplePlan::area_chroma_420(1, 8, 1, 3, 0.0, 1.0, false).expect("Bottom");
  assert_ne!(
    direct_area_2d(&plan, &chroma, 1),
    direct_area_2d(&bottom, &chroma, 1),
    "Top vertical output must differ from Bottom"
  );

  // Wider genuinely-overlapping Top plans (multi-column / multi-channel, H
  // downscale) all reproduce the direct span-sum reference exactly.
  for &(cw, luma_h, ow, oh, channels) in &[(2usize, 8, 2, 3, 2), (4, 12, 3, 5, 3)] {
    let plan =
      ResamplePlan::area_chroma_420(cw, luma_h, ow, oh, 0.0, 0.0, true).expect("valid Top plan");
    let chroma_h = luma_h.div_ceil(2);
    let mut src = std::vec![0u8; cw * chroma_h * channels];
    lcg_fill(&mut src, 0x37 ^ luma_h as u32);
    let mut streamed = std::vec![0u8; ow * oh * channels];
    let mut stream =
      AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels).expect("geometry");
    for y in 0..chroma_h {
      let row = &src[y * cw * channels..(y + 1) * cw * channels];
      stream
        .feed_row(y, row, true, |oy, fin| {
          streamed[oy * ow * channels..(oy + 1) * ow * channels].copy_from_slice(fin);
        })
        .expect("rows in order");
    }
    assert_eq!(
      streamed,
      direct_area_2d(&plan, &src, channels),
      "{cw}x{chroma_h}->{ow}x{oh} c{channels} Top stream vs direct span-sum"
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn top_forward_window_depth_matches_bottom() {
  // A Top downscale opens at most two rows (one forward reach), same as Bottom,
  // so the bounded 2-row window streams it. An EXACT 2:1 Top downscale shares
  // only the boundary row (window 1).
  let frac = ResamplePlan::area_chroma_420(2, 8, 2, 3, 0.0, 0.0, true).expect("Top frac");
  assert_eq!(
    frac.v().window_depth(),
    2,
    "a fractional Top downscale opens two rows"
  );
  let exact = ResamplePlan::area_chroma_420(2, 8, 2, 4, 0.0, 0.0, true).expect("Top exact");
  assert_eq!(
    exact.v().window_depth(),
    1,
    "an exact 2:1 Top downscale opens one row"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn co_sited_area_stream_allocates_no_lookahead() {
  // A co-sited (v_phase 0) plan's adjacent output V-spans share only their
  // boundary row, so the stream opens exactly one row at a time and must NOT
  // allocate the look-ahead accumulator — the pre-Bottom resource envelope is
  // preserved (a large co-sited resample that fit before must still fit).
  let cosited =
    ResamplePlan::area_chroma_420(2, 8, 2, 4, 0.0, 0.0, false).expect("valid co-sited plan");
  assert_eq!(
    cosited.v().window_depth(),
    1,
    "co-sited spans never overlap mid-row"
  );
  let s = AreaStream::<u8>::new(
    cosited.h(),
    cosited.v(),
    cosited.src_w(),
    cosited.src_h(),
    1,
  )
  .expect("realistic geometry");
  assert!(
    s.acc_next.is_empty(),
    "a co-sited stream must not allocate the look-ahead accumulator"
  );
  // A `v = 1` Bottom downscale opens two rows, so it DOES allocate acc_next.
  let bottom =
    ResamplePlan::area_chroma_420(2, 8, 2, 3, 0.0, 1.0, false).expect("valid Bottom plan");
  assert_eq!(
    bottom.v().window_depth(),
    2,
    "a Bottom downscale opens two rows"
  );
  let b = AreaStream::<u8>::new(bottom.h(), bottom.v(), bottom.src_w(), bottom.src_h(), 1)
    .expect("realistic geometry");
  assert!(
    !b.acc_next.is_empty(),
    "a Bottom downscale needs the look-ahead accumulator"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn phased_v_exceeding_two_open_rows_is_rejected() {
  // An expanding (upscale) phased-V plan can put one source chroma row inside
  // THREE output spans at once (e.g. luma_h=3 -> out_h=7), which the bounded
  // 2-row window cannot stream. `AreaStream::new` must reject it as
  // `UpscaleUnsupported` rather than silently drop the third open row's
  // contributions (the pre-fix debug_assert accepted `start == y`, and release
  // builds corrupted).
  let plan = ResamplePlan::area_chroma_420(2, 3, 2, 7, 0.0, 1.0, false).expect("plan builds");
  assert!(
    plan.v().window_depth() > 2,
    "the 3->7 Bottom shape opens more than two rows at once"
  );
  let err = AreaStream::<u8>::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)
    .expect_err("a >2-open-row plan must be rejected, not streamed");
  assert!(
    matches!(err, ResampleError::UpscaleUnsupported(_)),
    "expected UpscaleUnsupported, got {err:?}"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_rejects_out_of_order_duplicate_and_skipped_rows() {
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let row = [0u8; 8];
  let mut stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1).unwrap();

  // Out of order from the start: row 1 before row 0.
  let err = stream.feed_row(1, &row, true, |_, _| {}).unwrap_err();
  match err {
    ResampleError::OutOfSequenceRow(e) => {
      assert_eq!((e.expected(), e.got()), (0, 1));
    }
    other => panic!("expected OutOfSequenceRow, got {other:?}"),
  }
  // The rejected row must not have touched stream state.
  stream.feed_row(0, &row, true, |_, _| {}).unwrap();

  // Duplicate.
  let err = stream.feed_row(0, &row, true, |_, _| {}).unwrap_err();
  assert!(err.is_out_of_sequence_row());

  // Skipped.
  let err = stream.feed_row(2, &row, true, |_, _| {}).unwrap_err();
  match err {
    ResampleError::OutOfSequenceRow(e) => {
      assert_eq!((e.expected(), e.got()), (1, 2));
    }
    other => panic!("expected OutOfSequenceRow, got {other:?}"),
  }

  // reset() restarts the sequence.
  stream.reset();
  stream.feed_row(0, &row, true, |_, _| {}).unwrap();
}

#[test]
fn round_div_half_up_is_exact_and_overflow_free() {
  // Equivalence with (a + d/2) / d on small values, both parities.
  for d in 1u64..=9 {
    for a in 0u64..=200 {
      assert_eq!(round_div_half_up(a, d), (a + d / 2) / d, "a={a} d={d}");
    }
  }
  // Boundary: the naive form would wrap; the q/r form must not.
  let d = u64::MAX / 255;
  assert_eq!(round_div_half_up(d * 255, d), 255);
  assert_eq!(round_div_half_up(u64::MAX, u64::MAX), 1);
  assert_eq!(round_div_half_up(u64::MAX - 1, u64::MAX), 1);
  assert_eq!(round_div_half_up(u64::MAX / 2, u64::MAX), 0);
  assert_eq!(round_div_half_up(u64::MAX / 2 + 1, u64::MAX), 1);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_constant_input_is_constant() {
  let plan = AreaResampler::to(3, 2)
    .plan(7, 5)
    .expect("valid")
    .expect("non-identity");
  let src = std::vec![173u8; 7 * 5];
  assert!(stream_collect(&plan, &src, 1).iter().all(|&v| v == 173));
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
/// Same LCG as `ci/gen_cv2_goldens.py`: the parity sources are
/// synthesized identically on both sides, so the fixture carries only
/// cv2's outputs.
fn lcg_fill(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[test]
fn area_matches_cv2_inter_area_within_one_lsb() {
  // cv2's INTER_AREA uses its own fixed-point/float internals, so the
  // contract is +-1 LSB against our exact integer area mean — checked
  // across integer and fractional ratios, gray and interleaved RGB.
  for &(src_w, src_h, out_w, out_h, channels, seed, golden) in super::cv2_goldens::ALL {
    let mut src = std::vec![0u8; src_w * src_h * channels];
    lcg_fill(&mut src, seed);
    let plan = AreaResampler::to(out_w, out_h)
      .plan(src_w, src_h)
      .expect("valid downscale")
      .expect("non-identity");
    let mut stream =
      AreaStream::new(plan.h(), plan.v(), src_w, src_h, channels).expect("realistic geometry");
    let mut ours = std::vec![0u8; out_w * out_h * channels];
    for y in 0..src_h {
      let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
      stream
        .feed_row(y, row, true, |oy, finalized| {
          ours[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
        })
        .expect("rows in order");
    }
    assert_eq!(golden.len(), ours.len(), "{src_w}x{src_h}->{out_w}x{out_h}");
    for (i, (a, b)) in ours.iter().zip(golden.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "{src_w}x{src_h}->{out_w}x{out_h} c{channels} idx {i}: ours {a} vs cv2 {b}"
      );
    }
  }
}

#[test]
fn plan_error_display_names_geometry() {
  let upscale = ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(1920, 1080, 3840, 2160));
  assert!(upscale.is_upscale_unsupported());
  let msg = format!("{upscale}");
  assert!(msg.contains("1920x1080"), "{msg}");
  assert!(msg.contains("3840x2160"), "{msg}");

  let zero = ResampleError::ZeroOutputDimension(ZeroOutputDimension::new(0, 189));
  assert!(zero.is_zero_output_dimension());
  assert!(format!("{zero}").contains("0x189"));

  let overflow = ResampleError::Overflow(PlanGeometry::new(usize::MAX, 2, 3, 1));
  let alloc = ResampleError::AllocationFailed(PlanGeometry::new(usize::MAX, 1, 1, 1));
  assert!(alloc.is_allocation_failed());
  assert!(format!("{alloc}").contains("allocation"));
  assert!(overflow.is_overflow());
  let msg = format!("{overflow}");
  assert!(msg.contains("3x1"), "{msg}");
}

#[test]
fn area_plans_products_beyond_32_bits() {
  // 70_000 x 62_000 exceeds u32::MAX: span coordinates run in u64, so
  // 32-bit targets plan any geometry whose buffers fit (regression
  // cover runs under the miri i686 jobs).
  let plan = AreaResampler::to(62_000, 1)
    .plan(70_000, 2)
    .expect("coordinates must not overflow 32-bit usize")
    .expect("non-identity");
  let h = plan.h();
  assert_eq!(h.out_len(), 62_000);
  let (first_start, first) = h.span(0);
  assert_eq!(first_start, 0);
  assert_eq!(first.iter().sum::<usize>(), 70_000);
  let (last_start, last) = h.span(61_999);
  assert_eq!(last_start + last.len(), 70_000);
}

#[test]
fn area_taps_formula_is_exact() {
  // Weight-arena size has the closed form src + out - gcd(src, out):
  // every source cell contributes once, plus one shared straddle cell
  // per unaligned output boundary. Cross-checked against the arenas
  // the builder actually materializes.
  for src in 1..=24usize {
    for out in 1..=src {
      let expected = AxisSpans::area_taps(src, out).unwrap();
      let plan = AreaResampler::to(out, 1).plan(src, 1);
      if out == src {
        // Identity plans short-circuit before building spans; check
        // the formula degenerates to src.
        assert_eq!(expected, src);
        continue;
      }
      let plan = plan.expect("valid").expect("non-identity");
      let total: usize = (0..out).map(|j| plan.h().span(j).1.len()).sum();
      assert_eq!(total, expected, "src={src} out={out}");
    }
  }
  // Pure arithmetic at hostile magnitudes — no allocation involved.
  assert_eq!(AxisSpans::area_taps(usize::MAX, 1), Some(usize::MAX));
}

#[test]
fn area_tiny_output_huge_source_fails_recoverably() {
  // A hostile source dimension from untrusted metadata must surface a
  // structured error, not abort inside infallible allocation: the
  // weight arena for usize::MAX taps trips Vec::try_reserve_exact
  // capacity overflow deterministically.
  let err = AreaResampler::to(1, 1).plan(usize::MAX, 1).unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[test]
fn area_overflow_rejected() {
  // Hostile dimensions must surface a structured error, and which one
  // is pointer-width-dependent. On 64-bit targets the u64 span-grid
  // product src_w * out_w overflows and is rejected as Overflow. On
  // 32-bit targets that product always fits u64 — the grid
  // coordinates were widened precisely so 32-bit dims cannot overflow
  // them — so the same call runs on to the span arenas, whose
  // capacity-overflow reservation (entries times element size above
  // isize::MAX, no allocator touch) surfaces as AllocationFailed.
  let err = AreaResampler::to(usize::MAX / 2, 1)
    .plan(usize::MAX - 1, 1)
    .unwrap_err();
  #[cfg(target_pointer_width = "64")]
  assert!(err.is_overflow(), "got {err:?}");
  #[cfg(target_pointer_width = "32")]
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[test]
fn h_pass_simd_matches_scalar_bit_exact() {
  // Differential against the scalar reference at the h_tmp level —
  // the u8 finalize divide downstream could mask a small H-sum error,
  // so the raw u32 sums are compared directly. Geometries cover
  // single-chunk spans with tails (1920->336: 6-7 taps), multi-chunk
  // spans (640->7: ~92 taps), 2-tap spans whose final span lands on
  // the row-end staging path (4096->4095), tiny all-staged rows, and
  // an output width past the u16 weight bound (70000->66000), where
  // the dispatcher falls back to scalar.
  // Miri keeps every allocation's address live for the whole process,
  // and the i686 cell has 4 GB of address space shared by the entire
  // suite — the large geometries would exhaust it (the failure then
  // surfaces in whatever unrelated test runs last). The small cases
  // still cover chunked spans, staged row-ends, and both channel
  // counts under Miri; the full list runs on every native lane.
  let cases: &[(usize, usize, usize)] = if cfg!(miri) {
    &[(64, 7, 1), (64, 7, 3), (12, 11, 3), (8, 3, 1), (5, 4, 3)]
  } else {
    &[
      (1920, 336, 1),
      (1920, 336, 3),
      (640, 7, 1),
      (640, 7, 3),
      (4096, 4095, 1),
      (4096, 4095, 3),
      (5, 4, 3),
      (8, 3, 1),
      (70_000, 66_000, 1),
      // Wide-path boundaries for the AVX kernels (padded span length =
      // taps rounded up to a multiple of 8): the AVX-512 step consumes
      // 32 taps, AVX2 consumes 16, and the remainder falls to the
      // 128-bit step. These ratios place the remainder at each of 0 /
      // 8 / 16 / 24 taps after the wide chunks, for both channel
      // counts. Validated on real AVX2 hardware by the avx2-max
      // coverage job and under SDE for AVX-512.
      (256, 8, 1), // padded 32: one AVX-512 chunk, no remainder
      (256, 8, 3),
      (264, 8, 1), // padded 40: AVX-512 32 + 8 remainder
      (264, 8, 3),
      (336, 8, 1), // padded 48: AVX-512 32 + 16 remainder
      (336, 8, 3),
      (400, 8, 1), // padded 56: AVX-512 32 + 24 remainder
      (400, 8, 3),
      (120, 8, 1), // padded 16: one AVX2 chunk; AVX-512 tail-only
      (120, 8, 3),
      (160, 8, 1), // padded 24: AVX2 16 + 8; AVX-512 tail-only
      (160, 8, 3),
    ]
  };
  for &(src_w, out_w, channels) in cases {
    let plan = AreaResampler::to(out_w, 1)
      .plan(src_w, 2)
      .expect("valid geometry")
      .expect("strict downscale");
    let mut scalar = AreaStream::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut simd = AreaStream::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut row = std::vec![0u8; src_w * channels];
    for y in 0..2usize {
      lcg_fill(&mut row, (src_w * 31 + out_w * 7 + channels + y) as u32);
      let mut scalar_rows = std::vec::Vec::new();
      let mut simd_rows = std::vec::Vec::new();
      scalar
        .feed_row(y, &row, false, |oy, r| {
          scalar_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      simd
        .feed_row(y, &row, true, |oy, r| {
          simd_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      assert_eq!(
        scalar.h_tmp, simd.h_tmp,
        "h_tmp diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar.acc, simd.acc,
        "acc diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar_rows, simd_rows,
        "emitted rows diverged: src_w={src_w} out_w={out_w} c={channels}"
      );
    }
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
fn lcg_fill_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u16;
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[test]
fn h_pass_u16_simd_matches_scalar_bit_exact() {
  // u16 analogue of `h_pass_simd_matches_scalar_bit_exact`: the wider
  // (u64) H-sums and the u32-by-u64 V-pass split-multiply must match
  // the scalar reference bit-for-bit. Samples span the full 16-bit
  // range so the high bits a u8 path drops are exercised. Same boundary
  // geometries: single-chunk spans with tails, multi-chunk spans,
  // row-end staging (4096->4095), tiny all-staged rows, output past the
  // u16 weight bound (scalar fallback), and the AVX wide-path remainder
  // ladder (padded 32/40/48/56/16/24), both channel counts.
  let cases: &[(usize, usize, usize)] = if cfg!(miri) {
    &[(64, 7, 1), (64, 7, 3), (12, 11, 3), (8, 3, 1), (5, 4, 3)]
  } else {
    &[
      (1920, 336, 1),
      (1920, 336, 3),
      (640, 7, 1),
      (640, 7, 3),
      (4096, 4095, 1),
      (4096, 4095, 3),
      (5, 4, 3),
      (8, 3, 1),
      (70_000, 66_000, 1),
      (256, 8, 1),
      (256, 8, 3),
      (264, 8, 1),
      (264, 8, 3),
      (336, 8, 1),
      (336, 8, 3),
      (400, 8, 1),
      (400, 8, 3),
      (120, 8, 1),
      (120, 8, 3),
      (160, 8, 1),
      (160, 8, 3),
    ]
  };
  for &(src_w, out_w, channels) in cases {
    let plan = AreaResampler::to(out_w, 1)
      .plan(src_w, 2)
      .expect("valid geometry")
      .expect("strict downscale");
    let mut scalar = AreaStream::<u16>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut simd = AreaStream::<u16>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut row = std::vec![0u16; src_w * channels];
    for y in 0..2usize {
      lcg_fill_u16(&mut row, (src_w * 31 + out_w * 7 + channels + y) as u32);
      let mut scalar_rows = std::vec::Vec::new();
      let mut simd_rows = std::vec::Vec::new();
      scalar
        .feed_row(y, &row, false, |oy, r| {
          scalar_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      simd
        .feed_row(y, &row, true, |oy, r| {
          simd_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      assert_eq!(
        scalar.h_tmp, simd.h_tmp,
        "h_tmp diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar.acc, simd.acc,
        "acc diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar_rows, simd_rows,
        "emitted rows diverged: src_w={src_w} out_w={out_w} c={channels}"
      );
    }
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
fn lcg_fill_f32(buf: &mut [f32], seed: u32) {
  let mut state = seed;
  for v in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Non-negative values with a fractional part — the realistic shape
    // of float color samples (XYZ tristimulus, scene-referred RGB), and
    // the no-cancellation case, so the SIMD-vs-scalar reorder bound stays
    // tight.
    let int = ((state >> 8) & 0x3FFF) as f32;
    let frac = (state & 0xFF) as f32 / 256.0;
    *v = int + frac;
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
fn assert_close_rel_f64(a: &[f64], b: &[f64], rel: f64, ctx: &str) {
  assert_eq!(a.len(), b.len(), "{ctx} length");
  for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
    let tol = rel * x.abs().max(y.abs()) + 1e-6;
    assert!((x - y).abs() <= tol, "{ctx}[{i}]: {x} vs {y} (rel {rel})");
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[test]
fn h_pass_f32_simd_matches_scalar_within_tolerance() {
  // Float analogue of `h_pass_u16_simd_matches_scalar_bit_exact`. Float
  // addition does not associate, so the lane-parallel kernels match the
  // sequential scalar reference within a small tolerance, not
  // bit-for-bit — the parity contract for every float-stream SIMD tier.
  // The f64 H-sums and V-accumulators carry a tight relative bound (the
  // products are exact, so only the tap-sum order differs); the
  // finalized f32 rows match within a few ULPs. Same boundary-geometry
  // ladder as the integer differentials plus a 512x case (spans of ~512
  // taps, where the reorder is widest): single-chunk spans with tails,
  // multi-chunk spans, row-end staging, tiny all-staged rows, and the
  // AVX wide-path remainder widths, both channel counts.
  let cases: &[(usize, usize, usize)] = if cfg!(miri) {
    &[(64, 7, 1), (64, 7, 3), (12, 11, 3), (8, 3, 1), (5, 4, 3)]
  } else {
    &[
      (1920, 336, 1),
      (1920, 336, 3),
      (640, 7, 1),
      (640, 7, 3),
      (4096, 8, 1),
      (4096, 8, 3),
      (4096, 4095, 1),
      (4096, 4095, 3),
      (5, 4, 3),
      (8, 3, 1),
      (256, 8, 1),
      (256, 8, 3),
      (264, 8, 1),
      (264, 8, 3),
      (336, 8, 1),
      (336, 8, 3),
      (400, 8, 1),
      (400, 8, 3),
      (120, 8, 1),
      (160, 8, 3),
    ]
  };
  for &(src_w, out_w, channels) in cases {
    let plan = AreaResampler::to(out_w, 1)
      .plan(src_w, 2)
      .expect("valid geometry")
      .expect("strict downscale");
    let mut scalar = AreaStream::<f32>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut simd = AreaStream::<f32>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut row = std::vec![0f32; src_w * channels];
    for y in 0..2usize {
      lcg_fill_f32(&mut row, (src_w * 31 + out_w * 7 + channels + y) as u32);
      let mut scalar_rows = std::vec::Vec::new();
      let mut simd_rows = std::vec::Vec::new();
      scalar
        .feed_row(y, &row, false, |oy, r| scalar_rows.push((oy, r.to_vec())))
        .unwrap();
      simd
        .feed_row(y, &row, true, |oy, r| simd_rows.push((oy, r.to_vec())))
        .unwrap();
      assert_close_rel_f64(
        &scalar.h_tmp,
        &simd.h_tmp,
        1e-9,
        &std::format!("h_tmp src_w={src_w} out_w={out_w} c={channels} y={y}"),
      );
      assert_close_rel_f64(
        &scalar.acc,
        &simd.acc,
        1e-9,
        &std::format!("acc src_w={src_w} out_w={out_w} c={channels} y={y}"),
      );
      assert_eq!(scalar_rows.len(), simd_rows.len(), "emitted count");
      for ((soy, srow), (doy, drow)) in scalar_rows.iter().zip(simd_rows.iter()) {
        assert_eq!(soy, doy, "emitted index");
        for (i, (&a, &b)) in srow.iter().zip(drow.iter()).enumerate() {
          let tol = 1e-6 * a.abs().max(b.abs()) + 1e-3;
          assert!(
            (a - b).abs() <= tol,
            "emitted row src_w={src_w} out_w={out_w} c={channels} [{i}]: {a} vs {b}"
          );
        }
      }
    }
  }
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
#[test]
fn h_pass_f32_simd_matches_scalar_with_non_finite_padding() {
  // The zero-padded weight arena annihilates samples past a span's last
  // tap by multiplying them by a zero weight — exact for integers, but
  // `0.0 * NaN` and `0.0 * inf` are NaN, so a non-finite sample in a
  // padding lane would poison the span it pads. A padding lane
  // direct-loads a real neighbor the scalar reference never sums for
  // that output, so a non-finite value there must NOT leak into that
  // span. These integer-ratio downscales leave 6 padding lanes per
  // 2-tap span (and 5 per 3-tap span), all direct-loaded; placing
  // non-finite samples in the low pixels lands them in earlier spans'
  // padding. The SIMD and scalar streams must agree on which outputs are
  // finite (and on the finite values), not just on finite inputs.
  let cases: &[(usize, usize, usize)] =
    &[(16, 8, 1), (16, 8, 3), (24, 8, 1), (24, 8, 3), (48, 8, 3)];
  for &(src_w, out_w, channels) in cases {
    let plan = AreaResampler::to(out_w, 1)
      .plan(src_w, 2)
      .expect("valid geometry")
      .expect("strict downscale");
    let mut scalar = AreaStream::<f32>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut simd = AreaStream::<f32>::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut row = std::vec![0f32; src_w * channels];
    for y in 0..2usize {
      lcg_fill_f32(&mut row, (src_w * 17 + out_w * 5 + channels + y) as u32);
      // Non-finite samples in low pixels — padding lanes for earlier
      // spans, real taps for the spans they belong to.
      row[2 * channels] = f32::NAN;
      row[5 * channels + (channels - 1)] = f32::INFINITY;
      row[9 * channels] = f32::NEG_INFINITY;
      let mut s_rows = std::vec::Vec::new();
      let mut d_rows = std::vec::Vec::new();
      scalar
        .feed_row(y, &row, false, |oy, r| s_rows.push((oy, r.to_vec())))
        .unwrap();
      simd
        .feed_row(y, &row, true, |oy, r| d_rows.push((oy, r.to_vec())))
        .unwrap();
      assert_eq!(s_rows.len(), d_rows.len(), "emitted count");
      for ((_, srow), (_, drow)) in s_rows.iter().zip(d_rows.iter()) {
        for (i, (&a, &b)) in srow.iter().zip(drow.iter()).enumerate() {
          assert_eq!(
            a.is_finite(),
            b.is_finite(),
            "finiteness mismatch src_w={src_w} out_w={out_w} c={channels} [{i}]: scalar {a} simd {b}"
          );
          if a.is_finite() {
            let tol = 1e-6 * a.abs().max(b.abs()) + 1e-3;
            assert!((a - b).abs() <= tol, "value [{i}]: {a} vs {b}");
          } else {
            assert_eq!(
              a.is_nan(),
              b.is_nan(),
              "nan-vs-inf [{i}]: scalar {a} simd {b}"
            );
          }
        }
      }
    }
  }
}

// ---- Filter SIMD-vs-scalar stream parity ----------------------------------
//
// End-to-end differential: feeding identical rows to a scalar-path
// `FilterStream` (`use_simd = false`) and a SIMD-path one (`use_simd =
// true`) must yield emitted output rows that agree — bit-for-bit through the
// `f32`-narrowing finalize for the float stream, within `±1` for the integer
// streams (the SIMD H-pass reorders the `f64` tap sum, which can flip the
// half-rounded intermediate by one LSB). This drives the whole engine — H
// dispatch, the ring V-pass, and finalize — through the host's selected
// backend, complementing the dispatcher-level h_tmp differential in
// `row::dispatch::filter_reduce`.

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
mod filter_simd_parity {
  use super::super::{
    BlackmanSinc, CatmullRom, CubicBSpline, FilterKernel, FilterSample, FilterStream,
    FilteredResampler, Lanczos3, Resampler, SpanKind, Triangle,
  };

  /// Deterministic LCG byte stream (matches the resample suite's source).
  fn lcg(n: usize, seed: u32) -> std::vec::Vec<u8> {
    let mut state = seed;
    let mut out = std::vec![0u8; n];
    for b in &mut out {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *b = (state >> 8) as u8;
    }
    out
  }

  /// Streams `src` through a fresh [`FilterStream`] under `use_simd`,
  /// returning the `ow*oh*channels` output. The plan must be a filter plan.
  #[allow(clippy::too_many_arguments)]
  fn stream<S, K>(
    kernel: K,
    src: &[S],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    channels: usize,
    use_simd: bool,
  ) -> std::vec::Vec<S>
  where
    S: FilterSample,
    K: FilterKernel,
  {
    let plan = FilteredResampler::new(ow, oh, kernel)
      .plan(sw, sh)
      .expect("valid downscale")
      .expect("non-identity");
    assert_eq!(plan.kind(), SpanKind::Filter);
    let fh = plan.filter_h().expect("h windows");
    let fv = plan.filter_v().expect("v windows");
    let mut s = FilterStream::<S>::new(fh, fv, sw, sh, channels).expect("geometry");
    let mut out = std::vec![S::default(); ow * oh * channels];
    for y in 0..sh {
      let row = &src[y * sw * channels..(y + 1) * sw * channels];
      s.feed_row(y, row, use_simd, |oy, fin| {
        out[oy * ow * channels..(oy + 1) * ow * channels].copy_from_slice(fin);
      })
      .expect("rows in order");
    }
    out
  }

  /// (sw, sh, ow, oh): downscales spanning single-chunk windows with tails
  /// (Triangle/CatmullRom: 4-8 taps), wide multi-chunk windows (Lanczos3 on
  /// a heavy downscale: 12+ taps), and the row-end staging boundary.
  fn geoms() -> &'static [(usize, usize, usize, usize)] {
    if cfg!(miri) {
      &[(16, 12, 5, 4), (12, 12, 4, 3)]
    } else {
      &[
        (64, 48, 17, 13),
        (200, 60, 41, 19),
        (128, 96, 8, 7),
        (96, 17, 31, 6),
        (300, 200, 100, 67),
      ]
    }
  }

  #[test]
  fn u8_simd_matches_scalar_within_one_lsb() {
    let mut worst = 0u8;
    for &(sw, sh, ow, oh) in geoms() {
      for ch in [1usize, 3] {
        for (ki, kernel) in [0u8, 1, 2].into_iter().enumerate() {
          let src = lcg(sw * sh * ch, (sw * 7 + sh * 3 + ow + ch + ki) as u32 + 1);
          let (a, b) = match kernel {
            0 => (
              stream(Triangle, &src, sw, sh, ow, oh, ch, false),
              stream(Triangle, &src, sw, sh, ow, oh, ch, true),
            ),
            1 => (
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, false),
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, true),
            ),
            _ => (
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, false),
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, true),
            ),
          };
          assert_eq!(a.len(), b.len());
          for (i, (&s, &d)) in a.iter().zip(b.iter()).enumerate() {
            let diff = s.abs_diff(d);
            worst = worst.max(diff);
            assert!(
              diff <= 1,
              "u8 k{kernel} {sw}x{sh}->{ow}x{oh} c{ch} [{i}]: scalar {s} simd {d}"
            );
          }
        }
      }
    }
    std::eprintln!("filter u8 SIMD-vs-scalar: max abs diff = {worst}");
  }

  #[test]
  fn u16_simd_matches_scalar_within_one_lsb() {
    let mut worst = 0u16;
    for &(sw, sh, ow, oh) in geoms() {
      for ch in [1usize, 3] {
        for kernel in [0u8, 1, 2] {
          let raw = lcg(sw * sh * ch, (sw * 5 + sh * 11 + ow + ch) as u32 + 7);
          let src: std::vec::Vec<u16> = raw
            .iter()
            .map(|&v| (u16::from(v) << 8) | u16::from(v))
            .collect();
          let (a, b) = match kernel {
            0 => (
              stream(Triangle, &src, sw, sh, ow, oh, ch, false),
              stream(Triangle, &src, sw, sh, ow, oh, ch, true),
            ),
            1 => (
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, false),
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, true),
            ),
            _ => (
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, false),
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, true),
            ),
          };
          for (i, (&s, &d)) in a.iter().zip(b.iter()).enumerate() {
            let diff = s.abs_diff(d);
            worst = worst.max(diff);
            assert!(
              diff <= 1,
              "u16 k{kernel} {sw}x{sh}->{ow}x{oh} c{ch} [{i}]: scalar {s} simd {d}"
            );
          }
        }
      }
    }
    std::eprintln!("filter u16 SIMD-vs-scalar: max abs diff = {worst}");
  }

  #[test]
  fn f32_simd_matches_scalar_within_tolerance() {
    let mut worst = 0.0f32;
    for &(sw, sh, ow, oh) in geoms() {
      for ch in [1usize, 3] {
        for kernel in [0u8, 1, 2] {
          let src: std::vec::Vec<f32> = lcg(sw * sh * ch, (sw * 13 + sh * 5 + ow + ch) as u32 + 3)
            .into_iter()
            .map(|b| f32::from(b) + f32::from(b) / 256.0)
            .collect();
          let (a, b) = match kernel {
            0 => (
              stream(Triangle, &src, sw, sh, ow, oh, ch, false),
              stream(Triangle, &src, sw, sh, ow, oh, ch, true),
            ),
            1 => (
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, false),
              stream(CatmullRom, &src, sw, sh, ow, oh, ch, true),
            ),
            _ => (
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, false),
              stream(Lanczos3, &src, sw, sh, ow, oh, ch, true),
            ),
          };
          for (i, (&s, &d)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (s - d).abs();
            worst = worst.max(diff);
            let tol = 1e-4 * s.abs().max(d.abs()) + 1e-2;
            assert!(
              diff <= tol,
              "f32 k{kernel} {sw}x{sh}->{ow}x{oh} c{ch} [{i}]: scalar {s} simd {d}"
            );
          }
        }
      }
    }
    std::eprintln!("filter f32 SIMD-vs-scalar: max abs diff = {worst}");
  }

  #[test]
  fn phase4_kernels_resample_end_to_end_simd_matches_scalar() {
    // End-to-end wire-up for the two RFC #238 Phase 4 kernels: each routes
    // through `FilteredResampler<K>::plan` and the SAME kernel-agnostic
    // `FilterStream` apply path as the built-in kernels (no new apply code),
    // so it gets every SIMD backend for free. Resample a frame with each new
    // kernel on the u8, u16, and f32 streams and assert the SIMD pass matches
    // the scalar pass within the engine's per-element budget — proving the
    // all-backend apply path is exercised, not bypassed.
    for &(sw, sh, ow, oh) in geoms() {
      for ch in [1usize, 3] {
        // u8: byte-exact between scalar and SIMD on the q8 grid (+-1 LSB).
        let src_u8 = lcg(sw * sh * ch, (sw + sh + ow + ch) as u32 + 17);
        for (a, b) in [
          (
            stream(BlackmanSinc, &src_u8, sw, sh, ow, oh, ch, false),
            stream(BlackmanSinc, &src_u8, sw, sh, ow, oh, ch, true),
          ),
          (
            stream(CubicBSpline, &src_u8, sw, sh, ow, oh, ch, false),
            stream(CubicBSpline, &src_u8, sw, sh, ow, oh, ch, true),
          ),
        ] {
          assert_eq!(a.len(), ow * oh * ch);
          for (&s, &d) in a.iter().zip(b.iter()) {
            assert!(s.abs_diff(d) <= 1, "u8 {sw}x{sh}->{ow}x{oh}: {s} vs {d}");
          }
        }
        // u16: +-1 LSB on the full-precision float coefficients.
        let src_u16: std::vec::Vec<u16> = src_u8
          .iter()
          .map(|&v| (u16::from(v) << 8) | u16::from(v))
          .collect();
        for (a, b) in [
          (
            stream(BlackmanSinc, &src_u16, sw, sh, ow, oh, ch, false),
            stream(BlackmanSinc, &src_u16, sw, sh, ow, oh, ch, true),
          ),
          (
            stream(CubicBSpline, &src_u16, sw, sh, ow, oh, ch, false),
            stream(CubicBSpline, &src_u16, sw, sh, ow, oh, ch, true),
          ),
        ] {
          for (&s, &d) in a.iter().zip(b.iter()) {
            assert!(s.abs_diff(d) <= 1, "u16 {sw}x{sh}->{ow}x{oh}: {s} vs {d}");
          }
        }
        // f32: within the engine's float tolerance.
        let src_f32: std::vec::Vec<f32> = src_u8
          .iter()
          .map(|&v| f32::from(v) + f32::from(v) / 256.0)
          .collect();
        for (a, b) in [
          (
            stream(BlackmanSinc, &src_f32, sw, sh, ow, oh, ch, false),
            stream(BlackmanSinc, &src_f32, sw, sh, ow, oh, ch, true),
          ),
          (
            stream(CubicBSpline, &src_f32, sw, sh, ow, oh, ch, false),
            stream(CubicBSpline, &src_f32, sw, sh, ow, oh, ch, true),
          ),
        ] {
          for (&s, &d) in a.iter().zip(b.iter()) {
            let tol = 1e-4 * s.abs().max(d.abs()) + 1e-2;
            assert!((s - d).abs() <= tol, "f32 {sw}x{sh}->{ow}x{oh}: {s} vs {d}");
          }
        }
      }
    }
  }
}

// ---- PIL (Pillow) filter-resampler parity ---------------------------------
//
// The scalar separable filter resampler must reproduce Pillow's
// `Image.resize` **byte-exact** for u8, within +-1 LSB for u16, and within
// a small float tolerance for f32 (PIL `F`-mode unclamped). Sources are
// synthesized identically to `ci/gen_pil_goldens.py` (the same LCG +
// per-mode shaping), so the checked-in fixture carries only PIL's outputs.
// The u8 stream resamples on PIL's 8bpc fixed-point coefficient grid
// (`PRECISION_BITS = 22`) and accumulates in f64, so its `ss/2^22` and both
// passes' `clip8` finalize match Pillow bit-for-bit; u16 / f32 keep the
// full-precision coefficients (PIL's 32bpc paths use double coefficients,
// which f32 already matches within budget). Getting the half-pixel center
// convention exactly right (mirrored from PIL `precompute_coeffs`) is what
// makes all three hold.

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
mod pil_parity {
  use super::super::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, SpanKind,
    Triangle, pil_goldens,
  };

  /// The raw LCG byte stream — the shared source primitive (matches
  /// `ci/gen_pil_goldens.py::lcg_bytes`).
  fn lcg_bytes(n: usize, seed: u32) -> std::vec::Vec<u8> {
    let mut state = seed;
    let mut out = std::vec![0u8; n];
    for b in &mut out {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *b = (state >> 8) as u8;
    }
    out
  }

  /// u8 source (`L` / `RGB`): the LCG bytes verbatim.
  fn source_u8(sw: usize, sh: usize, channels: usize, seed: u32) -> std::vec::Vec<u8> {
    lcg_bytes(sw * sh * channels, seed)
  }

  /// u16 source (`I;16`): `(v << 8) | roll(v, 1)`, the numpy-roll shaping
  /// the generator uses to put signal in the low byte.
  fn source_u16(sw: usize, sh: usize, seed: u32) -> std::vec::Vec<u16> {
    let raw = lcg_bytes(sw * sh, seed);
    let n = raw.len();
    (0..n)
      .map(|i| {
        let v = u16::from(raw[i]);
        let rolled = u16::from(raw[(i + n - 1) % n]);
        (v << 8) | rolled
      })
      .collect()
  }

  /// f32 source (`F`): `raw + raw / 256.0`.
  fn source_f32(sw: usize, sh: usize, seed: u32) -> std::vec::Vec<f32> {
    lcg_bytes(sw * sh, seed)
      .into_iter()
      .map(|b| b as f32 + b as f32 / 256.0)
      .collect()
  }

  /// Source pixel count above which a PIL golden case is skipped under Miri.
  /// The large fixtures (up to `1920x1080`) resample hundreds of source rows
  /// through the real stream — and the SIMD variants drive every tap through
  /// Miri's NEON intrinsic shims — which runs for many minutes apiece under
  /// the interpreter and would push the job past its timeout. The small kept
  /// cases (the largest is the `16x12` upscale; everything else is `8x8` or
  /// tinier) still exercise the full apply path: chunked vs row-end-staged
  /// spans, the SIMD vs scalar H-pass tiers (hence the NEON `vaddvq` fallback),
  /// the multi-output accumulator ring, the upscale direction, and both channel
  /// counts — so Miri's UB checking is unchanged; the larger cases run on every
  /// native lane. No effect off Miri (the filter passes everything).
  const MIRI_MAX_SRC_PX: usize = 1_500;

  /// `(src_w, src_h)` of a PIL golden tuple, for the Miri size filter.
  trait GoldenDims {
    fn src_px(&self) -> usize;
  }
  impl<T: 'static> GoldenDims for (usize, usize, usize, usize, usize, u32, &'static [T]) {
    fn src_px(&self) -> usize {
      self.0 * self.1
    }
  }

  /// The golden cases to actually run: every case off Miri, and only the
  /// cases at or below [`MIRI_MAX_SRC_PX`] source pixels under Miri.
  fn miri_subset<G: GoldenDims>(all: &'static [G]) -> impl Iterator<Item = &'static G> {
    all
      .iter()
      .filter(|g| !cfg!(miri) || g.src_px() <= MIRI_MAX_SRC_PX)
  }

  /// Runs the filter stream for `kernel` over `src` (channels-interleaved,
  /// source width, rows in order) and returns the `ow*oh*channels` output.
  /// `use_simd` selects the SIMD H-pass tier (when the host backend permits)
  /// vs the scalar reference — both must land inside the PIL parity budget.
  #[allow(clippy::too_many_arguments)]
  fn run_with<S, K>(
    kernel: K,
    src: &[S],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    channels: usize,
    use_simd: bool,
  ) -> std::vec::Vec<S>
  where
    S: super::super::FilterSample,
    K: FilterKernel,
  {
    let plan = FilteredResampler::new(ow, oh, kernel)
      .plan(sw, sh)
      .expect("valid downscale")
      .expect("non-identity");
    assert_eq!(plan.kind(), SpanKind::Filter);
    let fh = plan.filter_h().expect("filter plan carries h windows");
    let fv = plan.filter_v().expect("filter plan carries v windows");
    let mut stream = FilterStream::<S>::new(fh, fv, sw, sh, channels).expect("realistic geometry");
    let mut out = std::vec![S::default(); ow * oh * channels];
    for y in 0..sh {
      let row = &src[y * sw * channels..(y + 1) * sw * channels];
      stream
        .feed_row(y, row, use_simd, |oy, finalized| {
          out[oy * ow * channels..(oy + 1) * ow * channels].copy_from_slice(finalized);
        })
        .expect("rows in order");
    }
    out
  }

  /// Scalar-path convenience wrapper over [`run_with`] — the historical
  /// `run` signature the per-kernel golden tests call.
  fn run<S, K>(
    kernel: K,
    src: &[S],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    channels: usize,
  ) -> std::vec::Vec<S>
  where
    S: super::super::FilterSample,
    K: FilterKernel,
  {
    run_with(kernel, src, sw, sh, ow, oh, channels, false)
  }

  /// Asserts every output of `ours` is within `tol` of `golden`, folding
  /// the worst integer diff into `worst`.
  fn assert_u_within(
    ours: &[u8],
    golden: &[u8],
    tol: u8,
    worst: &mut u8,
    ctx: core::fmt::Arguments<'_>,
  ) {
    assert_eq!(ours.len(), golden.len(), "{ctx} length");
    for (i, (a, b)) in ours.iter().zip(golden.iter()).enumerate() {
      let d = a.abs_diff(*b);
      *worst = (*worst).max(d);
      assert!(d <= tol, "{ctx} idx {i}: ours {a} vs pil {b}");
    }
  }

  #[test]
  fn filter_upscale_constant_stays_constant() {
    // out > in on both axes. Each output window's coefficients sum to 1 (PIL
    // normalization), so a constant source reconstructs to that same constant
    // at every output pixel, edges included — even through the negative lobes
    // of CatmullRom and Lanczos3. Exercises the whole upscale path: the plan
    // builds (the absorption + endpoint guards must not reject scale < 1), the
    // H/V passes run, and the stream accumulates into many simultaneously open
    // outputs and flushes the trailing ones after the final source row (an
    // unflushed output would stay at the zero default and break the constant).
    for ch in [1usize, 3] {
      let src = std::vec![123u8; 3 * 3 * ch];
      for (name, out) in [
        ("Triangle", run(Triangle, &src, 3, 3, 7, 7, ch)),
        ("CatmullRom", run(CatmullRom, &src, 3, 3, 7, 7, ch)),
        ("Lanczos3", run(Lanczos3, &src, 3, 3, 7, 7, ch)),
      ] {
        assert_eq!(out.len(), 7 * 7 * ch, "{name}: upscaled output length");
        assert!(
          out.iter().all(|&v| v == 123),
          "{name}: constant {ch}-channel upscale must stay constant, got {out:?}"
        );
      }
    }
  }

  #[test]
  fn filter_upscale_plan_builds_filter_spans() {
    // The downscale-only guard is gone: an upscale on either axis (here width
    // up 8->16, height down 8->4 — a mixed plan) plans the filter, neither an
    // error nor the identity.
    let plan = FilteredResampler::new(16, 4, Triangle)
      .plan(8, 8)
      .expect("upscale width must plan")
      .expect("non-identity");
    assert_eq!(plan.kind(), SpanKind::Filter);
    assert_eq!(plan.out_dims(), (16, 4));
  }

  /// Byte-exact u8 PIL tolerance. The u8 stream resamples on PIL's 8bpc
  /// fixed-point coefficient grid (`PRECISION_BITS = 22`), so its `f64`
  /// accumulation reproduces PIL's integer `ss/2^22` and both passes'
  /// `clip8` finalize match Pillow bit-for-bit — on downscale, enlarge, and
  /// mixed ratios alike. All three kernels reach 0; this is the regression
  /// floor.
  const PIL_U8_TOL: u8 = 0;

  #[test]
  fn u8_matches_pil() {
    let mut worst = 0u8;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run(Triangle, &src, sw, sh, ow, oh, ch);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("Triangle {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run(CatmullRom, &src, sw, sh, ow, oh, ch);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("CatmullRom {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run(Lanczos3, &src, sw, sh, ow, oh, ch);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("Lanczos3 {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    std::eprintln!("PIL u8 parity: max abs diff = {worst}");
  }

  /// u16 analogue of [`assert_u_within`].
  fn assert_u16_within(
    ours: &[u16],
    golden: &[u16],
    tol: u16,
    worst: &mut u16,
    ctx: core::fmt::Arguments<'_>,
  ) {
    assert_eq!(ours.len(), golden.len(), "{ctx} length");
    for (i, (a, b)) in ours.iter().zip(golden.iter()).enumerate() {
      let d = a.abs_diff(*b);
      *worst = (*worst).max(d);
      assert!(d <= tol, "{ctx} idx {i}: ours {a} vs pil {b}");
    }
  }

  #[test]
  fn u16_matches_pil_within_one_lsb() {
    let mut worst = 0u16;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run(Triangle, &src, sw, sh, ow, oh, ch);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("Triangle {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run(CatmullRom, &src, sw, sh, ow, oh, ch);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("CatmullRom {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run(Lanczos3, &src, sw, sh, ow, oh, ch);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("Lanczos3 {sw}x{sh}->{ow}x{oh}"),
      );
    }
    std::eprintln!("PIL u16 parity: max abs diff = {worst}");
  }

  /// f32 closeness — PIL `F`-mode is unclamped and reorders the taps vs our
  /// f64 accumulation, so parity is a small relative+absolute tolerance.
  fn assert_f32_within(
    ours: &[f32],
    golden: &[f32],
    worst: &mut f32,
    ctx: core::fmt::Arguments<'_>,
  ) {
    assert_eq!(ours.len(), golden.len(), "{ctx} length");
    for (i, (a, b)) in ours.iter().zip(golden.iter()).enumerate() {
      let d = (a - b).abs();
      *worst = worst.max(d);
      let tol = 1e-3 * a.abs().max(b.abs()) + 1e-2;
      assert!(d <= tol, "{ctx} idx {i}: ours {a} vs pil {b} (d {d})");
    }
  }

  #[test]
  fn f32_matches_pil_within_tolerance() {
    let mut worst = 0.0f32;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run(Triangle, &src, sw, sh, ow, oh, ch);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("Triangle {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run(CatmullRom, &src, sw, sh, ow, oh, ch);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("CatmullRom {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run(Lanczos3, &src, sw, sh, ow, oh, ch);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("Lanczos3 {sw}x{sh}->{ow}x{oh}"),
      );
    }
    std::eprintln!("PIL f32 parity: max abs diff = {worst}");
  }

  // ---- PIL parity with the SIMD H-pass active ----------------------------
  //
  // The three tests above drive the scalar reference (`use_simd = false`).
  // These re-run the same Pillow goldens with `use_simd = true`, so the host
  // backend's SIMD H-pass kernels (NEON / SSE4.1 / AVX2 / AVX-512 / wasm,
  // whichever the runtime selects) must themselves land inside the same
  // `±1`-LSB (u8 / u16) or small-float (f32) budget — not merely match the
  // scalar path. The integer kernels are not bit-identical to scalar (their
  // tap sum reorders in `f64`), so this is the load-bearing check that the
  // reorder error never widens the Pillow gap past one LSB.

  #[test]
  fn u8_matches_pil_simd() {
    let mut worst = 0u8;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run_with(Triangle, &src, sw, sh, ow, oh, ch, true);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("Triangle SIMD {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run_with(CatmullRom, &src, sw, sh, ow, oh, ch, true);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("CatmullRom SIMD {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_U8) {
      let src = source_u8(sw, sh, ch, seed);
      let ours = run_with(Lanczos3, &src, sw, sh, ow, oh, ch, true);
      assert_u_within(
        &ours,
        golden,
        PIL_U8_TOL,
        &mut worst,
        std::format_args!("Lanczos3 SIMD {sw}x{sh}->{ow}x{oh} c{ch}"),
      );
    }
    std::eprintln!("PIL u8 parity (SIMD): max abs diff = {worst}");
  }

  #[test]
  fn u16_matches_pil_within_one_lsb_simd() {
    let mut worst = 0u16;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run_with(Triangle, &src, sw, sh, ow, oh, ch, true);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("Triangle SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run_with(CatmullRom, &src, sw, sh, ow, oh, ch, true);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("CatmullRom SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_U16) {
      let src = source_u16(sw, sh, seed);
      let ours = run_with(Lanczos3, &src, sw, sh, ow, oh, ch, true);
      assert_u16_within(
        &ours,
        golden,
        1,
        &mut worst,
        std::format_args!("Lanczos3 SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    std::eprintln!("PIL u16 parity (SIMD): max abs diff = {worst}");
  }

  #[test]
  fn f32_matches_pil_within_tolerance_simd() {
    let mut worst = 0.0f32;
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::TRIANGLE_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run_with(Triangle, &src, sw, sh, ow, oh, ch, true);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("Triangle SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::CATMULLROM_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run_with(CatmullRom, &src, sw, sh, ow, oh, ch, true);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("CatmullRom SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    for &(sw, sh, ow, oh, ch, seed, golden) in miri_subset(pil_goldens::LANCZOS3_F32) {
      let src = source_f32(sw, sh, seed);
      let ours = run_with(Lanczos3, &src, sw, sh, ow, oh, ch, true);
      assert_f32_within(
        &ours,
        golden,
        &mut worst,
        std::format_args!("Lanczos3 SIMD {sw}x{sh}->{ow}x{oh}"),
      );
    }
    std::eprintln!("PIL f32 parity (SIMD): max abs diff = {worst}");
  }
}
