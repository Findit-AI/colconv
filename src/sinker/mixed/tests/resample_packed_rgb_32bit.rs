//! Fused-downscale coverage for the 32-bit packed RGB family (`Rgb96` /
//! `Rgba128`): the wire u32 row converts to a source-width host **`u32`** RGB(A)
//! row (the `BE` swap only — NO narrow), binning runs at full `u32` precision,
//! and every output narrows only after the bin — so `rgb_u16` / `rgba_u16` are
//! the exact `u32`-domain area / filter result narrowed `>> 16`, and the
//! u8 / `luma_u16` outputs a single further `>> 8` (net `>> 24` for u8). Binning
//! at `u32` and narrowing only after is **0-ULP** for both ranges (issue #289).

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, ResampleError, Resampler, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Rgb48, Rgb96, Rgb96Row, Rgba128, Rgba128Row, rgb48_to, rgb96_to, rgba128_to},
};
use mediaframe::frame::{Rgb48Frame, Rgb96Frame, Rgba128Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native u32 slice as LE-wire byte storage.
fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as LE-wire byte storage.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Exact `u32`-domain `2x2` block mean (round-half-up over `u128`) of channel
/// `c`, narrowed `>> 16` only AFTER binning — the 0-ULP oracle the area output
/// must equal. `channels` is 3 (RGB) or 4 (RGBA).
fn area_u32_narrowed(
  staged: &[u32],
  ox: usize,
  oy: usize,
  c: usize,
  src_w: usize,
  channels: usize,
) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * src_w + ox * 2 + dx) * channels + c] as u64;
    }
  }
  (((acc + 2) / 4) >> 16) as u16
}

/// INDEPENDENT separable-`f64` filter oracle — does NOT call the production
/// `FilterStream<u32>`. It reads only the plan's `FilterAxis` windows
/// (`span(j)` → `(start, coeffs)`) and reimplements the engine's exact two-pass
/// execution from scratch, so it can disprove a bug in the `u32` coefficient
/// **selection** (full-precision vs q8), intermediate rounding, final
/// clamp/narrow, or pass order:
///
/// - **H-pass** (per source row, per output col, per channel): `acc = Σ_k
///   f64(coeff_h[k]) · src[start_h+k]` in `f64`, then the intermediate is
///   `floor(acc + 0.5)` — round-half-up, **no clamp** (the `u32` `I`-mode rule).
/// - **V-pass** (per output row): `acc = Σ_k f64(coeff_v[k]) · h_inter[start_v+k]`,
///   then `floor(acc + 0.5).clamp(0, u32::MAX)`.
/// - **narrow**: `>> 16`.
///
/// The H-then-V order, the `f64::from(f32) · (u32 as f64)` widening, and the
/// `floor(x + 0.5)` rounding match the engine's scalar `u32` path bit-for-bit
/// (the `u32` tier is scalar-only, so there is no SIMD divergence), giving
/// 0-ULP while staying independent of the resampler under test.
fn separable_triangle_u32_narrowed(
  staged: &[u32],
  channels: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let plan = FilteredResampler::new(ow, oh, Triangle)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  // H-pass → quantized (round-half-up, NO clamp) f64 intermediate, sh × ow.
  let mut h_inter = vec![0.0f64; sh * ow * channels];
  for sy in 0..sh {
    for ox in 0..ow {
      let (start, coeffs) = fh.span(ox);
      for ch in 0..channels {
        let mut acc = 0.0f64;
        for (k, &w) in coeffs.iter().enumerate() {
          acc += f64::from(w) * staged[(sy * sw + start + k) * channels + ch] as f64;
        }
        h_inter[(sy * ow + ox) * channels + ch] = (acc + 0.5).floor();
      }
    }
  }
  // V-pass → round-half-up + clamp to u32, then narrow >> 16.
  let mut out = vec![0u16; oh * ow * channels];
  for oy in 0..oh {
    let (start, coeffs) = fv.span(oy);
    for ox in 0..ow {
      for ch in 0..channels {
        let mut acc = 0.0f64;
        for (k, &w) in coeffs.iter().enumerate() {
          acc += f64::from(w) * h_inter[((start + k) * ow + ox) * channels + ch];
        }
        let v = (acc + 0.5).floor().clamp(0.0, u32::MAX as f64) as u32;
        out[(oy * ow + ox) * channels + ch] = (v >> 16) as u16;
      }
    }
  }
  out
}

/// Per-channel full-range u32 ramps; the low 16 bits vary too so binning at
/// `u32` differs from the `>> 16` narrow-first mean (the 0-ULP fix is visible).
fn packed_frame_u32() -> Vec<u32> {
  let mut buf = vec![0u32; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 0x2000_0000 + (i as u32) * 0x0123_4567;
    px[1] = 0xC000_0000u32.wrapping_sub((i as u32) * 0x0098_7654);
    px[2] = 0x1000_0000 + ((i % 8) as u32) * 0x0555_5555;
  }
  buf
}

#[test]
fn rgb96_downscale_rgb_u16_is_exact_u32_area_mean() {
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    rgb96_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          area_u32_narrowed(&host, ox, oy, c, SRC, 3),
          "({ox},{oy}) c{c} — exact u32 mean narrowed >> 16"
        );
      }
    }
  }
}

#[test]
fn rgb96_derived_outputs_come_from_binned_rgb() {
  // Every attached output — native-depth u16 and narrowed u8 — must equal what
  // a direct full-res Rgb48 sink produces over the (exact) binned u16 RGB: once
  // the binned `u32` is narrowed `>> 16` to u16, Rgb96 shares the identical
  // derivation engine, so the binned u16 is the single source of truth for the
  // narrowed outputs.
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    rgb96_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: the full-res Rgb48 sink over the (exact) binned u16 RGB.
  let binned_wire = as_le_u16(&rgb_u16);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Rgb48Frame::new(&binned_wire, OUT as u32, OUT as u32, (OUT * 3) as u32);
    let mut sink = MixedSinker::<Rgb48>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    rgb48_to(&binned, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
fn rgb96_identity_plan_matches_new_sink() {
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgb96>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    rgb96_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    rgb96_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- Rgba128 (alpha-aware area resample) ------------------------------------

/// Full-range u32 RGBA ramps with real per-pixel alpha.
fn packed_rgba_frame_u32() -> Vec<u32> {
  let mut buf = vec![0u32; SRC * SRC * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    px[0] = 0x2000_0000 + (i as u32) * 0x0123_4567;
    px[1] = 0xC000_0000u32.wrapping_sub((i as u32) * 0x0098_7654);
    px[2] = 0x1000_0000 + ((i % 8) as u32) * 0x0555_5555;
    px[3] = 0x3000_0000 + (i as u32) * 0x0222_2222;
  }
  buf
}

#[test]
fn rgba128_downscale_rgba_u16_is_exact_u32_area_mean_incl_alpha() {
  let host = packed_rgba_frame_u32();
  // Wire-encode the frame's u32 samples so the LE frame decodes via `from_le`
  // on both endiannesses; the reference below stays in the logical domain.
  let hw = as_le_u32(&host);
  let src = Rgba128Frame::new(&hw, SRC as u32, SRC as u32, (SRC * 4) as u32);

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        assert_eq!(
          rgba_u16[(oy * OUT + ox) * 4 + c],
          area_u32_narrowed(&host, ox, oy, c, SRC, 4),
          "({ox},{oy}) c{c} (alpha is c3) — exact u32 mean narrowed >> 16"
        );
      }
    }
  }
}

#[test]
fn rgba128_identity_plan_matches_new_sink() {
  let host = packed_rgba_frame_u32();
  let src = Rgba128Frame::new(&host, SRC as u32, SRC as u32, (SRC * 4) as u32);

  let mut direct = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Rgba128>::new(SRC, SRC)
      .with_rgba_u16(&mut direct)
      .unwrap();
    rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_u16(&mut via_area)
        .unwrap();
    rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb96_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB-`u32`-scratch grow failpoint: Rgb96 stages its
    // wire row into the host-`u32` scratch, so the row-0 area stream box builds OK
    // (into a local) and the scratch grow — which runs AFTER the stream build and
    // BEFORE the field insert — refuses, surfacing AllocationFailed with BOTH the
    // stream field `None` and the freeze uncommitted (the commit-together atomic
    // shape; the pre-fix ordering would have left the stream committed).
    crate::sinker::mixed::arm_source_rgb_u32_scratch_failure();
    let err = sink
      .process(Rgb96Row::new(
        &wire[..SRC * 3],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the `u32` area stream
    // field must be `None` — the fix builds it into a LOCAL and inserts only after
    // the scratch grows, so a failed scratch leaves nothing committed. The
    // insert-before-scratch shape would leave it `Some` here, so this assertion FAILS
    // against that shape and PASSES against the fix (a `try_box` box failure cannot
    // make this distinction — it fails before any stream exists to insert).
    assert!(
      !sink.rgb_stream_u32_allocated(),
      "a scratch OOM must leave the rgb_stream_u32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgb96Row::new(
          &wire[r * SRC * 3..(r + 1) * SRC * 3],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgb_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba128_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = packed_rgba_frame_u32();
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot 4-channel `u32` SCRATCH grow failpoint: the row-0 area
    // stream box builds OK (into a local), then the `source_rgba_u32_scratch` grow —
    // which runs AFTER the stream build and BEFORE the field insert — refuses,
    // surfacing AllocationFailed with the rgba_stream_u32 field `None` and the freeze
    // uncommitted (the commit-together atomic shape). This exercises the exact
    // scratch-after-stream ordering boundary a box failure cannot pin.
    crate::sinker::mixed::arm_source_rgba_u32_scratch_failure();
    let err = sink
      .process(Rgba128Row::new(
        &host[..SRC * 4],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the 4-channel `u32` area
    // stream field must be `None` — built into a LOCAL and inserted only after the
    // scratch grows, so a failed scratch leaves nothing committed. The
    // insert-before-scratch shape would leave it `Some` here, so this assertion FAILS
    // against that shape and PASSES against the fix (a `try_box` box failure cannot
    // make this distinction).
    assert!(
      !sink.rgba_stream_u32_allocated(),
      "a scratch OOM must leave the rgba_stream_u32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgba128Row::new(
          &host[r * SRC * 4..(r + 1) * SRC * 4],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgba_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real rgba_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

// ---- 0-ULP parity-vs-u32-domain resample pins (issue #289, closed) ----------
// Binning at native `u32` and narrowing `>> 16` only AFTER the bin is 0-ULP for
// BOTH ranges (`rgb_u16` / `rgba_u16` are pure colour — range-independent — so a
// single oracle serves FR true and false). A single host-native fixture is
// re-encoded LE / BE so both endian arms decode the same logical values and
// must produce the identical 0-ULP output (area: exact u32 mean narrowed;
// filter: an INDEPENDENT separable-f64 Triangle oracle narrowed — see
// `separable_triangle_u32_narrowed`).

use mediaframe::frame::{Rgb96BeFrame, Rgba128BeFrame};

const FRP: usize = 4; // source side
const FRO: usize = 2; // output side

fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Nonzero low-16-bit u32 RGB ramp so the post-bin `>> 16` narrow is lossy.
fn frp_rgb96() -> Vec<u32> {
  (0..FRP * FRP * 3)
    .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(0xABCD))
    .collect()
}
/// Nonzero low-16-bit u32 RGBA ramp.
fn frp_rgba128() -> Vec<u32> {
  (0..FRP * FRP * 4)
    .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(0xBEEF))
    .collect()
}

#[test]
fn rgb96_fr_false_area_rgb_u16_is_u32_domain_mean() {
  let intended = frp_rgb96();
  let mut expected = vec![0u16; FRO * FRO * 3];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..3 {
        expected[(oy * FRO + ox) * 3 + c] = area_u32_narrowed(&intended, ox, oy, c, FRP, 3);
      }
    }
  }
  // LE arm
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96Frame::new(&le, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(FRP, FRP, AreaResampler::to(FRO, FRO))
        .unwrap()
        .with_rgb_u16(&mut out_le)
        .unwrap();
    rgb96_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  // BE arm
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink = MixedSinker::<Rgb96<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgb_u16(&mut out_be)
    .unwrap();
    crate::source::rgb96_to_endian::<_, true>(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(out_le, expected, "Rgb96 FR=false area rgb_u16 LE (0-ULP)");
  assert_eq!(out_be, expected, "Rgb96 FR=false area rgb_u16 BE (0-ULP)");
}

#[test]
fn rgba128_fr_false_area_rgba_u16_is_u32_domain_mean() {
  let intended = frp_rgba128();
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = area_u32_narrowed(&intended, ox, oy, c, FRP, 4);
      }
    }
  }
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128Frame::new(&le, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(FRP, FRP, AreaResampler::to(FRO, FRO))
        .unwrap()
        .with_rgba_u16(&mut out_le)
        .unwrap();
    rgba128_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink = MixedSinker::<Rgba128<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::rgba128_to_endian::<_, true>(&src, false, KernelMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(
    out_le, expected,
    "Rgba128 FR=false area rgba_u16 LE (0-ULP)"
  );
  assert_eq!(
    out_be, expected,
    "Rgba128 FR=false area rgba_u16 BE (0-ULP)"
  );
}

#[test]
fn rgb96_fr_false_filter_all_outputs_is_u32_domain() {
  let intended = frp_rgb96();
  // Independent oracle for the binned RGB; narrowed derives from a direct Rgb48.
  let oracle_rgb = separable_triangle_u32_narrowed(&intended, 3, FRP, FRP, FRO, FRO);
  let (ref_rgb, ref_luma, ref_lu16, ref_h, ref_s, ref_v) =
    rgb48_derive_ref(&oracle_rgb, FRO, false);

  for be in [false, true] {
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut luma = vec![0u8; FRO * FRO];
    let mut luma_u16 = vec![0u16; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];
    if be {
      let wire = as_be_u32(&intended);
      let src = Rgb96BeFrame::new(&wire, FRP as u32, FRP as u32, (FRP * 3) as u32);
      let mut sink = MixedSinker::<Rgb96<true>, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      crate::source::rgb96_to_endian::<_, true>(&src, false, KernelMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let wire = as_le_u32(&intended);
      let src = Rgb96Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 3) as u32);
      let mut sink = MixedSinker::<Rgb96, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      rgb96_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(rgb_u16, oracle_rgb, "Rgb96 filter rgb_u16 {tag} (0-ULP)");
    assert_eq!(rgb, ref_rgb, "Rgb96 filter rgb {tag}");
    assert_eq!(luma, ref_luma, "Rgb96 filter luma {tag}");
    assert_eq!(luma_u16, ref_lu16, "Rgb96 filter luma_u16 {tag}");
    assert_eq!(h, ref_h, "Rgb96 filter hsv-h {tag}");
    assert_eq!(s_, ref_s, "Rgb96 filter hsv-s {tag}");
    assert_eq!(v_, ref_v, "Rgb96 filter hsv-v {tag}");
  }
}

#[test]
fn rgba128_fr_false_filter_straight_4ch_all_outputs() {
  // Straight-4ch filter route (rgba attached): independent oracle for the binned
  // RGBA; EVERY output (rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, hsv).
  let intended = frp_rgba128();
  let oracle_rgba = separable_triangle_u32_narrowed(&intended, 4, FRP, FRP, FRO, FRO);
  let oracle_rgb = drop_alpha_u16(&oracle_rgba);
  let oracle_rgba_u8 = narrow_u16(&oracle_rgba);
  let (ref_rgb, ref_luma, ref_lu16, ref_h, ref_s, ref_v) =
    rgb48_derive_ref(&oracle_rgb, FRO, false);

  for be in [false, true] {
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut rgba = vec![0u8; FRO * FRO * 4];
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgba_u16 = vec![0u16; FRO * FRO * 4];
    let mut luma = vec![0u8; FRO * FRO];
    let mut luma_u16 = vec![0u16; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];
    if be {
      let wire = as_be_u32(&intended);
      let src = Rgba128BeFrame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128<true>, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      crate::source::rgba128_to_endian::<_, true>(&src, false, KernelMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let wire = as_le_u32(&intended);
      let src = Rgba128Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      rgba128_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(
      rgba_u16, oracle_rgba,
      "Rgba128 4ch filter rgba_u16 {tag} (0-ULP)"
    );
    assert_eq!(rgb_u16, oracle_rgb, "Rgba128 4ch filter rgb_u16 {tag}");
    assert_eq!(rgba, oracle_rgba_u8, "Rgba128 4ch filter rgba (u8) {tag}");
    assert_eq!(rgb, ref_rgb, "Rgba128 4ch filter rgb (u8) {tag}");
    assert_eq!(luma, ref_luma, "Rgba128 4ch filter luma {tag}");
    assert_eq!(luma_u16, ref_lu16, "Rgba128 4ch filter luma_u16 {tag}");
    assert_eq!(h, ref_h, "Rgba128 4ch filter hsv-h {tag}");
    assert_eq!(s_, ref_s, "Rgba128 4ch filter hsv-s {tag}");
    assert_eq!(v_, ref_v, "Rgba128 4ch filter hsv-v {tag}");
  }
}

// ---- Premultiplied-alpha area resample (0-ULP, u64-intermediate) -------------
// Under `AlphaMode::Premultiplied` the engine premultiplies the source at `u32`
// (`round(c·α / u32::MAX)`, `u64` intermediates), area-bins at `u32`, then
// un-premultiplies the binned row (`round(pm·MAX / α)`, 0 when binned α = 0)
// before the `>> 16` narrow + derive. The flipped #289 pins exercise the
// STRAIGHT paths only; this oracle independently reproduces that exact `u64`
// premult / unpremult chain and asserts equality (0-ULP), covering varying
// alpha, the alpha = 0 transparent edge, and LE + BE.

const PMAX: u64 = u32::MAX as u64;

/// Premultiplied-area `u32` oracle for the canonical `R, G, B, A` `u32` source
/// `src` (`src_w x src_w`): premultiply each pixel at `u32`, area-bin each 2x2
/// block (round-half-up over `u128`), zero-α-safe un-premultiply, then narrow
/// `>> 16` — the exact `u64`-intermediate chain the engine runs. Returns the
/// `out_w x out_w` straight `R, G, B, A` `u16`.
fn premult_area_rgba_u16(src: &[u32], src_w: usize, out_w: usize) -> Vec<u16> {
  let n = src_w * src_w;
  let mut pm = vec![0u64; n * 4];
  for i in 0..n {
    let a = src[i * 4 + 3] as u64;
    for c in 0..3 {
      pm[i * 4 + c] = (src[i * 4 + c] as u64 * a + PMAX / 2) / PMAX;
    }
    pm[i * 4 + 3] = a;
  }
  let mut out = vec![0u16; out_w * out_w * 4];
  for oy in 0..out_w {
    for ox in 0..out_w {
      let mut binned = [0u64; 4];
      for (c, b) in binned.iter_mut().enumerate() {
        let mut acc = 0u64;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += pm[((oy * 2 + dy) * src_w + ox * 2 + dx) * 4 + c];
          }
        }
        *b = (acc + 2) / 4;
      }
      let a = binned[3];
      for c in 0..3 {
        // The exact `checked_div` expression the engine's un-premultiply uses
        // (`None` ⇔ α == 0 ⇒ a transparent binned pixel exposes no colour).
        let straight = (binned[c] * PMAX + a / 2)
          .checked_div(a)
          .map_or(0, |q| q.min(PMAX));
        out[(oy * out_w + ox) * 4 + c] = (straight >> 16) as u16;
      }
      out[(oy * out_w + ox) * 4 + 3] = (a >> 16) as u16;
    }
  }
  out
}

/// Drop alpha from a packed `R, G, B, A` `u16` row to `R, G, B`.
fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  rgba
    .chunks_exact(4)
    .flat_map(|px| [px[0], px[1], px[2]])
    .collect()
}

/// Narrow every `u16` `>> 8`.
fn narrow_u16(v: &[u16]) -> Vec<u8> {
  v.iter().map(|&x| (x >> 8) as u8).collect()
}

/// Canonical `R, G, B, A` `u32` ramp with one fully-transparent (α = 0) 2x2
/// output block (top-left), so the un-premultiply's `α == 0` branch is hit.
fn premult_rgba128_fixture() -> Vec<u32> {
  let mut v = frp_rgba128();
  for &(sx, sy) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
    v[(sy * FRP + sx) * 4 + 3] = 0;
  }
  v
}

#[test]
fn rgba128_premult_area_matches_u32_premult_oracle() {
  let host = premult_rgba128_fixture();
  let oracle = premult_area_rgba_u16(&host, FRP, FRO);
  let straight_rgb = drop_alpha_u16(&oracle);

  // Direct full-res Rgb48 sink over the straight (un-premultiplied) binned RGB —
  // the reference for the narrowed u8 rgb / luma / hsv derives.
  let rgb_wire = as_le_u16(&straight_rgb);
  let mut ref_rgb = vec![0u8; FRO * FRO * 3];
  let mut ref_luma = vec![0u8; FRO * FRO];
  let mut ref_h = vec![0u8; FRO * FRO];
  let mut ref_s = vec![0u8; FRO * FRO];
  let mut ref_v = vec![0u8; FRO * FRO];
  {
    let binned = Rgb48Frame::new(&rgb_wire, FRO as u32, FRO as u32, (FRO * 3) as u32);
    let mut sink = MixedSinker::<Rgb48>::new(FRO, FRO)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    rgb48_to(&binned, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  for be in [false, true] {
    let mut rgba_u16 = vec![0u16; FRO * FRO * 4];
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgba = vec![0u8; FRO * FRO * 4];
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut luma = vec![0u8; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];
    if be {
      let wire = as_be_u32(&host);
      let src = Rgba128BeFrame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128<true>, AreaResampler>::with_resampler(
        FRP,
        FRP,
        AreaResampler::to(FRO, FRO),
      )
      .unwrap()
      .with_alpha_mode(AlphaMode::Premultiplied)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      crate::source::rgba128_to_endian::<_, true>(&src, true, KernelMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let wire = as_le_u32(&host);
      let src = Rgba128Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128, AreaResampler>::with_resampler(
        FRP,
        FRP,
        AreaResampler::to(FRO, FRO),
      )
      .unwrap()
      .with_alpha_mode(AlphaMode::Premultiplied)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    // Core premult/unpremult chain.
    assert_eq!(rgba_u16, oracle, "Rgba128 premult rgba_u16 {tag}");
    assert_eq!(rgb_u16, straight_rgb, "Rgba128 premult rgb_u16 {tag}");
    assert_eq!(rgba, narrow_u16(&oracle), "Rgba128 premult rgba (u8) {tag}");
    // Transparent edge: the α = 0 block exposes no colour (no bleed).
    assert_eq!(
      &rgba_u16[..4],
      &[0, 0, 0, 0],
      "Rgba128 transparent bled {tag}"
    );
    // Narrowed derives from the un-premultiplied straight colour.
    assert_eq!(rgb, ref_rgb, "Rgba128 premult rgb (u8) {tag}");
    assert_eq!(luma, ref_luma, "Rgba128 premult luma {tag}");
    assert_eq!(h, ref_h, "Rgba128 premult hsv-h {tag}");
    assert_eq!(s_, ref_s, "Rgba128 premult hsv-s {tag}");
    assert_eq!(v_, ref_v, "Rgba128 premult hsv-v {tag}");
  }
}

// ---- Straight-alpha DROP-alpha 3-channel route (Rgba128, area + filter) ------
// When straight alpha and the caller drops α (only `rgb` / `rgb_u16` / `luma` /
// `hsv` attached, no `rgba(_u16)`), Rgba128 takes the SEPARATE 3-channel route
// (`rgba128_to_rgb_u32_row` drop-α staging → `packed_rgb_u32_resample_emit`),
// distinct from the 4-channel and premult pins above. Pin `rgb_u16` against the
// 3-channel u32-domain oracle (staging + bin + narrow), and the narrowed
// derives against a direct full-res `Rgb48` sink over that binned RGB.

/// Direct full-res `Rgb48` sink over a binned `R, G, B` u16 row → the narrowed
/// `(rgb_u8, luma, luma_u16, h, s, v)` derives. `Rgb48`'s `luma_u16` is the
/// 8-bit-precision (narrowed) flavour, matching the `NATIVE_LUMA16 = false`
/// packed-RGB route (see `rgb96_derived_outputs_come_from_binned_rgb`).
/// `(rgb_u8, luma_u8, luma_u16, h, s, v)` narrowed derives.
type Rgb48Derives = (Vec<u8>, Vec<u8>, Vec<u16>, Vec<u8>, Vec<u8>, Vec<u8>);

fn rgb48_derive_ref(rgb_u16: &[u16], n: usize, full_range: bool) -> Rgb48Derives {
  let wire = as_le_u16(rgb_u16);
  let src = Rgb48Frame::new(&wire, n as u32, n as u32, (n * 3) as u32);
  let mut rgb = vec![0u8; n * n * 3];
  let mut luma = vec![0u8; n * n];
  let mut luma_u16 = vec![0u16; n * n];
  let mut h = vec![0u8; n * n];
  let mut s = vec![0u8; n * n];
  let mut v = vec![0u8; n * n];
  {
    let mut sink = MixedSinker::<Rgb48>::new(n, n)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    rgb48_to(&src, full_range, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  (rgb, luma, luma_u16, h, s, v)
}

/// Drop α from a canonical `R, G, B, A` u32 row → packed `R, G, B` u32 — the
/// engine's `rgba128_to_rgb_u32_row` staging (no narrow), the oracle input.
fn rgba128_drop_rgb_u32(host: &[u32]) -> Vec<u32> {
  host
    .chunks_exact(4)
    .flat_map(|px| [px[0], px[1], px[2]])
    .collect()
}

#[test]
fn rgba128_drop_alpha_area_route_is_u32_domain() {
  let host = frp_rgba128();
  let rgb_src = rgba128_drop_rgb_u32(&host);
  let mut oracle_rgb_u16 = vec![0u16; FRO * FRO * 3];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..3 {
        oracle_rgb_u16[(oy * FRO + ox) * 3 + c] = area_u32_narrowed(&rgb_src, ox, oy, c, FRP, 3);
      }
    }
  }
  let (ref_rgb, ref_luma, ref_lu16, ref_h, ref_s, ref_v) =
    rgb48_derive_ref(&oracle_rgb_u16, FRO, true);

  for be in [false, true] {
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut luma = vec![0u8; FRO * FRO];
    let mut luma_u16 = vec![0u16; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];
    if be {
      let wire = as_be_u32(&host);
      let src = Rgba128BeFrame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128<true>, AreaResampler>::with_resampler(
        FRP,
        FRP,
        AreaResampler::to(FRO, FRO),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      crate::source::rgba128_to_endian::<_, true>(&src, true, KernelMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let wire = as_le_u32(&host);
      let src = Rgba128Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128, AreaResampler>::with_resampler(
        FRP,
        FRP,
        AreaResampler::to(FRO, FRO),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(rgb_u16, oracle_rgb_u16, "Rgba128 drop-α area rgb_u16 {tag}");
    assert_eq!(rgb, ref_rgb, "Rgba128 drop-α area rgb {tag}");
    assert_eq!(luma, ref_luma, "Rgba128 drop-α area luma {tag}");
    assert_eq!(luma_u16, ref_lu16, "Rgba128 drop-α area luma_u16 {tag}");
    assert_eq!(h, ref_h, "Rgba128 drop-α area hsv-h {tag}");
    assert_eq!(s_, ref_s, "Rgba128 drop-α area hsv-s {tag}");
    assert_eq!(v_, ref_v, "Rgba128 drop-α area hsv-v {tag}");
  }
}

#[test]
fn rgba128_drop_alpha_filter_route_is_u32_domain() {
  let host = frp_rgba128();
  let rgb_src = rgba128_drop_rgb_u32(&host);
  let oracle_rgb_u16 = separable_triangle_u32_narrowed(&rgb_src, 3, FRP, FRP, FRO, FRO);
  let (ref_rgb, ref_luma, ref_lu16, ref_h, ref_s, ref_v) =
    rgb48_derive_ref(&oracle_rgb_u16, FRO, true);

  for be in [false, true] {
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut luma = vec![0u8; FRO * FRO];
    let mut luma_u16 = vec![0u16; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];
    if be {
      let wire = as_be_u32(&host);
      let src = Rgba128BeFrame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128<true>, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      crate::source::rgba128_to_endian::<_, true>(&src, true, KernelMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let wire = as_le_u32(&host);
      let src = Rgba128Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
      let mut sink = MixedSinker::<Rgba128, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s_, &mut v_)
      .unwrap();
      rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(
      rgb_u16, oracle_rgb_u16,
      "Rgba128 drop-α filter rgb_u16 {tag}"
    );
    assert_eq!(rgb, ref_rgb, "Rgba128 drop-α filter rgb {tag}");
    assert_eq!(luma, ref_luma, "Rgba128 drop-α filter luma {tag}");
    assert_eq!(luma_u16, ref_lu16, "Rgba128 drop-α filter luma_u16 {tag}");
    assert_eq!(h, ref_h, "Rgba128 drop-α filter hsv-h {tag}");
    assert_eq!(s_, ref_s, "Rgba128 drop-α filter hsv-s {tag}");
    assert_eq!(v_, ref_v, "Rgba128 drop-α filter hsv-v {tag}");
  }
}

#[test]
fn rgba128_premult_filter_rejects() {
  // Premultiplied alpha has no filter analogue: the Filter arm routes to the
  // area tail, which rejects with the typed `UnsupportedFilter` (no silent
  // corruption). The one remaining distinct Rgba128 branch.
  let host = frp_rgba128();
  let wire = as_le_u32(&host);
  let src = Rgba128Frame::new(&wire, FRP as u32, FRP as u32, (FRP * 4) as u32);
  let mut rgba_u16 = vec![0u16; FRO * FRO * 4];
  let mut sink = MixedSinker::<Rgba128, FilteredResampler<Triangle>>::with_resampler(
    FRP,
    FRP,
    FilteredResampler::new(FRO, FRO, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba_u16(&mut rgba_u16)
  .unwrap();
  let err = rgba128_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(crate::resample::ResampleError::UnsupportedFilter(_))
    ),
    "Rgba128 premult+filter must reject: {err:?}"
  );
}
