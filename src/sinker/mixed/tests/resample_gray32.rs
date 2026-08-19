//! Fused-downscale coverage for `Gray32` — the full-bit (u32) integer twin of
//! `Gray16`. The wire row converts to a host-native **`u32`** luma plane (the
//! wire `BE` swap only, no depth narrow), then a single 1-channel
//! `AreaStream<u32>` / `FilterStream<u32>` resamples that plane at **native
//! `u32` precision** and every attached output derives from each finalized
//! `u32` luma row using the very `gray32_to_*` kernels the direct path uses,
//! narrowing only afterwards. So every resampled output equals the direct
//! Gray32 sink run over a frame that already holds the `u32`-binned luma — a
//! 0-ULP parity-vs-direct for **both** `full_range` true and false (closes
//! issue #289; the prior `>> 16`-narrow-first staging was ≤1 LSB off the exact
//! `u32`-domain mean).

use crate::{
  KernelMatrix, PixelSink,
  frame::Gray32Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray32, Gray32Row, gray32_to, gray32_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native u32 slice as LE-encoded byte storage (the
/// `gray32le` plane contract), recovered via `u32::from_le`.
fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u32 slice as BE-encoded byte storage (the
/// `gray32be` plane contract), recovered via `u32::from_be`.
fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// A 32-bit luma ramp with a non-trivial low half so the `>> 16` narrow and the
/// u16-precision area mean both see real per-2x2-block variation.
fn ramp() -> Vec<u32> {
  let mut y = vec![0u32; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = ((i as u32).wrapping_mul(50_529_027)) ^ 0x0F0F_0F0F;
  }
  y
}

/// Exact 2x2-block area mean (round-half-up) of the **full `u32`** plane — the
/// native-`u32`-domain binned luma the Gray32 area resample produces (binning
/// the full samples, not their `>> 16` narrow). Each output then narrows only
/// in the derive, so this is the 0-ULP oracle the resampled `luma_u16` must
/// equal once narrowed `>> 16`.
fn block_mean_2x2_u32(plane: &[u32]) -> Vec<u32> {
  let mut out = vec![0u32; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u64;
      for dy in 0..2 {
        for dx in 0..2 {
          s += u64::from(plane[(oy * 2 + dy) * SRC + ox * 2 + dx]);
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u32;
    }
  }
  out
}

/// The binned `u32` plane narrowed `>> 16` — the resampled `luma_u16` (direct
/// `luma_u16` semantics applied **after** binning).
fn narrowed_u16_from_u32(binned: &[u32]) -> Vec<u16> {
  binned.iter().map(|&v| (v >> 16) as u16).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_downscale_luma_u16_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray32_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    luma_u16,
    narrowed_u16_from_u32(&block_mean_2x2_u32(&plane)),
    "luma_u16 must be the exact native-u32 2x2 block mean, narrowed >> 16 only after binning"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_all_outputs_match_direct_over_binned_luma() {
  // Every attached output must equal the direct Gray32 sink run over the
  // native-`u32`-binned luma frame — the 0-ULP resample contract (bin the
  // full `u32`, narrow only in each per-output derive).
  let plane = ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    gray32_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let binned = block_mean_2x2_u32(&plane);
  let binned_pix = as_le_u32(&binned);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Gray32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray32_to(&binned_frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_le_be_resample_outputs_identical() {
  // The binned luma is host-native, so the derive kernels must run with
  // `HOST_NATIVE_BE`. The LE-vs-BE parity catches a wrong const on either host:
  // LE and BE wire encodings of the same logical plane must resample identically.
  let plane = ramp();
  let pix_le = as_le_u32(&plane);
  let pix_be = as_be_u32(&plane);

  let mut le_luma_u16 = vec![0u16; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray32Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut le_luma_u16)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    gray32_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let mut be_luma_u16 = vec![0u16; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray32Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Gray32<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    gray32_to_endian::<_, true>(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  assert_eq!(
    le_luma_u16,
    narrowed_u16_from_u32(&block_mean_2x2_u32(&plane)),
    "luma_u16 not the native-u32 area mean narrowed >> 16"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_out_of_sequence_first_row_rejected_before_allocation() {
  // Feeding row 1 before row 0 is rejected with OutOfSequenceRow, before any
  // stream allocation — the atomic-preflight contract shared with Gray16.
  let pix = as_le_u32(&ramp());
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let g = &pix[SRC..2 * SRC];
  let err = sink.process(Gray32Row::for_tests(g, 1, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

// ---- Filter-plan routing ----------------------------------------------------
//
// The filter path stages the wire row as a host-native `u32` luma plane (the
// wire `BE` swap only, no depth narrow) and resamples it through the
// signed-coefficient single-channel `FilterStream<u32>` (the filter twin of the
// area bin). Gray32 carries the full native `u32` range, so the
// `FilterStream<u32>`'s `0..=u32::MAX` clamp *is* the native clamp — no extra
// clamp. So the filter `luma_u16` must equal a single-channel
// `FilterStream<u32>` resample of the source plane, narrowed `>> 16` only
// after binning (0-ULP).

use crate::resample::{
  CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
};

const FW: usize = 8;
const FH: usize = 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

/// A u32 ramp with a hard mid-column edge plus two textured interior rows so
/// filter windows straddling the edge produce real intermediate values,
/// exercising the full low half the native-`u32` filter stream must carry.
fn filter_ramp() -> Vec<u32> {
  let mut y = vec![0u32; FW * FH];
  for row in 0..FH {
    for col in 0..FW {
      y[row * FW + col] = if col < FW / 2 {
        0x1234_5678
      } else {
        0xC0DE_4321
      };
    }
  }
  for col in 0..FW {
    y[4 * FW + col] = (col as u32 * 8000) << 16;
    y[5 * FW + col] = (60000 - col as u32 * 8000) << 16;
  }
  y
}

/// Single-channel `FilterStream<u32>` resample of the host-native `u32` luma
/// plane — the Gray32 native-`u32` filter oracle (full `u32` range, so the
/// engine clamp is native). Returns the **raw binned `u32`**; each output
/// narrows only afterwards.
fn native_luma_filter_u32<K: FilterKernel>(
  kernel: K,
  luma_plane: &[u32],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u32> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u32>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u32; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(
        row,
        &luma_plane[row * sw..(row + 1) * sw],
        true,
        |oy, fin| {
          out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
        },
      )
      .expect("rows in order");
  }
  out
}

fn gray32_filter_luma_u16<K: FilterKernel + Copy>(ow: usize, oh: usize, kernel: K) -> Vec<u16> {
  let plane = filter_ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, FW as u32, FH as u32, FW as u32);
  let mut luma_u16 = vec![0u16; ow * oh];
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut luma = vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Gray32, FilteredResampler<K>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap();
    gray32_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  // Derived outputs follow from the resampled luma exactly as the gray32
  // derive kernels do: luma = luma_u16 >> 8, rgb broadcasts that byte.
  for (i, &lv) in luma_u16.iter().enumerate() {
    assert_eq!(
      luma[i],
      (lv >> 8) as u8,
      "luma must be luma_u16 >> 8 at {i}"
    );
    assert_eq!(rgb[i * 3], (lv >> 8) as u8, "rgb broadcasts luma_u16 >> 8");
  }
  luma_u16
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_filter_luma_u16_matches_native_oracle() {
  // Oracle: native-`u32` filter of the full source plane, narrowed `>> 16`
  // only after binning — the 0-ULP contract.
  let plane = filter_ramp();
  for (ow, oh) in [(FOUT_DOWN, FOUT_DOWN), (FUP, FUP)] {
    let tri = gray32_filter_luma_u16(ow, oh, Triangle);
    assert_eq!(
      tri,
      narrowed_u16_from_u32(&native_luma_filter_u32(Triangle, &plane, FW, FH, ow, oh)),
      "Triangle {ow}x{oh}"
    );
    let cat = gray32_filter_luma_u16(ow, oh, CatmullRom);
    assert_eq!(
      cat,
      narrowed_u16_from_u32(&native_luma_filter_u32(CatmullRom, &plane, FW, FH, ow, oh)),
      "CatmullRom {ow}x{oh}"
    );
    let lan = gray32_filter_luma_u16(ow, oh, Lanczos3);
    assert_eq!(
      lan,
      narrowed_u16_from_u32(&native_luma_filter_u32(Lanczos3, &plane, FW, FH, ow, oh)),
      "Lanczos3 {ow}x{oh}"
    );
  }
}

// ---- Limited-range (full_range=false) 0-ULP parity-vs-direct (issue #289) ----
//
// Closing #289, the limited-range Gray32 resample is now exact: binning at
// native `u32` and narrowing only in each per-output derive is byte-identical
// to the direct Gray32 limited sink run over the `u32`-binned luma. (The old
// narrow-first staging dropped the low 16 bits ahead of the limited-range
// affine, ≤1 LSB off — the gap is now closed.) Verified for area + filter,
// LE + BE, every output.

/// Resample a Gray32 `full_range = false` source (`host`, as host-native u32)
/// through `resampler` (wire `BE`) with every output attached, then assert
/// each output equals the **direct** Gray32 limited sink run over `binned` —
/// the native-`u32`-binned luma the engine produces (block mean for area,
/// `FilterStream<u32>` for filter). A clean parity-vs-direct: the resample is
/// 0-ULP, no longer the narrow-first ≤1-LSB approximation.
fn pin_limited_exact_over_u32_binned<R: Resampler, const BE: bool>(
  host: &[u32],
  binned: &[u32],
  resampler: R,
  label: &str,
) {
  const PSRC: usize = 8;
  const POUT: usize = 4;
  let wire = if BE { as_be_u32(host) } else { as_le_u32(host) };
  let frame = Gray32Frame::<BE>::new(&wire, PSRC as u32, PSRC as u32, PSRC as u32);
  let mut rgb = vec![0u8; POUT * POUT * 3];
  let mut rgb_u16 = vec![0u16; POUT * POUT * 3];
  let mut rgba = vec![0u8; POUT * POUT * 4];
  let mut rgba_u16 = vec![0u16; POUT * POUT * 4];
  let mut luma = vec![0u8; POUT * POUT];
  let mut luma_u16 = vec![0u16; POUT * POUT];
  let mut h = vec![0u8; POUT * POUT];
  let mut s = vec![0u8; POUT * POUT];
  let mut v = vec![0u8; POUT * POUT];
  {
    let mut sink = MixedSinker::<Gray32<BE>, R>::with_resampler(PSRC, PSRC, resampler)
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
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    // full_range = false — the limited-range path under test.
    gray32_to_endian::<_, BE>(&frame, false, sink.set_kernel_matrix(M)).unwrap();
  }

  // Reference: the direct (identity-plan) Gray32 limited sink over the
  // `u32`-binned luma frame. Equality proves the resample bins at native `u32`
  // and derives each output identically — 0-ULP, no narrow-first gap.
  let binned_wire = if BE {
    as_be_u32(binned)
  } else {
    as_le_u32(binned)
  };
  let ref_frame = Gray32Frame::<BE>::new(&binned_wire, POUT as u32, POUT as u32, POUT as u32);
  let mut ref_rgb = vec![0u8; POUT * POUT * 3];
  let mut ref_rgb_u16 = vec![0u16; POUT * POUT * 3];
  let mut ref_rgba = vec![0u8; POUT * POUT * 4];
  let mut ref_rgba_u16 = vec![0u16; POUT * POUT * 4];
  let mut ref_luma = vec![0u8; POUT * POUT];
  let mut ref_luma_u16 = vec![0u16; POUT * POUT];
  let mut ref_h = vec![0u8; POUT * POUT];
  let mut ref_s = vec![0u8; POUT * POUT];
  let mut ref_v = vec![0u8; POUT * POUT];
  {
    let mut ref_sink = MixedSinker::<Gray32<BE>>::new(POUT, POUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
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
    gray32_to_endian::<_, BE>(&ref_frame, false, ref_sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "{label} rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "{label} rgb_u16");
  assert_eq!(rgba, ref_rgba, "{label} rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "{label} rgba_u16");
  assert_eq!(luma, ref_luma, "{label} luma");
  assert_eq!(luma_u16, ref_luma_u16, "{label} luma_u16");
  assert_eq!(h, ref_h, "{label} h");
  assert_eq!(s, ref_s, "{label} s");
  assert_eq!(v, ref_v, "{label} v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_limited_resample_is_exact_u32_domain() {
  // Constant source with nonzero low 16 bits near a rounding threshold — the
  // case the narrow-first staging mis-rounded at limited range (#289).
  const V: u32 = 0x106d_edee;
  let host = vec![V; 8 * 8];
  // Area engine, LE + BE: the u32 block mean is the binned oracle.
  let area_binned = block_mean_2x2_u32(&host);
  pin_limited_exact_over_u32_binned::<_, false>(
    &host,
    &area_binned,
    AreaResampler::to(4, 4),
    "area LE",
  );
  pin_limited_exact_over_u32_binned::<_, true>(
    &host,
    &area_binned,
    AreaResampler::to(4, 4),
    "area BE",
  );
  // Filter engine (Triangle), LE + BE: a standalone FilterStream<u32> is the
  // binned oracle (a constant need not filter to itself in f64, so use the
  // actual native-u32 filter result).
  let filt_binned = native_luma_filter_u32(Triangle, &host, 8, 8, 4, 4);
  pin_limited_exact_over_u32_binned::<_, false>(
    &host,
    &filt_binned,
    FilteredResampler::new(4, 4, Triangle),
    "filter LE",
  );
  pin_limited_exact_over_u32_binned::<_, true>(
    &host,
    &filt_binned,
    FilteredResampler::new(4, 4, Triangle),
    "filter BE",
  );
}
