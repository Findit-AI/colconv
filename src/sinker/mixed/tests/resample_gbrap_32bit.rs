//! Fused-downscale coverage for the 32-bit planar GBR + alpha (`Gbrap32`)
//! family: the four `u32` planes are de-interleaved to a source-width host
//! **`u32`** RGBA row (the `BE` swap only — NO narrow), binning runs at full
//! `u32` precision, and every output narrows only after the bin — so the
//! native-depth `rgba_u16` output is the exact `u32`-domain area / filter
//! result narrowed `>> 16` (including alpha), **0-ULP** for both ranges (closes
//! issue #289). This mirrors `Rgba128`'s resample coverage at a planar source.

use crate::{
  ColorMatrix,
  resample::{AreaResampler, FilteredResampler, Resampler, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Rgb48, rgb48_to},
};
use mediaframe::frame::Rgb48Frame;

const SRC: usize = 8;
const OUT: usize = 4;

fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// One `u32` plane ramp with nonzero low-16 bits so the post-bin `>> 16` narrow
/// is genuinely lossy (matching the format contract); binning at `u32` keeps
/// those low bits, so a downscale is 0-ULP versus the exact `u32`-domain mean.
fn plane_u32(seed: u32) -> Vec<u32> {
  (0..SRC * SRC)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0xABCD))
    .collect()
}

/// Stage the four planes to the canonical host-native `R, G, B, A` `u32` row
/// (NO narrow), the exact input the native-`u32` engine bins.
fn staged_rgba_u32(g: &[u32], b: &[u32], r: &[u32], a: &[u32], side: usize) -> Vec<u32> {
  let mut out = vec![0u32; side * side * 4];
  for i in 0..side * side {
    out[i * 4] = r[i];
    out[i * 4 + 1] = g[i];
    out[i * 4 + 2] = b[i];
    out[i * 4 + 3] = a[i];
  }
  out
}

/// Exact `u32`-domain `2x2` block mean (round-half-up over `u128`), narrowed
/// `>> 16` only AFTER binning — the 0-ULP oracle the area `rgba_u16` must equal.
fn area_u32_narrowed(staged: &[u32], ox: usize, oy: usize, c: usize, src_w: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * src_w + ox * 2 + dx) * 4 + c] as u64;
    }
  }
  (((acc + 2) / 4) >> 16) as u16
}

/// INDEPENDENT separable-`f64` filter oracle — does NOT call the production
/// `FilterStream<u32>`. Reads only the plan's `FilterAxis` windows (`span(j)` →
/// `(start, coeffs)`) and reimplements the engine's exact two-pass `u32`
/// execution from scratch (H-pass: `floor(Σ f64(coeff)·src + 0.5)`, no clamp;
/// V-pass: `floor(Σ f64(coeff)·h_inter + 0.5).clamp(0, u32::MAX)`; then
/// `>> 16`) — H-then-V order, `f64::from(f32)·(u32 as f64)` widening, and
/// `floor(x+0.5)` rounding matching the engine's scalar `u32` path bit-for-bit
/// (no SIMD divergence for the scalar-only `u32` tier). 0-ULP, yet independent
/// of the resampler under test, so it disproves bugs in coefficient selection,
/// intermediate rounding, final clamp/narrow, and pass order.
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

#[test]
fn gbrap32_downscale_rgba_u16_is_exact_u32_area_mean_incl_alpha() {
  let g = plane_u32(0x0011_0007);
  let b = plane_u32(0x0033_0009);
  let r = plane_u32(0x0055_000B);
  let a = plane_u32(0x0077_000D);
  let staged = staged_rgba_u32(&g, &b, &r, &a, SRC);
  // Wire-encode the frame's u32 planes so the LE frame decodes them via
  // `from_le` on both endiannesses; `staged` stays in the logical domain.
  let (gw, bw, rw, aw) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap();

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  // luma_u16 attached so the 4-channel tail's NATIVE_LUMA16 = true native-luma
  // branch (`rgb_to_luma_u16_native_row`) is exercised here.
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut oracle_rgba = vec![0u16; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        oracle_rgba[(oy * OUT + ox) * 4 + c] = area_u32_narrowed(&staged, ox, oy, c, SRC);
      }
    }
  }
  assert_eq!(
    rgba_u16, oracle_rgba,
    "rgba_u16 — exact u32 mean narrowed >> 16 (alpha is c3)"
  );
  // Native luma_u16 == a direct Gbrap32 sink over the binned RGB (independent
  // Q15 path) — the NATIVE_LUMA16 = true parity.
  let oracle_rgb = drop_alpha_u16(&oracle_rgba);
  assert_eq!(
    luma_u16,
    gbrap32_native_luma_u16(&oracle_rgb, OUT, true),
    "luma_u16 — native Q15 from the binned RGB"
  );
}

#[test]
fn gbrap32_identity_plan_matches_new_sink() {
  let g = plane_u32(0x0011_0007);
  let b = plane_u32(0x0033_0009);
  let r = plane_u32(0x0055_000B);
  let a = plane_u32(0x0077_000D);
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &g, &b, &r, &a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap();

  let mut direct = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32>::new(SRC, SRC)
      .with_rgba_u16(&mut direct)
      .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(SRC, SRC),
    )
    .unwrap()
    .with_rgba_u16(&mut via_area)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- 0-ULP parity-vs-u32-domain resample pins (issue #289, closed) ----------
// Binning at native `u32` and narrowing `>> 16` only AFTER the bin is 0-ULP for
// BOTH `full_range = true` and false: each resampled `rgba_u16` equals the
// exact `u32`-domain area / filter narrowed `>> 16`, NOT the prior narrow-first
// ≤1-LSB approximation. A single host-native fixture is re-encoded LE / BE so
// both endian arms decode the same logical values and must produce the
// identical 0-ULP output. The `gbrap32_*` area / filter oracles bin the full
// `u32` samples (a crafted averaging-boundary fixture would differ from the
// narrow-first mean — see the `gap_r_plane` history), then narrow.

const FRP: usize = 4; // source side
const FRO: usize = 2; // output side

fn frp_plane(seed: u32) -> Vec<u32> {
  (0..FRP * FRP)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0xBEEF))
    .collect()
}

#[test]
fn gbrap32_fr_false_area_rgba_u16_is_u32_domain_mean() {
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let staged = staged_rgba_u32(&g, &b, &r, &a, FRP);
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = area_u32_narrowed(&staged, ox, oy, c, FRP);
      }
    }
  }

  // LE arm
  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_le)
    .unwrap();
    crate::source::gbrap32_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  // BE arm
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(
    out_le, expected,
    "Gbrap32 FR=false area rgba_u16 LE (0-ULP)"
  );
  assert_eq!(
    out_be, expected,
    "Gbrap32 FR=false area rgba_u16 BE (0-ULP)"
  );
}

/// Runs the Gbrap32 straight-4ch FILTER route (rgba attached) for LE + BE with
/// EVERY output attached, asserting each against the INDEPENDENT separable-f64
/// oracle (`rgba_u16` / `rgb_u16` / `rgba` / `rgb`) and a direct-sink reference
/// (`luma` / `hsv` via Rgb48; native `luma_u16` via a direct Gbrap32 sink).
fn assert_gbrap32_4ch_filter(g: &[u32], b: &[u32], r: &[u32], a: &[u32], full_range: bool) {
  let staged = staged_rgba_u32(g, b, r, a, FRP);
  let oracle_rgba = separable_triangle_u32_narrowed(&staged, 4, FRP, FRP, FRO, FRO);
  let oracle_rgb = drop_alpha_u16(&oracle_rgba);
  let oracle_rgba_u8 = narrow_u16(&oracle_rgba);
  let (ref_rgb, ref_luma, ref_h, ref_s, ref_v) = rgb48_rgb_luma_hsv(&oracle_rgb, FRO, full_range);
  let ref_lu16 = gbrap32_native_luma_u16(&oracle_rgb, FRO, full_range);

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
    macro_rules! attach {
      ($sink:expr) => {
        $sink
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
          .unwrap()
      };
    }
    if be {
      let (gb, bb, rb, ab) = (as_be_u32(g), as_be_u32(b), as_be_u32(r), as_be_u32(a));
      let src = crate::frame::Gbrap32BeFrame::try_new(
        &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      let sink =
        MixedSinker::<crate::source::Gbrap32<true>, FilteredResampler<Triangle>>::with_resampler(
          FRP,
          FRP,
          FilteredResampler::new(FRO, FRO, Triangle),
        )
        .unwrap();
      let mut sink = attach!(sink);
      crate::source::gbrap32_to_endian::<_, true>(&src, full_range, ColorMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let (gl, bl, rl, al) = (as_le_u32(g), as_le_u32(b), as_le_u32(r), as_le_u32(a));
      let src = crate::frame::Gbrap32LeFrame::try_new(
        &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      let sink =
        MixedSinker::<crate::source::Gbrap32, FilteredResampler<Triangle>>::with_resampler(
          FRP,
          FRP,
          FilteredResampler::new(FRO, FRO, Triangle),
        )
        .unwrap();
      let mut sink = attach!(sink);
      crate::source::gbrap32_to(&src, full_range, ColorMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(
      rgba_u16, oracle_rgba,
      "Gbrap32 4ch filter rgba_u16 {tag} (0-ULP)"
    );
    assert_eq!(rgb_u16, oracle_rgb, "Gbrap32 4ch filter rgb_u16 {tag}");
    assert_eq!(rgba, oracle_rgba_u8, "Gbrap32 4ch filter rgba (u8) {tag}");
    assert_eq!(rgb, ref_rgb, "Gbrap32 4ch filter rgb (u8) {tag}");
    assert_eq!(luma, ref_luma, "Gbrap32 4ch filter luma {tag}");
    assert_eq!(
      luma_u16, ref_lu16,
      "Gbrap32 4ch filter luma_u16 (native) {tag}"
    );
    assert_eq!(h, ref_h, "Gbrap32 4ch filter hsv-h {tag}");
    assert_eq!(s_, ref_s, "Gbrap32 4ch filter hsv-s {tag}");
    assert_eq!(v_, ref_v, "Gbrap32 4ch filter hsv-v {tag}");
  }
}

#[test]
fn gbrap32_fr_false_filter_all_outputs_is_u32_domain() {
  assert_gbrap32_4ch_filter(
    &frp_plane(0x0123_4567),
    &frp_plane(0x0246_8ACE),
    &frp_plane(0x0FED_CBA9),
    &frp_plane(0x0777_1333),
    false,
  );
}

/// A `4x4` R plane whose top-left `2x2` averaging block carries nonzero low-16
/// bits — `[0x0001_FFFF, 0x0000_FFFF, 0x0000_FFFF, 0x0000_FFFF]` — so a
/// narrow-first mean (`>> 16` first → `[1,0,0,0]` → `0`) differs from the exact
/// `u32`-domain mean (`((Σ + 2) / 4) >> 16 = 81919 >> 16 = 1`). The native-`u32`
/// engine now emits the latter; this fixture pins the boundary.
fn gap_r_plane() -> Vec<u32> {
  let mut p: Vec<u32> = (0..FRP * FRP)
    .map(|i| (i as u32).wrapping_mul(0x00AB_1357).wrapping_add(0xABCD))
    .collect();
  p[0] = 0x0001_FFFF;
  p[1] = 0x0000_FFFF;
  p[FRP] = 0x0000_FFFF; // (0,1)
  p[FRP + 1] = 0x0000_FFFF; // (1,1)
  p
}

#[test]
fn gbrap32_fr_true_area_rgba_u16_is_u32_domain_mean() {
  // Full-range resample is now u32-domain exact (issue #289): it bins the full
  // `u32` samples and narrows only after — NOT the `>> 16` narrow first. The
  // crafted `gap_r_plane` block exposes the boundary: its R(0,0) is the exact
  // u32 mean (`1`), not the mean-of-narrowed (`0`).
  let r = gap_r_plane();
  let g = frp_plane(0x0246_8ACE);
  let b = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let staged = staged_rgba_u32(&g, &b, &r, &a, FRP);
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = area_u32_narrowed(&staged, ox, oy, c, FRP);
      }
    }
  }
  // The crafted block makes R(0,0) the exact u32 mean (1), distinct from the
  // mean-of-narrowed (0) — so this fixture genuinely exercises the 0-ULP fix.
  assert_eq!(expected[0], 1, "exact u32-domain R(0,0)");

  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_le)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(out_le, expected, "Gbrap32 FR=true area rgba_u16 LE (0-ULP)");
  assert_eq!(out_be, expected, "Gbrap32 FR=true area rgba_u16 BE (0-ULP)");
}

#[test]
fn gbrap32_fr_true_filter_all_outputs_is_u32_domain() {
  // Full-range FILTER, crafted averaging-boundary R plane whose `0xFFFF` low
  // bits survive until after filtering: the exact u32-domain Triangle R(0,0) is
  // 1, not the mean-of-narrowed 0 (so the fixture genuinely exercises the fix).
  let mut r = std::vec![0x0000_FFFFu32; FRP * FRP];
  r[0] = 0x0001_FFFF; // (0,0)
  r[1] = 0x0001_FFFF; // (0,1)
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let a = frp_plane(0x0777_1333);
  // Fixture-validity: the INDEPENDENT oracle's R(0,0) is 1 (crosses the narrow
  // boundary), distinct from the mean-of-narrowed 0.
  let staged = staged_rgba_u32(&g, &b, &r, &a, FRP);
  let oracle = separable_triangle_u32_narrowed(&staged, 4, FRP, FRP, FRO, FRO);
  assert_eq!(oracle[0], 1, "independent u32-domain Triangle R(0,0)");
  assert_gbrap32_4ch_filter(&g, &b, &r, &a, true);
}

// ---- Premultiplied-alpha area resample (0-ULP, u64-intermediate) -------------
// Under `AlphaMode::Premultiplied` the G/B/R/A planes are de-interleaved to the
// canonical host-native `R, G, B, A` `u32` row, premultiplied at `u32`
// (`round(c·α / u32::MAX)`, `u64` intermediates), area-binned at `u32`, then
// un-premultiplied (`round(pm·MAX / α)`, 0 when binned α = 0) before the `>> 16`
// narrow + derive. The flipped #289 pins exercise the STRAIGHT path only; this
// oracle independently reproduces that exact `u64` premult / unpremult chain
// (varying alpha, the α = 0 transparent edge, LE + BE).

const PMAX: u64 = u32::MAX as u64;

fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Premultiplied-area `u32` oracle for the canonical `R, G, B, A` `u32` row
/// `src` (`src_w x src_w`): premultiply at `u32`, area-bin each 2x2 block
/// (round-half-up over `u128`), zero-α-safe un-premultiply, narrow `>> 16` — the
/// exact `u64`-intermediate chain the engine runs. Returns straight `R,G,B,A` u16.
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

fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  rgba
    .chunks_exact(4)
    .flat_map(|px| [px[0], px[1], px[2]])
    .collect()
}

fn narrow_u16(v: &[u16]) -> Vec<u8> {
  v.iter().map(|&x| (x >> 8) as u8).collect()
}

/// G/B/R/A `u32` plane ramps with one fully-transparent (α = 0) 2x2 output
/// block (top-left), so the un-premultiply's `α == 0` branch is hit.
fn premult_gbrap_planes() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let mut a = frp_plane(0x0777_1333);
  for &(sx, sy) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
    a[sy * FRP + sx] = 0;
  }
  (g, b, r, a)
}

#[test]
fn gbrap32_premult_area_matches_u32_premult_oracle() {
  let (g, b, r, a) = premult_gbrap_planes();
  let staged = staged_rgba_u32(&g, &b, &r, &a, FRP);
  let oracle = premult_area_rgba_u16(&staged, FRP, FRO);
  let straight_rgb = drop_alpha_u16(&oracle);

  // Direct full-res Rgb48 sink over the straight (un-premultiplied) binned RGB —
  // reference for the narrowed u8 rgb / luma / hsv derives.
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
    rgb48_to(&binned, true, ColorMatrix::Bt709, &mut sink).unwrap();
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
      let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
      let src = crate::frame::Gbrap32BeFrame::try_new(
        &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      let mut sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
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
      crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink)
        .unwrap();
    } else {
      let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
      let src = crate::frame::Gbrap32LeFrame::try_new(
        &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
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
      crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(rgba_u16, oracle, "Gbrap32 premult rgba_u16 {tag}");
    assert_eq!(rgb_u16, straight_rgb, "Gbrap32 premult rgb_u16 {tag}");
    assert_eq!(rgba, narrow_u16(&oracle), "Gbrap32 premult rgba (u8) {tag}");
    assert_eq!(
      &rgba_u16[..4],
      &[0, 0, 0, 0],
      "Gbrap32 transparent bled {tag}"
    );
    assert_eq!(rgb, ref_rgb, "Gbrap32 premult rgb (u8) {tag}");
    assert_eq!(luma, ref_luma, "Gbrap32 premult luma {tag}");
    assert_eq!(h, ref_h, "Gbrap32 premult hsv-h {tag}");
    assert_eq!(s_, ref_s, "Gbrap32 premult hsv-s {tag}");
    assert_eq!(v_, ref_v, "Gbrap32 premult hsv-v {tag}");
  }
}

// ---- Straight-alpha DROP-alpha 3-channel route (Gbrap32, area + filter) ------
// When straight alpha and the caller drops α (only `rgb` / `rgb_u16` / `luma` /
// `luma_u16` / `hsv`, no `rgba(_u16)`), Gbrap32 takes the SEPARATE 3-channel
// route (`gbr32_to_rgb_u32_row` G/B/R reorder + BE staging →
// `packed_rgb_u32_resample_emit::<true>`), distinct from the 4-channel and
// premult pins above — and the only route reaching the `NATIVE_LUMA16 = true`
// native-`luma_u16` branch of the 3-channel emit. Pin `rgb_u16` against the
// 3-channel u32-domain oracle (staging + bin + narrow); the narrowed rgb / luma
// / hsv against a direct `Rgb48` sink; and the **native** `luma_u16` against a
// direct `Gbrap32` sink over the same binned RGB (an independent Q15 path).

/// Exact `u32`-domain 2x2 block mean (round-half-up), narrowed `>> 16`, for a
/// **3-channel** `R, G, B` `u32` row.
fn area_rgb_u32_narrowed(staged: &[u32], ox: usize, oy: usize, c: usize, src_w: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * src_w + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  (((acc + 2) / 4) >> 16) as u16
}

/// G/B/R planar `u32` → packed canonical `R, G, B` `u32` (drop α, NO narrow) —
/// the engine's `gbr32_to_rgb_u32_row` staging, the 3-channel oracle input.
fn gbrap32_drop_rgb_u32(g: &[u32], b: &[u32], r: &[u32], side: usize) -> Vec<u32> {
  let mut out = vec![0u32; side * side * 3];
  for i in 0..side * side {
    out[i * 3] = r[i];
    out[i * 3 + 1] = g[i];
    out[i * 3 + 2] = b[i];
  }
  out
}

/// Direct full-res `Rgb48` sink over a binned `R, G, B` u16 row → narrowed
/// `(rgb_u8, luma, h, s, v)` (shared u8 kernels; `luma_u16` is the NATIVE flavour
/// for Gbrap32, taken from [`gbrap32_native_luma_u16`] instead).
/// `(rgb_u8, luma_u8, h, s, v)` narrowed derives (`luma_u16` is native — see
/// [`gbrap32_native_luma_u16`]).
type RgbLumaHsv = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

fn rgb48_rgb_luma_hsv(rgb_u16: &[u16], n: usize, full_range: bool) -> RgbLumaHsv {
  let wire = as_le_u16(rgb_u16);
  let src = Rgb48Frame::new(&wire, n as u32, n as u32, (n * 3) as u32);
  let mut rgb = vec![0u8; n * n * 3];
  let mut luma = vec![0u8; n * n];
  let mut h = vec![0u8; n * n];
  let mut s = vec![0u8; n * n];
  let mut v = vec![0u8; n * n];
  {
    let mut sink = MixedSinker::<Rgb48>::new(n, n)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    rgb48_to(&src, full_range, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  (rgb, luma, h, s, v)
}

/// Native `luma_u16` oracle for a binned `R, G, B` u16 row: re-encode it as
/// `Gbrap32` planes (widened `<< 16`, dummy α) and run a DIRECT `Gbrap32` sink —
/// its `gbr32_to_luma_u16_row` is the independent native-Q15 path the resample's
/// `rgb_to_luma_u16_native_row` must equal (`NATIVE_LUMA16 = true` parity).
fn gbrap32_native_luma_u16(rgb_u16: &[u16], n: usize, full_range: bool) -> Vec<u16> {
  let px = n * n;
  let mut g = vec![0u32; px];
  let mut b = vec![0u32; px];
  let mut r = vec![0u32; px];
  let a = vec![0u32; px];
  for i in 0..px {
    r[i] = (rgb_u16[i * 3] as u32) << 16;
    g[i] = (rgb_u16[i * 3 + 1] as u32) << 16;
    b[i] = (rgb_u16[i * 3 + 2] as u32) << 16;
  }
  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &gl, &bl, &rl, &al, n as u32, n as u32, n as u32, n as u32, n as u32, n as u32,
  )
  .unwrap();
  let mut luma_u16 = vec![0u16; px];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32>::new(n, n)
      .with_luma_u16(&mut luma_u16)
      .unwrap();
    crate::source::gbrap32_to(&src, full_range, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  luma_u16
}

#[test]
fn gbrap32_drop_alpha_area_route_is_u32_domain() {
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let rgb_src = gbrap32_drop_rgb_u32(&g, &b, &r, FRP);
  let mut oracle = vec![0u16; FRO * FRO * 3];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..3 {
        oracle[(oy * FRO + ox) * 3 + c] = area_rgb_u32_narrowed(&rgb_src, ox, oy, c, FRP);
      }
    }
  }
  let (ref_rgb, ref_luma, ref_h, ref_s, ref_v) = rgb48_rgb_luma_hsv(&oracle, FRO, true);
  let ref_lu16 = gbrap32_native_luma_u16(&oracle, FRO, true);
  run_gbrap32_drop_route(
    &g, &b, &r, &a, false, &oracle, &ref_rgb, &ref_luma, &ref_lu16, &ref_h, &ref_s, &ref_v, "area",
  );
}

#[test]
fn gbrap32_drop_alpha_filter_route_is_u32_domain() {
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let rgb_src = gbrap32_drop_rgb_u32(&g, &b, &r, FRP);
  let oracle = separable_triangle_u32_narrowed(&rgb_src, 3, FRP, FRP, FRO, FRO);
  let (ref_rgb, ref_luma, ref_h, ref_s, ref_v) = rgb48_rgb_luma_hsv(&oracle, FRO, true);
  let ref_lu16 = gbrap32_native_luma_u16(&oracle, FRO, true);
  run_gbrap32_drop_route(
    &g, &b, &r, &a, true, &oracle, &ref_rgb, &ref_luma, &ref_lu16, &ref_h, &ref_s, &ref_v, "filter",
  );
}

/// Runs the Gbrap32 straight-alpha drop-α 3-channel route for LE + BE (area or
/// filter per `is_filter`) and asserts every alpha-dropped output against the
/// supplied oracles.
#[allow(clippy::too_many_arguments)]
fn run_gbrap32_drop_route(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  is_filter: bool,
  oracle_rgb_u16: &[u16],
  ref_rgb: &[u8],
  ref_luma: &[u8],
  ref_lu16: &[u16],
  ref_h: &[u8],
  ref_s: &[u8],
  ref_v: &[u8],
  kind: &str,
) {
  for be in [false, true] {
    let mut rgb_u16 = vec![0u16; FRO * FRO * 3];
    let mut rgb = vec![0u8; FRO * FRO * 3];
    let mut luma = vec![0u8; FRO * FRO];
    let mut luma_u16 = vec![0u16; FRO * FRO];
    let mut h = vec![0u8; FRO * FRO];
    let mut s_ = vec![0u8; FRO * FRO];
    let mut v_ = vec![0u8; FRO * FRO];

    macro_rules! attach {
      ($sink:expr) => {
        $sink
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_hsv(&mut h, &mut s_, &mut v_)
          .unwrap()
      };
    }

    if be {
      let (gb, bb, rb, ab) = (as_be_u32(g), as_be_u32(b), as_be_u32(r), as_be_u32(a));
      let src = crate::frame::Gbrap32BeFrame::try_new(
        &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      if is_filter {
        let sink =
          MixedSinker::<crate::source::Gbrap32<true>, FilteredResampler<Triangle>>::with_resampler(
            FRP,
            FRP,
            FilteredResampler::new(FRO, FRO, Triangle),
          )
          .unwrap();
        let mut sink = attach!(sink);
        crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink)
          .unwrap();
      } else {
        let sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
          FRP,
          FRP,
          AreaResampler::to(FRO, FRO),
        )
        .unwrap();
        let mut sink = attach!(sink);
        crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink)
          .unwrap();
      }
    } else {
      let (gl, bl, rl, al) = (as_le_u32(g), as_le_u32(b), as_le_u32(r), as_le_u32(a));
      let src = crate::frame::Gbrap32LeFrame::try_new(
        &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
      )
      .unwrap();
      if is_filter {
        let sink =
          MixedSinker::<crate::source::Gbrap32, FilteredResampler<Triangle>>::with_resampler(
            FRP,
            FRP,
            FilteredResampler::new(FRO, FRO, Triangle),
          )
          .unwrap();
        let mut sink = attach!(sink);
        crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
      } else {
        let sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
          FRP,
          FRP,
          AreaResampler::to(FRO, FRO),
        )
        .unwrap();
        let mut sink = attach!(sink);
        crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
      }
    }
    let tag = if be { "BE" } else { "LE" };
    assert_eq!(
      rgb_u16, oracle_rgb_u16,
      "Gbrap32 drop-α {kind} rgb_u16 {tag}"
    );
    assert_eq!(rgb, ref_rgb, "Gbrap32 drop-α {kind} rgb {tag}");
    assert_eq!(luma, ref_luma, "Gbrap32 drop-α {kind} luma {tag}");
    assert_eq!(
      luma_u16, ref_lu16,
      "Gbrap32 drop-α {kind} luma_u16 (native) {tag}"
    );
    assert_eq!(h, ref_h, "Gbrap32 drop-α {kind} hsv-h {tag}");
    assert_eq!(s_, ref_s, "Gbrap32 drop-α {kind} hsv-s {tag}");
    assert_eq!(v_, ref_v, "Gbrap32 drop-α {kind} hsv-v {tag}");
  }
}

#[test]
fn gbrap32_premult_filter_rejects() {
  // Premultiplied alpha has no filter analogue: the Filter arm routes to the
  // area tail, which rejects with the typed `UnsupportedFilter`. The one
  // remaining distinct Gbrap32 branch.
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
  )
  .unwrap();
  let mut rgba_u16 = vec![0u16; FRO * FRO * 4];
  let mut sink =
    MixedSinker::<crate::source::Gbrap32, FilteredResampler<Triangle>>::with_resampler(
      FRP,
      FRP,
      FilteredResampler::new(FRO, FRO, Triangle),
    )
    .unwrap()
    .with_alpha_mode(AlphaMode::Premultiplied)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  let err = crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(crate::resample::ResampleError::UnsupportedFilter(_))
    ),
    "Gbrap32 premult+filter must reject: {err:?}"
  );
}
