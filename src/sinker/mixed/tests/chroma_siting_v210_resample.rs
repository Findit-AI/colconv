//! Chroma-siting-aware fused-downscale coverage for the exotic 10-bit **packed**
//! 4:2:2 `V210` — the packed-word twin of the high-bit `chroma_siting_y2xx_resample`
//! and the planar `chroma_siting_hibit_422_resample`.
//!
//! 4:2:2 subsamples chroma 2:1 horizontally only. The V210 word packing (12 x
//! 10-bit samples per 16-byte word = 6 pixels) de-packs to the SAME half-width
//! U / V planes a `Yuv422p10` frame holds, so every centered packed resample is
//! bit-identical to the centered `Yuv422p10` resample of those planes — the
//! strongest catch for a U/V swap in the de-pack. Routing the centered siting
//! (`Center` / `Top` / `Bottom`, [`chroma_422_center_sited_h`]) through the
//! resample means:
//!  - the **native fast tier** (`with_native(true)`) folds the #302 `1/4`–`3/4`
//!    triangle into the [`ResamplePlan::area_chroma_422`] chroma weights and bins
//!    the SUBSAMPLED `u16` chroma directly — its code-domain twin is
//!    [`bin_chroma_centered_u16`] (a single round over the x4 reconstruction);
//!  - the **encoded row-stage tier** (`with_native(false)`) reconstructs full-width
//!    `u16` chroma per source row then convert-then-bins 4:4:4 — the RGB-domain
//!    reconstruct-then-bin, pinned against a `Yuv444p10` resample of the
//!    independently reconstructed planes.
//!
//! V210 is area-only (no filter route), so unlike the Y2xx twin there is no filter
//! tier here. The logical 10-bit values de-pack identically to the low-packed
//! `Yuv422p10` / `Yuv444p10` oracle planes, so the siting math is endian-independent
//! (pinned by the BE<->LE parity test). Every centered assertion is an EXACT match
//! (single rounding), never a tolerance.

use super::*;
use crate::{
  ChromaLocation, KernelMatrix, PixelSink,
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{Yuv422p10, Yuv444p10, yuv422p10_to, yuv444p10_to},
};

const SRC: usize = 12;
const CW: usize = SRC / 2;
const OUT: usize = 6;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;
/// V210 byte stride for an `SRC`-wide row: `ceil(SRC / 6) * 16`.
const STRIDE: u32 = (SRC.div_ceil(6) * 16) as u32;
const MASK: u16 = 0x3FF;
const MID: u16 = 1 << 9;

// ---- V210 wire packing (mirrors resample_v210_native.rs) --------------

/// Pack 12 logical samples in V210 standard order
/// (`[Cb0, Y0, Cr0, Y1, Cb1, Y2, Cr1, Y3, Cb2, Y4, Cr2, Y5]`) into a 16-byte
/// word: three 10-bit samples per 32-bit LE word, top 2 bits unused.
fn pack_v210_word(samples: [u16; 12]) -> [u8; 16] {
  let mut out = [0u8; 16];
  let pack = |a: u16, b: u16, c: u16| -> u32 {
    (a as u32 & 0x3FF) | ((b as u32 & 0x3FF) << 10) | ((c as u32 & 0x3FF) << 20)
  };
  out[0..4].copy_from_slice(&pack(samples[0], samples[1], samples[2]).to_le_bytes());
  out[4..8].copy_from_slice(&pack(samples[3], samples[4], samples[5]).to_le_bytes());
  out[8..12].copy_from_slice(&pack(samples[6], samples[7], samples[8]).to_le_bytes());
  out[12..16].copy_from_slice(&pack(samples[9], samples[10], samples[11]).to_le_bytes());
  out
}

/// Pack per-pixel logical Y (`SRC * SRC`) and per-chroma-sample logical U / V
/// (`CW * SRC`, 4:2:2) into a `V210` LE byte plane. `SRC` is a multiple of 6, so
/// every word is full (no partial-word tail). The chroma for word pixel pair
/// (0,1)/(2,3)/(4,5) is the even-pixel chroma column.
fn pack_v210(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
  let words_per_row = SRC.div_ceil(6);
  let mut out = vec![0u8; words_per_row * 16 * SRC];
  for row in 0..SRC {
    for word in 0..words_per_row {
      let px = word * 6;
      let gy = |k: usize| -> u16 {
        if px + k < SRC {
          y[row * SRC + px + k]
        } else {
          0
        }
      };
      let gc = |c: &[u16], k: usize| -> u16 {
        let cu = px / 2 + k;
        if cu < CW { c[row * CW + cu] } else { 0 }
      };
      let samples: [u16; 12] = [
        gc(u, 0),
        gy(0),
        gc(v, 0),
        gy(1),
        gc(u, 1),
        gy(2),
        gc(v, 1),
        gy(3),
        gc(u, 2),
        gy(4),
        gc(v, 2),
        gy(5),
      ];
      let off = (row * words_per_row + word) * 16;
      out[off..off + 16].copy_from_slice(&pack_v210_word(samples));
    }
  }
  out
}

/// Re-encode a V210 LE byte plane as BE-encoded byte storage by byte-swapping
/// each 32-bit word (the kernel's `from_be` decode recovers the same samples).
fn v210_as_be(plane_le: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(plane_le.len());
  for chunk in plane_le.chunks_exact(4) {
    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    out.extend_from_slice(&w.to_be_bytes());
  }
  out
}

// ---- exact oracles ----------------------------------------------------

/// Round-half-up integer divide.
fn rdhu(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// Exact box-overlap area weights for `src -> out`, mirroring
/// `resample::AxisSpans::area`. Returns per output `(first source cell, overlaps)`.
fn area_weights(src: usize, out: usize) -> Vec<(usize, Vec<u64>)> {
  let (src64, out64) = (src as u64, out as u64);
  (0..out)
    .map(|o| {
      let lo = o as u64 * src64;
      let hi = lo + src64;
      let start = (lo / out64) as usize;
      let mut w = Vec::new();
      let mut i = start as u64;
      loop {
        let clo = i * out64;
        if clo >= hi {
          break;
        }
        let chi = clo + out64;
        let ov = chi.min(hi) - clo.max(lo);
        if ov == 0 {
          break;
        }
        w.push(ov);
        if chi >= hi {
          break;
        }
        i += 1;
      }
      (start, w)
    })
    .collect()
}

/// Co-sited box-average of a full-resolution `sw x sh` `u16` plane to `ow x oh`
/// (round-half-up) — the reference for a phase-free plane (luma, co-sited chroma).
fn bin_cosited_u16(plane: &[u16], sw: usize, sh: usize, ow: usize, oh: usize) -> Vec<u16> {
  let hw = area_weights(sw, ow);
  let vw = area_weights(sh, oh);
  let denom = (sw * sh) as u64;
  let mut out = vec![0u16; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * u64::from(plane[(vs + dy) * sw + hs + dx]);
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

/// The EXACT centered chroma oracle (`u16`): reconstruct the half-width `cw x ch`
/// chroma to full width with the #302 `1/4`–`3/4` triangle kept UNROUNDED (scaled
/// x4 to stay integral), then box-average to `ow x oh` with a SINGLE round-half-up
/// over `4·(2·cw)·ch`. The code-domain twin the folded
/// [`ResamplePlan::area_chroma_422`] weights realize.
fn bin_chroma_centered_u16(c: &[u16], cw: usize, ch: usize, ow: usize, oh: usize) -> Vec<u16> {
  let full = 2 * cw;
  let mut r4 = vec![0u64; full * ch];
  for r in 0..ch {
    let row = &c[r * cw..r * cw + cw];
    for j in 0..cw {
      let l = u64::from(row[j.saturating_sub(1)]);
      let m = u64::from(row[j]);
      let rt = u64::from(row[if j + 1 < cw { j + 1 } else { j }]);
      r4[r * full + 2 * j] = l + 3 * m;
      r4[r * full + 2 * j + 1] = 3 * m + rt;
    }
  }
  let hw = area_weights(full, ow);
  let vw = area_weights(ch, oh);
  let denom = (4 * full * ch) as u64;
  let mut out = vec![0u16; ow * oh];
  for (oy, (vs, vwin)) in vw.iter().enumerate() {
    for (ox, (hs, hwin)) in hw.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &vwt) in vwin.iter().enumerate() {
        let mut hsum = 0u64;
        for (dx, &hwt) in hwin.iter().enumerate() {
          hsum += hwt * r4[(vs + dy) * full + hs + dx];
        }
        s += vwt * hsum;
      }
      out[oy * ow + ox] = rdhu(s, denom) as u16;
    }
  }
  out
}

/// Independent #302 centered horizontal upsample (`1/4`–`3/4`, edge clamp,
/// round-half-up to `u16`) — the RGB-domain oracle's reconstruction step.
fn recon_full_row_u16(c: &[u16], cw: usize) -> Vec<u16> {
  let mut out = vec![0u16; 2 * cw];
  for j in 0..cw {
    let l = u32::from(c[j.saturating_sub(1)]);
    let m = u32::from(c[j]);
    let r = u32::from(c[if j + 1 < cw { j + 1 } else { j }]);
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u16;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u16;
  }
  out
}

// ---- source-grid fixtures ---------------------------------------------

/// A strong HORIZONTAL chroma ramp (so the centered triangle genuinely differs
/// from the co-sited nearest decode) plus a per-row tilt (a vertical mistake
/// would show). Returns LOGICAL planes.
fn ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let step = (MASK as u32 / 16).max(1);
  let mut y = vec![0u16; SRC * SRC];
  let mut u = vec![0u16; CW * SRC];
  let mut v = vec![0u16; CW * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = ((40 + i as u32 * 37) & MASK as u32) as u16;
  }
  for r in 0..SRC {
    for c in 0..CW {
      u[r * CW + c] = (step * c as u32 + step + r as u32 * 5).min(MASK as u32) as u16;
      v[r * CW + c] = (MASK as u32).saturating_sub(step * c as u32).max(step) as u16;
    }
  }
  (y, u, v)
}

/// Flat chroma: the centered triangle of a constant is that constant, so centered
/// must equal co-sited. Luma still varies.
fn flat() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut y = vec![0u16; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = ((40 + i as u32 * 29) & MASK as u32) as u16;
  }
  (y, vec![MID; CW * SRC], vec![MID; CW * SRC])
}

// ---- tier drivers -----------------------------------------------------

fn frame(buf: &[u8]) -> V210Frame<'_> {
  V210Frame::new(buf, SRC as u32, SRC as u32, STRIDE)
}

/// Drive an LE `V210` area resample for `rgb` (u8) + `rgb_u16`, at `loc` siting
/// and `native` tier.
fn run(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  loc: ChromaLocation,
  native: bool,
  simd: bool,
) -> (Vec<u8>, Vec<u16>) {
  let packed = pack_v210(y, u, v);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(native)
        .with_chroma_location(loc.clone())
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb16)
        .unwrap();
    v210_to(&frame(&packed), FR, M, &mut sink).unwrap();
  }
  (rgb, rgb16)
}

/// Drive a BE `V210` area resample (wire re-encoded BE).
fn run_be(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  loc: ChromaLocation,
  native: bool,
) -> (Vec<u8>, Vec<u16>) {
  let packed = v210_as_be(&pack_v210(y, u, v));
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb16 = vec![0u16; OUT * OUT * 3];
  {
    let be_frame = V210BeFrame::try_new(&packed, SRC as u32, SRC as u32, STRIDE).unwrap();
    let mut sink = MixedSinker::<V210<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_native(native)
    .with_chroma_location(loc.clone())
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb16)
    .unwrap();
    v210_to_endian::<_, true>(&be_frame, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb16)
}

/// The planar `Yuv422p10` twin of [`run`]: the SAME logical U / V driven through a
/// planar 4:2:2 resample. A centered packed decode must equal this byte-for-byte —
/// the de-pack / U-V-swap catch.
fn run_yuv422p(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  loc: ChromaLocation,
  native: bool,
  simd: bool,
) -> (Vec<u8>, Vec<u16>) {
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv422p10, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_native(native)
    .with_chroma_location(loc.clone())
    .with_simd(simd)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb16)
    .unwrap();
    let f = Yuv422p10Frame::new(
      y, u, v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
    );
    yuv422p10_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb16)
}

/// The centered NATIVE code-domain oracle: bin Y co-sited and U / V through the
/// exact centered chroma oracle to `OUT x OUT` (LOGICAL), then convert ONCE at
/// output width via an identity `Yuv444p10` sink.
fn native_oracle(y: &[u16], u: &[u16], v: &[u16], simd: bool) -> (Vec<u8>, Vec<u16>) {
  let yb = bin_cosited_u16(y, SRC, SRC, OUT, OUT);
  let ub = bin_chroma_centered_u16(u, CW, SRC, OUT, OUT);
  let vb = bin_chroma_centered_u16(v, CW, SRC, OUT, OUT);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv444p10>::new(OUT, OUT)
      .with_simd(simd)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb16)
      .unwrap();
    let f = Yuv444p10Frame::new(
      &yb, &ub, &vb, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
    );
    yuv444p10_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb16)
}

/// The centered RGB-domain oracle: reconstruct U / V to full width (LOGICAL u16)
/// with the #302 kernel, then run that `Yuv444p10` frame through an area resample —
/// convert-each-row-then-bin, exactly what the row-stage arm does.
fn rgb_domain_oracle(y: &[u16], u: &[u16], v: &[u16], simd: bool) -> (Vec<u8>, Vec<u16>) {
  let mut uf = vec![0u16; SRC * SRC];
  let mut vf = vec![0u16; SRC * SRC];
  for r in 0..SRC {
    uf[r * SRC..r * SRC + SRC].copy_from_slice(&recon_full_row_u16(&u[r * CW..r * CW + CW], CW));
    vf[r * SRC..r * SRC + SRC].copy_from_slice(&recon_full_row_u16(&v[r * CW..r * CW + CW], CW));
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb16 = vec![0u16; OUT * OUT * 3];
  {
    let f = Yuv444p10Frame::new(
      y, &uf, &vf, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink = MixedSinker::<Yuv444p10, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_native(false)
    .with_simd(simd)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb16)
    .unwrap();
    yuv444p10_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb16)
}

// ---- co-sited byte-identity (the regression contract) -----------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn cosited_group_is_byte_identical_across_tiers() {
  let (y, u, v) = ramp();
  for native in [true, false] {
    let base = run(&y, &u, &v, ChromaLocation::Unspecified, native, true);
    for loc in [
      ChromaLocation::Left,
      ChromaLocation::TopLeft,
      ChromaLocation::BottomLeft,
      ChromaLocation::other("unassigned-7"),
    ] {
      assert_eq!(
        run(&y, &u, &v, loc.clone(), native, true),
        base,
        "co-sited siting {loc:?} must keep the byte-identical decode (native={native})"
      );
    }
  }
}

// ---- centered packed == centered planar Yuv422p (U/V-swap catch) ------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_equals_planar_yuv422p_across_tiers() {
  let (y, u, v) = ramp();
  for loc in [
    ChromaLocation::Center,
    ChromaLocation::Top,
    ChromaLocation::Bottom,
  ] {
    for native in [true, false] {
      assert_eq!(
        run(&y, &u, &v, loc.clone(), native, true),
        run_yuv422p(&y, &u, &v, loc.clone(), native, true),
        "centered packed {loc:?} must equal centered planar Yuv422p10 (native={native})"
      );
    }
  }
}

// ---- centered native == the exact code-domain oracle ------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_native_equals_code_domain_oracle() {
  let (y, u, v) = ramp();
  let want = native_oracle(&y, &u, &v, true);
  for loc in [
    ChromaLocation::Center,
    ChromaLocation::Top,
    ChromaLocation::Bottom,
  ] {
    assert_eq!(
      run(&y, &u, &v, loc.clone(), true, true),
      want,
      "centered native {loc:?} must equal the code-domain reconstruct-then-bin oracle"
    );
  }
}

// ---- centered row-stage == RGB-domain reconstruct-then-bin ------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_row_stage_equals_rgb_reconstruct_then_bin() {
  let (y, u, v) = ramp();
  let want = rgb_domain_oracle(&y, &u, &v, true);
  for loc in [
    ChromaLocation::Center,
    ChromaLocation::Top,
    ChromaLocation::Bottom,
  ] {
    assert_eq!(
      run(&y, &u, &v, loc.clone(), false, true),
      want,
      "centered row-stage {loc:?} must equal the RGB-domain reconstruct-then-bin oracle"
    );
  }
}

// ---- SIMD == scalar ---------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_simd_matches_scalar() {
  let (y, u, v) = ramp();
  for native in [true, false] {
    assert_eq!(
      run(&y, &u, &v, ChromaLocation::Center, native, true),
      run(&y, &u, &v, ChromaLocation::Center, native, false),
      "centered SIMD vs scalar must agree (native={native})"
    );
  }
}

// ---- centered differs from co-sited on a ramp, equals it on flat ------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_differs_from_cosited_on_a_chroma_ramp() {
  let (y, u, v) = ramp();
  for native in [true, false] {
    let cos = run(&y, &u, &v, ChromaLocation::Left, native, true);
    let cen = run(&y, &u, &v, ChromaLocation::Center, native, true);
    assert_ne!(
      cos, cen,
      "centered must differ from co-sited on a chroma ramp (native={native})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn centered_equals_cosited_on_flat_chroma() {
  let (y, u, v) = flat();
  for native in [true, false] {
    assert_eq!(
      run(&y, &u, &v, ChromaLocation::Left, native, true),
      run(&y, &u, &v, ChromaLocation::Center, native, true),
      "centered must equal co-sited on flat chroma (native={native})"
    );
  }
}

// ---- wire endianness is siting-independent ----------------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn be_centered_matches_le() {
  let (y, u, v) = ramp();
  for native in [true, false] {
    assert_eq!(
      run_be(&y, &u, &v, ChromaLocation::Center, native),
      run(&y, &u, &v, ChromaLocation::Center, native, true),
      "BE centered decode must equal LE for the same logical planes (native={native})"
    );
  }
}

// ---- mid-frame siting change is rejected across tiers -----------------

/// One V210 row slice (`STRIDE` bytes) at the given source row.
fn row_slice(packed: &[u8], idx: usize) -> &[u8] {
  let stride = STRIDE as usize;
  &packed[idx * stride..(idx + 1) * stride]
}

/// Accept row 0 at `loc1` (freezes the phase), flip to `loc2`, feed the
/// IN-SEQUENCE row 1, and return its `process` result.
fn flip_row1<R>(
  mut sink: MixedSinker<'_, V210, R>,
  y: &[u16],
  u: &[u16],
  v: &[u16],
  loc1: ChromaLocation,
  loc2: ChromaLocation,
) -> Result<(), MixedSinkerError> {
  let packed = pack_v210(y, u, v);
  sink.set_chroma_location(loc1.clone());
  PixelSink::begin_frame(&mut sink, SRC as u32, SRC as u32).unwrap();
  let row0 = V210Row::new(row_slice(&packed, 0), 0, M, FR);
  PixelSink::process(&mut sink, row0).unwrap();
  sink.set_chroma_location(loc2.clone());
  let row1 = V210Row::new(row_slice(&packed, 1), 1, M, FR);
  PixelSink::process(&mut sink, row1)
}

#[test]
#[cfg_attr(miri, ignore = "SIMD row kernels use intrinsics unsupported by Miri")]
fn mid_frame_siting_change_rejected() {
  let (y, u, v) = ramp();
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
  ] {
    // Native fast tier.
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "native {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );

    // Encoded row-stage tier.
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    let err = flip_row1(sink, &y, &u, &v, loc1.clone(), loc2.clone()).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "row-stage {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}
