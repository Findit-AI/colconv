//! Fused-downscale coverage for the exotic 10-bit **packed 4:2:2** YUV NATIVE
//! fast tier — `V210` (LE + BE wire). The fixed-`BITS = 10` sibling of the
//! high-bit packed 4:2:2 `Y210` native tier
//! ([`resample_y2xx_native`](super::resample_y2xx_native)): both reuse the
//! high-bit non-4:2:0 PLANAR join
//! ([`yuv_planar16_process_native`](crate::sinker::mixed::planar_high_bit_native))
//! after de-packing the wire planes into wrapper-owned host-native LOGICAL u16
//! scratch — V210 with the intricate word packing (12 × 10-bit samples per
//! 16-byte word = four 32-bit LE/BE words = 6 pixels), Y210 with the YUYV
//! MSB-aligned u16 words.
//!
//! The native tier bins those planes straight to the output grid and converts
//! ONCE per output row at output width (4:4:4 kernels) — vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. The tiers differ in
//! colour SEMANTICS (native averages in YUV then converts; row-stage converts
//! then averages in RGB), so native is NOT byte-identical to row-stage — only
//! within a small tolerance in-gamut. Luma is bit-identical (both bin the same
//! de-packed native Y then narrow `>> 2`).
//!
//! The high-bit planar join now emits BOTH u8 `luma` and the native-depth
//! `luma_u16` (the clamped binned Y), so the V210 sink routes to native for
//! EVERY output set it exposes — attaching `luma_u16` no longer falls the
//! pipeline back to row-stage (which would silently change the rgb colour
//! semantics).
//!
//! THE DE-PACK IS THE CRUX. `native_equals_planar_twin` is the strong guard: it
//! de-packs the SAME wire into separate Yuv422p10 planes (an INDEPENDENT extract,
//! `pack_v210` → planar) and runs the PLANAR native sink, so any wrong V210 bit
//! position / word offset in the native wrapper shows up as a mismatch. The
//! partial-last-word variant (`native_equals_planar_twin_partial_word`) pins the
//! width-not-a-multiple-of-6 de-pack too.
//!
//! Tests:
//! - `native_equals_bin_then_convert_oracle`: the GROUND-TRUTH check — native
//!   output EXACTLY equals an independent bin-then-convert oracle (de-pack →
//!   area-bin each plane to OUTPUT res → ONE conversion through an
//!   identity-resolution `Yuv444p10` sink with the SAME native-depth kernels +
//!   `1023` clamp the native tier finalizes with). The luma oracle clamps
//!   INDEPENDENTLY.
//! - `native_equals_planar_twin` (+ `_partial_word`): native V210 == native
//!   `Yuv422p10` on the de-packed planes — the de-pack-correctness cross-check.
//! - `native_within_tolerance_of_rowstage` / `native_be_within_tolerance_of_rowstage_be`:
//!   the cv2 INTER_AREA parity bound + the `BE = HOST_NATIVE_BE` handoff proof.
//! - `native_luma_clamps_full_scale_y` / `rowstage_luma_clamps_full_scale_y`: the
//!   native-depth luma clamp at the achievable full-scale (1023) boundary, BOTH
//!   tiers.
//! - `native_luma_u16_equals_clamped_binned_y`: the native-depth `luma_u16`
//!   (clamped binned Y, NOT narrowed) vs an independent oracle, LE + BE, plus a
//!   rgb + luma_u16 sink takes native for BOTH.
//! - `luma_only_native_skips_chroma_planning`: a luma-only sink plans no chroma.
//! - the atomicity regressions on [`arm_v210_alloc_failure`] + the route freeze
//!   guard (both directions + the luma_u16-attach precedence).

use super::*;
use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Yuv422p10, Yuv444p10, yuv422p10_to, yuv444p10_to},
};

const SRC: usize = 12;
const CW: usize = SRC / 2;
const OUT: usize = 6;
const M: ColorMatrix = ColorMatrix::Bt601;
const FR: bool = true;
/// V210 byte stride for an `SRC`-wide row: `ceil(SRC / 6) * 16`.
const STRIDE: u32 = (SRC.div_ceil(6) * 16) as u32;
const MASK: u16 = 0x3FF;
const MID: u16 = 1 << 9;

/// In-gamut per-channel u8 tolerance between the native and row-stage tiers (the
/// two average in different domains and round independently per output pixel);
/// native correctness itself is pinned EXACTLY by
/// `native_equals_bin_then_convert_oracle`. Matches the high-bit packed 4:2:2
/// suite's bound.
const TOL_U8: u8 = 5;

// ---- V210 wire packing (mirrors resample_v210.rs / scalar/v210.rs) ----

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

/// Pack per-pixel native Y (`w * h`) and per-chroma-sample native U / V
/// (`(w / 2) * h`, 4:2:2) into a `V210` LE byte plane. `w` must be even; a final
/// partial word (w not a multiple of 6) zero-fills its unused high samples
/// (matching a real capture's undefined-but-zero tail). The chroma for word
/// pixel pair (0,1)/(2,3)/(4,5) is the even-pixel chroma column.
fn pack_v210_dims(y: &[u16], u: &[u16], v: &[u16], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let words_per_row = w.div_ceil(6);
  let mut out = std::vec![0u8; words_per_row * 16 * h];
  for row in 0..h {
    for word in 0..words_per_row {
      let px = word * 6;
      // Read up to 6 Y and 3 chroma pairs; clamp at the row's valid extent so a
      // partial last word leaves its unused samples zero.
      let gy = |k: usize| -> u16 { if px + k < w { y[row * w + px + k] } else { 0 } };
      let gc = |c: &[u16], k: usize| -> u16 {
        let cu = px / 2 + k;
        if cu < cw { c[row * cw + cu] } else { 0 }
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

/// `SRC x SRC` convenience wrapper over [`pack_v210_dims`].
fn pack_v210(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
  pack_v210_dims(y, u, v, SRC, SRC)
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

// ---- Source-grid fixtures ---------------------------------------------

/// Per-pixel logical Y ramp + per-chroma-sample logical U / V ramp, kept near
/// the legal-range middle so the converted RGB stays in gamut and the
/// native-vs-rowstage delta is per-pixel rounding. Returns the (Y, U, V) planes.
fn ramp_planes() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut y = std::vec![0u16; SRC * SRC];
  let mut u = std::vec![0u16; CW * SRC];
  let mut v = std::vec![0u16; CW * SRC];
  for row in 0..SRC {
    for x in 0..SRC {
      let i = row * SRC + x;
      y[i] = (MID as u32 + ((i as u32 * 37) % (MASK as u32 / 4))) as u16 & MASK;
    }
    for cx in 0..CW {
      let ci = row * CW + cx;
      u[ci] =
        (MID as u32 + ((ci as u32 * 53) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
      v[ci] =
        (MID as u32 + ((ci as u32 * 41) % (MASK as u32 / 8)) - (MASK as u32 / 16)) as u16 & MASK;
    }
  }
  (y, u, v)
}

fn ramp_packed() -> Vec<u8> {
  let (y, u, v) = ramp_planes();
  pack_v210(&y, &u, &v)
}

/// Uniform-gray plane: constant logical Y, neutral chroma (U = V = mid).
fn uniform_gray_packed(y: u16) -> Vec<u8> {
  let yp = std::vec![y & MASK; SRC * SRC];
  let chroma = std::vec![MID & MASK; CW * SRC];
  pack_v210(&yp, &chroma, &chroma)
}

/// Crafted VARYING illegal-chroma fixture: extreme alternating chroma over a
/// super-black->super-white Y ramp — many 2x2 blocks straddle the RGB clamp,
/// where native (average-in-YUV) and row-stage (convert-then-average) diverge.
fn out_of_gamut_packed() -> Vec<u8> {
  let mut y = std::vec![0u16; SRC * SRC];
  let mut u = std::vec![0u16; CW * SRC];
  let mut v = std::vec![0u16; CW * SRC];
  for row in 0..SRC {
    for x in 0..SRC {
      let i = row * SRC + x;
      y[i] = ((i as u32 * MASK as u32) / (SRC * SRC) as u32) as u16 & MASK;
    }
    for cx in 0..CW {
      let ci = row * CW + cx;
      let hi = ci.is_multiple_of(2);
      u[ci] = if hi { MASK } else { 0 };
      v[ci] = if hi { 0 } else { MASK };
    }
  }
  pack_v210(&y, &u, &v)
}

/// Full-scale-Y fixture: every Y at the native max `MASK`, legal chroma. A V210
/// sample CANNOT exceed `MASK` (the de-pack masks `& 0x3FF`), and an area mean of
/// `<= MASK` stays `<= MASK` — so the achievable boundary is the legal max. Pins
/// the native-depth luma clamp at that boundary.
fn full_scale_luma_packed() -> Vec<u8> {
  let (_, u, v) = ramp_planes();
  let yp = std::vec![MASK; SRC * SRC];
  pack_v210(&yp, &u, &v)
}

fn frame(buf: &[u8]) -> V210Frame<'_> {
  V210Frame::new(buf, SRC as u32, SRC as u32, STRIDE)
}

// ---- Independent de-pack (mirrors the row-stage `unpack_v210_word`) ----
//
// These extract the logical planes straight from the wire bytes, INDEPENDENTLY
// of the native wrapper's de-pack — so a wrong bit position in the wrapper is
// caught by the twin-parity / oracle comparison.

/// De-pack a V210 LE byte plane into the logical Y plane (`w * h`).
fn logical_y(packed: &[u8], w: usize, h: usize) -> Vec<u16> {
  let words_per_row = w.div_ceil(6);
  let mut y = std::vec![0u16; w * h];
  for row in 0..h {
    for word in 0..words_per_row {
      let off = (row * words_per_row + word) * 16;
      let lane =
        |l: usize| u32::from_le_bytes(packed[off + l * 4..off + l * 4 + 4].try_into().unwrap());
      let (w0, w1, w2, w3) = (lane(0), lane(1), lane(2), lane(3));
      let ys = [
        (w0 >> 10) & 0x3FF,
        w1 & 0x3FF,
        (w1 >> 20) & 0x3FF,
        (w2 >> 10) & 0x3FF,
        w3 & 0x3FF,
        (w3 >> 20) & 0x3FF,
      ];
      let px = word * 6;
      for (k, &yv) in ys.iter().enumerate() {
        if px + k < w {
          y[row * w + px + k] = yv as u16;
        }
      }
    }
  }
  y
}

/// De-pack a V210 LE byte plane into the logical U / V planes (`(w / 2) * h`).
fn logical_uv(packed: &[u8], w: usize, h: usize) -> (Vec<u16>, Vec<u16>) {
  let cw = w / 2;
  let words_per_row = w.div_ceil(6);
  let mut u = std::vec![0u16; cw * h];
  let mut v = std::vec![0u16; cw * h];
  for row in 0..h {
    for word in 0..words_per_row {
      let off = (row * words_per_row + word) * 16;
      let lane =
        |l: usize| u32::from_le_bytes(packed[off + l * 4..off + l * 4 + 4].try_into().unwrap());
      let (w0, w1, w2, w3) = (lane(0), lane(1), lane(2), lane(3));
      let us = [w0 & 0x3FF, (w1 >> 10) & 0x3FF, (w2 >> 20) & 0x3FF];
      let vs = [(w0 >> 20) & 0x3FF, w2 & 0x3FF, (w3 >> 10) & 0x3FF];
      let cu = word * 3;
      for k in 0..3 {
        if cu + k < cw {
          u[row * cw + cu + k] = us[k] as u16;
          v[row * cw + cu + k] = vs[k] as u16;
        }
      }
    }
  }
  (u, v)
}

/// Exact integer-ratio area mean (round-half-up) of an `in_w x in_h` u16 plane
/// down to `out_w x out_h`, binning each axis by its own ratio.
fn bin_to(plane: &[u16], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u16> {
  let (rx, ry) = (in_w / out_w, in_h / out_h);
  let denom = (rx * ry) as u32;
  let mut out = std::vec![0u16; out_w * out_h];
  for oy in 0..out_h {
    for ox in 0..out_w {
      let mut s = 0u32;
      for dy in 0..ry {
        for dx in 0..rx {
          s += plane[(oy * ry + dy) * in_w + ox * rx + dx] as u32;
        }
      }
      out[oy * out_w + ox] = ((s + denom / 2) / denom) as u16;
    }
  }
  out
}

// ---- Tier drivers -----------------------------------------------------

/// Drive the LE source through a tier for the native output set
/// (u8 RGB, u16 RGB, u8 luma). `native` toggles the bin-then-convert native fast
/// tier vs the convert-then-bin row-stage tier.
fn run(packed: &[u8], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    v210_to(&frame(packed), FR, M, &mut sink).unwrap();
  }
  (rgb, rgb_u16, luma)
}

/// Drive the BE source through a tier (the host-native-endian guard reference).
fn be_run(packed_le: &[u8], native: bool) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
  let p = v210_as_be(packed_le);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let be_frame = V210BeFrame::try_new(&p, SRC as u32, SRC as u32, STRIDE).unwrap();
    let mut sink = MixedSinker::<V210<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_native(native)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    v210_to_endian::<_, true>(&be_frame, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb_u16, luma)
}

/// The bin-then-convert oracle: de-pack the V210 words, area-bin every plane to
/// OUTPUT resolution (Y from `SRC x SRC`, chroma from `CW x SRC` — horizontal-only
/// subsample), then convert the full-output-width host-native planes ONCE through
/// an identity-resolution `Yuv444p10` sink. The luma oracle clamps INDEPENDENTLY.
fn oracle(packed: &[u8]) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
  let yl = logical_y(packed, SRC, SRC);
  let (u, v) = logical_uv(packed, SRC, SRC);
  let yb = bin_to(&yl, SRC, SRC, OUT, OUT);
  let ub = bin_to(&u, CW, SRC, OUT, OUT);
  let vb = bin_to(&v, CW, SRC, OUT, OUT);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  {
    // Chroma is binned to FULL output width, so feed a 4:4:4 sink.
    let mut sink = MixedSinker::<Yuv444p10>::new(OUT, OUT)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    let f = Yuv444p10Frame::new(
      &yb, &ub, &vb, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
    );
    yuv444p10_to(&f, FR, M, &mut sink).unwrap();
  }
  let luma: Vec<u8> = yb.iter().map(|&by| (by.min(MASK) >> 2) as u8).collect();
  (rgb, rgb_u16, luma)
}

/// Native `Yuv422p10` reference on the de-packed planes (an INDEPENDENT extract)
/// — the twin-parity cross-check that the V210 wrapper is a faithful de-pack in
/// front of the planar join. Source-resolution sink (the planar native tier bins).
fn planar_twin_native(packed: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u16>, Vec<u8>) {
  let yl = logical_y(packed, w, h);
  let (u, v) = logical_uv(packed, w, h);
  let cw = w / 2;
  let ow = w / 2;
  let oh = h / 2;
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut luma = std::vec![0u8; ow * oh];
  {
    let mut sink =
      MixedSinker::<Yuv422p10, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    let f = Yuv422p10Frame::new(
      &yl, &u, &v, w as u32, h as u32, w as u32, cw as u32, cw as u32,
    );
    yuv422p10_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgb_u16, luma)
}

fn max_delta_u8(a: &[u8], b: &[u8]) -> u8 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}
fn max_delta_u16(a: &[u16], b: &[u16]) -> u16 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}

// ---- Per-output parity ------------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_equals_bin_then_convert_oracle() {
  let packed = ramp_packed();
  let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
  let (o_rgb, o_rgb16, o_luma) = oracle(&packed);
  assert_eq!(
    n_rgb, o_rgb,
    "u8 rgb must equal the bin-then-convert oracle"
  );
  assert_eq!(
    n_rgb16, o_rgb16,
    "u16 rgb must equal the bin-then-convert oracle (clamp-for-clamp)"
  );
  assert_eq!(n_luma, o_luma, "luma must equal the binned-then-narrowed Y");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_equals_planar_twin() {
  // The V210 wrapper IS a de-pack in front of the planar join, so its output must
  // be bit-identical to feeding the de-packed planes straight to native Yuv422p10.
  // This catches any wrong V210 bit position / word offset in the wrapper.
  let packed = ramp_packed();
  let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
  let (t_rgb, t_rgb16, t_luma) = planar_twin_native(&packed, SRC, SRC);
  assert_eq!(n_rgb, t_rgb, "u8 rgb must match the planar twin");
  assert_eq!(n_rgb16, t_rgb16, "u16 rgb must match the planar twin");
  assert_eq!(n_luma, t_luma, "luma must match the planar twin");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_equals_planar_twin_partial_word() {
  // Width 8 is even but NOT a multiple of 6: each row is 2 words, the last
  // carrying only 2 valid pixels (1 chroma pair). The native de-pack must write
  // only the valid prefix; feeding the independently de-packed planes to the
  // planar twin proves the partial-word handling is correct.
  const W: usize = 8;
  const H: usize = 8;
  let cw = W / 2;
  let mut y = std::vec![0u16; W * H];
  let mut u = std::vec![0u16; cw * H];
  let mut v = std::vec![0u16; cw * H];
  for row in 0..H {
    for x in 0..W {
      y[row * W + x] =
        (MID as u32 + (((row * W + x) as u32 * 29) % (MASK as u32 / 4))) as u16 & MASK;
    }
    for cx in 0..cw {
      let ci = row * cw + cx;
      u[ci] = (MID as u32 + ((ci as u32 * 47) % (MASK as u32 / 8))) as u16 & MASK;
      v[ci] = (MID as u32 + ((ci as u32 * 31) % (MASK as u32 / 8))) as u16 & MASK;
    }
  }
  let packed = pack_v210_dims(&y, &u, &v, W, H);
  let stride = (W.div_ceil(6) * 16) as u32;

  let mut n_rgb = std::vec![0u8; (W / 2) * (H / 2) * 3];
  let mut n_luma = std::vec![0u8; (W / 2) * (H / 2)];
  {
    let mut sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(W, H, AreaResampler::to(W / 2, H / 2))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut n_rgb)
        .unwrap()
        .with_luma(&mut n_luma)
        .unwrap();
    v210_to(
      &V210Frame::new(&packed, W as u32, H as u32, stride),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
  }
  let (t_rgb, _, t_luma) = planar_twin_native(&packed, W, H);
  assert_eq!(
    n_rgb, t_rgb,
    "partial-word u8 rgb must match the planar twin"
  );
  assert_eq!(
    n_luma, t_luma,
    "partial-word luma must match the planar twin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_within_tolerance_of_rowstage() {
  let packed = ramp_packed();
  let (n_rgb, n_rgb16, n_luma) = run(&packed, true);
  let (r_rgb, r_rgb16, r_luma) = run(&packed, false);
  assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");
  let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
  assert!(
    d_u8 <= TOL_U8,
    "u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
  );
  let tol_u16: u16 = (TOL_U8 as u16) << 2;
  let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
  assert!(
    d_u16 <= tol_u16,
    "u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_be_matches_native_le() {
  // The native tier de-packs the wire words to host-native LOGICAL u16 BEFORE
  // binning, so BE and LE sources produce identical output.
  let packed = ramp_packed();
  let le = run(&packed, true);
  let be = be_run(&packed, true);
  assert_eq!(be.0, le.0, "BE u8 colour must match LE");
  assert_eq!(be.1, le.1, "BE u16 colour must match LE");
  assert_eq!(be.2, le.2, "BE luma must match LE");
}

/// The host-native-endian regression: BE native vs the correct BE row-stage
/// reference (it de-packs BE-wire bytes to host-native before converting), within
/// the same tolerances + luma bit-identical. Proves the `BE = HOST_NATIVE_BE`
/// handoff — a wrapper forwarding the source `BE` would byte-swap the
/// already-native scratch on a big-endian host.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_be_within_tolerance_of_rowstage_be() {
  let packed = ramp_packed();
  let (n_rgb, n_rgb16, n_luma) = be_run(&packed, true);
  let (r_rgb, r_rgb16, r_luma) = be_run(&packed, false);
  assert_eq!(n_luma, r_luma, "BE luma must be bit-identical across tiers");
  let d_u8 = max_delta_u8(&n_rgb, &r_rgb);
  assert!(
    d_u8 <= TOL_U8,
    "BE u8 native-vs-rowstage max delta {d_u8} exceeds tolerance {TOL_U8}"
  );
  let tol_u16: u16 = (TOL_U8 as u16) << 2;
  let d_u16 = max_delta_u16(&n_rgb16, &r_rgb16);
  assert!(
    d_u16 <= tol_u16,
    "BE u16 native-vs-rowstage max delta {d_u16} exceeds tolerance {tol_u16}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_luma_matches_inter_area_oracle() {
  // cv2 INTER_AREA parity for luma: the area-bin of the DE-PACKED logical Y,
  // narrowed. Guards the Y de-pack (the word 0/2 bit positions).
  let packed = ramp_packed();
  let (_, _, n_luma) = run(&packed, true);
  let y_ref = bin_to(&logical_y(&packed, SRC, SRC), SRC, SRC, OUT, OUT);
  let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> 2) as u8).collect();
  assert_eq!(
    n_luma, luma_ref,
    "native luma must equal the INTER_AREA bin of the de-packed Y"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_luma_clamps_full_scale_y() {
  // A full-scale (1023) binned Y must SATURATE through the native-depth clamp +
  // `>> 2` narrowing, never wrap. The oracle clamps independently.
  let packed = full_scale_luma_packed();
  let (_, _, n_luma) = run(&packed, true);
  let yb = bin_to(&logical_y(&packed, SRC, SRC), SRC, SRC, OUT, OUT);
  let expect: Vec<u8> = yb.iter().map(|&by| (by.min(MASK) >> 2) as u8).collect();
  assert_eq!(
    n_luma, expect,
    "full-scale binned Y must clamp to native-max before narrowing, not wrap"
  );
  let sat = (MASK >> 2) as u8;
  assert!(
    n_luma.iter().all(|&l| l == sat),
    "all full-scale luma must saturate to {sat}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rowstage_luma_clamps_full_scale_y() {
  // Same clamp on the ROW-STAGE (with_native(false)) path.
  let packed = full_scale_luma_packed();
  let (_, _, r_luma) = run(&packed, false);
  let yb = bin_to(&logical_y(&packed, SRC, SRC), SRC, SRC, OUT, OUT);
  let expect: Vec<u8> = yb.iter().map(|&by| (by.min(MASK) >> 2) as u8).collect();
  assert_eq!(
    r_luma, expect,
    "row-stage full-scale luma must clamp to native-max before narrowing, not wrap"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_gamut_native_vs_rowstage_pinned() {
  let packed = out_of_gamut_packed();
  let (n_rgb, _, n_luma) = run(&packed, true);
  let (r_rgb, _, r_luma) = run(&packed, false);
  assert_eq!(n_luma, r_luma, "luma stays bit-identical out of gamut");
  let d = max_delta_u8(&n_rgb, &r_rgb);
  assert!(
    d > TOL_U8,
    "crafted out-of-gamut case must diverge beyond the in-gamut tolerance {TOL_U8}, got {d}"
  );
  assert!(d < u8::MAX, "out-of-gamut delta stays bounded, got {d}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uniform_gray_leaves_color_unchanged() {
  // Independent-kernel guard (#37): a uniform-gray downscale must leave every
  // colour output equal to the direct conversion of a single pixel.
  let packed = uniform_gray_packed((MID as u32 + (MASK as u32 / 8)) as u16 & MASK);
  let (n_rgb, n_rgb16, _) = run(&packed, true);
  let mut ref_rgb = std::vec![0u8; SRC * SRC * 3];
  let mut ref_rgb16 = std::vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<V210>::new(SRC, SRC)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb16)
      .unwrap();
    v210_to(&frame(&packed), FR, M, &mut sink).unwrap();
  }
  for px in n_rgb.chunks_exact(3) {
    assert_eq!(px, &ref_rgb[..3], "uniform-gray u8 colour drifted");
  }
  for px in n_rgb16.chunks_exact(3) {
    assert_eq!(px, &ref_rgb16[..3], "uniform-gray u16 colour drifted");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn u8_and_u16_color_are_independent_bins() {
  // Independent-kernel guard (#37): narrowing the binned u16 colour to u8
  // diverges from the genuine u8 bin over a varying ramp.
  let packed = ramp_packed();
  let (n_rgb, n_rgb16, _) = run(&packed, true);
  let narrowed: Vec<u8> = n_rgb16.iter().map(|&c| (c >> 2) as u8).collect();
  assert_ne!(
    n_rgb, narrowed,
    "u8 colour must be an independent bin, not a narrowed u16 bin"
  );
}

/// The native fast tier emits the native-depth `luma_u16` directly (the clamped
/// binned Y, host-native u16 — NOT narrowed), so it equals an INDEPENDENT
/// clamped-binned-Y oracle bit-for-bit, LE + BE. AND a `rgb` + `luma_u16` sink
/// takes the NATIVE route for BOTH: the rgb matches the native bin-then-convert
/// oracle (not row-stage), proving that attaching `luma_u16` no longer changes
/// the rgb colour semantics.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_luma_u16_equals_clamped_binned_y() {
  let packed = ramp_packed();
  let luma_u16_oracle: Vec<u16> = bin_to(&logical_y(&packed, SRC, SRC), SRC, SRC, OUT, OUT)
    .iter()
    .map(|&by| by.min(MASK))
    .collect();

  // LE: a `luma_u16`-only native sink.
  {
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    {
      let mut sink =
        MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_native(true)
          .with_luma_u16(&mut luma_u16)
          .unwrap();
      v210_to(&frame(&packed), FR, M, &mut sink).unwrap();
    }
    assert_eq!(
      luma_u16, luma_u16_oracle,
      "native luma_u16 (LE) must equal the clamped-binned-Y oracle"
    );
  }

  // BE: same, through the BE wire — the native tier de-packs to host-native
  // LOGICAL u16 before binning, so it is bit-identical to LE.
  {
    let p = v210_as_be(&packed);
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    {
      let be_frame = V210BeFrame::try_new(&p, SRC as u32, SRC as u32, STRIDE).unwrap();
      let mut sink = MixedSinker::<V210<true>, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_native(true)
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      v210_to_endian::<_, true>(&be_frame, FR, M, &mut sink).unwrap();
    }
    assert_eq!(
      luma_u16, luma_u16_oracle,
      "native luma_u16 (BE) must equal the clamped-binned-Y oracle"
    );
  }

  // A `rgb` + `luma_u16` sink uses the NATIVE route for BOTH.
  {
    let mut rgb = std::vec![0u8; OUT * OUT * 3];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    {
      let mut sink =
        MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_native(true)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap();
      v210_to(&frame(&packed), FR, M, &mut sink).unwrap();
    }
    let (o_rgb, _, _) = oracle(&packed);
    assert_eq!(
      rgb, o_rgb,
      "rgb + luma_u16 must take the NATIVE route — rgb equals the native oracle, \
       not row-stage"
    );
    assert_eq!(
      luma_u16, luma_u16_oracle,
      "rgb + luma_u16: luma_u16 must equal the clamped-binned-Y oracle"
    );
  }
}

#[test]
fn no_outputs_is_a_no_op() {
  let packed = ramp_packed();
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true);
  v210_to(&frame(&packed), FR, M, &mut sink).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn resets_join_across_frames() {
  let (y1, u1, v1) = ramp_planes();
  let p1 = pack_v210(&y1, &u1, &v1);
  let inv = |p: &[u16]| -> Vec<u16> { p.iter().map(|&x| MASK - (x & MASK)).collect() };
  let p2 = pack_v210(&inv(&y1), &inv(&u1), &inv(&v1));
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    v210_to(&frame(&p1), FR, M, &mut sink).unwrap();
    v210_to(&frame(&p2), FR, M, &mut sink).unwrap();
  }
  let y_ref = bin_to(&logical_y(&p2, SRC, SRC), SRC, SRC, OUT, OUT);
  let luma_ref: Vec<u8> = y_ref.iter().map(|&c| (c >> 2) as u8).collect();
  assert_eq!(luma, luma_ref, "join did not reset between frames");
}

// ---- atomicity --------------------------------------------------------

/// One V210 row slice (`div_ceil(6) * 16` bytes) at the given source row.
fn row_slice(packed: &[u8], idx: usize) -> &[u8] {
  let stride = STRIDE as usize;
  &packed[idx * stride..(idx + 1) * stride]
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_first_row_rejected_and_does_not_poison_retry() {
  let packed = ramp_packed();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(V210Row::new(row_slice(&packed, 3), 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

/// A mid-frame output-set change must be rejected by the join's frozen-output
/// preflight BEFORE the wrapper de-pack scratch alloc — `ResampleOutputsChanged`,
/// never `AllocationFailed`.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn frozen_mid_frame_change_rejected_before_scratch_alloc() {
  let packed = ramp_packed();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Luma-only rows 0 and 1 freeze a luma-only output set.
  for r in 0..2 {
    sink
      .process(V210Row::new(row_slice(&packed, r), r, M, FR))
      .expect("luma-only rows freeze a luma-only output set");
  }
  // Attach u16 colour mid-frame, changing the output set, and arm the wrapper
  // scratch failpoint on the reserve the changed row reaches.
  sink.set_rgb_u16(&mut rgb_u16).unwrap();
  crate::sinker::mixed::arm_v210_alloc_failure();
  let err = sink
    .process(V210Row::new(row_slice(&packed, 2), 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame output change must reject as ResampleOutputsChanged before the \
     scratch alloc, got {err:?}"
  );
  assert!(
    rgb_u16.iter().all(|&b| b == 0),
    "rejected mid-frame-change row touched the new colour output"
  );
  // The failpoint is single-shot; prove it was NOT consumed via a fresh
  // in-sequence colour row that DOES fire it.
  let mut rgb_u16b = std::vec![0u16; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb_u16(&mut rgb_u16b)
      .unwrap();
  let err2 = sink2
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err2,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "armed failpoint must still be live and fire on the first in-sequence colour \
     reserve, got {err2:?}"
  );
}

/// Sequence rejection after a RECOVERABLE wrapper-scratch allocation failure:
/// because the wrapper now runs the COMPARE-ONLY preflight (the delegate owns the
/// single commit), a de-pack failure on an in-sequence colour row 0 leaves the
/// output set UNFROZEN and the join unbuilt, so a later OUT-OF-SEQUENCE row is
/// caught by the pre-compare first-row sequence check and must reject as
/// `OutOfSequenceRow`, never `AllocationFailed`.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn oos_after_recoverable_alloc_failure_rejected_before_scratch_alloc() {
  let packed = ramp_packed();
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  crate::sinker::mixed::arm_v210_alloc_failure();
  let err0 = sink
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err0,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the recoverable scratch failure on row 0 must surface AllocationFailed, got {err0:?}"
  );
  crate::sinker::mixed::arm_v210_alloc_failure();
  let err2 = sink
    .process(V210Row::new(row_slice(&packed, 2), 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err2,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "an out-of-sequence row after a recoverable scratch failure must reject as \
     OutOfSequenceRow, never AllocationFailed, got {err2:?}"
  );
  assert!(
    rgb_u16.iter().all(|&b| b == 0),
    "neither the recoverable-failure nor the out-of-sequence row touched the colour output"
  );
  // Drain the failpoint re-armed above: the out-of-sequence row rejected
  // BEFORE reaching the scratch grow, so the flag is still live. A fresh
  // in-sequence colour reserve fires it — both proving the re-arm survived
  // the OOS preflight AND leaving the thread-local failpoint disarmed so it
  // can never bleed into a later test. (`arm_*` failpoints are thread-local;
  // the native cargo runner gives each test its own thread, but the
  // wasm32-wasip1 harness runs every test on the one main thread, so a flag
  // left armed here would spuriously fail the next test that reaches a V210
  // scratch grow.) Mirrors the `resample_y2xx_native` sibling's drain.
  let mut rgb_u16b = std::vec![0u16; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb_u16(&mut rgb_u16b)
      .unwrap();
  let err3 = sink2
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err3,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the failpoint re-armed in step 2 must still be live and fire on the first \
     in-sequence colour reserve, got {err3:?}"
  );
}

// ---- frozen native-vs-row-stage route ---------------------------------

/// Flipping `set_native(true) -> false` mid-frame must reject as the
/// deterministic `NativeRouteChanged` BEFORE either tier consumes the row.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_to_rowstage_route_flip_mid_frame_rejected() {
  let packed = ramp_packed();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .expect("native row 0 freezes the route and succeeds");
  sink.set_native(false);
  let err = sink
    .process(V210Row::new(row_slice(&packed, 1), 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::NativeRouteChanged(_)),
    "a native -> row-stage mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
  );
}

/// The reverse flip `set_native(false) -> true` mid-frame must reject identically
/// — the guard catches BOTH directions.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rowstage_to_native_route_flip_mid_frame_rejected() {
  let packed = ramp_packed();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(false)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .expect("row-stage row 0 freezes the route and succeeds");
  sink.set_native(true);
  let err = sink
    .process(V210Row::new(row_slice(&packed, 1), 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::NativeRouteChanged(_)),
    "a row-stage -> native mid-frame route flip must reject as NativeRouteChanged, got {err:?}"
  );
}

/// Attaching `luma_u16` MID-FRAME (after a native u8-luma row froze the output
/// set) must be classified by the FROZEN-OUTPUT check as `ResampleOutputsChanged`,
/// NOT by the route guard as `NativeRouteChanged` — `take_native = native` is
/// invariant to `luma_u16`, so the row enters the native delegate and the
/// frozen-output preflight reports the genuine output-set change.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_u16_attach_mid_frame_rejected_as_outputs_changed() {
  let packed = ramp_packed();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V210Row::new(row_slice(&packed, 0), 0, M, FR))
    .expect("native luma row 0 freezes the output set and the route");
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(V210Row::new(row_slice(&packed, 1), 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "a mid-frame luma_u16 attach must reject as ResampleOutputsChanged (the \
     frozen-output check), not NativeRouteChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "the rejected mid-frame-change row must not touch the new luma_u16 output"
  );
}

/// A luma-only native sink must NOT plan or allocate any chroma state. Armed with
/// the planar join's chroma-planning failpoint: a luma-only row leaves it
/// unconsumed (so the run succeeds), while a colour row reaches chroma planning
/// and fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  let packed = uniform_gray_packed(MID);
  crate::sinker::mixed::arm_planar_hb_native_chroma_failure();

  // Luma-only: the chroma failpoint is armed but never reached -> Ok.
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    v210_to(&frame(&packed), FR, M, &mut sink).expect("luma-only native must not plan chroma");
  }

  // Colour: the still-armed failpoint fires at chroma planning -> Err.
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    v210_to(&frame(&packed), FR, M, &mut sink).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}
