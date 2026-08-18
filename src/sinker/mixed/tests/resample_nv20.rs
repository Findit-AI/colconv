//! Fused-downscale coverage for **NV20** — the **low-bit-packed** 10-bit
//! semi-planar 4:2:2 twin of [`P210`](crate::source::P210) (LE + BE wire).
//! A full-width Y plane plus one **interleaved** half-width / **full-height**
//! `U,V,U,V…` plane; NV20 stores its 10 active bits in the **low** 10 of each
//! `u16` (`& 0x03FF`) where P210 stores them in the high 10 (`>> 6`).
//!
//! The conversion suite ([`super::nv20`]) pins the NV20 **direct** (identity)
//! path; this module pins the three NON-identity resample tiers the NV20 sinker
//! wires in [`p2xx.rs`](crate::sinker::mixed::subsampled_4_2_2_high_bit) — the
//! row-stage **area** tail, the **filter** tail, and the **native** area
//! decimator — each of which feeds NV20's low-bit Y de-pack
//! (`logical & ((1 << BITS) - 1)`) and, for native, the `LOW_PACKED = true`
//! monomorphization of the shared depack. A wrong mask/shift on the resample
//! path (as opposed to the direct path) would corrupt resized frames; nothing
//! else constructs an NV20 resample plan.
//!
//! Two complementary oracles:
//!
//! - **Independent area-bin-of-direct** (mirrors the P2xx `rgb_u8_matches_
//!   area_bin_of_direct` family): drive a direct identity NV20 sink at source
//!   resolution and 2x2-block-mean its output (luma area-bins the de-packed
//!   logical Y then narrows). Pins the NV20 area + native tiers on their own.
//! - **★ Cross-packing equivalence** (the load-bearing check, the resample
//!   twin of [`super::nv20::nv20_low_bit_decodes_same_as_p210_with_equal_
//!   logical_sample`]): NV20(low-packed logical `L`) resampled **==** P210
//!   (high-packed `L << 6`) resampled, output-for-output, for the AREA
//!   (native on + off) and FILTER tiers, LE + BE. The SAME logical samples
//!   must produce the SAME resampled output regardless of low-vs-high packing —
//!   proving the resample path extracts the active bits correctly, not just the
//!   direct path. P210's resampled outputs are themselves pinned to
//!   bin-then-convert / area-bin oracles in the sibling P2xx suites, so this
//!   equivalence transfers that ground truth onto NV20's low-bit resample path.

use crate::{
  KernelMatrix,
  // `P210Frame` bakes in `BITS = 10` (`PnFrame422<'a, 10>`), so an
  // endian-generic builder needs the underlying `PnFrame422<'_, 10, BE>`.
  frame::{Nv20Frame, PnFrame422},
  resample::{AreaResampler, CatmullRom, FilteredResampler, Lanczos3, Triangle},
  sinker::MixedSinker,
  source::{Nv20, P210, nv20_to_endian, p210_to_endian},
};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;
const BITS: u32 = 10;
const MASK: u16 = ((1u32 << BITS) - 1) as u16;

/// Encode a `u16` as host-independent **wire** byte storage in the order
/// selected by `BE` (the `*Frame` plane contract; recovered via `from_le` /
/// `from_be` inside the kernels).
fn enc(v: u16, be: bool) -> u16 {
  if be {
    u16::from_ne_bytes(v.to_be_bytes())
  } else {
    u16::from_ne_bytes(v.to_le_bytes())
  }
}

/// Per-pixel logical Y ramp + per-chroma-sample logical (U, V) ramp, all
/// within the 10-bit range. The SAME logical planes back both an NV20 and a
/// P210 frame (each packed in its own convention), so the cross-packing
/// equivalence compares like for like.
fn logical_ramp(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>) {
  let cw = sw / 2;
  let mut y = vec![0u16; sw * sh];
  let mut uv = vec![0u16; cw * sh * 2];
  for (i, yi) in y.iter_mut().enumerate() {
    *yi = ((40u32 + i as u32 * 37) & MASK as u32) as u16;
  }
  for i in 0..cw * sh {
    let u = ((300u32 + i as u32 * 53) & MASK as u32) as u16;
    let v = ((MASK as u32).wrapping_sub(i as u32 * 41) as u16) & MASK;
    uv[2 * i] = u;
    uv[2 * i + 1] = v;
  }
  (y, uv)
}

/// A varying, always-NON-ZERO 6-bit pattern for the high (padding) bits of
/// NV20 word `i`. NV20's active bits are the low 10 (`& 0x03FF`); the high 6
/// are padding the de-pack MUST discard. Real wire data does not zero them, so
/// the fixtures inject garbage here (mirroring the direct-path `0x1234`
/// stray-bit tests). The `| 0b00_0001` floor on the 6-bit pattern keeps every
/// stray non-zero even when `i` lands on a multiple of 64, so a regression that
/// drops the mask (reads the full `u16`) leaks these bits and diverges from the
/// L-based oracles below — that is the load-bearing property.
const HIGH_MASK: u16 = (1u16 << (16 - BITS)) - 1; // 0b0011_1111
fn stray_high(i: usize) -> u16 {
  (((i as u16).wrapping_mul(37).wrapping_add(1) & HIGH_MASK) | 0b00_0001) << BITS
}

/// Build an NV20 (low-bit-packed) wire frame pair from logical planes: the
/// active bits stay in the low 10 (`logical & 0x03FF`), with STRAY non-zero
/// garbage OR'd into the high 6 padding bits (`stray_high`), encoded in `BE`
/// order. The de-pack's `& 0x03FF` must discard the stray bits, so the planes
/// still carry exactly the logical `L` — making the cross-packing /
/// area-bin-of-direct oracles load-bearing for the mask (without it the stray
/// bits leak and the NV20 outputs diverge from the L-based references).
fn nv20_planes(y_log: &[u16], uv_log: &[u16], be: bool) -> (Vec<u16>, Vec<u16>) {
  (
    y_log
      .iter()
      .enumerate()
      .map(|(i, &v)| enc((v & MASK) | stray_high(i), be))
      .collect(),
    uv_log
      .iter()
      .enumerate()
      .map(|(i, &v)| enc((v & MASK) | stray_high(i), be))
      .collect(),
  )
}

/// Build a P210 (high-bit-packed) wire frame pair carrying the SAME logical
/// planes: `logical << (16 - BITS)` (active bits in the high 10), `BE` order.
fn p210_planes(y_log: &[u16], uv_log: &[u16], be: bool) -> (Vec<u16>, Vec<u16>) {
  let shift = 16 - BITS;
  (
    y_log
      .iter()
      .map(|&v| enc((v & MASK) << shift, be))
      .collect(),
    uv_log
      .iter()
      .map(|&v| enc((v & MASK) << shift, be))
      .collect(),
  )
}

/// Every numeric output an NV20 / P210 sink can emit. Collected together so a
/// single equivalence assertion covers the whole surface (the rgb buffer is
/// the kernel output, hsv derives from the same rgb row — Strategy A).
struct Outputs {
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  luma: Vec<u8>,
  hsv: (Vec<u8>, Vec<u8>, Vec<u8>),
}

/// Plan selector: AREA (native toggled) vs FILTER (one of the three kernels).
#[derive(Clone, Copy)]
enum Tier {
  Area { native: bool },
  Filter(Kernel),
}

#[derive(Clone, Copy)]
enum Kernel {
  Triangle,
  CatmullRom,
  Lanczos3,
}

/// Drive an **NV20** source through `tier` for the full output set (rgb / rgba
/// / rgb_u16 / rgba_u16 / luma / hsv), const-generic over the wire endianness.
/// The `Nv20Frame<'_, BE>` plane data must already be wire-encoded for `BE`.
#[allow(clippy::too_many_arguments)]
fn run_nv20<const BE: bool>(
  yp: &[u16],
  uvp: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  tier: Tier,
  loc: crate::ChromaLocation,
) -> Outputs {
  let src = Nv20Frame::<BE>::new(yp, uvp, sw as u32, sh as u32, sw as u32, sw as u32);
  let mut o = blank_outputs(ow, oh);
  macro_rules! drive {
    ($sink:expr) => {{
      let mut sink = $sink
        .with_chroma_location(loc.clone())
        .with_rgb(&mut o.rgb)
        .unwrap()
        .with_rgba(&mut o.rgba)
        .unwrap()
        .with_rgb_u16(&mut o.rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut o.rgba_u16)
        .unwrap()
        .with_luma(&mut o.luma)
        .unwrap()
        .with_hsv(&mut o.hsv.0, &mut o.hsv.1, &mut o.hsv.2)
        .unwrap();
      nv20_to_endian::<_, BE>(&src, FR, M, &mut sink).unwrap();
    }};
  }
  match tier {
    Tier::Area { native } => {
      drive!(
        MixedSinker::<Nv20<BE>, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(native)
      );
    }
    Tier::Filter(k) => match k {
      Kernel::Triangle => drive!(
        MixedSinker::<Nv20<BE>, FilteredResampler<Triangle>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, Triangle),
        )
        .unwrap()
      ),
      Kernel::CatmullRom => drive!(
        MixedSinker::<Nv20<BE>, FilteredResampler<CatmullRom>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, CatmullRom),
        )
        .unwrap()
      ),
      Kernel::Lanczos3 => drive!(
        MixedSinker::<Nv20<BE>, FilteredResampler<Lanczos3>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, Lanczos3),
        )
        .unwrap()
      ),
    },
  }
  o
}

/// Drive a **P210** source through `tier` — the high-bit-packed reference. The
/// `P210Frame<'_, BE>` plane data must already be wire-encoded for `BE`.
#[allow(clippy::too_many_arguments)]
fn run_p210<const BE: bool>(
  yp: &[u16],
  uvp: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  tier: Tier,
  loc: crate::ChromaLocation,
) -> Outputs {
  let src = PnFrame422::<10, BE>::new(yp, uvp, sw as u32, sh as u32, sw as u32, sw as u32);
  let mut o = blank_outputs(ow, oh);
  macro_rules! drive {
    ($sink:expr) => {{
      let mut sink = $sink
        .with_chroma_location(loc.clone())
        .with_rgb(&mut o.rgb)
        .unwrap()
        .with_rgba(&mut o.rgba)
        .unwrap()
        .with_rgb_u16(&mut o.rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut o.rgba_u16)
        .unwrap()
        .with_luma(&mut o.luma)
        .unwrap()
        .with_hsv(&mut o.hsv.0, &mut o.hsv.1, &mut o.hsv.2)
        .unwrap();
      p210_to_endian::<_, BE>(&src, FR, M, &mut sink).unwrap();
    }};
  }
  match tier {
    Tier::Area { native } => {
      drive!(
        MixedSinker::<P210<BE>, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(native)
      );
    }
    Tier::Filter(k) => match k {
      Kernel::Triangle => drive!(
        MixedSinker::<P210<BE>, FilteredResampler<Triangle>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, Triangle),
        )
        .unwrap()
      ),
      Kernel::CatmullRom => drive!(
        MixedSinker::<P210<BE>, FilteredResampler<CatmullRom>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, CatmullRom),
        )
        .unwrap()
      ),
      Kernel::Lanczos3 => drive!(
        MixedSinker::<P210<BE>, FilteredResampler<Lanczos3>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, Lanczos3),
        )
        .unwrap()
      ),
    },
  }
  o
}

fn blank_outputs(ow: usize, oh: usize) -> Outputs {
  Outputs {
    rgb: vec![0u8; ow * oh * 3],
    rgba: vec![0u8; ow * oh * 4],
    rgb_u16: vec![0u16; ow * oh * 3],
    rgba_u16: vec![0u16; ow * oh * 4],
    luma: vec![0u8; ow * oh],
    hsv: (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]),
  }
}

fn assert_outputs_eq(got: &Outputs, want: &Outputs, ctx: &str) {
  assert_eq!(got.rgb, want.rgb, "{ctx}: rgb");
  assert_eq!(got.rgba, want.rgba, "{ctx}: rgba");
  assert_eq!(got.rgb_u16, want.rgb_u16, "{ctx}: rgb_u16");
  assert_eq!(got.rgba_u16, want.rgba_u16, "{ctx}: rgba_u16");
  assert_eq!(got.luma, want.luma, "{ctx}: luma");
  assert_eq!(got.hsv.0, want.hsv.0, "{ctx}: hsv H");
  assert_eq!(got.hsv.1, want.hsv.1, "{ctx}: hsv S");
  assert_eq!(got.hsv.2, want.hsv.2, "{ctx}: hsv V");
}

// ---- Independent area-bin-of-direct oracle (NV20 stands on its own) -------

/// 2x2-block area mean (round-half-up) of an `sw x sh` interleaved-channel u8
/// plane (`ch` channels per pixel) down to `(sw/2) x (sh/2)`.
fn block_mean_2x2_u8(src: &[u8], sw: usize, sh: usize, ch: usize) -> Vec<u8> {
  let (ow, oh) = (sw / 2, sh / 2);
  let mut out = vec![0u8; ow * oh * ch];
  for oy in 0..oh {
    for ox in 0..ow {
      for c in 0..ch {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += src[((oy * 2 + dy) * sw + ox * 2 + dx) * ch + c] as u32;
          }
        }
        out[(oy * ow + ox) * ch + c] = ((s + 2) / 4) as u8;
      }
    }
  }
  out
}

/// 2x2-block area mean (round-half-up) of an `sw x sh` interleaved-channel u16
/// plane down to `(sw/2) x (sh/2)`.
fn block_mean_2x2_u16(src: &[u16], sw: usize, sh: usize, ch: usize) -> Vec<u16> {
  let (ow, oh) = (sw / 2, sh / 2);
  let mut out = vec![0u16; ow * oh * ch];
  for oy in 0..oh {
    for ox in 0..ow {
      for c in 0..ch {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += src[((oy * 2 + dy) * sw + ox * 2 + dx) * ch + c] as u32;
          }
        }
        out[(oy * ow + ox) * ch + c] = ((s + 2) / 4) as u16;
      }
    }
  }
  out
}

/// Direct (identity) NV20 conversion at source resolution for the full output
/// set — the input the area-bin-of-direct oracle averages.
fn direct_nv20<const BE: bool>(yp: &[u16], uvp: &[u16], sw: usize, sh: usize) -> Outputs {
  let src = Nv20Frame::<BE>::new(yp, uvp, sw as u32, sh as u32, sw as u32, sw as u32);
  let mut o = blank_outputs(sw, sh);
  let mut sink = MixedSinker::<Nv20<BE>>::new(sw, sh)
    .with_rgb(&mut o.rgb)
    .unwrap()
    .with_rgba(&mut o.rgba)
    .unwrap()
    .with_rgb_u16(&mut o.rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut o.rgba_u16)
    .unwrap()
    .with_luma(&mut o.luma)
    .unwrap()
    .with_hsv(&mut o.hsv.0, &mut o.hsv.1, &mut o.hsv.2)
    .unwrap();
  nv20_to_endian::<_, BE>(&src, FR, M, &mut sink).unwrap();
  o
}

/// The area-bin-of-direct oracle: 2x2 block-mean the direct NV20 conversion.
/// `rgb` / `rgba` / `rgb_u16` / `rgba_u16` bin the converted source pixels;
/// `luma` bins the de-packed logical Y then narrows; hsv is re-derived from
/// the binned u8 rgb (`rgb_to_hsv_row`), matching Strategy A. Used to pin the
/// ROW-STAGE area tier (convert-then-bin) exactly.
fn area_bin_of_direct<const BE: bool>(
  yp_log: &[u16],
  yp_wire: &[u16],
  uvp_wire: &[u16],
  sw: usize,
  sh: usize,
) -> Outputs {
  let d = direct_nv20::<BE>(yp_wire, uvp_wire, sw, sh);
  let (ow, oh) = (sw / 2, sh / 2);
  let rgb = block_mean_2x2_u8(&d.rgb, sw, sh, 3);

  // luma oracle: area-bin the de-packed logical Y, then narrow `>> (BITS - 8)`.
  let logical_y: Vec<u16> = yp_log.iter().map(|&v| v & MASK).collect();
  let y_binned = block_mean_2x2_u16(&logical_y, sw, sh, 1);
  let luma: Vec<u8> = y_binned.iter().map(|&v| (v >> (BITS - 8)) as u8).collect();

  // hsv oracle: re-derive from the binned u8 rgb (the same rgb the resample
  // tier feeds `rgb_to_hsv_row`).
  let (mut hh, mut ss, mut vv) = (vec![0u8; ow * oh], vec![0u8; ow * oh], vec![0u8; ow * oh]);
  crate::row::rgb_to_hsv_row(&rgb, &mut hh, &mut ss, &mut vv, ow * oh, false);

  Outputs {
    rgb,
    rgba: block_mean_2x2_u8(&d.rgba, sw, sh, 4),
    rgb_u16: block_mean_2x2_u16(&d.rgb_u16, sw, sh, 3),
    rgba_u16: block_mean_2x2_u16(&d.rgba_u16, sw, sh, 4),
    luma,
    hsv: (hh, ss, vv),
  }
}

// ---- Independent area-bin oracle: row-stage area, LE + BE ------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_rowstage_matches_area_bin_of_direct_le() {
  let (sw, sh) = (8, 8);
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, false);
  let got = run_nv20::<false>(
    &yp,
    &uvp,
    sw,
    sh,
    sw / 2,
    sh / 2,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
  );
  let want = area_bin_of_direct::<false>(&y_log, &yp, &uvp, sw, sh);
  // hsv excluded from the rgb_to_hsv_row scalar-vs-SIMD detail: the resample
  // tier may run hsv through the host SIMD path while the oracle pins scalar.
  // Compare every output except hsv here; hsv is cross-checked via the P210
  // equivalence (same tier on both sides) below.
  assert_eq!(got.rgb, want.rgb, "LE rowstage area rgb");
  assert_eq!(got.rgba, want.rgba, "LE rowstage area rgba");
  assert_eq!(got.rgb_u16, want.rgb_u16, "LE rowstage area rgb_u16");
  assert_eq!(got.rgba_u16, want.rgba_u16, "LE rowstage area rgba_u16");
  assert_eq!(got.luma, want.luma, "LE rowstage area luma");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_rowstage_matches_area_bin_of_direct_be() {
  let (sw, sh) = (8, 8);
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, true);
  let got = run_nv20::<true>(
    &yp,
    &uvp,
    sw,
    sh,
    sw / 2,
    sh / 2,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
  );
  let want = area_bin_of_direct::<true>(&y_log, &yp, &uvp, sw, sh);
  assert_eq!(got.rgb, want.rgb, "BE rowstage area rgb");
  assert_eq!(got.rgba, want.rgba, "BE rowstage area rgba");
  assert_eq!(got.rgb_u16, want.rgb_u16, "BE rowstage area rgb_u16");
  assert_eq!(got.rgba_u16, want.rgba_u16, "BE rowstage area rgba_u16");
  assert_eq!(got.luma, want.luma, "BE rowstage area luma");
}

/// The native fast tier's luma is bit-identical to the area-bin of the
/// de-packed logical Y (cv2 INTER_AREA parity) — guards the NV20 native
/// `LOW_PACKED = true` Y de-pack (`& 0x03FF`). Colour averages in YUV (native)
/// vs RGB (row-stage) so only luma is bit-exact here; native colour
/// correctness is pinned by the P210 equivalence below.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_luma_is_inter_area_bin_of_depacked_y() {
  let (sw, sh) = (8, 8);
  let (y_log, uv_log) = logical_ramp(sw, sh);
  for &be in &[false, true] {
    let (luma, want) = if be {
      let (yp, uvp) = nv20_planes(&y_log, &uv_log, true);
      let got = run_nv20::<true>(
        &yp,
        &uvp,
        sw,
        sh,
        sw / 2,
        sh / 2,
        Tier::Area { native: true },
        crate::ChromaLocation::Left,
      );
      let want = area_bin_of_direct::<true>(&y_log, &yp, &uvp, sw, sh).luma;
      (got.luma, want)
    } else {
      let (yp, uvp) = nv20_planes(&y_log, &uv_log, false);
      let got = run_nv20::<false>(
        &yp,
        &uvp,
        sw,
        sh,
        sw / 2,
        sh / 2,
        Tier::Area { native: true },
        crate::ChromaLocation::Left,
      );
      let want = area_bin_of_direct::<false>(&y_log, &yp, &uvp, sw, sh).luma;
      (got.luma, want)
    };
    assert_eq!(
      luma, want,
      "native luma must be the INTER_AREA bin of the de-packed Y (be={be})"
    );
  }
}

// ---- ★ Cross-packing equivalence: NV20(low L) == P210(high L<<6) -----------
//
// The load-bearing resample-path check. The SAME logical samples, packed two
// different ways, must resample to the SAME output on every tier — proving the
// NV20 resample path's low-bit extraction (`& 0x03FF`, and the native
// `LOW_PACKED = true` depack) reads the active bits correctly. P210's resample
// output is independently pinned (resample_p2xx_high_bit{,_native}), so this
// transfers that ground truth onto NV20.

fn assert_nv20_equals_p210<const BE: bool>(
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  tier: Tier,
  loc: crate::ChromaLocation,
  ctx: &str,
) {
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (nv_y, nv_uv) = nv20_planes(&y_log, &uv_log, BE);
  let (p_y, p_uv) = p210_planes(&y_log, &uv_log, BE);
  let nv = run_nv20::<BE>(&nv_y, &nv_uv, sw, sh, ow, oh, tier, loc.clone());
  let p = run_p210::<BE>(&p_y, &p_uv, sw, sh, ow, oh, tier, loc);
  assert_outputs_eq(&nv, &p, ctx);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_rowstage_nv20_equals_p210_le() {
  assert_nv20_equals_p210::<false>(
    8,
    8,
    4,
    4,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
    "LE area row-stage",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_rowstage_nv20_equals_p210_be() {
  assert_nv20_equals_p210::<true>(
    8,
    8,
    4,
    4,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
    "BE area row-stage",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_native_nv20_equals_p210_le() {
  assert_nv20_equals_p210::<false>(
    8,
    8,
    4,
    4,
    Tier::Area { native: true },
    crate::ChromaLocation::Left,
    "LE area native",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_native_nv20_equals_p210_be() {
  assert_nv20_equals_p210::<true>(
    8,
    8,
    4,
    4,
    Tier::Area { native: true },
    crate::ChromaLocation::Left,
    "BE area native",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_nv20_equals_p210_le() {
  for k in [Kernel::Triangle, Kernel::CatmullRom, Kernel::Lanczos3] {
    assert_nv20_equals_p210::<false>(
      8,
      8,
      4,
      4,
      Tier::Filter(k),
      crate::ChromaLocation::Left,
      "LE filter",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_nv20_equals_p210_be() {
  for k in [Kernel::Triangle, Kernel::CatmullRom, Kernel::Lanczos3] {
    assert_nv20_equals_p210::<true>(
      8,
      8,
      4,
      4,
      Tier::Filter(k),
      crate::ChromaLocation::Left,
      "BE filter",
    );
  }
}

/// The native-on vs native-off distinction must be load-bearing for NV20 too:
/// over the ramp the two averaging domains (YUV vs RGB) diverge on colour, so
/// `with_native(true)` is NOT a no-op rename of the row-stage path. (Luma stays
/// bit-identical across tiers — checked above.) Guards against a native route
/// that silently falls through to row-stage.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_and_rowstage_color_differ_on_ramp() {
  let (sw, sh) = (8, 8);
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, false);
  let native = run_nv20::<false>(
    &yp,
    &uvp,
    sw,
    sh,
    4,
    4,
    Tier::Area { native: true },
    crate::ChromaLocation::Left,
  );
  let rowstage = run_nv20::<false>(
    &yp,
    &uvp,
    sw,
    sh,
    4,
    4,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
  );
  assert_eq!(
    native.luma, rowstage.luma,
    "luma is bit-identical across tiers"
  );
  assert_ne!(
    native.rgb_u16, rowstage.rgb_u16,
    "native (average-in-YUV) and row-stage (convert-then-average) colour must differ on the ramp"
  );
}

// ---- Non-multiple tail geometry (partial bins) ----------------------------
//
// Source dims not an integer multiple of the output exercise the area-bin
// partial-coverage path and the filter's fractional sampling. The cross-
// packing equivalence holds regardless of geometry, so it is the right oracle:
// a tail mask/shift bug breaks NV20 vs P210 just as an aligned one would.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_tail_geometry_nv20_equals_p210_le() {
  // 6x6 -> 4x4: x and y ratios 6/4 are non-integer, so output bins straddle
  // source pixels with fractional coverage. Width 6 keeps NV20's even-width
  // 4:2:2 contract.
  assert_nv20_equals_p210::<false>(
    6,
    6,
    4,
    4,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
    "LE area tail 6->4",
  );
  assert_nv20_equals_p210::<false>(
    6,
    6,
    4,
    4,
    Tier::Area { native: true },
    crate::ChromaLocation::Left,
    "LE area tail native 6->4",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_tail_geometry_nv20_equals_p210_be() {
  assert_nv20_equals_p210::<true>(
    6,
    6,
    4,
    4,
    Tier::Area { native: false },
    crate::ChromaLocation::Left,
    "BE area tail 6->4",
  );
  assert_nv20_equals_p210::<true>(
    6,
    6,
    4,
    4,
    Tier::Area { native: true },
    crate::ChromaLocation::Left,
    "BE area tail native 6->4",
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_tail_geometry_nv20_equals_p210() {
  // 6x6 -> 5x5 (downscale, non-integer) and 4x4 -> 7x7 (upscale) both exercise
  // fractional kernel sampling. LE + BE, all three kernels.
  for k in [Kernel::Triangle, Kernel::CatmullRom, Kernel::Lanczos3] {
    assert_nv20_equals_p210::<false>(
      6,
      6,
      5,
      5,
      Tier::Filter(k),
      crate::ChromaLocation::Left,
      "LE filter tail 6->5",
    );
    assert_nv20_equals_p210::<true>(
      6,
      6,
      5,
      5,
      Tier::Filter(k),
      crate::ChromaLocation::Left,
      "BE filter tail 6->5",
    );
    assert_nv20_equals_p210::<false>(
      4,
      4,
      7,
      7,
      Tier::Filter(k),
      crate::ChromaLocation::Left,
      "LE filter upscale 4->7",
    );
  }
}

// ---- ★ Identity CENTERED chroma siting (RFC #238 NV20-PR1) ------------------
//
// The IDENTITY (no-resample) centered path is net-new in PR1: a centered NV20
// identity decode reconstructs full-width chroma via the low-packed
// `chroma_upsample_2to1_center_h_p0xx::<10, true>` + the low-packed `nv20_444_*`
// 4:4:4 kernels, instead of the co-sited fused decode. The load-bearing checks:
//
//  1. NV20-centered == P210-centered (cross-packing): the SAME logical samples,
//     packed low (NV20, with STRAY dirty high bits) vs high (P210), decode
//     identically — proving the low-packed 4:4:4 kernel + upsample sanitize the
//     dirty bits exactly as P210's `>> 6` shift and reconstruct the same chroma.
//     The identity twin of `assert_nv20_equals_p210`.
//  2. centered != co-sited on a horizontal chroma ramp (the phase-0.5 shift is
//     observable) — the negative control.
//  3. centered == co-sited on FLAT chroma (the 1/4–3/4 blend of equal samples,
//     with edge clamp, is a no-op) — the positive control.

/// Identity (no-resample) NV20 conversion at `loc` for the full output set.
#[cfg(feature = "yuv-planar")]
fn direct_nv20_sited<const BE: bool>(
  yp: &[u16],
  uvp: &[u16],
  sw: usize,
  sh: usize,
  loc: crate::ChromaLocation,
) -> Outputs {
  let src = Nv20Frame::<BE>::new(yp, uvp, sw as u32, sh as u32, sw as u32, sw as u32);
  let mut o = blank_outputs(sw, sh);
  let mut sink = MixedSinker::<Nv20<BE>>::new(sw, sh)
    .with_chroma_location(loc.clone())
    .with_rgb(&mut o.rgb)
    .unwrap()
    .with_rgba(&mut o.rgba)
    .unwrap()
    .with_rgb_u16(&mut o.rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut o.rgba_u16)
    .unwrap()
    .with_luma(&mut o.luma)
    .unwrap()
    .with_hsv(&mut o.hsv.0, &mut o.hsv.1, &mut o.hsv.2)
    .unwrap();
  nv20_to_endian::<_, BE>(&src, FR, M, &mut sink).unwrap();
  o
}

/// Identity (no-resample) P210 conversion at `loc` — the high-bit-packed
/// reference for the centered cross-packing check.
#[cfg(feature = "yuv-planar")]
fn direct_p210_sited<const BE: bool>(
  yp: &[u16],
  uvp: &[u16],
  sw: usize,
  sh: usize,
  loc: crate::ChromaLocation,
) -> Outputs {
  let src = PnFrame422::<10, BE>::new(yp, uvp, sw as u32, sh as u32, sw as u32, sw as u32);
  let mut o = blank_outputs(sw, sh);
  let mut sink = MixedSinker::<P210<BE>>::new(sw, sh)
    .with_chroma_location(loc.clone())
    .with_rgb(&mut o.rgb)
    .unwrap()
    .with_rgba(&mut o.rgba)
    .unwrap()
    .with_rgb_u16(&mut o.rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut o.rgba_u16)
    .unwrap()
    .with_luma(&mut o.luma)
    .unwrap()
    .with_hsv(&mut o.hsv.0, &mut o.hsv.1, &mut o.hsv.2)
    .unwrap();
  p210_to_endian::<_, BE>(&src, FR, M, &mut sink).unwrap();
  o
}

/// Logical Y ramp + FLAT (constant) chroma — the positive control for the
/// centered==co-sited check (a constant survives the phase-0.5 blend unchanged).
#[cfg(feature = "yuv-planar")]
fn logical_flat_chroma(sw: usize, sh: usize) -> (Vec<u16>, Vec<u16>) {
  let cw = sw / 2;
  let mut y = vec![0u16; sw * sh];
  for (i, yi) in y.iter_mut().enumerate() {
    *yi = ((40u32 + i as u32 * 37) & MASK as u32) as u16;
  }
  // Constant U = 500, V = 520 across every chroma sample.
  let mut uv = vec![0u16; cw * sh * 2];
  for i in 0..cw * sh {
    uv[2 * i] = 500 & MASK;
    uv[2 * i + 1] = 520 & MASK;
  }
  (y, uv)
}

#[cfg(feature = "yuv-planar")]
fn assert_identity_center_nv20_equals_p210<const BE: bool>(sw: usize, sh: usize, ctx: &str) {
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (nv_y, nv_uv) = nv20_planes(&y_log, &uv_log, BE); // low-packed + STRAY dirty bits
  let (p_y, p_uv) = p210_planes(&y_log, &uv_log, BE); // high-packed
  let center = crate::ChromaLocation::Center;
  let nv = direct_nv20_sited::<BE>(&nv_y, &nv_uv, sw, sh, center.clone());
  let p = direct_p210_sited::<BE>(&p_y, &p_uv, sw, sh, center);
  assert_outputs_eq(&nv, &p, ctx);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_nv20_equals_p210_le() {
  assert_identity_center_nv20_equals_p210::<false>(8, 4, "LE identity centered NV20==P210");
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_nv20_equals_p210_be() {
  assert_identity_center_nv20_equals_p210::<true>(8, 4, "BE identity centered NV20==P210");
}

/// Centered ≠ co-sited on a horizontal chroma ramp: the phase-0.5 reconstruction
/// shifts chroma, so the colour outputs must diverge from the fused co-sited
/// decode (luma is siting-independent, so compare rgb only).
#[cfg(feature = "yuv-planar")]
fn assert_identity_center_differs_from_cosited<const BE: bool>(sw: usize, sh: usize) {
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (nv_y, nv_uv) = nv20_planes(&y_log, &uv_log, BE);
  let centered = direct_nv20_sited::<BE>(&nv_y, &nv_uv, sw, sh, crate::ChromaLocation::Center);
  let cosited = direct_nv20_sited::<BE>(&nv_y, &nv_uv, sw, sh, crate::ChromaLocation::Left);
  assert_ne!(
    centered.rgb, cosited.rgb,
    "centered chroma siting must differ from co-sited on a horizontal ramp (be={BE})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_differs_from_cosited_on_ramp_le() {
  assert_identity_center_differs_from_cosited::<false>(8, 4);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_differs_from_cosited_on_ramp_be() {
  assert_identity_center_differs_from_cosited::<true>(8, 4);
}

/// Centered == co-sited on FLAT chroma: the 1/4–3/4 blend (+ edge clamp) of a
/// constant is that constant, so the reconstructed decode is byte-identical to
/// the fused co-sited decode across the whole output set.
#[cfg(feature = "yuv-planar")]
fn assert_identity_center_equals_cosited_on_flat<const BE: bool>(sw: usize, sh: usize, ctx: &str) {
  let (y_log, uv_log) = logical_flat_chroma(sw, sh);
  let (nv_y, nv_uv) = nv20_planes(&y_log, &uv_log, BE);
  let centered = direct_nv20_sited::<BE>(&nv_y, &nv_uv, sw, sh, crate::ChromaLocation::Center);
  let cosited = direct_nv20_sited::<BE>(&nv_y, &nv_uv, sw, sh, crate::ChromaLocation::Left);
  assert_outputs_eq(&centered, &cosited, ctx);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_equals_cosited_on_flat_le() {
  assert_identity_center_equals_cosited_on_flat::<false>(8, 4, "LE flat centered==cosited");
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_center_equals_cosited_on_flat_be() {
  assert_identity_center_equals_cosited_on_flat::<true>(8, 4, "BE flat centered==cosited");
}

/// Per-frame chroma-siting freeze on the IDENTITY path: process the first
/// output-bearing row at one siting, flip `set_chroma_location`, then process
/// the next in-sequence row — the sink must reject it with `ChromaSitingChanged`
/// (a mid-frame flip would decode a mixture of centered + co-sited chroma into
/// ONE frame). Both directions (co-sited ⇆ centered).
#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_mid_frame_siting_flip_rejected() {
  use crate::{ChromaLocation, PixelSink, sinker::MixedSinkerError, source::Nv20Row};
  let (sw, sh) = (8usize, 4usize);
  let cw = sw / 2;
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, false); // LE wire
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
  ] {
    let mut rgb = vec![0u8; sw * sh * 3];
    let mut sink = MixedSinker::<Nv20>::new(sw, sh)
      .with_chroma_location(loc1.clone())
      .with_rgb(&mut rgb)
      .unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    let row0 = Nv20Row::new(&yp[0..sw], &uvp[0..2 * cw], 0, M, FR);
    PixelSink::process(&mut sink, row0).unwrap();
    sink.set_chroma_location(loc2.clone());
    let row1 = Nv20Row::new(&yp[sw..2 * sw], &uvp[2 * cw..4 * cw], 1, M, FR);
    let err = PixelSink::process(&mut sink, row1).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "identity {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}

// ---- ★ Resample CENTERED chroma siting: NV20-centered == P210-centered ------
//
// The resample twin of the identity centered cross-check above, and of the
// co-sited `assert_nv20_equals_p210` family: routing `ChromaLocation::Center`
// through the 3 resample tiers, the SAME logical samples packed low (NV20, with
// STRAY dirty high bits) vs high (P210) must resample IDENTICALLY. The sibling
// `chroma_siting_hibit_422_semiplanar_resample` suite pins P210-centered to the
// planar `Yuv422p` / `Yuv444p` oracle EXACTLY, so NV20-centered == P210-centered
// transitively pins NV20's centered resample to the true oracle across the
// native, row-stage, and filter tiers.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_center_nv20_equals_p210_le() {
  for native in [false, true] {
    assert_nv20_equals_p210::<false>(
      8,
      8,
      4,
      4,
      Tier::Area { native },
      crate::ChromaLocation::Center,
      "LE area centered",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_center_nv20_equals_p210_be() {
  for native in [false, true] {
    assert_nv20_equals_p210::<true>(
      8,
      8,
      4,
      4,
      Tier::Area { native },
      crate::ChromaLocation::Center,
      "BE area centered",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_center_nv20_equals_p210_le() {
  for k in [Kernel::Triangle, Kernel::CatmullRom, Kernel::Lanczos3] {
    assert_nv20_equals_p210::<false>(
      8,
      8,
      4,
      4,
      Tier::Filter(k),
      crate::ChromaLocation::Center,
      "LE filter centered",
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn filter_center_nv20_equals_p210_be() {
  for k in [Kernel::Triangle, Kernel::CatmullRom, Kernel::Lanczos3] {
    assert_nv20_equals_p210::<true>(
      8,
      8,
      4,
      4,
      Tier::Filter(k),
      crate::ChromaLocation::Center,
      "BE filter centered",
    );
  }
}

/// Non-vacuity: centered ≠ co-sited on a horizontal chroma ramp for EVERY
/// resample tier. A "centered" arm that silently stayed on the co-sited decode
/// would make these equal, so the phase-0.5 reconstruction must move the colour
/// output — otherwise the cross-checks above would pass vacuously.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn resample_centered_differs_from_cosited_on_ramp() {
  let (sw, sh) = (8, 8);
  let (ow, oh) = (4, 4);
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, false);
  for tier in [
    Tier::Area { native: false },
    Tier::Area { native: true },
    Tier::Filter(Kernel::Triangle),
    Tier::Filter(Kernel::CatmullRom),
    Tier::Filter(Kernel::Lanczos3),
  ] {
    let centered = run_nv20::<false>(
      &yp,
      &uvp,
      sw,
      sh,
      ow,
      oh,
      tier,
      crate::ChromaLocation::Center,
    );
    let cosited = run_nv20::<false>(&yp, &uvp, sw, sh, ow, oh, tier, crate::ChromaLocation::Left);
    assert_ne!(
      centered.rgb_u16, cosited.rgb_u16,
      "centered chroma siting must differ from co-sited on a horizontal ramp"
    );
  }
}

/// Per-frame chroma-siting freeze on the RESAMPLE path (the resample-branch
/// `frozen_chroma_centered` CHECK, the twin of PR1's identity-path freeze): feed
/// the first output-bearing row at one siting, flip `set_chroma_location`, then
/// feed the next in-sequence row — the resample sink must reject it with
/// `ChromaSitingChanged` (a mid-frame flip would bin a mixture of centered +
/// co-sited chroma into ONE frame). Both directions, on the row-stage area tier.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn resample_mid_frame_siting_flip_rejected() {
  use crate::{ChromaLocation, PixelSink, sinker::MixedSinkerError, source::Nv20Row};
  let (sw, sh) = (8usize, 4usize);
  let (ow, oh) = (4usize, 2usize);
  let cw = sw / 2;
  let (y_log, uv_log) = logical_ramp(sw, sh);
  let (yp, uvp) = nv20_planes(&y_log, &uv_log, false); // LE wire
  for (loc1, loc2) in [
    (ChromaLocation::Center, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Center),
  ] {
    let mut rgb = vec![0u8; ow * oh * 3];
    let mut sink =
      MixedSinker::<Nv20, AreaResampler>::with_resampler(sw, sh, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(false)
        .with_chroma_location(loc1.clone())
        .with_rgb(&mut rgb)
        .unwrap();
    PixelSink::begin_frame(&mut sink, sw as u32, sh as u32).unwrap();
    let row0 = Nv20Row::new(&yp[0..sw], &uvp[0..2 * cw], 0, M, FR);
    PixelSink::process(&mut sink, row0).unwrap();
    sink.set_chroma_location(loc2.clone());
    let row1 = Nv20Row::new(&yp[sw..2 * sw], &uvp[2 * cw..4 * cw], 1, M, FR);
    let err = PixelSink::process(&mut sink, row1).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "resample {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
  }
}
