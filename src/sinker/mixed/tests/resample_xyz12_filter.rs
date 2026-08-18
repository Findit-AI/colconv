//! Filter-resample coverage for the packed-CIE-XYZ-12-bit family
//! ([`Xyz12`](crate::source::Xyz12)) routed through the separable filter
//! engine.
//!
//! `Xyz12` bins in **linear XYZ** `f32` (SMPTE ST 428-1 inverse-OETF
//! only, pre-matrix); the gamut matrix + sRGB OETF + narrow run per
//! finalized output row, exactly as on the area path. The engine's `f32`
//! filter is parity-within-tolerance versus PIL (not 0-ULP), and the
//! two float outputs (`xyz_f32` / `rgb_f32`) are full-range by design, so
//! there is no per-channel PIL golden. Instead this mirrors the
//! per-channel equivalence the packed-RGBA / `Rgbf32` filter tests use:
//!
//! 1. **Per-channel equivalence.** The binned linear XYZ a 3-channel
//!    `Xyz12` filter resample produces (its `xyz_f32` output) must equal
//!    **bit-for-bit** the single-channel [`FilterStream<f32>`] resample
//!    of each source XYZ plane — the *same engine*, run per plane —
//!    because the merged engine filters each channel independently. The
//!    derived integer / f16 outputs must then equal the direct path's
//!    matrix + OETF + narrow applied to that binned XYZ. Covered for
//!    `Triangle` / `CatmullRom` / `Lanczos3` across a downscale (8 -> 4)
//!    and an upscale (4 -> 7).
//! 2. **Output contract.** A `CatmullRom` / `Lanczos3` edge that
//!    overshoots must push the **unclamped** outputs (`xyz_f32` /
//!    `rgb_f32`) out of `[0, 1]` — full-range float preserved, mirroring
//!    the area path's `xyz12_downscale_preserves_hdr_and_out_of_gamut`
//!    contract — while the **clamped** outputs (`rgb` / `rgb_u16` /
//!    `rgb_f16`) stay in range with no wrap (their narrows clamp `[0, 1]`,
//!    so a signed-coefficient overshoot cannot wrap them).
//! 3. **Filter-plan-accepted regression.** A filter plan must no longer
//!    raise `UnsupportedFilter` at the `Xyz12` fence.

use crate::{
  KernelGamut,
  frame::Xyz12LeFrame,
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  row::scalar::{
    xyz12::{matmul3_xyz_rgb, narrow_unit_to_u8, narrow_unit_to_u16, oetf_srgb},
    xyz12_constants::xyz_to_rgb_matrix,
  },
  sinker::MixedSinker,
  source::{Xyz12Le, xyz12_to},
};

/// Encodes a 12-bit code into the high-bit-packed LE wire `u16`
/// (`code << 4`, then LE bytes reinterpreted as host-native). Per FFmpeg
/// `AV_PIX_FMT_XYZ12LE`: active 12 bits in `[15:4]`, low 4 bits zero.
/// Host-independent (same logical wire value on LE and BE hosts).
fn pack12_le(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_le_bytes())
}

/// A `w x h` wire frame whose 12-bit codes vary per pixel and channel so
/// every filter window sees distinct neighbours (a channel mix-up or a
/// row/column transpose diverges immediately).
fn varying_frame(w: usize, h: usize) -> Vec<u16> {
  let mut buf = vec![0u16; w * h * 3];
  for (i, c) in buf.iter_mut().enumerate() {
    let code = ((i * 173 + 11) % 4096) as u16;
    *c = pack12_le(code);
  }
  buf
}

/// Full-res `with_xyz_f32` of a wire frame — the production source of
/// truth for source-width **linear XYZ** (SMPTE ST 428-1 inverse-OETF,
/// no matrix). This is the per-plane filter oracle's input.
fn full_res_linear_xyz(wire: &[u16], w: usize, h: usize, gamut: KernelGamut) -> Vec<f32> {
  let src = Xyz12LeFrame::try_new(wire, w as u32, h as u32, (w * 3) as u32).unwrap();
  let mut xyz = vec![0.0f32; w * h * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(w, h)
    .with_simd(false)
    .with_xyz_f32(&mut xyz)
    .unwrap();
  xyz12_to(&src, gamut, &mut sink).unwrap();
  xyz
}

/// Single-channel filter resample of channel `c` of a packed `X, Y, Z`
/// linear-XYZ `f32` plane, via the merged engine's [`FilterStream<f32>`]
/// (channels = 1) — the per-channel oracle. The 3-channel `Xyz12` filter
/// resample's binned channel `c` must equal this **bit-for-bit**: same
/// engine, same coefficients, run independently per plane.
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter<K: FilterKernel>(
  kernel: K,
  packed_xyz: &[f32],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  use_simd: bool,
) -> Vec<f32> {
  let mut plane = vec![0.0f32; sw * sh];
  for (dst, px) in plane.iter_mut().zip(packed_xyz.chunks_exact(3)) {
    *dst = px[c];
  }
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<f32>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0.0f32; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], use_simd, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Runs the `Xyz12` filter sink over a wire frame and returns every
/// resampled output that the per-channel equivalence asserts on.
struct FilterOutputs {
  rgb: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgb_f32: Vec<f32>,
  xyz_f32: Vec<f32>,
  rgb_f16: Vec<half::f16>,
}

fn xyz12_filter_outputs<K: FilterKernel + Copy>(
  wire: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  gamut: KernelGamut,
  kernel: K,
) -> FilterOutputs {
  let src = Xyz12LeFrame::try_new(wire, sw as u32, sh as u32, (sw * 3) as u32).unwrap();
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgb_u16 = vec![0u16; ow * oh * 3];
  let mut rgb_f32 = vec![0.0f32; ow * oh * 3];
  let mut xyz_f32 = vec![0.0f32; ow * oh * 3];
  let mut rgb_f16 = vec![half::f16::ZERO; ow * oh * 3];
  {
    let mut sink = MixedSinker::<Xyz12Le, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_simd(false)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap()
    .with_xyz_f32(&mut xyz_f32)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16)
    .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  FilterOutputs {
    rgb,
    rgb_u16,
    rgb_f32,
    xyz_f32,
    rgb_f16,
  }
}

/// Asserts the 3-channel `Xyz12` filter outputs equal the per-channel
/// single-plane [`FilterStream<f32>`] resample of the source XYZ planes,
/// then the direct path's matrix + OETF + narrow applied to that binned
/// XYZ. Returns the max per-channel `xyz_f32` diff (exactly 0 — same
/// engine).
#[allow(clippy::too_many_arguments)]
fn assert_filter_is_per_channel<K: FilterKernel + Copy>(
  kernel: K,
  wire: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  gamut: KernelGamut,
  ctx: &str,
) -> f32 {
  let got = xyz12_filter_outputs(wire, sw, sh, ow, oh, gamut, kernel);
  let src_xyz = full_res_linear_xyz(wire, sw, sh, gamut);
  let m = xyz_to_rgb_matrix(gamut);

  // Rebuild the binned linear XYZ from 3 independent single-channel
  // filter runs — the per-channel oracle.
  let planes: [Vec<f32>; 3] = [
    channel_plane_filter(kernel, &src_xyz, 0, sw, sh, ow, oh, true),
    channel_plane_filter(kernel, &src_xyz, 1, sw, sh, ow, oh, true),
    channel_plane_filter(kernel, &src_xyz, 2, sw, sh, ow, oh, true),
  ];

  let mut max_diff = 0.0f32;
  for (px, ((xyz_out, rgb_f32_out), (rgb_out, rgb_u16_out))) in got
    .xyz_f32
    .chunks_exact(3)
    .zip(got.rgb_f32.chunks_exact(3))
    .zip(got.rgb.chunks_exact(3).zip(got.rgb_u16.chunks_exact(3)))
    .enumerate()
  {
    let binned = [planes[0][px], planes[1][px], planes[2][px]];
    let rgb_f16_out = &got.rgb_f16[px * 3..px * 3 + 3];
    // 1. The `xyz_f32` output IS the binned linear XYZ — must equal the
    //    per-plane filter, bit-for-bit (same engine, run per channel).
    for (c, (&g, &want)) in xyz_out.iter().zip(binned.iter()).enumerate() {
      let diff = (g - want).abs();
      if diff > max_diff {
        max_diff = diff;
      }
      assert_eq!(
        g.to_bits(),
        want.to_bits(),
        "{ctx} xyz_f32 px {px} c{c}: {g} vs per-plane filter {want}"
      );
    }
    // 2. The derived outputs = the direct path's narrow over the binned
    //    XYZ (matrix -> linear RGB -> OETF -> clamp/scale/narrow).
    let lin = matmul3_xyz_rgb(&m, binned);
    for (c, &lin_c) in lin.iter().enumerate() {
      assert_eq!(
        rgb_f32_out[c].to_bits(),
        lin_c.to_bits(),
        "{ctx} rgb_f32 px {px} c{c}"
      );
      let oetf = oetf_srgb(lin_c);
      assert_eq!(
        rgb_out[c],
        narrow_unit_to_u8(oetf),
        "{ctx} rgb px {px} c{c}"
      );
      assert_eq!(
        rgb_u16_out[c],
        narrow_unit_to_u16(oetf),
        "{ctx} rgb_u16 px {px} c{c}"
      );
      assert_eq!(
        rgb_f16_out[c].to_bits(),
        half::f16::from_f32(oetf.clamp(0.0, 1.0)).to_bits(),
        "{ctx} rgb_f16 px {px} c{c}"
      );
    }
  }
  assert_eq!(max_diff, 0.0, "{ctx}: per-channel xyz_f32 diff must be 0");
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_downscale_filter_is_per_channel() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let wire = varying_frame(SW, SH);
  for &gamut in &[
    KernelGamut::DciP3,
    KernelGamut::Rec709,
    KernelGamut::Rec2020,
  ] {
    assert_filter_is_per_channel(Triangle, &wire, SW, SH, OW, OH, gamut, "triangle down");
    assert_filter_is_per_channel(CatmullRom, &wire, SW, SH, OW, OH, gamut, "catmullrom down");
    assert_filter_is_per_channel(Lanczos3, &wire, SW, SH, OW, OH, gamut, "lanczos3 down");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_upscale_filter_is_per_channel() {
  const SW: usize = 4;
  const SH: usize = 4;
  const OW: usize = 7;
  const OH: usize = 7;
  let wire = varying_frame(SW, SH);
  for &gamut in &[
    KernelGamut::DciP3,
    KernelGamut::Rec709,
    KernelGamut::Rec2020,
  ] {
    assert_filter_is_per_channel(Triangle, &wire, SW, SH, OW, OH, gamut, "triangle up");
    assert_filter_is_per_channel(CatmullRom, &wire, SW, SH, OW, OH, gamut, "catmullrom up");
    assert_filter_is_per_channel(Lanczos3, &wire, SW, SH, OW, OH, gamut, "lanczos3 up");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_filter_preserves_hdr_and_out_of_gamut_while_clamped_outputs_stay_in_range() {
  // A high-contrast edge of HDR (code 0xFFF, Y-only -> linear Y > 1 +
  // out-of-gamut negative R/B under Rec.709) against black drives the
  // signed-coefficient kernels (CatmullRom / Lanczos3) to overshoot. The
  // full-range float outputs (xyz_f32 / rgb_f32) must carry the overshoot
  // out of [0, 1] (full-range preserved, mirroring the area path's
  // HDR / out-of-gamut contract); the clamped outputs (rgb / rgb_u16 /
  // rgb_f16) must stay in range with no wrap.
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 6;
  const OH: usize = 6;
  let gamut = KernelGamut::Rec709;

  // Half-and-half edge: left columns max-Y (HDR + out-of-gamut), right
  // columns black. A reconstruction filter overshoots near the boundary.
  let mut wire = vec![0u16; SW * SH * 3];
  for sy in 0..SH {
    for sx in 0..SW {
      let i = (sy * SW + sx) * 3;
      if sx < SW / 2 {
        wire[i] = pack12_le(0x000);
        wire[i + 1] = pack12_le(0xFFF);
        wire[i + 2] = pack12_le(0x000);
      }
      // right half stays 0 (black)
    }
  }

  for &kernel_name in &["catmullrom", "lanczos3"] {
    let outs = match kernel_name {
      "catmullrom" => xyz12_filter_outputs(&wire, SW, SH, OW, OH, gamut, CatmullRom),
      _ => xyz12_filter_outputs(&wire, SW, SH, OW, OH, gamut, Lanczos3),
    };

    // Full-range float: the signed-coefficient overshoot must escape
    // [0, 1] in BOTH directions — `xyz_f32` keeps the HDR (> 1) Y, and
    // `rgb_f32` keeps both the HDR (> 1) overshoot and the out-of-gamut /
    // ringing undershoot (< 0). These are the unclamped contract.
    assert!(
      outs.xyz_f32.iter().any(|&v| v > 1.0),
      "{kernel_name}: filtered xyz_f32 lost the HDR (> 1) overshoot"
    );
    let any_over = outs.rgb_f32.iter().any(|&v| v > 1.0);
    let any_under = outs.rgb_f32.iter().any(|&v| v < 0.0);
    assert!(
      any_over,
      "{kernel_name}: filtered rgb_f32 lost the HDR (> 1) overshoot"
    );
    assert!(
      any_under,
      "{kernel_name}: filtered rgb_f32 lost the out-of-gamut / ringing undershoot (< 0)"
    );

    // Clamped outputs cannot wrap: each must equal the saturating narrow
    // of the SAME unclamped linear RGB the full-range `rgb_f32` carries
    // (narrow_unit_to_* / the f16 path clamp [0, 1]). So an overshoot
    // saturates UP and an undershoot saturates DOWN — neither wraps.
    for (i, &lin) in outs.rgb_f32.iter().enumerate() {
      let oetf = oetf_srgb(lin);
      assert_eq!(
        outs.rgb[i],
        narrow_unit_to_u8(oetf),
        "{kernel_name}: rgb[{i}] is not the [0,1]-clamped narrow of rgb_f32 (wrap?)"
      );
      assert_eq!(
        outs.rgb_u16[i],
        narrow_unit_to_u16(oetf),
        "{kernel_name}: rgb_u16[{i}] is not the [0,1]-clamped narrow of rgb_f32 (wrap?)"
      );
      assert_eq!(
        outs.rgb_f16[i].to_bits(),
        half::f16::from_f32(oetf.clamp(0.0, 1.0)).to_bits(),
        "{kernel_name}: rgb_f16[{i}] is not the [0,1]-clamped narrow of rgb_f32 (wrap?)"
      );
      let f = outs.rgb_f16[i].to_f32();
      assert!(
        (0.0..=1.0).contains(&f),
        "{kernel_name}: rgb_f16[{i}] = {f} escaped [0, 1]"
      );
    }

    // The two saturation directions actually occur: an overshoot pixel
    // pins the clamped u8 to 255 (saturated up, not wrapped to a small
    // value), and an undershoot pixel pins it to 0 (saturated down, not
    // wrapped to ~255).
    let over_i = outs
      .rgb_f32
      .iter()
      .position(|&v| oetf_srgb(v) > 1.0)
      .expect("an overshoot sample exists");
    assert_eq!(
      outs.rgb[over_i], 255,
      "{kernel_name}: overshoot did not saturate rgb to 255 (wrapped?)"
    );
    let under_i = outs
      .rgb_f32
      .iter()
      .position(|&v| oetf_srgb(v) < 0.0)
      .expect("an undershoot sample exists");
    assert_eq!(
      outs.rgb[under_i], 0,
      "{kernel_name}: undershoot did not saturate rgb to 0 (wrapped?)"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_filter_plan_is_accepted() {
  // Regression: a filter plan must no longer raise `UnsupportedFilter` at
  // the Xyz12 fence — the routed Filter arm runs the engine and populates
  // every attached output (the sentinel is gone).
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let wire = varying_frame(SW, SH);
  let src = Xyz12LeFrame::try_new(&wire, SW as u32, SH as u32, (SW * 3) as u32).unwrap();

  let sentinel = f32::from_bits(0x7FC0_1234); // a quiet-NaN sentinel
  let mut xyz_f32 = vec![sentinel; OW * OH * 3];
  {
    let mut sink = MixedSinker::<Xyz12Le, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_xyz_f32(&mut xyz_f32)
    .unwrap();
    xyz12_to(&src, KernelGamut::DciP3, &mut sink).unwrap();
  }
  assert!(
    xyz_f32.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate xyz_f32 (no UnsupportedFilter)"
  );
}
