//! Fused-downscale coverage for the packed-CIE-XYZ-12-bit family
//! ([`Xyz12`]): the wire row converts to source-width **linear XYZ**
//! `f32` (SMPTE ST 428-1 inverse-OETF only, pre-matrix), binning runs
//! in float so the area mean is taken in linear light, and every output
//! is derived from each finalized binned linear-XYZ row by the direct
//! DCP pipeline's math (matrix + gamma + clamp/scale + narrow).
//!
//! Two oracles:
//!
//! 1. **Bit-exact (constant blocks).** When every source pixel of an
//!    output block carries the same 12-bit triple, the area mean of the
//!    block's linear XYZ is that constant (exact in the f64-accumulated
//!    f32 stream), so every resampled output equals the *direct
//!    full-res* `Xyz12` conversion sampled at the block's top-left
//!    pixel — both run scalar, giving bit-exact parity for rgb / rgba /
//!    rgb_u16 / rgba_u16 / rgb_f32 / xyz_f32 / rgb_f16 / rgba_f16 /
//!    luma / luma_u16 / hsv. This is the strong oracle and exercises the
//!    matrix/gamma-commute-with-bin property directly.
//! 2. **Linear-light mean (varying within block, ±tolerance).** A frame
//!    with codes that vary inside each block: the resampled `xyz_f32`
//!    must match the f64 block-mean of the source linear XYZ (taken from
//!    a full-res `with_xyz_f32`), and `rgb_f32` must match the gamut
//!    matrix applied to that binned XYZ — proving the bin truly averages
//!    in linear light and that `M . mean(xyz) == mean(M . xyz)`.

use crate::{
  DcpTargetGamut,
  frame::Xyz12LeFrame,
  resample::AreaResampler,
  row::scalar::{xyz12::matmul3_xyz_rgb, xyz12_constants::xyz_to_rgb_matrix},
  sinker::MixedSinker,
  source::{Xyz12Le, xyz12_to},
};

// NOTE on the missing direct-row out-of-sequence / mid-frame-output-
// change / rejected-first-row-poison tests: those require injecting a
// hand-built `Xyz12Row` at a non-sequential index via
// `sink.process(...)` (the pattern the `resample_rgbf32` /
// `resample_rgb48` tests use). `Xyz12Row::new` is `pub(crate)` in the
// upstream `mediaframe` crate, so colconv cannot construct one — there
// is no public Row constructor or `From` impl. The sequence-check and
// the conditional-ordered frozen-output snapshot (the sink's resample
// route calls the shared compare-only `resample_preflight_check_only`,
// then commits the `frozen_outputs_check` freeze only after every stream
// / scratch allocation succeeds) are the *same* shared helpers exercised
// bit-for-bit by the `resample_rgbf32`
// `..._rejected_first_row_does_not_poison_output_retry` test, so the
// ordering logic is covered; only the xyz-specific driver to feed a bad
// order is inexpressible here. The commit-together atomicity IS covered
// directly below via the scratch-OOM failpoints (source-XYZ and the
// emit-owned narrow), which a walker can drive at row 0. `begin_frame`'s
// stream reset is covered by
// `xyz12_begin_frame_resets_stream_between_frames`.

const SRC: usize = 8;
const OUT: usize = 4;

/// f32 tolerance for the binned-value comparisons. The f32 area stream
/// accumulates in f64 and the linear-light block-mean is computed in
/// f64 here, so the only divergence is the final f32 cast and any
/// reorder of the H/V reduce — comfortably under 1e-4.
const TOL: f32 = 1e-4;

/// Encodes a 12-bit code into the high-bit-packed LE wire `u16`
/// (`code << 4`, then LE bytes reinterpreted as host-native). Per
/// FFmpeg `AV_PIX_FMT_XYZ12LE`: active 12 bits in `[15:4]`, low 4 bits
/// zero. Host-independent (same logical wire value on LE and BE hosts).
fn pack12_le(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_le_bytes())
}

/// Builds a frame in which every source pixel of each `2x2` output block
/// carries a constant `(X, Y, Z)` 12-bit triple that varies per block.
/// The triples deliberately include `y`-only-max blocks (which drive
/// out-of-gamut negative R/B and HDR > 1 in linear RGB) and a mid /
/// max-white block. Returns the packed wire plane (`SRC * SRC * 3`
/// `u16`, stride `3 * SRC`).
fn constant_block_frame() -> Vec<u16> {
  // Per-block 12-bit (X, Y, Z) triples for the OUT x OUT = 4x4 blocks.
  // Index by [oy][ox]; chosen to span the active range and the
  // out-of-gamut / HDR corners.
  let blocks: [[(u16, u16, u16); OUT]; OUT] = [
    [
      (0x000, 0x000, 0x000),
      (0x800, 0x800, 0x800),
      (0xFFF, 0xFFF, 0xFFF),
      (0x000, 0xFFF, 0x000),
    ],
    [
      (0x123, 0x456, 0x789),
      (0xABC, 0x0DE, 0x0F0),
      (0x400, 0x200, 0x600),
      (0xFFF, 0x000, 0x000),
    ],
    [
      (0x000, 0x000, 0xFFF),
      (0x333, 0x999, 0x222),
      (0x700, 0x800, 0x900),
      (0x0F0, 0x0FF, 0x00F),
    ],
    [
      (0xFFF, 0x800, 0x400),
      (0x100, 0x900, 0x100),
      (0x555, 0x555, 0x555),
      (0xC00, 0x300, 0xE00),
    ],
  ];
  let mut buf = vec![0u16; SRC * SRC * 3];
  for sy in 0..SRC {
    for sx in 0..SRC {
      let (x, y, z) = blocks[sy / 2][sx / 2];
      let i = (sy * SRC + sx) * 3;
      buf[i] = pack12_le(x);
      buf[i + 1] = pack12_le(y);
      buf[i + 2] = pack12_le(z);
    }
  }
  buf
}

/// Builds a frame whose 12-bit codes vary within every block (so the
/// area mean is a genuine average, not a single representative pixel).
fn varying_frame() -> Vec<u16> {
  let mut buf = vec![0u16; SRC * SRC * 3];
  for (i, c) in buf.iter_mut().enumerate() {
    // A per-element ramp through the active 12-bit range; neighbouring
    // pixels (hence within-block pixels) differ.
    let code = ((i * 137) % 4096) as u16;
    *c = pack12_le(code);
  }
  buf
}

/// Full-res `with_xyz_f32` of a wire frame — the production source of
/// truth for linear XYZ (SMPTE ST 428-1 inverse-OETF, no matrix).
fn full_res_linear_xyz(wire: &[u16], gamut: DcpTargetGamut) -> Vec<f32> {
  let src = Xyz12LeFrame::try_new(wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let mut xyz = vec![0.0f32; SRC * SRC * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(SRC, SRC)
    .with_simd(false)
    .with_xyz_f32(&mut xyz)
    .unwrap();
  xyz12_to(&src, gamut, &mut sink).unwrap();
  xyz
}

/// Exact f64 `2x2` block mean of a full-res linear-XYZ plane.
fn block_mean_linear_xyz(src_xyz: &[f32], ox: usize, oy: usize, c: usize) -> f64 {
  let mut acc = 0.0f64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += src_xyz[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as f64;
    }
  }
  acc / 4.0
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_constant_block_outputs_match_direct_full_res() {
  // The strong oracle: every resampled output of a constant-block frame
  // equals the direct full-res Xyz12 conversion sampled at the block's
  // top-left pixel (both scalar -> bit-exact). Covers rgb, rgba,
  // rgb_u16, rgba_u16, rgb_f32, xyz_f32, rgb_f16, rgba_f16, luma,
  // luma_u16, hsv.
  for &gamut in &[
    DcpTargetGamut::DciP3,
    DcpTargetGamut::Rec709,
    DcpTargetGamut::Rec2020,
  ] {
    let wire = constant_block_frame();
    let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

    // Resampled (scalar) outputs at OUT x OUT.
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let mut rgba = vec![0u8; OUT * OUT * 4];
    let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
    let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
    let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
    let mut rgb_f16 = vec![half::f16::ZERO; OUT * OUT * 3];
    let mut rgba_f16 = vec![half::f16::ZERO; OUT * OUT * 4];
    let mut luma = vec![0u8; OUT * OUT];
    let mut luma_u16 = vec![0u16; OUT * OUT];
    let mut hh = vec![0u8; OUT * OUT];
    let mut ss = vec![0u8; OUT * OUT];
    let mut vv = vec![0u8; OUT * OUT];
    {
      let mut sink = MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_simd(false)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap()
      .with_xyz_f32(&mut xyz_f32)
      .unwrap()
      .with_rgb_f16(&mut rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut rgba_f16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
      xyz12_to(&src, gamut, &mut sink).unwrap();
    }

    // Direct full-res (scalar) outputs at SRC x SRC.
    let mut d_rgb = vec![0u8; SRC * SRC * 3];
    let mut d_rgba = vec![0u8; SRC * SRC * 4];
    let mut d_rgb_u16 = vec![0u16; SRC * SRC * 3];
    let mut d_rgba_u16 = vec![0u16; SRC * SRC * 4];
    let mut d_rgb_f32 = vec![0.0f32; SRC * SRC * 3];
    let mut d_xyz_f32 = vec![0.0f32; SRC * SRC * 3];
    let mut d_rgb_f16 = vec![half::f16::ZERO; SRC * SRC * 3];
    let mut d_rgba_f16 = vec![half::f16::ZERO; SRC * SRC * 4];
    let mut d_luma = vec![0u8; SRC * SRC];
    let mut d_luma_u16 = vec![0u16; SRC * SRC];
    let mut d_h = vec![0u8; SRC * SRC];
    let mut d_s = vec![0u8; SRC * SRC];
    let mut d_v = vec![0u8; SRC * SRC];
    {
      let mut sink = MixedSinker::<Xyz12Le>::new(SRC, SRC)
        .with_simd(false)
        .with_rgb(&mut d_rgb)
        .unwrap()
        .with_rgba(&mut d_rgba)
        .unwrap()
        .with_rgb_u16(&mut d_rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut d_rgba_u16)
        .unwrap()
        .with_rgb_f32(&mut d_rgb_f32)
        .unwrap()
        .with_xyz_f32(&mut d_xyz_f32)
        .unwrap()
        .with_rgb_f16(&mut d_rgb_f16)
        .unwrap()
        .with_rgba_f16(&mut d_rgba_f16)
        .unwrap()
        .with_luma(&mut d_luma)
        .unwrap()
        .with_luma_u16(&mut d_luma_u16)
        .unwrap()
        .with_hsv(&mut d_h, &mut d_s, &mut d_v)
        .unwrap();
      xyz12_to(&src, gamut, &mut sink).unwrap();
    }

    for oy in 0..OUT {
      for ox in 0..OUT {
        // Top-left source pixel of this block.
        let sp = (oy * 2) * SRC + ox * 2;
        let op = oy * OUT + ox;
        for c in 0..3 {
          assert_eq!(
            rgb[op * 3 + c],
            d_rgb[sp * 3 + c],
            "rgb ({ox},{oy}) c{c} gamut={gamut:?}"
          );
          assert_eq!(
            rgb_u16[op * 3 + c],
            d_rgb_u16[sp * 3 + c],
            "rgb_u16 ({ox},{oy}) c{c}"
          );
          assert_eq!(
            rgb_f32[op * 3 + c].to_bits(),
            d_rgb_f32[sp * 3 + c].to_bits(),
            "rgb_f32 ({ox},{oy}) c{c} gamut={gamut:?}"
          );
          assert_eq!(
            xyz_f32[op * 3 + c].to_bits(),
            d_xyz_f32[sp * 3 + c].to_bits(),
            "xyz_f32 ({ox},{oy}) c{c} gamut={gamut:?}"
          );
          assert_eq!(
            rgb_f16[op * 3 + c].to_bits(),
            d_rgb_f16[sp * 3 + c].to_bits(),
            "rgb_f16 ({ox},{oy}) c{c}"
          );
        }
        for c in 0..4 {
          assert_eq!(
            rgba[op * 4 + c],
            d_rgba[sp * 4 + c],
            "rgba ({ox},{oy}) c{c}"
          );
          assert_eq!(
            rgba_u16[op * 4 + c],
            d_rgba_u16[sp * 4 + c],
            "rgba_u16 ({ox},{oy}) c{c}"
          );
          assert_eq!(
            rgba_f16[op * 4 + c].to_bits(),
            d_rgba_f16[sp * 4 + c].to_bits(),
            "rgba_f16 ({ox},{oy}) c{c}"
          );
        }
        assert_eq!(luma[op], d_luma[sp], "luma ({ox},{oy}) gamut={gamut:?}");
        assert_eq!(luma_u16[op], d_luma_u16[sp], "luma_u16 ({ox},{oy})");
        assert_eq!(hh[op], d_h[sp], "hsv.h ({ox},{oy})");
        assert_eq!(ss[op], d_s[sp], "hsv.s ({ox},{oy})");
        assert_eq!(vv[op], d_v[sp], "hsv.v ({ox},{oy})");
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_downscale_xyz_f32_is_linear_light_area_mean() {
  // Varying-within-block frame: the binned xyz_f32 must equal the f64
  // block-mean of the source linear XYZ (the linear-light mean),
  // proving the bin actually averages in linear light.
  let gamut = DcpTargetGamut::Rec709;
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let src_xyz = full_res_linear_xyz(&wire, gamut);

  let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(false)
        .with_xyz_f32(&mut xyz_f32)
        .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let got = xyz_f32[(oy * OUT + ox) * 3 + c];
        let want = block_mean_linear_xyz(&src_xyz, ox, oy, c) as f32;
        assert!(
          (got - want).abs() <= TOL,
          "xyz_f32 ({ox},{oy}) c{c}: {got} vs {want}"
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
fn xyz12_downscale_rgb_f32_is_matrix_of_binned_xyz() {
  // rgb_f32 of the binned (linear-light averaged) XYZ must equal the
  // gamut matrix applied to the linear-light block-mean — i.e. the
  // matrix commutes with the bin (`M . mean(xyz) == mean(M . xyz)`).
  let gamut = DcpTargetGamut::DciP3;
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let src_xyz = full_res_linear_xyz(&wire, gamut);
  let m = xyz_to_rgb_matrix(gamut);

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(false)
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let bx = block_mean_linear_xyz(&src_xyz, ox, oy, 0) as f32;
      let by = block_mean_linear_xyz(&src_xyz, ox, oy, 1) as f32;
      let bz = block_mean_linear_xyz(&src_xyz, ox, oy, 2) as f32;
      let want = matmul3_xyz_rgb(&m, [bx, by, bz]);
      for c in 0..3 {
        let got = rgb_f32[(oy * OUT + ox) * 3 + c];
        assert!(
          (got - want[c]).abs() <= TOL,
          "rgb_f32 ({ox},{oy}) c{c}: {got} vs {}",
          want[c]
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
fn xyz12_downscale_preserves_hdr_and_out_of_gamut() {
  // The y-only-max constant block (code (0, 0xFFF, 0)) under Rec.709
  // produces linear XYZ Y ~= 1.09 (HDR > 1 in the Y channel) and, after
  // the matrix, out-of-gamut negative R/B in linear RGB. Both must
  // survive the bin into xyz_f32 / rgb_f32 (no clamp on the float
  // paths).
  let gamut = DcpTargetGamut::Rec709;
  let wire = constant_block_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(false)
        .with_xyz_f32(&mut xyz_f32)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  // Block (ox=3, oy=0) is the y-only-max block.
  let op = 3;
  assert!(
    xyz_f32[op * 3 + 1] > 1.0,
    "y-only-max binned Y should be HDR > 1, got {}",
    xyz_f32[op * 3 + 1]
  );
  assert!(
    rgb_f32.iter().any(|&v| v < 0.0),
    "binned rgb_f32 clamped away out-of-gamut negatives"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_simd_downscale_matches_scalar_on_constant_blocks() {
  // Constant blocks make the f32 area reduce exact even under SIMD
  // (summing identical values is exact), so the SIMD-binned outputs are
  // bit-identical to the scalar-binned outputs. Guards the SIMD area
  // reduce dispatch on the linear-XYZ path.
  let gamut = DcpTargetGamut::DciP3;
  let wire = constant_block_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let run = |simd: bool| {
    let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_xyz_f32(&mut xyz_f32)
        .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
    (rgb_u16, xyz_f32)
  };
  let (s_rgb_u16, s_xyz) = run(false);
  let (v_rgb_u16, v_xyz) = run(true);
  assert_eq!(s_rgb_u16, v_rgb_u16, "rgb_u16 scalar vs simd");
  for (a, b) in s_xyz.iter().zip(v_xyz.iter()) {
    assert_eq!(a.to_bits(), b.to_bits(), "xyz_f32 scalar vs simd");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_identity_plan_matches_new_sink() {
  // An identity (out == src) area plan must reproduce the direct sink
  // byte-for-byte on every output.
  let gamut = DcpTargetGamut::Rec709;
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut d_rgb = vec![0u8; SRC * SRC * 3];
  let mut d_xyz = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Xyz12Le>::new(SRC, SRC)
      .with_rgb(&mut d_rgb)
      .unwrap()
      .with_xyz_f32(&mut d_xyz)
      .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  let mut a_rgb = vec![0u8; SRC * SRC * 3];
  let mut a_xyz = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut a_rgb)
        .unwrap()
        .with_xyz_f32(&mut a_xyz)
        .unwrap();
    xyz12_to(&src, gamut, &mut sink).unwrap();
  }
  assert_eq!(d_rgb, a_rgb, "rgb identity");
  for (a, b) in d_xyz.iter().zip(a_xyz.iter()) {
    assert_eq!(a.to_bits(), b.to_bits(), "xyz_f32 identity");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_no_output_sink_is_a_noop() {
  // A resampled sink with no attached output neither allocates the
  // stream nor grows the staging scratch — the documented legal no-op.
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut sink =
    MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();
  assert!(
    !sink.xyz_stream_f32_allocated(),
    "no-output sink allocated the linear-XYZ stream"
  );
  assert_eq!(
    sink.xyz_scratch_f32_capacity(),
    0,
    "no-output sink grew the linear-XYZ scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_f32_only_downscale_does_not_size_the_narrow_scratch() {
  // An f32-only sink (xyz_f32 + rgb_f32) derives every output directly
  // from the binned float row, so the u8 narrow scratch is never sized.
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_xyz_f32(&mut xyz_f32)
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "f32-only sink grew the u8 narrow scratch"
  );

  // Positive control: attaching a u8 output sizes the narrow scratch.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink2).unwrap();
  assert!(
    sink2.rgb_scratch_capacity() >= OUT * 3,
    "u8 output did not size the narrow scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xyz12_begin_frame_resets_stream_between_frames() {
  // Two consecutive frames through the same sink must each produce the
  // correct downscaled result — begin_frame resets the area stream and
  // the frozen-output snapshot so the second frame is independent.
  let gamut = DcpTargetGamut::Rec709;
  let wire_a = constant_block_frame();
  let wire_b = varying_frame();
  let src_a = Xyz12LeFrame::try_new(&wire_a, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let src_b = Xyz12LeFrame::try_new(&wire_b, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut xyz_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_simd(false)
      .with_xyz_f32(&mut xyz_f32)
      .unwrap();
  xyz12_to(&src_a, gamut, &mut sink).unwrap();
  // Second frame reuses the sink; begin_frame inside xyz12_to resets.
  xyz12_to(&src_b, gamut, &mut sink).unwrap();

  // Result must equal frame B's linear-light block mean (not a blend
  // with frame A's accumulator).
  let src_xyz_b = full_res_linear_xyz(&wire_b, gamut);
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let got = xyz_f32[(oy * OUT + ox) * 3 + c];
        let want = block_mean_linear_xyz(&src_xyz_b, ox, oy, c) as f32;
        assert!(
          (got - want).abs() <= TOL,
          "frame B xyz_f32 ({ox},{oy}) c{c}: {got} vs {want}"
        );
      }
    }
  }
}

// RFC #238 tail-collapse atomicity: the source-width linear-XYZ `f32` staging
// scratch is grown AFTER the area stream builds into a LOCAL and BEFORE the field
// insert + freeze, so a scratch OOM leaves the stream field `None` AND
// `resample_outputs` uncommitted — a retryable first row. `xyz_f32` alone does not
// stage through the emit's narrow scratch, so the source-XYZ grow is the first
// fallible step after the stream build.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn first_row_source_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let gamut = DcpTargetGamut::Rec709;
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let mut xyz = std::vec![0f32; OUT * OUT * 3];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_xyz_f32(&mut xyz)
        .unwrap();
    // Arm the single-shot source-linear-XYZ scratch grow failpoint: the row-0 area
    // stream box builds OK (into a LOCAL) and the source-XYZ staging grow — before the
    // field insert + freeze — refuses, surfacing AllocationFailed with BOTH the stream
    // field `None` and the freeze uncommitted (the commit-together atomic shape).
    crate::sinker::mixed::arm_source_xyz_f32_scratch_failure();
    let err = xyz12_to(&src, gamut, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        crate::sinker::MixedSinkerError::Resample(
          crate::resample::ResampleError::AllocationFailed(_)
        )
      ),
      "first-row source-XYZ scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the area stream field must
    // be `None` — the fix builds it into a LOCAL and inserts only after the scratch
    // grows. The insert-before-scratch shape would leave it `Some` here, so this
    // assertion FAILS against that shape and PASSES against the fix.
    assert!(
      !sink.xyz_stream_f32_allocated(),
      "a scratch OOM must leave the xyz_stream_f32 field None (no partial commit)"
    );
    // DIRECT freeze-rollback proof (same frame, BEFORE any begin_frame reset): the
    // output-set snapshot must NOT be committed. A freeze-before-scratch regression
    // would commit it here even though the stream insertion is still delayed — this
    // catches the freeze half that the walker retry below (which `begin_frame`-resets
    // the field) cannot.
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    // Extra recoverability coverage — attaching rgb (a CHANGED output set) and
    // re-walking (begin_frame re-sequences from row 0) is ACCEPTED and drives output.
    sink.set_rgb(&mut rgb).unwrap();
    xyz12_to(&src, gamut, &mut sink)
      .expect("frame replay after a first-row source-XYZ scratch OOM must succeed");
  }
  assert!(
    xyz.iter().any(|&v| v != 0.0),
    "the recovered frame produced real xyz_f32 output (scratch OOM was recoverable)"
  );
}

// RFC #238 tail-collapse atomicity: the emit's OWN out-width narrow (u8 RGB) scratch
// is PRE-GROWN before the freeze (the same predicate `xyz12_resample_emit` uses), so
// a narrow-scratch OOM — after the stream build AND the source-XYZ grow both
// succeed — still leaves the stream field `None` AND `resample_outputs` uncommitted.
// `rgb` stages through that narrow scratch, so its grow is reachable.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn first_row_emit_narrow_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let gamut = DcpTargetGamut::Rec709;
  let wire = varying_frame();
  let src = Xyz12LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut xyz = std::vec![0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Xyz12Le, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    // Arm the source-RGB (emit narrow) scratch grow failpoint: the row-0 stream box
    // builds OK, the source-XYZ staging grows OK, and the emit-owned narrow scratch —
    // pre-grown BEFORE the field insert + freeze — refuses. Proves the emit-owned
    // scratch is pre-grown BEFORE the commit, so nothing fallible runs post-freeze.
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = xyz12_to(&src, gamut, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        crate::sinker::MixedSinkerError::Resample(
          crate::resample::ResampleError::AllocationFailed(_)
        )
      ),
      "first-row emit-narrow scratch OOM must surface AllocationFailed, got {err:?}"
    );
    assert!(
      !sink.xyz_stream_f32_allocated(),
      "an emit-owned scratch OOM must leave the xyz_stream_f32 field None (no partial commit)"
    );
    // DIRECT freeze-rollback proof (same frame, BEFORE any begin_frame reset): the
    // emit-owned scratch is pre-grown BEFORE the freeze, so its OOM must leave the
    // output-set snapshot uncommitted. A freeze-before-scratch regression would commit
    // it here — this catches the freeze half the walker retry below cannot.
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    // Extra recoverability coverage — attaching xyz_f32 (a CHANGED output set) and
    // re-walking is ACCEPTED and drives output.
    sink.set_xyz_f32(&mut xyz).unwrap();
    xyz12_to(&src, gamut, &mut sink)
      .expect("frame replay after a first-row emit-narrow scratch OOM must succeed");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (scratch OOM was recoverable)"
  );
}
