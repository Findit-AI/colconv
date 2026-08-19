//! Fused-downscale coverage for the exotic 10-bit packed 4:2:2 YUV
//! source `V210` (12 x 10-bit samples per 16-byte word = 6 pixels, four
//! 32-bit LE words). The 4:2:2 sibling of the `Y210` / `Y212` / `Y216`
//! suite (`resample_y2xx.rs`) — same engine route, different wire
//! packing.
//!
//! V210 routes through [`packed_yuv422_triple_resample`] at `BITS = 10`,
//! the same shared 4:2:2 helper the `y2xx` family uses, with **three**
//! independent native-precision binnings — the u8 and u16 YUV→RGB
//! kernels round and scale *independently*, and luma is native Y:
//! - **u8 colour (rgb / rgba / hsv)** bins a converted source-width u8
//!   RGB row (`v210_to_rgb_row_endian`).
//! - **u16 colour (rgb_u16 / rgba_u16)** bins a converted source-width
//!   native u16 RGB row at native 10-bit depth
//!   (`v210_to_rgb_u16_row_endian`).
//! - **luma / luma_u16** bin the de-interleaved native Y
//!   (`v210_to_luma_u16_row_endian`); luma_u16 is the binned native Y,
//!   luma is `binned_Y >> (10 - 8)`.
//!
//! Each output is byte-identical to the area-bin of the **direct**
//! full-resolution conversion (convert-then-bin), so the oracles below
//! drive a direct identity V210 sink at source resolution and
//! 2x2-block-mean its output. The uniform-gray counterexample pins the
//! real parity bug: deriving u8 colour by narrowing the u16 bin would
//! change a uniform-gray downscale's colour, so the u8 group must bin its
//! own u8 conversion.
//!
//! `SRC = 12` is the smallest legal width (6-pixel words, even); `OUT = 6`
//! gives an exact 2x ratio so the area mean of each output pixel is the
//! round-half-up mean of a 2x2 source block.

use super::*;
use crate::{
  KernelMatrix,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
};

const SRC: usize = 12;
const OUT: usize = 6;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

// ---- V210 wire packing ------------------------------------------------

/// Pack 12 logical samples in V210 standard order
/// (`[Cb0, Y0, Cr0, Y1, Cb1, Y2, Cr1, Y3, Cb2, Y4, Cr2, Y5]`) into a
/// 16-byte word: three 10-bit samples per 32-bit LE word, top 2 bits
/// unused. Mirrors `pack_v210_word_for_test` in `tests/v210.rs`.
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

/// Pack per-pixel native Y (`SRC * SRC`) and per-chroma-pair native U / V
/// (`(SRC / 2) * SRC`, 4:2:2) into a `V210` LE byte plane. Width must be a
/// multiple of 6 (full words; `SRC = 12` → 2 words per row). The chroma
/// for word pixel pair (0,1) / (2,3) / (4,5) is taken from the
/// even-pixel chroma column, matching the V210 spec sample positions.
fn pack_v210(y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
  let cw = SRC / 2;
  let words_per_row = SRC / 6;
  let mut out = std::vec![0u8; words_per_row * 16 * SRC];
  for row in 0..SRC {
    for word in 0..words_per_row {
      let px = word * 6;
      let cu = px / 2;
      let samples: [u16; 12] = [
        u[row * cw + cu],
        y[row * SRC + px],
        v[row * cw + cu],
        y[row * SRC + px + 1],
        u[row * cw + cu + 1],
        y[row * SRC + px + 2],
        v[row * cw + cu + 1],
        y[row * SRC + px + 3],
        u[row * cw + cu + 2],
        y[row * SRC + px + 4],
        v[row * cw + cu + 2],
        y[row * SRC + px + 5],
      ];
      let off = (row * words_per_row + word) * 16;
      out[off..off + 16].copy_from_slice(&pack_v210_word(samples));
    }
  }
  out
}

const STRIDE: u32 = (SRC / 6 * 16) as u32;

/// Re-encode a V210 LE byte plane as **BE-encoded** byte storage by
/// byte-swapping each 32-bit word. The kernel's `load_endian_u32::<true>`
/// recovers the same 10-bit samples. Mirrors `v210_as_be` in
/// `tests/v210.rs`.
fn v210_as_be(plane_le: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(plane_le.len());
  for chunk in plane_le.chunks_exact(4) {
    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    out.extend_from_slice(&w.to_be_bytes());
  }
  out
}

// ---- Source-grid ramps ------------------------------------------------

/// Per-pixel `(Y, U, V)` ramp packed into a `V210` plane. Chroma is
/// sampled at one column per 2-pixel pair (4:2:2). Interior 10-bit codes
/// so every kernel sees real math.
fn ramp_packed() -> Vec<u8> {
  let cw = SRC / 2;
  let mut y = std::vec![0u16; SRC * SRC];
  let mut u = std::vec![0u16; cw * SRC];
  let mut v = std::vec![0u16; cw * SRC];
  for row in 0..SRC {
    for x in 0..SRC {
      y[row * SRC + x] = ((40u32 + (row * SRC + x) as u32 * 17) & 0x3FF) as u16;
    }
    for cx in 0..cw {
      u[row * cw + cx] = ((300u32 + cx as u32 * 53 + row as u32 * 11) & 0x3FF) as u16;
      v[row * cw + cx] = (0x3FFu32.wrapping_sub(cx as u32 * 41 + row as u32 * 7) & 0x3FF) as u16;
    }
  }
  pack_v210(&y, &u, &v)
}

/// Uniform-gray plane: constant Y, neutral chroma (U = V = 512). Binning
/// a uniform frame is identity, so every resampled colour output must
/// equal the direct full-res conversion.
fn uniform_gray_packed(y: u16) -> Vec<u8> {
  let cw = SRC / 2;
  let y_plane = std::vec![y & 0x3FF; SRC * SRC];
  let chroma = std::vec![512u16; cw * SRC];
  pack_v210(&y_plane, &chroma, &chroma)
}

/// Saturated-chroma plane: constant Y, extreme U/V — the case where
/// RGB-derived luma would clamp away from the Y plane.
fn saturated_packed(y: u16) -> Vec<u8> {
  let cw = SRC / 2;
  let y_plane = std::vec![y & 0x3FF; SRC * SRC];
  let u = std::vec![0x3FFu16; cw * SRC];
  let v = std::vec![0u16; cw * SRC];
  pack_v210(&y_plane, &u, &v)
}

fn frame(buf: &[u8]) -> V210Frame<'_> {
  V210Frame::new(buf, SRC as u32, SRC as u32, STRIDE)
}

// ---- Exact 2x2 block-area means (round-half-up) -----------------------

fn block_mean_2x2_rgb_u8(rgb: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u8;
      }
    }
  }
  out
}

fn block_mean_2x2_u16(plane: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

fn block_mean_2x2_rgb_u16(rgb: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u16;
      }
    }
  }
  out
}

// ---- Direct full-resolution oracles -----------------------------------

fn direct_rgb_u8(packed: &[u8]) -> Vec<u8> {
  let mut rgb = std::vec![0u8; SRC * SRC * 3];
  let mut sink = MixedSinker::<V210>::new(SRC, SRC)
    .with_rgb(&mut rgb)
    .unwrap();
  v210_to(&frame(packed), FR, sink.set_kernel_matrix(M)).unwrap();
  rgb
}

fn direct_rgb_u16(packed: &[u8]) -> Vec<u16> {
  let mut rgb = std::vec![0u16; SRC * SRC * 3];
  let mut sink = MixedSinker::<V210>::new(SRC, SRC)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  v210_to(&frame(packed), FR, sink.set_kernel_matrix(M)).unwrap();
  rgb
}

fn direct_luma_u16(packed: &[u8]) -> Vec<u16> {
  let mut y = std::vec![0u16; SRC * SRC];
  let mut sink = MixedSinker::<V210>::new(SRC, SRC)
    .with_luma_u16(&mut y)
    .unwrap();
  v210_to(&frame(packed), FR, sink.set_kernel_matrix(M)).unwrap();
  y
}

// ---- Per-output area-bin parity ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_u8_matches_area_bin_of_direct() {
  let packed = ramp_packed();
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgb, block_mean_2x2_rgb_u8(&direct_rgb_u8(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_u16_is_exact_native_area_mean() {
  let packed = ramp_packed();
  let mut rgb = std::vec![0u16; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb_u16(&mut rgb)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgb, block_mean_2x2_rgb_u16(&direct_rgb_u16(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_is_native_y_area_mean() {
  let packed = ramp_packed();
  let (mut luma, mut luma_u16) = (std::vec![0u8; OUT * OUT], std::vec![0u16; OUT * OUT]);
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let y_ref = block_mean_2x2_u16(&direct_luma_u16(&packed));
  assert_eq!(luma_u16, y_ref, "luma_u16 = native-Y area mean");
  // luma is the binned native Y narrowed `>> (10 - 8)`, matching a direct
  // conversion of the area-downscaled native frame.
  let luma_ref: Vec<u8> = y_ref.iter().map(|&y| (y >> 2) as u8).collect();
  assert_eq!(luma, luma_ref, "luma = binned native Y >> 2");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uniform_gray_color_unchanged_counterexample() {
  // The high-bit-YUV parity bug: deriving u8 colour by narrowing the u16
  // bin changes a uniform-gray downscale's colour. With a uniform-gray
  // frame, binning is identity, so every colour output must equal the
  // direct full-res conversion (also uniform).
  let packed = uniform_gray_packed(512);
  let direct_u8 = direct_rgb_u8(&packed);
  let direct_u16 = direct_rgb_u16(&packed);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  // Every direct full-res pixel is the same gray; the resampled pixels
  // must match it exactly (not a narrowed-u16 approximation).
  let g_u8 = &direct_u8[..3];
  for px in rgb.chunks_exact(3) {
    assert_eq!(px, g_u8, "uniform-gray rgb must equal the direct gray");
  }
  for px in rgba.chunks_exact(4) {
    assert_eq!(&px[..3], g_u8, "uniform-gray rgba colour");
    assert_eq!(px[3], 0xFF, "uniform-gray rgba alpha");
  }
  let g_u16 = &direct_u16[..3];
  for px in rgb_u16.chunks_exact(3) {
    assert_eq!(px, g_u16, "uniform-gray rgb_u16 must equal the direct gray");
  }
  // HSV of a uniform-gray frame: achromatic (H = 0, S = 0).
  assert!(hh.iter().all(|&h| h == 0), "uniform-gray hsv H");
  assert!(ss.iter().all(|&s| s == 0), "uniform-gray hsv S");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_from_native_y_under_saturated_chroma() {
  // Constant Y, extreme U/V: the area-downscaled Y is constant, so
  // luma-from-Y stays exactly Y. RGB-derived luma would clamp away.
  let y: u16 = 0x3FF / 4;
  let packed = saturated_packed(y);
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert!(
    luma_u16.iter().all(|&v| v == y),
    "luma_u16 must be native Y ({y}), not RGB-derived; got {luma_u16:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn all_outputs_combo() {
  // Every output attached: each must match its own oracle, proving the
  // three binnings (u8 colour, native u16 colour, native Y) coexist.
  let packed = ramp_packed();
  let rgb_u8_ref = block_mean_2x2_rgb_u8(&direct_rgb_u8(&packed));
  let rgb_u16_ref = block_mean_2x2_rgb_u16(&direct_rgb_u16(&packed));
  let y_ref = block_mean_2x2_u16(&direct_luma_u16(&packed));

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
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
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgb, rgb_u8_ref, "all-outputs rgb");
  for (px, rgb_px) in rgba.chunks_exact(4).zip(rgb_u8_ref.chunks_exact(3)) {
    assert_eq!(&px[..3], rgb_px, "all-outputs rgba colour");
    assert_eq!(px[3], 0xFF, "all-outputs rgba alpha");
  }
  assert_eq!(rgb_u16, rgb_u16_ref, "all-outputs rgb_u16");
  for (px, rgb_px) in rgba_u16.chunks_exact(4).zip(rgb_u16_ref.chunks_exact(3)) {
    assert_eq!(&px[..3], rgb_px, "all-outputs rgba_u16 colour");
    assert_eq!(px[3], 1023, "all-outputs rgba_u16 alpha");
  }
  assert_eq!(luma_u16, y_ref, "all-outputs luma_u16");
  let luma_ref: Vec<u8> = y_ref.iter().map(|&y| (y >> 2) as u8).collect();
  assert_eq!(luma, luma_ref, "all-outputs luma");
  // HSV from the binned u8 RGB.
  let mut hh_ref = std::vec![0u8; OUT * OUT];
  let mut ss_ref = std::vec![0u8; OUT * OUT];
  let mut vv_ref = std::vec![0u8; OUT * OUT];
  crate::row::rgb_to_hsv_row(
    &rgb_u8_ref,
    &mut hh_ref,
    &mut ss_ref,
    &mut vv_ref,
    OUT * OUT,
    false,
  );
  assert_eq!(hh, hh_ref, "all-outputs hsv H");
  assert_eq!(ss, ss_ref, "all-outputs hsv S");
  assert_eq!(vv, vv_ref, "all-outputs hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn with_rgb_equals_with_rgba() {
  // Strategy A: attaching both rgb and rgba yields identical colour bytes
  // (alpha = 0xFF), and the rgb plane equals the rgb-only oracle.
  let packed = ramp_packed();
  let rgb_ref = block_mean_2x2_rgb_u8(&direct_rgb_u8(&packed));
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgb, rgb_ref, "with_rgb plane");
  for (px, rgb_px) in rgba.chunks_exact(4).zip(rgb.chunks_exact(3)) {
    assert_eq!(&px[..3], rgb_px, "with_rgba colour matches with_rgb");
    assert_eq!(px[3], 0xFF, "with_rgba alpha");
  }
}

// ---- SIMD-vs-scalar parity across widths ------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn simd_matches_scalar_across_widths() {
  // Pseudo-random V210 across multiple 6-multiple widths (covering the
  // main loop + scalar tail of every backend block size) must bin
  // identically with SIMD on vs forced scalar. Mask each 32-bit word's
  // top 2 unused bits so the packed payload stays valid 10-bit fields.
  for sw in [6usize, 12, 18, 24, 30, 36, 48] {
    let oh = 2usize;
    let ow = sw / 2;
    let mut buf = std::vec![0u8; sw.div_ceil(6) * 16 * 4];
    pseudo_random_u8(&mut buf, 0xC0FFEE ^ sw as u32);
    for i in (0..buf.len()).step_by(4) {
      buf[i + 3] &= 0x3F;
    }
    let src = V210Frame::new(&buf, sw as u32, 4, (sw.div_ceil(6) * 16) as u32);

    let mut rgb_simd = std::vec![0u8; ow * oh * 3];
    let mut rgb_scalar = std::vec![0u8; ow * oh * 3];
    let mut luma16_simd = std::vec![0u16; ow * oh];
    let mut luma16_scalar = std::vec![0u16; ow * oh];
    {
      let mut sink = force_row_stage(
        MixedSinker::<V210, AreaResampler>::with_resampler(sw, 4, AreaResampler::to(ow, oh))
          .unwrap(),
      )
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_luma_u16(&mut luma16_simd)
      .unwrap();
      v210_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }
    {
      let mut sink = force_row_stage(
        MixedSinker::<V210, AreaResampler>::with_resampler(sw, 4, AreaResampler::to(ow, oh))
          .unwrap(),
      )
      .with_simd(false)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_luma_u16(&mut luma16_scalar)
      .unwrap();
      v210_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }
    assert_eq!(
      rgb_simd, rgb_scalar,
      "V210 resample rgb SIMD≠scalar at width {sw}"
    );
    assert_eq!(
      luma16_simd, luma16_scalar,
      "V210 resample luma_u16 SIMD≠scalar at width {sw}"
    );
  }
}

// ---- LE/BE parity -----------------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn le_be_outputs_identical() {
  // LE and BE wire encodings of the same logical plane must produce
  // identical resampled outputs: the binned row is host-native and the
  // derive kernels recover it with `HOST_NATIVE_BE`, so a wrong wire
  // const on either host shows up as an LE/BE divergence.
  let packed_le = ramp_packed();
  let packed_be = v210_as_be(&packed_le);

  let mut le_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut le_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut le_luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut le_rgb)
    .unwrap()
    .with_rgb_u16(&mut le_rgb_u16)
    .unwrap()
    .with_luma_u16(&mut le_luma_u16)
    .unwrap();
    v210_to(&frame(&packed_le), FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let mut be_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut be_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut be_luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let be_frame = V210BeFrame::try_new(&packed_be, SRC as u32, SRC as u32, STRIDE).unwrap();
    let mut sink = force_row_stage(
      MixedSinker::<V210<true>, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap(),
    )
    .with_rgb(&mut be_rgb)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap();
    v210_to_endian::<_, true>(&be_frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  assert_eq!(le_rgb, be_rgb, "rgb LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
}

// ---- Identity / no-op / lifecycle -------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_plan_matches_new_sink() {
  let packed = ramp_packed();
  let mut direct = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<V210>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let mut via_area = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap(),
    )
    .with_rgb(&mut via_area)
    .unwrap();
    v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn no_outputs_is_a_no_op() {
  let packed = ramp_packed();
  let mut sink = force_row_stage(
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  );
  v210_to(&frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  assert!(
    !sink.luma_stream_u16_allocated(),
    "no-output sink allocated a luma stream"
  );
  assert!(
    !sink.rgb_stream_allocated(),
    "no-output sink allocated an rgb stream"
  );
  assert!(
    !sink.rgb_stream_u16_allocated(),
    "no-output sink allocated a u16 rgb stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn resets_streams_across_frames() {
  // A reused sink must reset all three streams each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let p1 = ramp_packed();
  // Frame 2: invert each 10-bit sample by rebuilding from inverted planes.
  let cw = SRC / 2;
  let mut y = std::vec![0u16; SRC * SRC];
  let mut u = std::vec![0u16; cw * SRC];
  let mut v = std::vec![0u16; cw * SRC];
  for row in 0..SRC {
    for x in 0..SRC {
      y[row * SRC + x] = 0x3FF - (((40u32 + (row * SRC + x) as u32 * 17) & 0x3FF) as u16);
    }
    for cx in 0..cw {
      u[row * cw + cx] = 0x3FF - (((300u32 + cx as u32 * 53 + row as u32 * 11) & 0x3FF) as u16);
      v[row * cw + cx] =
        0x3FF - ((0x3FFu32.wrapping_sub(cx as u32 * 41 + row as u32 * 7) & 0x3FF) as u16);
    }
  }
  let p2 = pack_v210(&y, &u, &v);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    v210_to(&frame(&p1), FR, sink.set_kernel_matrix(M)).unwrap();
    v210_to(&frame(&p2), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  // Frame 2's outputs must reflect frame 2 (reset succeeded).
  assert_eq!(luma_u16, block_mean_2x2_u16(&direct_luma_u16(&p2)));
  assert_eq!(rgb_u16, block_mean_2x2_rgb_u16(&direct_rgb_u16(&p2)));
}

// ---- Sequence + freeze ordering ---------------------------------------

/// One V210 row slice (`div_ceil(6) * 16` bytes) at the given source row.
fn row_slice(packed: &[u8], idx: usize) -> &[u8] {
  let stride = STRIDE as usize;
  &packed[idx * stride..(idx + 1) * stride]
}

#[test]
fn out_of_sequence_first_row_rejected_before_allocation() {
  let packed = ramp_packed();
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb(&mut rgb)
  .unwrap()
  .with_rgb_u16(&mut rgb_u16)
  .unwrap()
  .with_luma_u16(&mut luma_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(V210Row::for_tests(row_slice(&packed, 3), 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.luma_stream_u16_allocated()
      && !sink.rgb_stream_allocated()
      && !sink.rgb_stream_u16_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(
    rgb.iter().all(|&b| b == 0)
      && rgb_u16.iter().all(|&b| b == 0)
      && luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated output"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rejects_mid_frame_out_of_sequence() {
  let packed = ramp_packed();
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_luma_u16(&mut luma_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V210Row::for_tests(row_slice(&packed, 0), 0, M, FR))
    .unwrap();
  let err = sink
    .process(V210Row::for_tests(row_slice(&packed, 2), 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rejects_mid_frame_output_change() {
  let packed = ramp_packed();
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V210Row::for_tests(row_slice(&packed, 0), 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(V210Row::for_tests(row_slice(&packed, 1), 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after reconfiguring the output set
  // succeeds instead of tripping ResampleOutputsChanged against a
  // snapshot the rejected row should never have committed.
  let packed = ramp_packed();
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut sink = force_row_stage(
    MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(V210Row::for_tests(row_slice(&packed, 3), 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  sink.set_luma_u16(&mut luma_u16).unwrap();
  sink
    .process(V210Row::for_tests(row_slice(&packed, 0), 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

// ---- Planar Yuv422p10 parity oracle -----------------------------------
//
// V210 is just a packed byte-stream of 4:2:2 10-bit planar data, so its
// resampled RGB must equal the area-bin of the **direct** Yuv422p10
// conversion of the same samples (convert-then-bin against the canonical
// planar reference, not just V210's own direct path). Gated `yuv-planar`
// (the planar source) so the v210-solo `--tests` build still compiles.
// `Yuv422p10` has no resample route yet, so the reference is its direct
// full-res RGB, area-binned here.

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_matches_area_bin_of_direct_yuv422p10() {
  let cw = SRC / 2;
  let mut y_plane = std::vec![0u16; SRC * SRC];
  let mut u_plane = std::vec![0u16; cw * SRC];
  let mut v_plane = std::vec![0u16; cw * SRC];
  pseudo_random_u16_low_n_bits(&mut y_plane, 0xC0FFEE, 10);
  pseudo_random_u16_low_n_bits(&mut u_plane, 0xBADF00D, 10);
  pseudo_random_u16_low_n_bits(&mut v_plane, 0xFEEDFACE, 10);

  let planar = Yuv422p10Frame::new(
    &y_plane, &u_plane, &v_plane, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let packed = pack_v210(&y_plane, &u_plane, &v_plane);
  let v210 = V210Frame::new(&packed, SRC as u32, SRC as u32, STRIDE);

  // Reference: direct full-res Yuv422p10 RGB, area-binned 2x2.
  let mut rgb_planar_direct = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv422p10>::new(SRC, SRC)
      .with_rgb(&mut rgb_planar_direct)
      .unwrap();
    yuv422p10_to(&planar, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let planar_ref = block_mean_2x2_rgb_u8(&rgb_planar_direct);

  // V210 resampled RGB for the same samples.
  let mut rgb_packed = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V210, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb_packed)
    .unwrap();
    v210_to(&v210, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(
    rgb_packed, planar_ref,
    "V210 resampled RGB must equal the area-bin of direct Yuv422p10"
  );
}
