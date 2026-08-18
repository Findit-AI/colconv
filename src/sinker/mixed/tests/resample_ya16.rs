//! Alpha-aware fused-downscale coverage for the packed 16-bit gray+alpha
//! source (`Ya16`, LE + BE).
//!
//! `Ya16` decodes each packed `[Y, A]` u16 row into the canonical host-
//! native source-width `R, G, B, A` u16 row with `R = G = B = Y`
//! (`ya16_to_rgba_u16_row::<BE>`) and feeds the **same** 4-channel high-bit
//! packed-RGBA resample tail (`packed_rgba_u16_resample::<16, …>`) the
//! `Rgba64` / `Bgra64` / `Gbrap16` sources take — so binning the direct
//! RGBA yields `(binY, binY, binY, binA)`, byte-identical to binning Y once
//! and duplicating. This suite asserts the alpha contract at native 16-bit
//! depth:
//! - native `rgba_u16` is the exact 2x2 native block mean (alpha averaged,
//!   not forced opaque);
//! - the u8 / narrowed outputs derive from a single `>> 8` narrowing of the
//!   straight color;
//! - premultiplied bins premultiplied color and un-premultiplies against
//!   `0xFFFF` (transparent pixels never bleed);
//! - **native-Y luma**: `luma_u16` is the native Y pass-through and `luma`
//!   (u8) is `Y >> 8`, taken from an INDEPENDENT native-Y area bin (the Y
//!   plane fed through its own single-channel u16 stream), NEVER from the
//!   alpha- or range-affected color. This is byte-exact to the direct
//!   `ya16_to_luma*` kernels for every matrix, every range, AND every alpha
//!   mode — NOT `rgb_to_luma*`, which for the SMPTE-240M matrix (Q15 weights
//!   summing to 32769, not 32768) deviates by up to 2 LSB, and NOT the
//!   color path, which under premultiplied collapses to `mean(Y*A)/mean(A)`.
//!   `smpte240m_native_luma_is_y_not_matrix_derived` pins the matrix
//!   divergence; the `limited_range_*` tests pin the range divergence; and
//!   `premultiplied_nonuniform_alpha_*` pins the alpha divergence (incl. the
//!   `(0,65535),(65535,0),... -> 32768` case);
//! - LE and BE wire encodings produce byte-identical output;
//! - the alpha mode is frozen per frame, re-armed across frames, and a
//!   mid-frame flip / out-of-sequence row is rejected (driven directly via
//!   the publicly-constructible `Ya16Row`).

use crate::{
  KernelMatrix, PixelSink,
  frame::{Ya16BeFrame, Ya16Frame},
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Ya16, Ya16Row, ya16_to, ya16_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
/// Limited (studio) range — exercises the native-Y luma path against the
/// range-dependent `rgb_to_luma*` it must NOT use.
const FR_LIMITED: bool = false;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native u16 slice as LE-encoded wire byte storage (the
/// `ya16le` plane contract); recovered via `u16::from_le`.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded wire byte storage (the
/// `ya16be` plane contract); recovered via `u16::from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Pseudo-random host-native packed `[Y, A]` u16 plane (`SRC * SRC * 2`
/// elements); alpha varies (not all-opaque).
fn host_packed(seed: u32) -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 2];
  super::pseudo_random_u16_low_n_bits(&mut buf, seed, 16);
  buf
}

/// Canonical host-native `R, G, B, A` u16 of one host packed `[Y, A]`
/// plane: `R = G = B = Y`, `A` passed through. Only the `rgb`-gated
/// fractional-ratio oracle (against the packed-`Rgba64` source) consumes
/// it.
#[cfg(feature = "rgb")]
fn canonical_from_packed(packed: &[u16], n: usize) -> Vec<u16> {
  let mut out = std::vec![0u16; n * 4];
  for i in 0..n {
    let y = packed[i * 2];
    let a = packed[i * 2 + 1];
    out[i * 4] = y;
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = y;
    out[i * 4 + 3] = a;
  }
  out
}

/// Round-half-up 2x2 block mean of a canonical RGBA u16 plane (every
/// channel, alpha included), computed at u16 precision.
fn block_mean_rgba(src: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u64;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u64;
          }
        }
        out[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u16;
      }
    }
  }
  out
}

/// Round-half-up 2x2 block mean of the **native Y plane** of a host-packed
/// `[Y, A]` u16 source (`Y = packed[2*i]`), at u16 precision — the
/// alpha-independent native-Y area-downscale oracle. This is `mean(Y)`,
/// NOT the color path's `mean(Y*A)/mean(A)`, so it is the correct `luma`
/// source under every alpha mode.
fn block_mean_native_y(packed: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u64;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += packed[((oy * 2 + dy) * SRC + ox * 2 + dx) * 2] as u64;
        }
      }
      out[oy * OUT + ox] = ((acc + 2) / 4) as u16;
    }
  }
  out
}

/// Premultiply one canonical RGBA u16 plane in place against `0xFFFF`.
fn premultiply(plane: &mut [u16]) {
  const MAX: u32 = 0xFFFF;
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + MAX / 2) / MAX) as u16;
    }
  }
}

/// Un-premultiply one binned canonical RGBA u16 plane against `0xFFFF`.
fn unpremultiply(plane: &[u16]) -> Vec<u16> {
  const MAX: u32 = 0xFFFF;
  let mut out = std::vec![0u16; plane.len()];
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * MAX + a / 2)
        .checked_div(a)
        .map_or(0, |q| q.min(MAX)) as u16;
    }
    o[3] = i[3];
  }
  out
}

/// Drop alpha from a canonical RGBA u16 plane → packed RGB u16.
fn drop_alpha(rgba: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Narrow a canonical RGBA u16 plane to u8 via `>> 8`.
fn narrow_rgba_u8(rgba: &[u16]) -> Vec<u8> {
  rgba.iter().map(|&v| (v >> 8) as u8).collect()
}

/// Narrow an RGB u16 plane to u8 via `>> 8`.
fn narrow_rgb_u8(rgb: &[u16]) -> Vec<u8> {
  rgb.iter().map(|&v| (v >> 8) as u8).collect()
}

/// Full-resolution canonical RGBA u16 of the source — a direct (identity)
/// `Ya16` conversion over an LE-wire frame. The oracles bin / premultiply
/// this.
fn direct_rgba_u16(host_packed: &[u16]) -> Vec<u16> {
  let wire = as_le_u16(host_packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);
  let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
  let mut sink = MixedSinker::<Ya16>::new(SRC, SRC)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  rgba_u16
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_rgba_u16_is_block_mean_of_direct() {
  let packed = host_packed(0x51A1);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_alpha_is_averaged_not_forced_opaque() {
  let mut packed = host_packed(0x9E37);
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[1] = (i as u32 * 919) as u16;
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgba_u16,
    block_mean_rgba(&direct_rgba_u16(&packed)),
    "block mean"
  );
  assert!(
    rgba_u16.chunks_exact(4).any(|px| px[3] != 0xFFFF),
    "resampled alpha was forced opaque — area-mean alpha lost"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_all_outputs_derive_from_binned_color() {
  let packed = host_packed(0xBEEF);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_luma_u16(&mut lu16)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut v)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba_u16(&packed));
  assert_eq!(rgba_u16, binned, "rgba_u16 == native block mean");
  assert_eq!(
    rgb_u16,
    drop_alpha(&binned),
    "rgb_u16 == drop-alpha(binned)"
  );
  assert_eq!(rgba, narrow_rgba_u8(&binned), "rgba == narrowed binned");
  assert_eq!(
    rgb,
    narrow_rgb_u8(&drop_alpha(&binned)),
    "rgb == narrowed drop-alpha(binned)"
  );

  // luma / luma_u16: native Y. The binned color's R channel IS the binned
  // Y (R = Y at every source pixel), so a direct `Ya16` conversion of the
  // binned Y is the byte-exact reference: luma_u16 native pass-through,
  // luma `>> 8`. hsv is the achromatic V = Y>>8.
  let binned_rgb = drop_alpha(&binned);
  let mut binned_packed = std::vec![0u16; OUT * OUT * 2];
  for i in 0..OUT * OUT {
    binned_packed[i * 2] = binned_rgb[i * 3]; // R == Y
    binned_packed[i * 2 + 1] = 0xFFFF;
  }
  let binned_wire = as_le_u16(&binned_packed);
  let binned_frame = Ya16Frame::new(&binned_wire, OUT as u32, OUT as u32, (OUT * 2) as u32);
  let mut luma_ref = std::vec![0u8; OUT * OUT];
  let mut lu16_ref = std::vec![0u16; OUT * OUT];
  let mut h_ref = std::vec![0u8; OUT * OUT];
  let mut s_ref = std::vec![0u8; OUT * OUT];
  let mut v_ref = std::vec![0u8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Ya16>::new(OUT, OUT)
      .with_luma(&mut luma_ref)
      .unwrap()
      .with_luma_u16(&mut lu16_ref)
      .unwrap()
      .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
      .unwrap();
    ya16_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, luma_ref, "luma (Y >> 8)");
  assert_eq!(lu16, lu16_ref, "luma_u16 (native Y pass-through)");
  assert_eq!(h, h_ref, "hsv H");
  assert_eq!(s, s_ref, "hsv S");
  assert_eq!(v, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn smpte240m_native_luma_is_y_not_matrix_derived() {
  // The SMPTE-240M Q15 luma weights sum to 32769 (not 32768), so deriving
  // the native u16 luma from R=G=B=Y via the matrix path
  // (`rgb_to_luma_u16_native_row`) deviates from native Y by up to 2 LSB at
  // Y≈49152. The direct `Ya16` luma_u16 is the native Y pass-through, so the
  // resampled luma_u16 must equal the binned-Y R channel — NOT the matrix
  // derivation. Pick a uniform-per-block Y near the worst-case so the
  // matrix path would visibly diverge.
  const SM: KernelMatrix = KernelMatrix::Smpte240m;
  let mut packed = std::vec![0u16; SRC * SRC * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    // Y constant within each 2x2 block so the block mean is exactly Y; the
    // matrix path's per-pixel rounding then shows against native Y.
    let block = (i % SRC) / 2 + (i / (SRC * 2)) * (SRC / 2);
    px[0] = 49000u16.wrapping_add((block as u16) * 37); // near worst case
    px[1] = 0xFFFF; // opaque
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya16_to(&frame, FR, SM, &mut sink).unwrap();
  }

  // Native-Y reference: the area-mean of the native Y plane (exact,
  // matrix- and alpha-independent).
  let native_y = block_mean_native_y(&packed);
  assert_eq!(
    lu16, native_y,
    "luma_u16 must be native Y, not matrix-derived"
  );

  // And confirm the matrix path WOULD differ here (guards the test itself):
  // `rgb_to_luma_u16_native_row` over (Y,Y,Y) for SMPTE-240M.
  let (kr, kg, kb) = (6947i64, 22971i64, 2851i64); // SMPTE-240M Q15
  let rnd: i64 = 1 << 14;
  let matrix_luma: Vec<u16> = native_y
    .iter()
    .map(|&y| {
      let y = y as i64;
      ((kr * y + kg * y + kb * y + rnd) >> 15).clamp(0, 65535) as u16
    })
    .collect();
  assert_ne!(
    native_y, matrix_luma,
    "test fixture failed to exercise the matrix/native-Y divergence"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_matches_premult_bin_unpremult_oracle() {
  let packed = host_packed(0x1234);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut pm = direct_rgba_u16(&packed);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba_u16, oracle, "premult rgba_u16");
  assert_eq!(rgb_u16, drop_alpha(&oracle), "premult rgb_u16");
  assert_eq!(rgba, narrow_rgba_u8(&oracle), "premult rgba (narrowed)");
  // luma_u16 under premult is the area-mean of the NATIVE Y plane
  // (`mean(Y)`, native pass-through) — alpha-INDEPENDENT, NOT the color
  // path's `mean(Y*A)/mean(A)` (the un-premultiplied straight R). Compare
  // to the native-Y bin oracle, which equals the direct `ya16_to_luma_u16_row`.
  let binned_y = block_mean_native_y(&packed);
  let (_, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR);
  assert_eq!(lu16, lu16_ref, "premult luma_u16 (native-Y bin oracle)");
  assert_eq!(
    lu16, binned_y,
    "premult luma_u16 == native-Y bin (pass-through)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_nonuniform_alpha_luma_is_native_y_bin_not_color() {
  // The Ya16 analogue of the cited 8-bit counterexample, tiled over every
  // 2x2 block:
  //   (Y, A) = (0, 65535), (65535, 0), (0, 65535), (65535, 0)
  // Native-Y mean = (0 + 65535 + 0 + 65535 + 2) / 4 = 32768 (round-half-up),
  // so luma_u16 = 32768 (native pass-through) and luma = 32768 >> 8 = 128.
  // The premultiplied color collapses to mean(Y*A)/mean(A): the binned
  // premult-color R is 0, so an un-premultiplied (color-derived) luma_u16
  // would be 0 — the bug. Native-Y luma_u16 must be 32768.
  let mut packed = std::vec![0u16; SRC * SRC * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    let odd = (i % SRC) % 2 == 1;
    px[0] = if odd { 65535 } else { 0 }; // Y
    px[1] = if odd { 0 } else { 65535 }; // A
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  assert!(
    lu16.iter().all(|&y| y == 32768),
    "premult non-uniform-alpha luma_u16 must be native-Y mean 32768, got {lu16:?}"
  );
  assert!(
    luma.iter().all(|&y| y == 128),
    "premult non-uniform-alpha luma must be 32768 >> 8 = 128, got {luma:?}"
  );

  // Equals the native-Y bin oracle (direct `ya16_to_luma*` of the
  // block-meaned native Y), NOT the un-premultiplied color R (= 0 here).
  let binned_y = block_mean_native_y(&packed);
  assert!(binned_y.iter().all(|&y| y == 32768), "native-Y bin sanity");
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR);
  assert_eq!(luma, luma_ref, "premult luma == native-Y bin oracle");
  assert_eq!(lu16, lu16_ref, "premult luma_u16 == native-Y bin oracle");

  // Guard the test: the color-derived luma_u16 (un-premultiplied straight R)
  // really would be 0 here, pinning the divergence the bug produced.
  let mut pm = direct_rgba_u16(&packed);
  premultiply(&mut pm);
  let color_oracle = unpremultiply(&block_mean_rgba(&pm));
  let color_luma_r: Vec<u16> = color_oracle.chunks_exact(4).map(|px| px[0]).collect();
  assert!(
    color_luma_r.iter().all(|&r| r == 0),
    "fixture failed to exercise the color-vs-native-Y divergence"
  );
  assert_ne!(
    lu16, color_luma_r,
    "luma_u16 must NOT be the color-derived R"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_transparent_block_does_not_bleed() {
  let mut packed = host_packed(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    packed[i * 2] = 60000; // Y
    packed[i * 2 + 1] = 0; // A
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    &rgba_u16[..4],
    &[0, 0, 0, 0],
    "transparent block bled color"
  );

  let mut pm = direct_rgba_u16(&packed);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba_u16, oracle, "premult output != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn le_be_parity() {
  let packed = host_packed(0xC0DE);

  let render_le = || {
    let wire = as_le_u16(&packed);
    let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
    (rgba_u16, rgba)
  };
  let render_be = || {
    let wire = as_be_u16(&packed);
    let frame = Ya16BeFrame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink = MixedSinker::<Ya16<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    ya16_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
    (rgba_u16, rgba)
  };
  assert_eq!(render_le(), render_be(), "LE/BE outputs diverge");
}

#[test]
fn default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Ya16>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_plan_matches_direct() {
  let packed = host_packed(0x0F0F);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgba_u16,
    direct_rgba_u16(&packed),
    "identity plan == direct"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cross_frame_reset_reuses_streams() {
  let packed = host_packed(0x5151);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn accepts_alpha_mode_change_across_frames() {
  let packed = host_packed(0xB2B2);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    ya16_to(&frame, FR, M, &mut sink).expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm = direct_rgba_u16(&packed);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba_u16, oracle, "premult frame 2 output");
}

// The fractional-ratio reference reuses the packed-`Rgba64` source as an
// independent area-engine oracle (gated on `rgb`, where its frame / walker
// live); a `gray`-solo build covers fractional ratios via the shared
// `resample_packed_rgba_16bit` suite.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn fractional_ratio_matches_direct_then_bin() {
  // 8 -> 3 fractional downscale: `Ya16`'s decoded canonical RGBA must match
  // the packed-`Rgba64` source fed the same canonical row at the same plan.
  const F: usize = 3;
  let packed = host_packed(0xF2AC);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba_u16 = std::vec![0u16; F * F * 4];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(F, F))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let canonical = canonical_from_packed(&packed, SRC * SRC);
  let canonical_wire = as_le_u16(&canonical);
  let mut rgba_u16_ref = std::vec![0u16; F * F * 4];
  {
    use crate::source::rgba64_to;
    use mediaframe::frame::Rgba64Frame;
    let rsrc = Rgba64Frame::new(&canonical_wire, SRC as u32, SRC as u32, (SRC * 4) as u32);
    let mut sink = MixedSinker::<crate::source::Rgba64<false>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(F, F),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_ref)
    .unwrap();
    rgba64_to(&rsrc, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgba_u16, rgba_u16_ref,
    "Ya16 8->3 != packed-Rgba64 8->3 of canonical"
  );
}

// ---- direct-row freeze / sequencing (Ya16Row is publicly constructible) ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mid_frame_alpha_mode_flip_is_rejected() {
  let packed = host_packed(0x33AA);
  let wire = as_le_u16(&packed);
  let row_elems = SRC * 2;
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Ya16Row::new(&wire[..row_elems], 0, M, FR))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Ya16Row::new(&wire[row_elems..2 * row_elems], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame alpha flip not rejected: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_first_row_is_rejected() {
  let packed = host_packed(0x44BB);
  let wire = as_le_u16(&packed);
  let row_elems = SRC * 2;
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Ya16Row::new(&wire[row_elems..2 * row_elems], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row not rejected: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_sink_is_a_noop() {
  let packed = host_packed(0x4242);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);
  let mut sink =
    MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
}

// ---- limited-range (full_range = false) native-Y luma regression --------------
//
// The direct `Ya16` luma is `Y >> 8` and luma_u16 the native Y pass-through
// (`ya16_to_luma_row` / `ya16_to_luma_u16_row`): neither applies the matrix or
// range. luma_u16 was already native (`NATIVE_Y_LUMA`); luma (u8) was derived
// from the binned RGB via `rgb_to_luma_row`, which equals `Y >> 8` only at
// `full_range = true`. A `full_range = false` row therefore corrupted the
// narrowed grayscale. These tests pin the limited-range case for BOTH outputs.

/// Direct (identity) `Ya16` luma / luma_u16 of a binned-Y plane at the given
/// range — the byte-exact native-Y oracle. `binned_y[i]` is the binned native
/// Y (alpha forced opaque, irrelevant to luma).
fn direct_luma_of_binned_y(binned_y: &[u16], full_range: bool) -> (Vec<u8>, Vec<u16>) {
  let n = binned_y.len();
  let mut packed = std::vec![0u16; n * 2];
  for (i, &y) in binned_y.iter().enumerate() {
    packed[i * 2] = y;
    packed[i * 2 + 1] = 0xFFFF;
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, n as u32, 1, (n * 2) as u32);
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  let mut sink = MixedSinker::<Ya16>::new(n, 1)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut lu16)
    .unwrap();
  ya16_to(&frame, full_range, M, &mut sink).unwrap();
  (luma, lu16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn limited_range_luma_is_native_y_not_rgb_derived() {
  let packed = host_packed(0xCAFE);
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya16_to(&frame, FR_LIMITED, M, &mut sink).unwrap();
  }

  // Native-Y oracle: the area-mean of the native Y plane (alpha-independent
  // by construction).
  let binned_y = block_mean_native_y(&packed);
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR_LIMITED);
  assert_eq!(luma, luma_ref, "limited-range luma must be native Y >> 8");
  let y_narrowed: Vec<u8> = binned_y.iter().map(|&y| (y >> 8) as u8).collect();
  assert_eq!(luma, y_narrowed, "limited-range luma == binned Y >> 8");
  assert_eq!(
    lu16, lu16_ref,
    "limited-range luma_u16 must be native Y pass-through"
  );
  assert_eq!(lu16, binned_y, "limited-range luma_u16 == binned native Y");

  // Native Y is range-independent: the same source at full range yields the
  // identical luma. A range-derived luma (u8) would differ here.
  let mut luma_fr = std::vec![0u8; OUT * OUT];
  let mut lu16_fr = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma_fr)
        .unwrap()
        .with_luma_u16(&mut lu16_fr)
        .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, luma_fr, "native-Y luma must be range-independent");
  assert_eq!(lu16, lu16_fr, "native-Y luma_u16 must be range-independent");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn limited_range_y16_luma_is_native_not_rgb_scaled() {
  // Limited-range counterexample at 16-bit: a uniform native Y. The direct
  // `Ya16` luma is `Y >> 8` and luma_u16 is Y; a limited-range
  // `rgb_to_luma_row` of (Y,Y,Y) >> 8 would scale the narrowed gray. Use a
  // low Y (the 16-bit analogue of the 8-bit Y = 16 case): Y = 0x1000 → narrows
  // to 0x10 = 16.
  const Y: u16 = 0x1000;
  let mut packed = std::vec![0u16; SRC * SRC * 2];
  for px in packed.chunks_exact_mut(2) {
    px[0] = Y;
    px[1] = 0xFFFF; // opaque
  }
  let wire = as_le_u16(&packed);
  let frame = Ya16Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya16_to(&frame, FR_LIMITED, M, &mut sink).unwrap();
  }
  assert!(
    luma.iter().all(|&y| y == (Y >> 8) as u8),
    "limited-range luma must stay native Y >> 8 = {}, got {luma:?}",
    (Y >> 8) as u8
  );
  assert!(
    lu16.iter().all(|&y| y == Y),
    "limited-range luma_u16 must stay native Y = {Y}, got {lu16:?}"
  );
}
