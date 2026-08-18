//! Alpha-aware fused-downscale coverage for the high-bit **planar** 4:4:4
//! YUV-with-alpha family — `Yuva444p9` / `Yuva444p10` / `Yuva444p12` /
//! `Yuva444p14` / `Yuva444p16` (LE + BE wire). Low-packed `u16` Y / U / V /
//! A planes, full-width chroma (no upsampling).
//!
//! These extend the 8-bit `Yuva444p` route to native depth: they route
//! through the same packed-YUVA tail
//! ([`packed_yuva444_resample`](super::super::packed_yuva444_resample)) with
//! **three** independent native-precision binnings — but unlike the 8-bit
//! sink (no u16 colour outputs) the high-bit sinks ACTIVATE the independent
//! u16 colour path:
//! - **u8 colour (rgb / rgba / hsv)** bins the converted u8 RGBA row
//!   (`yuva444pN_to_rgba_row_endian`, real source α). Under
//!   `AlphaMode::Premultiplied` the row is premultiplied at MAX = 255.
//! - **u16 colour (rgb_u16 / rgba_u16)** bins the **independent** native u16
//!   RGBA row (`yuva444pN_to_rgba_u16_row_endian`) — never a narrowing of the
//!   u8 bin. Alpha is the real native-depth plane value; premultiply /
//!   un-premultiply uses MAX = `(1 << BITS) - 1`.
//! - **luma** bins the **low-packed** native Y (a raw host-native copy of the
//!   Y plane — planar YUVA Y stores logical values directly, NOT the
//!   high-bit-packed semi-planar P-format de-pack); `luma = binned_Y >>
//!   (BITS - 8)`, `luma_u16` is the host-native binned Y. Alpha- and
//!   range-independent.
//!
//! Each output is byte-identical to the area-bin of a **direct** identity
//! conversion (convert-then-bin), so the colour oracles drive a direct sink
//! at source resolution and 2x2-block-mean its output (premultiplying first
//! for the premultiplied mode), and the luma oracle area-bins the native Y
//! plane then narrows. The independence + anti-correlated-alpha
//! counterexamples pin the real parity bugs: narrowing the u16 bin to u8
//! would change a varying ramp's colour, and deriving luma from the
//! premultiplied colour would diverge from the native-Y mean when alpha
//! varies.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const FR_LIMITED: bool = false;

/// Re-encode a host-native u16 slice as host-independent LE-wire byte
/// storage (the `*LeFrame` plane contract), recovered via `from_le`.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as host-independent BE-wire byte
/// storage (the `*BeFrame` plane contract), recovered via `from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Round-half-up 2x2-block mean of an `SRC`-grid 4-channel u8 RGBA plane
/// (alpha included).
fn block_mean_rgba_u8(src: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Round-half-up 2x2-block mean of an `SRC`-grid 4-channel u16 RGBA plane.
fn block_mean_rgba_u16(src: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u16;
      }
    }
  }
  out
}

/// Round-half-up 2x2-block mean of an `SRC`-grid u16 plane (host-native
/// logical values) — the native-Y oracle source.
fn block_mean_u16(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((acc + 2) / 4) as u16;
    }
  }
  out
}

fn drop_alpha_u8(rgba: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Premultiply a canonical RGBA plane in place at the given channel maximum
/// (`255` for u8, `(1 << BITS) - 1` for native u16) — matches the tail's
/// `premultiply_rgba_row_in_place`.
fn premultiply_u8(plane: &mut [u8]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + 127) / 255) as u8;
    }
  }
}

fn premultiply_u16(plane: &mut [u16], max: u32) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + max / 2) / max) as u16;
    }
  }
}

/// Un-premultiply a binned RGBA plane (guard A == 0 → 0) at the given
/// maximum — matches the tail's `unpremultiply_binned_rgba_into`.
fn unpremultiply_u8(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; plane.len()];
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * 255 + a / 2)
        .checked_div(a)
        .map_or(0, |q| q.min(255)) as u8;
    }
    o[3] = i[3];
  }
  out
}

fn unpremultiply_u16(plane: &[u16], max: u32) -> Vec<u16> {
  let mut out = vec![0u16; plane.len()];
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * max + a / 2)
        .checked_div(a)
        .map_or(0, |q| q.min(max)) as u16;
    }
    o[3] = i[3];
  }
  out
}

// A per-depth macro keeps the five near-identical suites in lockstep while
// naming each test after its bit depth. `$frame_le` / `$frame_be` are the
// LE / BE frame types, `$marker` the source marker, `$row` the row type,
// `$walker` the LE walker, `$walker_be` the `_endian` walker, `$bits` the
// active depth.
macro_rules! yuva444p_high_bit_resample_suite {
  (
    $mod:ident, $frame_le:ident, $frame_be:ident, $marker:ident, $row:ident,
    $walker:ident, $walker_be:ident, $bits:literal,
  ) => {
    mod $mod {
      use super::*;
      use crate::{
        frame::{$frame_be, $frame_le},
        source::{$marker, $row, $walker, $walker_be},
      };

      const MASK: u16 = ((1u32 << $bits) - 1) as u16;
      const MAXV: u32 = (1u32 << $bits) - 1;

      /// #334: the masked-Y twin missed this YUVA 4:4:4 arm, so its `luma` /
      /// `luma_u16` feeds de-interleaved the native Y UNMASKED — an
      /// out-of-range sample (`MASK + 1`; a no-op at 16-bit) escaped CLAMPED to
      /// the native max instead of DEPTH-MASKED (`& MASK`). A uniform dirty Y
      /// must publish `luma_u16 == clean & MASK` and `luma == that >> (BITS -
      /// 8)` on the area AND filter tiers, both wire endiannesses.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn luma_u16_masks_dirty_high_bit_y() {
        let clean: u16 = (MASK / 3) & MASK;
        let dirty: u16 = clean | MASK.wrapping_add(1);
        let want_u16 = std::vec![clean; OUT * OUT];
        let want_u8 = std::vec![(clean >> ($bits - 8)) as u8; OUT * OUT];
        let mid = MASK / 2;
        // Chroma / alpha are unread by a luma-only sink; keep them neutral.
        let yp = std::vec![dirty; SRC * SRC];
        let u = std::vec![mid; SRC * SRC];
        let v = std::vec![mid; SRC * SRC];
        let a = std::vec![MASK; SRC * SRC];
        let (y_le, u_le, v_le, a_le) =
          (as_le_u16(&yp), as_le_u16(&u), as_le_u16(&v), as_le_u16(&a));
        let (y_be, u_be, v_be, a_be) =
          (as_be_u16(&yp), as_be_u16(&u), as_be_u16(&v), as_be_u16(&a));

        let mut luma = std::vec![0u8; OUT * OUT];
        let mut lu16 = std::vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut lu16)
          .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le, &a_le), FR, M, &mut sink).unwrap();
        }
        assert_eq!(lu16, want_u16, "LE area luma_u16 not depth-masked");
        assert_eq!(luma, want_u8, "LE area luma not depth-masked");

        let mut be_luma = std::vec![0u8; OUT * OUT];
        let mut be_lu16 = std::vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma(&mut be_luma)
          .unwrap()
          .with_luma_u16(&mut be_lu16)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be, &a_be), FR, M, &mut sink).unwrap();
        }
        assert_eq!(be_lu16, want_u16, "BE area luma_u16 not depth-masked");
        assert_eq!(be_luma, want_u8, "BE area luma not depth-masked");

        let mut luma = std::vec![0u8; OUT * OUT];
        let mut lu16 = std::vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut lu16)
          .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le, &a_le), FR, M, &mut sink).unwrap();
        }
        assert_eq!(lu16, want_u16, "LE filter luma_u16 not depth-masked");
        assert_eq!(luma, want_u8, "LE filter luma not depth-masked");

        let mut be_luma = std::vec![0u8; OUT * OUT];
        let mut be_lu16 = std::vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_luma(&mut be_luma)
          .unwrap()
          .with_luma_u16(&mut be_lu16)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be, &a_be), FR, M, &mut sink).unwrap();
        }
        assert_eq!(be_lu16, want_u16, "BE filter luma_u16 not depth-masked");
        assert_eq!(be_luma, want_u8, "BE filter luma not depth-masked");
      }

      /// Pseudo-random Y / U / V / A planes (all full-resolution in 4:4:4,
      /// low-packed at `$bits`). Alpha varies (not all-opaque).
      fn planes(seed: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let mut y = vec![0u16; SRC * SRC];
        let mut u = vec![0u16; SRC * SRC];
        let mut v = vec![0u16; SRC * SRC];
        let mut a = vec![0u16; SRC * SRC];
        super::super::pseudo_random_u16_low_n_bits(&mut y, seed, $bits);
        super::super::pseudo_random_u16_low_n_bits(&mut u, seed ^ 0x1111_1111, $bits);
        super::super::pseudo_random_u16_low_n_bits(&mut v, seed ^ 0x2222_2222, $bits);
        super::super::pseudo_random_u16_low_n_bits(&mut a, seed ^ 0x3333_3333, $bits);
        (y, u, v, a)
      }

      fn frame<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16], a: &'a [u16]) -> $frame_le<'a> {
        $frame_le::try_new(
          y, u, v, a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
        )
        .unwrap()
      }
      fn frame_be<'a>(y: &'a [u16], u: &'a [u16], v: &'a [u16], a: &'a [u16]) -> $frame_be<'a> {
        $frame_be::try_new(
          y, u, v, a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
        )
        .unwrap()
      }

      /// Direct (identity) full-resolution canonical u8 RGBA — the u8 colour
      /// oracle source (alpha = `a >> (BITS - 8)`).
      fn direct_rgba_u8(y: &[u16], u: &[u16], v: &[u16], a: &[u16], fr: bool) -> Vec<u8> {
        let mut rgba = vec![0u8; SRC * SRC * 4];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgba(&mut rgba)
          .unwrap();
        $walker(&frame(y, u, v, a), fr, M, &mut sink).unwrap();
        rgba
      }
      /// Direct (identity) full-resolution canonical native u16 RGBA — the
      /// independent u16 colour oracle source (alpha = native `a`).
      fn direct_rgba_u16(y: &[u16], u: &[u16], v: &[u16], a: &[u16], fr: bool) -> Vec<u16> {
        let mut rgba = vec![0u16; SRC * SRC * 4];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgba_u16(&mut rgba)
          .unwrap();
        $walker(&frame(y, u, v, a), fr, M, &mut sink).unwrap();
        rgba
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgba_u8_and_u16_are_block_mean_of_direct() {
        let (y, u, v, a) = planes(0x51A1);
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgba,
          block_mean_rgba_u8(&direct_rgba_u8(&y, &u, &v, &a, FR)),
          "straight rgba (u8) == block mean of direct"
        );
        assert_eq!(
          rgba_u16,
          block_mean_rgba_u16(&direct_rgba_u16(&y, &u, &v, &a, FR)),
          "straight rgba_u16 (independent) == block mean of direct"
        );
        assert!(
          rgba.chunks_exact(4).any(|px| px[3] != 0xFF),
          "u8 resampled alpha was forced opaque — area-mean alpha lost"
        );
        assert!(
          rgba_u16.chunks_exact(4).any(|px| px[3] != MASK),
          "u16 resampled alpha was forced opaque — area-mean alpha lost"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_all_outputs_derive_correctly() {
        let (y, u, v, a) = planes(0xBEEF);

        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut luma = vec![0u8; OUT * OUT];
        let mut hh = vec![0u8; OUT * OUT];
        let mut ss = vec![0u8; OUT * OUT];
        let mut vv = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
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
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }

        let binned_u8 = block_mean_rgba_u8(&direct_rgba_u8(&y, &u, &v, &a, FR));
        assert_eq!(rgba, binned_u8, "rgba == block mean");
        let binned_rgb_u8 = drop_alpha_u8(&binned_u8);
        assert_eq!(rgb, binned_rgb_u8, "rgb == drop-alpha(binned)");

        let binned_u16 = block_mean_rgba_u16(&direct_rgba_u16(&y, &u, &v, &a, FR));
        assert_eq!(rgba_u16, binned_u16, "rgba_u16 == block mean (independent)");
        assert_eq!(
          rgb_u16,
          drop_alpha_u16(&binned_u16),
          "rgb_u16 == drop-alpha"
        );

        let y_binned = block_mean_u16(&y);
        let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "luma == native-Y bin >> (BITS - 8)");

        let mut h_ref = vec![0u8; OUT * OUT];
        let mut s_ref = vec![0u8; OUT * OUT];
        let mut v_ref = vec![0u8; OUT * OUT];
        crate::row::rgb_to_hsv_row(
          &binned_rgb_u8,
          &mut h_ref,
          &mut s_ref,
          &mut v_ref,
          OUT * OUT,
          false,
        );
        assert_eq!(hh, h_ref, "hsv H");
        assert_eq!(ss, s_ref, "hsv S");
        assert_eq!(vv, v_ref, "hsv V");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn u8_color_is_not_a_narrowing_of_u16() {
        // The u8 and u16 YUVA→RGBA kernels round and scale independently;
        // narrowing the binned u16 colour to u8 (`>> (BITS - 8)`) diverges
        // from the genuine u8 bin. Each must match its OWN oracle.
        let (y, u, v, a) = planes(0x1234);
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgba,
          block_mean_rgba_u8(&direct_rgba_u8(&y, &u, &v, &a, FR))
        );
        assert_eq!(
          rgba_u16,
          block_mean_rgba_u16(&direct_rgba_u16(&y, &u, &v, &a, FR))
        );
        let narrowed: Vec<u8> = rgba_u16.iter().map(|&c| (c >> ($bits - 8)) as u8).collect();
        assert_ne!(
          rgba, narrowed,
          "u8 colour must be an independent bin, not a narrowed u16 bin"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_matches_premult_bin_unpremult_oracle() {
        let (y, u, v, a) = planes(0x77AA);

        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        let mut luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }

        // u8: premult at MAX = 255.
        let mut pm8 = direct_rgba_u8(&y, &u, &v, &a, FR);
        premultiply_u8(&mut pm8);
        let oracle8 = unpremultiply_u8(&block_mean_rgba_u8(&pm8));
        assert_eq!(rgba, oracle8, "premult rgba (u8)");
        assert_eq!(rgb, drop_alpha_u8(&oracle8), "premult rgb (u8)");

        // u16: premult at MAX = (1 << BITS) - 1, independent of u8.
        let mut pm16 = direct_rgba_u16(&y, &u, &v, &a, FR);
        premultiply_u16(&mut pm16, MAXV);
        let oracle16 = unpremultiply_u16(&block_mean_rgba_u16(&pm16), MAXV);
        assert_eq!(rgba_u16, oracle16, "premult rgba_u16 (independent)");
        assert_eq!(rgb_u16, drop_alpha_u16(&oracle16), "premult rgb_u16");

        // luma is native Y, NOT colour-derived — unaffected by premult.
        let y_binned = block_mean_u16(&y);
        let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "premult luma == native-Y bin >> shift");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_transparent_block_does_not_bleed() {
        let (mut y, u, v, mut a) = planes(0xABCD);
        for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
          let i = off.1 * SRC + off.0;
          y[i] = MASK;
          a[i] = 0;
        }
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          &rgba[..4],
          &[0, 0, 0, 0],
          "u8 transparent block bled colour"
        );
        assert_eq!(
          &rgba_u16[..4],
          &[0, 0, 0, 0],
          "u16 transparent block bled colour"
        );
        let mut pm8 = direct_rgba_u8(&y, &u, &v, &a, FR);
        premultiply_u8(&mut pm8);
        assert_eq!(
          rgba,
          unpremultiply_u8(&block_mean_rgba_u8(&pm8)),
          "u8 premult != oracle"
        );
        let mut pm16 = direct_rgba_u16(&y, &u, &v, &a, FR);
        premultiply_u16(&mut pm16, MAXV);
        assert_eq!(
          rgba_u16,
          unpremultiply_u16(&block_mean_rgba_u16(&pm16), MAXV),
          "u16 premult != oracle"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_nonuniform_alpha_luma_is_native_y_not_colour() {
        // (Y, A) = (0, MAX), (MAX, 0) alternating columns → native-Y mean is
        // mid, but premult colour R collapses to 0. luma must follow native Y.
        let mut y = vec![0u16; SRC * SRC];
        let mut a = vec![0u16; SRC * SRC];
        for i in 0..SRC * SRC {
          let odd = !(i % SRC).is_multiple_of(2);
          y[i] = if odd { MASK } else { 0 };
          a[i] = if odd { 0 } else { MASK };
        }
        let mid = MASK / 2;
        let u = vec![mid; SRC * SRC];
        let v = vec![mid; SRC * SRC];

        let mut luma = vec![0u8; OUT * OUT];
        let mut lu16 = vec![0u16; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut lu16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        let y_binned = block_mean_u16(&y);
        let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "premult luma == native-Y bin");
        assert_eq!(lu16, y_binned, "premult luma_u16 == native-Y bin");

        // The premult colour R really does collapse to 0 here (fixture check).
        let mut pm16 = direct_rgba_u16(&y, &u, &v, &a, FR);
        premultiply_u16(&mut pm16, MAXV);
        let color = unpremultiply_u16(&block_mean_rgba_u16(&pm16), MAXV);
        let color_r: Vec<u16> = color.chunks_exact(4).map(|px| px[0]).collect();
        assert!(
          color_r.iter().all(|&r| r == 0),
          "fixture failed to exercise the divergence"
        );
        assert_ne!(lu16, color_r, "luma_u16 must NOT be the colour-derived R");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_y_luma_is_range_independent() {
        let (y, u, v, a) = planes(0xCAFE);
        let render = |fr: bool| {
          let mut luma = vec![0u8; OUT * OUT];
          let mut lu16 = vec![0u16; OUT * OUT];
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut lu16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), fr, M, &mut sink).unwrap();
          (luma, lu16)
        };
        let (luma_lim, lu16_lim) = render(FR_LIMITED);
        let (luma_full, lu16_full) = render(FR);
        let y_binned = block_mean_u16(&y);
        let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(
          luma_lim, luma_ref,
          "limited-range luma == native-Y bin >> shift"
        );
        assert_eq!(lu16_lim, y_binned, "limited-range luma_u16 == native-Y bin");
        assert_eq!(
          luma_lim, luma_full,
          "native-Y luma must be range-independent"
        );
        assert_eq!(
          lu16_lim, lu16_full,
          "native-Y luma_u16 must be range-independent"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_and_premult_differ_under_varying_alpha() {
        let (y, u, v, mut a) = planes(0x0BAD);
        for (i, px) in a.iter_mut().enumerate() {
          *px = ((16u32 + (i as u32).wrapping_mul(5)) & MASK as u32) as u16;
        }
        let render = |mode: AlphaMode| {
          let mut rgba = vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(mode)
          .with_rgba(&mut rgba)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
          rgba
        };
        assert_ne!(
          render(AlphaMode::Straight),
          render(AlphaMode::Premultiplied),
          "alpha mode had no effect"
        );
      }

      #[test]
      fn default_alpha_mode_is_straight() {
        let sink = MixedSinker::<$marker>::new(SRC, SRC);
        assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn chroma_full_res_under_saturated_chroma() {
        // 4:4:4 carries per-pixel chroma, so a high-frequency chroma pattern
        // must bin to the exact 2x2 mean of the direct conversion — for both
        // the u8 and the independent u16 colour binnings.
        let yc = MASK / 2;
        let y = vec![yc; SRC * SRC];
        let mut u = vec![0u16; SRC * SRC];
        let mut v = vec![0u16; SRC * SRC];
        let a = vec![(MASK as u32 * 4 / 5) as u16; SRC * SRC];
        for i in 0..SRC * SRC {
          let cb = ((i % SRC) + (i / SRC)).is_multiple_of(2);
          u[i] = if cb { MASK } else { 0 };
          v[i] = if cb { 0 } else { MASK };
        }
        let mut rgb = vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgb,
          drop_alpha_u8(&block_mean_rgba_u8(&direct_rgba_u8(&y, &u, &v, &a, FR))),
          "4:4:4 chroma rgb == block mean of direct"
        );
        assert_eq!(
          rgb_u16,
          drop_alpha_u16(&block_mean_rgba_u16(&direct_rgba_u16(&y, &u, &v, &a, FR))),
          "4:4:4 chroma rgb_u16 == block mean of direct"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_direct() {
        let (y, u, v, a) = planes(0x0F0F);
        let mut rgba = vec![0u8; SRC * SRC * 4];
        let mut rgba_u16 = vec![0u16; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(
          rgba,
          direct_rgba_u8(&y, &u, &v, &a, FR),
          "identity rgba == direct"
        );
        assert_eq!(
          rgba_u16,
          direct_rgba_u16(&y, &u, &v, &a, FR),
          "identity rgba_u16 == direct"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn le_be_outputs_identical() {
        let (y, u, v, a) = planes(0x33AA);
        let (y_le, u_le, v_le, a_le) = (as_le_u16(&y), as_le_u16(&u), as_le_u16(&v), as_le_u16(&a));
        let (y_be, u_be, v_be, a_be) = (as_be_u16(&y), as_be_u16(&u), as_be_u16(&v), as_be_u16(&a));

        let mut le_rgba = vec![0u8; OUT * OUT * 4];
        let mut le_rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut le_luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut le_rgba)
          .unwrap()
          .with_rgba_u16(&mut le_rgba_u16)
          .unwrap()
          .with_luma(&mut le_luma)
          .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le, &a_le), FR, M, &mut sink).unwrap();
        }

        let mut be_rgba = vec![0u8; OUT * OUT * 4];
        let mut be_rgba_u16 = vec![0u16; OUT * OUT * 4];
        let mut be_luma = vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut be_rgba)
          .unwrap()
          .with_rgba_u16(&mut be_rgba_u16)
          .unwrap()
          .with_luma(&mut be_luma)
          .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be, &a_be), FR, M, &mut sink).unwrap();
        }

        assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
        assert_eq!(le_rgba_u16, be_rgba_u16, "rgba_u16 LE/BE diverge");
        assert_eq!(le_luma, be_luma, "luma LE/BE diverge");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn simd_matches_scalar_and_fractional() {
        let (y, u, v, a) = planes(0x1357);
        let run = |simd: bool, ow: usize| {
          let mut rgba = vec![0u8; ow * ow * 4];
          let mut rgba_u16 = vec![0u16; ow * ow * 4];
          let mut luma = vec![0u8; ow * ow];
          {
            let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
              SRC,
              SRC,
              AreaResampler::to(ow, ow),
            )
            .unwrap()
            .with_simd(simd)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_rgba_u16(&mut rgba_u16)
            .unwrap()
            .with_luma(&mut luma)
            .unwrap();
            $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
          }
          (rgba, rgba_u16, luma)
        };
        assert_eq!(
          run(true, OUT),
          run(false, OUT),
          "integer-ratio SIMD != scalar"
        );
        // 8 -> 3 fractional ratio.
        assert_eq!(
          run(true, 3),
          run(false, 3),
          "fractional-ratio SIMD != scalar"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cross_frame_reset_and_alpha_rearm() {
        let (y, u, v, a) = planes(0x5151);
        let mut rgba = vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          // Frame 1 (straight), then frame 2 with a re-armed premult mode.
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
          sink.set_alpha_mode(AlphaMode::Premultiplied);
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink)
            .expect("a fresh frame must accept a different alpha mode");
        }
        let mut pm = direct_rgba_u8(&y, &u, &v, &a, FR);
        premultiply_u8(&mut pm);
        assert_eq!(
          rgba,
          unpremultiply_u8(&block_mean_rgba_u8(&pm)),
          "premult frame 2 output"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn mid_frame_alpha_mode_flip_is_rejected() {
        let (y, u, v, a) = planes(0x44BB);
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(
            &y[..SRC],
            &u[..SRC],
            &v[..SRC],
            &a[..SRC],
            0,
            M,
            FR,
          ))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(
            &y[SRC..2 * SRC],
            &u[SRC..2 * SRC],
            &v[SRC..2 * SRC],
            &a[SRC..2 * SRC],
            1,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "mid-frame alpha flip not rejected: {err:?}"
        );
      }

      #[test]
      fn out_of_sequence_first_row_is_rejected() {
        let (y, u, v, a) = planes(0x2244);
        let mut rgba = vec![0u8; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let r = 2 * SRC;
        let err = sink
          .process($row::new(
            &y[r..r + SRC],
            &u[r..r + SRC],
            &v[r..r + SRC],
            &a[r..r + SRC],
            2,
            M,
            FR,
          ))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "out-of-sequence first row not rejected: {err:?}"
        );
        assert!(rgba.iter().all(|&b| b == 0), "rejected row mutated output");
      }

      #[test]
      fn no_output_sink_is_a_noop() {
        let (y, u, v, a) = planes(0x4242);
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_identity_luma_u16_is_native_y() {
        // The direct (NoopResampler) path must also emit luma_u16 — the
        // host-native logical Y — and luma as `Y >> (BITS - 8)`; without the
        // direct-path emission `with_luma_u16` would silently never write.
        let (y, u, v, a) = planes(0x9C9C);
        let mut luma = vec![0u8; SRC * SRC];
        let mut lu16 = vec![0u16; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut lu16)
            .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(lu16, y, "direct luma_u16 == native logical Y");
        let luma_ref: Vec<u8> = y.iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(luma, luma_ref, "direct luma == Y >> (BITS - 8)");
      }

      /// #334 R2 (identity tier): the direct (NoopResampler)
      /// `yuva444p_high_bit_process` luma path de-interleaved the native Y into
      /// `luma_u16` / 8-bit `luma` WITHOUT the `(1 << BITS) - 1` depth mask, so a
      /// geometry-only-accepted frame carrying out-of-range Y (bits above BITS; a
      /// no-op at 16-bit) published the raw dirty value to `luma_u16` while the
      /// RGB/RGBA kernels masked the same sample via `extract_hb`. A uniform dirty
      /// Y must publish `luma_u16 == clean & MASK` and `luma == that >> (BITS -
      /// 8)`, both wire endiannesses. Mirrors the resample-tier
      /// `luma_u16_masks_dirty_high_bit_y` for the no-resampler path.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_identity_luma_masks_dirty_high_bit_y() {
        let clean: u16 = (MASK / 3) & MASK;
        let dirty: u16 = clean | MASK.wrapping_add(1);
        let want_u16 = std::vec![clean; SRC * SRC];
        let want_u8 = std::vec![(clean >> ($bits - 8)) as u8; SRC * SRC];
        let mid = MASK / 2;
        // Chroma / alpha are unread by a luma-only sink; keep them in-range so
        // ONLY the dirty Y exercises the masked luma path.
        let yp = std::vec![dirty; SRC * SRC];
        let u = std::vec![mid; SRC * SRC];
        let v = std::vec![mid; SRC * SRC];
        let a = std::vec![MASK; SRC * SRC];
        let (y_le, u_le, v_le, a_le) =
          (as_le_u16(&yp), as_le_u16(&u), as_le_u16(&v), as_le_u16(&a));
        let (y_be, u_be, v_be, a_be) =
          (as_be_u16(&yp), as_be_u16(&u), as_be_u16(&v), as_be_u16(&a));

        let mut luma = std::vec![0u8; SRC * SRC];
        let mut lu16 = std::vec![0u16; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut lu16)
            .unwrap();
          $walker(&frame(&y_le, &u_le, &v_le, &a_le), FR, M, &mut sink).unwrap();
        }
        assert_eq!(lu16, want_u16, "LE direct luma_u16 not depth-masked");
        assert_eq!(luma, want_u8, "LE direct luma not depth-masked");

        let mut be_luma = std::vec![0u8; SRC * SRC];
        let mut be_lu16 = std::vec![0u16; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker<true>>::new(SRC, SRC)
            .with_luma(&mut be_luma)
            .unwrap()
            .with_luma_u16(&mut be_lu16)
            .unwrap();
          $walker_be::<_, true>(&frame_be(&y_be, &u_be, &v_be, &a_be), FR, M, &mut sink).unwrap();
        }
        assert_eq!(be_lu16, want_u16, "BE direct luma_u16 not depth-masked");
        assert_eq!(be_luma, want_u8, "BE direct luma not depth-masked");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_luma_u16_with_hsv_no_rgb_buffer_writes_both() {
        // #263 PR 8: on the direct (NoopResampler) path, `with_luma_u16` +
        // `with_hsv` with NO rgb / rgba plane attached routes HSV through
        // the matching direct `yuv*p*_to_hsv_row_endian` kernel — RGB-free
        // (no rgb scratch). Both outputs must be produced: luma_u16 is the
        // native logical Y; HSV must match the RGB-attached oracle (same
        // kernel — direct vs derived-from-RGB is the only difference).
        let (y, u, v, a) = planes(0x7E57);

        // RGB-free path: luma_u16 + hsv only.
        let mut lu16 = vec![0u16; SRC * SRC];
        let mut hh = vec![0u8; SRC * SRC];
        let mut ss = vec![0u8; SRC * SRC];
        let mut vv = vec![0u8; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_luma_u16(&mut lu16)
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
          // White-box: the direct HSV path is RGB-free — the rgb scratch
          // is never grown.
          assert_eq!(
            sink.rgb_scratch_capacity(),
            0,
            "HSV-only direct path must not allocate the rgb scratch"
          );
        }
        assert_eq!(lu16, y, "no-rgb direct luma_u16 == native logical Y");

        // Oracle: same source with rgb attached (HSV derives from the
        // caller RGB buffer) — HSV must be identical.
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut oh = vec![0u8; SRC * SRC];
        let mut os = vec![0u8; SRC * SRC];
        let mut ov = vec![0u8; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_hsv(&mut oh, &mut os, &mut ov)
            .unwrap();
          $walker(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
        }
        assert_eq!(hh, oh, "direct H == rgb-attached H");
        assert_eq!(ss, os, "direct S == rgb-attached S");
        assert_eq!(vv, ov, "direct V == rgb-attached V");
        // The direct path actually ran a real conversion (not all-zero).
        assert!(
          hh.iter().chain(ss.iter()).chain(vv.iter()).any(|&b| b != 0),
          "HSV direct path produced no output"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_hsv_only_is_rgb_free_and_infallible() {
        // #263 PR 8: the direct HSV-only path (`with_luma` +
        // `with_luma_u16` + `with_hsv`, NO rgb / rgba plane) now routes HSV
        // through the matching direct `yuv*p*_to_hsv_row_endian` kernel —
        // RGB-free (no rgb scratch). Proof: arm the rgb-scratch allocation
        // failpoint (which would surface `AllocationFailed` if the path
        // still grew the scratch); the row must instead SUCCEED, leave the
        // scratch unallocated, and write every output. The failpoint is
        // take-on-read, so disarm it after to avoid leaking into a later
        // same-thread test.
        let (y, u, v, a) = planes(0x7E57);
        let mut luma = vec![0u8; SRC * SRC];
        let mut lu16 = vec![0u16; SRC * SRC];
        let mut hh = vec![0u8; SRC * SRC];
        let mut ss = vec![0u8; SRC * SRC];
        let mut vv = vec![0u8; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut lu16)
            .unwrap()
            .with_hsv(&mut hh, &mut ss, &mut vv)
            .unwrap();
          sink.begin_frame(SRC as u32, SRC as u32).unwrap();
          super::super::super::arm_rgb_scratch_alloc_failure();
          sink
            .process($row::new(
              &y[..SRC],
              &u[..SRC],
              &v[..SRC],
              &a[..SRC],
              0,
              M,
              FR,
            ))
            .expect("HSV-only direct row must be RGB-free (no scratch alloc)");
          assert_eq!(
            sink.rgb_scratch_capacity(),
            0,
            "HSV-only direct path must not allocate the rgb scratch"
          );
        }
        super::super::super::disarm_rgb_scratch_alloc_failure();
        let lu16_ref: Vec<u16> = y[..SRC].to_vec();
        assert_eq!(
          &lu16[..SRC],
          &lu16_ref[..],
          "direct luma_u16 == native logical Y"
        );
        let luma_ref: Vec<u8> = y[..SRC].iter().map(|&p| (p >> ($bits - 8)) as u8).collect();
        assert_eq!(
          &luma[..SRC],
          &luma_ref[..],
          "direct luma == Y >> (BITS - 8)"
        );
        let mut rgb = vec![0u8; SRC * SRC * 3];
        let mut oh = vec![0u8; SRC * SRC];
        let mut os = vec![0u8; SRC * SRC];
        let mut ov = vec![0u8; SRC * SRC];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_hsv(&mut oh, &mut os, &mut ov)
            .unwrap();
          sink.begin_frame(SRC as u32, SRC as u32).unwrap();
          sink
            .process($row::new(
              &y[..SRC],
              &u[..SRC],
              &v[..SRC],
              &a[..SRC],
              0,
              M,
              FR,
            ))
            .unwrap();
        }
        assert_eq!(&hh[..SRC], &oh[..SRC], "direct H == rgb-attached H");
        assert_eq!(&ss[..SRC], &os[..SRC], "direct S == rgb-attached S");
        assert_eq!(&vv[..SRC], &ov[..SRC], "direct V == rgb-attached V");
      }
    }
  };
}

yuva444p_high_bit_resample_suite!(
  yuva444p9,
  Yuva444p9LeFrame,
  Yuva444p9BeFrame,
  Yuva444p9,
  Yuva444p9Row,
  yuva444p9_to,
  yuva444p9_to_endian,
  9,
);
yuva444p_high_bit_resample_suite!(
  yuva444p10,
  Yuva444p10LeFrame,
  Yuva444p10BeFrame,
  Yuva444p10,
  Yuva444p10Row,
  yuva444p10_to,
  yuva444p10_to_endian,
  10,
);
yuva444p_high_bit_resample_suite!(
  yuva444p12,
  Yuva444p12LeFrame,
  Yuva444p12BeFrame,
  Yuva444p12,
  Yuva444p12Row,
  yuva444p12_to,
  yuva444p12_to_endian,
  12,
);
yuva444p_high_bit_resample_suite!(
  yuva444p14,
  Yuva444p14LeFrame,
  Yuva444p14BeFrame,
  Yuva444p14,
  Yuva444p14Row,
  yuva444p14_to,
  yuva444p14_to_endian,
  14,
);
yuva444p_high_bit_resample_suite!(
  yuva444p16,
  Yuva444p16LeFrame,
  Yuva444p16BeFrame,
  Yuva444p16,
  Yuva444p16Row,
  yuva444p16_to,
  yuva444p16_to_endian,
  16,
);
