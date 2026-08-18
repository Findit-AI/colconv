//! Alpha-aware fused-downscale coverage for the high-bit planar GBR+alpha
//! family (`Gbrap10` / `Gbrap12` / `Gbrap14` / `Gbrap16`).
//!
//! Each `GbrapN` de-interleaves its native-depth G/B/R/A planes into the
//! canonical host-native `R, G, B, A` u16 row the high-bit packed-RGBA
//! sources stage (`gbra_to_rgba_u16_high_bit_row`) and feeds the **same**
//! 4-channel high-bit packed-RGBA resample tail
//! (`packed_rgba_u16_resample::<BITS>`) — so this suite is the planar twin
//! of `resample_packed_rgba_16bit`, asserting the identical alpha contract
//! at the source's native depth:
//! - native `rgba_u16` is the exact 2x2 native block mean (alpha averaged,
//!   not forced opaque);
//! - the u8 / narrowed outputs derive from a single `>> (BITS - 8)`
//!   narrowing of the straight color;
//! - premultiplied bins premultiplied color and un-premultiplies against
//!   the **native** max `(1 << BITS) - 1` (transparent pixels never bleed);
//! - LE and BE wire encodings produce byte-identical output.
//!
//! `GbrapNRow::new` is `pub(crate)` in `mediaframe`, so (as with the
//! `resample_gbrp` / `resample_gbr_high_bit` suites) a high-bit GBR+alpha
//! row can only reach `process` through the in-order walker; the mid-frame
//! alpha-mode-freeze / out-of-sequence rejections are covered by the
//! shared-tail `resample_packed_rgba_16bit` suite against the exact same
//! `check_frozen_alpha_mode` / `packed_rgba_u16_resample` functions.

use super::*;
use crate::{
  KernelMatrix,
  resample::AreaResampler,
  sinker::{AlphaMode, MixedSinker},
};

const SRC: usize = 8;
const OUT: usize = 4;
const MATRIX: KernelMatrix = KernelMatrix::Bt709;

/// Pseudo-random canonical host-native `R, G, B, A` plane, every sample
/// masked to `BITS` so it is a legal native code; alpha varies.
fn canonical_frame<const BITS: u32>(seed: u32) -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 4];
  pseudo_random_u16_low_n_bits(&mut buf, seed, BITS);
  buf
}

/// Scatter a canonical interleaved `R, G, B, A` u16 plane into the four
/// `(g, b, r, a)` planes a `GbrapHighBitFrame` carries.
fn planes_from_canonical(rgba: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r, mut a) = (
    std::vec![0u16; n],
    std::vec![0u16; n],
    std::vec![0u16; n],
    std::vec![0u16; n],
  );
  for i in 0..n {
    r[i] = rgba[i * 4];
    g[i] = rgba[i * 4 + 1];
    b[i] = rgba[i * 4 + 2];
    a[i] = rgba[i * 4 + 3];
  }
  (g, b, r, a)
}

/// Re-encode host-native u16 to wire byte storage per endianness, so a
/// fixture reads back identically on LE/BE hosts.
fn as_wire(host: &[u16], be: bool) -> Vec<u16> {
  host
    .iter()
    .map(|&v| if be { as_be_u16(v) } else { as_le_u16(v) })
    .collect()
}

/// Round-half-up 2x2 block mean over native u16 canonical RGBA, all four
/// channels (alpha included).
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

/// Premultiply one canonical RGBA plane in place against `max`.
fn premultiply(plane: &mut [u16], max: u32) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + max / 2) / max) as u16;
    }
  }
}

/// Un-premultiply one binned canonical RGBA plane against `max`.
fn unpremultiply(plane: &[u16], max: u32) -> Vec<u16> {
  let mut out = std::vec![0u16; plane.len()];
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

/// Drop alpha from a canonical RGBA u16 plane → packed RGB u16.
fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Narrow a canonical RGBA u16 plane to u8 via `>> shift`.
fn narrow_rgba_u8(rgba: &[u16], shift: u32) -> Vec<u8> {
  rgba.iter().map(|&v| (v >> shift) as u8).collect()
}

/// Narrow an RGB u16 plane to u8 via `>> shift`.
fn narrow_rgb_u8(rgb_u16: &[u16], shift: u32) -> Vec<u8> {
  rgb_u16.iter().map(|&v| (v >> shift) as u8).collect()
}

macro_rules! gbrap_high_bit_resample_suite {
  (
    $modname:ident,
    $marker:ident,
    $le_frame:ident,
    $be_frame:ident,
    $walk:ident,
    $walk_endian:ident,
    $gbrp_marker:ident,
    $gbrp_walk:ident,
    $bits:literal
  ) => {
    mod $modname {
      use super::*;

      const BITS: u32 = $bits;
      const MAX: u32 = (1u32 << BITS) - 1;
      const SHIFT: u32 = BITS - 8;

      fn le_frame<'a>(
        g: &'a [u16],
        b: &'a [u16],
        r: &'a [u16],
        a: &'a [u16],
        w: usize,
        h: usize,
      ) -> crate::frame::$le_frame<'a> {
        crate::frame::$le_frame::try_new(
          g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
        )
        .unwrap()
      }

      /// Native-u16 canonical RGBA of the source — a direct (identity)
      /// `GbrapN` conversion over an LE-wire frame. The oracles consume it.
      fn direct_rgba_u16(host: &[u16]) -> Vec<u16> {
        let (g, b, r, a) = planes_from_canonical(host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);
        let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
        let mut sink = MixedSinker::<crate::source::$marker<false>>::new(SRC, SRC)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
        crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        rgba_u16
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgba_u16_is_block_mean_of_direct() {
        let host = canonical_frame::<BITS>(0x51A1);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_alpha_is_averaged_not_forced_opaque() {
        let mut host = canonical_frame::<BITS>(0x9E37);
        for (i, px) in host.chunks_exact_mut(4).enumerate() {
          px[3] = ((i as u32 * 37) & MAX) as u16;
        }
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)), "block mean");
        assert!(
          rgba_u16.chunks_exact(4).any(|px| px[3] != MAX as u16),
          "resampled alpha was forced opaque — area-mean alpha lost"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_all_outputs_derive_from_binned_color() {
        // Every output flavour at once: native u16 copies the binned color,
        // the u8 / narrowed outputs derive from `>> (BITS - 8)`, and luma /
        // hsv match a direct GbrpN conversion of the binned RGB.
        let host = canonical_frame::<BITS>(0xBEEF);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

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
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
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
          .with_luma_u16(&mut lu16)
          .unwrap()
          .with_hsv(&mut h, &mut s, &mut v)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }

        let binned = block_mean_rgba(&direct_rgba_u16(&host));
        assert_eq!(rgba_u16, binned, "rgba_u16 == native block mean");
        assert_eq!(rgb_u16, drop_alpha_u16(&binned), "rgb_u16 == drop-alpha(binned)");
        assert_eq!(rgba, narrow_rgba_u8(&binned, SHIFT), "rgba == narrowed binned");
        assert_eq!(
          rgb,
          narrow_rgb_u8(&drop_alpha_u16(&binned), SHIFT),
          "rgb == narrowed drop-alpha(binned)"
        );

        // luma / luma_u16 / hsv from a direct GbrpN conversion of the binned
        // RGB (GBR->luma == RGB->luma; luma_u16 at native precision).
        let binned_rgb = drop_alpha_u16(&binned);
        let (bg, bb, br) = planes_from_packed_rgb_u16(&binned_rgb, OUT * OUT);
        let (bgw, bbw, brw) = (as_wire(&bg, false), as_wire(&bb, false), as_wire(&br, false));
        let binned_src = crate::frame::GbrpHighBitFrame::<BITS>::try_new(
          &bgw,
          &bbw,
          &brw,
          OUT as u32,
          OUT as u32,
          OUT as u32,
          OUT as u32,
          OUT as u32,
        )
        .unwrap();
        let mut luma_ref = std::vec![0u8; OUT * OUT];
        let mut lu16_ref = std::vec![0u16; OUT * OUT];
        let mut h_ref = std::vec![0u8; OUT * OUT];
        let mut s_ref = std::vec![0u8; OUT * OUT];
        let mut v_ref = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$gbrp_marker<false>>::new(OUT, OUT)
            .with_luma(&mut luma_ref)
            .unwrap()
            .with_luma_u16(&mut lu16_ref)
            .unwrap()
            .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
            .unwrap();
          crate::source::$gbrp_walk(&binned_src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(luma, luma_ref, "luma (narrowed)");
        assert_eq!(lu16, lu16_ref, "luma_u16 (native, full parity)");
        assert_eq!(h, h_ref, "hsv H");
        assert_eq!(s, s_ref, "hsv S");
        assert_eq!(v, v_ref, "hsv V");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_matches_premult_bin_unpremult_oracle() {
        let host = canonical_frame::<BITS>(0x1234);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }

        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let binned = block_mean_rgba(&pm);
        let oracle = unpremultiply(&binned, MAX);
        assert_eq!(rgba_u16, oracle, "premult rgba_u16");
        assert_eq!(rgb_u16, drop_alpha_u16(&oracle), "premult rgb_u16");
        assert_eq!(rgba, narrow_rgba_u8(&oracle, SHIFT), "premult rgba (narrowed)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_transparent_block_does_not_bleed() {
        let mut host = canonical_frame::<BITS>(0xABCD);
        let hi = MAX as u16;
        for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
          let i = off.1 * SRC + off.0;
          host[i * 4] = hi - 100;
          host[i * 4 + 1] = hi - 200;
          host[i * 4 + 2] = hi - 300;
          host[i * 4 + 3] = 0;
        }
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(&rgba_u16[..4], &[0, 0, 0, 0], "transparent block bled color");

        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let oracle = unpremultiply(&block_mean_rgba(&pm), MAX);
        assert_eq!(rgba_u16, oracle, "premult output != oracle");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn le_be_parity() {
        // The same host values through LE and BE wire frames must produce
        // byte-identical resampled output.
        let host = canonical_frame::<BITS>(0xC0DE);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);

        let render_le = || {
          let (gw, bw, rw, aw) = (
            as_wire(&g, false),
            as_wire(&b, false),
            as_wire(&r, false),
            as_wire(&a, false),
          );
          let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);
          let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
          let mut rgba = std::vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          (rgba_u16, rgba)
        };
        let render_be = || {
          let (gw, bw, rw, aw) = (
            as_wire(&g, true),
            as_wire(&b, true),
            as_wire(&r, true),
            as_wire(&a, true),
          );
          let src = crate::frame::$be_frame::try_new(
            &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
            SRC as u32,
          )
          .unwrap();
          let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
          let mut rgba = std::vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<crate::source::$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk_endian(&src, true, MATRIX, &mut sink).unwrap();
          (rgba_u16, rgba)
        };
        assert_eq!(render_le(), render_be(), "LE/BE outputs diverge");
      }

      #[test]
      fn default_alpha_mode_is_straight() {
        let sink = MixedSinker::<crate::source::$marker<false>>::new(SRC, SRC);
        assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
        assert!(sink.alpha_mode().is_straight());
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_direct() {
        let host = canonical_frame::<BITS>(0x0F0F);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, direct_rgba_u16(&host), "identity plan == direct");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cross_frame_reset_reuses_streams() {
        let host = canonical_frame::<BITS>(0x5151);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn accepts_alpha_mode_change_across_frames() {
        // begin_frame (re-run by the walker each frame) re-arms the frozen
        // mode, so frame 2 may flip to Premultiplied without a false
        // ResampleOutputsChanged.
        let host = canonical_frame::<BITS>(0xB2B2);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          sink.set_alpha_mode(AlphaMode::Premultiplied);
          crate::source::$walk(&src, true, MATRIX, &mut sink)
            .expect("a fresh frame must accept a different alpha mode");
        }
        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let oracle = unpremultiply(&block_mean_rgba(&pm), MAX);
        assert_eq!(rgba_u16, oracle, "premult frame 2 output");
      }
    }
  };
}

/// Scatter a packed-RGB u16 buffer into `(g, b, r)` planes — shared by the
/// per-format suites' luma/hsv reference (binned RGB → GbrpN).
fn planes_from_packed_rgb_u16(rgb: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r) = (std::vec![0u16; n], std::vec![0u16; n], std::vec![0u16; n]);
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

gbrap_high_bit_resample_suite!(
  gbrap10,
  Gbrap10,
  Gbrap10LeFrame,
  Gbrap10BeFrame,
  gbrap10_to,
  gbrap10_to_endian,
  Gbrp10,
  gbrp10_to,
  10
);
gbrap_high_bit_resample_suite!(
  gbrap12,
  Gbrap12,
  Gbrap12LeFrame,
  Gbrap12BeFrame,
  gbrap12_to,
  gbrap12_to_endian,
  Gbrp12,
  gbrp12_to,
  12
);
gbrap_high_bit_resample_suite!(
  gbrap14,
  Gbrap14,
  Gbrap14LeFrame,
  Gbrap14BeFrame,
  gbrap14_to,
  gbrap14_to_endian,
  Gbrp14,
  gbrp14_to,
  14
);
gbrap_high_bit_resample_suite!(
  gbrap16,
  Gbrap16,
  Gbrap16LeFrame,
  Gbrap16BeFrame,
  gbrap16_to,
  gbrap16_to_endian,
  Gbrp16,
  gbrp16_to,
  16
);

// ---- Filter-resample routing (Tier 10b -> separable filter engine) ------
//
// Straight-alpha `GbrapN` de-interleaves its native-depth G/B/R/A planes
// into the canonical host-native `R, G, B, A` u16 row and runs the SAME
// shared 4-channel high-bit packed-RGBA filter tail
// (`packed_rgba_u16_filter_resample::<BITS, true>`, the native-luma_u16 mode)
// the straight `Rgba64` source runs (it takes `<16, false>`, the narrowed
// luma_u16 mode) — so a `GbrapN` filter `rgba_u16` is byte-identical to the
// `Rgba64`
// filter `rgba_u16` of the same logical pixels, AFTER the reference is
// clamped to the native max `(1 << BITS) - 1` (the 16-bit `Rgba64` oracle
// does not clamp a signed-kernel overshoot to a sub-16-bit ceiling; the
// `GbrapN` tail does — a no-op for `Gbrap16`). PIL filters R, G, B, A
// independently with no premultiplication, so the filter path is reached
// only for straight alpha.

/// `Gbrap10` filter coverage: a feature-independent acceptance test (also
/// guards `gbr`-solo), the native-range + no-wrap overshoot contract for the
/// sub-16-bit clamp, and the `Rgba64`-parity oracle (gated on `rgb`).
mod gbrap10_filter {
  use super::*;
  use crate::resample::FilteredResampler;

  const BITS: u32 = 10;
  const NATIVE_MAX: u16 = (1 << BITS) - 1; // 1023
  const SHIFT: u32 = BITS - 8;

  fn le_frame<'a>(
    g: &'a [u16],
    b: &'a [u16],
    r: &'a [u16],
    a: &'a [u16],
    w: usize,
    h: usize,
  ) -> crate::frame::Gbrap10LeFrame<'a> {
    crate::frame::Gbrap10LeFrame::try_new(
      g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
    )
    .unwrap()
  }

  /// A `GbrapN` filter plan must be accepted and produce a real (non-
  /// sentinel) output — no `Rgba64` reference, so this also guards the
  /// `gbr`-solo build where the rgb-gated oracle below compiles out.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn filter_plan_is_accepted() {
    use crate::resample::Triangle;
    let host = canonical_frame::<BITS>(0x7A1C);
    let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
    let (gw, bw, rw, aw) = (
      as_wire(&g, false),
      as_wire(&b, false),
      as_wire(&r, false),
      as_wire(&a, false),
    );
    let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

    let mut rgba_u16 = std::vec![0xABCDu16; OUT * OUT * 4];
    {
      let mut sink =
        MixedSinker::<crate::source::Gbrap10<false>, FilteredResampler<Triangle>>::with_resampler(
          SRC,
          SRC,
          FilteredResampler::new(OUT, OUT, Triangle),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
      crate::source::gbrap10_to(&src, true, MATRIX, &mut sink)
        .expect("GbrapN filter plan must be accepted (routed)");
    }
    assert!(
      rgba_u16.iter().any(|&v| v != 0xABCD),
      "routed filter plan must write the rgba_u16 buffer"
    );
  }

  /// A sharp `0 -> native-max` horizontal step, uniform vertically, all
  /// three colour channels equal and a fully-opaque alpha. A signed kernel
  /// enlarging this overshoots above `NATIVE_MAX` on the high side.
  fn step_edge(w: usize, h: usize) -> Vec<u16> {
    let mut buf = std::vec![0u16; w * h * 4];
    for (i, px) in buf.chunks_exact_mut(4).enumerate() {
      let x = i % w;
      let v = if x >= w / 2 { NATIVE_MAX } else { 0 };
      px.copy_from_slice(&[v, v, v, NATIVE_MAX]);
    }
    buf
  }

  /// Native-range clamp + no-wrap contract for the sub-16-bit filter. A
  /// `CatmullRom` / `Lanczos3` negative lobe overshoots a near-max edge, so
  /// a finalized sample can exceed `1023` even though the `FilterStream`
  /// clamps only to the full u16 range. For `Gbrap10` that overshoot must
  /// clip to `1023` before any output is derived: native `rgba_u16` stays
  /// `<= 1023`, and the narrowed `rgba` u8 at a ceiling pixel is `1023 >> 2
  /// = 255`, never the wrapped small value an un-clamped overshoot makes.
  fn assert_clamped_and_narrowed<K>(name: &str, kernel: K)
  where
    K: crate::resample::FilterKernel + Copy,
  {
    const SW: usize = 4;
    const SD: usize = 7;
    let host = step_edge(SW, SW);
    let (g, b, r, a) = planes_from_canonical(&host, SW * SW);
    let (gw, bw, rw, aw) = (
      as_wire(&g, false),
      as_wire(&b, false),
      as_wire(&r, false),
      as_wire(&a, false),
    );
    let src = le_frame(&gw, &bw, &rw, &aw, SW, SW);

    let mut rgba_u16 = std::vec![0u16; SD * SD * 4];
    let mut rgba_u8 = std::vec![0u8; SD * SD * 4];
    {
      let mut sink =
        MixedSinker::<crate::source::Gbrap10<false>, FilteredResampler<K>>::with_resampler(
          SW,
          SW,
          FilteredResampler::new(SD, SD, kernel),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba(&mut rgba_u8)
        .unwrap();
      crate::source::gbrap10_to(&src, true, MATRIX, &mut sink).unwrap();
    }

    // (a) Every native sample (alpha included) is within the 10-bit range.
    assert!(
      rgba_u16.iter().all(|&v| v <= NATIVE_MAX),
      "{name}: rgba_u16 must stay <= {NATIVE_MAX}; max was {}",
      rgba_u16.iter().copied().max().unwrap()
    );
    // The step overshoots, so a clipped-high (== ceiling) edge must exist —
    // else the test is not exercising the overshoot it claims to.
    assert!(
      rgba_u16.contains(&NATIVE_MAX),
      "{name}: expected a clipped-high (== {NATIVE_MAX}) edge in rgba_u16"
    );
    // (b) Each narrowed u8 is the `>> 2` of the clamped native sample, so a
    // ceiling pixel narrows to 255 — never a wrap (`1062 >> 2 = 265 as u8 =
    // 9`).
    for (&hi, &lo) in rgba_u16.iter().zip(rgba_u8.iter()) {
      assert_eq!(
        lo,
        (hi >> SHIFT) as u8,
        "{name}: u8 must be the narrowing of the clamped native sample"
      );
      if hi == NATIVE_MAX {
        assert_eq!(
          lo, 255,
          "{name}: clipped-high edge must narrow to 255, not a wrap"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn catmullrom_overshoot_is_clamped_to_native_max() {
    use crate::resample::CatmullRom;
    assert_clamped_and_narrowed("catmullrom", CatmullRom);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn lanczos3_overshoot_is_clamped_to_native_max() {
    use crate::resample::Lanczos3;
    assert_clamped_and_narrowed("lanczos3", Lanczos3);
  }
}

/// Native-depth `luma_u16` contract for the 4-channel `GbrapN` filter tail:
/// attaching an alpha output (so the sink routes through the 4-channel
/// `packed_rgba_u16_filter_resample` rather than the 3-channel emit) must NOT
/// downgrade `luma_u16` from native precision to the narrowed 0..255
/// flavor. The oracle is the area path's own "native, full parity" rule
/// (`straight_all_outputs_derive_from_binned_color`): `luma_u16` equals a
/// direct `GbrpN` conversion of the drop-alpha binned RGB — applied here to
/// the FILTER's binned frame (its `rgba_u16` output). Feature-independent (no
/// `Rgba64`), so it also guards the `gbr`-solo build.
mod filter_native_luma_u16 {
  use super::*;
  use crate::resample::{CatmullRom, FilteredResampler, Lanczos3, Triangle};

  macro_rules! gbrap_filter_native_luma_tests {
    ($mod:ident, $marker:ident, $le:ident, $walk:ident, $bits:literal) => {
      mod $mod {
        use super::*;

        const BITS: u32 = $bits;
        const SHIFT: u32 = BITS - 8;

        /// Run the `GbrapN` FILTER sink with both `rgba_u16` AND `luma_u16`
        /// attached (the alpha output forces the 4-channel tail), returning
        /// `(rgba_u16, luma_u16)`.
        fn filter_rgba_and_luma_u16<K>(
          canonical: &[u16],
          ow: usize,
          oh: usize,
          kernel: K,
        ) -> (std::vec::Vec<u16>, std::vec::Vec<u16>)
        where
          K: crate::resample::FilterKernel,
        {
          let (g, b, r, a) = planes_from_canonical(canonical, SRC * SRC);
          let (gw, bw, rw, aw) = (
            as_wire(&g, false),
            as_wire(&b, false),
            as_wire(&r, false),
            as_wire(&a, false),
          );
          let src = crate::frame::$le::try_new(
            &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
            SRC as u32,
          )
          .unwrap();
          let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
          let mut luma_u16 = std::vec![0u16; ow * oh];
          {
            let mut sink =
              MixedSinker::<crate::source::$marker<false>, FilteredResampler<K>>::with_resampler(
                SRC,
                SRC,
                FilteredResampler::new(ow, oh, kernel),
              )
              .unwrap()
              .with_rgba_u16(&mut rgba_u16)
              .unwrap()
              .with_luma_u16(&mut luma_u16)
              .unwrap();
            crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          }
          (rgba_u16, luma_u16)
        }

        fn assert_native_luma_u16<K>(kernel: K, ow: usize, oh: usize, seed: u32, ctx: &str)
        where
          K: crate::resample::FilterKernel + Copy,
        {
          let canonical = canonical_frame::<BITS>(seed);
          let (rgba_u16, filter_luma_u16) = filter_rgba_and_luma_u16(&canonical, ow, oh, kernel);

          // Native-depth oracle: the exact `rgb_to_luma_u16_native_row` the
          // area / direct paths run on the binned drop-alpha RGB — applied
          // here to the FILTER's own `rgba_u16` output. (The filter already
          // clamps `rgba_u16` to the native max, so the oracle input is
          // in-range.) This is the value the area path's
          // `straight_all_outputs_derive_from_binned_color` asserts as
          // "native, full parity".
          let binned_rgb = drop_alpha_u16(&rgba_u16);
          let mut native_ref = std::vec![0u16; ow * oh];
          crate::row::rgb_to_luma_u16_native_row(
            &binned_rgb,
            &mut native_ref,
            ow * oh,
            MATRIX,
            true,
            BITS,
          );

          assert_eq!(
            filter_luma_u16, native_ref,
            "{ctx}: filter luma_u16 must be NATIVE depth (rgb_to_luma_u16_native_row of the binned RGB), not narrowed"
          );

          // Discrimination guard: the narrowed flavor (the pre-fix bug) is the
          // 8-bit Y' of the `>> SHIFT` RGB, zero-extended via
          // `rgb_to_luma_u16_row`. Assert it genuinely differs from the native
          // oracle, so this test would FAIL on the pre-fix narrowed path.
          let narrowed_rgb: std::vec::Vec<u8> =
            binned_rgb.iter().map(|&v| (v >> SHIFT) as u8).collect();
          let mut narrowed_luma = std::vec![0u16; ow * oh];
          crate::row::rgb_to_luma_u16_row(
            &narrowed_rgb,
            &mut narrowed_luma,
            ow * oh,
            MATRIX,
            true,
            true,
          );
          assert_ne!(
            native_ref, narrowed_luma,
            "{ctx}: oracle is non-discriminating (native == narrowed); pick a frame that differs"
          );
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn filter_luma_u16_is_native_depth() {
          let s = stringify!($mod).len() as u32;
          assert_native_luma_u16(Triangle, OUT, OUT, 0xC11A ^ s, "triangle");
          assert_native_luma_u16(CatmullRom, OUT, OUT, 0xC22B ^ s, "catmullrom");
          assert_native_luma_u16(Lanczos3, OUT, OUT, 0xC33C ^ s, "lanczos3");
        }
      }
    };
  }

  gbrap_filter_native_luma_tests!(gbrap10, Gbrap10, Gbrap10LeFrame, gbrap10_to, 10);
  gbrap_filter_native_luma_tests!(gbrap12, Gbrap12, Gbrap12LeFrame, gbrap12_to, 12);
  gbrap_filter_native_luma_tests!(gbrap14, Gbrap14, Gbrap14LeFrame, gbrap14_to, 14);
  gbrap_filter_native_luma_tests!(gbrap16, Gbrap16, Gbrap16LeFrame, gbrap16_to, 16);
}

/// `Rgba64`-parity oracle for both `Gbrap10` (sub-16-bit, native clamp
/// applied to the reference) and `Gbrap16` (full 16-bit, clamp a no-op): a
/// `GbrapN` filter `rgba_u16` is byte-identical to the `Rgba64` filter
/// `rgba_u16` of the same logical pixels. Gated on `rgb` (the oracle
/// source); the `gbr`-solo build relies on the acceptance test above.
#[cfg(feature = "rgb")]
mod filter_parity {
  use super::*;
  use crate::resample::{CatmullRom, FilteredResampler, Lanczos3, Triangle};

  /// Run the `Rgba64` filter sink over a canonical host-native `R, G, B, A`
  /// plane at `ow x oh`, returning native `rgba_u16`. Clamped to `native_max`
  /// per channel so a signed-kernel overshoot matches the sub-16-bit `GbrapN`
  /// tail (no-op when `native_max == u16::MAX`).
  fn rgba64_filter_rgba_u16<K>(
    canonical: &[u16],
    sw: usize,
    sh: usize,
    ow: usize,
    oh: usize,
    kernel: K,
    native_max: u16,
  ) -> Vec<u16>
  where
    K: crate::resample::FilterKernel,
  {
    let wire = as_wire(canonical, false);
    let src = crate::frame::Rgba64Frame::new(&wire, sw as u32, sh as u32, (sw * 4) as u32);
    let mut out = std::vec![0u16; ow * oh * 4];
    {
      let mut sink =
        MixedSinker::<crate::source::Rgba64<false>, FilteredResampler<K>>::with_resampler(
          sw,
          sh,
          FilteredResampler::new(ow, oh, kernel),
        )
        .unwrap()
        .with_rgba_u16(&mut out)
        .unwrap();
      crate::source::rgba64_to(&src, true, MATRIX, &mut sink).unwrap();
    }
    for v in &mut out {
      *v = (*v).min(native_max);
    }
    out
  }

  macro_rules! gbrap_filter_parity_tests {
    ($mod:ident, $marker:ident, $walker:ident, $le:ident, $bits:literal) => {
      mod $mod {
        use super::*;

        const BITS: u32 = $bits;
        const NATIVE_MAX: u16 = ((1u32 << BITS) - 1) as u16;

        /// Run the `GbrapN` filter sink over a canonical `R, G, B, A` plane
        /// (scattered to G/B/R/A planes) at `ow x oh`, returning native
        /// `rgba_u16` — the value compared against the `Rgba64` oracle.
        fn gbrap_filter_rgba_u16<K>(
          canonical: &[u16],
          sw: usize,
          sh: usize,
          ow: usize,
          oh: usize,
          kernel: K,
        ) -> Vec<u16>
        where
          K: crate::resample::FilterKernel,
        {
          let (g, b, r, a) = planes_from_canonical(canonical, sw * sh);
          let (gw, bw, rw, aw) = (
            as_wire(&g, false),
            as_wire(&b, false),
            as_wire(&r, false),
            as_wire(&a, false),
          );
          let src = crate::frame::$le::try_new(
            &gw, &bw, &rw, &aw, sw as u32, sh as u32, sw as u32, sw as u32, sw as u32, sw as u32,
          )
          .unwrap();
          let mut out = std::vec![0u16; ow * oh * 4];
          {
            let mut sink =
              MixedSinker::<crate::source::$marker<false>, FilteredResampler<K>>::with_resampler(
                sw,
                sh,
                FilteredResampler::new(ow, oh, kernel),
              )
              .unwrap()
              .with_rgba_u16(&mut out)
              .unwrap();
            crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
          }
          out
        }

        fn assert_matches<K>(kernel: K, ow: usize, oh: usize, seed: u32, ctx: &str)
        where
          K: crate::resample::FilterKernel + Copy,
        {
          let canonical = canonical_frame::<BITS>(seed);
          let got = gbrap_filter_rgba_u16(&canonical, SRC, SRC, ow, oh, kernel);
          let want = rgba64_filter_rgba_u16(&canonical, SRC, SRC, ow, oh, kernel, NATIVE_MAX);
          assert_eq!(got, want, "{ctx}: rgba_u16 vs Rgba64 filter oracle");
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn downscale_filter_rgba_u16_matches_rgba64() {
          let s = stringify!($mod).len() as u32;
          assert_matches(Triangle, OUT, OUT, 0xF11A ^ s, "triangle/down");
          assert_matches(CatmullRom, OUT, OUT, 0xF22B ^ s, "catmullrom/down");
          assert_matches(Lanczos3, OUT, OUT, 0xF33C ^ s, "lanczos3/down");
        }

        #[test]
        #[cfg_attr(
          miri,
          ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
        )]
        fn upscale_filter_rgba_u16_matches_rgba64() {
          // 8 -> 7 enlarge — exercises the filter engine's grow path.
          const UP: usize = 7;
          let s = stringify!($mod).len() as u32;
          assert_matches(Triangle, UP, UP, 0xE11A ^ s, "triangle/up");
          assert_matches(CatmullRom, UP, UP, 0xE22B ^ s, "catmullrom/up");
          assert_matches(Lanczos3, UP, UP, 0xE33C ^ s, "lanczos3/up");
        }
      }
    };
  }

  gbrap_filter_parity_tests!(gbrap10, Gbrap10, gbrap10_to, Gbrap10LeFrame, 10);
  gbrap_filter_parity_tests!(gbrap16, Gbrap16, gbrap16_to, Gbrap16LeFrame, 16);
}
