//! Alpha-aware fused-downscale coverage for the high-bit packed
//! straight/premult RGBA family (`Rgba64` / `Bgra64`).
//!
//! The 4-channel u16 tail bins canonical `R, G, B, A` at native 16-bit
//! depth so resampled alpha is a real area mean (not the forced-opaque
//! `0xFFFF` the 3-channel u16 path emitted). Straight: resampled
//! `rgba_u16` == block-mean of the direct full-res `rgba_u16`; the u8
//! `rgba` is the `>> 8` narrowing. Premultiplied: resampled `rgba_u16`
//! == un-premultiply(block-mean(premultiply(direct full-res
//! `rgba_u16`))) — the oracle mirrors the impl's exact integer ops.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{
    Bgra64, Bgra64Row, Rgba64, Rgba64Row, bgra64_to, bgra64_to_endian, rgba64_to, rgba64_to_endian,
  },
};
use mediaframe::frame::{Bgra64BeFrame, Bgra64Frame, Rgba64BeFrame, Rgba64Frame};

const SRC: usize = 8;
const OUT: usize = 4;
const MAX: u32 = 65535;

/// Re-encode a host-native u16 slice as wire byte storage for the given
/// endianness, so a fixture reads back identically on LE and BE hosts.
/// Built on the shared per-element `as_le_u16` / `as_be_u16` so those
/// helpers stay live under the `rgb` feature.
fn as_wire_u16(host: &[u16], be: bool) -> Vec<u16> {
  host
    .iter()
    .map(|&v| {
      if be {
        super::as_be_u16(v)
      } else {
        super::as_le_u16(v)
      }
    })
    .collect()
}

/// Pseudo-random full-range u16 host values (incl. varying alpha) so
/// every binned channel sees real averaging and a forced-opaque alpha
/// path diverges. Local LCG so the helper compiles under the `rgb`
/// feature alone (the shared `pseudo_random_u16_low_n_bits` is gated to
/// other families).
fn host_frame(seed: u32) -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 4];
  let mut state = seed;
  for b in &mut buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u16;
  }
  buf
}

/// Round-half-up 2x2 block mean of a canonical native-u16 RGBA plane.
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

fn premultiply(plane: &mut [u16]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + MAX / 2) / MAX) as u16;
    }
  }
}

fn unpremultiply(plane: &[u16]) -> Vec<u16> {
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

fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  let mut rgb = std::vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in rgb.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  rgb
}

fn narrow_rgba_u8(rgba: &[u16]) -> Vec<u8> {
  rgba.iter().map(|&v| (v >> 8) as u8).collect()
}

fn narrow_rgb_u8(rgb_u16: &[u16]) -> Vec<u8> {
  rgb_u16.iter().map(|&v| (v >> 8) as u8).collect()
}

macro_rules! rgba16_resample_suite {
  ($modname:ident, $marker:ident, $row:ident, $walk:ident, $walk_endian:ident, $frame:ident, $beframe:ident) => {
    mod $modname {
      use super::*;

      /// Direct full-resolution canonical native-u16 RGBA conversion at
      /// source width (identity plan), from an LE-wire fixture.
      fn direct_rgba_u16(host: &[u16]) -> Vec<u16> {
        let wire = as_wire_u16(host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);
        let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
        let mut sink = MixedSinker::<$marker<false>>::new(SRC, SRC)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
        $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        rgba_u16
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgba_u16_is_block_mean_of_direct() {
        let host = host_frame(0x51A1 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_alpha_is_averaged_not_forced_opaque() {
        let mut host = host_frame(0x9E37 ^ stringify!($modname).len() as u32);
        for (i, px) in host.chunks_exact_mut(4).enumerate() {
          px[3] = (i as u16).wrapping_mul(901); // varied, mostly != 0xFFFF
        }
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
        assert!(
          rgba_u16.chunks_exact(4).any(|px| px[3] != 0xFFFF),
          "alpha was forced opaque — the bug is back"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_all_outputs_derive_from_binned_color() {
        // Every attached output — native u16, narrowed u8, luma, hsv —
        // must match a direct full-res sink over the binned native RGBA.
        let host = host_frame(0xBEEF ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);

        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut luma_u16 = std::vec![0u16; OUT * OUT];
        let mut h = std::vec![0u8; OUT * OUT];
        let mut s_ = std::vec![0u8; OUT * OUT];
        let mut v_ = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
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
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_hsv(&mut h, &mut s_, &mut v_)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let binned = block_mean_rgba(&direct_rgba_u16(&host));
        assert_eq!(rgba_u16, binned, "rgba_u16");
        assert_eq!(rgb_u16, drop_alpha_u16(&binned), "rgb_u16");
        assert_eq!(rgba, narrow_rgba_u8(&binned), "rgba (narrowed)");
        assert_eq!(rgb, narrow_rgb_u8(&drop_alpha_u16(&binned)), "rgb (narrowed)");

        // luma / luma_u16 / hsv: direct full-res sink over the binned
        // native RGB (a `Rgb48` LE fixture).
        let binned_rgb = drop_alpha_u16(&binned);
        let binned_rgb_wire = as_wire_u16(&binned_rgb, false);
        let rgb_src = mediaframe::frame::Rgb48Frame::new(
          &binned_rgb_wire,
          OUT as u32,
          OUT as u32,
          (OUT * 3) as u32,
        );
        let mut ref_luma = std::vec![0u8; OUT * OUT];
        let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
        let mut ref_h = std::vec![0u8; OUT * OUT];
        let mut ref_s = std::vec![0u8; OUT * OUT];
        let mut ref_v = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::Rgb48>::new(OUT, OUT)
            .with_luma(&mut ref_luma)
            .unwrap()
            .with_luma_u16(&mut ref_luma_u16)
            .unwrap()
            .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
            .unwrap();
          crate::source::rgb48_to(&rgb_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(luma, ref_luma, "luma");
        assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
        assert_eq!(h, ref_h, "h");
        assert_eq!(s_, ref_s, "s");
        assert_eq!(v_, ref_v, "v");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_matches_premult_bin_unpremult_oracle() {
        let host = host_frame(0x1234 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm);
        let binned = block_mean_rgba(&pm);
        let oracle = unpremultiply(&binned);
        assert_eq!(rgba_u16, oracle, "premult rgba_u16");
        assert_eq!(rgb_u16, drop_alpha_u16(&oracle), "premult rgb_u16");
        assert_eq!(rgba, narrow_rgba_u8(&oracle), "premult rgba (narrowed)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_transparent_block_does_not_bleed() {
        let mut host = host_frame(0xABCD ^ stringify!($modname).len() as u32);
        for &(sx, sy) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
          let p = (sy * SRC + sx) * 4;
          host[p] = 60000;
          host[p + 1] = 50000;
          host[p + 2] = 40000;
          host[p + 3] = 0;
        }
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(&rgba_u16[..4], &[0, 0, 0, 0], "transparent block bled color");
        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm);
        assert_eq!(rgba_u16, unpremultiply(&block_mean_rgba(&pm)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn le_be_parity() {
        // Same logical values, LE vs BE wire, must produce identical
        // host-native binned output.
        let host = host_frame(0xC0DE ^ stringify!($modname).len() as u32);

        let wire_le = as_wire_u16(&host, false);
        let src_le = $frame::new(&wire_le, SRC as u32, SRC as u32, (SRC * 4) as u32);
        let mut out_le = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut out_le)
          .unwrap();
          $walk(&src_le, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let wire_be = as_wire_u16(&host, true);
        let src_be = $beframe::new(&wire_be, SRC as u32, SRC as u32, (SRC * 4) as u32);
        let mut out_be = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut out_be)
          .unwrap();
          $walk_endian(&src_be, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(out_le, out_be, "LE/BE parity");
      }

      #[test]
      fn default_alpha_mode_is_straight() {
        let sink = MixedSinker::<$marker<false>>::new(SRC, SRC);
        assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_direct() {
        let host = host_frame(0x0F0F ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);
        let mut via_area = std::vec![0u16; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgba_u16(&mut via_area)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(via_area, direct_rgba_u16(&host));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cross_frame_reset_reuses_streams() {
        let host = host_frame(0x5151 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let src = $frame::new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32);
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_sequence_first_row_rejected_before_allocation() {
        let host = host_frame(0x6262 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let row3 = &wire[3 * SRC * 4..4 * SRC * 4];
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let err = sink
          .process($row::new(row3, 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(
          err,
          MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
        ));
        // The out-of-sequence first row is rejected before the 4-channel u16 area
        // stream is built — nothing is committed for a row that never ran.
        assert!(
          !sink.rgba_stream_u16_allocated(),
          "stream allocated for a rejected out-of-sequence row"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_output_change() {
        let host = host_frame(0x7373 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_luma(&mut luma).unwrap();
        let err = sink
          .process($row::new(&wire[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
        assert!(luma.iter().all(|&b| b == 0));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejected_first_row_does_not_poison_output_retry() {
        let host = host_frame(0x8484 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let row3 = &wire[3 * SRC * 4..4 * SRC * 4];
        let err = sink
          .process($row::new(row3, 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(
          err,
          MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
        ));
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        sink.set_rgb_u16(&mut rgb_u16).unwrap();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .expect("row 0 must succeed after a rejected out-of-sequence first row");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_alpha_mode_change() {
        // The alpha mode is frozen at the first resampled row; flipping it
        // mid-frame would mix straight and premultiplied rows in one
        // stream, so it is rejected deterministically rather than silently
        // producing output that matches neither mode.
        let host = host_frame(0x9595 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&wire[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged on mid-frame alpha-mode flip, got {err:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_alpha_mode_route_switch() {
        // Premultiplied rgb-only routes through the 4-channel tail; a flip
        // to Straight would reroute a later row to the 3-channel tail and
        // bypass the per-tail freeze. The pre-route check rejects it. A
        // horizontal-only resample (height unchanged) emits per input row
        // — the case most exposed to a stale-stream overwrite.
        let host = host_frame(0xA1A1 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgb = std::vec![0u8; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, SRC),
        )
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Straight);
        let err = sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged on mid-frame alpha-mode route-switch, got {err:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn accepts_alpha_mode_change_across_frames() {
        // begin_frame resets the frozen alpha mode, so a fresh frame may
        // use a different mode without a false ResampleOutputsChanged.
        let host = host_frame(0xB2B2 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for y in 0..SRC {
          sink
            .process($row::new(&wire[y * SRC * 4..(y + 1) * SRC * 4], y, KernelMatrix::Bt709, true))
            .unwrap();
        }
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for y in 0..SRC {
          sink
            .process($row::new(&wire[y * SRC * 4..(y + 1) * SRC * 4], y, KernelMatrix::Bt709, true))
            .expect("a fresh frame must accept a different alpha mode");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_straight_to_premultiplied_route_switch() {
        // Straight rgb_u16-only routes through the 3-channel tail, which the
        // sink-level freeze still arms; a flip to Premultiplied (which would
        // reroute to the 4-channel tail) is rejected.
        let host = host_frame(0xC3C3 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgb_u16 = std::vec![0u16; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, SRC),
        )
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged on straight-to-premultiplied route-switch, got {err:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_route_switch_after_out_of_sequence_first_row() {
        // The mode snapshot is taken at begin_frame, so an out-of-sequence
        // first row and its retry do not disturb it: a later
        // Straight->Premultiplied flip is still rejected before any second
        // stream feed. rgb_u16-only, horizontal-only resample.
        let host = host_frame(0xD4D4 ^ stringify!($modname).len() as u32);
        let wire = as_wire_u16(&host, false);
        let mut rgb_u16 = std::vec![0u16; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker<false>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, SRC),
        )
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&wire[3 * SRC * 4..4 * SRC * 4], 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        sink
          .process($row::new(&wire[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&wire[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged after OOS first row + route-switch, got {err:?}"
        );
      }
    }
  };
}

rgba16_resample_suite!(
  rgba64,
  Rgba64,
  Rgba64Row,
  rgba64_to,
  rgba64_to_endian,
  Rgba64Frame,
  Rgba64BeFrame
);
rgba16_resample_suite!(
  bgra64,
  Bgra64,
  Bgra64Row,
  bgra64_to,
  bgra64_to_endian,
  Bgra64Frame,
  Bgra64BeFrame
);

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba64_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = host_frame(0x9151);
  let wire = as_wire_u16(&host, false);
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgba64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot 4-channel u16 SCRATCH grow failpoint: the row-0 area stream
    // box builds OK (into a local), then the source_rgba_u16_scratch grow — which runs
    // AFTER the stream build and BEFORE the field insert — refuses, surfacing
    // AllocationFailed with the rgba_stream_u16 field `None` and the freeze
    // uncommitted (the commit-together atomic shape). This exercises the exact
    // scratch-after-stream ordering boundary a box failure cannot pin.
    crate::sinker::mixed::arm_source_rgba_u16_scratch_failure();
    let err = sink
      .process(Rgba64Row::new(
        &wire[..SRC * 4],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the 4-channel u16 area
    // stream field must be `None` — built into a LOCAL and inserted only after the
    // scratch grows, so a failed scratch leaves nothing committed. The insert-before-
    // scratch shape would leave it `Some` here, so this assertion FAILS against that
    // shape and PASSES against the fix (a box failure cannot make this distinction).
    assert!(
      !sink.rgba_stream_u16_allocated(),
      "a scratch OOM must leave the rgba_stream_u16 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgba64Row::new(
          &wire[r * SRC * 4..(r + 1) * SRC * 4],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgba_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real rgba_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}
