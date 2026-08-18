//! Alpha-aware fused-downscale coverage for the packed straight/premult
//! RGBA 8-bit family (`Rgba` / `Bgra` / `Argb` / `Abgr`).
//!
//! The 4-channel tail bins canonical `R, G, B, A` so resampled alpha is
//! a real area mean — the forced-opaque-`0xFF` bug the 3-channel RGB
//! path hit. Straight mode: resampled `rgba` == 2x2 block-mean of the
//! direct full-res `rgba`. Premultiplied mode: resampled `rgba` ==
//! un-premultiply(block-mean(premultiply(direct full-res `rgba`))) —
//! the oracle mirrors the impl's exact integer ops, so it is byte-exact.

use crate::{
  KernelMatrix, PixelSink,
  resample::{
    AreaResampler, CatmullRom, FilterStream, FilteredResampler, Lanczos3, ResampleError, Resampler,
    Triangle,
  },
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{
    Abgr, AbgrRow, Argb, ArgbRow, Bgra, BgraRow, Rgba, RgbaRow, abgr_to, argb_to, bgra_to, rgba_to,
  },
};
use mediaframe::frame::{AbgrFrame, ArgbFrame, BgraFrame, RgbaFrame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Pseudo-random source bytes (incl. varying alpha) so every binned
/// channel — alpha included — sees real averaging, and a forced-opaque
/// alpha path would diverge immediately.
fn packed_frame(seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC * 4];
  super::pseudo_random_u8(&mut buf, seed);
  buf
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane (every
/// channel, alpha included) — the contract for integer-ratio area
/// downscale.
fn block_mean_rgba(src: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT * 4];
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

/// Premultiply one canonical RGBA plane in place — `round(c * a / 255)`
/// per color channel, alpha untouched. Mirrors the impl's exact op.
fn premultiply(plane: &mut [u8]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + 127) / 255) as u8;
    }
  }
}

/// Un-premultiply one binned canonical RGBA plane — `round(c' * 255 /
/// a)` clamped to 255, color 0 when `a == 0`; alpha copied. Mirrors the
/// impl's exact op.
fn unpremultiply(plane: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; plane.len()];
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

/// Drop alpha from a canonical RGBA plane to packed RGB.
fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut rgb = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in rgb.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  rgb
}

/// Single-channel byte-exact filter resample of channel `c` of a
/// canonical `R, G, B, A` plane, via the merged engine's `FilterStream`
/// (channels = 1). This is the per-channel PIL oracle: the u8 filter is
/// byte-exact versus Pillow per channel (proven by the engine's
/// `u8_matches_pil`), and PIL resizes RGBA by filtering each channel
/// independently with **no premultiplication** — so a 4-channel RGBA
/// filter resample's channel `c` must equal this single-plane filter.
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter<K: crate::resample::FilterKernel>(
  kernel: K,
  canonical: &[u8],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  use_simd: bool,
) -> Vec<u8> {
  let mut plane = std::vec![0u8; sw * sh];
  for (dst, px) in plane.iter_mut().zip(canonical.chunks_exact(4)) {
    *dst = px[c];
  }
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u8>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = std::vec![0u8; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], use_simd, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Asserts a 4-channel RGBA filter resample (interleaved `R, G, B, A`)
/// equals the per-channel single-plane filter of the canonical source —
/// i.e. each of R/G/B/A independently matches the byte-exact engine.
#[allow(clippy::too_many_arguments)]
fn assert_rgba_is_per_channel_filter<K: crate::resample::FilterKernel + Copy>(
  kernel: K,
  resampled_rgba: &[u8],
  canonical: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) {
  for c in 0..4 {
    let plane = channel_plane_filter(kernel, canonical, c, sw, sh, ow, oh, true);
    for (i, &p) in plane.iter().enumerate() {
      assert_eq!(
        resampled_rgba[i * 4 + c],
        p,
        "{ctx} channel {c} px {i}: rgba {} vs per-plane filter {p}",
        resampled_rgba[i * 4 + c],
      );
    }
  }
}

macro_rules! rgba8_resample_suite {
  ($modname:ident, $marker:ident, $row:ident, $walk:ident, $frame:ident, $perm:expr) => {
    mod $modname {
      use super::*;

      /// Canonical-channel → source-byte permutation: `src[k] =
      /// canonical[PERM[k]]` (`canonical` is `R, G, B, A`). Lets tests
      /// build a fixture in canonical space (where slot 3 is always
      /// alpha) and emit the format's wire layout, so alpha-position
      /// differences (`Argb` / `Abgr` lead with α) stay correct.
      const PERM: [usize; 4] = $perm;

      /// Encode a canonical `R, G, B, A` plane into this format's source
      /// bytes.
      fn encode(canonical: &[u8]) -> Vec<u8> {
        let mut src = std::vec![0u8; canonical.len()];
        for (s, c) in src.chunks_exact_mut(4).zip(canonical.chunks_exact(4)) {
          for k in 0..4 {
            s[k] = c[PERM[k]];
          }
        }
        src
      }

      /// Direct full-resolution canonical RGBA conversion at source
      /// width (identity plan) — the per-format `with_rgba` reference the
      /// straight oracle bins and the premult oracle premultiplies.
      fn direct_rgba(pix: &[u8]) -> Vec<u8> {
        let src = $frame::try_new(pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
        let mut rgba = std::vec![0u8; SRC * SRC * 4];
        let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
          .with_rgba(&mut rgba)
          .unwrap();
        $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        rgba
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgba_is_block_mean_of_direct_rgba() {
        let pix = packed_frame(0x51A1 ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(rgba, block_mean_rgba(&direct_rgba(&pix)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_alpha_is_averaged_not_forced_opaque() {
        // Canonical alpha ramps across the byte range; the binned alpha
        // must be the area mean, never a constant 0xFF (the bug this whole
        // tail fixes). Built in canonical space then encoded so α lands at
        // the format's true wire position.
        let mut canonical = packed_frame(0x9E37 ^ stringify!($modname).len() as u32);
        for (i, px) in canonical.chunks_exact_mut(4).enumerate() {
          px[3] = (i as u8).wrapping_mul(3);
        }
        let pix = encode(&canonical);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        // direct_rgba canonicalizes, so its alpha is `canonical`'s alpha.
        assert_eq!(rgba, block_mean_rgba(&direct_rgba(&pix)));
        // Counterexample guard: a forced-opaque path would have set every
        // alpha to 0xFF — prove at least one output alpha differs.
        assert!(
          rgba.chunks_exact(4).any(|px| px[3] != 0xFF),
          "alpha was forced opaque — the bug is back"
        );
        // The output alphas are exactly the canonical-alpha area means.
        for (oy, ox) in (0..OUT).flat_map(|y| (0..OUT).map(move |x| (y, x))) {
          let mut acc = 0u32;
          for dy in 0..2 {
            for dx in 0..2 {
              acc += canonical[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + 3] as u32;
            }
          }
          assert_eq!(rgba[(oy * OUT + ox) * 4 + 3], ((acc + 2) / 4) as u8, "alpha mean");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgb_luma_hsv_derive_from_binned_color() {
        let pix = packed_frame(0xBEEF ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut h = std::vec![0u8; OUT * OUT];
        let mut s_ = std::vec![0u8; OUT * OUT];
        let mut v_ = std::vec![0u8; OUT * OUT];
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
          .with_luma(&mut luma)
          .unwrap()
          .with_hsv(&mut h, &mut s_, &mut v_)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        // rgb == drop-alpha of the binned (straight) color.
        let binned = block_mean_rgba(&direct_rgba(&pix));
        assert_eq!(rgb, drop_alpha(&binned), "rgb");

        // luma / hsv == direct full-res sink over the binned RGB.
        let binned_rgb = drop_alpha(&binned);
        let rgb_src = mediaframe::frame::Rgb24Frame::new(
          &binned_rgb,
          OUT as u32,
          OUT as u32,
          (OUT * 3) as u32,
        );
        let mut ref_luma = std::vec![0u8; OUT * OUT];
        let mut ref_h = std::vec![0u8; OUT * OUT];
        let mut ref_s = std::vec![0u8; OUT * OUT];
        let mut ref_v = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::Rgb24>::new(OUT, OUT)
            .with_luma(&mut ref_luma)
            .unwrap()
            .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
            .unwrap();
          crate::source::rgb24_to(&rgb_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(luma, ref_luma, "luma");
        assert_eq!(h, ref_h, "h");
        assert_eq!(s_, ref_s, "s");
        assert_eq!(v_, ref_v, "v");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_rgba_matches_premult_bin_unpremult_oracle() {
        let pix = packed_frame(0x1234 ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut rgb = std::vec![0u8; OUT * OUT * 3];
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
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let mut pm = direct_rgba(&pix);
        premultiply(&mut pm);
        let binned = block_mean_rgba(&pm);
        let oracle = unpremultiply(&binned);
        assert_eq!(rgba, oracle, "premult rgba");
        assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_transparent_block_does_not_bleed() {
        // A 2x2 canonical block where every pixel is transparent (A=0)
        // but carries a loud stored color. Premult binning zeroes the
        // premultiplied color; binned-A is 0; un-premultiply yields RGB 0
        // — no bleed. Encoded so α lands at the format's wire position.
        let mut canonical = packed_frame(0xABCD ^ stringify!($modname).len() as u32);
        for &(sx, sy) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
          let p = (sy * SRC + sx) * 4;
          canonical[p] = 250;
          canonical[p + 1] = 240;
          canonical[p + 2] = 230;
          canonical[p + 3] = 0; // fully transparent
        }
        let pix = encode(&canonical);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        // The fully-transparent block: A=0, and RGB must be 0 (no bleed).
        assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled color");
        // And it matches the oracle everywhere.
        let mut pm = direct_rgba(&pix);
        premultiply(&mut pm);
        let oracle = unpremultiply(&block_mean_rgba(&pm));
        assert_eq!(rgba, oracle);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_and_premult_differ_under_varying_alpha() {
        // Guard that the mode flag actually changes behaviour: with
        // non-trivial alpha the two oracles produce different RGB.
        let mut canonical = packed_frame(0x77AA ^ stringify!($modname).len() as u32);
        for (i, px) in canonical.chunks_exact_mut(4).enumerate() {
          px[3] = 16u8.wrapping_add((i as u8).wrapping_mul(5));
        }
        let pix = encode(&canonical);
        let render = |mode: AlphaMode| {
          let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
          let mut rgba = std::vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(mode)
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
        assert!(sink.alpha_mode().is_straight());
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_direct_rgba() {
        let pix = packed_frame(0x0F0F ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let mut via_area = std::vec![0u8; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgba(&mut via_area)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(via_area, direct_rgba(&pix));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn no_output_sink_is_a_noop() {
        let pix = packed_frame(0x4242 ^ stringify!($modname).len() as u32);
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        // A no-output sink neither sequences nor allocates — feeding any
        // row index (even out of order) is a legal no-op.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let row3 = &pix[3 * SRC * 4..4 * SRC * 4];
        sink.process($row::new(row3, 3, KernelMatrix::Bt709, true)).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cross_frame_reset_reuses_streams() {
        let pix = packed_frame(0x5151 ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

        let first;
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
          // Capture row 0's output, then run a second frame through the
          // same sink — begin_frame must reset the stream so the result
          // reproduces.
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        first = block_mean_rgba(&direct_rgba(&pix));
        assert_eq!(rgba, first);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn out_of_sequence_first_row_rejected() {
        let pix = packed_frame(0x6262 ^ stringify!($modname).len() as u32);
        let row3 = &pix[3 * SRC * 4..4 * SRC * 4];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let err = sink
          .process($row::new(row3, 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(
          err,
          MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
        ));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_mid_frame_output_change() {
        let pix = packed_frame(0x7373 ^ stringify!($modname).len() as u32);
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
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
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_luma(&mut luma).unwrap();
        let err = sink
          .process($row::new(&pix[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
        assert!(luma.iter().all(|&b| b == 0), "rejected row mutated new output");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejected_first_row_does_not_poison_output_retry() {
        // A rejected out-of-sequence first row must store no frozen-output
        // snapshot, so retrying row 0 after reconfiguring the output set
        // succeeds instead of tripping ResampleOutputsChanged.
        let pix = packed_frame(0x8484 ^ stringify!($modname).len() as u32);
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let row3 = &pix[3 * SRC * 4..4 * SRC * 4];
        let err = sink
          .process($row::new(row3, 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(matches!(
          err,
          MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
        ));
        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        sink.set_rgb(&mut rgb).unwrap();
        sink
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
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
        let pix = packed_frame(0x9595 ^ stringify!($modname).len() as u32);
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
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
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&pix[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
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
        let pix = packed_frame(0xA1A1 ^ stringify!($modname).len() as u32);
        let mut rgb = std::vec![0u8; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
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
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Straight);
        let err = sink
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
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
        let pix = packed_frame(0xB2B2 ^ stringify!($modname).len() as u32);
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
        // Frame 1 under the default Straight, fully processed.
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for y in 0..SRC {
          sink
            .process($row::new(&pix[y * SRC * 4..(y + 1) * SRC * 4], y, KernelMatrix::Bt709, true))
            .unwrap();
        }
        // Frame 2 under Premultiplied must be accepted (the freeze re-arms).
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        for y in 0..SRC {
          sink
            .process($row::new(&pix[y * SRC * 4..(y + 1) * SRC * 4], y, KernelMatrix::Bt709, true))
            .expect("a fresh frame must accept a different alpha mode");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn rejects_straight_to_premultiplied_route_switch() {
        // Straight rgb-only routes through the 3-channel tail, which the
        // sink-level freeze still arms; a flip to Premultiplied (which would
        // reroute to the 4-channel tail) is rejected.
        let pix = packed_frame(0xC3C3 ^ stringify!($modname).len() as u32);
        let mut rgb = std::vec![0u8; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, SRC),
        )
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        sink
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
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
        // first row and its retry do not disturb it (unlike a first-row
        // proxy): a later Straight->Premultiplied flip is still rejected
        // before any second stream feed. Horizontal-only resample.
        let pix = packed_frame(0xD4D4 ^ stringify!($modname).len() as u32);
        let mut rgb = std::vec![0u8; OUT * SRC * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, SRC),
        )
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        // Out-of-sequence first row — rejected, snapshot undisturbed.
        sink
          .process($row::new(&pix[3 * SRC * 4..4 * SRC * 4], 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        // Retry row 0 under the (snapshotted) Straight mode — accepted.
        sink
          .process($row::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
          .unwrap();
        // Flip to Premultiplied, then feed row 1 — rejected before any feed.
        sink.set_alpha_mode(AlphaMode::Premultiplied);
        let err = sink
          .process($row::new(&pix[SRC * 4..2 * SRC * 4], 1, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
          "expected ResampleOutputsChanged after OOS first row + route-switch, got {err:?}"
        );
      }

      /// Runs a real-alpha filter resample (rgba attached) and asserts each
      /// of R/G/B/A equals the byte-exact single-plane filter — proving the
      /// 4-channel filter is PIL-byte-exact and channel-independent (no
      /// premultiplication). Swept over Triangle/CatmullRom/Lanczos3 for a
      /// downscale and an enlarge.
      fn filter_rgba_per_channel<K: crate::resample::FilterKernel + Copy>(
        kernel: K,
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        seed: u32,
        ctx: &str,
      ) {
        let canonical = {
          let mut buf = std::vec![0u8; sw * sh * 4];
          crate::sinker::mixed::tests::pseudo_random_u8(&mut buf, seed);
          buf
        };
        let pix = encode(&canonical);
        let src = $frame::try_new(&pix, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
        let mut rgba = std::vec![0u8; ow * oh * 4];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<K>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, kernel),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_rgba_is_per_channel_filter(kernel, &rgba, &canonical, sw, sh, ow, oh, ctx);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_real_alpha_is_byte_exact_per_channel_downscale() {
        let s = stringify!($modname).len() as u32;
        filter_rgba_per_channel(Triangle, 8, 8, 4, 4, 0xF11A ^ s, "Triangle 8x8->4x4");
        filter_rgba_per_channel(CatmullRom, 8, 8, 4, 4, 0xF22B ^ s, "CatmullRom 8x8->4x4");
        filter_rgba_per_channel(Lanczos3, 8, 8, 4, 4, 0xF33C ^ s, "Lanczos3 8x8->4x4");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_real_alpha_is_byte_exact_per_channel_upscale() {
        let s = stringify!($modname).len() as u32;
        filter_rgba_per_channel(Triangle, 4, 4, 8, 8, 0xE11A ^ s, "Triangle 4x4->8x8");
        filter_rgba_per_channel(CatmullRom, 4, 4, 8, 8, 0xE22B ^ s, "CatmullRom 4x4->8x8");
        filter_rgba_per_channel(Lanczos3, 4, 4, 8, 8, 0xE33C ^ s, "Lanczos3 4x4->8x8");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_real_alpha_rgb_is_drop_alpha_of_filtered_rgba() {
        // With both rgb and rgba attached under a filter plan, rgb must be
        // the alpha-dropped filtered RGBA (the same color the rgba output
        // carries) — proving the drop-alpha derivation rides the 4-channel
        // filter, not a separately-filtered RGB.
        let s = stringify!($modname).len() as u32;
        let canonical = {
          let mut buf = std::vec![0u8; 8 * 8 * 4];
          crate::sinker::mixed::tests::pseudo_random_u8(&mut buf, 0xD0DA ^ s);
          buf
        };
        let pix = encode(&canonical);
        let src = $frame::try_new(&pix, 8, 8, 32).unwrap();
        let mut rgba = std::vec![0u8; 4 * 4 * 4];
        let mut rgb = std::vec![0u8; 4 * 4 * 3];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(rgb, drop_alpha(&rgba), "rgb is drop-alpha of filtered rgba");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_plan_is_accepted_not_unsupported() {
        // Regression: the 4-channel filter fence is gone — a filter plan on
        // a real-alpha packed-RGBA source is accepted (rgba attached routes
        // the 4-channel filter; rgb-only routes the 3-channel filter).
        let pix = packed_frame(0xC0DE ^ stringify!($modname).len() as u32);
        let src = $frame::try_new(&pix, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink)
            .expect("a filter plan must be accepted for real-alpha packed RGBA");
        }
        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, FilteredResampler<Triangle>>::with_resampler(
            SRC,
            SRC,
            FilteredResampler::new(OUT, OUT, Triangle),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
          $walk(&src, true, KernelMatrix::Bt709, &mut sink)
            .expect("a filter plan must be accepted for an rgb-only sink too");
        }
      }
    }
  };
}

#[test]
fn alpha_mode_str_and_predicates() {
  assert_eq!(AlphaMode::Straight.as_str(), "straight");
  assert_eq!(AlphaMode::Premultiplied.as_str(), "premultiplied");
  assert!(AlphaMode::Straight.is_straight());
  assert!(AlphaMode::Premultiplied.is_premultiplied());
  assert_eq!(AlphaMode::default(), AlphaMode::Straight);
}

// PERM[k] = which canonical channel (R=0,G=1,B=2,A=3) the source byte k
// holds. Rgba: R,G,B,A. Bgra: B,G,R,A. Argb: A,R,G,B. Abgr: A,B,G,R.
rgba8_resample_suite!(rgba, Rgba, RgbaRow, rgba_to, RgbaFrame, [0, 1, 2, 3]);
rgba8_resample_suite!(bgra, Bgra, BgraRow, bgra_to, BgraFrame, [2, 1, 0, 3]);
rgba8_resample_suite!(argb, Argb, ArgbRow, argb_to, ArgbFrame, [3, 0, 1, 2]);
rgba8_resample_suite!(abgr, Abgr, AbgrRow, abgr_to, AbgrFrame, [3, 2, 1, 0]);

/// RFC #238 tail-collapse atomicity (4-channel): the in-place-upgraded
/// `packed_rgba_resample` commits the output-set freeze only AFTER the
/// (fallible) 4-channel area-stream box allocation succeeds. A first-row
/// stream-alloc refusal returns the typed `AllocationFailed` leaving
/// `resample_outputs` UNCOMMITTED (the output untouched), so a row-0 retry with a
/// CHANGED output set (hsv added) is ACCEPTED — not mis-rejected as
/// `ResampleOutputsChanged` — and drives real output. The pre-collapse split
/// (freeze-before-alloc) would have frozen `{rgba}` and rejected the retry.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_first_row_stream_alloc_failure_leaves_freeze_uncommitted_for_retry() {
  let pix = packed_frame(0xA70A);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgba, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot box-alloc failpoint: the row-0 4-channel area-stream box
    // alloc refuses, surfacing the recoverable AllocationFailed with the freeze
    // uncommitted (the helper freezes only after the alloc succeeds).
    crate::resample::arm_box_failure();
    let err = sink
      .process(RgbaRow::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row 4-channel stream box-alloc refusal must surface AllocationFailed, got {err:?}"
    );
    // The box alloc fails before the field insert, so the stream field stays None.
    assert!(
      !sink.rgba_stream_allocated(),
      "a box-alloc-refused first row must leave the rgba_stream field None"
    );
    // The freeze was never committed — attaching hsv (a CHANGED output set) and
    // replaying the frame from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_hsv(&mut h, &mut s, &mut v).unwrap();
    for r in 0..SRC {
      sink
        .process(RgbaRow::new(
          &pix[r * SRC * 4..(r + 1) * SRC * 4],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row 4-channel stream-alloc OOM must succeed");
    }
  }
  assert!(
    rgba.iter().any(|&b| b != 0),
    "the recovered frame produced real RGBA output (box-alloc refusal was recoverable)"
  );
}

/// RFC #238 tail-collapse atomicity (4-channel, scratch OOM): confirms the same
/// commit-together shape in `packed_rgba_resample` — the RGBA stream builds into a
/// LOCAL, then the drop-alpha RGB scratch (`source_rgb_scratch`, needed here
/// because `hsv` is attached) grows before the stream field insert + freeze. A
/// scratch OOM AFTER the successful stream build leaves the stream field `None`
/// AND `resample_outputs` uncommitted, so a row-0 retry with a CHANGED output set
/// (luma added) is ACCEPTED and drives real output.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgba_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let pix = packed_frame(0x5C0A);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgba, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    // Attach hsv so the drop-alpha RGB scratch (`source_rgb_scratch`) is grown.
    sink.set_hsv(&mut h, &mut s, &mut v).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB-scratch grow failpoint: the row-0 RGBA stream
    // box builds OK, then the drop-alpha RGB scratch grow refuses — surfacing
    // AllocationFailed with BOTH the stream field `None` and the freeze
    // uncommitted (the commit-together atomic shape).
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(RgbaRow::new(&pix[..SRC * 4], 0, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row 4-channel drop-alpha scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the 4-channel stream
    // field must be `None` — the fix builds it into a LOCAL and inserts only after
    // the scratch grows. The R1-buggy shape (insert BEFORE the scratch) would leave
    // it `Some` here, so this FAILS against the buggy shape and PASSES against the
    // fix.
    assert!(
      !sink.rgba_stream_allocated(),
      "a scratch OOM must leave the rgba_stream field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(RgbaRow::new(
          &pix[r * SRC * 4..(r + 1) * SRC * 4],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row 4-channel scratch OOM must succeed");
    }
  }
  assert!(
    rgba.iter().any(|&b| b != 0),
    "the recovered frame produced real RGBA output (scratch OOM was recoverable)"
  );
}
