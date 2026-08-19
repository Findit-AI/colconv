//! Per-channel filter-resample equivalence for the high-bit packed
//! straight RGBA `u16` family (`Rgba64` / `Bgra64`).
//!
//! The 4-channel u16 filter tail bins canonical `R, G, B, A` at native
//! 16-bit depth, filtering each channel independently with **no
//! premultiplication** — exactly PIL's RGBA `resize` semantics. So each
//! output channel of a real-alpha filter resample must equal, byte for
//! byte, the single-plane [`FilterStream<u16>`] resample of that channel
//! through the *same* merged engine. No PIL golden is needed: the engine's
//! own `u16_matches_pil` proves the single-plane filter is PIL-exact per
//! channel, and this file proves the 4-channel interleaved filter reduces
//! to that per channel.
//!
//! `Rgba64` / `Bgra64` are 16 bits per channel (native max `65535`), so the
//! `FilterStream<u16>` `0..=65535` clamp *is* the native clamp — a signed
//! kernel's overshoot is clipped to the same ceiling in both the 4-channel
//! and the single-plane runs, so the equivalence is an exact equality with
//! no extra native-depth clamp.

use crate::{
  KernelMatrix, PixelSink,
  resample::{
    CatmullRom, FilterStream, FilteredResampler, Lanczos3, ResampleError, Resampler, Triangle,
  },
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Bgra64, Bgra64Row, Rgba64, Rgba64Row, bgra64_to, rgba64_to},
};
use mediaframe::frame::{Bgra64Frame, Rgba64Frame};

/// Host-native pseudo-random full-range u16 values (incl. varying alpha) so
/// every channel — alpha included — sees real filtering. Local LCG so the
/// helper compiles under the `rgb` feature alone.
fn host_rgba(seed: u32, len: usize) -> Vec<u16> {
  let mut buf = std::vec![0u16; len];
  let mut state = seed;
  for b in &mut buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u16;
  }
  buf
}

/// Re-encode a host-native u16 slice as **LE-wire** byte storage so a
/// fixture reads back identically on LE (no-op) and BE (byte-swap) hosts.
fn as_le_wire(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|&v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Single-channel byte-exact filter resample of channel `c` of a canonical
/// `R, G, B, A` u16 plane, via the merged engine's `FilterStream<u16>`
/// (channels = 1) — the per-channel oracle. The u16 filter is byte-exact
/// versus Pillow per channel; PIL resizes RGBA by filtering each channel
/// independently with no premultiplication, so a 4-channel RGBA filter
/// resample's channel `c` must equal this single-plane filter.
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter<K: crate::resample::FilterKernel>(
  kernel: K,
  canonical: &[u16],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let mut plane = std::vec![0u16; sw * sh];
  for (dst, px) in plane.iter_mut().zip(canonical.chunks_exact(4)) {
    *dst = px[c];
  }
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u16>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = std::vec![0u16; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Asserts a 4-channel native-u16 RGBA filter resample (interleaved
/// `R, G, B, A`) equals the per-channel single-plane filter of the
/// canonical source — i.e. each of R/G/B/A independently matches the
/// byte-exact engine.
#[allow(clippy::too_many_arguments)]
fn assert_rgba_u16_is_per_channel_filter<K: crate::resample::FilterKernel + Copy>(
  kernel: K,
  resampled_rgba_u16: &[u16],
  canonical: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) {
  for c in 0..4 {
    let plane = channel_plane_filter(kernel, canonical, c, sw, sh, ow, oh);
    for (i, &p) in plane.iter().enumerate() {
      assert_eq!(
        resampled_rgba_u16[i * 4 + c],
        p,
        "{ctx} channel {c} px {i}: rgba_u16 {} vs per-plane filter {p}",
        resampled_rgba_u16[i * 4 + c],
      );
    }
  }
}

macro_rules! rgba_u16_filter_suite {
  ($modname:ident, $marker:ident, $row:ident, $walk:ident, $frame:ident, $perm:expr) => {
    mod $modname {
      use super::*;

      /// Canonical-channel → source-byte permutation: `src[k] =
      /// canonical[PERM[k]]` (`canonical` is `R, G, B, A`). `Rgba64` is the
      /// identity; `Bgra64` swaps R↔B. Alpha is always slot 3 (both formats
      /// trail alpha), so it is never permuted.
      const PERM: [usize; 4] = $perm;

      /// Encode a canonical `R, G, B, A` host-native u16 plane into this
      /// format's host-native source layout.
      fn encode(canonical: &[u16]) -> Vec<u16> {
        let mut src = std::vec![0u16; canonical.len()];
        for (s, c) in src.chunks_exact_mut(4).zip(canonical.chunks_exact(4)) {
          for k in 0..4 {
            s[k] = c[PERM[k]];
          }
        }
        src
      }

      /// Run the real-alpha filter sink (rgba_u16 attached) over a canonical
      /// source at `ow` x `oh` under `kernel`, returning native `rgba_u16`.
      fn filter_rgba_u16<K: crate::resample::FilterKernel + Copy>(
        kernel: K,
        canonical: &[u16],
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
      ) -> Vec<u16> {
        let wire = as_le_wire(&encode(canonical));
        let src = $frame::new(&wire, sw as u32, sh as u32, (sw * 4) as u32);
        let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<K>>::with_resampler(
            sw,
            sh,
            FilteredResampler::new(ow, oh, kernel),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
        }
        rgba_u16
      }

      fn per_channel<K: crate::resample::FilterKernel + Copy>(
        kernel: K,
        sw: usize,
        sh: usize,
        ow: usize,
        oh: usize,
        seed: u32,
        ctx: &str,
      ) {
        let canonical = host_rgba(seed, sw * sh * 4);
        let rgba_u16 = filter_rgba_u16(kernel, &canonical, sw, sh, ow, oh);
        assert_rgba_u16_is_per_channel_filter(kernel, &rgba_u16, &canonical, sw, sh, ow, oh, ctx);
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_real_alpha_is_byte_exact_per_channel_downscale() {
        let s = stringify!($modname).len() as u32;
        per_channel(Triangle, 8, 8, 4, 4, 0xF11A ^ s, "Triangle 8x8->4x4");
        per_channel(CatmullRom, 8, 8, 4, 4, 0xF22B ^ s, "CatmullRom 8x8->4x4");
        per_channel(Lanczos3, 8, 8, 4, 4, 0xF33C ^ s, "Lanczos3 8x8->4x4");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_real_alpha_is_byte_exact_per_channel_upscale() {
        let s = stringify!($modname).len() as u32;
        per_channel(Triangle, 4, 4, 7, 7, 0xE11A ^ s, "Triangle 4x4->7x7");
        per_channel(CatmullRom, 4, 4, 7, 7, 0xE22B ^ s, "CatmullRom 4x4->7x7");
        per_channel(Lanczos3, 4, 4, 7, 7, 0xE33C ^ s, "Lanczos3 4x4->7x7");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_rgb_u16_is_drop_alpha_of_filtered_rgba_u16() {
        // With both rgb_u16 and rgba_u16 attached under a filter plan,
        // rgb_u16 must be the alpha-dropped filtered RGBA (the same color
        // the rgba_u16 output carries) — proving the drop-alpha derivation
        // rides the 4-channel filter, not a separately-filtered RGB.
        let s = stringify!($modname).len() as u32;
        let canonical = host_rgba(0xD0DA ^ s, 8 * 8 * 4);
        let wire = as_le_wire(&encode(&canonical));
        let src = $frame::new(&wire, 8, 8, 32);
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        let mut rgb_u16 = std::vec![0u16; 4 * 4 * 3];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
        }
        let drop_alpha: Vec<u16> = rgba_u16
          .chunks_exact(4)
          .flat_map(|px| px[..3].iter().copied())
          .collect();
        assert_eq!(rgb_u16, drop_alpha, "rgb_u16 is drop-alpha of filtered rgba_u16");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_u8_rgba_is_narrowing_of_native_filtered_rgba() {
        // The narrowed u8 `rgba` rides the same 4-channel filter: each u8 is
        // the `>> 8` of the native-u16 filtered RGBA. (Alpha narrows too —
        // it is a real filtered channel, never forced opaque.)
        let s = stringify!($modname).len() as u32;
        let canonical = host_rgba(0xC0DE ^ s, 8 * 8 * 4);
        let wire = as_le_wire(&encode(&canonical));
        let src = $frame::new(&wire, 8, 8, 32);
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        let mut rgba = std::vec![0u8; 4 * 4 * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
        }
        let narrowed: Vec<u8> = rgba_u16.iter().map(|&v| (v >> 8) as u8).collect();
        assert_eq!(rgba, narrowed, "u8 rgba is the >>8 narrowing of the filtered rgba_u16");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_filter_is_unsupported() {
        // Premultiplied alpha has no filter analogue (the filter engine
        // cannot un-premultiply), so a filter plan under Premultiplied
        // surfaces the typed `UnsupportedFilter` rather than emitting
        // straight-filtered premultiplied color.
        let canonical = host_rgba(0x9595 ^ stringify!($modname).len() as u32, 8 * 8 * 4);
        let wire = as_le_wire(&encode(&canonical));
        let src = $frame::new(&wire, 8, 8, 32);
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
          8,
          8,
          FilteredResampler::new(4, 4, Triangle),
        )
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        let err = $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
          ),
          "expected UnsupportedFilter for a premultiplied filter plan, got {err:?}"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_plan_is_accepted_not_unsupported() {
        // Regression: a filter plan on a real-alpha packed-RGBA u16 source is
        // accepted (rgba_u16 attached routes the 4-channel filter; an
        // rgb_u16-only sink routes the 3-channel filter).
        let canonical = host_rgba(0xC0DE ^ stringify!($modname).len() as u32, 8 * 8 * 4);
        let wire = as_le_wire(&encode(&canonical));
        let src = $frame::new(&wire, 8, 8, 32);
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709))
            .expect("a filter plan must be accepted for real-alpha packed RGBA u16");
        }
        let mut rgb_u16 = std::vec![0u16; 4 * 4 * 3];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709))
            .expect("a filter plan must be accepted for an rgb_u16-only sink too");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn filter_out_of_sequence_first_row_rejected() {
        // Atomicity: the per-kind filter stream's sequence check raises
        // OutOfSequenceRow before any scratch staging or allocation.
        let canonical = host_rgba(0x6262 ^ stringify!($modname).len() as u32, 8 * 8 * 4);
        let wire = as_le_wire(&encode(&canonical));
        let row3 = &wire[3 * 8 * 4..4 * 8 * 4];
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
          8,
          8,
          FilteredResampler::new(4, 4, Triangle),
        )
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
        sink.begin_frame(8, 8).unwrap();
        let err = sink
          .process($row::for_tests(row3, 3, KernelMatrix::Bt709, true))
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
      fn filter_cross_frame_reset_reuses_stream() {
        // begin_frame resets the filter stream, so a second frame reproduces.
        let s = stringify!($modname).len() as u32;
        let canonical = host_rgba(0x5151 ^ s, 8 * 8 * 4);
        let first = filter_rgba_u16(Triangle, &canonical, 8, 8, 4, 4);

        let wire = as_le_wire(&encode(&canonical));
        let src = $frame::new(&wire, 8, 8, 32);
        let mut rgba_u16 = std::vec![0u16; 4 * 4 * 4];
        {
          let mut sink = MixedSinker::<$marker<false>, FilteredResampler<Triangle>>::with_resampler(
            8,
            8,
            FilteredResampler::new(4, 4, Triangle),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
          $walk(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
        }
        assert_eq!(rgba_u16, first, "second frame must reproduce after reset");
      }
    }
  };
}

// PERM[k] = which canonical channel (R=0,G=1,B=2,A=3) source byte k holds.
// Rgba64: R,G,B,A (identity). Bgra64: B,G,R,A (R↔B swap; alpha stays slot 3).
rgba_u16_filter_suite!(
  rgba64,
  Rgba64,
  Rgba64Row,
  rgba64_to,
  Rgba64Frame,
  [0, 1, 2, 3]
);
rgba_u16_filter_suite!(
  bgra64,
  Bgra64,
  Bgra64Row,
  bgra64_to,
  Bgra64Frame,
  [2, 1, 0, 3]
);
