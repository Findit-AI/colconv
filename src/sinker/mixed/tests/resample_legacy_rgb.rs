//! Fused-downscale coverage for the legacy 16-bit packed-RGB family
//! (`Rgb565` / `Bgr565` / `Rgb555` / `Bgr555` / `Rgb444` / `Bgr444`).
//!
//! These sources bin at **native** bit depth: each packed pixel is
//! unpacked to its native R/G/B channels (5/6/5, 5/5/5 or 4/4/4 values,
//! NOT bit-expanded), the area mean is taken over those native channels
//! (half-up), the binned channels are re-packed into the source's packed
//! word, and every output is the **exact** direct kernel run over that
//! re-packed binned frame. So the oracle for every output is a direct
//! conversion of the area-downscaled source-format frame — the same
//! single-binned-frame contract `Rgb48` / `X2Rgb10` / `Gbrp16` follow.
//!
//! Native-depth binning differs observably from binning the
//! **bit-expanded** u8 stream and re-narrowing: the latter loses low-bit
//! detail. See [`counterexample_*`] below, which constructs a block the
//! two approaches disagree on and asserts the native-depth result.
//!
//! Everything here builds under `--features "std rgb-legacy"` alone: the
//! oracle re-packs the native-binned channels by hand and feeds the
//! direct legacy-RGB sink — it does **not** depend on the `rgb` feature's
//! `Rgb24` / `rgb24_to` / `Rgb24Frame`.

use super::*;
use crate::resample::{AreaResampler, ResampleError};

// ---- Per-format bit layout (canonical R, G, B native channels) ----------
//
// `unpack` mirrors each format's `*_to_rgb_u16_row` (native bits, no
// expansion); `pack` is its inverse, building the source's packed
// little-endian word. The fused path bins the `unpack`ed channels and
// re-packs with `pack`, so these define the native-depth oracle.

#[derive(Clone, Copy)]
struct Layout {
  /// (shift, mask) for the R, G, B channels in the source word.
  fields: [(u32, u16); 3],
}

impl Layout {
  const RGB565: Layout = Layout {
    fields: [(11, 0x1F), (5, 0x3F), (0, 0x1F)],
  };
  const BGR565: Layout = Layout {
    fields: [(0, 0x1F), (5, 0x3F), (11, 0x1F)],
  };
  const RGB555: Layout = Layout {
    fields: [(10, 0x1F), (5, 0x1F), (0, 0x1F)],
  };
  const BGR555: Layout = Layout {
    fields: [(0, 0x1F), (5, 0x1F), (10, 0x1F)],
  };
  const RGB444: Layout = Layout {
    fields: [(8, 0x0F), (4, 0x0F), (0, 0x0F)],
  };
  const BGR444: Layout = Layout {
    fields: [(0, 0x0F), (4, 0x0F), (8, 0x0F)],
  };

  /// Native R, G, B channels of a packed source word.
  fn unpack(&self, px: u16) -> [u16; 3] {
    [
      (px >> self.fields[0].0) & self.fields[0].1,
      (px >> self.fields[1].0) & self.fields[1].1,
      (px >> self.fields[2].0) & self.fields[2].1,
    ]
  }

  /// Re-pack native R, G, B channels into the source word.
  fn pack(&self, rgb: [u16; 3]) -> u16 {
    ((rgb[0] & self.fields[0].1) << self.fields[0].0)
      | ((rgb[1] & self.fields[1].1) << self.fields[1].0)
      | ((rgb[2] & self.fields[2].1) << self.fields[2].0)
  }
}

/// LE-wire byte storage for a packed-`u16` frame.
fn pack_frame(words: &[u16]) -> std::vec::Vec<u8> {
  let mut buf = std::vec::Vec::with_capacity(words.len() * 2);
  for w in words {
    buf.extend_from_slice(&w.to_le_bytes());
  }
  buf
}

/// Exact native-depth 2x2 block mean (round-half-up) of a packed
/// source-word frame: unpack each contributing pixel to native channels,
/// average per channel, re-pack into one binned source word per output
/// pixel. This is the source-format frame the direct path converts.
fn binned_frame_2x2(
  src: &[u16],
  src_w: usize,
  out_w: usize,
  out_h: usize,
  layout: Layout,
) -> std::vec::Vec<u16> {
  let mut out = std::vec![0u16; out_w * out_h];
  for oy in 0..out_h {
    for ox in 0..out_w {
      let mut acc = [0u64; 3];
      for dy in 0..2 {
        for dx in 0..2 {
          let px = src[(oy * 2 + dy) * src_w + ox * 2 + dx];
          let ch = layout.unpack(px);
          for c in 0..3 {
            acc[c] += ch[c] as u64;
          }
        }
      }
      let binned = [
        ((acc[0] + 2) / 4) as u16,
        ((acc[1] + 2) / 4) as u16,
        ((acc[2] + 2) / 4) as u16,
      ];
      out[oy * out_w + ox] = layout.pack(binned);
    }
  }
  out
}

/// Pseudo-random packed source words masked to the format's used bits.
fn random_words(n: usize, seed: u32, layout: Layout) -> std::vec::Vec<u16> {
  let used = (layout.fields[0].1 << layout.fields[0].0)
    | (layout.fields[1].1 << layout.fields[1].0)
    | (layout.fields[2].1 << layout.fields[2].0);
  let mut state = seed;
  let mut out = std::vec![0u16; n];
  for w in &mut out {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *w = ((state >> 8) as u16) & used;
  }
  out
}

// ---- Per-format generated tests -----------------------------------------

macro_rules! legacy_resample_tests {
  (
    mod_name:  $mod_name:ident,
    marker:    $marker:ident,
    row_ty:    $row_ty:ident,
    frame:     $frame:ident,
    walker:    $walker:ident,
    layout:    $layout:expr,
  ) => {
    mod $mod_name {
      use super::*;

      const SRC: usize = 8;
      const OUT: usize = 4;

      /// rgb_u16 is the exact native-depth 2x2 area mean of the source's
      /// native channels.
      #[test]
      fn rgb_u16_is_native_area_mean() {
        let words = random_words(SRC * SRC, 0x1234_5678, $layout);
        let wire = pack_frame(&words);
        let src = $frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let binned = binned_frame_2x2(&words, SRC, OUT, OUT, $layout);
        for oy in 0..OUT {
          for ox in 0..OUT {
            let want = $layout.unpack(binned[oy * OUT + ox]);
            for c in 0..3 {
              assert_eq!(
                rgb_u16[(oy * OUT + ox) * 3 + c],
                want[c],
                "({ox},{oy}) c{c}"
              );
            }
          }
        }
      }

      /// Every attached output equals a direct conversion of the
      /// native-binned source-format frame (NOT an RGB888-binned one).
      #[test]
      fn all_outputs_match_direct_of_binned_frame() {
        let words = random_words(SRC * SRC, 0x0BAD_F00D, $layout);
        let wire = pack_frame(&words);
        let src = $frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut luma_u16 = std::vec![0u16; OUT * OUT];
        let mut hh = std::vec![0u8; OUT * OUT];
        let mut ss = std::vec![0u8; OUT * OUT];
        let mut vv = std::vec![0u8; OUT * OUT];
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
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut luma_u16)
          .unwrap()
          .with_hsv(&mut hh, &mut ss, &mut vv)
          .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        // Oracle: the direct sink over the native-binned source frame.
        let binned = binned_frame_2x2(&words, SRC, OUT, OUT, $layout);
        let binned_wire = pack_frame(&binned);
        let bsrc = $frame::try_new(&binned_wire, OUT as u32, OUT as u32, (OUT * 2) as u32).unwrap();

        let mut r_rgb = std::vec![0u8; OUT * OUT * 3];
        let mut r_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut r_rgba = std::vec![0u8; OUT * OUT * 4];
        let mut r_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut r_luma = std::vec![0u8; OUT * OUT];
        let mut r_luma_u16 = std::vec![0u16; OUT * OUT];
        let mut r_h = std::vec![0u8; OUT * OUT];
        let mut r_s = std::vec![0u8; OUT * OUT];
        let mut r_v = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker>::new(OUT, OUT)
            .with_rgb(&mut r_rgb)
            .unwrap()
            .with_rgb_u16(&mut r_rgb_u16)
            .unwrap()
            .with_rgba(&mut r_rgba)
            .unwrap()
            .with_rgba_u16(&mut r_rgba_u16)
            .unwrap()
            .with_luma(&mut r_luma)
            .unwrap()
            .with_luma_u16(&mut r_luma_u16)
            .unwrap()
            .with_hsv(&mut r_h, &mut r_s, &mut r_v)
            .unwrap();
          $walker(&bsrc, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        assert_eq!(rgb, r_rgb, "rgb");
        assert_eq!(rgb_u16, r_rgb_u16, "rgb_u16");
        assert_eq!(rgba, r_rgba, "rgba");
        assert_eq!(rgba_u16, r_rgba_u16, "rgba_u16");
        assert_eq!(luma, r_luma, "luma");
        assert_eq!(luma_u16, r_luma_u16, "luma_u16");
        assert_eq!(hh, r_h, "h");
        assert_eq!(ss, r_s, "s");
        assert_eq!(vv, r_v, "v");
      }

      /// The identity plan (output geometry == source) matches the
      /// non-resampling `new` sink byte-for-byte.
      #[test]
      fn identity_plan_matches_new() {
        let words = random_words(SRC * SRC, 0xFEED_FACE, $layout);
        let wire = pack_frame(&words);
        let src = $frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut direct = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC)
            .with_rgb_u16(&mut direct)
            .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        let mut via_area = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgb_u16(&mut via_area)
          .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(direct, via_area);
      }

      /// A sink with no attached output is a legal no-op: it neither
      /// errors nor writes, and never allocates the stream/scratches.
      #[test]
      fn no_output_is_noop() {
        let words = random_words(SRC * SRC, 0x5151_5151, $layout);
        let wire = pack_frame(&words);
        let src = $frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        assert!(!sink.rgb_stream_allocated(), "stream allocated for no-op");
        assert_eq!(sink.legacy_rgb_native_scratch_capacity(), 0);
        assert_eq!(sink.legacy_rgb_packed_scratch_capacity(), 0);
      }

      /// The out-width u8 RGB staging row is sized only when a u8-RGB-
      /// derived output (rgb / luma / luma_u16 / hsv) is attached and
      /// `rgb` is absent: a native-`u16`-only sink never grows it; a luma
      /// sink does.
      #[test]
      fn u16_only_sink_does_not_size_rgb_stage() {
        let words = random_words(SRC * SRC, 0x2468_ACE0, $layout);
        let wire = pack_frame(&words);
        let src = $frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        assert_eq!(
          sink.rgb_scratch_capacity(),
          0,
          "native-u16-only sink grew the u8 RGB stage"
        );
        // The re-pack scratch is still sized — the rgb_u16 kernel reads it.
        assert!(sink.legacy_rgb_packed_scratch_capacity() >= 2 * OUT);

        // Positive control: a luma sink (no `rgb`) sizes the u8 RGB stage.
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut sink2 = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
        $walker(&src, true, KernelMatrix::Bt709, &mut sink2).unwrap();
        assert!(
          sink2.rgb_scratch_capacity() >= 3 * OUT,
          "luma sink did not size the u8 RGB stage"
        );
      }

      /// `begin_frame` resets the stream so a second frame bins cleanly:
      /// two frames through one sink match two fresh sinks.
      #[test]
      fn cross_frame_reset() {
        let words_a = random_words(SRC * SRC, 0x1111_2222, $layout);
        let words_b = random_words(SRC * SRC, 0x3333_4444, $layout);
        let wire_a = pack_frame(&words_a);
        let wire_b = pack_frame(&words_b);
        let fa = $frame::try_new(&wire_a, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();
        let fb = $frame::try_new(&wire_b, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

        let mut reused = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut reused)
          .unwrap();
          $walker(&fa, true, KernelMatrix::Bt709, &mut sink).unwrap();
          $walker(&fb, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        let mut fresh = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut fresh)
          .unwrap();
          $walker(&fb, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(reused, fresh, "second frame must match a fresh sink");
      }

      /// An out-of-sequence first row is rejected before the stream or
      /// the staging scratches are allocated.
      #[test]
      fn out_of_sequence_first_row_rejected_before_alloc() {
        let words = random_words(SRC * SRC, 0xABCD_1234, $layout);
        let wire = pack_frame(&words);
        let row3 = &wire[3 * SRC * 2..4 * SRC * 2];

        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        sink.begin_frame(SRC as u32, SRC as u32).unwrap();
        let err = sink
          .process($row_ty::new(row3, 3, KernelMatrix::Bt709, true))
          .unwrap_err();
        assert!(
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
          ),
          "expected OutOfSequenceRow, got {err:?}"
        );
        assert!(!sink.rgb_stream_allocated(), "stream allocated for rejected row");
        assert_eq!(
          sink.legacy_rgb_native_scratch_capacity(),
          0,
          "native scratch grown for rejected row"
        );
        assert_eq!(
          sink.legacy_rgb_packed_scratch_capacity(),
          0,
          "packed scratch grown for rejected row"
        );
      }

      /// Changing the attached output set mid-frame is rejected.
      #[test]
      fn mid_frame_output_change_rejected() {
        let words = random_words(SRC * SRC, 0x7777_8888, $layout);
        let wire = pack_frame(&words);

        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut luma = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<$marker, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap();
          sink.begin_frame(SRC as u32, SRC as u32).unwrap();
          sink
            .process($row_ty::new(&wire[..SRC * 2], 0, KernelMatrix::Bt709, true))
            .unwrap();
          sink.set_luma(&mut luma).unwrap();
          let err = sink
            .process($row_ty::new(
              &wire[SRC * 2..SRC * 4],
              1,
              KernelMatrix::Bt709,
              true,
            ))
            .unwrap_err();
          assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
        }
        assert!(luma.iter().all(|&l| l == 0), "rejected output must stay unwritten");
      }
    }
  };
}

legacy_resample_tests! {
  mod_name: rgb565,
  marker:   Rgb565,
  row_ty:   Rgb565Row,
  frame:    Rgb565Frame,
  walker:   rgb565_to,
  layout:   Layout::RGB565,
}
legacy_resample_tests! {
  mod_name: bgr565,
  marker:   Bgr565,
  row_ty:   Bgr565Row,
  frame:    Bgr565Frame,
  walker:   bgr565_to,
  layout:   Layout::BGR565,
}
legacy_resample_tests! {
  mod_name: rgb555,
  marker:   Rgb555,
  row_ty:   Rgb555Row,
  frame:    Rgb555Frame,
  walker:   rgb555_to,
  layout:   Layout::RGB555,
}
legacy_resample_tests! {
  mod_name: bgr555,
  marker:   Bgr555,
  row_ty:   Bgr555Row,
  frame:    Bgr555Frame,
  walker:   bgr555_to,
  layout:   Layout::BGR555,
}
legacy_resample_tests! {
  mod_name: rgb444,
  marker:   Rgb444,
  row_ty:   Rgb444Row,
  frame:    Rgb444Frame,
  walker:   rgb444_to,
  layout:   Layout::RGB444,
}
legacy_resample_tests! {
  mod_name: bgr444,
  marker:   Bgr444,
  row_ty:   Bgr444Row,
  frame:    Bgr444Frame,
  walker:   bgr444_to,
  layout:   Layout::BGR444,
}

/// Ratio-independence: for a **non-integer** area ratio (7 -> 3 on each
/// axis) every output still equals a direct conversion of the binned
/// source-format frame. The binned frame is read back from the fused
/// `rgb_u16` output (native channels == binned channels) and re-packed —
/// the `Rgb48` oracle pattern — so this holds for any ratio the area
/// stream plans, not just integer block means.
#[test]
fn rgb565_non_integer_ratio_all_outputs_match_direct_of_binned() {
  const SRC: usize = 7;
  const OUT: usize = 3;
  let words = random_words(SRC * SRC, 0xC0FF_EE11, Layout::RGB565);
  let wire = pack_frame(&words);
  let src = Rgb565Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb565, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // The binned source frame == the native channels the fused `rgb_u16`
  // already holds, re-packed into source words.
  let binned: std::vec::Vec<u16> = rgb_u16
    .chunks_exact(3)
    .map(|c| Layout::RGB565.pack([c[0], c[1], c[2]]))
    .collect();
  let binned_wire = pack_frame(&binned);
  let bsrc = Rgb565Frame::try_new(&binned_wire, OUT as u32, OUT as u32, (OUT * 2) as u32).unwrap();

  let mut r_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut r_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut r_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut r_luma = std::vec![0u8; OUT * OUT];
  let mut r_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut r_h = std::vec![0u8; OUT * OUT];
  let mut r_s = std::vec![0u8; OUT * OUT];
  let mut r_v = std::vec![0u8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Rgb565>::new(OUT, OUT)
      .with_rgb(&mut r_rgb)
      .unwrap()
      .with_rgba(&mut r_rgba)
      .unwrap()
      .with_rgba_u16(&mut r_rgba_u16)
      .unwrap()
      .with_luma(&mut r_luma)
      .unwrap()
      .with_luma_u16(&mut r_luma_u16)
      .unwrap()
      .with_hsv(&mut r_h, &mut r_s, &mut r_v)
      .unwrap();
    rgb565_to(&bsrc, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb, r_rgb, "rgb");
  assert_eq!(rgba, r_rgba, "rgba");
  assert_eq!(rgba_u16, r_rgba_u16, "rgba_u16");
  assert_eq!(luma, r_luma, "luma");
  assert_eq!(luma_u16, r_luma_u16, "luma_u16");
  assert_eq!(hh, r_h, "h");
  assert_eq!(ss, r_s, "s");
  assert_eq!(vv, r_v, "v");
}

// ---- Counterexample: native-depth vs RGB888-space binning ---------------
//
// The block R5 = [0, 1, 0, 1] area-means to 1 at NATIVE depth
// (round-half-up of 2/4); binning the bit-expanded [0, 8, 0, 8] means to
// 4, then `>> 3` = 0 — losing the low bit. The fused path must produce
// the native-depth result (1).

#[test]
fn counterexample_rgb565_r_low_bit_survives() {
  // 2x2 block, R5 channel = [0, 1, 0, 1]; G6 / B5 = 0.
  let r5 = [0u16, 1, 0, 1];
  let words: std::vec::Vec<u16> = r5.iter().map(|&r| Layout::RGB565.pack([r, 0, 0])).collect();
  let wire = pack_frame(&words);
  let src = Rgb565Frame::try_new(&wire, 2, 2, 4).unwrap();

  let mut rgb_u16 = std::vec![0u16; 3];
  {
    let mut sink =
      MixedSinker::<Rgb565, AreaResampler>::with_resampler(2, 2, AreaResampler::to(1, 1))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(
    rgb_u16[0], 1,
    "native-depth area mean of R5 [0,1,0,1] must round to 1, not the \
     RGB888-space 0"
  );

  // And the u8 RGB output is the direct expansion of the native-binned R5
  // = 1 (i.e. `(1 << 3) | (1 >> 2)` = 8), NOT the RGB888-space mean (4).
  let mut rgb = std::vec![0u8; 3];
  {
    let mut sink =
      MixedSinker::<Rgb565, AreaResampler>::with_resampler(2, 2, AreaResampler::to(1, 1))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  // `rgb565_to_rgb_row` expands R5=1 via `(1 << 3) | (1 >> 2)` = 8;
  // binning in expanded space would have produced the mean 4 instead.
  assert_eq!(rgb[0], 8, "u8 R must expand the native bin (8), not 4");
}

/// Same low-bit-survival check for a 4-bit channel (RGB444): R4 =
/// [0, 1, 1, 0] area-means to 1 at native depth (round-half-up of 2/4).
/// Binning in expanded space ([0, 17, 17, 0], mean 9) would not reproduce
/// the native signal; the native path yields 1.
#[test]
fn counterexample_rgb444_r_low_bit_survives() {
  let r4 = [0u16, 1, 1, 0];
  let words: std::vec::Vec<u16> = r4.iter().map(|&r| Layout::RGB444.pack([r, 0, 0])).collect();
  let wire = pack_frame(&words);
  let src = Rgb444Frame::try_new(&wire, 2, 2, 4).unwrap();

  let mut rgb_u16 = std::vec![0u16; 3];
  {
    let mut sink =
      MixedSinker::<Rgb444, AreaResampler>::with_resampler(2, 2, AreaResampler::to(1, 1))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb_u16[0], 1, "native-depth 4-bit area mean must survive");
}

// =========================================================================
// Legacy bit-packed (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1) fused-downscale coverage
// (Rgb8 / Bgr8 / Rgb4Byte / Bgr4Byte — 1 byte/pixel;
//  Rgb4 / Bgr4 — 4 bits/pixel, two pixels per byte).
//
// Same native-depth area-binning contract as the 16-bit family: each packed
// pixel is unpacked to its native R/G/B channels, the 2x2 area mean is taken
// over those native channels (round-half-up), and the binned channels are
// re-packed into the source's byte/nibble layout — so `rgb_u16` is exactly
// that native area mean and the rest are the direct kernels over the re-packed
// binned frame.
// =========================================================================

#[derive(Clone, Copy)]
struct LowbitLayout {
  /// (shift, mask) for the R, G, B channels in the packed byte/nibble.
  fields: [(u32, u8); 3],
  /// 4-bpp two-pixels-per-byte (even pixel = high nibble) when set.
  nibble: bool,
}

impl LowbitLayout {
  const RGB8: LowbitLayout = LowbitLayout {
    fields: [(5, 0x07), (2, 0x07), (0, 0x03)],
    nibble: false,
  };
  const BGR8: LowbitLayout = LowbitLayout {
    fields: [(0, 0x07), (3, 0x07), (6, 0x03)],
    nibble: false,
  };
  const RGB4BYTE: LowbitLayout = LowbitLayout {
    fields: [(3, 0x01), (1, 0x03), (0, 0x01)],
    nibble: false,
  };
  const BGR4BYTE: LowbitLayout = LowbitLayout {
    fields: [(0, 0x01), (1, 0x03), (3, 0x01)],
    nibble: false,
  };
  const RGB4: LowbitLayout = LowbitLayout {
    fields: [(3, 0x01), (1, 0x03), (0, 0x01)],
    nibble: true,
  };
  const BGR4: LowbitLayout = LowbitLayout {
    fields: [(0, 0x01), (1, 0x03), (3, 0x01)],
    nibble: true,
  };

  fn unpack(&self, px: u8) -> [u8; 3] {
    [
      (px >> self.fields[0].0) & self.fields[0].1,
      (px >> self.fields[1].0) & self.fields[1].1,
      (px >> self.fields[2].0) & self.fields[2].1,
    ]
  }

  fn pack(&self, ch: [u8; 3]) -> u8 {
    ((ch[0] & self.fields[0].1) << self.fields[0].0)
      | ((ch[1] & self.fields[1].1) << self.fields[1].0)
      | ((ch[2] & self.fields[2].1) << self.fields[2].0)
  }

  /// Bytes per packed row of `width` pixels.
  fn row_bytes(&self, width: usize) -> usize {
    if self.nibble {
      width.div_ceil(2)
    } else {
      width
    }
  }

  /// Read the packed byte/nibble of pixel `(x, y)`.
  fn read(&self, plane: &[u8], width: usize, x: usize, y: usize) -> u8 {
    let stride = self.row_bytes(width);
    if self.nibble {
      let byte = plane[y * stride + (x >> 1)];
      if x & 1 == 0 { byte >> 4 } else { byte & 0x0F }
    } else {
      plane[y * width + x]
    }
  }
}

/// Pseudo-random packed source plane (`row_bytes(width) * height` bytes).
fn random_lowbit_plane(
  layout: LowbitLayout,
  width: usize,
  height: usize,
  seed: u32,
) -> std::vec::Vec<u8> {
  let n = layout.row_bytes(width) * height;
  let mut state = seed;
  (0..n)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state >> 16) as u8
    })
    .collect()
}

/// Native-depth 2x2 area mean (round-half-up) re-packed into the source's
/// byte/nibble layout — the binned source-format frame the direct path
/// converts. Returns one packed value per output pixel.
fn binned_lowbit_2x2(
  layout: LowbitLayout,
  src: &[u8],
  src_w: usize,
  out_w: usize,
  out_h: usize,
) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; out_w * out_h];
  for oy in 0..out_h {
    for ox in 0..out_w {
      let mut acc = [0u32; 3];
      for dy in 0..2 {
        for dx in 0..2 {
          let ch = layout.unpack(layout.read(src, src_w, ox * 2 + dx, oy * 2 + dy));
          for c in 0..3 {
            acc[c] += u32::from(ch[c]);
          }
        }
      }
      out[oy * out_w + ox] = layout.pack([
        ((acc[0] + 2) / 4) as u8,
        ((acc[1] + 2) / 4) as u8,
        ((acc[2] + 2) / 4) as u8,
      ]);
    }
  }
  out
}

macro_rules! lowbit_resample_tests {
  (
    mod_name: $mod_name:ident,
    marker:   $marker:ident,
    frame:    $frame:ident,
    walker:   $walker:ident,
    layout:   $layout:expr,
  ) => {
    mod $mod_name {
      use super::*;

      const SRC: usize = 8;
      const OUT: usize = 4;

      /// `rgb_u16` is the exact native-depth 2x2 area mean of the source's
      /// native channels (re-extracted from the re-packed binned frame).
      #[test]
      #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
      fn rgb_u16_is_native_area_mean() {
        let layout = $layout;
        let plane = random_lowbit_plane(layout, SRC, SRC, 0x1234_5678);
        let stride = layout.row_bytes(SRC) as u32;
        let src = $frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();

        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink =
            MixedSinker::<$marker, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
              .unwrap()
              .with_rgb_u16(&mut rgb_u16)
              .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        let binned = binned_lowbit_2x2(layout, &plane, SRC, OUT, OUT);
        for oy in 0..OUT {
          for ox in 0..OUT {
            let want = layout.unpack(binned[oy * OUT + ox]);
            for c in 0..3 {
              assert_eq!(rgb_u16[(oy * OUT + ox) * 3 + c], u16::from(want[c]), "({ox},{oy}) c{c}");
            }
          }
        }
      }

      /// Every attached output equals a direct (`new()`) conversion of the
      /// independently-binned source-format frame.
      #[test]
      #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
      fn all_outputs_match_direct_of_binned_frame() {
        let layout = $layout;
        let plane = random_lowbit_plane(layout, SRC, SRC, 0x0BAD_F00D);
        let stride = layout.row_bytes(SRC) as u32;
        let src = $frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();

        // Build the binned source-format frame as packed bytes/nibbles.
        let binned = binned_lowbit_2x2(layout, &plane, SRC, OUT, OUT);
        let out_stride = layout.row_bytes(OUT);
        let mut binned_plane = std::vec![0u8; out_stride * OUT];
        for oy in 0..OUT {
          for ox in 0..OUT {
            let v = binned[oy * OUT + ox];
            if layout.nibble {
              if ox & 1 == 0 {
                binned_plane[oy * out_stride + (ox >> 1)] = v << 4;
              } else {
                binned_plane[oy * out_stride + (ox >> 1)] |= v;
              }
            } else {
              binned_plane[oy * out_stride + ox] = v;
            }
          }
        }
        let binned_src = $frame::try_new(&binned_plane, OUT as u32, OUT as u32, out_stride as u32).unwrap();

        // Reference: direct (non-resampled) conversion of the binned frame.
        let (mut r_rgb, mut r_rgba) = (std::vec![0u8; OUT * OUT * 3], std::vec![0u8; OUT * OUT * 4]);
        let (mut r_rgbu, mut r_rgbau) = (std::vec![0u16; OUT * OUT * 3], std::vec![0u16; OUT * OUT * 4]);
        let (mut r_l, mut r_lu) = (std::vec![0u8; OUT * OUT], std::vec![0u16; OUT * OUT]);
        let (mut r_h, mut r_s, mut r_v) = (std::vec![0u8; OUT * OUT], std::vec![0u8; OUT * OUT], std::vec![0u8; OUT * OUT]);
        {
          let mut sink = MixedSinker::<$marker>::new(OUT, OUT)
            .with_rgb(&mut r_rgb).unwrap()
            .with_rgba(&mut r_rgba).unwrap()
            .with_rgb_u16(&mut r_rgbu).unwrap()
            .with_rgba_u16(&mut r_rgbau).unwrap()
            .with_luma(&mut r_l).unwrap()
            .with_luma_u16(&mut r_lu).unwrap()
            .with_hsv(&mut r_h, &mut r_s, &mut r_v).unwrap();
          $walker(&binned_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        // Actual: fused area-resample of the full-res frame.
        let (mut a_rgb, mut a_rgba) = (std::vec![0u8; OUT * OUT * 3], std::vec![0u8; OUT * OUT * 4]);
        let (mut a_rgbu, mut a_rgbau) = (std::vec![0u16; OUT * OUT * 3], std::vec![0u16; OUT * OUT * 4]);
        let (mut a_l, mut a_lu) = (std::vec![0u8; OUT * OUT], std::vec![0u16; OUT * OUT]);
        let (mut a_h, mut a_s, mut a_v) = (std::vec![0u8; OUT * OUT], std::vec![0u8; OUT * OUT], std::vec![0u8; OUT * OUT]);
        {
          let mut sink =
            MixedSinker::<$marker, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
              .unwrap()
              .with_rgb(&mut a_rgb).unwrap()
              .with_rgba(&mut a_rgba).unwrap()
              .with_rgb_u16(&mut a_rgbu).unwrap()
              .with_rgba_u16(&mut a_rgbau).unwrap()
              .with_luma(&mut a_l).unwrap()
              .with_luma_u16(&mut a_lu).unwrap()
              .with_hsv(&mut a_h, &mut a_s, &mut a_v).unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }

        assert_eq!(a_rgb, r_rgb, "rgb");
        assert_eq!(a_rgba, r_rgba, "rgba");
        assert_eq!(a_rgbu, r_rgbu, "rgb_u16");
        assert_eq!(a_rgbau, r_rgbau, "rgba_u16");
        assert_eq!(a_l, r_l, "luma");
        assert_eq!(a_lu, r_lu, "luma_u16");
        assert_eq!(a_h, r_h, "hsv H");
        assert_eq!(a_s, r_s, "hsv S");
        assert_eq!(a_v, r_v, "hsv V");
      }

      /// An identity plan (`OUT == SRC`) reproduces the non-resampled path.
      #[test]
      #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
      fn identity_plan_matches_new() {
        let layout = $layout;
        let plane = random_lowbit_plane(layout, SRC, SRC, 0x5EED_1111);
        let stride = layout.row_bytes(SRC) as u32;
        let src = $frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();

        let mut via_plan = std::vec![0u16; SRC * SRC * 3];
        let mut via_new = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink =
            MixedSinker::<$marker, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
              .unwrap()
              .with_rgb_u16(&mut via_plan)
              .unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        {
          let mut sink = MixedSinker::<$marker>::new(SRC, SRC).with_rgb_u16(&mut via_new).unwrap();
          $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
        }
        assert_eq!(via_plan, via_new, "identity plan diverges from new()");
      }

      /// A no-output sink with a plan is a clean no-op (no allocation, no error).
      #[test]
      #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
      fn no_output_is_noop() {
        let layout = $layout;
        let plane = random_lowbit_plane(layout, SRC, SRC, 0x9999);
        let stride = layout.row_bytes(SRC) as u32;
        let src = $frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();
        let mut sink =
          MixedSinker::<$marker, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT)).unwrap();
        $walker(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
      }
    }
  };
}

lowbit_resample_tests! { mod_name: rgb8, marker: Rgb8, frame: Rgb8Frame, walker: rgb8_to, layout: LowbitLayout::RGB8, }
lowbit_resample_tests! { mod_name: bgr8, marker: Bgr8, frame: Bgr8Frame, walker: bgr8_to, layout: LowbitLayout::BGR8, }
lowbit_resample_tests! { mod_name: rgb4_byte, marker: Rgb4Byte, frame: Rgb4ByteFrame, walker: rgb4_byte_to, layout: LowbitLayout::RGB4BYTE, }
lowbit_resample_tests! { mod_name: bgr4_byte, marker: Bgr4Byte, frame: Bgr4ByteFrame, walker: bgr4_byte_to, layout: LowbitLayout::BGR4BYTE, }
lowbit_resample_tests! { mod_name: rgb4, marker: Rgb4, frame: Rgb4Frame, walker: rgb4_to, layout: LowbitLayout::RGB4, }
lowbit_resample_tests! { mod_name: bgr4, marker: Bgr4, frame: Bgr4Frame, walker: bgr4_to, layout: LowbitLayout::BGR4, }

// =========================================================================
// RFC #238 tail-collapse atomicity: a first-row scratch OOM AFTER the area
// stream builds into a LOCAL and BEFORE the field insert + freeze leaves the
// stream field `None` AND `resample_outputs` uncommitted — a retryable first
// row, provable directly (no walker retry needed) via `rgb_stream_allocated()`
// and `resample_outputs_frozen()`. Covers BOTH legacy families (16-bit packed
// + 8-/4-bpp) and BOTH scratch classes reachable through the shared
// `source_rgb_scratch` grow failpoint:
//   * source-width native staging (an `rgb_u16`-only sink stages nothing through
//     the emit's u8 RGB scratch, so the native grow is the first
//     `source_rgb_scratch` after the stream build), and
//   * the emit-owned u8 RGB stage (a luma-only sink — rgb absent — pre-grows it
//     before the freeze, so its OOM is the first `source_rgb_scratch`).
// The emit-owned re-pack scratch uses a distinct `try_reserve_exact` helper, so
// the single-shot failpoint targets exactly the two `source_rgb_scratch` grows.
// =========================================================================

/// Non-packed 16-bit family, source-width native scratch: an `rgb_u16`-only sink
/// grows only the native staging row through `source_rgb_scratch`, so the armed
/// failpoint fires there — after the stream box builds, before the commit.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn rgb565_first_row_native_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  const SRC: usize = 8;
  const OUT: usize = 4;
  let words = random_words(SRC * SRC, 0xA110_C8ED, Layout::RGB565);
  let wire = pack_frame(&words);
  let src = Rgb565Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb565, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    // The row-0 area stream box builds OK (into a LOCAL); the source-width native
    // grow — before the field insert + freeze — refuses, surfacing AllocationFailed
    // with BOTH the stream field `None` and the freeze uncommitted.
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row native scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof: the stream is built into a LOCAL and inserted
    // only after the scratch grows, so it must be `None` here. The
    // insert-before-scratch shape would leave it `Some` and FAIL this.
    assert!(
      !sink.rgb_stream_allocated(),
      "a scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    // DIRECT freeze-rollback proof (same frame, BEFORE any begin_frame reset): the
    // output-set snapshot must NOT be committed. The freeze-before-scratch shape
    // would commit it and FAIL this — the half a walker retry cannot observe.
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    // Recoverability: attaching rgb (a CHANGED output set) and re-walking is ACCEPTED.
    sink.set_rgb(&mut rgb).unwrap();
    rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink)
      .expect("frame replay after a first-row native scratch OOM must succeed");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (scratch OOM was recoverable)"
  );
}

/// Non-packed 16-bit family, emit-owned u8 RGB stage: a luma-only sink (rgb
/// absent) pre-grows the emit's u8 RGB stage before the freeze, so the armed
/// failpoint fires on THAT grow — proving the emit-owned scratch is pre-grown, so
/// nothing fallible runs post-freeze.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn rgb565_first_row_emit_stage_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  const SRC: usize = 8;
  const OUT: usize = 4;
  let words = random_words(SRC * SRC, 0x5CA7_C400, Layout::RGB565);
  let wire = pack_frame(&words);
  let src = Rgb565Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 2) as u32).unwrap();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb565, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row emit-stage scratch OOM must surface AllocationFailed, got {err:?}"
    );
    assert!(
      !sink.rgb_stream_allocated(),
      "an emit-owned scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    sink.set_rgb(&mut rgb).unwrap();
    rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink)
      .expect("frame replay after a first-row emit-stage scratch OOM must succeed");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (scratch OOM was recoverable)"
  );
}

/// Packed 8-/4-bpp family, source-width native scratch (twin of the 16-bit case).
#[test]
#[cfg(feature = "std")]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn rgb8_first_row_native_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  const SRC: usize = 8;
  const OUT: usize = 4;
  let layout = LowbitLayout::RGB8;
  let plane = random_lowbit_plane(layout, SRC, SRC, 0xA110_C8ED);
  let stride = layout.row_bytes(SRC) as u32;
  let src = Rgb8Frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = rgb8_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row native scratch OOM must surface AllocationFailed, got {err:?}"
    );
    assert!(
      !sink.rgb_stream_allocated(),
      "a scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    sink.set_rgb(&mut rgb).unwrap();
    rgb8_to(&src, true, KernelMatrix::Bt709, &mut sink)
      .expect("frame replay after a first-row native scratch OOM must succeed");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (scratch OOM was recoverable)"
  );
}

/// Packed 8-/4-bpp family, emit-owned u8 RGB stage (twin of the 16-bit case).
#[test]
#[cfg(feature = "std")]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn rgb8_first_row_emit_stage_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  const SRC: usize = 8;
  const OUT: usize = 4;
  let layout = LowbitLayout::RGB8;
  let plane = random_lowbit_plane(layout, SRC, SRC, 0x5CA7_C400);
  let stride = layout.row_bytes(SRC) as u32;
  let src = Rgb8Frame::try_new(&plane, SRC as u32, SRC as u32, stride).unwrap();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = rgb8_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row emit-stage scratch OOM must surface AllocationFailed, got {err:?}"
    );
    assert!(
      !sink.rgb_stream_allocated(),
      "an emit-owned scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    sink.set_rgb(&mut rgb).unwrap();
    rgb8_to(&src, true, KernelMatrix::Bt709, &mut sink)
      .expect("frame replay after a first-row emit-stage scratch OOM must succeed");
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (scratch OOM was recoverable)"
  );
}
