//! 10 / 12 / 14 / 16-bit Bayer — single-plane mosaic source
//! carrying **low-packed** `u16` samples.
//!
//! Shape mirrors [`super::bayer`] for the 8-bit case but with a
//! `u16` plane and a `BITS` const generic. Sinks consume
//! [`BayerRow16<'_, BITS>`] (different row type from the 8-bit
//! [`super::BayerRow`] so the type system pins the input bit depth
//! at the sink boundary).
//!
//! Sample convention is **low-packed**: active samples occupy the
//! low `BITS` bits of each `u16`, valid range
//! `[0, (1 << BITS) - 1]`. This matches the planar
//! [`Yuv420pFrame16`](crate::frame::Yuv420pFrame16) family.
//! [`crate::frame::BayerFrame16::try_new_checked`] rejects samples
//! whose value exceeds `(1 << BITS) - 1`. **Note:** this is the
//! opposite of [`PnFrame`](crate::frame::PnFrame) (high-bit-packed
//! semi-planar `u16`); if your upstream provides high-bit-packed
//! Bayer, right-shift by `(16 - BITS)` before constructing
//! [`BayerFrame16`](crate::frame::BayerFrame16).

use crate::{
  PixelSink, SourceFormat,
  frame::BayerFrame16,
  raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance, fuse_wb_ccm},
  sealed::Sealed,
};

/// Zero-sized marker for the high-bit-depth Bayer source family.
/// Parameterized on the active bit depth `BITS` ∈ {10, 12, 14, 16}.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Bayer16<const BITS: u32>;

impl<const BITS: u32> Sealed for Bayer16<BITS> {}
impl<const BITS: u32> SourceFormat for Bayer16<BITS> {}

/// Type-aliased markers for readability at call sites.
pub type Bayer10 = Bayer16<10>;
/// 12-bit Bayer source marker.
pub type Bayer12 = Bayer16<12>;
/// 14-bit Bayer source marker.
pub type Bayer14 = Bayer16<14>;
/// 16-bit Bayer source marker.
pub type Bayer16Bit = Bayer16<16>;

/// One output row of a high-bit-depth Bayer source handed to a
/// [`BayerSink16<BITS>`].
///
/// Carries `&[u16]` slices for `above` / `mid` / `below`, the row
/// index, the pattern, the demosaic algorithm, and the **unscaled**
/// fused `M = CCM · diag(wb)` 3×3. Output-bit-depth scaling
/// (multiply by `255 / ((1 << BITS) - 1)` for u8 output; identity
/// for low-packed u16 output) is the kernel's job.
///
/// **Boundary contract: mirror-by-2** — see [`super::BayerRow`]
/// for the full discussion. Top edge supplies `above = mid_row(1)`,
/// bottom edge supplies `below = mid_row(h - 2)`; replicate
/// fallback applies only when `height < 2`. Custom sinks must
/// honor this convention.
#[derive(Debug, Clone, Copy)]
pub struct BayerRow16<'a, const BITS: u32> {
  above: &'a [u16],
  mid: &'a [u16],
  below: &'a [u16],
  row: usize,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  m: [[f32; 3]; 3],
}

impl<'a, const BITS: u32> BayerRow16<'a, BITS> {
  /// Bundles one row of a high-bit-depth Bayer source for a
  /// [`BayerSink16<BITS>`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    above: &'a [u16],
    mid: &'a [u16],
    below: &'a [u16],
    row: usize,
    pattern: BayerPattern,
    demosaic: BayerDemosaic,
    m: [[f32; 3]; 3],
  ) -> Self {
    Self {
      above,
      mid,
      below,
      row,
      pattern,
      demosaic,
      m,
    }
  }

  /// Row above `mid` per the **mirror-by-2** boundary contract:
  /// `mid_row(row - 1)` for interior rows; `mid_row(1)` at the top
  /// edge. See [`super::BayerRow::above`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn above(&self) -> &'a [u16] {
    self.above
  }

  /// The row currently being produced — `width` `u16` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mid(&self) -> &'a [u16] {
    self.mid
  }

  /// Row below `mid` per the **mirror-by-2** boundary contract:
  /// `mid_row(row + 1)` for interior rows; `mid_row(h - 2)` at the
  /// bottom edge. See [`super::BayerRow::below`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn below(&self) -> &'a [u16] {
    self.below
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Row parity (`row & 1`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row_parity(&self) -> u32 {
    (self.row & 1) as u32
  }

  /// The Bayer pattern this frame uses.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn pattern(&self) -> BayerPattern {
    self.pattern
  }

  /// The demosaic algorithm requested by the caller.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn demosaic(&self) -> BayerDemosaic {
    self.demosaic
  }

  /// Borrow the fused `M = CCM · diag(wb)` transform. Unscaled —
  /// kernels apply the input/output bit-depth scaling themselves.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn m(&self) -> &[[f32; 3]; 3] {
    &self.m
  }

  /// Active bit depth — 10, 12, 14, or 16.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }
}

/// Sinks that consume high-bit-depth Bayer rows at a fixed `BITS`.
pub trait BayerSink16<const BITS: u32>:
  for<'a> PixelSink<Input<'a> = BayerRow16<'a, BITS>>
{
}

/// Walks a [`BayerFrame16<BITS>`] row by row, handing each row to
/// the sink along with the precomputed `M = CCM · diag(wb)` 3×3.
///
/// **Sample-range pre-pass.** Before any sink interaction, the
/// walker scans the source plane and panics if any sample
/// violates the low-packed contract (`value < (1 << BITS)`). The
/// panic fires **before** [`crate::PixelSink::begin_frame`], so
/// the sink's output buffers are guaranteed untouched on contract
/// violation — no half-converted frames. The fallible alternative
/// is to construct the frame via
/// [`crate::frame::BayerFrame16::try_new_checked`], which surfaces
/// the same condition as [`crate::frame::BayerFrame16Error::SampleOutOfRange`]
/// at frame-construction time.
///
/// The pre-pass is one OR-fold across the entire plane (single
/// instruction per `u16`); cost is well below the demosaic +
/// matmul that follows. For `BITS == 16` the check is a no-op
/// (every `u16` is valid; const-folded by the compiler).
///
/// **Allocation profile.** Zero per-row and zero per-frame heap
/// allocation, identical to the 8-bit [`super::bayer_to`].
pub fn bayer16_to<const BITS: u32, S: BayerSink16<BITS>>(
  src: &BayerFrame16<'_, BITS>,
  pattern: BayerPattern,
  demosaic: BayerDemosaic,
  wb: WhiteBalance,
  ccm: ColorCorrectionMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  // Upfront sample-range validation — runs *before* `begin_frame`
  // so that a contract violation (caller used `try_new` with
  // mispacked / stale-high-bits data) panics with the user's
  // output buffers untouched, instead of mid-stream after earlier
  // rows have already been written.
  crate::row::assert_bayer16_plane_in_range::<BITS>(src.data());

  sink.begin_frame(src.width(), src.height())?;

  let m = fuse_wb_ccm(&wb, &ccm);

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let plane = src.data();

  for row in 0..h {
    // Mirror-by-2 row clamp; see [`super::bayer::bayer_to`] for
    // the rationale (CFA-parity preservation at boundaries).
    let above_row = if row == 0 {
      if h >= 2 { 1 } else { 0 }
    } else {
      row - 1
    };
    let below_row = if row + 1 == h {
      if h >= 2 { h - 2 } else { h - 1 }
    } else {
      row + 1
    };

    let above = &plane[above_row * stride..above_row * stride + w];
    let mid = &plane[row * stride..row * stride + w];
    let below = &plane[below_row * stride..below_row * stride + w];

    sink.process(BayerRow16::<BITS>::new(
      above, mid, below, row, pattern, demosaic, m,
    ))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests {
  use super::*;
  use crate::{
    frame::Bayer12Frame,
    row::{bayer16_to_rgb_row, bayer16_to_rgb_u16_row},
  };
  use core::convert::Infallible;

  /// Sink that walks each `BayerRow16<BITS>` through the public
  /// u8-output dispatcher with SIMD off.
  struct CaptureRgbU8<'a, const BITS: u32> {
    out: &'a mut [u8],
    width: u32,
  }

  impl<const BITS: u32> PixelSink for CaptureRgbU8<'_, BITS> {
    type Input<'b> = BayerRow16<'b, BITS>;
    type Error = Infallible;

    fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
      self.width = width;
      Ok(())
    }

    fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
      let r = row.row();
      let w = self.width as usize;
      let off = r * w * 3;
      let dst = &mut self.out[off..off + 3 * w];
      bayer16_to_rgb_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        dst,
        false,
      );
      Ok(())
    }
  }

  impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU8<'_, BITS> {}

  /// Sink that walks each `BayerRow16<BITS>` through the public
  /// u16-output dispatcher with SIMD off.
  struct CaptureRgbU16<'a, const BITS: u32> {
    out: &'a mut [u16],
    width: u32,
  }

  impl<const BITS: u32> PixelSink for CaptureRgbU16<'_, BITS> {
    type Input<'b> = BayerRow16<'b, BITS>;
    type Error = Infallible;

    fn begin_frame(&mut self, width: u32, _height: u32) -> Result<(), Self::Error> {
      self.width = width;
      Ok(())
    }

    fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
      let r = row.row();
      let w = self.width as usize;
      let off = r * w * 3;
      let dst = &mut self.out[off..off + 3 * w];
      bayer16_to_rgb_u16_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        dst,
        false,
      );
      Ok(())
    }
  }

  impl<const BITS: u32> BayerSink16<BITS> for CaptureRgbU16<'_, BITS> {}

  /// Build a 12-bit low-packed RGGB Bayer plane from per-channel
  /// nominal values (each 0..=4095). Bayer16 is low-packed: samples
  /// occupy the low 12 bits of each `u16`, no shift required.
  fn solid_rggb_12bit(width: u32, height: u32, r: u16, g: u16, b: u16) -> std::vec::Vec<u16> {
    let w = width as usize;
    let h = height as usize;
    let mut data = std::vec![0u16; w * h];
    for y in 0..h {
      for x in 0..w {
        let v = match (y & 1, x & 1) {
          (0, 0) => r,
          (0, 1) => g,
          (1, 0) => g,
          (1, 1) => b,
          _ => unreachable!(),
        };
        data[y * w + x] = v;
      }
    }
    data
  }

  fn assert_full_frame_u8(rgb: &[u8], w: u32, h: u32, expect: (u8, u8, u8)) {
    let w = w as usize;
    let h = h as usize;
    for y in 0..h {
      for x in 0..w {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "u8 px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "u8 px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "u8 px ({x},{y}) B");
      }
    }
  }

  fn assert_full_frame_u16(rgb: &[u16], w: u32, h: u32, expect: (u16, u16, u16)) {
    let w = w as usize;
    let h = h as usize;
    for y in 0..h {
      for x in 0..w {
        let i = (y * w + x) * 3;
        assert_eq!(rgb[i], expect.0, "u16 px ({x},{y}) R");
        assert_eq!(rgb[i + 1], expect.1, "u16 px ({x},{y}) G");
        assert_eq!(rgb[i + 2], expect.2, "u16 px ({x},{y}) B");
      }
    }
  }

  #[test]
  fn bayer12_solid_red_rggb_yields_u8_red_full_frame() {
    // 12-bit max = 4095. R = 4095 (white at this channel), G = B = 0.
    // Mirror-by-2 boundary handling means every output pixel
    // matches, including borders and corners.
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_full_frame_u8(&rgb, w, h, (255, 0, 0));
  }

  #[test]
  fn bayer12_solid_red_rggb_yields_u16_red_full_frame() {
    let (w, h) = (8u32, 6u32);
    let raw = solid_rggb_12bit(w, h, 4095, 0, 0);
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sink = CaptureRgbU16::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_full_frame_u16(&rgb, w, h, (4095, 0, 0));
  }

  #[test]
  fn bayer12_uniform_value_yields_uniform_u8_output() {
    // Every sample = 2048 (low-packed 12-bit midgray). u8 output:
    // 2048 / 4095 * 255 ≈ 127.53 → 128 everywhere (uniform input
    // so edge clamping doesn't shift the value).
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![2048u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &c in &rgb {
      assert!((c as i32 - 128).abs() <= 1, "got {c}");
    }
  }

  #[test]
  fn bayer12_uniform_value_yields_uniform_u16_output() {
    // Every sample = 4095 (max 12-bit low-packed). u16 output
    // should be 4095 (low-packed full white).
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![4095u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sink = CaptureRgbU16::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &c in &rgb {
      assert_eq!(c, 4095);
    }
  }

  #[test]
  fn bayer10_low_packed_white_yields_full_scale_u8() {
    // 10-bit low-packed white (1023). u8 output should be 255.
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![1023u16; (w * h) as usize];
    let frame = crate::frame::Bayer10Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<10> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &c in &rgb {
      assert_eq!(c, 255, "10-bit low-packed white must scale to u8 255");
    }
  }

  #[test]
  fn bayer14_low_packed_white_yields_full_scale_u16() {
    // 14-bit low-packed white (16383). u16 output should be 16383.
    let (w, h) = (8u32, 6u32);
    let raw = std::vec![16383u16; (w * h) as usize];
    let frame = crate::frame::Bayer14Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u16; (w * h * 3) as usize];
    let mut sink = CaptureRgbU16::<14> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    for &c in &rgb {
      assert_eq!(c, 16383, "14-bit low-packed white must stay 16383");
    }
  }

  /// The public Bayer16 dispatcher panics in **release mode** if
  /// any sample's high bits are non-zero (low-packed contract
  /// violation — the only out-of-range case `try_new` doesn't
  /// catch). Test runs in both debug and release.
  #[test]
  #[should_panic(expected = "Bayer16 plane contains samples")]
  fn bayer12_dispatcher_panics_on_sample_above_max() {
    let (w, h) = (4u32, 2u32);
    let mut raw = std::vec![100u16; (w * h) as usize];
    raw[3] = 4096; // just above 12-bit max
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
  }

  /// Codex-recommended regression: MSB-aligned 12-bit midgray
  /// (e.g., `2048 << 4 = 0x8000`) is exactly the common
  /// packing-mismatch bug, where a caller forgot to right-shift
  /// before constructing the `Bayer12Frame`. Should be rejected
  /// loudly, not silently produce saturated output.
  #[test]
  #[should_panic(expected = "Bayer16 plane contains samples")]
  fn bayer12_dispatcher_rejects_msb_aligned_input() {
    let (w, h) = (4u32, 2u32);
    let raw = std::vec![0x8000u16; (w * h) as usize]; // MSB-aligned 12-bit midgray
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<12> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
  }

  /// Codex-recommended regression: Bayer12Frame built via plain
  /// `try_new` (geometry-only) with one bad sample in a *later*
  /// row. The walker must panic **before** any output is written,
  /// not mid-stream after earlier rows have been mutated.
  #[test]
  fn bayer12_walker_panics_before_first_sink_mutation_on_bad_sample() {
    let (w, h) = (4u32, 8u32);
    let mut raw = std::vec![100u16; (w * h) as usize];
    // Bad sample in row 6 — far enough that several rows would
    // process cleanly under a per-row check (rows 0..5), exposing
    // the partial-mutation bug Codex's adversarial review caught.
    let off = (6 * w) as usize + 2;
    raw[off] = 4096; // exceeds 12-bit max
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();

    // The output buffer is filled with a sentinel; if the walker
    // partially mutates it before panicking, the test will detect
    // it (the sentinel won't all survive). We capture both the
    // pre-state and the panic.
    let mut rgb = std::vec![0xAAu8; (w * h * 3) as usize];
    let pre_rgb = rgb.clone();
    // Invoke the walker through `MixedSinker`, the same path a
    // safe API user would take. Catch the panic so we can inspect
    // the output buffer state afterward.
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let mut sinker = crate::sinker::MixedSinker::<Bayer16<12>>::new(w as usize, h as usize)
        .with_rgb(&mut rgb)
        .unwrap();
      bayer16_to(
        &frame,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        WhiteBalance::neutral(),
        ColorCorrectionMatrix::identity(),
        &mut sinker,
      )
    }))
    .is_err();
    assert!(panicked, "walker must panic on out-of-range Bayer12 sample");
    // Critical assertion: the output buffer is byte-identical to
    // its pre-walker state. No partial mutation.
    assert_eq!(
      rgb, pre_rgb,
      "walker panicked AFTER mutating the output buffer; partial output contract violated"
    );
  }

  /// At BITS=16 every `u16` is valid; the dispatcher's bad-bit
  /// mask is zero so the check is a no-op and 0xFFFF passes.
  #[test]
  fn bayer16bit_dispatcher_accepts_full_u16_range() {
    let (w, h) = (4u32, 2u32);
    let raw = std::vec![0xFFFFu16; (w * h) as usize];
    let frame = crate::frame::Bayer16Frame::try_new(&raw, w, h, w).unwrap();
    let mut rgb = std::vec![0u8; (w * h * 3) as usize];
    let mut sink = CaptureRgbU8::<16> {
      out: &mut rgb,
      width: 0,
    };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    // Solid 0xFFFF saturates to 255 on every channel.
    for &c in &rgb {
      assert_eq!(c, 255);
    }
  }

  #[test]
  fn bayer12_walker_calls_sink_once_per_row() {
    struct CountSink<const BITS: u32> {
      rows: u32,
    }
    impl<const BITS: u32> PixelSink for CountSink<BITS> {
      type Input<'a> = BayerRow16<'a, BITS>;
      type Error = Infallible;
      fn process(&mut self, _row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
        self.rows += 1;
        Ok(())
      }
    }
    impl<const BITS: u32> BayerSink16<BITS> for CountSink<BITS> {}

    let (w, h) = (8u32, 6u32);
    let raw = std::vec![0u16; (w * h) as usize];
    let frame = Bayer12Frame::try_new(&raw, w, h, w).unwrap();
    let mut sink = CountSink::<12> { rows: 0 };
    bayer16_to(
      &frame,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
    assert_eq!(sink.rows, h);
  }
}
