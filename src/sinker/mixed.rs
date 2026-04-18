//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format; v0.1 ships
//! the [`Yuv420p`](crate::yuv::Yuv420p) impl.

use core::marker::PhantomData;

use std::vec::Vec;

use crate::{
  HsvBuffers, PixelSink, SourceFormat,
  row::{rgb_to_hsv_row, yuv_420_to_rgb_row},
  yuv::{Yuv420p, Yuv420pRow, Yuv420pSink},
};

/// A sink that writes any subset of `{RGB, Luma, HSV}` into
/// caller-provided buffers.
///
/// Each output is optional — provide `Some(buffer)` to have that
/// channel written, leave it `None` to skip. Providing no outputs is
/// legal (the kernel still walks the source and calls `process`
/// for each row, but nothing is written).
///
/// When HSV is requested **without** RGB, `MixedSinker` keeps a single
/// row of intermediate RGB in an internal scratch buffer (allocated
/// lazily on first use). If RGB output is also requested, the user's
/// RGB buffer serves as the intermediate for HSV and no scratch is
/// allocated.
///
/// # Type parameter
///
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Bgr24`, etc.
/// Each format provides its own `impl PixelSink for MixedSinker<'_, F>`
/// (the only `impl` landed in v0.1 is for [`Yuv420p`]).
pub struct MixedSinker<'a, F: SourceFormat> {
  rgb: Option<&'a mut [u8]>,
  luma: Option<&'a mut [u8]>,
  hsv: Option<HsvBuffers<'a>>,
  width: usize,
  /// Lazily grown to `3 * width` bytes when HSV is requested without a
  /// user RGB buffer. Empty otherwise.
  rgb_scratch: Vec<u8>,
  /// Whether row primitives dispatch to their SIMD backend. Defaults
  /// to `true`; benchmarks flip this with [`Self::with_simd`] /
  /// [`Self::set_simd`] to A/B test scalar vs SIMD on the same frame.
  simd: bool,
  _fmt: PhantomData<F>,
}

impl<F: SourceFormat> MixedSinker<'_, F> {
  /// Creates an empty [`MixedSinker`] for the given output width in
  /// pixels. No outputs are requested until `with_rgb` / `with_luma` /
  /// `with_hsv` are called on the builder.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(width: usize) -> Self {
    Self {
      rgb: None,
      luma: None,
      hsv: None,
      width,
      rgb_scratch: Vec::new(),
      simd: true,
      _fmt: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write RGB.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb(&self) -> bool {
    self.rgb.is_some()
  }

  /// Returns `true` iff the sinker will write luma.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma(&self) -> bool {
    self.luma.is_some()
  }

  /// Returns `true` iff the sinker will write HSV.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_hsv(&self) -> bool {
    self.hsv.is_some()
  }

  /// Frame width in pixels. Output buffers are expected to be at
  /// least `width * height * bytes_per_pixel` bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// Returns `true` iff row primitives dispatch to their SIMD backend.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn simd(&self) -> bool {
    self.simd
  }

  /// Toggles the SIMD dispatch in place. See [`Self::with_simd`] for the
  /// consuming builder variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_simd(&mut self, simd: bool) -> &mut Self {
    self.simd = simd;
    self
  }

  /// Sets whether row primitives dispatch to their SIMD backend.
  /// Defaults to `true` — pass `false` to force the scalar reference
  /// path (intended for benchmarks, fuzzing, and differential
  /// testing). See [`Self::set_simd`] for the in‑place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, simd: bool) -> Self {
    self.set_simd(simd);
    self
  }
}

impl<'a, F: SourceFormat> MixedSinker<'a, F> {
  /// Attaches a packed 24-bit RGB output buffer.
  /// `buf.len()` must be `>= width * height * 3`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_rgb(mut self, buf: &'a mut [u8]) -> Self {
    self.set_rgb(buf);
    self
  }

  /// Attaches a packed 24-bit RGB output buffer.
  /// `buf.len()` must be `>= width * height * 3`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_rgb(&mut self, buf: &'a mut [u8]) -> &mut Self {
    self.rgb = Some(buf);
    self
  }

  /// Attaches a single-plane luma output buffer.
  /// `buf.len()` must be `>= width * height`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_luma(mut self, buf: &'a mut [u8]) -> Self {
    self.set_luma(buf);
    self
  }

  /// Attaches a single-plane luma output buffer.
  /// `buf.len()` must be `>= width * height`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_luma(&mut self, buf: &'a mut [u8]) -> &mut Self {
    self.luma = Some(buf);
    self
  }

  /// Attaches three HSV output planes.
  /// Each plane's length must be `>= width * height`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_hsv(mut self, h: &'a mut [u8], s: &'a mut [u8], v: &'a mut [u8]) -> Self {
    self.set_hsv(h, s, v);
    self
  }

  /// Attaches three HSV output planes.
  /// Each plane's length must be `>= width * height`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_hsv(&mut self, h: &'a mut [u8], s: &'a mut [u8], v: &'a mut [u8]) -> &mut Self {
    self.hsv = Some(HsvBuffers { h, s, v });
    self
  }
}

// ---- Yuv420p impl --------------------------------------------------------

impl PixelSink for MixedSinker<'_, Yuv420p> {
  type Input<'r> = Yuv420pRow<'r>;

  fn process(&mut self, row: Yuv420pRow<'_>) {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    // Split-borrow so the `rgb_scratch` path and the `hsv` write don't
    // collide with the `rgb` read-after-write chain below.
    let Self {
      rgb,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    // Luma — YUV420p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[idx * w..(idx + 1) * w].copy_from_slice(&row.y()[..w]);
    }

    let want_rgb = rgb.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_hsv {
      return;
    }

    // Pick where the RGB row lands. If the caller wants RGB in their
    // own buffer, write directly there; otherwise use the scratch.
    // Either way, the slice we hold is `&mut [u8]` that we then
    // reborrow as `&[u8]` for the HSV step.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => &mut buf[idx * w * 3..(idx + 1) * w * 3],
      None => {
        if rgb_scratch.len() < w * 3 {
          rgb_scratch.resize(w * 3, 0);
        }
        &mut rgb_scratch[..w * 3]
      }
    };

    // Fused YUV→RGB: upsample chroma in registers inside the row
    // primitive, no intermediate memory.
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    // HSV from the RGB row we just wrote.
    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[idx * w..(idx + 1) * w],
        &mut hsv.s[idx * w..(idx + 1) * w],
        &mut hsv.v[idx * w..(idx + 1) * w],
        w,
        use_simd,
      );
    }
  }
}

impl Yuv420pSink for MixedSinker<'_, Yuv420p> {}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{ColorMatrix, frame::Yuv420pFrame, yuv::yuv420p_to};

  fn solid_yuv420p_frame(
    width: u32,
    height: u32,
    y: u8,
    u: u8,
    v: u8,
  ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    (
      std::vec![y; w * h],
      std::vec![u; cw * ch],
      std::vec![v; cw * ch],
    )
  }

  #[test]
  fn luma_only_copies_y_plane() {
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 42, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut luma = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16).with_luma(&mut luma);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
  }

  #[test]
  fn bgr_only_converts_gray_to_gray() {
    // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16).with_rgb(&mut rgb);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn hsv_only_allocates_scratch_and_produces_gray_hsv() {
    // Neutral gray → H=0, S=0, V=~128. No RGB buffer provided.
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut h = std::vec![0xFFu8; 16 * 8];
    let mut s = std::vec![0xFFu8; 16 * 8];
    let mut v = std::vec![0xFFu8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16).with_hsv(&mut h, &mut s, &mut v);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
  }

  #[test]
  fn mixed_all_three_outputs_populated() {
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 200, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16)
      .with_rgb(&mut rgb)
      .with_luma(&mut luma)
      .with_hsv(&mut h, &mut s, &mut v);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    // Luma = Y plane verbatim.
    assert!(luma.iter().all(|&y| y == 200));
    // RGB gray.
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    // HSV of gray.
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  fn bgr_with_hsv_uses_user_buffer_not_scratch() {
    // When caller provides RGB, the scratch should remain empty (Vec len 0).
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 100, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut rgb = std::vec![0u8; 16 * 8 * 3];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16)
      .with_rgb(&mut rgb)
      .with_hsv(&mut h, &mut s, &mut v);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    assert_eq!(
      sink.rgb_scratch.len(),
      0,
      "scratch should stay unallocated when RGB buffer is provided"
    );
  }

  #[test]
  fn with_simd_false_matches_with_simd_true() {
    // A/B test: same frame, one sinker forces scalar, the other uses
    // SIMD. NEON is bit‑exact to scalar so outputs must match.
    let w = 32usize;
    let h = 16usize;
    let (yp, up, vp) = solid_yuv420p_frame(w as u32, h as u32, 180, 60, 200);
    let src = Yuv420pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
    );

    let mut bgr_simd = std::vec![0u8; w * h * 3];
    let mut bgr_scalar = std::vec![0u8; w * h * 3];

    let mut sink_simd = MixedSinker::<Yuv420p>::new(w).with_rgb(&mut bgr_simd);
    let mut sink_scalar = MixedSinker::<Yuv420p>::new(w)
      .with_rgb(&mut bgr_scalar)
      .with_simd(false);
    assert!(sink_simd.simd());
    assert!(!sink_scalar.simd());

    yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_simd);
    yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar);

    assert_eq!(bgr_simd, bgr_scalar);
  }

  #[test]
  fn stride_padded_source_reads_correct_pixels() {
    // 16×8 frame, Y stride 32 (padding), chroma stride 16.
    let w = 16usize;
    let h = 8usize;
    let y_stride = 32usize;
    let c_stride = 16usize;
    let mut yp = std::vec![0xFFu8; y_stride * h]; // padding = 0xFF
    let mut up = std::vec![0xFFu8; c_stride * h / 2];
    let mut vp = std::vec![0xFFu8; c_stride * h / 2];
    // Write actual pixel data in non-padding bytes.
    for row in 0..h {
      for x in 0..w {
        yp[row * y_stride + x] = 50;
      }
    }
    for row in 0..h / 2 {
      for x in 0..w / 2 {
        up[row * c_stride + x] = 128;
        vp[row * c_stride + x] = 128;
      }
    }

    let src = Yuv420pFrame::new(
      &yp,
      &up,
      &vp,
      w as u32,
      h as u32,
      y_stride as u32,
      c_stride as u32,
      c_stride as u32,
    );

    let mut luma = std::vec![0u8; w * h];
    let mut sink = MixedSinker::<Yuv420p>::new(w).with_luma(&mut luma);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    assert!(
      luma.iter().all(|&y| y == 50),
      "padding bytes leaked into output"
    );
  }
}
