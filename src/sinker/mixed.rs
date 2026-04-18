//! [`MixedSinker`] — the common "I want some subset of {BGR, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format; v0.1 ships
//! the [`Yuv420p`](crate::yuv::Yuv420p) impl.

use core::marker::PhantomData;

use std::vec::Vec;

use crate::{
  PixelSink, SourceFormat,
  row::{bgr_to_hsv_row, yuv_420_to_bgr_row},
  yuv::{Yuv420p, Yuv420pRow, Yuv420pSink},
};

/// A sink that writes any subset of `{BGR, Luma, HSV}` into
/// caller-provided buffers.
///
/// Each output is optional — provide `Some(buffer)` to have that
/// channel written, leave it `None` to skip. Providing no outputs is
/// legal (the kernel still walks the source and calls `process_row`
/// for each row, but nothing is written).
///
/// When HSV is requested **without** BGR, `MixedSinker` keeps a single
/// row of intermediate BGR in an internal scratch buffer (allocated
/// lazily on first use). If BGR output is also requested, the user's
/// BGR buffer serves as the intermediate for HSV and no scratch is
/// allocated.
///
/// # Type parameter
///
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Bgr24`, etc.
/// Each format provides its own `impl PixelSink for MixedSinker<'_, F>`
/// (the only `impl` landed in v0.1 is for [`Yuv420p`]).
pub struct MixedSinker<'a, F: SourceFormat> {
  bgr: Option<&'a mut [u8]>,
  luma: Option<&'a mut [u8]>,
  hsv: Option<HsvBuffers<'a>>,
  width: usize,
  /// Lazily grown to `3 * width` bytes when HSV is requested without a
  /// user BGR buffer. Empty otherwise.
  bgr_scratch: Vec<u8>,
  _fmt: PhantomData<F>,
}

/// The three output planes for HSV, bundled so `MixedSinker` stores a
/// single `Option<HsvBuffers>` rather than three independent options.
pub struct HsvBuffers<'a> {
  /// Hue plane (OpenCV 8-bit: `H ∈ [0, 179]`), at least
  /// `width * height` bytes.
  pub h: &'a mut [u8],
  /// Saturation plane (`S ∈ [0, 255]`), at least `width * height` bytes.
  pub s: &'a mut [u8],
  /// Value plane (`V ∈ [0, 255]`), at least `width * height` bytes.
  pub v: &'a mut [u8],
}

impl<F: SourceFormat> MixedSinker<'_, F> {
  /// Creates an empty [`MixedSinker`] for the given output width in
  /// pixels. No outputs are requested until `with_bgr` / `with_luma` /
  /// `with_hsv` are called on the builder.
  #[inline]
  pub fn new(width: usize) -> Self {
    Self {
      bgr: None,
      luma: None,
      hsv: None,
      width,
      bgr_scratch: Vec::new(),
      _fmt: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write BGR.
  #[inline]
  pub fn produces_bgr(&self) -> bool {
    self.bgr.is_some()
  }

  /// Returns `true` iff the sinker will write luma.
  #[inline]
  pub fn produces_luma(&self) -> bool {
    self.luma.is_some()
  }

  /// Returns `true` iff the sinker will write HSV.
  #[inline]
  pub fn produces_hsv(&self) -> bool {
    self.hsv.is_some()
  }

  /// Frame width in pixels. Output buffers are expected to be at
  /// least `width * height * bytes_per_pixel` bytes.
  #[inline]
  pub const fn width(&self) -> usize {
    self.width
  }
}

impl<'a, F: SourceFormat> MixedSinker<'a, F> {
  /// Attaches a packed 24-bit BGR output buffer.
  /// `buf.len()` must be `>= width * height * 3`.
  #[inline]
  pub fn with_bgr(mut self, buf: &'a mut [u8]) -> Self {
    self.bgr = Some(buf);
    self
  }

  /// Attaches a single-plane luma output buffer.
  /// `buf.len()` must be `>= width * height`.
  #[inline]
  pub fn with_luma(mut self, buf: &'a mut [u8]) -> Self {
    self.luma = Some(buf);
    self
  }

  /// Attaches three HSV output planes.
  /// Each plane's length must be `>= width * height`.
  #[inline]
  pub fn with_hsv(mut self, h: &'a mut [u8], s: &'a mut [u8], v: &'a mut [u8]) -> Self {
    self.hsv = Some(HsvBuffers { h, s, v });
    self
  }
}

// ---- Yuv420p impl --------------------------------------------------------

impl PixelSink for MixedSinker<'_, Yuv420p> {
  type Input<'r> = Yuv420pRow<'r>;

  fn process_row(&mut self, row: Yuv420pRow<'_>) {
    let w = self.width;
    let idx = row.row;

    // Split-borrow so the `bgr_scratch` path and the `hsv` write don't
    // collide with the `bgr` read-after-write chain below.
    let Self {
      bgr,
      luma,
      hsv,
      bgr_scratch,
      ..
    } = self;

    // Luma — YUV420p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[idx * w..(idx + 1) * w].copy_from_slice(&row.y[..w]);
    }

    let want_bgr = bgr.is_some();
    let want_hsv = hsv.is_some();
    if !want_bgr && !want_hsv {
      return;
    }

    // Pick where the BGR row lands. If the caller wants BGR in their
    // own buffer, write directly there; otherwise use the scratch.
    // Either way, the slice we hold is `&mut [u8]` that we then
    // reborrow as `&[u8]` for the HSV step.
    let bgr_row: &mut [u8] = match bgr.as_deref_mut() {
      Some(buf) => &mut buf[idx * w * 3..(idx + 1) * w * 3],
      None => {
        if bgr_scratch.len() < w * 3 {
          bgr_scratch.resize(w * 3, 0);
        }
        &mut bgr_scratch[..w * 3]
      }
    };

    // Fused YUV→BGR: upsample chroma in registers inside the row
    // primitive, no intermediate memory.
    yuv_420_to_bgr_row(
      row.y,
      row.u_half,
      row.v_half,
      bgr_row,
      w,
      row.matrix,
      row.full_range,
    );

    // HSV from the BGR row we just wrote.
    if let Some(hsv) = hsv.as_mut() {
      bgr_to_hsv_row(
        bgr_row,
        &mut hsv.h[idx * w..(idx + 1) * w],
        &mut hsv.s[idx * w..(idx + 1) * w],
        &mut hsv.v[idx * w..(idx + 1) * w],
        w,
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
    // Neutral chroma → gray BGR; solid Y=128 → ~128 in every BGR byte.
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 128, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut bgr = std::vec![0u8; 16 * 8 * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(16).with_bgr(&mut bgr);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    for px in bgr.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1);
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn hsv_only_allocates_scratch_and_produces_gray_hsv() {
    // Neutral gray → H=0, S=0, V=~128. No BGR buffer provided.
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

    let mut bgr = std::vec![0u8; 16 * 8 * 3];
    let mut luma = std::vec![0u8; 16 * 8];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16)
      .with_bgr(&mut bgr)
      .with_luma(&mut luma)
      .with_hsv(&mut h, &mut s, &mut v);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    // Luma = Y plane verbatim.
    assert!(luma.iter().all(|&y| y == 200));
    // BGR gray.
    for px in bgr.chunks(3) {
      assert!(px[0].abs_diff(200) <= 1);
    }
    // HSV of gray.
    assert!(h.iter().all(|&b| b == 0));
    assert!(s.iter().all(|&b| b == 0));
    assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
  }

  #[test]
  fn bgr_with_hsv_uses_user_buffer_not_scratch() {
    // When caller provides BGR, the scratch should remain empty (Vec len 0).
    let (yp, up, vp) = solid_yuv420p_frame(16, 8, 100, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

    let mut bgr = std::vec![0u8; 16 * 8 * 3];
    let mut h = std::vec![0u8; 16 * 8];
    let mut s = std::vec![0u8; 16 * 8];
    let mut v = std::vec![0u8; 16 * 8];
    let mut sink = MixedSinker::<Yuv420p>::new(16)
      .with_bgr(&mut bgr)
      .with_hsv(&mut h, &mut s, &mut v);
    yuv420p_to(&src, true, ColorMatrix::Bt601, &mut sink);

    assert_eq!(
      sink.bgr_scratch.len(),
      0,
      "scratch should stay unallocated when BGR buffer is provided"
    );
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
