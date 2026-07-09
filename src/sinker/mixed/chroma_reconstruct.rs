//! L2 shared chroma-reconstruction stage (RFC #238).
//!
//! The element-generic **centered** (#302 phase-0.5) chroma reconstruction that
//! the `Yuv422p` identity decode — and, in later RFC #238 PRs, every centered
//! resample arm plus the other subsampled layouts — funnels through, so one
//! siting implementation serves them all. [`reconstruct_chroma`] stages the
//! half-width `U` / `V` planes into the caller's already-reserved full-width
//! scratch by reusing the shipped #302 scalar upsamplers (it does **not**
//! re-implement the `1/4`–`3/4` MPEG-1 / JPEG math); the per-element /
//! per-layout differences live behind the [`ChromaCenterUpsampler`] trait, so a
//! new chroma element type or memory layout extends the stage with one trait
//! impl rather than another copy of the buffer-splitting wrapper.
//!
//! Wired today for the 8-bit ([`ChromaU8`]) and high-bit ([`ChromaU16`]) planar
//! 4:2:2 identity paths. The designed extension axes (later PRs, hence the
//! trait rather than two free functions):
//!
//! - **new element / layout kernels** — a [`ChromaCenterUpsampler`] impl over
//!   the semi-planar `_p0xx` interleaved-UV upsampler
//!   ([`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx))
//!   or a packed-YUV kernel, reachable from the matching identity / resample
//!   arms;
//! - **the vertical delay-line** — the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
//!   4:2:0 even-row vertical box-blend
//!   ([`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h))
//!   and its one-row chroma lookback, threaded as a row-index + `&mut state`
//!   extension of [`reconstruct_chroma`]. 4:2:2 is subsampled horizontally only
//!   — its siting reduces to the horizontal axis (cf.
//!   [`chroma_422_center_sited_h`](super::chroma_422_center_sited_h)) — so this
//!   first cut carries no vertical phase.

/// One chroma element type's **centered horizontal** (#302 phase-0.5)
/// reconstruction primitive — the per-element knob the element-generic
/// [`reconstruct_chroma`] L2 stage turns. Each impl delegates to the matching
/// shipped #302 scalar upsampler, so the `1/4`–`3/4` reconstruction (and its
/// edge clamp, wire byte order, and low-`BITS` masking) lives in exactly one
/// place per element type.
pub(crate) trait ChromaCenterUpsampler {
  /// Chroma sample element — `u8` (8-bit planar) or `u16` (high-bit planar).
  type Elem: Copy;

  /// Reconstructs one half-width chroma plane (`half`) to full width at the
  /// centered phase-0.5 position, writing `width` samples into `full`.
  /// `use_simd` threads the sink's SIMD toggle (`MixedSinker::with_simd`)
  /// into the shared dispatcher, byte-identical to the scalar fallback.
  fn upsample_center_h(
    &self,
    half: &[Self::Elem],
    full: &mut [Self::Elem],
    width: usize,
    use_simd: bool,
  );
}

/// 8-bit planar centered chroma upsampler — delegates to
/// [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h).
/// The shared separate-plane `u8` kernel the planar (and, once later PRs
/// repoint them, the de-interleaved semi-planar / packed) 4:2:x centered paths
/// reach.
pub(crate) struct ChromaU8;

impl ChromaCenterUpsampler for ChromaU8 {
  type Elem = u8;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn upsample_center_h(&self, half: &[u8], full: &mut [u8], width: usize, use_simd: bool) {
    crate::row::chroma_upsample_2to1_center_h_row(half, full, width, use_simd);
  }
}

/// 8-bit planar centered chroma upsampler for the `1→4` subsampled layouts
/// (4:1:1 / 4:1:0) — delegates to the SIMD-dispatched
/// [`chroma_upsample_4to1_center_h_row`](crate::row::chroma_upsample_4to1_center_h_row)
/// (byte-identical to the scalar
/// [`chroma_upsample_4to1_center_h`](crate::row::scalar::chroma_upsample_4to1_center_h)
/// per tier). The quarter-width sibling of [`ChromaU8`]: the trait's `half`
/// argument carries the quarter-width chroma plane, reconstructed to `width`
/// full-width samples.
pub(crate) struct Chroma411U8;

impl ChromaCenterUpsampler for Chroma411U8 {
  type Elem = u8;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn upsample_center_h(&self, quarter: &[u8], full: &mut [u8], width: usize, use_simd: bool) {
    crate::row::chroma_upsample_4to1_center_h_row(quarter, full, width, use_simd);
  }
}

/// High-bit (`9`…`16`-bit low-packed) planar centered chroma upsampler —
/// delegates to
/// [`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16),
/// masking each sample to the low `BITS` and operating in the source's wire
/// byte order (`big_endian`) so the reconstructed full-width chroma feeds the
/// matching high-bit 4:4:4 decode kernel bit-identically per tier. `BITS` is a
/// const generic (threaded into the per-sample mask, `u16::MAX` / a no-op at
/// `BITS = 16`); the endianness is the one runtime knob the `u8` sibling lacks.
pub(crate) struct ChromaU16<const BITS: u32> {
  /// Whether the source's `u16` chroma samples are big-endian wire order.
  pub big_endian: bool,
}

impl<const BITS: u32> ChromaCenterUpsampler for ChromaU16<BITS> {
  type Elem = u16;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn upsample_center_h(&self, half: &[u16], full: &mut [u16], width: usize, use_simd: bool) {
    crate::row::chroma_upsample_2to1_center_h_u16_row::<BITS>(
      half,
      full,
      width,
      self.big_endian,
      use_simd,
    );
  }
}

/// Element-generic L2 chroma reconstruction (RFC #238): horizontally
/// reconstructs the half-width `u_half` / `v_half` chroma planes of a
/// **centered-sited** 4:2:x source to full width into the already-reserved
/// `chroma_full` scratch, returning the two full-width slices
/// `(u_full, v_full)` the 4:4:4 decode reads. The buffer is split
/// `[0..width]` = U, `[width..2*width]` = V; each half is filled by `kernel`'s
/// [`ChromaCenterUpsampler::upsample_center_h`] (the shipped #302 phase-0.5
/// upsampler for that element type) — so this stage owns only the shared buffer
/// geometry, never the siting math.
///
/// **Infallible**: the caller must have grown `chroma_full` to `>= 2 * width`
/// up front (every centered identity path reserves it before writing any output
/// row — the crate's #180 / #314 preflight-ordering atomicity contract), so the
/// split below cannot panic and `2 * width` cannot overflow here.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn reconstruct_chroma<'s, K: ChromaCenterUpsampler>(
  kernel: K,
  chroma_full: &'s mut [K::Elem],
  u_half: &[K::Elem],
  v_half: &[K::Elem],
  width: usize,
  use_simd: bool,
) -> (&'s [K::Elem], &'s [K::Elem]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved to >= 2 * width before reconstruct_chroma"
  );
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  kernel.upsample_center_h(u_half, u_full, width, use_simd);
  kernel.upsample_center_h(v_half, v_full, width, use_simd);
  (u_full, v_full)
}
