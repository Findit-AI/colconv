//! Sinker impl for the packed AYUV64 source format — Ship 12d (Tier 5
//! 16-bit packed YUV 4:4:4 with real source alpha).
//!
//! AYUV64 (FFmpeg `AV_PIX_FMT_AYUV64LE`) packs **four u16 slots per
//! pixel** (`[A, Y, U, V]`). All channels are 16-bit native — no
//! padding bits, no shift required. The A slot is **real source alpha**
//! — not padding. The packed slice type is `&[u16]`, with `4 x width`
//! u16 elements per row. There is no chroma subsampling — every pixel
//! carries its own independent A / Y / U / V quadruple (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α is
//!   depth-converted to u8 via `>> 8`** from slot 0 of each pixel.
//! - `with_rgb_u16` — packed YUV → RGB u16 pipeline; alpha discarded;
//!   i64 chroma path.
//! - `with_rgba_u16` — packed YUV → RGBA u16 pipeline; **source α u16
//!   is written direct** (no conversion); i64 chroma path.
//! - `with_luma` — extracts the Y u16 from slot 1 and downshifts `>> 8`
//!   to u8; no YUV→RGB pipeline.
//! - `with_luma_u16` — extracts the Y u16 at full 16-bit native depth;
//!   no YUV→RGB pipeline.
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! ## Alpha semantics (`§ 7.2` / Tier 5 spec rules)
//!
//! - **Standalone RGBA u8** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `ayuv64_to_rgba_row` runs directly — source α is
//!   depth-converted via `>> 8` in the kernel.
//! - **Standalone RGBA u16** (`with_rgba_u16` attached, no
//!   `with_rgb_u16`): `ayuv64_to_rgba_u16_row` runs directly — source
//!   α is written direct as u16.
//! - **RGB + RGBA** (both attached, with or without HSV): Strategy A+
//!   combo — `with_rgb` calls `ayuv64_to_rgb_row` (chroma kernel runs
//!   ONCE); `with_rgba` is derived by `expand_rgb_to_rgba_row` (writes
//!   α=`0xFF`) followed by
//!   `alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0` to
//!   overwrite the α slot from the packed source (slot 0, depth-conv
//!   `>> 8`). Output is byte-identical to calling `ayuv64_to_rgba_row`
//!   directly (spec § 3.2 / § 7.2).
//! - **RGB u16 + RGBA u16** (both attached): same A+ pattern on the u16
//!   path — `expand_rgb_u16_to_rgba_u16_row::<16>` fans out, then
//!   `copy_alpha_packed_u16x4_at_0` overwrites α from packed slot 0.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_yuva444_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    ayuv64_to_hsv_row, ayuv64_to_luma_row, ayuv64_to_luma_u16_row, ayuv64_to_rgb_row,
    ayuv64_to_rgb_u16_row, ayuv64_to_rgba_row, ayuv64_to_rgba_u16_row, expand_rgb_to_rgba_row,
    expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row,
  },
  source::{Ayuv64, Ayuv64Row, Ayuv64Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, Ayuv64<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. When AYUV64 is the
  /// source, the per-pixel alpha value is **sourced from the A u16 at
  /// slot 0 of each pixel quadruple**, depth-converted to u8 via `>> 8`
  /// — not forced to `0xFF`.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths (standalone or
  /// combined with `with_rgb` / `with_hsv`). When standalone (no
  /// `with_rgb` / `with_hsv`), `ayuv64_to_rgba_row` runs directly.
  /// When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0`
  /// overwrites the α slot — output is byte-identical to the standalone
  /// path (spec § 3.2 / § 7.2).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer. Native 16-bit depth;
  /// length is measured in `u16` **elements** (`width x height x 3`).
  /// Alpha is discarded.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if
  /// `buf.len() < width x height x 3`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer. Native 16-bit
  /// depth; source α u16 at slot 0 of each pixel quadruple is written
  /// **direct** (no conversion). Length is measured in `u16`
  /// **elements** (`width x height x 4`).
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through (u16 direct) is guaranteed in **all** paths.
  /// When standalone (no `with_rgb_u16`), `ayuv64_to_rgba_u16_row` runs
  /// directly. When combined with `with_rgb_u16`, Strategy A+ applies:
  /// `expand_rgb_u16_to_rgba_u16_row::<16>` fans out the u16 RGB row and
  /// `alpha_extract::copy_alpha_packed_u16x4_at_0` overwrites
  /// the α slot — output is byte-identical to the standalone path (spec
  /// § 3.2 / § 7.2).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a native-depth **`u16`** luma output buffer. The 16-bit Y
  /// value at slot 1 of each AYUV64 quadruple is written direct (no
  /// shift — 16-bit native). Length is measured in `u16` **elements**
  /// (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if
  /// `buf.len() < width x height`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_pixels()?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> Ayuv64Sink<BE> for MixedSinker<'_, Ayuv64<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Ayuv64<BE>, R> {
  type Input<'r> = Ayuv64Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 + u16 RGBA colour streams and the
    // independent native-Y u16 luma stream (all lazily created in
    // `process`) and re-arm the alpha-mode snapshot, mirroring the
    // alpha-aware packed-RGBA / `Ya` sinks.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Ayuv64Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // AYUV64 row = `width x 4` u16 elements (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Ayuv64Packed,
        idx,
        packed_expected,
        row.packed().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Ayuv64` is packed 4:4:4 YUV **with real 16-bit
    // source alpha** (the A u16 of each `[A, Y, U, V]` quadruple) and the
    // most demanding alpha family — it must reproduce four independently
    // rounding outputs. Route through the packed-YUVA tail at
    // `SRC_BITS = 16`, which carries THREE independent binnings:
    // - u8 colour bins `ayuv64_to_rgba_row` (u8 `YUV→RGB`, α `>> 8`) → rgb
    //   / rgba / hsv;
    // - u16 colour bins the INDEPENDENT `ayuv64_to_rgba_u16_row` (native
    //   `YUV→RGB`, α direct) → rgb_u16 / rgba_u16 — never a narrowing of the
    //   u8 bin (the u8 and u16 `YUV→RGB` kernels round and scale
    //   independently);
    // - native Y bins `ayuv64_to_luma_u16_row` (Y from slot 1) → luma_u16
    //   (native) / luma (`>> 8`), alpha- and range-independent.
    // Both colour streams bin premultiplied under `AlphaMode::Premultiplied`
    // (each at its own depth max) and un-premultiply per output row. `BE`
    // is propagated from the parent `Ayuv64Frame<'_, BE>` to each kernel.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        plan,
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return packed_yuva444_resample::<BITS>(
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        alpha_mode,
        |dst| ayuv64_to_rgba_row(packed, dst, w, matrix, full_range, use_simd, BE),
        |dst| ayuv64_to_rgba_u16_row(packed, dst, w, matrix, full_range, use_simd, BE),
        |dst| ayuv64_to_luma_u16_row(packed, dst, w, use_simd, BE),
      );
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Output mode resolution — hoisted ABOVE the luma writes so the atomicity
    // preflight runs ahead of any output mutation. `Ayuv64` also exposes u16
    // colour outputs, so the scratch-arm gate must include them.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // merged packed 4:2:2 / planar / semi-planar siblings): the sole growable
    // scratch is the u8 RGB row buffer, taken when the u8 colour decode needs
    // an RGB row but no caller RGB buffer is borrowable. With `!want_rgb` the
    // combo `need_rgb_kernel` decode reaches `rgb_row_buf_or_scratch`'s scratch
    // arm whenever HSV is wanted alongside ANY other colour output — i.e.
    // `want_rgba || want_rgb_u16 || want_rgba_u16` (the `!want_hsv_direct`
    // condition; the u16 colour outputs derive HSV off the same u8 RGB row).
    // Reserve it BEFORE the luma / luma_u16 writes so an allocator refusal is a
    // recoverable `AllocationFailed` with the outputs untouched. (`with_rgb`
    // borrows the caller buffer and never allocates; the u16 colour paths slice
    // caller buffers; the standalone-RGBA fast paths use dedicated kernels.)
    if want_hsv && !want_rgb && (want_rgba || want_rgb_u16 || want_rgba_u16) {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Luma u8 — extract Y value from slot 1 of each AYUV64 quadruple
    // and downshift `>> 8` to u8.
    if let Some(buf) = luma.as_deref_mut() {
      ayuv64_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }

    // Luma u16 — extract Y value at native 16-bit depth (written direct,
    // no shift).
    if let Some(buf) = luma_u16.as_deref_mut() {
      ayuv64_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }

    // HSV-without-any-RGB/RGBA (u8 or u16) goes through the direct
    // `ayuv64_to_hsv_row` kernel — no source-width RGB scratch (the SIMD
    // path stages a fixed 64-pixel 8-bit RGB chunk internally). When any
    // RGB/RGBA output is also attached, HSV derives off the converted u8
    // RGB row for free via the convert-once path, so `need_rgb_kernel`
    // keeps that kernel alive. (Unlike the planar 4:4:4 sink, this sink
    // runs its u16 RGB/RGBA paths AFTER the u8 block, so the HSV-direct
    // early return must also exclude pending u16 work.) Resample
    // row-stage HSV-only is a #263 follow-up — HSV stays correct via the
    // convert-once path.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba && !want_rgb_u16 && !want_rgba_u16;
    let need_rgb_kernel = want_rgb || (want_hsv && !want_hsv_direct);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      ayuv64_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // Standalone RGBA u8 fast path — spec § 7.2: when only RGBA u8 (no
    // RGB, no HSV) is requested AND no u16 work is needed, run the
    // dedicated RGBA kernel directly and return early. Source α is
    // depth-converted via `>> 8` in the kernel.
    if want_rgba && !need_rgb_kernel && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // Standalone RGBA u16 fast path — when only RGBA u16 (no RGB u16) is
    // requested AND no u8 work is needed, run the dedicated kernel
    // directly and return early; source α is written direct as u16.
    if want_rgba_u16 && !want_rgb_u16 && !need_rgb_kernel && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_u16_row(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // ===== Combo / mixed paths =====
    //
    // Reached when at least two of {rgb, rgba, hsv, rgb_u16, rgba_u16}
    // are attached, or when the single standalone fast paths didn't fire.

    // u8 RGB path — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    if need_rgb_kernel {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      ayuv64_to_rgb_row(
        packed,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );

      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          use_simd,
        );
      }

      // Strategy A+ combo (u8): RGBA also attached — derive from the
      // just-computed RGB row (writes α=0xFF), then overwrite α slot from
      // packed source (slot 0, depth-conv >> 8). Output is byte-identical
      // to ayuv64_to_rgba_row directly (spec § 3.2 / § 7.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // BE propagated from the parent `Ayuv64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Ayuv64<BE>>` monomorphization.
        crate::row::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0::<BE>(
          packed, rgba_row, w, use_simd,
        );
      }
    }

    // Standalone RGBA u8 path — want_rgba without need_rgb_kernel (so
    // want_rgba must be combined with want_rgb_u16 or want_rgba_u16 only).
    // Run ayuv64_to_rgba_row directly; source α depth-converted >> 8.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    }

    // u16 RGB path — run when rgb_u16 is attached.
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      ayuv64_to_rgb_u16_row(
        packed,
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );

      // Strategy A+ combo (u16): RGBA u16 also attached — derive from the
      // just-computed u16 RGB row (writes α=max at 16 bits), then overwrite
      // α slot from packed source (slot 0, u16 direct). Output is
      // byte-identical to ayuv64_to_rgba_u16_row directly (spec § 3.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // BE propagated from the parent `Ayuv64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Ayuv64<BE>>` monomorphization.
        crate::row::alpha_extract::copy_alpha_packed_u16x4_at_0::<BE>(
          packed,
          rgba_u16_row,
          w,
          use_simd,
        );
      }
    }

    // Standalone RGBA u16 path — want_rgba_u16 without want_rgb_u16 (so
    // want_rgba_u16 must be combined with need_rgb_kernel or want_rgba).
    // Run ayuv64_to_rgba_u16_row directly; source α u16 written direct.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_u16_row(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    }

    Ok(())
  }
}
