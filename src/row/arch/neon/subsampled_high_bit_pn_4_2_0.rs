use crate::{ColorMatrix, row::scalar};

use super::*;

/// Apply per-lane byte-swap to a `uint16x8x2_t` pair when the source
/// (wire) endian differs from the host's native u16 byte order.
///
/// `vld2q_u16` materializes lanes in the **host-native** u16 byte order
/// regardless of the wire encoding, so the swap must trigger on
/// `BE != HOST_NATIVE_BE`. Truth table:
///
/// | wire `BE` | host       | gate    | action            |
/// |-----------|------------|---------|-------------------|
/// | `false`   | LE         | `false` | no swap (LE→LE)   |
/// | `false`   | BE         | `true`  | swap (LE→BE)      |
/// | `true`    | LE         | `true`  | swap (BE→LE)      |
/// | `true`    | BE         | `false` | no swap (BE→BE)   |
///
/// `BE` and [`HOST_NATIVE_BE`](super::HOST_NATIVE_BE) are both
/// compile-time constants, so the gate folds and the unused branch is
/// eliminated. Reuses the shared [`bswap_u16x8_if_be`](super::bswap_u16x8_if_be)
/// helper so both lanes go through the same target-aware path.
#[inline(always)]
unsafe fn deinterleave_endian<const BE: bool>(pair: uint16x8x2_t) -> uint16x8x2_t {
  unsafe {
    uint16x8x2_t(
      bswap_u16x8_if_be::<BE>(pair.0),
      bswap_u16x8_if_be::<BE>(pair.1),
    )
  }
}

/// De-packs a host-native `uint16x8_t` of wire samples to its `BITS`-bit
/// active value: a right shift by `16 - BITS` (`vshlq_u16` with the
/// negative `shr_count`) for the high-bit-packed P-formats
/// (`LOW_PACKED = false`), or a low mask `& ((1 << BITS) - 1)`
/// (`low_mask`) for low-bit-packed NV20 (`LOW_PACKED = true`). The NEON
/// twin of the scalar `depack_pn`. `LOW_PACKED` is const, so the unused
/// branch is eliminated; byte-identical to the scalar de-pack per lane.
#[inline(always)]
pub(crate) unsafe fn depack_u16x8<const LOW_PACKED: bool>(
  raw: uint16x8_t,
  shr_count: int16x8_t,
  low_mask: uint16x8_t,
) -> uint16x8_t {
  unsafe {
    if LOW_PACKED {
      vandq_u16(raw, low_mask)
    } else {
      vshlq_u16(raw, shr_count)
    }
  }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}: P010, P012)
/// → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`yuv_420p_n_to_rgb_row`]:
/// - UV is semi‑planar interleaved (`U0, V0, U1, V1, …`), split in
///   one shot via `vld2q_u16` (returns separate U and V vectors).
/// - Each `u16` load is **right‑shifted by `16 - BITS`** — 6 for
///   P010, 4 for P012 — extracting the `BITS` active bits from the
///   high bits of each `u16` and clearing the low bits. The shift
///   runs via `vshlq_u16` with a negative loop‑invariant count so a
///   single kernel serves all supported bit depths.
///
/// After the shift, the rest of the pipeline is identical to the
/// low‑bit‑packed planar path — same `chroma_i16x8` / `scale_y` /
/// `chroma_dup` / `vst3q_u8` write, with `range_params_n::<BITS, 8>`
/// scaling.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS, BE>`] across all
/// supported `BITS` values and endian modes.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12}`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) → packed
/// **8-bit** RGB. Low-bit twin of [`p_n_to_rgb_row`] — `& 0x03FF`
/// extraction (`LOW_PACKED = true`) instead of `>> 6`. Pinned to
/// `BITS = 10`. Byte-identical to [`scalar::nv20_to_rgb_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv20_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV20 → packed **8-bit RGBA** (`R, G, B, 0xFF`). Low-bit twin of
/// [`p_n_to_rgba_row`]. Byte-identical to
/// [`scalar::nv20_to_rgba_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv20_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// NEON high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as [`p_n_to_rgb_row`];
/// 4 bpp store with constant alpha.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON kernel for [`p_n_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p_n_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha vector).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // P016 (BITS=16) routes through `p16_to_rgb_or_rgba_row` (i64 chroma);
  // attempting `::<16, _>` here would silently overflow on high chroma.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // High-bit-packed (`LOW_PACKED = false`): `vshlq_u16` right-shifts by
    // -(16 - BITS) to extract the active high bits. Low-bit-packed NV20
    // (`LOW_PACKED = true`): the `low_mask` keeps the active low `BITS`.
    // One of these is dead per monomorphization (both consts).
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let low_mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads; BE-swapped before the high-bit
      // extraction shift (swap_bytes must precede >> (16-BITS) to get
      // correct active-bit alignment from BE wire format).
      let y_raw_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_raw_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let y_vec_lo = depack_u16x8::<LOW_PACKED>(y_raw_lo, shr_count, low_mask);
      let y_vec_hi = depack_u16x8::<LOW_PACKED>(y_raw_hi, shr_count, low_mask);

      // Semi‑planar UV: `vld2q_u16` loads 16 interleaved `u16` elements
      // and returns (evens, odds) = (U, V) in one shot.  For BE input,
      // byte-swap each deinterleaved vector before the de-pack.
      let uv_raw = vld2q_u16(uv_half.as_ptr().add(x));
      let uv_swapped = deinterleave_endian::<BE>(uv_raw);
      let u_vec = depack_u16x8::<LOW_PACKED>(uv_swapped.0, shr_count, low_mask);
      let v_vec = depack_u16x8::<LOW_PACKED>(uv_swapped.1, shr_count, low_mask);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      if ALPHA {
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      // Route the scalar tail through the same `LOW_PACKED` de-pack as the
      // SIMD body so the high- and low-packed tails match their vector path.
      scalar::p_n_to_rgb_or_rgba_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
    }
  }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output,
/// `yuv420p10le` / `yuv420p12le` convention — not P010/P012).
///
/// Same structure as [`super::neon::p_n_to_rgb_row`] up to the
/// chroma compute; the only differences are:
/// - `range_params_n::<BITS, BITS>` → larger scales targeting the
///   native‑depth output range.
/// - Clamp is explicit min/max to `[0, (1 << BITS) - 1]` via
///   [`clamp_u10`](crate::row::arch::neon::clamp_u10) — the helper
///   name is historical; the actual max is derived from `BITS` at
///   the call site (1023 for P010, 4095 for P012).
/// - Writes use two `vst3q_u16` calls per 16‑pixel block.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS, BE>`] for the
/// monomorphized `BITS` and endian mode.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV20 → packed **native-depth `u16`** RGB (low-bit-packed output,
/// `[0, 1023]`). Low-bit twin of [`p_n_to_rgb_u16_row`]. Byte-identical
/// to [`scalar::nv20_to_rgb_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv20_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, false, BE, true>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON NV20 → packed **native-depth `u16` RGBA** (low-bit-packed;
/// alpha `1023`). Low-bit twin of [`p_n_to_rgba_u16_row`].
/// Byte-identical to [`scalar::nv20_to_rgba_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_u16_row`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv20_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, true, BE, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth) — matches `scalar::p_n_to_rgba_u16_row`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`], plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared NEON Pn → native-depth `u16` kernel. `ALPHA = false` writes
/// RGB triples via `vst3q_u16`; `ALPHA = true` writes RGBA quads via
/// `vst4q_u16` with constant alpha `(1 << BITS) - 1`. P016 has its
/// own kernel family — never routed here.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let low_mask = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_raw_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_raw_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let y_vec_lo = depack_u16x8::<LOW_PACKED>(y_raw_lo, shr_count, low_mask);
      let y_vec_hi = depack_u16x8::<LOW_PACKED>(y_raw_hi, shr_count, low_mask);
      let uv_raw = vld2q_u16(uv_half.as_ptr().add(x));
      let uv_swapped = deinterleave_endian::<BE>(uv_raw);
      let u_vec = depack_u16x8::<LOW_PACKED>(uv_swapped.0, shr_count, low_mask);
      let v_vec = depack_u16x8::<LOW_PACKED>(uv_swapped.1, shr_count, low_mask);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      // Route the scalar tail through the same `LOW_PACKED` de-pack as the
      // SIMD body (see the u8 kernel's tail).
      scalar::p_n_to_rgb_or_rgba_u16_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
    }
  }
}
/// NEON P016 (semi-planar 16-bit) → packed 8-bit RGB.
///
/// UV is interleaved (`U0, V0, U1, V1, …`), split via `vld2q_u16`.
/// Byte-identical to [`scalar::p16_to_rgb_row::<BE>`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON P016 (semi-planar 4:2:0, full 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P016 kernel for [`p16_to_rgb_row`] (`ALPHA = false`,
/// `vst3q_u8`) and [`p16_to_rgba_row`] (`ALPHA = true`, `vst4q_u8`
/// with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // P016 has no shift — load, optionally byte-swap, direct use.
      let y_vec_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_vec_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let uv_raw = vld2q_u16(uv_half.as_ptr().add(x));
      let uv_swapped = deinterleave_endian::<BE>(uv_raw);
      let u_vec = uv_swapped.0;
      let v_vec = uv_swapped.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed native-depth u16 RGB.
///
/// Byte-identical to [`scalar::p16_to_rgb_u16_row::<BE>`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF` — matches `scalar::p16_to_rgba_u16_row::<BE>`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = vdupq_n_u16(0xFFFF);
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_vec_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let uv_raw = vld2q_u16(uv_half.as_ptr().add(x));
      let uv_swapped = deinterleave_endian::<BE>(uv_raw);
      let u_vec = uv_swapped.0;
      let v_vec = uv_swapped.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
      let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
      let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
      let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
      let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
      let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
      let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
      let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
      let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
      let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
      let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
      let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      let r_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, alpha_u16),
        );
        vst4q_u16(
          out.as_mut_ptr().add(x * 4 + 32),
          uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
        );
        vst3q_u16(
          out.as_mut_ptr().add(x * 3 + 24),
          uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
        );
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- High-bit semi-planar P-format → HSV (staged via a reused 8-bit RGB chunk) ----
//
// The NEON twins of the scalar `p_n_to_hsv_row` / `p16_to_hsv_row`
// kernels. Rather than re-derive an HSV-specific register pipeline, each
// fills a small fixed reused **8-bit** RGB scratch (one `HSV_CHUNK`-pixel
// chunk at a time) using the EXISTING NEON `p_n_to_rgb_row::<BITS, BE>` /
// `p16_to_rgb_row::<BE>` kernel of this file — so the chunk filler IS the
// production 8-bit RGB kernel — then runs the NEON `rgb_to_hsv_row` on
// the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` (and the P016
// analogue) within the NEON tier — the same 8-bit RGB intermediate the
// existing high-bit HSV path uses — with no source-width RGB allocation.
// The scalar tail of each underlying RGB kernel handles widths below the
// SIMD block, so no separate tail is needed here. `HSV_CHUNK` (64) is a
// multiple of 2, so every chunk offset lands on a chroma-sample boundary
// for the 1→2 (4:2:0) interleaved-UV shape (`uv` byte offset = pixel
// offset).
//
// The chunked driver is defined locally (mirroring the semi-planar 8-bit
// `nv_to_hsv_via_rgb_chunks`) rather than reusing the planar
// `yuv_to_hsv_via_rgb_chunks`, because this file compiles under
// `yuv-semi-planar` alone while the planar driver lives behind the
// `yuv-planar` gate. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared NEON driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing NEON RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the NEON [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(p*_to_rgb_row(...))` within
/// the NEON tier, with no source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// NEON must be available, and `fill_rgb` must uphold the underlying RGB
/// kernel's safety contract for each chunk. Each of `h_out` / `s_out` /
/// `v_out` must be `>= width`.
#[inline]
unsafe fn pn_hsv_via_rgb_chunks(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  mut fill_rgb: impl FnMut(usize, usize, &mut [u8]),
) {
  let mut scratch = [0u8; HSV_CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(HSV_CHUNK);
    fill_rgb(offset, n, &mut scratch[..n * 3]);
    // SAFETY: NEON verified by the wrapper's `#[target_feature]`; the
    // chunk and the output sub-slices are all length `n`.
    unsafe {
      rgb_to_hsv_row(
        &scratch[..n * 3],
        &mut h_out[offset..offset + n],
        &mut s_out[offset..offset + n],
        &mut v_out[offset..offset + n],
        n,
      );
    }
    offset += n;
  }
}

/// NEON: high-bit-packed semi-planar 4:2:0 (P010/P012) → planar HSV
/// bytes (OpenCV encoding), staged via the reused-8-bit-RGB-chunk
/// pattern over the NEON [`p_n_to_rgb_row`] + [`rgb_to_hsv_row`].
/// Const-generic over `BITS ∈ {10, 12}` and `BE`. Byte-identical to
/// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` within the NEON
/// tier.
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the NEON high-bit 4:2:0 RGB
  // kernel under the same contract (its own scalar tail covers small n).
  // The same `LOW_PACKED` de-pack threads through the shared RGB kernel.
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p_n_to_rgb_or_rgba_row::<BITS, false, BE, LOW_PACKED>(
        &y[offset..],
        &uv_half[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// NEON NV20 → planar HSV bytes (OpenCV encoding), the low-bit twin of
/// [`p_n_to_hsv_row`] (`LOW_PACKED = true`, `BITS = 10`). Byte-identical
/// to `rgb_to_hsv_row(nv20_to_rgb_row::<BE>(...))` within the NEON tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv20_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: forwarded to the shared NEON HSV kernel with `LOW_PACKED = true`.
  unsafe {
    p_n_to_hsv_row::<10, BE, true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// NEON: P016 (semi-planar 4:2:0, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the NEON [`p16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Byte-
/// identical to `rgb_to_hsv_row(p16_to_rgb_row::<BE>(...))` within the
/// NEON tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the NEON P016 RGB kernel under the same contract (its own
  // scalar tail covers small n).
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p16_to_rgb_row::<BE>(&y[offset..], &uv_half[offset..], rgb, n, matrix, full_range);
    });
  }
}
