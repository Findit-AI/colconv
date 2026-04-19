//! Scalar reference implementations of the row primitives.
//!
//! Always compiled. SIMD backends live in [`super::arch`] and dispatch
//! to these as their tail fallback. Per-call dispatch in
//! [`super`]`::{yuv_420_to_rgb_row, rgb_to_hsv_row}` picks the best
//! backend at the module boundary.

use crate::ColorMatrix;

// ---- YUV 4:2:0 → RGB (fused: upsample + convert) ----------------------

/// Converts one row of 4:2:0 YUV — Y at full width, U/V at half-width —
/// directly to packed RGB. Chroma is nearest-neighbor upsampled **in
/// registers** inside the kernel; no intermediate memory traffic.
///
/// `full_range = true` interprets Y in `[0, 255]` and chroma in
/// `[0, 255]` (JPEG / `yuvjNNNp` convention). `full_range = false`
/// interprets Y in `[16, 235]` and chroma in `[16, 240]` (broadcast /
/// limited-range convention).
///
/// Output is packed `R, G, B` triples: `rgb_out[3*x] = R`,
/// `rgb_out[3*x + 1] = G`, `rgb_out[3*x + 2] = B`.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);

  // Process two pixels per iteration — they share one chroma sample.
  // Round-to-nearest on every Q15 shift by adding 1 << 14 before the
  // `>> 15`, so 219 * (255/219 in Q15) cleanly produces 255 at the top
  // of limited-range without a 254-truncation bias.
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = ((u_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;

    // Single-round per channel keeps the math faithful to a 1×2 3x3
    // matrix multiply. All six coefficients are used; standard
    // matrices (BT.601 / 709 / 2020) have `r_u = b_v = 0` so those
    // terms vanish. YCgCo uses all six.
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Pixel x.
    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    // Pixel x+1 shares chroma.
    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

/// NV12 (semi‑planar 4:2:0, UV-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_row_impl::<false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// NV21 (semi‑planar 4:2:0, VU-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_row_impl::<true>(y, vu_half, rgb_out, width, matrix, full_range);
}

/// Shared scalar kernel for NV12 (SWAP_UV=false) and NV21
/// (SWAP_UV=true). Identical math and numerical contract to
/// [`yuv_420_to_rgb_row`]; the only difference is chroma byte order
/// in the interleaved plane. `const` generic drives compile-time
/// monomorphization — each wrapper is inlined with the branch
/// eliminated.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `uv_or_vu_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn nv12_or_nv21_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu_half.len() >= width, "chroma row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    // NV12: even byte = U, odd byte = V.
    // NV21: even byte = V, odd byte = U.
    let (u_byte, v_byte) = if SWAP_UV {
      (uv_or_vu_half[c_idx * 2 + 1], uv_or_vu_half[c_idx * 2])
    } else {
      (uv_or_vu_half[c_idx * 2], uv_or_vu_half[c_idx * 2 + 1])
    };
    let u_d = ((u_byte as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_byte as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn clamp_u8(v: i32) -> u8 {
  v.clamp(0, 255) as u8
}

// ---- High-bit-depth YUV 4:2:0 → RGB (BITS ∈ {10, 12, 14}) -------------

/// Converts one row of high-bit-depth 4:2:0 YUV (`u16` samples in the
/// low `BITS` bits of each element) directly to **8-bit** packed RGB.
///
/// `BITS` is the active input bit depth (10/12/14). Chroma bias is
/// `128 << (BITS - 8)` and the Q15 coefficients plus i32 intermediates
/// work unchanged across all three depths — only the range‑scaling
/// params ([`range_params_n`]) change with `BITS`. 16‑bit input is
/// not handled here because the i32 chroma sum would overflow.
///
/// Output semantics match [`yuv_420_to_rgb_row`]: the final clamp is
/// to `[0, 255]`, so the scale inside [`range_params_n`] targets an
/// 8‑bit output range — the kernel sheds the extra `BITS - 8` bits of
/// source precision inline rather than converting first at `BITS` and
/// then downshifting. This keeps the fast path a single Q15 shift.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  // Arithmetic notes for the 10/12/14‑bit path — every op matches
  // a concrete SIMD intrinsic so scalar and the 5 SIMD backends
  // stay **bit‑identical** on every input, including mispacked
  // `p010`‑style data with bits above BITS set:
  //
  // - The `y_off` / `bias` subtract is done in **i16 with wrap**
  //   (`wrapping_sub`) to mirror `vsubq_s16` / `_mm_sub_epi16` /
  //   `i16x8_sub`. For in‑range samples the result fits i16
  //   un‑wrapped, so this is a no‑op on the hot path.
  // - The Q15 coefficient path (`r_u * u_d + r_v * v_d + RND`) uses
  //   `wrapping_mul` / `wrapping_add` to mirror `_mm_mullo_epi32` /
  //   `_mm_add_epi32` / `vmulq_s32` / `vaddq_s32`, all of which wrap
  //   on i32 overflow by contract. Valid input cannot overflow
  //   (chroma sum max ≈ 10⁹ for 14‑bit, well within i32), so again
  //   no behavior change on the hot path; malformed input no longer
  //   trips debug‑mode overflow panics that SIMD silently wraps past.
  let y_off_i16 = y_off as i16;
  let bias_i16 = bias as i16;
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(
      (u_half[c_idx] as i16).wrapping_sub(bias_i16) as i32,
      c_scale,
    );
    let v_d = q15_scale(
      (v_half[c_idx] as i16).wrapping_sub(bias_i16) as i32,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] as i16).wrapping_sub(y_off_i16) as i32, y_scale);
    rgb_out[x * 3] = clamp_u8(y0.wrapping_add(r_chroma));
    rgb_out[x * 3 + 1] = clamp_u8(y0.wrapping_add(g_chroma));
    rgb_out[x * 3 + 2] = clamp_u8(y0.wrapping_add(b_chroma));

    let y1 = q15_scale((y[x + 1] as i16).wrapping_sub(y_off_i16) as i32, y_scale);
    rgb_out[(x + 1) * 3] = clamp_u8(y1.wrapping_add(r_chroma));
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1.wrapping_add(g_chroma));
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1.wrapping_add(b_chroma));

    x += 2;
  }
}

/// `(sample * scale_q15 + RND) >> 15` using wrapping i32 arithmetic
/// — matches the SIMD backends' implicit wrap‑on‑overflow semantics.
/// RND = `1 << 14` for round‑to‑nearest.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_scale(sample: i32, scale_q15: i32) -> i32 {
  sample.wrapping_mul(scale_q15).wrapping_add(1 << 14) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15` using wrapping i32
/// arithmetic — mirrors the SIMD `_mm_mullo_epi32` /
/// `_mm_add_epi32` / `vmulq_s32` / `vaddq_s32` chain.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_chroma(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  c_u
    .wrapping_mul(u_d)
    .wrapping_add(c_v.wrapping_mul(v_d))
    .wrapping_add(1 << 14)
    >> 15
}

/// Converts one row of high‑bit‑depth 4:2:0 YUV to **`u16`** packed
/// RGB at the **input's native bit depth** (`BITS`).
///
/// Output is **low‑bit‑packed**: for 10‑bit input each `u16` holds a
/// value in `[0, 1023]` with the upper 6 bits zero — matching
/// FFmpeg's `yuv420p10le` convention. 12‑ and 14‑bit inputs produce
/// `[0, 4095]` / `[0, 16383]` respectively, again in the low bits.
///
/// This is **not** the FFmpeg `p010` layout: `p010` puts samples in
/// the **high** 10 bits of each `u16` (effectively `sample << 6`).
/// Callers routing this output to a p010 consumer must shift left
/// by `16 - BITS`.
///
/// This is the fidelity‑preserving path: no bits are shed inside the
/// conversion, so the output retains the full dynamic range of the
/// source for HDR tone mapping, 10‑bit scene analysis, and similar
/// downstream work. Callers who only need 8‑bit output should prefer
/// [`yuv_420p_n_to_rgb_row`], which is ~2× faster.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;

  // Wrapping arithmetic throughout — see matching comment in
  // [`yuv_420p_n_to_rgb_row`]. Keeps scalar bit‑identical to every
  // SIMD backend on out‑of‑range input.
  let y_off_i16 = y_off as i16;
  let bias_i16 = bias as i16;
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(
      (u_half[c_idx] as i16).wrapping_sub(bias_i16) as i32,
      c_scale,
    );
    let v_d = q15_scale(
      (v_half[c_idx] as i16).wrapping_sub(bias_i16) as i32,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] as i16).wrapping_sub(y_off_i16) as i32, y_scale);
    rgb_out[x * 3] = y0.wrapping_add(r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = y0.wrapping_add(g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = y0.wrapping_add(b_chroma).clamp(0, out_max) as u16;

    let y1 = q15_scale((y[x + 1] as i16).wrapping_sub(y_off_i16) as i32, y_scale);
    rgb_out[(x + 1) * 3] = y1.wrapping_add(r_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 1] = y1.wrapping_add(g_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 2] = y1.wrapping_add(b_chroma).clamp(0, out_max) as u16;

    x += 2;
  }
}

/// Chroma bias for input bit depth `BITS` — `128 << (BITS - 8)`.
/// 128 for 8‑bit, 512 for 10‑bit, 2048 for 12‑bit, 8192 for 14‑bit.
/// Exposed at module visibility so SIMD backends can reuse it.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn chroma_bias<const BITS: u32>() -> i32 {
  128i32 << (BITS - 8)
}

/// Range‑scaling params `(y_off, y_scale_q15, c_scale_q15)` for the
/// high‑bit‑depth kernel family.
///
/// `BITS` is the input bit depth (10 / 12 / 14); `OUT_BITS` is the
/// target output range (8 for u8‑packed RGB, equal to `BITS` for
/// native‑depth `u16` output).
///
/// The scales are chosen so that after `((sample - y_off) * scale + RND) >> 15`
/// the result lies in `[0, (1 << OUT_BITS) - 1]` without further
/// downshifting. This keeps the fast path a single Q15 multiply for
/// both output widths.
///
/// - Full range: luma and chroma both use the same scale, mapping
///   `[0, in_max]` to `[0, out_max]`. Same shape as 8‑bit's
///   `(0, 1<<15, 1<<15)` for `BITS == OUT_BITS`.
/// - Limited range: luma maps `[16·k, 235·k]` to `[0, out_max]`,
///   chroma maps `[16·k, 240·k]` to `[0, out_max]`, where
///   `k = 1 << (BITS - 8)`. Matches FFmpeg's `AVCOL_RANGE_MPEG`
///   semantics.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn range_params_n<const BITS: u32, const OUT_BITS: u32>(
  full_range: bool,
) -> (i32, i32, i32) {
  let in_max: i64 = (1i64 << BITS) - 1;
  let out_max: i64 = (1i64 << OUT_BITS) - 1;
  if full_range {
    // `scale = round((out_max << 15) / in_max)`. For `BITS == OUT_BITS`
    // the quotient is exactly `1 << 15` (no rounding needed); for
    // 10‑bit→8‑bit it's `(255 << 15) / 1023 ≈ 8167`.
    let scale = ((out_max << 15) + in_max / 2) / in_max;
    (0, scale as i32, scale as i32)
  } else {
    let y_off = 16i32 << (BITS - 8);
    let y_range: i64 = 219i64 << (BITS - 8);
    let c_range: i64 = 224i64 << (BITS - 8);
    let y_scale = ((out_max << 15) + y_range / 2) / y_range;
    let c_scale = ((out_max << 15) + c_range / 2) / c_range;
    (y_off, y_scale as i32, c_scale as i32)
  }
}

/// Range-scaling params: `(y_off, y_scale_q15, c_scale_q15)`.
///
/// Full range: no offset, unit scales (Q15 = 2^15).
///
/// Limited range: map Y from `[16, 235]` to `[0, 255]` via
/// `y_scaled = (y - 16) * (255 / 219)`; map chroma from `[16, 240]`
/// to `[0, 255]` via `c_scaled = (c - 128) * (255 / 224)`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn range_params(full_range: bool) -> (i32, i32, i32) {
  if full_range {
    (0, 1 << 15, 1 << 15)
  } else {
    //  255 / 219 ≈ 1.164383; * 2^15 ≈ 38142.
    //  255 / 224 ≈ 1.138393; * 2^15 ≈ 37306.
    (16, 38142, 37306)
  }
}

/// Q15 YUV → RGB coefficients for a given matrix.
///
/// Full generalized 3×3 matrix:
/// - `R = Y + r_u·u_d + r_v·v_d`
/// - `G = Y + g_u·u_d + g_v·v_d`
/// - `B = Y + b_u·u_d + b_v·v_d`
///
/// where `u_d = U - 128`, `v_d = V - 128`. Standard matrices
/// (BT.601, BT.709, BT.2020-NCL, SMPTE 240M, FCC) have sparse layout
/// with `r_u = b_v = 0`; YCgCo uses all six entries.
pub(super) struct Coefficients {
  r_u: i32,
  r_v: i32,
  g_u: i32,
  g_v: i32,
  b_u: i32,
  b_v: i32,
}

impl Coefficients {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn for_matrix(m: ColorMatrix) -> Self {
    match m {
      // BT.601: r_v=1.402, g_u=-0.344136, g_v=-0.714136, b_u=1.772.
      ColorMatrix::Bt601 | ColorMatrix::Fcc => Self {
        r_u: 0,
        r_v: 45941,
        g_u: -11277,
        g_v: -23401,
        b_u: 58065,
        b_v: 0,
      },
      // BT.709: r_v=1.5748, g_u=-0.1873, g_v=-0.4681, b_u=1.8556.
      ColorMatrix::Bt709 => Self {
        r_u: 0,
        r_v: 51606,
        g_u: -6136,
        g_v: -15339,
        b_u: 60808,
        b_v: 0,
      },
      // BT.2020-NCL: r_v=1.4746, g_u=-0.164553, g_v=-0.571353, b_u=1.8814.
      ColorMatrix::Bt2020Ncl => Self {
        r_u: 0,
        r_v: 48325,
        g_u: -5391,
        g_v: -18722,
        b_u: 61653,
        b_v: 0,
      },
      // SMPTE 240M: r_v=1.576, g_u=-0.2253, g_v=-0.4767, b_u=1.826.
      ColorMatrix::Smpte240m => Self {
        r_u: 0,
        r_v: 51642,
        g_u: -7383,
        g_v: -15620,
        b_u: 59834,
        b_v: 0,
      },
      // YCgCo per H.273 MatrixCoefficients = 8.
      //   U plane → Cg, V plane → Co (biased by 128 each).
      //   R = Y - (Cg - 128) + (Co - 128) = Y - u_d + v_d
      //   G = Y + (Cg - 128)              = Y + u_d
      //   B = Y - (Cg - 128) - (Co - 128) = Y - u_d - v_d
      // Each coefficient is ±1.0 → ±32768 in Q15.
      ColorMatrix::YCgCo => Self {
        r_u: -32768,
        r_v: 32768,
        g_u: 32768,
        g_v: 0,
        b_u: -32768,
        b_v: -32768,
      },
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_u(&self) -> i32 {
    self.r_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn r_v(&self) -> i32 {
    self.r_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_u(&self) -> i32 {
    self.g_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn g_v(&self) -> i32 {
    self.g_v
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_u(&self) -> i32 {
    self.b_u
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn b_v(&self) -> i32 {
    self.b_v
  }
}

// ---- RGB → HSV ----------------------------------------------------------

// ---- HSV division LUTs (OpenCV `cv2.COLOR_RGB2HSV` compatible) --------
//
// Replace the f32 divisions in the scalar HSV path with an integer
// multiply + table lookup. Produces byte‑exact output against OpenCV
// for 8‑bit RGB → HSV on every pixel.
//
// `HSV_SHIFT = 12` gives 1044480 / v (saturation divisor) and 122880 /
// delta (hue divisor) as the raw Q12 reciprocals. Both fit in i32, and
// the subsequent `diff * table[x]` product (max 255 × 1044480 ≈ 2.66e8)
// also fits in i32 comfortably.
//
// Total `.rodata` cost: 2 KB (two 256‑entry i32 tables). Always fits
// in L1D on every modern CPU, so lookups average ~4 cycles.

const HSV_SHIFT: u32 = 12;
const HSV_RND: i32 = 1 << (HSV_SHIFT - 1);

/// `sdiv_table[v] = round((255 << 12) / v)`. `sdiv_table[0] = 0`
/// (saturation is undefined at v=0; the caller forces `s = 0` there).
const SDIV_TABLE: [i32; 256] = {
  let mut t = [0i32; 256];
  let mut i = 1usize;
  while i < 256 {
    let n: i32 = 255 << HSV_SHIFT;
    t[i] = (n + (i as i32) / 2) / (i as i32);
    i += 1;
  }
  t
};

/// `hdiv_table[delta] = round((30 << 12) / delta)`. The factor is 30
/// (not 60) because OpenCV's u8 hue range is `[0, 180)` instead of
/// `[0, 360)` — every 2° collapses to one unit. `hdiv_table[0] = 0`
/// (hue is undefined at delta=0; the caller forces `h = 0` there).
const HDIV_TABLE: [i32; 256] = {
  let mut t = [0i32; 256];
  let mut i = 1usize;
  while i < 256 {
    let n: i32 = 30 << HSV_SHIFT;
    t[i] = (n + (i as i32) / 2) / (i as i32);
    i += 1;
  }
  t
};

/// Converts one row of packed RGB to three planar HSV bytes matching
/// OpenCV `cv2.COLOR_RGB2HSV` semantics: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`.
///
/// Uses integer LUT arithmetic (no f32 divisions), producing byte‑
/// exact output against OpenCV's uint8 HSV conversion.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(h_out.len() >= width, "H row too short");
  debug_assert!(s_out.len() >= width, "S row too short");
  debug_assert!(v_out.len() >= width, "V row too short");
  for x in 0..width {
    let r = rgb[x * 3] as i32;
    let g = rgb[x * 3 + 1] as i32;
    let b = rgb[x * 3 + 2] as i32;
    let (h, s, v) = rgb_to_hsv_pixel(r, g, b);
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = v;
  }
}

/// Scalar RGB → HSV for a single pixel, using the shared division LUTs.
/// All arithmetic is integer; the two divisions `s = 255*delta/v` and
/// `h = 30*diff/delta` become `(operand * table[divisor] + RND) >> 12`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb_to_hsv_pixel(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
  let v = r.max(g.max(b));
  let min = r.min(g.min(b));
  let delta = v - min;

  // S = round(255 * delta / v), s = 0 when v = 0.
  //
  // SDIV_TABLE[0] = 0 so the expression evaluates to (delta * 0 + RND)
  // >> 12 = 0 when v = 0. Delta is also 0 in that case (min = v = 0),
  // but the explicit table entry makes the reasoning obvious.
  let s = ((delta * SDIV_TABLE[v as usize]) + HSV_RND) >> HSV_SHIFT;

  let h = if delta == 0 {
    0
  } else if v == r {
    let diff = g - b;
    let h_raw = ((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT;
    if h_raw < 0 { h_raw + 180 } else { h_raw }
  } else if v == g {
    let diff = b - r;
    (((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT) + 60
  } else {
    let diff = r - g;
    (((diff * HDIV_TABLE[delta as usize]) + HSV_RND) >> HSV_SHIFT) + 120
  };

  (h.clamp(0, 179) as u8, s.clamp(0, 255) as u8, v as u8)
}

// ---- BGR ↔ RGB byte swap ------------------------------------------------

/// Swaps the outer two channels of each packed RGB / BGR triple
/// (byte 0 ↔ byte 2), leaving the middle byte (G) untouched.
///
/// This is the shared implementation behind both `bgr_to_rgb_row` and
/// `rgb_to_bgr_row` — the transformation is a self‑inverse.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");
  for x in 0..width {
    let i = x * 3;
    output[i] = input[i + 2];
    output[i + 1] = input[i + 1];
    output[i + 2] = input[i];
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // ---- yuv_420_to_rgb_row ----------------------------------------------

  #[test]
  fn yuv420_rgb_black() {
    // Full-range Y=0, neutral chroma → black.
    let y = [0u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
  }

  #[test]
  fn yuv420_rgb_white_full_range() {
    let y = [255u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
  }

  #[test]
  fn yuv420_rgb_gray_is_gray() {
    let y = [128u8; 4];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    for x in 0..4 {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(r, g);
      assert_eq!(g, b);
      assert!(r.abs_diff(128) <= 1, "got {r}");
    }
  }

  #[test]
  fn yuv420_rgb_chroma_shared_across_pair() {
    // Two Y values with same chroma: differing Y produces differing
    // luminance but same chroma-driven offsets. Validates that pixel x
    // and x+1 share the upsampled chroma sample.
    let y = [50u8, 200, 50, 200];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    // With neutral chroma, output is gray = Y.
    assert_eq!(rgb[0], 50);
    assert_eq!(rgb[3], 200);
    assert_eq!(rgb[6], 50);
    assert_eq!(rgb[9], 200);
  }

  #[test]
  fn yuv420_rgb_limited_range_black_and_white() {
    // Y=16 → black, Y=235 → white in limited range.
    let y = [16u8, 16, 235, 235];
    let u = [128u8; 2];
    let v = [128u8; 2];
    let mut rgb = [0u8; 12];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
    for x in 0..2 {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!((r, g, b), (0, 0, 0), "limited-range Y=16 should be black");
    }
    for x in 2..4 {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(
        (r, g, b),
        (255, 255, 255),
        "limited-range Y=235 should be white"
      );
    }
  }

  #[test]
  fn yuv420_rgb_ycgco_neutral_is_gray() {
    // Y=128, Cg=128 (U), Co=128 (V) — neutral chroma → gray.
    let y = [128u8; 2];
    let u = [128u8; 1]; // Cg
    let v = [128u8; 1]; // Co
    let mut rgb = [0u8; 6];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
    for px in rgb.chunks(3) {
      assert!(px[0].abs_diff(128) <= 1, "RGB should be gray, got {rgb:?}");
      assert_eq!(px[0], px[1]);
      assert_eq!(px[1], px[2]);
    }
  }

  #[test]
  fn yuv420_rgb_ycgco_high_cg_is_green() {
    // U plane = Cg; Cg > 128 means green-ward shift.
    // Expected math (Y=128, Cg=200, Co=128):
    //   u_d = 72, v_d = 0
    //   R = 128 - 72 + 0 = 56
    //   G = 128 + 72     = 200
    //   B = 128 - 72 - 0 = 56
    let y = [128u8; 2];
    let u = [200u8; 1]; // Cg = 200 (green-ward)
    let v = [128u8; 1]; // Co neutral
    let mut rgb = [0u8; 6];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
    for px in rgb.chunks(3) {
      // Allow ±1 for Q15 rounding. RGB order: [R, G, B].
      assert!(px[0].abs_diff(56) <= 1, "expected R≈56, got {rgb:?}");
      assert!(px[1].abs_diff(200) <= 1, "expected G≈200, got {rgb:?}");
      assert!(px[2].abs_diff(56) <= 1, "expected B≈56, got {rgb:?}");
    }
  }

  #[test]
  fn yuv420_rgb_ycgco_high_co_is_red() {
    // V plane = Co; Co > 128 means orange/red-ward shift.
    // Expected (Y=128, Cg=128, Co=200):
    //   u_d = 0, v_d = 72
    //   R = 128 - 0 + 72 = 200
    //   G = 128 + 0      = 128
    //   B = 128 - 0 - 72 = 56
    let y = [128u8; 2];
    let u = [128u8; 1]; // Cg neutral
    let v = [200u8; 1]; // Co = 200 (orange-ward)
    let mut rgb = [0u8; 6];
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb, 2, ColorMatrix::YCgCo, true);
    for px in rgb.chunks(3) {
      // RGB order: [R, G, B].
      assert!(px[0].abs_diff(200) <= 1, "expected R≈200, got {rgb:?}");
      assert!(px[1].abs_diff(128) <= 1, "expected G≈128, got {rgb:?}");
      assert!(px[2].abs_diff(56) <= 1, "expected B≈56, got {rgb:?}");
    }
  }

  #[test]
  fn yuv420_rgb_bt601_vs_bt709_differ_for_chroma() {
    // Moderate chroma (V=200) so the red channel doesn't saturate on
    // either matrix — saturating both and then diffing gives zero.
    let y = [128u8; 2];
    let u = [128u8; 1];
    let v = [200u8; 1];
    let mut b601 = [0u8; 6];
    let mut b709 = [0u8; 6];
    yuv_420_to_rgb_row(&y, &u, &v, &mut b601, 2, ColorMatrix::Bt601, true);
    yuv_420_to_rgb_row(&y, &u, &v, &mut b709, 2, ColorMatrix::Bt709, true);
    // Sum of per-channel absolute differences — robust to which
    // particular channel the two matrices disagree on.
    let sad: i32 = b601
      .iter()
      .zip(b709.iter())
      .map(|(a, b)| (*a as i32 - *b as i32).abs())
      .sum();
    assert!(
      sad > 20,
      "BT.601 vs BT.709 outputs should materially differ: {b601:?} vs {b709:?}"
    );
  }

  // ---- rgb_to_hsv_row --------------------------------------------------

  #[test]
  fn hsv_gray_has_no_hue_no_sat() {
    let rgb = [128u8; 3];
    let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
    assert_eq!((h[0], s[0], v[0]), (0, 0, 128));
  }

  #[test]
  fn hsv_pure_red_matches_opencv() {
    // OpenCV RGB2HSV: red = (R=255, G=0, B=0) → H = 0, S = 255, V = 255.
    let rgb = [255u8, 0, 0];
    let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
    assert_eq!((h[0], s[0], v[0]), (0, 255, 255));
  }

  #[test]
  fn hsv_pure_green_matches_opencv() {
    // Green (R=0, G=255, B=0) → H = 60 in OpenCV 8-bit (120° / 2).
    let rgb = [0u8, 255, 0];
    let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
    assert_eq!((h[0], s[0], v[0]), (60, 255, 255));
  }

  #[test]
  fn hsv_pure_blue_matches_opencv() {
    // Blue (R=0, G=0, B=255) → H = 120 (240° / 2).
    let rgb = [0u8, 0, 255];
    let (mut h, mut s, mut v) = ([0u8; 1], [0u8; 1], [0u8; 1]);
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, 1);
    assert_eq!((h[0], s[0], v[0]), (120, 255, 255));
  }

  // ---- yuv_420p_n_to_rgb_row (10-bit → u8) -----------------------------

  #[test]
  fn yuv420p10_rgb_black_full_range() {
    // Y=0, neutral chroma (512 in 10-bit) → black.
    let y = [0u16; 4];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
  }

  #[test]
  fn yuv420p10_rgb_white_full_range() {
    // 10-bit full-range white is Y=1023.
    let y = [1023u16; 4];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
  }

  #[test]
  fn yuv420p10_rgb_gray_is_gray() {
    // Mid-gray 10-bit Y=512 ↔ 8-bit 128. Within ±1 for Q15 rounding.
    let y = [512u16; 4];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    for x in 0..4 {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(r, g);
      assert_eq!(g, b);
      assert!(r.abs_diff(128) <= 1, "got {r}");
    }
  }

  #[test]
  fn yuv420p10_rgb_limited_range_black_and_white() {
    // 10-bit limited: Y=64 → black, Y=940 → white.
    let y = [64u16, 64, 940, 940];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, false);
    assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
    assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
    assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
    assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
  }

  #[test]
  fn yuv420p10_rgb_chroma_shared_across_pair() {
    // Two 10-bit Y values sharing chroma: output is gray = Y>>2.
    let y = [200u16, 800, 200, 800];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    // Full-range 10→8 scale = 255/1023, so Y=200 → 50, Y=800 → 199.4 → 199.
    // Allow ±1 for Q15 rounding.
    assert!(rgb[0].abs_diff(50) <= 1, "got {}", rgb[0]);
    assert!(rgb[3].abs_diff(199) <= 1, "got {}", rgb[3]);
    assert!(rgb[6].abs_diff(50) <= 1, "got {}", rgb[6]);
    assert!(rgb[9].abs_diff(199) <= 1, "got {}", rgb[9]);
  }

  // ---- yuv_420p_n_to_rgb_u16_row (10-bit → 10-bit u16) ----------------

  #[test]
  fn yuv420p10_rgb_u16_black_full_range() {
    let y = [0u16; 4];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u16; 12];
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
  }

  #[test]
  fn yuv420p10_rgb_u16_white_full_range() {
    // 10-bit input Y=1023, full-range scale=1 → output Y=1023 on each channel.
    let y = [1023u16; 4];
    let u = [512u16; 2];
    let v = [512u16; 2];
    let mut rgb = [0u16; 12];
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
  }

  #[test]
  fn yuv420p10_rgb_u16_limited_range_endpoints() {
    // Limited-range: Y=64 → 0, Y=940 → 1023 in 10-bit output.
    let y = [64u16, 940];
    let u = [512u16; 1];
    let v = [512u16; 1];
    let mut rgb = [0u16; 6];
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb, 2, ColorMatrix::Bt709, false);
    assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
    assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
  }

  #[test]
  fn yuv420p10_rgb_u16_preserves_full_10bit_precision() {
    // Sanity: the u16 path retains native-depth precision, so two
    // inputs that round to the same u8 are distinguishable in u16.
    // Full-range Y=200 vs Y=201: same u8 output (50 vs 50) but
    // distinct u16 outputs (200 vs 201).
    let y = [200u16, 201];
    let u = [512u16; 1];
    let v = [512u16; 1];
    let mut rgb8 = [0u8; 6];
    let mut rgb16 = [0u16; 6];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb8, 2, ColorMatrix::Bt601, true);
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb16, 2, ColorMatrix::Bt601, true);
    assert_eq!(rgb8[0], rgb8[3]);
    assert_ne!(rgb16[0], rgb16[3]);
  }

  #[test]
  fn yuv420p10_bt709_ycgco_differ_for_chroma() {
    // Non-neutral chroma — different matrices produce different RGB.
    let y = [512u16; 2];
    let u = [512u16; 1];
    let v = [800u16; 1];
    let mut bt709 = [0u8; 6];
    let mut ycgco = [0u8; 6];
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut bt709, 2, ColorMatrix::Bt709, true);
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut ycgco, 2, ColorMatrix::YCgCo, true);
    let sad: i32 = bt709
      .iter()
      .zip(ycgco.iter())
      .map(|(a, b)| (*a as i32 - *b as i32).abs())
      .sum();
    assert!(
      sad > 20,
      "matrices should materially differ: {bt709:?} vs {ycgco:?}"
    );
  }
}
