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
  yuv_420_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Same as [`yuv_420_to_rgb_row`] but writes packed `R, G, B, A`
/// quadruplets, with `A = 0xFF` (opaque) for every pixel. The first
/// three bytes per pixel are byte-identical to what
/// [`yuv_420_to_rgb_row`] would write — only the per-pixel stride
/// (4 vs 3) and the alpha byte differ. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
}

/// Shared scalar kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false`,
/// 3 bytes / pixel) and [`yuv_420_to_rgba_row`] (`ALPHA = true`,
/// 4 bytes / pixel — 4th is opaque `0xFF`). The math is identical;
/// only the per-pixel store differs. `const` generic drives
/// compile-time monomorphization — each public wrapper is inlined
/// with the branch eliminated.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

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
    let r0 = clamp_u8(y0 + r_chroma);
    let g0 = clamp_u8(y0 + g_chroma);
    let b0 = clamp_u8(y0 + b_chroma);
    out[x * bpp] = r0;
    out[x * bpp + 1] = g0;
    out[x * bpp + 2] = b0;
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    // Pixel x+1 shares chroma.
    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let r1 = clamp_u8(y1 + r_chroma);
    let g1 = clamp_u8(y1 + g_chroma);
    let b1 = clamp_u8(y1 + b_chroma);
    out[(x + 1) * bpp] = r1;
    out[(x + 1) * bpp + 1] = g1;
    out[(x + 1) * bpp + 2] = b1;
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// NV12 (semi‑planar 4:2:0, UV-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
    y, uv_half, rgb_out, width, matrix, full_range,
  );
}

/// NV21 (semi‑planar 4:2:0, VU-ordered) → packed RGB. Thin wrapper
/// over [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
    y, vu_half, rgb_out, width, matrix, full_range,
  );
}

/// NV12 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv12_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
    y, uv_half, rgba_out, width, matrix, full_range,
  );
}

/// NV21 → packed `R, G, B, A` quadruplets with constant `A = 0xFF`.
/// First three bytes per pixel are byte-identical to
/// [`nv21_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
    y, vu_half, rgba_out, width, matrix, full_range,
  );
}

/// Shared scalar kernel for NV12 (`SWAP_UV = false`) / NV21
/// (`SWAP_UV = true`) at 3 bpp (`ALPHA = false`) or 4 bpp + opaque
/// alpha (`ALPHA = true`). Identical math to [`yuv_420_to_rgb_row`];
/// the only differences are chroma byte order in the interleaved
/// plane and the per-pixel store stride. Both `const` generics drive
/// compile-time monomorphization — each wrapper is inlined with both
/// branches eliminated.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `uv_or_vu_half.len() >= width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu_half.len() >= width, "chroma row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

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
    let r0 = clamp_u8(y0 + r_chroma);
    let g0 = clamp_u8(y0 + g_chroma);
    let b0 = clamp_u8(y0 + b_chroma);
    out[x * bpp] = r0;
    out[x * bpp + 1] = g0;
    out[x * bpp + 2] = b0;
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let r1 = clamp_u8(y1 + r_chroma);
    let g1 = clamp_u8(y1 + g_chroma);
    let b1 = clamp_u8(y1 + b_chroma);
    out[(x + 1) * bpp] = r1;
    out[(x + 1) * bpp + 1] = g1;
    out[(x + 1) * bpp + 2] = b1;
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// NV24 (semi-planar 4:4:4, UV-ordered) → packed RGB. Thin wrapper
/// over [`nv24_or_nv42_to_rgb_row_impl`] with `SWAP_UV = false`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_row_impl::<false>(y, uv, rgb_out, width, matrix, full_range);
}

/// NV42 (semi-planar 4:4:4, VU-ordered) → packed RGB. Thin wrapper
/// over [`nv24_or_nv42_to_rgb_row_impl`] with `SWAP_UV = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  nv24_or_nv42_to_rgb_row_impl::<true>(y, vu, rgb_out, width, matrix, full_range);
}

/// Shared scalar kernel for NV24 (SWAP_UV=false) and NV42
/// (SWAP_UV=true). Identical math and numerical contract to
/// [`yuv_420_to_rgb_row`]; the difference from NV12/NV21 is
/// 4:4:4 — one UV pair per Y pixel, no chroma upsampling.
/// No width parity constraint.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_or_vu.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn nv24_or_nv42_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_or_vu.len() >= 2 * width, "chroma row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel. No upsampling.
    let (u_byte, v_byte) = if SWAP_UV {
      (uv_or_vu[x * 2 + 1], uv_or_vu[x * 2])
    } else {
      (uv_or_vu[x * 2], uv_or_vu[x * 2 + 1])
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
  }
}

/// YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// One UV pair per Y pixel, U/V from separate planes. Same
/// arithmetic as [`nv24_to_rgb_row`] (4:4:4 semi-planar) but
/// without the deinterleave step — U and V come pre-separated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar → packed `R, G, B, A` quadruplets with constant
/// `A = 0xFF`. First three bytes per pixel are byte-identical to
/// [`yuv_444_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
}

/// Shared scalar kernel for [`yuv_444_to_rgb_row`] (`ALPHA = false`,
/// 3 bpp) and [`yuv_444_to_rgba_row`] (`ALPHA = true`, 4 bpp + opaque
/// alpha). Math is identical; only the per-pixel store stride
/// differs. `const` generic monomorphizes per call site, so the
/// `if ALPHA` branches are eliminated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel, no subsampling.
    let u_d = ((u[x] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v[x] as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
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
  // Compile-time guard — fails monomorphization for any BITS outside
  // {9, 10, 12, 14}. 16 would overflow the Q15 chroma sum (16-bit lives
  // in `yuv_420p16_to_rgb_row`'s i64 chroma family); 8 belongs to the
  // non-const-generic `yuv_420_to_rgb_row`. Without this guard a release
  // build instantiating ::<16> would silently produce wrong output.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let mask = bits_mask::<BITS>();

  // Every sample is AND‑masked to the low `BITS` bits on load. This
  // eliminates architecture‑dependent divergence on mispacked input
  // (e.g. `p010`‑style buffers where the 10 active bits sit in the
  // high bits of each `u16`): after masking, every backend sees the
  // same in‑range sample, so the whole Q15 pipeline stays bounded
  // (intermediate chroma sums fit i16 as designed, no saturating
  // narrow loses information). For valid input every mask is a
  // no‑op. For malformed input the "wrong" output is identical
  // across scalar + all 5 SIMD backends.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale((u_half[c_idx] & mask) as i32 - bias, c_scale);
    let v_d = q15_scale((v_half[c_idx] & mask) as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] & mask) as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    let y1 = q15_scale((y[x + 1] & mask) as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

/// `(sample * scale_q15 + RND) >> 15`. With input masked to BITS,
/// the `sample * scale` product cannot overflow i32 for any
/// reasonable `OUT_BITS ≤ 16`, so plain arithmetic is sufficient.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_scale(sample: i32, scale_q15: i32) -> i32 {
  (sample * scale_q15 + (1 << 14)) >> 15
}

/// `(c_u * u_d + c_v * v_d + RND) >> 15`. Chroma sum max ≈ 10⁹ for
/// 14‑bit masked input, well within i32.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_chroma(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  (c_u * u_d + c_v * v_d + (1 << 14)) >> 15
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
  // Compile-time guard — see note on `yuv_420p_n_to_rgb_row`. The
  // 16-bit u16-output path is `yuv_420p16_to_rgb_u16_row` (i64 chroma
  // family).
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let mask = bits_mask::<BITS>();

  // Every sample AND‑masked to the low `BITS` bits — see matching
  // comment in [`yuv_420p_n_to_rgb_row`]. Critical for the native‑
  // depth u16 output path: `range_params_n::<10, 10>` uses
  // `y_scale = c_scale = 32768` (unit Q15 for BITS==OUT_BITS full
  // range), so an unmasked out‑of‑range sample would push `u_d` /
  // `v_d` to ±32256 and the subsequent `coeff * v_d` exceeds i16
  // range — breaking the SIMD kernels' `vqmovn_s32` narrow step.
  // Masking keeps every intermediate bounded by design.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale((u_half[c_idx] & mask) as i32 - bias, c_scale);
    let v_d = q15_scale((v_half[c_idx] & mask) as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] & mask) as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;

    let y1 = q15_scale((y[x + 1] & mask) as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = (y1 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;

    x += 2;
  }
}

/// YUV 4:4:4 planar high‑bit‑depth → **u8** packed RGB. Const‑generic
/// over `BITS ∈ {10, 12, 14}`. 1:1 chroma per Y pixel (no chroma
/// pair, no upsampling). Math is identical to
/// [`yuv_420p_n_to_rgb_row`] except each pixel gets its own U / V.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Compile-time guard — fails monomorphization for any BITS outside
  // {10, 12, 14}. The 16-bit path lives in `yuv_444p16_to_rgb_row`
  // (i32 u8-output kernel family). Without this guard a caller
  // invoking ::<16> would reach the NEON clamp where
  // `(1 << BITS) - 1 as i16` silently wraps to -1.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let mask = bits_mask::<BITS>();

  for x in 0..width {
    // 4:4:4: one UV pair per pixel, no subsampling.
    let u_d = q15_scale((u[x] & mask) as i32 - bias, c_scale);
    let v_d = q15_scale((v[x] & mask) as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] & mask) as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);
  }
}

/// YUV 4:4:4 planar high‑bit‑depth → **native‑depth `u16`** packed RGB.
/// Const‑generic over `BITS ∈ {10, 12, 14}`. Low‑bit‑packed output.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Compile-time guard — see note on `yuv_444p_n_to_rgb_row`. The
  // 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row` (i64
  // chroma family).
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let mask = bits_mask::<BITS>();

  for x in 0..width {
    let u_d = q15_scale((u[x] & mask) as i32 - bias, c_scale);
    let v_d = q15_scale((v[x] & mask) as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] & mask) as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
  }
}

// ---- 16-bit YUV 4:2:0 → RGB (parallel kernel family) -------------------
//
// At 16 bits the chroma multiply-add `c_u * u_d + c_v * v_d` splits
// into two regimes by output target:
//
// - **16 → u8**: the Q15 scale knocks `u_d` / `v_d` down to u8 range
//   (max ±150 at limited range, ±128 at full). Products like
//   `60808 * 150 = 9.1M` and their sums stay well within i32, so the
//   i32 pipeline used by 10/12/14 works unchanged at BITS = 16 — the
//   kernels below reuse that structure without widening.
// - **16 → u16**: the Q15 scale is a near-identity (32768 at full
//   range), so `u_d` / `v_d` can reach ±32768. `coeff * u_d` alone
//   reaches ~1.99·10⁹ (close to i32 max); the full chroma sum
//   reaches ~3.68·10⁹ — overflows i32. The u16 kernels below widen
//   the chroma multiply-add to i64 (via [`q15_chroma64`]) and narrow
//   back after the `>> 15`.
//
// All four functions are dedicated 16-bit entry points (not
// const-generic) so each monomorphization picks the right precision
// path without a runtime branch.

/// `(c_u * u_d + c_v * v_d + RND) >> 15` computed in i64. Chroma sum
/// max ≈ 4.3·10⁹ at 16-bit limited range — above i32 but well within
/// i64. Result after the shift is bounded by ~130 000 so the final
/// `as i32` narrow is lossless.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_chroma64(c_u: i32, u_d: i32, c_v: i32, v_d: i32) -> i32 {
  let sum = (c_u as i64) * (u_d as i64) + (c_v as i64) * (v_d as i64);
  ((sum + (1 << 14)) >> 15) as i32
}

/// `(sample * scale_q15 + RND) >> 15` computed in i64. For 16-bit
/// samples at limited-range 16 → u16 scaling, `sample * y_scale` can
/// reach ~2.35·10⁹ — just over i32::MAX — when unclamped `u16` input
/// exceeds the nominal limited-range Y max. Result after the shift
/// is bounded by ~65 536 so the final `as i32` narrow is lossless.
#[cfg_attr(not(tarpaulin), inline(always))]
fn q15_scale64(sample: i32, scale_q15: i32) -> i32 {
  (((sample as i64) * (scale_q15 as i64) + (1 << 14)) >> 15) as i32
}

/// Converts one row of **16-bit** YUV 4:2:0 (samples in the full
/// `u16` range) to **8-bit** packed RGB. At 16 → u8 the Q15 scale
/// confines chroma to u8 range, so the i32 chroma pipeline used by
/// 10/12/14 applies unchanged here — this kernel is structurally
/// identical to [`yuv_420p_n_to_rgb_row`] at a hypothetical
/// `BITS = 16`, just without the AND-mask (no upper-bit-zero
/// guarantee to enforce at 16 bits).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgb_row(
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
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  // No AND-mask needed at 16-bit — every u16 is already a valid
  // sample. `q15_chroma` (i32) is enough for u8 output because the
  // output-target scaling keeps `u_d * coeff` well within i32.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(u_half[c_idx] as i32 - bias, c_scale);
    let v_d = q15_scale(v_half[c_idx] as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    let y1 = q15_scale(y[x + 1] as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

/// Converts one row of **16-bit** YUV 4:2:0 to **native-depth `u16`**
/// packed RGB — full-range output in `[0, 65535]`. **Runs the
/// chroma matrix multiply in i64** to accommodate the wider
/// `coeff × u_d` product at 16 → 16-bit scaling.
///
/// # Panics (debug builds)
///
/// Same contract as [`yuv_420p16_to_rgb_row`] plus `rgb_out` is
/// measured in `u16` elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p16_to_rgb_u16_row(
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
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(u_half[c_idx] as i32 - bias, c_scale);
    let v_d = q15_scale(v_half[c_idx] as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;

    let y1 = q15_scale64(y[x + 1] as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = (y1 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;

    x += 2;
  }
}

/// YUV 4:4:4 planar **16‑bit** → packed **8‑bit** RGB. Same i32
/// chroma pipeline as 10/12/14 (output‑range scaling keeps `coeff × u_d`
/// inside i32 for u8 target). 1:1 chroma per Y pixel, no width parity.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_d = q15_scale(u[x] as i32 - bias, c_scale);
    let v_d = q15_scale(v[x] as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);
  }
}

/// YUV 4:4:4 planar **16‑bit** → packed **native‑depth `u16`** RGB.
/// Widens chroma matrix multiply to i64 (Bt2020 `b_u × u_d` reaches
/// ~2.31·10⁹ at limited‑range 16→u16 — overflows i32). Y path widens
/// via [`q15_scale64`] to handle unclamped Y samples above the
/// limited‑range nominal max.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  for x in 0..width {
    let u_d = q15_scale(u[x] as i32 - bias, c_scale);
    let v_d = q15_scale(v[x] as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
  }
}

/// Converts one row of **P016** (semi-planar 4:2:0 with UV
/// interleaved, full `u16` samples) to **8-bit** packed RGB. At 16
/// bits there is no "high-bit-packed" vs "low-bit-packed" distinction
/// (every bit is active), so this kernel matches
/// [`yuv_420p16_to_rgb_row`] semantically — only the chroma plane
/// layout differs (interleaved vs. two half-width planes). Uses the
/// i32 chroma pipeline (same reasoning as the planar u8 kernel).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2];
    let v_sample = uv_half[c_idx * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    let y1 = q15_scale(y[x + 1] as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

/// Converts one row of **P016** to **native-depth `u16`** packed
/// RGB — full-range output in `[0, 65535]`. Chroma matrix multiply
/// runs in i64 (same reasoning as [`yuv_420p16_to_rgb_u16_row`]).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2];
    let v_sample = uv_half[c_idx * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;

    let y1 = q15_scale64(y[x + 1] as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = (y1 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;

    x += 2;
  }
}

// ---- P010 (semi-planar 10-bit, high-bit-packed) → RGB ------------------

/// Converts one row of P010 (semi‑planar 4:2:0 with UV interleaved,
/// `BITS` active bits in the **high** `BITS` of each `u16`) to
/// **8‑bit** packed RGB.
///
/// Structurally identical to [`nv12_to_rgb_row`] plus the per‑sample
/// shift: each `u16` load is extracted to its `BITS`‑bit value via
/// `sample >> (16 - BITS)`, then the same Q15 pipeline as
/// [`yuv_420p_n_to_rgb_row`] runs with the same `BITS`. For `BITS ==
/// 10` this is P010 (`>> 6`); for `BITS == 12` it's P012 (`>> 4`).
/// Mispacked input — e.g. a low‑bit‑packed buffer handed to this
/// kernel — has its active low bits discarded (producing near‑black
/// output), matching every SIMD backend.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // High-bit-packed Pn kernels are only defined for BITS in {10, 12}.
  // Outside that set, `16 - BITS` could under/overflow and the Q15
  // coefficient table has no corresponding entry. Caught here before
  // the SIMD dispatcher hands control to unsafe code.
  debug_assert!(
    BITS == 10 || BITS == 12,
    "p_n_to_rgb_row only supports BITS in {{10, 12}}"
  );
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let shift = 16 - BITS;

  // Each `u16` load is converted to its `BITS`-bit sample with
  // `>> (16 - BITS)` — 6 for P010, 4 for P012. Extracts the upper
  // bits and leaves the result in `[0, (1 << BITS) - 1]`. If
  // low-packed input (`yuv420p10le`, `yuv420p12le`) is handed to
  // this kernel by mistake, the shift discards the active low bits
  // rather than recovering the intended value. No hot-path cost:
  // one shift per load.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2] >> shift;
    let v_sample = uv_half[c_idx * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);

    let y1 = q15_scale((y[x + 1] >> shift) as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = clamp_u8(y1 + r_chroma);
    rgb_out[(x + 1) * 3 + 1] = clamp_u8(y1 + g_chroma);
    rgb_out[(x + 1) * 3 + 2] = clamp_u8(y1 + b_chroma);

    x += 2;
  }
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0
/// (`BITS` ∈ {10, 12}: P010, P012) to **native‑depth `u16`**
/// packed RGB — samples are **low‑bit‑packed** on output
/// (`[0, (1 << BITS) - 1]` in the low bits of each `u16`, upper bits
/// zero), matching the `yuv420p10le` / `yuv420p12le` convention —
/// **not** the P010/P012 high‑bit packing. Callers feeding a P010/
/// P012 consumer must shift the output left by `16 - BITS`.
///
/// Mirrors [`yuv_420p_n_to_rgb_u16_row`] on the math side; the only
/// differences are the input shift (`sample >> (16 - BITS)` to
/// extract the `BITS`-bit value from the high-bit packing) and the
/// interleaved UV layout.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // See `p_n_to_rgb_row` for the BITS range rationale. Duplicated
  // here so either entry point catches misuse on its own.
  debug_assert!(
    BITS == 10 || BITS == 12,
    "p_n_to_rgb_u16_row only supports BITS in {{10, 12}}"
  );
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let shift = 16 - BITS;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = uv_half[c_idx * 2] >> shift;
    let v_sample = uv_half[c_idx * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;

    let y1 = q15_scale((y[x + 1] >> shift) as i32 - y_off, y_scale);
    rgb_out[(x + 1) * 3] = (y1 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[(x + 1) * 3 + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;

    x += 2;
  }
}

// ---- Pn 4:4:4 (semi-planar high-bit-packed) → RGB ----------------------
//
// Mirrors `p_n_to_rgb_*<BITS>` but with full-width interleaved UV: one
// `U, V` pair per pixel (= `2 * width` u16 elements per row), no
// horizontal duplication. Same `>> (16 - BITS)` extraction at load
// time. BITS ∈ {10, 12} on the i32 Q15 pipeline; BITS = 16 lives in
// `p_n_444_16_to_rgb_*` because the chroma multiply-add overflows
// i32 at u16 output (same rationale as p16 / yuv_444p16).

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **8-bit** packed RGB. `BITS ∈ {10, 12}`. Each `u16` load is
/// shifted right by `16 - BITS` to extract the active value before
/// running the standard Q15 i32 pipeline.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let shift = 16 - BITS;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel — uv_full[x*2] = U, uv_full[x*2+1] = V.
    let u_sample = uv_full[x * 2] >> shift;
    let v_sample = uv_full[x * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);
  }
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **native-depth `u16`** packed RGB — low-bit-packed output (the
/// `BITS` active bits in the **low** bits of each `u16`, upper bits
/// zero), matching the [`yuv_444p_n_to_rgb_u16_row`] convention.
/// `BITS ∈ {10, 12}`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let shift = 16 - BITS;

  for x in 0..width {
    let u_sample = uv_full[x * 2] >> shift;
    let v_sample = uv_full[x * 2 + 1] >> shift;
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((y[x] >> shift) as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
  }
}

/// Converts one row of P416 (semi-planar 4:4:4, 16-bit, full UV) to
/// **8-bit** packed RGB. Y and chroma both stay on i32 — same logic
/// as `p16_to_rgb_row` plus the full-width UV layout.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_sample = uv_full[x * 2];
    let v_sample = uv_full[x * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = clamp_u8(y0 + r_chroma);
    rgb_out[x * 3 + 1] = clamp_u8(y0 + g_chroma);
    rgb_out[x * 3 + 2] = clamp_u8(y0 + b_chroma);
  }
}

/// Converts one row of P416 to **native-depth `u16`** packed RGB —
/// full-range output in `[0, 65535]`. Chroma multiply-add runs in i64
/// (same rationale as `p16_to_rgb_u16_row` and
/// `yuv_444p16_to_rgb_u16_row`: `coeff × u_d` overflows i32 at 16
/// bits for the BT.2020 blue coefficient).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  for x in 0..width {
    let u_sample = uv_full[x * 2];
    let v_sample = uv_full[x * 2 + 1];
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(y[x] as i32 - y_off, y_scale);
    rgb_out[x * 3] = (y0 + r_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    rgb_out[x * 3 + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
  }
}

/// Compile‑time sample mask for `BITS`: `(1 << BITS) - 1` as `u16`.
/// Returns `0x03FF` for 10‑bit, `0x0FFF` for 12‑bit, `0x3FFF` for
/// 14‑bit. SIMD backends splat this into a vector constant and AND
/// every load against it.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) const fn bits_mask<const BITS: u32>() -> u16 {
  ((1u32 << BITS) - 1) as u16
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
    // 10‑bit→8‑bit it's `(255 << 15) / 1023 ≈ 8167.5`, which rounds to 8168.
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

// =============================================================================
// Bayer demosaic + WB + CCM
// =============================================================================

/// Scalar bilinear demosaic + 3×3 matmul for one row of an 8-bit
/// Bayer plane.
///
/// Walker hands three row-aligned slices via the **mirror-by-2**
/// boundary contract: `above` is `mid_row(row - 1)` for interior
/// rows and `mid_row(1)` at the top edge; `below` is
/// `mid_row(row + 1)` for interior rows and `mid_row(h - 2)` at
/// the bottom edge (replicate fallback when `height < 2`). `mid`
/// is the row being produced. All three share the row's pixel
/// width (`mid.len()`); column edges mirror-by-2 inside this
/// kernel for the same CFA-parity reason.
///
/// `m` is the precomputed `CCM · diag(wb)` 3×3 transform — the
/// walker fuses the two parameters once at frame entry so per-pixel
/// arithmetic stays a single matmul.
///
/// Output is packed `R, G, B` bytes — `3 * mid.len()` u8.
///
/// Bilinear demosaic: at each Bayer site, the directly-sampled
/// channel passes through; the two missing channels are filled from
/// the cardinal-or-diagonal 4-neighborhood (averaged). Soft but
/// numerically stable; the standard "first pass" reconstruction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bayer_to_rgb_row(
  above: &[u8],
  mid: &[u8],
  below: &[u8],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  _demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u8],
) {
  let w = mid.len();
  debug_assert_eq!(above.len(), w, "above row length must match mid");
  debug_assert_eq!(below.len(), w, "below row length must match mid");
  debug_assert!(rgb_out.len() >= 3 * w, "rgb_out too short");

  let (r_par, b_par) = pattern_phases(pattern);
  let rp = (row_parity & 1) as usize;

  for x in 0..w {
    let cp = x & 1;
    let (r, g, b) = bilinear_demosaic_at(w, x, rp, cp, r_par, b_par, |sel, i| match sel {
      BayerRowSel::Above => above[i] as f32,
      BayerRowSel::Mid => mid[i] as f32,
      BayerRowSel::Below => below[i] as f32,
    });
    let r_out = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    let g_out = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    let b_out = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    rgb_out[3 * x] = clamp_u8_round(r_out);
    rgb_out[3 * x + 1] = clamp_u8_round(g_out);
    rgb_out[3 * x + 2] = clamp_u8_round(b_out);
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn clamp_u8_round(v: f32) -> u8 {
  if v <= 0.0 {
    0
  } else if v >= 255.0 {
    255
  } else {
    (v + 0.5) as u8
  }
}

/// Returns `(R-site parity, B-site parity)` where each parity is
/// `(row & 1, col & 1)`. The two greens occupy the remaining
/// parities.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pattern_phases(p: crate::raw::BayerPattern) -> ((usize, usize), (usize, usize)) {
  use crate::raw::BayerPattern::*;
  match p {
    Rggb => ((0, 0), (1, 1)),
    Bggr => ((1, 1), (0, 0)),
    Grbg => ((0, 1), (1, 0)),
    Gbrg => ((1, 0), (0, 1)),
  }
}

/// Selector for the demosaic indexer — picks which of the three
/// row slices the closure should read from.
#[derive(Clone, Copy)]
enum BayerRowSel {
  Above,
  Mid,
  Below,
}

/// Demosaic a Bayer site at column `x`. Generic over a sample
/// reader so the body can be shared between the 8-bit and the
/// 16-bit Bayer kernels — the closure handles the type-specific
/// `u8` / `u16` slice indexing and casts to f32. Returns the
/// reconstructed `(R, G, B)` in the input's native f32 range —
/// the caller bakes any output-bit-depth scale at write time.
#[cfg_attr(not(tarpaulin), inline(always))]
fn bilinear_demosaic_at<F>(
  width: usize,
  x: usize,
  rp: usize,
  cp: usize,
  r_par: (usize, usize),
  b_par: (usize, usize),
  read: F,
) -> (f32, f32, f32)
where
  F: Fn(BayerRowSel, usize) -> f32,
{
  let center = read(BayerRowSel::Mid, x);
  let n = read(BayerRowSel::Above, x);
  let s = read(BayerRowSel::Below, x);
  // **Mirror-by-2** column clamp. Replicate clamp (`x = 0 → x`,
  // `x = w-1 → x`) breaks Bayer parity: at column 0 of an RGGB
  // R-site, the "west" tap would read the same R sample as the
  // center, contaminating the G average with red. Mirror-by-2
  // (`-1 → 1`, `w → w-2`) preserves parity because Bayer tiles in
  // 2×2, so skipping two columns lands on the same CFA color the
  // missing-tap site would have provided. Falls back to replicate
  // when `width < 2` (no useful Bayer interpretation at that size).
  let w_idx = if x == 0 {
    if width >= 2 { 1 } else { 0 }
  } else {
    x - 1
  };
  let e_idx = if x + 1 == width {
    if width >= 2 { width - 2 } else { width - 1 }
  } else {
    x + 1
  };
  let west = read(BayerRowSel::Mid, w_idx);
  let east = read(BayerRowSel::Mid, e_idx);
  let nw = read(BayerRowSel::Above, w_idx);
  let ne = read(BayerRowSel::Above, e_idx);
  let sw = read(BayerRowSel::Below, w_idx);
  let se = read(BayerRowSel::Below, e_idx);

  if (rp, cp) == r_par {
    (
      center,
      (n + s + west + east) * 0.25,
      (nw + ne + sw + se) * 0.25,
    )
  } else if (rp, cp) == b_par {
    (
      (nw + ne + sw + se) * 0.25,
      (n + s + west + east) * 0.25,
      center,
    )
  } else {
    let on_red_row = rp == r_par.0;
    if on_red_row {
      ((west + east) * 0.5, center, (n + s) * 0.5)
    } else {
      ((n + s) * 0.5, center, (west + east) * 0.5)
    }
  }
}

/// 10/12/14/16-bit Bayer → packed `u8` RGB.
///
/// `above` / `mid` / `below` are **low-packed** `u16` row slices —
/// every sample must satisfy `value < (1 << BITS)`, with the high
/// `16 - BITS` bits zero. The
/// [`crate::frame::BayerFrame16::try_new`] constructor validates
/// this contract on every active sample, so callers using
/// [`crate::raw::bayer16_to`] are guaranteed in-range input. Direct
/// row-API callers passing raw `&[u16]` slices are responsible for
/// the same contract; out-of-range samples violate it but the
/// kernel is sound (no panic, no UB) — it produces saturated
/// output and contaminates demosaic neighbor averages.
///
/// `m` is the unscaled `CCM · diag(wb)`; this kernel bakes the
/// input→u8 rescale (`255 / ((1 << BITS) - 1)`) into output values
/// at write time.
///
/// Output: `3 * mid.len()` `u8` packed `R, G, B`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bayer16_to_rgb_row<const BITS: u32>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  _demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u8],
) {
  const { assert!(BITS == 10 || BITS == 12 || BITS == 14 || BITS == 16) };
  let w = mid.len();
  debug_assert_eq!(above.len(), w);
  debug_assert_eq!(below.len(), w);
  debug_assert!(rgb_out.len() >= 3 * w);
  // Sample-range contract: caller guarantees every sample is
  // `< (1 << BITS)` (low-packed convention). For walker callers
  // this is upheld by `BayerFrame16::try_new` (which validates
  // every active sample at construction); direct row-API callers
  // accept the contract — out-of-range samples produce
  // defined-but-saturated output, no panic, no UB.

  let (r_par, b_par) = pattern_phases(pattern);
  let rp = (row_parity & 1) as usize;
  let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
  let max_in = max_valid as f32;
  let out_scale = 255.0 / max_in;

  for x in 0..w {
    let cp = x & 1;
    let (r, g, b) = bilinear_demosaic_at(w, x, rp, cp, r_par, b_par, |sel, i| match sel {
      BayerRowSel::Above => above[i] as f32,
      BayerRowSel::Mid => mid[i] as f32,
      BayerRowSel::Below => below[i] as f32,
    });
    let r_out = (m[0][0] * r + m[0][1] * g + m[0][2] * b) * out_scale;
    let g_out = (m[1][0] * r + m[1][1] * g + m[1][2] * b) * out_scale;
    let b_out = (m[2][0] * r + m[2][1] * g + m[2][2] * b) * out_scale;
    rgb_out[3 * x] = clamp_u8_round(r_out);
    rgb_out[3 * x + 1] = clamp_u8_round(g_out);
    rgb_out[3 * x + 2] = clamp_u8_round(b_out);
  }
}

/// 10/12/14/16-bit Bayer → packed `u16` RGB (low-packed at `BITS`).
///
/// `above` / `mid` / `below` are **low-packed** `u16` row slices —
/// every sample must satisfy `value < (1 << BITS)`. Output range
/// is `[0, (1 << BITS) - 1]` per channel; since input and output
/// share the same scale, the matmul result feeds `clamp_u16_round`
/// directly with no extra rescale. Out-of-range samples violate
/// the contract — see [`bayer16_to_rgb_row`] for the details.
///
/// Output: `3 * mid.len()` `u16` packed `R, G, B`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bayer16_to_rgb_u16_row<const BITS: u32>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  _demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u16],
) {
  const { assert!(BITS == 10 || BITS == 12 || BITS == 14 || BITS == 16) };
  let w = mid.len();
  debug_assert_eq!(above.len(), w);
  debug_assert_eq!(below.len(), w);
  debug_assert!(rgb_out.len() >= 3 * w);
  // Same sample-range contract as `bayer16_to_rgb_row<BITS>`; for
  // walker callers the contract is upheld by
  // `BayerFrame16::try_new` (which validates every active sample
  // at construction); direct row-API callers accept the contract
  // and out-of-range samples produce defined-but-saturated output
  // (no panic, no UB).

  let (r_par, b_par) = pattern_phases(pattern);
  let rp = (row_parity & 1) as usize;
  let max_valid: u16 = ((1u32 << BITS) - 1) as u16;
  let max_out = max_valid as f32;

  for x in 0..w {
    let cp = x & 1;
    let (r, g, b) = bilinear_demosaic_at(w, x, rp, cp, r_par, b_par, |sel, i| match sel {
      BayerRowSel::Above => above[i] as f32,
      BayerRowSel::Mid => mid[i] as f32,
      BayerRowSel::Below => below[i] as f32,
    });
    let r_out = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    let g_out = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    let b_out = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    rgb_out[3 * x] = clamp_u16_round(r_out, max_out);
    rgb_out[3 * x + 1] = clamp_u16_round(g_out, max_out);
    rgb_out[3 * x + 2] = clamp_u16_round(b_out, max_out);
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn clamp_u16_round(v: f32, max: f32) -> u16 {
  if v <= 0.0 {
    0
  } else if v >= max {
    max as u16
  } else {
    (v + 0.5) as u16
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

  // ---- p010_to_rgb_row (P010 → u8) ---------------------------------------
  //
  // P010 samples: 10 active bits in the HIGH 10 of each u16.
  // White Y = 1023 << 6 = 0xFFC0, neutral UV = 512 << 6 = 0x8000.

  #[test]
  fn p010_rgb_black_full_range() {
    // Y = 0, neutral UV → black.
    let y = [0u16; 4];
    let uv = [0x8000u16, 0x8000, 0x8000, 0x8000]; // U0 V0 U1 V1
    let mut rgb = [0u8; 12];
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 0), "got {rgb:?}");
  }

  #[test]
  fn p010_rgb_white_full_range() {
    // Y = 0xFFC0 = 1023 << 6, neutral UV → white.
    let y = [0xFFC0u16; 4];
    let uv = [0x8000u16, 0x8000, 0x8000, 0x8000];
    let mut rgb = [0u8; 12];
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 255), "got {rgb:?}");
  }

  #[test]
  fn p010_rgb_gray_is_gray() {
    // 10-bit mid-gray Y=512 → P010 Y = 512 << 6 = 0x8000.
    let y = [0x8000u16; 4];
    let uv = [0x8000u16; 4];
    let mut rgb = [0u8; 12];
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
    for x in 0..4 {
      let (r, g, b) = (rgb[x * 3], rgb[x * 3 + 1], rgb[x * 3 + 2]);
      assert_eq!(r, g);
      assert_eq!(g, b);
      assert!(r.abs_diff(128) <= 1, "got {r}");
    }
  }

  #[test]
  fn p010_rgb_limited_range_endpoints() {
    // 10-bit limited black Y=64 → P010 = 64 << 6 = 0x1000.
    // 10-bit limited white Y=940 → P010 = 940 << 6 = 0xEB00.
    let y = [0x1000u16, 0x1000, 0xEB00, 0xEB00];
    let uv = [0x8000u16, 0x8000, 0x8000, 0x8000];
    let mut rgb = [0u8; 12];
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, false);
    assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
    assert_eq!((rgb[3], rgb[4], rgb[5]), (0, 0, 0));
    assert_eq!((rgb[6], rgb[7], rgb[8]), (255, 255, 255));
    assert_eq!((rgb[9], rgb[10], rgb[11]), (255, 255, 255));
  }

  #[test]
  fn p010_matches_yuv420p10_when_shifted() {
    // Handing the same logical samples to P010 (high-packed) and
    // yuv420p10 (low-packed) must produce the same RGB output.
    let y_p10 = [200u16, 800, 500, 700]; // 10-bit values
    let u_p10 = [600u16, 400]; // 10-bit values
    let v_p10 = [300u16, 900]; // 10-bit values

    let y_p010: [u16; 4] = core::array::from_fn(|i| y_p10[i] << 6);
    let uv_p010: [u16; 4] = [u_p10[0] << 6, v_p10[0] << 6, u_p10[1] << 6, v_p10[1] << 6];

    let mut rgb_p10 = [0u8; 12];
    let mut rgb_p010 = [0u8; 12];
    yuv_420p_n_to_rgb_row::<10>(
      &y_p10,
      &u_p10,
      &v_p10,
      &mut rgb_p10,
      4,
      ColorMatrix::Bt709,
      true,
    );
    p_n_to_rgb_row::<10>(
      &y_p010,
      &uv_p010,
      &mut rgb_p010,
      4,
      ColorMatrix::Bt709,
      true,
    );
    assert_eq!(rgb_p10, rgb_p010);
  }

  // ---- p010_to_rgb_u16_row (P010 → native-depth u16) --------------------

  #[test]
  fn p010_rgb_u16_white_full_range() {
    let y = [0xFFC0u16; 4];
    let uv = [0x8000u16; 4];
    let mut rgb = [0u16; 12];
    p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb, 4, ColorMatrix::Bt601, true);
    assert!(rgb.iter().all(|&c| c == 1023), "got {rgb:?}");
  }

  #[test]
  fn p010_rgb_u16_limited_range_endpoints() {
    let y = [0x1000u16, 0xEB00];
    let uv = [0x8000u16, 0x8000];
    let mut rgb = [0u16; 6];
    p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb, 2, ColorMatrix::Bt709, false);
    assert_eq!((rgb[0], rgb[1], rgb[2]), (0, 0, 0));
    assert_eq!((rgb[3], rgb[4], rgb[5]), (1023, 1023, 1023));
  }
}
