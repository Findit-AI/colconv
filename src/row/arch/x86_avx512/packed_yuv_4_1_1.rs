//! AVX‑512 kernels for the Tier 5.25 packed YUV 4:1:1 source
//! (UYYVYY411).
//!
//! Per‑block layout (6 bytes / 4 pixels): `[U, Y0, Y1, V, Y2, Y3]`.
//! Each (U, V) chroma pair is shared by 4 adjacent luma samples
//! (1 → 4 horizontal chroma fan‑out).
//!
//! ## Per‑iter pipeline (64 px / 96 input bytes)
//!
//! AVX‑512 lacks a cross‑lane single‑instruction byte permute
//! (`vpermb` is AVX‑512_VBMI which we don't gate on). The deinterleave
//! is therefore decomposed into four parallel 16‑px windows using the
//! same SSE4.1 shuffle pattern; the extracted Y / U / V bytes stay in
//! `__m128i` registers and are concatenated directly into `__m512i`s
//! via `_mm512_inserti32x4` + `_mm512_permutexvar_epi32` (no per‑iter
//! stack scratch; see [`deinterleave_64px`]).
//!
//! 1. Eight 16‑byte loads — `(p0_a, p0_b)` per window covering bytes
//!    `[w*24 .. w*24+16]` and `[w*24+8 .. w*24+24]` for `w ∈ 0..4`.
//!    Loop bound `x + 64 <= width` plus the `packed.len() >= width *
//!    3 / 2` contract guarantee 96 readable bytes.
//! 2. Per window: extract 16 Y bytes (two `_mm_shuffle_epi8` +
//!    `_mm_unpacklo_epi64`) and 4 U + 4 V bytes (two
//!    `_mm_shuffle_epi8` + OR). Concatenate the four windows' Y
//!    vectors via `_mm512_inserti32x4` to produce `y_vec`. Concatenate
//!    the four `uv` vectors the same way and split via
//!    `_mm512_permutexvar_epi32` into a low 128‑bit lane carrying
//!    `[u0,u1,u2,u3]` (one u32 per window) and a second lane carrying
//!    the four V quadlets.
//! 3. Widen U / V to i32x16 each, run the standard AVX‑512 Q15 chroma
//!    math (`(cu*u_d + cv*v_d + RND) >> 15`) producing 16 i16 chroma
//!    values per channel.
//! 4. Fan each of 16 chroma i16 to 4 adjacent lanes (1 → 4 upsample)
//!    via `_mm512_permutexvar_epi32` + `_mm512_shuffle_epi8`,
//!    yielding two i16x32 chroma vectors covering the 64 Y pixels.
//! 5. Standard `scale_y` + saturating add + `narrow_u8x64` →
//!    `write_rgb_64` / `write_rgba_64`.
//! 6. Scalar tail for `width % 64 != 0`.
//!
//! ## Removed stack spill
//!
//! Earlier revisions of this kernel held the 4 windows' Y / U / V
//! bytes in `[0u8; 64]` and `[0u8; 16]` stack arrays, then reloaded
//! them via `_mm512_loadu_si512` / `_mm_loadu_si128`. Each iteration
//! paid for stack zero‑init + 5 stores + 3 loads of bytes already
//! resident in vector registers. The current path keeps everything in
//! registers (4 × `_mm512_inserti32x4` + 1 × `_mm512_permutexvar_epi32`
//! for U/V; 4 × `_mm512_inserti32x4` for Y), shaving ~96 + 16 + 16
//! bytes of per‑iter stack traffic.

use super::*;

/// Loads 4 overlapping 16‑byte windows from `packed` starting at
/// `block` and returns the 64 deinterleaved Y bytes as one `__m512i`
/// plus the 16 U bytes and 16 V bytes packed into the low 128 bits of
/// two `__m128i` registers — all in registers, no stack scratch.
///
/// # Safety
///
/// Caller has verified AVX‑512F + AVX‑512BW. `packed.len() >= block +
/// 96` (i.e. at least 96 bytes readable from `packed.as_ptr().add(block)`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn deinterleave_64px(
  packed: &[u8],
  block: usize,
  y_mask_p0: __m128i,
  y_mask_p1: __m128i,
  uv_mask_p0: __m128i,
  uv_mask_p1: __m128i,
  uv_split_perm: __m512i,
) -> (__m512i, __m128i, __m128i) {
  // SAFETY: AVX‑512BW availability is the caller's obligation;
  // `packed.len() >= block + 96` is the documented contract.
  unsafe {
    // Per 16‑px window deinterleave (4 windows, w ∈ 0..4): two
    // overlapping 16‑byte loads at offsets `block + w*24 + {0,8}`.
    // Each window produces 16 Y bytes and 4 U + 4 V bytes that we
    // keep in `__m128i` registers.
    let p0_a = _mm_loadu_si128(packed.as_ptr().add(block).cast());
    let p0_b = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
    let p1_a = _mm_loadu_si128(packed.as_ptr().add(block + 24).cast());
    let p1_b = _mm_loadu_si128(packed.as_ptr().add(block + 32).cast());
    let p2_a = _mm_loadu_si128(packed.as_ptr().add(block + 48).cast());
    let p2_b = _mm_loadu_si128(packed.as_ptr().add(block + 56).cast());
    let p3_a = _mm_loadu_si128(packed.as_ptr().add(block + 72).cast());
    let p3_b = _mm_loadu_si128(packed.as_ptr().add(block + 80).cast());

    let y0 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p0_a, y_mask_p0),
      _mm_shuffle_epi8(p0_b, y_mask_p1),
    );
    let y1 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p1_a, y_mask_p0),
      _mm_shuffle_epi8(p1_b, y_mask_p1),
    );
    let y2 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p2_a, y_mask_p0),
      _mm_shuffle_epi8(p2_b, y_mask_p1),
    );
    let y3 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p3_a, y_mask_p0),
      _mm_shuffle_epi8(p3_b, y_mask_p1),
    );

    // Concatenate 4 × `__m128i` Y into one `__m512i`.
    let mut y_vec = _mm512_castsi128_si512(y0);
    y_vec = _mm512_inserti32x4::<1>(y_vec, y1);
    y_vec = _mm512_inserti32x4::<2>(y_vec, y2);
    y_vec = _mm512_inserti32x4::<3>(y_vec, y3);

    // Per‑window UV: low 4 bytes = U[w0..w3], bytes 4..7 = V[w0..w3];
    // bytes 8..15 are zero (the OR masks only target bytes 0..7).
    let uv0 = _mm_or_si128(
      _mm_shuffle_epi8(p0_a, uv_mask_p0),
      _mm_shuffle_epi8(p0_b, uv_mask_p1),
    );
    let uv1 = _mm_or_si128(
      _mm_shuffle_epi8(p1_a, uv_mask_p0),
      _mm_shuffle_epi8(p1_b, uv_mask_p1),
    );
    let uv2 = _mm_or_si128(
      _mm_shuffle_epi8(p2_a, uv_mask_p0),
      _mm_shuffle_epi8(p2_b, uv_mask_p1),
    );
    let uv3 = _mm_or_si128(
      _mm_shuffle_epi8(p3_a, uv_mask_p0),
      _mm_shuffle_epi8(p3_b, uv_mask_p1),
    );

    // Stack the 4 windows into one `__m512i`, one window per 128‑bit lane.
    // After insert: lane w (i32 lanes 4w..4w+3) = [w_U, w_V, 0, 0].
    let mut uv_vec = _mm512_castsi128_si512(uv0);
    uv_vec = _mm512_inserti32x4::<1>(uv_vec, uv1);
    uv_vec = _mm512_inserti32x4::<2>(uv_vec, uv2);
    uv_vec = _mm512_inserti32x4::<3>(uv_vec, uv3);

    // `uv_split_perm` = `[0,4,8,12, 1,5,9,13, dc, dc, …]`. After
    // permute: i32 lanes 0..3 = `[w0_U, w1_U, w2_U, w3_U]` (= 16 U
    // bytes contiguous in the low 128‑bit lane); i32 lanes 4..7 =
    // `[w0_V, w1_V, w2_V, w3_V]` (= 16 V bytes in the second lane).
    let uv_packed = _mm512_permutexvar_epi32(uv_split_perm, uv_vec);
    let u_packed = _mm512_castsi512_si128(uv_packed);
    let v_packed = _mm512_extracti32x4_epi32::<1>(uv_packed);

    (y_vec, u_packed, v_packed)
  }
}

/// Y‑only deinterleave for the luma kernels — same window pattern as
/// [`deinterleave_64px`] but skips the U/V work.
///
/// # Safety
///
/// Caller has verified AVX‑512F + AVX‑512BW. `packed.len() >= block + 96`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn deinterleave_64px_y_only(
  packed: &[u8],
  block: usize,
  y_mask_p0: __m128i,
  y_mask_p1: __m128i,
) -> __m512i {
  // SAFETY: forwarded from the caller's `target_feature` context.
  unsafe {
    let p0_a = _mm_loadu_si128(packed.as_ptr().add(block).cast());
    let p0_b = _mm_loadu_si128(packed.as_ptr().add(block + 8).cast());
    let p1_a = _mm_loadu_si128(packed.as_ptr().add(block + 24).cast());
    let p1_b = _mm_loadu_si128(packed.as_ptr().add(block + 32).cast());
    let p2_a = _mm_loadu_si128(packed.as_ptr().add(block + 48).cast());
    let p2_b = _mm_loadu_si128(packed.as_ptr().add(block + 56).cast());
    let p3_a = _mm_loadu_si128(packed.as_ptr().add(block + 72).cast());
    let p3_b = _mm_loadu_si128(packed.as_ptr().add(block + 80).cast());
    let y0 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p0_a, y_mask_p0),
      _mm_shuffle_epi8(p0_b, y_mask_p1),
    );
    let y1 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p1_a, y_mask_p0),
      _mm_shuffle_epi8(p1_b, y_mask_p1),
    );
    let y2 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p2_a, y_mask_p0),
      _mm_shuffle_epi8(p2_b, y_mask_p1),
    );
    let y3 = _mm_unpacklo_epi64(
      _mm_shuffle_epi8(p3_a, y_mask_p0),
      _mm_shuffle_epi8(p3_b, y_mask_p1),
    );
    let mut y_vec = _mm512_castsi128_si512(y0);
    y_vec = _mm512_inserti32x4::<1>(y_vec, y1);
    y_vec = _mm512_inserti32x4::<2>(y_vec, y2);
    y_vec = _mm512_inserti32x4::<3>(y_vec, y3);
    y_vec
  }
}

/// AVX‑512 UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX‑512 UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    uyyvyy411_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// Generic UYYVYY411 → RGB / RGBA AVX‑512 kernel. 64 px / iter.
///
/// # Safety
///
/// Caller has verified AVX‑512F + AVX‑512BW. `packed.len() >= width *
/// 3 / 2`. `width` is a multiple of 4. `out.len() >= bpp * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn uyyvyy411_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm512_set1_epi8(-1);

    // Per‑16‑px deinterleave masks (SSE4.1‑style, applied to __m128i).
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let uv_mask_p0 = _mm_setr_epi8(0, 6, 12, -1, 3, 9, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let uv_mask_p1 = _mm_setr_epi8(
      -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    // 1 → 4 chroma fan‑out mask (per‑128‑bit‑lane). Each AVX‑512 lane
    // has 8 i16 lanes (16 bytes); we read bytes 0..3 (2 i16 = 2 chroma
    // values) and fan each to 4 i16 lanes.
    let dup_mask_lane = [
      0i8, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, // lane 0..3 share
    ];
    let mut dup_mask_arr = [0i8; 64];
    for i in 0..4 {
      dup_mask_arr[i * 16..(i + 1) * 16].copy_from_slice(&dup_mask_lane);
    }
    let dup_mask = _mm512_loadu_si512(dup_mask_arr.as_ptr().cast());

    // Cross‑lane permute indices that arrange 2 chroma values into the
    // low 4 bytes of each 128‑bit AVX‑512 lane.
    //
    // After `permutexvar_epi64(pack_fixup, packs_epi32(x, x))` (used
    // below), the 16 chroma i16 values sit in i16 lanes 0..15 (low 256
    // bits) of `r_chroma`, with the high 256 bits an exact duplicate.
    // Equivalently, the 8 valid 32‑bit i32 chunks 0..7 contain pairs
    // `[(c0,c1), (c2,c3), (c4,c5), (c6,c7), (c8,c9), (c10,c11),
    // (c12,c13), (c14,c15)]`.
    //
    // `chroma_low` covers Y[0..32]:
    //   lane 0 chunk 0 = source chunk 0 = (c0, c1)
    //   lane 1 chunk 0 = source chunk 1 = (c2, c3)
    //   lane 2 chunk 0 = source chunk 2 = (c4, c5)
    //   lane 3 chunk 0 = source chunk 3 = (c6, c7)
    //
    // `chroma_high` covers Y[32..64]:
    //   lane 0 chunk 0 = source chunk 4 = (c8, c9)
    //   lane 1 chunk 0 = source chunk 5 = (c10, c11)
    //   lane 2 chunk 0 = source chunk 6 = (c12, c13)
    //   lane 3 chunk 0 = source chunk 7 = (c14, c15)
    //
    // 16-element i32 permute index: chunk 0 of each output 128-bit lane
    // gets the desired source chunk; chunks 1..3 of each lane are
    // don't-care (the per-lane shuffle only reads bytes 0..3).
    let perm_low = _mm512_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
    let perm_high = _mm512_setr_epi32(4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0);

    // U/V split permute (see `deinterleave_64px`): gathers i32 lane 0
    // (= 4 U bytes) of windows 0..3 into low 128 bits, and i32 lane 1
    // (= 4 V bytes) into the second 128 bits. High half is don't‑care.
    let uv_split_perm = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0);

    let mut x = 0usize;
    while x + 64 <= width {
      let block = (x / 4) * 6;
      let (y_vec, u_packed, v_packed) = deinterleave_64px(
        packed,
        block,
        y_mask_p0,
        y_mask_p1,
        uv_mask_p0,
        uv_mask_p1,
        uv_split_perm,
      );

      // Widen 16 U / 16 V bytes → i32x16 each.
      let u_i32 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(u_packed), _mm512_set1_epi32(128));
      let v_i32 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(v_packed), _mm512_set1_epi32(128));
      let u_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_v,
      ));

      // (cu * u_d + cv * v_d + RND) >> 15 in i32x16 → pack to i16.
      // `packs_epi32(x, x)` produces per-lane interleaved output;
      // `permutexvar_epi64(pack_fixup, ...)` with `[0,2,4,6,1,3,5,7]`
      // restores natural element order so c[0..16] sit contiguously in
      // the low 256 bits (high 256 bits = duplicate).
      let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
      let r_i32 = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cru, u_d), _mm512_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cgu, u_d), _mm512_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(cbu, u_d), _mm512_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));
      let r_chroma = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(r_i32, r_i32));
      let g_chroma = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(g_i32, g_i32));
      let b_chroma = _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(b_i32, b_i32));

      // Fan‑out each chroma channel to 64 Y pixels.
      let r_for_lo = _mm512_permutexvar_epi32(perm_low, r_chroma);
      let g_for_lo = _mm512_permutexvar_epi32(perm_low, g_chroma);
      let b_for_lo = _mm512_permutexvar_epi32(perm_low, b_chroma);
      let r_for_hi = _mm512_permutexvar_epi32(perm_high, r_chroma);
      let g_for_hi = _mm512_permutexvar_epi32(perm_high, g_chroma);
      let b_for_hi = _mm512_permutexvar_epi32(perm_high, b_chroma);
      let r_dup_lo = _mm512_shuffle_epi8(r_for_lo, dup_mask);
      let g_dup_lo = _mm512_shuffle_epi8(g_for_lo, dup_mask);
      let b_dup_lo = _mm512_shuffle_epi8(b_for_lo, dup_mask);
      let r_dup_hi = _mm512_shuffle_epi8(r_for_hi, dup_mask);
      let g_dup_hi = _mm512_shuffle_epi8(g_for_hi, dup_mask);
      let b_dup_hi = _mm512_shuffle_epi8(b_for_hi, dup_mask);

      // Y path identical to packed_yuv_8bit.
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    // Scalar tail.
    if x < width {
      let tail_block = (x / 4) * 6;
      let tail_packed = &packed[tail_block..(width / 4) * 6];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::uyyvyy411_to_rgba_row(tail_packed, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::uyyvyy411_to_rgb_row(tail_packed, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- Packed YUV 4:1:1 (8-bit) → HSV (staged via a reused RGB chunk) --
//
// The SIMD twin of the scalar `uyyvyy411_to_hsv_row` kernel. Reuses the
// LOCAL packed-family driver `packed_hsv_via_rgb_chunks` (defined in the
// sibling `packed_yuv_8bit` module) to fill a small reused RGB scratch
// via the EXISTING AVX-512 `uyyvyy411_to_rgb_row` kernel, then runs this
// backend's `rgb_to_hsv_row` on the chunk. Byte-identical to
// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier with no
// source-width RGB allocation. `HSV_CHUNK` is a multiple of 4, so every
// chunk offset lands on a 6-byte / 4-pixel block boundary.

/// AVX-512: UYYVYY411 (4:1:1) → planar HSV bytes (OpenCV encoding),
/// staged via the reused-RGB-chunk pattern over this backend's
/// [`uyyvyy411_to_rgb_row`] + `rgb_to_hsv_row`. Byte-identical to
/// `rgb_to_hsv_row(uyyvyy411_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn uyyvyy411_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: SIMD verified; the shared chunk driver forwards the per-chunk
  // sub-slices to this backend's UYYVYY411 RGB kernel under the same
  // contract. The packed byte offset for the chunk at pixel `offset` (a
  // multiple of 4) is `offset * 3 / 2` (6 bytes per 4-pixel block).
  unsafe {
    super::packed_yuv_8bit::packed_hsv_via_rgb_chunks(
      h_out,
      s_out,
      v_out,
      width,
      |offset, n, rgb| {
        uyyvyy411_to_rgb_row(&packed[offset * 3 / 2..], rgb, n, matrix, full_range);
      },
    );
  }
}

/// AVX‑512 UYYVYY411 → 8-bit luma extraction. 64 px / iter.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 64 <= width {
      let block = (x / 4) * 6;
      let y_vec = deinterleave_64px_y_only(packed, block, y_mask_p0, y_mask_p1);
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 64;
    }
    if x < width {
      let tail_block = (x / 4) * 6;
      scalar::uyyvyy411_to_luma_row(
        &packed[tail_block..(width / 4) * 6],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

/// AVX‑512 UYYVYY411 → u16 luma extraction (zero-extended Y bytes). 64
/// px / iter.
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with `out.len() >= width`
/// `u16` elements.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  debug_assert!(packed.len() >= width * 3 / 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let y_mask_p0 = _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask_p1 = _mm_setr_epi8(5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 64 <= width {
      let block = (x / 4) * 6;
      let y_vec = deinterleave_64px_y_only(packed, block, y_mask_p0, y_mask_p1);
      // Widen 64 u8 → 64 u16 via two `_mm512_cvtepu8_epi16` calls on
      // the two 256‑bit halves.
      let y_lo_256 = _mm512_castsi512_si256(y_vec);
      let y_hi_256 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let w_lo = _mm512_cvtepu8_epi16(y_lo_256);
      let w_hi = _mm512_cvtepu8_epi16(y_hi_256);
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), w_lo);
      _mm512_storeu_si512(out.as_mut_ptr().add(x + 32).cast(), w_hi);
      x += 64;
    }
    if x < width {
      let tail_block = (x / 4) * 6;
      scalar::uyyvyy411_to_luma_u16_row(
        &packed[tail_block..(width / 4) * 6],
        &mut out[x..width],
        width - x,
      );
    }
  }
}
