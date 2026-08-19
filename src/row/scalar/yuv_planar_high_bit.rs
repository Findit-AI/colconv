use super::{load_u16, *};

/// Recovers the `BITS`-bit logical sample from one **endian-normalized** wire
/// `u16` for the high-bit planar YUV family.
///
/// `MSB = false` (low-bit-packed `Yuv444pN`, samples in the low `BITS`) masks
/// the active low `BITS` (`& ((1 << BITS) - 1)`); `MSB = true` (MSB-aligned
/// `Yuv444pNMsb`, samples in the high `BITS`, FFmpeg `shift = 16 - BITS`)
/// right-shifts the active high `BITS` down (`>> (16 - BITS)`). `value` must
/// already be host-native (apply [`load_u16`] first). Both `BITS` and `MSB`
/// are const, so the branch folds and the unused path is eliminated per
/// monomorphisation. Mirrors the semi-planar [`depack_pn`](super::subsampled_high_bit_pn)
/// helper; mispacked input has the wrong bits discarded (the deliberate,
/// SIMD-matching failure mode).
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_hb<const BITS: u32, const MSB: bool>(value: u16) -> u16 {
  if MSB {
    value >> (16 - BITS)
  } else {
    value & bits_mask::<BITS>()
  }
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
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_row::<BITS, false, false, BE>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of high‑bit‑depth 4:2:0 YUV (`u16` samples in the
/// low `BITS` bits) directly to **8-bit** packed **RGBA**. Same numerical
/// contract as [`yuv_420p_n_to_rgb_row`]; the only differences are the
/// per-pixel stride (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `rgba_out.len() >= 4 * width` (other slices: same as RGB variant).
//
// Scalar prep for Ship 8 Tranche 5a: the public dispatcher
// `row::yuv420p10_to_rgba_row` (and its u16 sibling) lands in the
// follow-up SIMD/dispatcher PR. Until then this thin wrapper has no
// caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, false, BE>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 high‑bit‑depth → **u8** packed **RGBA**. Same numerical
/// contract as [`yuv_420p_n_to_rgba_row`] for R/G/B; the per-pixel
/// alpha byte is sourced from `a_src` (depth-converted by
/// `BITS - 8` shift) instead of being constant `0xFF`. Used by the
/// YUVA 4:2:0 source family ([`crate::source::Yuva420p9`] /
/// [`crate::source::Yuva420p10`] in tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `a_src.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
// Reachable from the arch tails (`yuv_420p_n_to_rgb_or_rgba_row<…,
// ALPHA_SRC = true>` and friends) which always reference this symbol
// at syntax expansion regardless of the monomorphization. Const-eval
// prunes the call when `ALPHA_SRC = false`, so under `yuv-planar`
// alone Rust sees it as dead. See `yuv_planar_8bit.rs` for the
// per-item `#[allow(dead_code)]` rationale.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_rgba_with_alpha_src_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, true, BE>(
    y,
    u_half,
    v_half,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared kernel for [`yuv_420p_n_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp store), [`yuv_420p_n_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bpp store with constant
/// `0xFF` alpha) and
/// [`yuv_420p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp store with depth-converted source alpha).
///
/// The compiler monomorphizes into separate functions per
/// `(ALPHA, ALPHA_SRC)`; the const branches are DCE'd at each call
/// site. `a_src` is `None` for both `ALPHA_SRC = false` flavors —
/// reading it is a const-disabled branch in those monomorphizations,
/// so callers pay zero overhead for the strategy parameter.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Compile-time guard — fails monomorphization for any BITS outside
  // {9, 10, 12, 14}. 16 would overflow the Q15 chroma sum (16-bit lives
  // in `yuv_420p16_to_rgb_row`'s i64 chroma family); 8 belongs to the
  // non-const-generic `yuv_420_to_rgb_or_rgba_row`. Without this guard a
  // release build instantiating ::<16, _, _> would silently produce wrong
  // output.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let mask = bits_mask::<BITS>();

  // Every sample is AND‑masked to the low `BITS` bits on load (after
  // optional BE byte-swap). This eliminates architecture‑dependent
  // divergence on mispacked input. For valid input every mask is a
  // no‑op. For malformed input the "wrong" output is identical
  // across scalar + all 5 SIMD backends.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(
      (load_u16::<BE>(u_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      (load_u16::<BE>(v_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((load_u16::<BE>(y[x]) & mask) as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies the wrapper
      // passed Some(_), validated above by debug_assert.
      // Mask the source alpha to BITS like Y/U/V — `try_new` admits
      // out-of-range u16 samples, and an unmasked overrange value
      // (e.g. 1024 at BITS=10) would shift down to 256 → cast-to-u8 0,
      // silently turning over-range alpha into transparent output.
      let a_u16 = load_u16::<BE>(a_src.as_ref().unwrap()[x]) & mask;
      out[x * bpp + 3] = (a_u16 >> (BITS - 8)) as u8;
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = q15_scale((load_u16::<BE>(y[x + 1]) & mask) as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = clamp_u8(y1 + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1 + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1 + b_chroma);
    if ALPHA_SRC {
      let a_u16 = load_u16::<BE>(a_src.as_ref().unwrap()[x + 1]) & mask;
      out[(x + 1) * bpp + 3] = (a_u16 >> (BITS - 8)) as u8;
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
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
/// [`yuv_420p_n_to_rgb_row`], which is ~2x faster.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false, false, BE>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of high‑bit‑depth 4:2:0 YUV → **native‑depth `u16`
/// packed RGBA**. Same numerical contract as
/// [`yuv_420p_n_to_rgb_u16_row`]; the only differences are the
/// per-pixel stride (4 vs 3 `u16` elements) and the alpha element,
/// `(1 << BITS) - 1` (opaque maximum at the input bit depth).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `rgba_out.len() >= 4 * width` (other slices: same as RGB variant).
//
// Scalar prep for Ship 8 Tranche 5b: the public dispatcher
// `row::yuv420p10_to_rgba_u16_row` lands in the follow-up SIMD/dispatcher
// PR. Until then this thin wrapper has no caller.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, false, BE>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 high‑bit‑depth → **native‑depth `u16`** packed
/// **RGBA**. Same numerical contract as
/// [`yuv_420p_n_to_rgba_u16_row`] for R/G/B; the per-pixel alpha
/// element is sourced from `a_src` (already at the source's native
/// bit depth) instead of being the opaque maximum
/// `(1 << BITS) - 1`. Used by the YUVA 4:2:0 source family
/// ([`crate::source::Yuva420p9`] / [`crate::source::Yuva420p10`] in
/// tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `a_src.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
// See `yuv_420p_n_to_rgba_with_alpha_src_row` for the per-item
// `#[allow(dead_code)]` rationale.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, true, BE>(
    y,
    u_half,
    v_half,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared kernel for [`yuv_420p_n_to_rgb_u16_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp store), [`yuv_420p_n_to_rgba_u16_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bpp store with opaque alpha
/// `(1 << BITS) - 1`) and
/// [`yuv_420p_n_to_rgba_u16_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp store with native-depth source alpha).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Compile-time guard — see note on `yuv_420p_n_to_rgb_or_rgba_row`.
  // The 16-bit u16-output path is `yuv_420p16_to_rgb_or_rgba_u16_row`
  // (i64 chroma family).
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let mask = bits_mask::<BITS>();
  let alpha_max: u16 = out_max as u16;

  // Every sample AND‑masked to the low `BITS` bits (after optional BE
  // byte-swap). See matching comment in `yuv_420p_n_to_rgb_or_rgba_row`.
  // Masking keeps every intermediate bounded by design.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(
      (load_u16::<BE>(u_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      (load_u16::<BE>(v_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((load_u16::<BE>(y[x]) & mask) as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // Mask the source alpha to BITS like Y/U/V — `try_new` admits
      // out-of-range u16 samples, and the documented native-depth
      // output range is `[0, (1 << BITS) - 1]`. Without masking, an
      // overrange `1024` at BITS=10 would leak straight to output.
      out[x * bpp + 3] = load_u16::<BE>(a_src.as_ref().unwrap()[x]) & mask;
    } else if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }

    let y1 = q15_scale((load_u16::<BE>(y[x + 1]) & mask) as i32 - y_off, y_scale);
    out[(x + 1) * bpp] = (y1 + r_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      out[(x + 1) * bpp + 3] = load_u16::<BE>(a_src.as_ref().unwrap()[x + 1]) & mask;
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = alpha_max;
    }

    x += 2;
  }
}

/// YUV 4:4:4 planar high‑bit‑depth → **u8** packed RGB. Const‑generic
/// over `BITS ∈ {9, 10, 12, 14}`. 1:1 chroma per Y pixel (no chroma
/// pair, no upsampling). Math is identical to
/// [`yuv_420p_n_to_rgb_row`] except each pixel gets its own U / V.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false, BE, false>(
    y, u, v, None, rgb_out, width, matrix, full_range,
  );
}

/// MSB-aligned YUV 4:4:4 planar high-bit-depth → **u8** packed RGB. The
/// recovery-shift twin of [`yuv_444p_n_to_rgb_row`] for the `Yuv444pNMsb`
/// family (samples in the high `BITS` bits, `shift = 16 - BITS`). Pinned to
/// `BITS ∈ {10, 12}` (FFmpeg ships only `yuv444p10msb` / `yuv444p12msb`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_msb_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false, BE, true>(
    y, u, v, None, rgb_out, width, matrix, full_range,
  );
}

/// MSB-aligned YUV 4:4:4 planar high-bit-depth → **u8** packed **RGBA**
/// (opaque alpha `0xFF`). The recovery-shift twin of
/// [`yuv_444p_n_to_rgba_row`] for the `Yuv444pNMsb` family.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_msb_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false, BE, true>(
    y, u, v, None, rgba_out, width, matrix, full_range,
  );
}

/// YUV 4:4:4 planar high‑bit‑depth → **u8** packed **RGBA**. Same
/// numerical contract as [`yuv_444p_n_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false, BE, false>(
    y, u, v, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:4:4 planar high‑bit‑depth → **u8** packed **RGBA**. Same
/// numerical contract as [`yuv_444p_n_to_rgba_row`] for R/G/B; the
/// per-pixel alpha byte is sourced from `a_src` (depth-converted by
/// `BITS - 8` shift) instead of being constant `0xFF`. Used by the
/// YUVA source family ([`crate::source::Yuva444p10`] in tranche 8b‑1a).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
// See `yuv_420p_n_to_rgba_with_alpha_src_row` for the per-item
// `#[allow(dead_code)]` rationale.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgba_with_alpha_src_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, true, BE, false>(
    y,
    u,
    v,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared kernel for [`yuv_444p_n_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp store), [`yuv_444p_n_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bpp store with constant
/// `0xFF` alpha) and
/// [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp store with depth-converted source alpha).
///
/// The compiler monomorphizes into separate functions per
/// `(ALPHA, ALPHA_SRC)`; the const branches are DCE'd at each call
/// site. `a_src` is `None` for both `ALPHA_SRC = false` flavors —
/// reading it is a const-disabled branch in those monomorphizations,
/// so callers pay zero overhead for the strategy parameter.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
  const MSB: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Compile-time guard — fails monomorphization for any BITS outside
  // {9, 10, 12, 14}. The 16-bit path lives in `yuv_444p16_to_rgb_row`
  // (i32 u8-output kernel family). Without this guard a caller
  // invoking ::<16, _, _> would reach the NEON clamp where
  // `(1 << BITS) - 1 as i16` silently wraps to -1.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  // The MSB-aligned variant has no alpha plane (FFmpeg ships only
  // `yuv444p{10,12}msb`), so `ALPHA_SRC` is always low-bit-masked.
  const { assert!(!MSB || !ALPHA_SRC) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  for x in 0..width {
    // 4:4:4: one UV pair per pixel, no subsampling. The sample is recovered
    // from the wire `u16` via [`extract_hb`]: low-bit mask (`MSB = false`,
    // `Yuv444pN`) or `>> (16 - BITS)` recovery shift (`MSB = true`,
    // `Yuv444pNMsb`).
    let u_d = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(u[x])) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(v[x])) as i32 - bias,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies the wrapper
      // passed Some(_), validated above by debug_assert.
      // Mask the source alpha to BITS like Y/U/V — `try_new` admits
      // out-of-range u16 samples, and an unmasked overrange value
      // (e.g. 1024 at BITS=10) would shift down to 256 → cast-to-u8 0,
      // silently turning over-range alpha into transparent output.
      let a_u16 = load_u16::<BE>(a_src.as_ref().unwrap()[x]) & bits_mask::<BITS>();
      out[x * bpp + 3] = (a_u16 >> (BITS - 8)) as u8;
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// YUV 4:4:4 planar high‑bit‑depth → **native‑depth `u16`** packed RGB.
/// Const‑generic over `BITS ∈ {9, 10, 12, 14}`. Low‑bit‑packed output.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false, BE, false>(
    y, u, v, None, rgb_out, width, matrix, full_range,
  );
}

/// MSB-aligned YUV 4:4:4 planar high-bit-depth → **native-depth `u16`**
/// packed RGB (low-bit-packed output). The recovery-shift twin of
/// [`yuv_444p_n_to_rgb_u16_row`] for the `Yuv444pNMsb` family.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_msb_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false, BE, true>(
    y, u, v, None, rgb_out, width, matrix, full_range,
  );
}

/// MSB-aligned YUV 4:4:4 planar high-bit-depth → **native-depth `u16`**
/// packed **RGBA** (opaque alpha `(1 << BITS) - 1`). The recovery-shift twin
/// of [`yuv_444p_n_to_rgba_u16_row`] for the `Yuv444pNMsb` family.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_msb_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false, BE, true>(
    y, u, v, None, rgba_out, width, matrix, full_range,
  );
}

/// YUV 4:4:4 planar high‑bit‑depth → **native‑depth `u16`** packed
/// **RGBA**. Same numerical contract as [`yuv_444p_n_to_rgb_u16_row`];
/// the only differences are the per-pixel stride (4 vs 3 `u16`
/// elements) and the alpha element, `(1 << BITS) - 1` (opaque maximum
/// at the input bit depth).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false, BE, false>(
    y, u, v, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:4:4 planar high‑bit‑depth → **native‑depth `u16`** packed
/// **RGBA**. Same numerical contract as [`yuv_444p_n_to_rgba_u16_row`]
/// for R/G/B; the per-pixel alpha element is sourced from `a_src`
/// (already at the source's native bit depth) instead of being the
/// opaque maximum `(1 << BITS) - 1`. Used by the YUVA source family
/// ([`crate::source::Yuva444p10`] in tranche 8b‑1a).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
// See `yuv_420p_n_to_rgba_with_alpha_src_row` for the per-item
// `#[allow(dead_code)]` rationale.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, true, BE, false>(
    y,
    u,
    v,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared kernel for [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp store), [`yuv_444p_n_to_rgba_u16_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bpp store with opaque alpha
/// `(1 << BITS) - 1`) and
/// [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp store with native-depth source alpha).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
  const MSB: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Compile-time guard — see note on `yuv_444p_n_to_rgb_or_rgba_row`.
  // The 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row` (i64
  // chroma family).
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  // The MSB-aligned variant carries no alpha plane.
  const { assert!(!MSB || !ALPHA_SRC) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let alpha_max: u16 = out_max as u16;

  for x in 0..width {
    // `MSB = false` masks the low `BITS`; `MSB = true` recovers via
    // `>> (16 - BITS)` — see [`extract_hb`].
    let u_d = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(u[x])) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(v[x])) as i32 - bias,
      c_scale,
    );

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      extract_hb::<BITS, MSB>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // Mask the source alpha to BITS like Y/U/V — `try_new` admits
      // out-of-range u16 samples, and the documented native-depth
      // output range is `[0, (1 << BITS) - 1]`. Without masking, an
      // overrange `1024` at BITS=10 would leak straight to output.
      out[x * bpp + 3] = load_u16::<BE>(a_src.as_ref().unwrap()[x]) & bits_mask::<BITS>();
    } else if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }
  }
}

// ---- High-bit-depth planar YUV → HSV (direct: no RGB scratch) ---------
//
// The display-referred twins of the `yuv_*p_n_to_rgb_row` kernels above
// (BITS ∈ {9, 10, 12, 14}), fused with the OpenCV HSV quantizer. Each
// shares the EXACT per-pixel **8-bit-output** Q15 decode
// (`Coefficients::for_matrix` + `range_params_n::<BITS, 8>` + the AND
// mask + the same chroma-subsampling shape) as its `_to_rgb_row`
// sibling, then feeds the decoded `(r, g, b)` straight into
// [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes — never
// materializing a packed-RGB row. The HSV output is 8-bit
// (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`) regardless of source depth,
// because the existing high-bit HSV path is `rgb_to_hsv_row` applied to
// the **8-bit** `yuv_*p_n_to_rgb_row` output. These kernels are
// therefore byte-identical to `rgb_to_hsv_row(yuv_*p_n_to_rgb_row::<BITS,
// BE>(...))` but allocate no RGB intermediate. The SIMD backends mirror
// them via a small reused 8-bit-RGB chunk filled by the existing SIMD
// `_to_rgb_row` kernel plus the SIMD `rgb_to_hsv_row`.

/// High-bit-depth YUV 4:2:0 planar → planar HSV bytes (OpenCV
/// `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`).
/// Const-generic over `BITS ∈ {9, 10, 12, 14}` and `BE` (source byte
/// order), exactly like [`yuv_420p_n_to_rgb_row`]. Chroma is half-width,
/// nearest-neighbor 1→2 upsampled per pixel pair. Also serves 4:2:2
/// (half-width chroma, full height — identical per-row shape).
///
/// Byte-identical to `rgb_to_hsv_row(yuv_420p_n_to_rgb_row::<BITS,
/// BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, and each of `h_out` / `s_out` /
///   `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420p_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Same compile-time guard as `yuv_420p_n_to_rgb_or_rgba_row`: the i32
  // 8-bit-output Q15 pipeline is valid for {9, 10, 12, 14}; 16 lives in
  // `yuv_420p16_to_hsv_row`, 8 in `yuv_420_to_hsv_row`.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let mask = bits_mask::<BITS>();

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = q15_scale(
      (load_u16::<BE>(u_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      (load_u16::<BE>(v_half[c_idx]) & mask) as i32 - bias,
      c_scale,
    );
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((load_u16::<BE>(y[x]) & mask) as i32 - y_off, y_scale);
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h0;
    s_out[x] = s0;
    v_out[x] = v0;

    let y1 = q15_scale((load_u16::<BE>(y[x + 1]) & mask) as i32 - y_off, y_scale);
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1 + r_chroma) as i32,
      clamp_u8(y1 + g_chroma) as i32,
      clamp_u8(y1 + b_chroma) as i32,
    );
    h_out[x + 1] = h1;
    s_out[x + 1] = s1;
    v_out[x + 1] = v1;

    x += 2;
  }
}

/// High-bit-depth YUV 4:4:4 planar → planar HSV bytes. Const-generic
/// over `BITS ∈ {9, 10, 12, 14}` and `BE`, exactly like
/// [`yuv_444p_n_to_rgb_row`]. One UV pair per pixel (no subsampling).
/// Also serves 4:4:0 (the chroma row duplicated across two Y rows by
/// the walker — the per-row shape is identical to 4:4:4).
///
/// Byte-identical to `rgb_to_hsv_row(yuv_444p_n_to_rgb_row::<BITS,
/// BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`, and
///   each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();
  let mask = bits_mask::<BITS>();

  for x in 0..width {
    let u_d = q15_scale((load_u16::<BE>(u[x]) & mask) as i32 - bias, c_scale);
    let v_d = q15_scale((load_u16::<BE>(v[x]) & mask) as i32 - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale((load_u16::<BE>(y[x]) & mask) as i32 - y_off, y_scale);
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  }
}

/// MSB-aligned YUV 4:4:4 planar high-bit-depth → planar HSV bytes. The
/// recovery-shift twin of [`yuv_444p_n_to_hsv_row`] for the `Yuv444pNMsb`
/// family (`BITS ∈ {10, 12}`, samples in the high `BITS`, recovered via
/// `>> (16 - BITS)`). Byte-identical to
/// `rgb_to_hsv_row(yuv_444p_n_msb_to_rgb_row::<BITS, BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`, and
///   each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_msb_to_hsv_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  for x in 0..width {
    let u_d = q15_scale(
      extract_hb::<BITS, true>(load_u16::<BE>(u[x])) as i32 - bias,
      c_scale,
    );
    let v_d = q15_scale(
      extract_hb::<BITS, true>(load_u16::<BE>(v[x])) as i32 - bias,
      c_scale,
    );
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      extract_hb::<BITS, true>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  }
}
