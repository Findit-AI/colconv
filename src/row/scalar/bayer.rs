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
