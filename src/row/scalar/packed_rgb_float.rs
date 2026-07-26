// ---- Tier 9 packed-float-RGB helpers (Rgbf32) -------------------------
//
// Compact float→integer conversion kernels behind the [`Rgbf32`]
// source-side sinker family. Each pixel is `3 x f32` (linear R, G, B).
// HDR values > 1.0 saturate to the output range; values < 0.0 clamp
// to 0.
//
// This file provides the scalar reference / fallback implementations;
// SIMD dispatch lives in `row::dispatch::rgb_float_ops` and per-arch
// backends in `row::arch::*::packed_rgb_float`.
//
// Rounding convention: `(value * out_max).round()` then saturating cast.
// Matches the symmetric error bound the rest of the float→integer
// conversions in this crate use (see `scalar::mod` rounding docs).

/// Round-to-nearest-even (IEEE 754 default rounding mode) for a
/// non-negative `f32` known to fit in `u32` after the cast. This is
/// the reference rounding rule that the SIMD backends' saturating-
/// convert intrinsics use (`vcvtnq_u32_f32` on NEON,
/// `_mm{,256,512}_cvtps_epi32` with default MXCSR on x86,
/// `f32x4_nearest` + `i32x4_trunc_sat_f32x4` on wasm).
///
/// `f32::round` would round half-away-from-zero (e.g. `0.5 → 1`),
/// which diverges from SIMD on exact `.5` inputs. `f32::round_ties_even`
/// matches SIMD but is `std`-only on f32 — `pixon` is `no_std`-capable,
/// so we implement the rule manually using integer arithmetic.
#[cfg_attr(not(tarpaulin), inline(always))]
fn round_ties_even_nonneg(x: f32) -> f32 {
  // After `clamp(0.0, scale)` with `scale ≤ 65535`, `x as u32` is
  // lossless (truncates toward zero, no saturation). NaN→0 also matches
  // the SIMD backends' NaN-handling (NaN is not a valid sample for
  // colour-converted output; clamp would drop it via `min(NaN, 1) = NaN`
  // → `(NaN * 255).round_ties_even() = NaN` → `as u8 = 0` in scalar,
  // and `vcvtnq_u32_f32(NaN) = 0` likewise).
  if !x.is_finite() {
    return 0.0;
  }
  let i = x as u32;
  let frac = x - (i as f32);
  if frac < 0.5 {
    i as f32
  } else if frac > 0.5 {
    (i + 1) as f32
  } else {
    // Exact half — round to even.
    if i & 1 == 0 { i as f32 } else { (i + 1) as f32 }
  }
}

/// Clamps a single `f32` value to `[0, 1]` and scales to a `u8` using
/// round-to-nearest-even — matches every SIMD backend's saturating-
/// convert behavior. NaN → `0`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn f32_to_u8_clamped(v: f32) -> u8 {
  let clamped = v.clamp(0.0, 1.0);
  let scaled = clamped * 255.0;
  round_ties_even_nonneg(scaled) as u8
}

/// Clamps a single `f32` value to `[0, 1]` and scales to a `u16`.
/// See [`f32_to_u8_clamped`] for the rounding-mode rationale.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn f32_to_u16_clamped(v: f32) -> u16 {
  let clamped = v.clamp(0.0, 1.0);
  let scaled = clamped * 65535.0;
  round_ties_even_nonneg(scaled) as u16
}

/// Read one f32 element from `rgb_in[i]`, decoding the IEEE 754 bit
/// pattern from `BE` byte order to host-native byte order. Scalar
/// endian-aware load for Rgbf32 streams.
///
/// `from_be` / `from_le` are target-endian aware: a no-op when the
/// stored byte order matches the host, a byte-swap when they differ.
/// Mirrors the SIMD `load_endian_*::<BE>` helpers' semantics so LE and
/// BE hosts produce identical decoded values.
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_f32<const BE: bool>(rgb_in: &[f32], i: usize) -> f32 {
  let bits = rgb_in[i].to_bits();
  f32::from_bits(if BE {
    u32::from_be(bits)
  } else {
    u32::from_le(bits)
  })
}

/// Read one `half::f16` element from `rgb_in[i]`, decoding the bit
/// pattern from `BE` byte order to host-native. Scalar endian-aware
/// load for Rgbf16 streams.
#[cfg_attr(not(tarpaulin), inline(always))]
fn load_f16<const BE: bool>(rgb_in: &[half::f16], i: usize) -> half::f16 {
  let bits = rgb_in[i].to_bits();
  half::f16::from_bits(if BE {
    u16::from_be(bits)
  } else {
    u16::from_le(bits)
  })
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u8`
/// output. Each `f32` is clamped to `[0, 1]` and scaled by 255.
///
/// `BE` selects the **encoded byte order** of the input buffer:
/// `false` = LE-encoded on disk/wire, `true` = BE-encoded. This is
/// independent of the host CPU's native byte order — a swap happens
/// only when the encoded order differs from the host CPU's native order
/// (handled internally via `u32::from_le` / `u32::from_be`, both
/// target-endian-aware).
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_row<const BE: bool>(rgb_in: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, i));
    rgb_out[i + 1] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, i + 1));
    rgb_out[i + 2] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, i + 2));
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u8`
/// output with `A = 0xFF` (the float source has no alpha).
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, s));
    rgba_out[d + 1] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, s + 1));
    rgba_out[d + 2] = f32_to_u8_clamped(load_f32::<BE>(rgb_in, s + 2));
    rgba_out[d + 3] = 0xFF;
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u16`
/// output. Each `f32` is clamped to `[0, 1]` and scaled by 65535.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, i));
    rgb_out[i + 1] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, i + 1));
    rgb_out[i + 2] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, i + 2));
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u16`
/// output with `A = 0xFFFF`.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, s));
    rgba_out[d + 1] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, s + 1));
    rgba_out[d + 2] = f32_to_u16_clamped(load_f32::<BE>(rgb_in, s + 2));
    rgba_out[d + 3] = 0xFFFF;
  }
}

/// **Lossless** float pass-through: copies the packed `R, G, B` `f32`
/// row into the output buffer without conversion. Source HDR values
/// (> 1.0) and negatives are preserved bit-exact.
///
/// When `BE = true` the input is byte-swapped (big-endian → host-native)
/// so the output is always host-native `f32`.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  // Fast path: encoded byte order matches host-native — pure memcpy.
  // (LE-encoded data on LE host, or BE-encoded data on BE host.)
  // The const-generic `BE == HOST_NATIVE_BE` branch is dead-code-
  // eliminated per monomorphization, so this becomes a single
  // `copy_from_slice` call with no swap loop.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  if BE == HOST_NATIVE_BE {
    rgb_out[..width * 3].copy_from_slice(&rgb_in[..width * 3]);
    return;
  }
  // Slow path: encoded byte order differs from host — byte-swap each
  // f32 element via `u32::from_be` / `u32::from_le` (the dead branch
  // is eliminated since `BE` is const). Output is always host-native.
  for (dst, src) in rgb_out[..width * 3]
    .iter_mut()
    .zip(rgb_in[..width * 3].iter())
  {
    let bits = src.to_bits();
    *dst = f32::from_bits(if BE {
      u32::from_be(bits)
    } else {
      u32::from_le(bits)
    });
  }
}

// ---- Tier 9 — Rgbf16 scalar row kernels --------------------------------
//
// Each kernel widens the 16-bit half-precision input to `f32` on the fly
// and then applies the same clamping / scaling logic as the corresponding
// `rgbf32_to_*_row` function above.  No intermediate heap allocation is
// needed because the widening and the per-element math are interleaved in
// the same tight loop.

/// Converts packed `R, G, B` 16-bit half-precision float input to packed
/// `R, G, B` `u8` output.  Each `half::f16` is widened to `f32`, then
/// clamped to `[0, 1]` and scaled by 255.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, i).to_f32());
    rgb_out[i + 1] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, i + 1).to_f32());
    rgb_out[i + 2] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, i + 2).to_f32());
  }
}

/// Converts packed `R, G, B` 16-bit half-precision float input to packed
/// `R, G, B, A` `u8` output with `A = 0xFF` (the half-float source has no
/// alpha channel).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgba_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, s).to_f32());
    rgba_out[d + 1] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, s + 1).to_f32());
    rgba_out[d + 2] = f32_to_u8_clamped(load_f16::<BE>(rgb_in, s + 2).to_f32());
    rgba_out[d + 3] = 0xFF;
  }
}

/// Converts packed `R, G, B` 16-bit half-precision float input to packed
/// `R, G, B` `u16` output.  Each `half::f16` is widened to `f32`, then
/// clamped to `[0, 1]` and scaled by 65535.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, i).to_f32());
    rgb_out[i + 1] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, i + 1).to_f32());
    rgb_out[i + 2] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, i + 2).to_f32());
  }
}

/// Converts packed `R, G, B` 16-bit half-precision float input to packed
/// `R, G, B, A` `u16` output with `A = 0xFFFF`.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, s).to_f32());
    rgba_out[d + 1] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, s + 1).to_f32());
    rgba_out[d + 2] = f32_to_u16_clamped(load_f16::<BE>(rgb_in, s + 2).to_f32());
    rgba_out[d + 3] = 0xFFFF;
  }
}

/// Widens each `half::f16` element to `f32`, preserving HDR values
/// (> 1.0) and negatives bit-exactly through the widen step.  Output
/// is `f32`; no clamping is applied.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  for (dst, src) in rgb_out[..width * 3]
    .iter_mut()
    .zip(rgb_in[..width * 3].iter())
  {
    let bits = src.to_bits();
    let host_bits = if BE {
      u16::from_be(bits)
    } else {
      u16::from_le(bits)
    };
    *dst = half::f16::from_bits(host_bits).to_f32();
  }
}

/// **Lossless** pass-through: copies the packed `R, G, B` `half::f16` row
/// into the output buffer without any conversion.  Source HDR values and
/// negatives are preserved bit-exact.
///
/// When `BE = true` the input values are byte-swapped to host-native order
/// on output.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  // Fast path: encoded byte order matches host-native — pure memcpy.
  // Mirrors the `rgbf32_to_rgb_f32_row` fast path; the const-generic
  // `BE == HOST_NATIVE_BE` branch is dead-code-eliminated per
  // monomorphization, so this becomes a single `copy_from_slice`.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  if BE == HOST_NATIVE_BE {
    rgb_out[..width * 3].copy_from_slice(&rgb_in[..width * 3]);
    return;
  }
  // Slow path: encoded byte order differs from host — byte-swap each
  // f16 element via `u16::from_be` / `u16::from_le`. Output is always
  // host-native f16.
  for (dst, src) in rgb_out[..width * 3]
    .iter_mut()
    .zip(rgb_in[..width * 3].iter())
  {
    let bits = src.to_bits();
    *dst = half::f16::from_bits(if BE {
      u16::from_be(bits)
    } else {
      u16::from_le(bits)
    });
  }
}

// ---- Tier 9 — Rgbaf32 scalar row kernels (4-channel, real alpha) -------
//
// The alpha-bearing twin of the `rgbf32_to_*_row` kernels above. Input is
// packed `R, G, B, A` `f32` (`4 * width` elements). Two output families:
//   - **drop-alpha** (`*_to_rgb*`): emit `R, G, B` only, alpha discarded.
//   - **real-alpha** (`*_to_rgba*`): emit `R, G, B, A`, the source alpha
//     clamped + scaled exactly like a colour channel (integer outputs) or
//     passed through losslessly (float outputs).
// Every per-element conversion reuses the same `f32_to_u8_clamped` /
// `f32_to_u16_clamped` / `load_f32` helpers, so an `Rgbaf32` output is
// byte-identical to the `Rgbf32` output of its `R, G, B` for the dropped-
// alpha paths.

/// Converts packed `R, G, B, A` `f32` input to packed `R, G, B` `u8`
/// output, **dropping** the source alpha. Each colour `f32` is clamped to
/// `[0, 1]` and scaled by 255.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgb_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = f32_to_u8_clamped(load_f32::<BE>(rgba_in, s));
    rgb_out[d + 1] = f32_to_u8_clamped(load_f32::<BE>(rgba_in, s + 1));
    rgb_out[d + 2] = f32_to_u8_clamped(load_f32::<BE>(rgba_in, s + 2));
  }
}

/// Converts packed `R, G, B, A` `f32` input to packed `R, G, B, A` `u8`
/// output with **real** alpha — every channel (alpha included) is clamped
/// to `[0, 1]` and scaled by 255. Purely elementwise over `4 * width`.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgba_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for (i, dst) in rgba_out[..width * 4].iter_mut().enumerate() {
    *dst = f32_to_u8_clamped(load_f32::<BE>(rgba_in, i));
  }
}

/// Converts packed `R, G, B, A` `f32` input to packed `R, G, B` `u16`
/// output, dropping alpha. Each colour `f32` is clamped to `[0, 1]` and
/// scaled by 65535.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = f32_to_u16_clamped(load_f32::<BE>(rgba_in, s));
    rgb_out[d + 1] = f32_to_u16_clamped(load_f32::<BE>(rgba_in, s + 1));
    rgb_out[d + 2] = f32_to_u16_clamped(load_f32::<BE>(rgba_in, s + 2));
  }
}

/// Converts packed `R, G, B, A` `f32` input to packed `R, G, B, A` `u16`
/// output with real alpha — every channel clamped to `[0, 1]` and scaled
/// by 65535. Purely elementwise over `4 * width`.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for (i, dst) in rgba_out[..width * 4].iter_mut().enumerate() {
    *dst = f32_to_u16_clamped(load_f32::<BE>(rgba_in, i));
  }
}

/// **Lossless** float drop-alpha: copies the packed `R, G, B` channels of
/// the `f32` RGBA row into a packed `R, G, B` `f32` output (alpha dropped).
/// HDR values (> 1.0) and negatives are preserved bit-exact.
///
/// When `BE = true` the input is byte-swapped (big-endian → host-native).
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = load_f32::<BE>(rgba_in, s);
    rgb_out[d + 1] = load_f32::<BE>(rgba_in, s + 1);
    rgb_out[d + 2] = load_f32::<BE>(rgba_in, s + 2);
  }
}

/// **Lossless** float pass-through: copies the packed `R, G, B, A` `f32`
/// row into the output buffer (4 channels, alpha included) without
/// conversion. HDR values and negatives are preserved bit-exact.
///
/// When `BE = true` the input is byte-swapped (big-endian → host-native)
/// so the output is always host-native `f32`.
///
/// # Panics
///
/// Panics (any build profile) if `rgba_in.len() < 4 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf32_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_f32_out row too short");
  // Fast path: encoded byte order matches host-native — pure memcpy
  // (mirrors `rgbf32_to_rgb_f32_row`; the const branch is DCE'd).
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  if BE == HOST_NATIVE_BE {
    rgba_out[..width * 4].copy_from_slice(&rgba_in[..width * 4]);
    return;
  }
  for (dst, src) in rgba_out[..width * 4]
    .iter_mut()
    .zip(rgba_in[..width * 4].iter())
  {
    let bits = src.to_bits();
    *dst = f32::from_bits(if BE {
      u32::from_be(bits)
    } else {
      u32::from_le(bits)
    });
  }
}

// ---- Tier 9 — Rgbaf16 scalar row kernels (4-channel, real alpha) -------
//
// The half-precision twin of the `rgbaf32_*` kernels: each `half::f16`
// element is widened to `f32` on the fly (drop-alpha / integer / widen
// paths) or copied bit-exact (the `_f16_*` lossless paths). Input is
// packed `R, G, B, A` `half::f16` (`4 * width` elements).

/// Packed `R, G, B, A` `f16` → packed `R, G, B` `u8` (drop alpha).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgb_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = f32_to_u8_clamped(load_f16::<BE>(rgba_in, s).to_f32());
    rgb_out[d + 1] = f32_to_u8_clamped(load_f16::<BE>(rgba_in, s + 1).to_f32());
    rgb_out[d + 2] = f32_to_u8_clamped(load_f16::<BE>(rgba_in, s + 2).to_f32());
  }
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B, A` `u8` (real alpha,
/// elementwise clamp + scale by 255).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgba_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for (i, dst) in rgba_out[..width * 4].iter_mut().enumerate() {
    *dst = f32_to_u8_clamped(load_f16::<BE>(rgba_in, i).to_f32());
  }
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B` `u16` (drop alpha).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = f32_to_u16_clamped(load_f16::<BE>(rgba_in, s).to_f32());
    rgb_out[d + 1] = f32_to_u16_clamped(load_f16::<BE>(rgba_in, s + 1).to_f32());
    rgb_out[d + 2] = f32_to_u16_clamped(load_f16::<BE>(rgba_in, s + 2).to_f32());
  }
}

/// Packed `R, G, B, A` `f16` → packed `R, G, B, A` `u16` (real alpha,
/// elementwise clamp + scale by 65535).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for (i, dst) in rgba_out[..width * 4].iter_mut().enumerate() {
    *dst = f32_to_u16_clamped(load_f16::<BE>(rgba_in, i).to_f32());
  }
}

/// Widening drop-alpha: packed `R, G, B, A` `f16` → packed `R, G, B` `f32`
/// (alpha dropped). HDR values and negatives preserved through the widen.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = load_f16::<BE>(rgba_in, s).to_f32();
    rgb_out[d + 1] = load_f16::<BE>(rgba_in, s + 1).to_f32();
    rgb_out[d + 2] = load_f16::<BE>(rgba_in, s + 2).to_f32();
  }
}

/// Widening pass-through: packed `R, G, B, A` `f16` → packed `R, G, B, A`
/// `f32` (4 channels, alpha widened too). HDR and negatives preserved.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_f32_out row too short");
  for (i, dst) in rgba_out[..width * 4].iter_mut().enumerate() {
    *dst = load_f16::<BE>(rgba_in, i).to_f32();
  }
}

/// **Lossless** drop-alpha: packed `R, G, B, A` `f16` → packed `R, G, B`
/// `f16` (alpha dropped, no conversion). Source bit-patterns preserved.
///
/// When `BE = true` the input values are byte-swapped to host-native on
/// output.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgb_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  for x in 0..width {
    let s = x * 4;
    let d = x * 3;
    rgb_out[d] = load_f16::<BE>(rgba_in, s);
    rgb_out[d + 1] = load_f16::<BE>(rgba_in, s + 1);
    rgb_out[d + 2] = load_f16::<BE>(rgba_in, s + 2);
  }
}

/// **Lossless** pass-through: packed `R, G, B, A` `f16` → packed
/// `R, G, B, A` `f16` (4 channels) without conversion. Source bit-patterns
/// preserved bit-exact.
///
/// When `BE = true` the input values are byte-swapped to host-native on
/// output.
///
/// # Panics
///
/// Panics if `rgba_in.len() < 4 * width` or `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbaf16_to_rgba_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_f16_out row too short");
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  if BE == HOST_NATIVE_BE {
    rgba_out[..width * 4].copy_from_slice(&rgba_in[..width * 4]);
    return;
  }
  for (dst, src) in rgba_out[..width * 4]
    .iter_mut()
    .zip(rgba_in[..width * 4].iter())
  {
    let bits = src.to_bits();
    *dst = half::f16::from_bits(if BE {
      u16::from_be(bits)
    } else {
      u16::from_le(bits)
    });
  }
}
