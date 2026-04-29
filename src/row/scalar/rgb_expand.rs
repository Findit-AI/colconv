// ---- RGB → RGBA expand (Strategy A combined-buffer optimization) ------

/// Reads packed `R, G, B` triples and writes packed `R, G, B, A`
/// quadruplets with `A = 0xFF` (opaque). Used by `MixedSinker` impls
/// when callers attach **both** `with_rgb` and `with_rgba`: instead
/// of running the YUV→RGB math twice (once per output format), we
/// run the RGB kernel into the user's RGB buffer and then expand
/// here to derive the RGBA buffer with a single per-byte pass.
///
/// The 3W read is L1-hot from the just-completed RGB write, so the
/// effective memory traffic is roughly 3W RGB write + 4W RGBA write
/// = 7W per row — same as the existing native-RGBA path, but with
/// only one pass through the YUV→RGB math instead of two. See
/// `docs/color-conversion-functions.md` § Ship 8 for the full
/// design discussion (Strategy A vs the alternative B "combined
/// kernel writes both per pixel" deferred to a future PR).
///
/// # Panics (debug builds)
///
/// - `rgb.len() >= 3 * width`
/// - `rgba_out.len() >= 4 * width`
// Only the `MixedSinker` Strategy A fan-out calls this; that lives in
// `crate::sinker::mixed`, gated on `feature = "std"` / `"alloc"`. Without
// either feature the helper would be unused and `-D dead_code` (set by
// `cargo clippy -- -D warnings` on CI) would fail the build.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn expand_rgb_to_rgba_row(rgb: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  // `chunks_exact` lets the compiler hoist the bounds checks out of the
  // loop and keep the per-pixel store as four register writes — tighter
  // codegen than the `[x * 3 + k]` indexing form on the Strategy A hot
  // path (RGB→RGBA fan-out called once per row when both buffers are
  // attached).
  for (rgb_px, rgba_px) in rgb[..width * 3]
    .chunks_exact(3)
    .zip(rgba_out[..width * 4].chunks_exact_mut(4))
  {
    rgba_px[0] = rgb_px[0];
    rgba_px[1] = rgb_px[1];
    rgba_px[2] = rgb_px[2];
    rgba_px[3] = 0xFF;
  }
}

/// `u16` analogue of [`expand_rgb_to_rgba_row`]: copy each `u16` RGB
/// triple into a `u16` RGBA quadruple, with the alpha element set to
/// `(1 << BITS) - 1` (opaque maximum at the input bit depth). Used by
/// `MixedSinker` Strategy A on the **u16** path when both
/// `with_rgb_u16` and `with_rgba_u16` are attached — runs the YUV→RGB
/// math once into the u16 RGB buffer, then this helper fans out to the
/// u16 RGBA buffer with no second per-pixel kernel call.
///
/// `BITS` is a `const` parameter so the alpha constant resolves at
/// compile time per format (10 / 12 / 16 etc.); the compiler folds the
/// `(1 << BITS) - 1` expression to a literal in each monomorphization.
///
/// # Panics (debug builds)
///
/// - `rgb.len() >= 3 * width` (`u16` elements)
/// - `rgba_out.len() >= 4 * width` (`u16` elements)
//
// Scalar prep for Ship 8 Tranche 5: the consumer (MixedSinker Strategy A
// on the u16 path) lands in the follow-up Tranche 5b PR. `dead_code`
// allow lets this prep PR ship the foundation without the eventual call
// site.
#[cfg(any(feature = "std", feature = "alloc"))]
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn expand_rgb_u16_to_rgba_u16_row<const BITS: u32>(
  rgb: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(BITS > 0 && BITS <= 16);
  }

  let rgb_len = width.checked_mul(3).expect("rgb row length overflow");
  let rgba_len = width.checked_mul(4).expect("rgba row length overflow");

  debug_assert!(rgb.len() >= rgb_len, "rgb row too short");
  debug_assert!(rgba_out.len() >= rgba_len, "rgba_out row too short");

  let alpha_max: u16 = ((1u32 << BITS) - 1) as u16;
  for (rgb_px, rgba_px) in rgb[..rgb_len]
    .chunks_exact(3)
    .zip(rgba_out[..rgba_len].chunks_exact_mut(4))
  {
    rgba_px[0] = rgb_px[0];
    rgba_px[1] = rgb_px[1];
    rgba_px[2] = rgb_px[2];
    rgba_px[3] = alpha_max;
  }
}
