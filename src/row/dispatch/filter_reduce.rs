//! Runtime SIMD dispatcher for the separable **filter** resampler's H/V
//! passes — the signed twin of [`area_reduce`](super::area_reduce).
//!
//! The arena ([`FilterPaddedSpans`]) is the signed analogue of
//! [`PaddedSpans`](super::area_reduce::PaddedSpans): per-span `f32`
//! coefficients padded to a multiple of 8 with zeros (zero annihilates),
//! so the kernels run pure wide loads and the dispatcher does O(1) binds
//! before each unsafe kernel. The H-pass leaves the **raw `f64` dot
//! product** in `h_tmp`; the stream quantizes it per element type
//! afterwards. Unlike the integer area dispatchers the kernels do not
//! reproduce the scalar reference bit-for-bit — float addition does not
//! associate, so the reordered tap sum lands within a small tolerance
//! (the integer streams within `±1` LSB of Pillow). The V-pass is
//! element-wise (mul+add) and stays bit-equal to the scalar reference.

#![cfg_attr(
  any(not(feature = "std"), not(any(feature = "rgb", feature = "gray"))),
  allow(dead_code)
)]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use std::vec::Vec;

use crate::row::scalar::{filter_reduce as scalar, filter_reduce::FilterElem};
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

/// Plan-time SIMD staging for the filter H-pass: the span starts, the
/// per-span **real tap count** (so a kernel masks only the trailing
/// padding lanes, never a real in-window zero coefficient), a zero-padded
/// `f32` copy of the signed coefficient arena (every span padded to a
/// multiple of 8 so kernels run pure wide loads), and the furthest source
/// cell any real span touches (a padded span may overhang the row by up to
/// 7 cells; kernels stage those final chunks through stack copies).
///
/// Invalid states are unrepresentable: the fields are private and only
/// [`Self::build`] populates them, so a value existing proves per-span
/// 8-multiple padding, monotonic offsets, a `ksize` matching each span's
/// real length, and an accurate `max_reach` — the dispatcher then needs
/// only O(1) binds per row (shape vs the plan, reach vs the row) before
/// entering an unsafe kernel.
///
/// The signed twin of [`PaddedSpans`](super::area_reduce::PaddedSpans):
/// `f32` coefficients carry any value, so there is no `u16` weight bound
/// to screen — only the geometry and allocation guards remain. Unlike the
/// integer arena it also keeps `ksize`: an integer span's padding zeros
/// always annihilate (the samples are finite), but a signed-filter span
/// over `f32` input can carry a non-finite sample under a **real** zero
/// coefficient, where `0.0 * NaN == NaN` must survive exactly as the
/// scalar reference computes it — so the kernels distinguish a padding
/// lane (mask to `+0.0`) from a real zero-coefficient lane (multiply as
/// is) by lane index against `ksize`.
#[derive(Debug)]
pub(crate) struct FilterPaddedSpans {
  starts: Vec<usize>,
  /// Real (unpadded) tap count per span — the lane boundary the kernels
  /// mask beyond. `ksize[j] == offsets[j + 1] - offsets[j]`.
  ksize: Vec<usize>,
  coeffs: Vec<f32>,
  off: Vec<usize>,
  max_reach: usize,
}

impl FilterPaddedSpans {
  /// Builds the staging arena from a plan's filter spans. Returns `None`
  /// — routing the dispatcher to the scalar reference — on arithmetic
  /// overflow or when an allocation is refused: the arena is an optional
  /// accelerator, never a reason to fail stream creation.
  pub(crate) fn build(starts: &[usize], offsets: &[usize], coeffs: &[f32]) -> Option<Self> {
    let out = starts.len();
    if offsets.len() != out + 1 {
      return None;
    }
    let mut total = 0usize;
    let mut max_reach = 0usize;
    for j in 0..out {
      let k = offsets[j + 1].checked_sub(offsets[j])?;
      let k_pad = k.div_ceil(8).checked_mul(8)?;
      total = total.checked_add(k_pad)?;
      max_reach = max_reach.max(starts[j].checked_add(k)?);
    }
    if coeffs.len() < offsets[out] {
      return None;
    }
    let mut cf = Vec::new();
    let mut off = Vec::new();
    let mut starts_own = Vec::new();
    let mut ksize = Vec::new();
    if cf.try_reserve_exact(total).is_err()
      || off.try_reserve_exact(out + 1).is_err()
      || starts_own.try_reserve_exact(out).is_err()
      || ksize.try_reserve_exact(out).is_err()
    {
      return None;
    }
    starts_own.extend_from_slice(starts);
    off.push(0);
    for j in 0..out {
      let span = &coeffs[offsets[j]..offsets[j + 1]];
      ksize.push(span.len());
      cf.extend_from_slice(span);
      cf.resize(cf.len() + (span.len().div_ceil(8) * 8 - span.len()), 0.0);
      off.push(cf.len());
    }
    Some(Self {
      starts: starts_own,
      ksize,
      coeffs: cf,
      off,
      max_reach,
    })
  }
}

/// The crate element types the filter dispatcher handles. Each carries the
/// per-arch H-pass + V-pass kernel selection so the dispatcher functions
/// stay element-generic, mirroring how the area engine has one dispatcher
/// per element. A blanket scalar fallback covers every target and every
/// channel count other than 1 / 3.
///
/// # Safety
///
/// The `h_*`/`v` methods enter unsafe SIMD kernels; callers must have
/// completed the [`FilterPaddedSpans`] construction-proof binds first.
pub(crate) trait FilterSimdElem: FilterElem + Copy + Default {
  /// 1-channel H-pass into `h_tmp` via the highest available SIMD tier.
  /// `ksize[j]` is span `j`'s real tap count, so a kernel masks only the
  /// trailing padding lanes — a real in-window zero coefficient still
  /// multiplies its (possibly non-finite) sample exactly as scalar does.
  ///
  /// # Safety
  ///
  /// The arena binds to this call (shape + reach proven by the caller).
  unsafe fn h_c1(
    row: &[Self],
    starts: &[usize],
    ksize: &[usize],
    coeffs: &[f32],
    off: &[usize],
    h_tmp: &mut [f64],
  );

  /// 3-channel H-pass into `h_tmp` via the highest available SIMD tier.
  /// See [`Self::h_c1`] for the `ksize` padding-mask contract.
  ///
  /// # Safety
  ///
  /// The arena binds to this call (shape + reach proven by the caller).
  unsafe fn h_c3(
    row: &[Self],
    starts: &[usize],
    ksize: &[usize],
    coeffs: &[f32],
    off: &[usize],
    h_tmp: &mut [f64],
  );
}

/// Generates a `FilterSimdElem` impl whose `h_c1`/`h_c3` bodies are the
/// per-arch `cfg_select!`, naming this element's concrete kernel entry
/// points (`filter_h_reduce_row_<elem>_c{1,3}`). One macro keeps the three
/// element impls identical bar the kernel-name infix.
macro_rules! impl_filter_simd_elem {
  ($elem:ty, $c1:ident, $c3:ident) => {
    impl FilterSimdElem for $elem {
      #[cfg_attr(not(tarpaulin), inline(always))]
      unsafe fn h_c1(
        row: &[$elem],
        starts: &[usize],
        ksize: &[usize],
        coeffs: &[f32],
        off: &[usize],
        h_tmp: &mut [f64],
      ) {
        cfg_select! {
          target_arch = "aarch64" => {
            if neon_available() {
              // SAFETY: NEON available; arena bound by the caller.
              unsafe { arch::neon::filter_reduce::$c1(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          target_arch = "x86_64" => {
            if avx512_available() {
              // SAFETY: AVX-512F+BW verified; arena bound by the caller.
              unsafe { arch::x86_avx512::filter_reduce::$c1(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
            if avx2_available() {
              // SAFETY: AVX2 verified; arena bound by the caller.
              unsafe { arch::x86_avx2::filter_reduce::$c1(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
            if sse41_available() {
              // SAFETY: SSE4.1 verified; arena bound by the caller.
              unsafe { arch::x86_sse41::filter_reduce::$c1(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          target_arch = "wasm32" => {
            if simd128_available() {
              // SAFETY: simd128 enabled; arena bound by the caller.
              unsafe { arch::wasm_simd128::filter_reduce::$c1(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          _ => {}
        }
        // No SIMD tier ran (none available — e.g. `pixon_force_scalar`, or
        // a CPU lacking the feature). Fall back on the scalar reference, but
        // over the PADDED arena bounded by `ksize`: the real-span reference
        // (`filter_h_reduce_row_c1`) takes raw `offsets`, whereas here only
        // the padded `off` / `coeffs` are in scope, and iterating a padded
        // span's full (8-multiple) length would read the padding lanes that
        // overhang the row — the overhang a real kernel stages through stack
        // copies. `ksize` bounds the sum to the real taps (padding coeffs are
        // zero, so the result is identical to the unpadded reference).
        scalar::filter_h_reduce_row_padded_c1(row, starts, ksize, coeffs, off, h_tmp);
      }

      #[cfg_attr(not(tarpaulin), inline(always))]
      unsafe fn h_c3(
        row: &[$elem],
        starts: &[usize],
        ksize: &[usize],
        coeffs: &[f32],
        off: &[usize],
        h_tmp: &mut [f64],
      ) {
        cfg_select! {
          target_arch = "aarch64" => {
            if neon_available() {
              // SAFETY: NEON available; arena bound by the caller.
              unsafe { arch::neon::filter_reduce::$c3(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          target_arch = "x86_64" => {
            if avx512_available() {
              // SAFETY: AVX-512F+BW verified; arena bound by the caller.
              unsafe { arch::x86_avx512::filter_reduce::$c3(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
            if avx2_available() {
              // SAFETY: AVX2 verified; arena bound by the caller.
              unsafe { arch::x86_avx2::filter_reduce::$c3(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
            if sse41_available() {
              // SAFETY: SSE4.1 verified; arena bound by the caller.
              unsafe { arch::x86_sse41::filter_reduce::$c3(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          target_arch = "wasm32" => {
            if simd128_available() {
              // SAFETY: simd128 enabled; arena bound by the caller.
              unsafe { arch::wasm_simd128::filter_reduce::$c3(row, starts, ksize, coeffs, off, h_tmp); }
              return;
            }
          },
          _ => {}
        }
        // See `h_c1`: bound the padded-arena scalar fallback to `ksize` real
        // taps so the absent SIMD kernel's row overhang is never read.
        scalar::filter_h_reduce_row_padded_c3(row, starts, ksize, coeffs, off, h_tmp);
      }
    }
  };
}

impl_filter_simd_elem!(u8, filter_h_reduce_row_u8_c1, filter_h_reduce_row_u8_c3);
impl_filter_simd_elem!(u16, filter_h_reduce_row_u16_c1, filter_h_reduce_row_u16_c3);
impl_filter_simd_elem!(f32, filter_h_reduce_row_f32_c1, filter_h_reduce_row_f32_c3);

/// The `u32` filter element ships **no per-arch SIMD kernel** — the integer
/// SIMD H-passes top out at `u16`, and a `u32` sample widens to the shared
/// `f64` accumulation domain losslessly anyway, so the exact scalar `f64`
/// reduce *is* the production H-pass (and is deterministic, hence 0-ULP). The
/// impl is hand-written (not via [`impl_filter_simd_elem`]) to delegate
/// straight to the padded-arena scalar reference, bypassing the `cfg_select!`
/// SIMD dispatch the macro would emit. The `u32` source formats are rare and
/// this is the downscale path, so the scalar `f64` dot product is the right
/// cost/precision trade.
impl FilterSimdElem for u32 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  unsafe fn h_c1(
    row: &[u32],
    starts: &[usize],
    ksize: &[usize],
    coeffs: &[f32],
    off: &[usize],
    h_tmp: &mut [f64],
  ) {
    scalar::filter_h_reduce_row_padded_c1(row, starts, ksize, coeffs, off, h_tmp);
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  unsafe fn h_c3(
    row: &[u32],
    starts: &[usize],
    ksize: &[usize],
    coeffs: &[f32],
    off: &[usize],
    h_tmp: &mut [f64],
  ) {
    scalar::filter_h_reduce_row_padded_c3(row, starts, ksize, coeffs, off, h_tmp);
  }
}

/// Runtime-dispatched filter H-pass of one source row into `h_tmp`
/// (`starts.len() * channels` raw `f64` dot products).
///
/// `padded` is the plan-time staging arena; kernels consume its
/// internally-proven spans exclusively. `None` — like channel counts
/// other than 1 and 3, and `use_simd == false` — routes to the scalar
/// reference, which every backend matches within the float tolerance.
///
/// # Panics
///
/// When a supplied arena does not bind to this call: span count differing
/// from the plan's, or reach past the row — both are internal contract
/// bugs, not fallback cases.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_h_reduce_row<S: FilterSimdElem>(
  row: &[S],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  coeffs: &[f32],
  padded: Option<&FilterPaddedSpans>,
  h_tmp: &mut [f64],
  use_simd: bool,
) {
  // Release-mode safety boundary before any unsafe SIMD dispatch — the
  // per-arch helpers only debug_assert these.
  let out = starts.len();
  assert!(offsets.len() == out + 1, "offsets shape");
  assert!(h_tmp.len() >= out * channels, "h_tmp too short");
  assert!(coeffs.len() >= offsets[out], "coeffs arena too short");

  if use_simd
    && (channels == 1 || channels == 3)
    && let Some(p) = padded
  {
    // O(1) binds completing the construction proof for this call: same
    // span count as the plan (so the h_tmp bound covers kernel writes)
    // and every real span inside the row (so kernel index arithmetic
    // cannot wrap; the padded overhang past a span's last coefficient
    // stages through guarded stack copies).
    assert!(
      p.starts.len() == out && p.ksize.len() == out,
      "padded arena shape"
    );
    assert!(
      p.off.len() == out + 1 && p.off[out] <= p.coeffs.len(),
      "padded arena layout"
    );
    assert!(
      p.max_reach <= row.len() / channels,
      "padded arena exceeds row"
    );
    if channels == 1 {
      // SAFETY: the binds above complete the arena's construction proof
      // for this row; the kernel reads only proven spans.
      unsafe { S::h_c1(row, &p.starts, &p.ksize, &p.coeffs, &p.off, h_tmp) };
    } else {
      // SAFETY: as above, 3-channel variant.
      unsafe { S::h_c3(row, &p.starts, &p.ksize, &p.coeffs, &p.off, h_tmp) };
    }
    return;
  }
  if channels == 1 {
    scalar::filter_h_reduce_row_c1(row, starts, offsets, coeffs, h_tmp);
  } else if channels == 3 {
    scalar::filter_h_reduce_row_c3(row, starts, offsets, coeffs, h_tmp);
  } else {
    filter_h_reduce_row_scalar_generic(row, channels, starts, offsets, coeffs, h_tmp);
  }
}

/// Scalar H-pass for channel counts other than 1 / 3 (no SIMD kernel) —
/// the generic per-channel dot product the stream would otherwise inline.
#[cfg_attr(not(tarpaulin), inline(always))]
fn filter_h_reduce_row_scalar_generic<S: FilterElem>(
  row: &[S],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  coeffs: &[f32],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[offsets[j]..offsets[j + 1]];
    let base = j * channels;
    for ch in 0..channels {
      let mut acc = 0.0f64;
      for (k, &w) in span.iter().enumerate() {
        acc += f64::from(w) * row[(start + k) * channels + ch].widen();
      }
      h_tmp[base + ch] = acc;
    }
  }
}

/// Runtime-dispatched filter V-pass AXPY: `acc[i] += w * h_tmp[i]` over the
/// H-reduced (and quantized) row, in `f64`. `w` is one signed `f32`
/// vertical coefficient. The V-pass is element-wise — no reordering — so
/// every backend matches the scalar reference bit-for-bit (only the H-pass
/// carries the float tolerance); `use_simd == false` takes the reference.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32, use_simd: bool) {
  // Release-mode safety boundary before any unsafe SIMD dispatch — the
  // per-arch helpers only debug_assert this.
  assert!(h_tmp.len() >= acc.len(), "h_tmp too short");

  if !use_simd {
    return scalar::filter_v_accumulate(acc, h_tmp, w);
  }
  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
  ))]
  {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON available; bounds asserted above.
          unsafe { arch::neon::filter_reduce::filter_v_accumulate(acc, h_tmp, w); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F verified at runtime; bounds asserted above.
          unsafe { arch::x86_avx512::filter_reduce::filter_v_accumulate(acc, h_tmp, w); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified at runtime; bounds asserted above.
          unsafe { arch::x86_avx2::filter_reduce::filter_v_accumulate(acc, h_tmp, w); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified at runtime; bounds asserted above.
          unsafe { arch::x86_sse41::filter_reduce::filter_v_accumulate(acc, h_tmp, w); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 enabled at compile time; bounds asserted above.
          unsafe { arch::wasm_simd128::filter_reduce::filter_v_accumulate(acc, h_tmp, w); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::filter_v_accumulate(acc, h_tmp, w);
}

#[cfg(all(test, feature = "std"))]
mod tests;
