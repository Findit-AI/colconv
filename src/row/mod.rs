//! Crate-internal row-level primitives.
//!
//! These are the composable units that Sinks call on each row handed
//! to them by a source kernel. Source kernels are pure row walkers;
//! the actual arithmetic lives here.
//!
//! Backends:
//! - [`scalar`] — always compiled, reference implementation.
//! - [`arch::neon`] — aarch64 NEON.
//! - Future: `x86_ssse3`, `x86_sse41`, `x86_avx2`, `x86_avx512`,
//!   `wasm_simd128`, each gated on the appropriate `target_arch` /
//!   `target_feature` cfg.
//!
//! Dispatch model: every backend is selected at call time by runtime
//! CPU feature detection — `is_aarch64_feature_detected!` /
//! `is_x86_feature_detected!` under `feature = "std"`, or compile‑time
//! `cfg!(target_feature = ...)` in no‑std builds. `std`'s runtime
//! detection caches the result in an atomic, so per‑call overhead is a
//! single relaxed load plus a branch. Each SIMD kernel itself carries
//! `#[target_feature(enable = "...")]` so its intrinsics execute in an
//! explicitly feature‑enabled context, not one inherited from the
//! target's default features.
//!
//! Output guarantees: every backend is either byte‑identical to
//! [`scalar`] or differs by at most 1 LSB per channel (documented per
//! backend). Tests in [`super::arch`] enforce this contract.

pub(crate) mod arch;
pub(crate) mod scalar;

use crate::ColorMatrix;

/// Converts one row of 4:2:0 YUV to packed BGR.
///
/// Dispatches to the best available backend for the current target.
/// See [`scalar::yuv_420_to_bgr_row_scalar`] for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend. Benchmarks flip this to compare scalar vs SIMD
/// directly on the same input; production code should pass `true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_bgr_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  bgr_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  if use_simd {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
      // SAFETY: `neon_available()` verified NEON is present on this
      // CPU. Bounds / parity invariants are the caller's obligation
      // (same contract as the scalar reference); they are checked
      // with `debug_assert` in debug builds.
      unsafe {
        arch::neon::yuv_420_to_bgr_row_neon(y, u_half, v_half, bgr_out, width, matrix, full_range);
      }
      return;
    }

    // Future x86_64 cascade (avx512 → avx2 → sse4.1 → ssse3) slots in
    // here, each branch guarded by the matching `is_x86_feature_detected!`
    // / `cfg!(target_feature = ...)` pair.
  }

  scalar::yuv_420_to_bgr_row_scalar(y, u_half, v_half, bgr_out, width, matrix, full_range);
}

/// Converts one row of packed BGR to planar HSV (OpenCV 8‑bit
/// encoding). See [`scalar::bgr_to_hsv_row_scalar`] for semantics.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr_to_hsv_row(
  bgr: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  scalar::bgr_to_hsv_row_scalar(bgr, h_out, s_out, v_out, width);
}

// ---- runtime CPU feature detection -----------------------------------
//
// Each `*_available` helper returns `true` iff the named feature is
// present. `feature = "std"` branches use std's cached
// `is_*_feature_detected!` macros (atomic load + branch after the
// first call). No‑std branches fall back to `cfg!(target_feature = ...)`
// which is resolved at compile time. Helpers are only compiled for
// targets where the corresponding feature exists.

/// NEON availability on aarch64.
#[cfg(all(target_arch = "aarch64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn neon_available() -> bool {
  std::arch::is_aarch64_feature_detected!("neon")
}

/// NEON availability on aarch64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn neon_available() -> bool {
  cfg!(target_feature = "neon")
}
