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

/// Converts one row of 4:2:0 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See [`scalar::yuv_420_to_rgb_row`] for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend. Benchmarks flip this to compare scalar vs SIMD
/// directly on the same input; production code should pass `true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (same contract as the scalar reference); they are checked
          // with `debug_assert` in debug builds.
          unsafe {
            arch::neon::yuv_420_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          // Bounds / parity invariants are the caller's obligation.
          unsafe {
            arch::x86_avx512::yuv_420_to_rgb_row(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (same contract as the scalar reference); they are checked
          // with `debug_assert` in debug builds.
          unsafe {
            arch::x86_avx2::yuv_420_to_rgb_row(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          // Bounds / parity invariants are the caller's obligation
          // (same contract as the scalar reference).
          unsafe {
            arch::x86_sse41::yuv_420_to_rgb_row(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      // Future x86_64 tiers (avx512 promoted above AVX2, ssse3 below
      // SSE4.1) slot in here, each branch guarded by the matching
      // `is_x86_feature_detected!` / `cfg!(target_feature = ...)` pair.
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: `simd128_available()` (compile‑time
          // `cfg!(target_feature = "simd128")`) verified that simd128
          // is on. WASM has no runtime detection — the module's SIMD
          // support is fixed at produce‑time. Bounds / parity
          // invariants are the caller's obligation.
          unsafe {
            arch::wasm_simd128::yuv_420_to_rgb_row(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend (riscv64, powerpc, …) fall
        // through to the scalar path below.
      }
    }
  }

  scalar::yuv_420_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of packed RGB to planar HSV (OpenCV 8‑bit
/// encoding). See [`scalar::rgb_to_hsv_row`] for semantics.
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend (same semantics as `yuv_420_to_rgb_row`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds invariants are the caller's obligation,
          // checked with `debug_assert` in debug builds.
          unsafe {
            arch::neon::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      _ => {
        // Other targets currently fall through to scalar until HSV
        // SIMD backends land for them (x86 cascade and wasm_simd128 are
        // follow‑ups to the NEON kernel).
      }
    }
  }

  scalar::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
}

/// Rewrites a row of packed BGR to packed RGB by swapping the outer
/// two channels (byte 0 ↔ byte 2) of every triple. `input` and
/// `output` must not alias.
///
/// The underlying transformation is self‑inverse, so
/// [`rgb_to_bgr_row`] shares the same implementation — use whichever
/// name reads more naturally at the call site.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr_to_rgb_row(bgr: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  swap_rb_channels_row(bgr, rgb_out, width, use_simd);
}

/// Rewrites a row of packed RGB to packed BGR by swapping the outer
/// two channels. See [`bgr_to_rgb_row`] — this is an alias that reads
/// more naturally for the opposite direction.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb_to_bgr_row(rgb: &[u8], bgr_out: &mut [u8], width: usize, use_simd: bool) {
  swap_rb_channels_row(rgb, bgr_out, width, use_simd);
}

/// Shared dispatcher behind `bgr_to_rgb_row` / `rgb_to_bgr_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn swap_rb_channels_row(input: &[u8], output: &mut [u8], width: usize, use_simd: bool) {
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 just verified.
          unsafe {
            arch::x86_avx2::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 just verified.
          unsafe {
            arch::x86_sse41::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::bgr_rgb_swap_row(input, output, width);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::bgr_rgb_swap_row(input, output, width);
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

/// AVX2 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx2_available() -> bool {
  std::arch::is_x86_feature_detected!("avx2")
}

/// AVX2 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn avx2_available() -> bool {
  cfg!(target_feature = "avx2")
}

/// SSE4.1 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn sse41_available() -> bool {
  std::arch::is_x86_feature_detected!("sse4.1")
}

/// SSE4.1 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn sse41_available() -> bool {
  cfg!(target_feature = "sse4.1")
}

/// AVX‑512 (F + BW) availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx512_available() -> bool {
  std::arch::is_x86_feature_detected!("avx512bw")
}

/// AVX‑512 (F + BW) availability on x86_64 — no‑std variant
/// (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn avx512_available() -> bool {
  cfg!(target_feature = "avx512bw")
}

/// simd128 availability on wasm32. WASM has no runtime CPU detection
/// (SIMD support is fixed at module produce time), so this is always
/// a compile‑time check regardless of the `std` feature.
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn simd128_available() -> bool {
  cfg!(target_feature = "simd128")
}
