//! 8-bit YUV 4:2:0 → RGB / RGBA dispatchers (`yuv_420_to_rgb_row`,
//! `yuv_420_to_rgba_row`). Extracted from the parent `dispatch::yuv420`
//! module per source format for organization.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  KernelMatrix,
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};
// `ChromaDerivedNcl` resolves its coefficients from the signalled primaries,
// so the primaries-aware dispatchers need the type.
use crate::Primaries;

/// Converts one row of 4:2:0 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_420_to_rgb_row` for the full semantic
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
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary. The unsafe SIMD
  // kernels below rely on these invariants for bounds‑free pointer
  // arithmetic, so we validate in *release* builds too — not just
  // under `debug_assert!`. Kernels keep their own `debug_assert!`s as
  // internal sanity checks.
  //
  // `rgb_min` uses `checked_mul` because `3 * width` can wrap `usize`
  // on 32‑bit targets (wasm32, i686) for extreme widths. Without the
  // guard, a wrapped product could admit an undersized `rgb_out` and
  // let the scalar loop's `x * 3` indexing or a SIMD kernel's
  // pointer arithmetic run off the end.
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

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
      }
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
      }
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
      }
      _ => {
        // Targets without a SIMD backend (riscv64, powerpc, …) fall
        // through to the scalar path below.
      }
    }
  }

  scalar::yuv_420_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of 4:2:0 YUV to packed **RGBA** (8-bit).
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output). The first three bytes per pixel are
/// byte-identical to what [`yuv_420_to_rgb_row`] would write.
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 4` multiplication via [`rgba_row_bytes`].
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::yuv_420_to_rgba_row(y, u_half, v_half, rgba_out, width, matrix, full_range);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::yuv_420_to_rgba_row(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::yuv_420_to_rgba_row(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::yuv_420_to_rgba_row(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time availability verified.
          unsafe {
            arch::wasm_simd128::yuv_420_to_rgba_row(
              y, u_half, v_half, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::yuv_420_to_rgba_row(y, u_half, v_half, rgba_out, width, matrix, full_range);
}

/// [`yuv_420_to_rgb_row`] that additionally honours
/// [`KernelMatrix::ChromaDerivedNcl`] (ITU-T H.273 `MatrixCoefficients =
/// 12`), whose `Kr` / `Kb` are *derived* from the signalled colour
/// `primaries` rather than fixed by the matrix tag.
///
/// For `ChromaDerivedNcl` with primaries that carry chromaticities, the
/// coefficients are resolved once via
/// `scalar::Coefficients::for_matrix_with_primaries` and the row runs on the
/// **scalar reference**: no SIMD kernel can derive
/// this set from the matrix tag, and routing the matrix deterministically to
/// scalar keeps it free of any SIMD-vs-scalar split. Every other matrix — and
/// `ChromaDerivedNcl` without usable primaries — falls through to
/// [`yuv_420_to_rgb_row`] unchanged (byte-identical, full SIMD dispatch,
/// the prior BT.709 fallback for an unresolved `ChromaDerivedNcl`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_rgb_row_primaries(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  primaries: &Primaries,
  full_range: bool,
  use_simd: bool,
) {
  if matches!(matrix, KernelMatrix::ChromaDerivedNcl) && primaries.chromaticities().is_some() {
    // Same release-build boundary asserts as `yuv_420_to_rgb_row` (the
    // scalar kernel only `debug_assert!`s its bounds).
    assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
    let rgb_min = rgb_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u_half.len() >= width / 2, "u_half row too short");
    assert!(v_half.len() >= width / 2, "v_half row too short");
    assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
    let coeffs = scalar::Coefficients::for_matrix_with_primaries(matrix, primaries);
    scalar::yuv_420_to_rgb_row_with_coeffs(y, u_half, v_half, rgb_out, width, coeffs, full_range);
    return;
  }
  yuv_420_to_rgb_row(
    y, u_half, v_half, rgb_out, width, matrix, full_range, use_simd,
  );
}

/// [`yuv_420_to_rgba_row`] with the [`KernelMatrix::ChromaDerivedNcl`]
/// primaries-derived path — the RGBA twin of [`yuv_420_to_rgb_row_primaries`]
/// (alpha `0xFF`, opaque). See it for the routing rationale.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_rgba_row_primaries(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  primaries: &Primaries,
  full_range: bool,
  use_simd: bool,
) {
  if matches!(matrix, KernelMatrix::ChromaDerivedNcl) && primaries.chromaticities().is_some() {
    assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
    let rgba_min = rgba_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u_half.len() >= width / 2, "u_half row too short");
    assert!(v_half.len() >= width / 2, "v_half row too short");
    assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");
    let coeffs = scalar::Coefficients::for_matrix_with_primaries(matrix, primaries);
    scalar::yuv_420_to_rgba_row_with_coeffs(y, u_half, v_half, rgba_out, width, coeffs, full_range);
    return;
  }
  yuv_420_to_rgba_row(
    y, u_half, v_half, rgba_out, width, matrix, full_range, use_simd,
  );
}

/// Converts one row of 4:2:0 YUV **directly** to planar HSV bytes
/// (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`, `S, V ∈
/// [0, 255]`), without materializing a source-width RGB row. Output is
/// byte-identical to `rgb_to_hsv_row(yuv_420_to_rgb_row(...))` within
/// the selected tier. Also serves 4:2:2 (identical per-row chroma
/// shape).
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_420_to_hsv_row` for the full semantic specification.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_420_to_hsv_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — see
  // [`yuv_420_to_rgb_row`] for rationale. The unsafe SIMD kernels stage
  // a fixed 64-pixel RGB chunk internally (no source-width RGB
  // allocation), so only the source / output bounds need validating
  // here.
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present. Bounds
          // are the caller's obligation (asserted above).
          unsafe {
            arch::neon::yuv_420_to_hsv_row(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::yuv_420_to_hsv_row(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::yuv_420_to_hsv_row(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::yuv_420_to_hsv_row(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time availability verified.
          unsafe {
            arch::wasm_simd128::yuv_420_to_hsv_row(
              y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::yuv_420_to_hsv_row(
    y, u_half, v_half, h_out, s_out, v_out, width, matrix, full_range,
  );
}
