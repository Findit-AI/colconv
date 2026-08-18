//! 8-bit YUV 4:4:4 → RGB / RGBA dispatchers (`yuv_444_to_rgb_row`,
//! `yuv_444_to_rgba_row`). Extracted from the parent
//! `dispatch::yuv444` module per source format for organization.

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
// which the centered chroma-siting decode routes through these 4:4:4 kernels.
use crate::Primaries;

/// Converts one row of YUV 4:4:4 planar to packed RGB. Dispatches
/// to the best available SIMD backend for the current target.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the difference
/// is 4:4:4 chroma — one U / V pair per Y pixel, full-width chroma
/// planes, no chroma upsampling, no width parity constraint. See
/// `scalar::yuv_444_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
}

/// Converts one row of YUV 4:4:4 planar to packed **RGBA** (8-bit).
/// Same numerical contract as [`yuv_444_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel). `rgba_out.len() >= 4 * width`.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range);
}

/// [`yuv_444_to_rgb_row`] that additionally honours
/// [`KernelMatrix::ChromaDerivedNcl`] (ITU-T H.273 `MatrixCoefficients =
/// 12`), whose `Kr` / `Kb` are derived from the signalled colour
/// `primaries`. The 4:4:4 twin of `yuv_420_to_rgb_row_primaries`: the
/// centered chroma-siting (#302) `Yuv420p` decode upsamples its 4:2:0 chroma
/// to full width and feeds it here, so this seam must also resolve
/// `ChromaDerivedNcl` from primaries rather than the matrix tag.
///
/// For `ChromaDerivedNcl` with chromaticity-bearing primaries the
/// coefficients are resolved once via
/// `scalar::Coefficients::for_matrix_with_primaries` and the row runs on the
/// **scalar reference** — no SIMD kernel can derive this set from the tag,
/// and the deterministic scalar route keeps it free of any SIMD-vs-scalar
/// split. Every other matrix, and an unresolvable `ChromaDerivedNcl`, fall
/// through to [`yuv_444_to_rgb_row`] unchanged (byte-identical, full SIMD).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgb_row_primaries(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  primaries: &Primaries,
  full_range: bool,
  use_simd: bool,
) {
  if matches!(matrix, KernelMatrix::ChromaDerivedNcl) && primaries.chromaticities().is_some() {
    let rgb_min = rgb_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
    let coeffs = scalar::Coefficients::for_matrix_with_primaries(matrix, primaries);
    scalar::yuv_444_to_rgb_row_with_coeffs(y, u, v, rgb_out, width, coeffs, full_range);
    return;
  }
  yuv_444_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// [`yuv_444_to_rgba_row`] with the [`KernelMatrix::ChromaDerivedNcl`]
/// primaries-derived path — the RGBA twin of [`yuv_444_to_rgb_row_primaries`]
/// (alpha `0xFF`, opaque). See it for the routing rationale.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_rgba_row_primaries(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  primaries: &Primaries,
  full_range: bool,
  use_simd: bool,
) {
  if matches!(matrix, KernelMatrix::ChromaDerivedNcl) && primaries.chromaticities().is_some() {
    let rgba_min = rgba_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");
    let coeffs = scalar::Coefficients::for_matrix_with_primaries(matrix, primaries);
    scalar::yuv_444_to_rgba_row_with_coeffs(y, u, v, rgba_out, width, coeffs, full_range);
    return;
  }
  yuv_444_to_rgba_row(y, u, v, rgba_out, width, matrix, full_range, use_simd);
}

/// Converts one row of 4:4:4 YUV **directly** to planar HSV bytes
/// (OpenCV encoding), without materializing a source-width RGB row.
/// Output is byte-identical to `rgb_to_hsv_row(yuv_444_to_rgb_row(...))`
/// within the selected tier. Also serves 4:4:0 (identical per-row chroma
/// shape).
///
/// `use_simd = false` forces the scalar reference path. See
/// `scalar::yuv_444_to_hsv_row` for the semantic specification.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_444_to_hsv_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified; bounds asserted above.
          unsafe {
            arch::neon::yuv_444_to_hsv_row(
              y, u, v, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444_to_hsv_row(
              y, u, v, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444_to_hsv_row(
              y, u, v, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444_to_hsv_row(
              y, u, v, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444_to_hsv_row(
              y, u, v, h_out, s_out, v_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444_to_hsv_row(y, u, v, h_out, s_out, v_out, width, matrix, full_range);
}
