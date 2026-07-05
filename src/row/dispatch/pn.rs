//! Semi-planar 4:4:4 (P410 / P412 / P416) dispatchers — RGB + RGBA
//! for both 8-bit and native-depth `u16` outputs. Extracted from
//! `row::mod` for organization.
//!
//! Internal `pub(crate)` helpers `p_n_444_to_rgb_row` /
//! `p_n_444_to_rgb_u16_row` provide the BITS-generic dispatch shared
//! by P410/P412 (`BITS = 10/12`); P416 has its own dedicated kernels
//! (full u16 range; the BITS-generic path doesn't apply).
//!
//! P010 / P012 / P016 (semi-planar 4:2:0) live in `dispatch::yuv420`
//! since they share the 4:2:0 chroma layout with the planar
//! yuv420p9/10/12/14/16 family.

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
  ColorMatrix,
  row::{rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar, uv_full_row_elems},
};

// ---- Pn semi-planar 4:4:4 (P410 / P412 / P416) → RGB --------------------
//
// Same shape as the 4:2:0 / 4:2:2 P-family kernels but with full-width
// interleaved UV (one `U, V` pair per pixel = `2 * width` u16 elements
// per row). BITS ∈ {10, 12} run on the const-generic Q15 i32 family;
// BITS = 16 runs on the dedicated parallel i64-chroma family
// (chroma multiply-add overflows i32 at 16-bit u16 output).

/// Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB
/// dispatcher. Const-generic over `BITS`; dispatches to the best
/// available backend (NEON / SSE4.1 / AVX2 / AVX-512 / wasm simd128),
/// falling back to scalar when no SIMD backend is available or
/// `use_simd` is false.
///
/// Crate-private — public consumers go through the per-format
/// dispatchers (`p410_to_rgb_row`, `p412_to_rgb_row`) which fix
/// `BITS` to a literal.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_444_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe {
            arch::x86_avx512::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe {
            arch::wasm_simd128::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_444_to_rgb_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → native-depth **u16**
/// RGB dispatcher. Output is low-bit-packed (active bits in low
/// `BITS` of each `u16`). Same dispatch shape as
/// [`p_n_444_to_rgb_row`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_444_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_elems(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_444_to_rgb_u16_row::<BITS, BE>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB dispatcher.
/// Y stays on i32 (output-range scaling keeps `coeff × u_d` within
/// i32 for u8 output); chroma multiply-add also stays on i32.
/// Dedicated entry point because the Q15 const-generic family is
/// pinned to BITS ∈ {10, 12}.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgb_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_16_to_rgb_row::<false>(y, uv_full, rgb_out, width, matrix, full_range),
    scalar::p_n_444_16_to_rgb_row::<true>(y, uv_full, rgb_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p416_to_rgb_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p416_to_rgb_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P416 → native-depth **u16** RGB dispatcher (`[0, 65535]`). Chroma
/// multiply-add runs on i64 (overflow safety at 16-bit u16 output);
/// see scalar reference for the rationale.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgb_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgb_min = rgb_row_elems(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          dispatch_be!(
            unsafe { arch::neon::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_16_to_rgb_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range),
    scalar::p_n_444_16_to_rgb_u16_row::<true>(y, uv_full, rgb_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p416_to_rgb_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p416_to_rgb_u16_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P410 → packed u8 RGB. Endian-aware thin wrapper at `BITS = 10`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgb_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_rgb_row::<10, true>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  } else {
    p_n_444_to_rgb_row::<10, false>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`p410_to_rgb_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p410_to_rgb_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P410 → native-depth u16 RGB (10-bit low-packed output). Endian-aware
/// thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgb_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_rgb_u16_row::<10, true>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  } else {
    p_n_444_to_rgb_u16_row::<10, false>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`p410_to_rgb_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p410_to_rgb_u16_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P412 → packed u8 RGB. Endian-aware thin wrapper at `BITS = 12`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgb_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_rgb_row::<12, true>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  } else {
    p_n_444_to_rgb_row::<12, false>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`p412_to_rgb_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p412_to_rgb_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P412 → native-depth u16 RGB (12-bit low-packed output). Endian-aware
/// thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgb_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_rgb_u16_row::<12, true>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  } else {
    p_n_444_to_rgb_u16_row::<12, false>(y, uv_full, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`p412_to_rgb_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p412_to_rgb_u16_row_endian(
    y, uv_full, rgb_out, width, matrix, full_range, use_simd, false,
  );
}

/// P410 (semi-planar 4:4:4, 10-bit high-packed) → packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgba_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_to_rgba_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_to_rgba_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p410_to_rgba_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p410_to_rgba_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// P410 → **native-depth `u16`** packed **RGBA** — output is
/// low-bit-packed (`[0, 1023]`); alpha element is `1023`.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgba_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_elems(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_to_rgba_u16_row::<10, false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_to_rgba_u16_row::<10, true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p410_to_rgba_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p410_to_rgba_u16_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// P412 (semi-planar 4:4:4, 12-bit high-packed) → packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgba_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_to_rgba_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_to_rgba_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p412_to_rgba_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p412_to_rgba_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// P412 → **native-depth `u16`** packed **RGBA** — output is
/// low-bit-packed (`[0, 4095]`); alpha element is `4095`.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgba_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_elems(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_to_rgba_u16_row::<12, false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_to_rgba_u16_row::<12, true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p412_to_rgba_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p412_to_rgba_u16_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit** **RGBA**
/// (`R, G, B, 0xFF`). Routes through the dedicated 16-bit scalar
/// kernel (`scalar::p_n_444_16_to_rgba_row`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgba_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_16_to_rgba_row::<false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_16_to_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p416_to_rgba_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p416_to_rgba_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// P416 → **native-depth `u16`** packed **RGBA** — full-range output
/// `[0, 65535]`; alpha element is `0xFFFF`. Routes through the
/// dedicated 16-bit u16-output scalar kernel
/// (`scalar::p_n_444_16_to_rgba_u16_row`) — i64 chroma multiply.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgba_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_elems(width);
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_16_to_rgba_u16_row::<false>(y, uv_full, rgba_out, width, matrix, full_range),
    scalar::p_n_444_16_to_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`p416_to_rgba_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  p416_to_rgba_u16_row_endian(
    y, uv_full, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

// ---- Pn semi-planar 4:4:4 (P410 / P412 / P416) → HSV (direct) -----------

/// Pn 4:4:4 (`BITS ∈ {10, 12}`) → planar HSV bytes dispatcher, const-generic
/// over `BITS`, `BE` (source byte order), and `LOW_PACKED` (high-bit-packed
/// P410/P412 vs low-bit-packed NV20). Dispatches to the best available
/// backend, falling back to scalar when no SIMD backend is available or
/// `use_simd` is false. The `LOW_PACKED` const threads to the shared
/// de-pack, so P410/P412 (`false`) stay byte-identical while NV20 (`true`)
/// masks the low `BITS`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_444_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe { arch::x86_avx512::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe { arch::wasm_simd128::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_444_to_hsv_row::<BITS, BE, LOW_PACKED>(
    y, uv_full, h_out, s_out, v_out, width, matrix, full_range,
  );
}

/// P410 (semi-planar 4:4:4, 10-bit high-packed) → planar HSV bytes
/// (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`), without materializing a source-width RGB row.
/// Byte-identical to `rgb_to_hsv_row(p410_to_rgb_row_endian(...))` within
/// the selected tier — the SIMD path stages a fixed 64-pixel 8-bit RGB
/// chunk internally. See `scalar::p_n_444_to_hsv_row::<10, false, false>`
/// for the reference. `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p410_to_hsv_row_endian(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_hsv_row::<10, true, false>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    p_n_444_to_hsv_row::<10, false, false>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}

/// P412 (semi-planar 4:4:4, 12-bit high-packed) → planar HSV bytes
/// (OpenCV encoding). Byte-identical to
/// `rgb_to_hsv_row(p412_to_rgb_row_endian(...))` within the selected tier;
/// the SIMD path stages a fixed 64-pixel 8-bit RGB chunk internally. See
/// `scalar::p_n_444_to_hsv_row::<12, false, false>` for the reference.
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p412_to_hsv_row_endian(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_hsv_row::<12, true, false>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    p_n_444_to_hsv_row::<12, false, false>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}

/// P416 (semi-planar 4:4:4, 16-bit) → planar HSV bytes (OpenCV encoding).
/// Byte-identical to `rgb_to_hsv_row(p416_to_rgb_row_endian(...))` within
/// the selected tier; the SIMD path stages a fixed 64-pixel 8-bit RGB
/// chunk internally. See `scalar::p_n_444_16_to_hsv_row::<false>` for the
/// reference. `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p416_to_hsv_row_endian(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  macro_rules! dispatch_be {
    ($call_le:expr, $call_be:expr) => {
      if big_endian { $call_be } else { $call_le }
    };
  }

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          dispatch_be!(
            unsafe { arch::neon::p_n_444_16_to_hsv_row::<false>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::neon::p_n_444_16_to_hsv_row::<true>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::p_n_444_16_to_hsv_row::<false>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::p_n_444_16_to_hsv_row::<true>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::p_n_444_16_to_hsv_row::<false>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::p_n_444_16_to_hsv_row::<true>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::p_n_444_16_to_hsv_row::<false>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::p_n_444_16_to_hsv_row::<true>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::p_n_444_16_to_hsv_row::<false>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::p_n_444_16_to_hsv_row::<true>(y, uv_full, h_out, s_out, v_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::p_n_444_16_to_hsv_row::<false>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range
    ),
    scalar::p_n_444_16_to_hsv_row::<true>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range
    )
  );
}

// ---- NV20 semi-planar 4:4:4 low-bit-packed (BITS = 10) dispatchers -------
//
// The low-bit-packed (`LOW_PACKED = true`, `& 0x03FF`) twins of the P410
// `p_n_444_*` dispatchers — same backends and shared kernels, but the
// de-pack keeps the active value in the LOW 10 bits (NV20's packing) rather
// than the high 10 (P410/P412). These decode the full-width interleaved
// chroma plane the NV20 centered-siting identity path stages (RFC #238).

/// NV20 4:4:4 low-bit-packed (BITS = 10) → packed **u8** RGB (`ALPHA =
/// false`) / RGBA (`ALPHA = true`) dispatcher. Const-generic over `ALPHA`
/// and `BE`; dispatches to the best available backend, falling back to
/// scalar when no SIMD backend is available or `use_simd` is false. Routes
/// the shared `p_n_444_to_rgb_or_rgba_row` kernel with `LOW_PACKED = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn nv20_444_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let out_min = if ALPHA {
    rgba_row_bytes(width)
  } else {
    rgb_row_bytes(width)
  };
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(out.len() >= out_min, "out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe { arch::x86_avx512::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe { arch::wasm_simd128::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_444_to_rgb_or_rgba_row::<10, ALPHA, BE, true>(
    y, uv_full, out, width, matrix, full_range,
  );
}

/// NV20 4:4:4 low-bit-packed (BITS = 10) → **native-depth `u16`** RGB
/// (`ALPHA = false`) / RGBA (`ALPHA = true`) dispatcher. Output is
/// low-bit-packed 10-bit (`yuv420p10le` convention). Same dispatch shape as
/// [`nv20_444_to_rgb_or_rgba_row`]; routes the shared
/// `p_n_444_to_rgb_or_rgba_u16_row` kernel with `LOW_PACKED = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn nv20_444_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let out_min = if ALPHA {
    rgba_row_elems(width)
  } else {
    rgb_row_elems(width)
  };
  let uv_min = uv_full_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_full.len() >= uv_min, "uv_full row too short");
  assert!(out.len() >= out_min, "out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe { arch::neon::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe { arch::x86_avx512::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe { arch::wasm_simd128::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(y, uv_full, out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_444_to_rgb_or_rgba_u16_row::<10, ALPHA, BE, true>(
    y, uv_full, out, width, matrix, full_range,
  );
}

/// NV20 4:4:4 low-bit-packed → packed **u8** RGB. Endian-aware thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_444_to_rgb_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    nv20_444_to_rgb_or_rgba_row::<false, true>(
      y, uv_full, rgb_out, width, matrix, full_range, use_simd,
    );
  } else {
    nv20_444_to_rgb_or_rgba_row::<false, false>(
      y, uv_full, rgb_out, width, matrix, full_range, use_simd,
    );
  }
}

/// NV20 4:4:4 low-bit-packed → packed **8-bit RGBA** (`R, G, B, 0xFF`).
/// Endian-aware thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_444_to_rgba_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    nv20_444_to_rgb_or_rgba_row::<true, true>(
      y, uv_full, rgba_out, width, matrix, full_range, use_simd,
    );
  } else {
    nv20_444_to_rgb_or_rgba_row::<true, false>(
      y, uv_full, rgba_out, width, matrix, full_range, use_simd,
    );
  }
}

/// NV20 4:4:4 low-bit-packed → **native-depth `u16`** RGB (10-bit
/// low-packed output). Endian-aware thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_444_to_rgb_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    nv20_444_to_rgb_or_rgba_u16_row::<false, true>(
      y, uv_full, rgb_out, width, matrix, full_range, use_simd,
    );
  } else {
    nv20_444_to_rgb_or_rgba_u16_row::<false, false>(
      y, uv_full, rgb_out, width, matrix, full_range, use_simd,
    );
  }
}

/// NV20 4:4:4 low-bit-packed → **native-depth `u16`** RGBA (10-bit
/// low-packed output; alpha element `1023`). Endian-aware thin wrapper.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_444_to_rgba_u16_row_endian(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    nv20_444_to_rgb_or_rgba_u16_row::<true, true>(
      y, uv_full, rgba_out, width, matrix, full_range, use_simd,
    );
  } else {
    nv20_444_to_rgb_or_rgba_u16_row::<true, false>(
      y, uv_full, rgba_out, width, matrix, full_range, use_simd,
    );
  }
}

/// NV20 4:4:4 low-bit-packed → planar HSV bytes (OpenCV encoding).
/// Endian-aware thin wrapper over the shared `p_n_444_to_hsv_row`
/// dispatcher with `LOW_PACKED = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv20_444_to_hsv_row_endian(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    p_n_444_to_hsv_row::<10, true, true>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    p_n_444_to_hsv_row::<10, false, true>(
      y, uv_full, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}
