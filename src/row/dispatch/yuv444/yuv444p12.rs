//! 12-bit planar YUV 4:4:4 dispatchers — 4 variants.

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
  row::{rgba_row_bytes, rgba_row_elems, scalar},
};
// The open descriptor vocabulary: the non-affine splices below gate on a
// matrix tag that has no `KernelMatrix` spelling, so it cannot ride the row
// and arrives separately as `tag`.
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::ColorMatrix;
// Used only by the `any(std, alloc)`-gated non-affine Smpte2085 row wrappers
// (the base RGB dispatcher preflights inside the kernel, not here).
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::row::{rgb_row_bytes, rgb_row_elems};

use super::{yuv_444p_n_to_rgb_row, yuv_444p_n_to_rgb_u16_row};

/// YUV 4:4:4 planar 12-bit → u8 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    yuv_444p_n_to_rgb_row::<12, true>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  } else {
    yuv_444p_n_to_rgb_row::<12, false>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`yuv444p12_to_rgb_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p12_to_rgb_row_endian(y, u, v, rgb_out, width, matrix, full_range, use_simd, false);
}

/// YUV 4:4:4 planar 12-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    yuv_444p_n_to_rgb_u16_row::<12, true>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  } else {
    yuv_444p_n_to_rgb_u16_row::<12, false>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
  }
}

/// LE-only wrapper around [`yuv444p12_to_rgb_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p12_to_rgb_u16_row_endian(y, u, v, rgb_out, width, matrix, full_range, use_simd, false);
}

/// Converts one row of **12-bit** YUV 4:4:4 to packed **8-bit**
/// **RGBA** (`R, G, B, 0xFF`).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
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
            unsafe { arch::neon::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range),
    scalar::yuv_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`yuv444p12_to_rgba_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p12_to_rgba_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **12-bit** YUV 4:4:4 to **native-depth `u16`**
/// packed **RGBA** — output is low-bit-packed (`[0, 4095]`); alpha
/// element is `4095`.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  let rgba_min = rgba_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
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
            unsafe { arch::neon::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::neon::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx512::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_avx2::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::x86_sse41::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range); },
            unsafe { arch::wasm_simd128::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range); }
          );
          return;
        }
      },
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, matrix, full_range),
    scalar::yuv_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, matrix, full_range)
  );
}

/// LE-only wrapper around [`yuv444p12_to_rgba_u16_row_endian`]; preserves the pre-endian-aware
/// public signature so existing little-endian callers compile unchanged.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv444p12_to_rgba_u16_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **12-bit** YUV 4:4:4 **directly** to planar
/// HSV bytes (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`), without materializing a source-width RGB row.
/// Byte-identical to `rgb_to_hsv_row(yuv444p12_to_rgb_row_endian
/// (...))` within the selected tier — the SIMD path stages a fixed
/// 64-pixel 8-bit RGB chunk internally. Also serves 4:4:0.
///
/// Thin endian-dispatching wrapper over the BITS-generic
/// [`super::yuv_444p_n_to_hsv_row`]. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_hsv_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
  big_endian: bool,
) {
  if big_endian {
    super::yuv_444p_n_to_hsv_row::<12, true>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  } else {
    super::yuv_444p_n_to_hsv_row::<12, false>(
      y, u, v, h_out, s_out, v_out, width, matrix, full_range, use_simd,
    );
  }
}

// ---- ICtCp (BT.2100, H.273 MatrixCoefficients = 14) routing -------------
//
// Transfer-aware siblings of the affine `*_endian` dispatchers, mirroring
// the `ChromaDerivedNcl` `*_primaries` pattern: when the matrix is
// `ColorMatrix::Ictcp` **and** the source carries a PQ/HLG transfer, the row
// decodes through the dedicated scalar non-affine kernel
// ([`scalar::ictcp`]); otherwise it delegates byte-identically to the affine
// `*_endian` dispatcher. `ICtCp` is scalar-only (the transcendental EOTF
// does not vectorize), so `use_simd` is honoured only on the affine
// fallback. Gated on the transcendental tier (`std`/`alloc`, via `libm`)
// `scalar::ictcp` itself requires; without it an `Ictcp` source falls back
// to the affine path.
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::Transfer;
#[cfg(any(feature = "std", feature = "alloc"))]
use scalar::ictcp::{self, IctcpTransfer};

/// [`yuv444p12_to_rgb_row_endian`] with the `ColorMatrix::Ictcp` non-affine
/// decode spliced in for PQ/HLG `transfer`. See the module routing note.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row_ictcp_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Ictcp)
    && let Some(tf) = IctcpTransfer::for_transfer(transfer)
  {
    if big_endian {
      ictcp::ictcp_444p_n_to_rgb_row::<12, true>(y, u, v, rgb_out, width, full_range, tf);
    } else {
      ictcp::ictcp_444p_n_to_rgb_row::<12, false>(y, u, v, rgb_out, width, full_range, tf);
    }
    return;
  }
  yuv444p12_to_rgb_row_endian(
    y, u, v, rgb_out, width, matrix, full_range, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_row_endian`] with the `ColorMatrix::Ictcp` non-affine
/// decode (opaque alpha) for PQ/HLG `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row_ictcp_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Ictcp)
    && let Some(tf) = IctcpTransfer::for_transfer(transfer)
  {
    if big_endian {
      ictcp::ictcp_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, full_range, tf);
    } else {
      ictcp::ictcp_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, full_range, tf);
    }
    return;
  }
  yuv444p12_to_rgba_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgb_u16_row_endian`] with the `ColorMatrix::Ictcp`
/// non-affine decode for PQ/HLG `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row_ictcp_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Ictcp)
    && let Some(tf) = IctcpTransfer::for_transfer(transfer)
  {
    if big_endian {
      ictcp::ictcp_444p_n_to_rgb_u16_row::<12, true>(y, u, v, rgb_out, width, full_range, tf);
    } else {
      ictcp::ictcp_444p_n_to_rgb_u16_row::<12, false>(y, u, v, rgb_out, width, full_range, tf);
    }
    return;
  }
  yuv444p12_to_rgb_u16_row_endian(
    y, u, v, rgb_out, width, matrix, full_range, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_u16_row_endian`] with the `ColorMatrix::Ictcp`
/// non-affine decode (opaque alpha `0xFFFF`) for PQ/HLG `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row_ictcp_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Ictcp)
    && let Some(tf) = IctcpTransfer::for_transfer(transfer)
  {
    if big_endian {
      ictcp::ictcp_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, full_range, tf);
    } else {
      ictcp::ictcp_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, full_range, tf);
    }
    return;
  }
  yuv444p12_to_rgba_u16_row_endian(
    y, u, v, rgba_out, width, matrix, full_range, use_simd, big_endian,
  );
}

// ---- ChromaDerivedCl (BT.2020 CL, H.273 MatrixCoefficients = 13) routing -
//
// Transfer-and-primaries-aware siblings of the affine `*_endian` dispatchers,
// the constant-luminance analogue of the ICtCp `*_ictcp_endian` splice: when
// the matrix is `ColorMatrix::ChromaDerivedCl` **and** the source carries
// BT.2020 primaries (the gamut CL is published for; the transfer selects the
// 10-/12-bit OETF constants), the row decodes through the dedicated scalar
// non-affine kernel ([`scalar::cl`]); otherwise it delegates byte-identically
// to the affine `*_endian` dispatcher (the prior BT.709 fallback for an
// unresolved chromaticity-derived matrix). CL is scalar-only (the
// transcendental OETF does not vectorize), so `use_simd` is honoured only on
// the affine fallback. Gated on the same transcendental tier (`std`/`alloc`)
// `scalar::cl` requires.
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::Primaries;
#[cfg(any(feature = "std", feature = "alloc"))]
use scalar::cl::{self, ClSystem};

/// [`yuv444p12_to_rgb_row_endian`] with the `ColorMatrix::ChromaDerivedCl`
/// constant-luminance decode spliced in for BT.2020 primaries. See the module
/// routing note.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row_chroma_derived_cl_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::ChromaDerivedCl)
    && let Some(system) = ClSystem::resolve(primaries, transfer)
  {
    if big_endian {
      cl::cl_444p_n_to_rgb_row::<12, true>(y, u, v, rgb_out, width, full_range, system);
    } else {
      cl::cl_444p_n_to_rgb_row::<12, false>(y, u, v, rgb_out, width, full_range, system);
    }
    return;
  }
  // Not a resolvable CL row: chain to the ICtCp dispatcher, which splices the
  // other non-affine decode (mutually exclusive matrix) and otherwise
  // delegates byte-identically to the affine `*_endian` path.
  yuv444p12_to_rgb_row_ictcp_endian(
    y, u, v, rgb_out, width, matrix, tag, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_row_endian`] with the `ColorMatrix::ChromaDerivedCl`
/// constant-luminance decode (opaque alpha) for BT.2020 primaries.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row_chroma_derived_cl_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::ChromaDerivedCl)
    && let Some(system) = ClSystem::resolve(primaries, transfer)
  {
    if big_endian {
      cl::cl_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, full_range, system);
    } else {
      cl::cl_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, full_range, system);
    }
    return;
  }
  yuv444p12_to_rgba_row_ictcp_endian(
    y, u, v, rgba_out, width, matrix, tag, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgb_u16_row_endian`] with the `ColorMatrix::ChromaDerivedCl`
/// constant-luminance decode for BT.2020 primaries.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row_chroma_derived_cl_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::ChromaDerivedCl)
    && let Some(system) = ClSystem::resolve(primaries, transfer)
  {
    if big_endian {
      cl::cl_444p_n_to_rgb_u16_row::<12, true>(y, u, v, rgb_out, width, full_range, system);
    } else {
      cl::cl_444p_n_to_rgb_u16_row::<12, false>(y, u, v, rgb_out, width, full_range, system);
    }
    return;
  }
  yuv444p12_to_rgb_u16_row_ictcp_endian(
    y, u, v, rgb_out, width, matrix, tag, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_u16_row_endian`] with the `ColorMatrix::ChromaDerivedCl`
/// constant-luminance decode (opaque alpha `(1 << 12) - 1`) for BT.2020
/// primaries.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row_chroma_derived_cl_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::ChromaDerivedCl)
    && let Some(system) = ClSystem::resolve(primaries, transfer)
  {
    if big_endian {
      cl::cl_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, full_range, system);
    } else {
      cl::cl_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, full_range, system);
    }
    return;
  }
  yuv444p12_to_rgba_u16_row_ictcp_endian(
    y, u, v, rgba_out, width, matrix, tag, full_range, transfer, use_simd, big_endian,
  );
}

// ---- IptC2 (H.273 MatrixCoefficients = 15) routing ----------------------
//
// The Dolby Vision Profile 5 base colour space, the non-affine sibling of the
// ICtCp / ChromaDerivedCl splices and the head of the non-affine dispatcher
// chain: when the matrix is `ColorMatrix::IptC2` **and** the source carries a
// PQ transfer (the only transfer IPT-C2 is defined for), the row decodes
// through the dedicated scalar non-affine kernel ([`scalar::iptc2`]);
// otherwise it delegates byte-identically to the `*_chroma_derived_cl_endian`
// dispatcher — the next link, which splices its own (mutually exclusive)
// matrix and finally the affine `*_endian` path. IPT-C2 is scalar-only (the
// transcendental PQ EOTF/OETF does not vectorize), so `use_simd` is honoured
// only on the delegated fallback. Gated on the same transcendental tier
// (`std`/`alloc`) `scalar::iptc2` requires; `primaries` is threaded through
// unused here so the delegated CL link can still resolve.
#[cfg(any(feature = "std", feature = "alloc"))]
use scalar::iptc2::{self, IptC2Transfer};

/// [`yuv444p12_to_rgb_row_chroma_derived_cl_endian`] with the
/// `ColorMatrix::IptC2` non-affine decode spliced in for a PQ `transfer`. See
/// the module routing note.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row_iptc2_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::IptC2) && IptC2Transfer::for_transfer(transfer).is_some() {
    if big_endian {
      iptc2::iptc2_444p_n_to_rgb_row::<12, true>(y, u, v, rgb_out, width, full_range);
    } else {
      iptc2::iptc2_444p_n_to_rgb_row::<12, false>(y, u, v, rgb_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgb_row_chroma_derived_cl_endian(
    y, u, v, rgb_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_row_chroma_derived_cl_endian`] with the
/// `ColorMatrix::IptC2` non-affine decode (opaque alpha) for a PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row_iptc2_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::IptC2) && IptC2Transfer::for_transfer(transfer).is_some() {
    if big_endian {
      iptc2::iptc2_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, full_range);
    } else {
      iptc2::iptc2_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgba_row_chroma_derived_cl_endian(
    y, u, v, rgba_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgb_u16_row_chroma_derived_cl_endian`] with the
/// `ColorMatrix::IptC2` non-affine decode for a PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row_iptc2_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::IptC2) && IptC2Transfer::for_transfer(transfer).is_some() {
    if big_endian {
      iptc2::iptc2_444p_n_to_rgb_u16_row::<12, true>(y, u, v, rgb_out, width, full_range);
    } else {
      iptc2::iptc2_444p_n_to_rgb_u16_row::<12, false>(y, u, v, rgb_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgb_u16_row_chroma_derived_cl_endian(
    y, u, v, rgb_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_u16_row_chroma_derived_cl_endian`] with the
/// `ColorMatrix::IptC2` non-affine decode (opaque alpha `(1 << 12) - 1`) for a
/// PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row_iptc2_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::IptC2) && IptC2Transfer::for_transfer(transfer).is_some() {
    if big_endian {
      iptc2::iptc2_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, full_range);
    } else {
      iptc2::iptc2_444p_n_to_rgba_u16_row::<12, false>(y, u, v, rgba_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgba_u16_row_chroma_derived_cl_endian(
    y, u, v, rgba_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

// ---- Smpte2085 (H.273 MatrixCoefficients = 11) routing ------------------
//
// The SMPTE ST 2085 "Y'D'zD'x" X'Y'Z' colour-difference model, the non-affine
// sibling of the IptC2 / ICtCp / ChromaDerivedCl splices and the new head of
// the non-affine dispatcher chain: when the matrix is `ColorMatrix::Smpte2085`
// **and** the source carries a PQ transfer (the only transfer SMPTE 2085 is
// defined for), the row decodes through the dedicated scalar non-affine kernel
// ([`scalar::smpte2085`]); otherwise it delegates byte-identically to the
// `*_iptc2_endian` dispatcher — the next link, which splices its own (mutually
// exclusive) matrix and finally the CL / ICtCp / affine path. SMPTE 2085 is
// scalar-only (the transcendental PQ EOTF/OETF does not vectorize), so
// `use_simd` is honoured only on the delegated fallback. Gated on the same
// transcendental tier (`std`/`alloc`) `scalar::smpte2085` requires; `primaries`
// is threaded through unused here so the delegated CL link can still resolve.
#[cfg(any(feature = "std", feature = "alloc"))]
use scalar::smpte2085::{self, Smpte2085Transfer};

/// [`yuv444p12_to_rgb_row_iptc2_endian`] with the `ColorMatrix::Smpte2085`
/// non-affine decode spliced in for a PQ `transfer`. See the module routing
/// note.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row_smpte2085_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Smpte2085) && Smpte2085Transfer::for_transfer(transfer).is_some() {
    let rgb_min = rgb_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
    if big_endian {
      smpte2085::smpte2085_444p_n_to_rgb_row::<12, true>(y, u, v, rgb_out, width, full_range);
    } else {
      smpte2085::smpte2085_444p_n_to_rgb_row::<12, false>(y, u, v, rgb_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgb_row_iptc2_endian(
    y, u, v, rgb_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_row_iptc2_endian`] with the `ColorMatrix::Smpte2085`
/// non-affine decode (opaque alpha) for a PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_row_smpte2085_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Smpte2085) && Smpte2085Transfer::for_transfer(transfer).is_some() {
    let rgba_min = rgba_row_bytes(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");
    if big_endian {
      smpte2085::smpte2085_444p_n_to_rgba_row::<12, true>(y, u, v, rgba_out, width, full_range);
    } else {
      smpte2085::smpte2085_444p_n_to_rgba_row::<12, false>(y, u, v, rgba_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgba_row_iptc2_endian(
    y, u, v, rgba_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgb_u16_row_iptc2_endian`] with the `ColorMatrix::Smpte2085`
/// non-affine decode for a PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row_smpte2085_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Smpte2085) && Smpte2085Transfer::for_transfer(transfer).is_some() {
    let rgb_min = rgb_row_elems(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
    if big_endian {
      smpte2085::smpte2085_444p_n_to_rgb_u16_row::<12, true>(y, u, v, rgb_out, width, full_range);
    } else {
      smpte2085::smpte2085_444p_n_to_rgb_u16_row::<12, false>(y, u, v, rgb_out, width, full_range);
    }
    return;
  }
  yuv444p12_to_rgb_u16_row_iptc2_endian(
    y, u, v, rgb_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}

/// [`yuv444p12_to_rgba_u16_row_iptc2_endian`] with the `ColorMatrix::Smpte2085`
/// non-affine decode (opaque alpha `(1 << 12) - 1`) for a PQ `transfer`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgba_u16_row_smpte2085_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  tag: &ColorMatrix,
  primaries: &Primaries,
  full_range: bool,
  transfer: &Transfer,
  use_simd: bool,
  big_endian: bool,
) {
  if matches!(tag, ColorMatrix::Smpte2085) && Smpte2085Transfer::for_transfer(transfer).is_some() {
    let rgba_min = rgba_row_elems(width);
    assert!(y.len() >= width, "y row too short");
    assert!(u.len() >= width, "u row too short");
    assert!(v.len() >= width, "v row too short");
    assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");
    if big_endian {
      smpte2085::smpte2085_444p_n_to_rgba_u16_row::<12, true>(y, u, v, rgba_out, width, full_range);
    } else {
      smpte2085::smpte2085_444p_n_to_rgba_u16_row::<12, false>(
        y, u, v, rgba_out, width, full_range,
      );
    }
    return;
  }
  yuv444p12_to_rgba_u16_row_iptc2_endian(
    y, u, v, rgba_out, width, matrix, tag, primaries, full_range, transfer, use_simd, big_endian,
  );
}
