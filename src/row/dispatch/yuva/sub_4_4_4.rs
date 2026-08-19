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

// ---- YUVA 4:4:4 RGBA dispatchers --------------------------------------
//
// Per-row dispatchers for Yuva444p (8-bit, Ship 8b‑6), Yuva444p9 /
// Yuva444p10 / Yuva444p12 / Yuva444p14 (BITS-generic Q15 i32 family),
// and Yuva444p16 (dedicated i64 16-bit family). Both the u8 and
// native-depth `u16` RGBA dispatchers route through per-arch SIMD
// wrappers, mirroring the non-alpha siblings.
//
// The high-bit dispatchers (`yuva444p9/10/12/14/16`) each expose an
// `_endian` entry point that threads a runtime `big_endian: bool`
// through every backend (scalar / NEON / SSE4.1 / AVX2 / AVX-512BW /
// wasm-simd128) via the kernels' `<BITS, BE>` (or `<BE>` for 16-bit)
// const-generic pair, including the alpha-source u16 load. The
// pre-existing LE-only public function is preserved as a one-line
// wrapper that forwards `big_endian = false`, matching the pattern
// established for non-alpha YUV high-bit dispatchers in commit
// `1c2df3d`.

/// Converts one row of **8-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. R / G / B are produced by the same Q15 i32 8-bit kernel
/// that backs [`yuv444p_to_rgba_row`]; the per-pixel alpha byte is
/// **sourced from `a`** (one byte per pixel, full-width) instead of
/// being constant `0xFF`.
///
/// `use_simd = false` forces the scalar reference path; otherwise
/// per-arch dispatch matches [`yuv444p_to_rgba_row`]'s pattern.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
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
  assert!(a.len() >= width, "a row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444_to_rgba_with_alpha_src_row(
              y, u, v, a, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar::yuv_444_to_rgba_with_alpha_src_row(y, u, v, a, rgba_out, width, matrix, full_range);
}

// ---- BITS-generic 9/10/12/14 helpers (mirror non-alpha pattern) ----

macro_rules! impl_yuva444p_n_endian_pair {
  (
    $bits:literal,
    $endian_u8:ident,
    $le_u8:ident,
    $endian_u16:ident,
    $le_u16:ident
  ) => {
    /// 4:4:4 YUVA high-bit (`BITS`) → packed u8 RGBA. Endian-aware
    /// variant: `big_endian = true` selects the BE-encoded `u16` plane
    /// contract for Y / U / V **and** the alpha source plane.
    #[cfg_attr(not(tarpaulin), inline(always))]
    #[allow(clippy::too_many_arguments)]
    pub fn $endian_u8(
      y: &[u16],
      u: &[u16],
      v: &[u16],
      a: &[u16],
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
      assert!(a.len() >= width, "a row too short");
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
                unsafe {
                  arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::neon::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          target_arch = "x86_64" => {
            if avx512_available() {
              // SAFETY: AVX‑512BW verified.
              dispatch_be!(
                unsafe {
                  arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_avx512::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
            if avx2_available() {
              // SAFETY: AVX2 verified.
              dispatch_be!(
                unsafe {
                  arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_avx2::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
            if sse41_available() {
              // SAFETY: SSE4.1 verified.
              dispatch_be!(
                unsafe {
                  arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_sse41::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          target_arch = "wasm32" => {
            if simd128_available() {
              // SAFETY: simd128 compile‑time verified.
              dispatch_be!(
                unsafe {
                  arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::wasm_simd128::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          _ => {}
        }
      }

      dispatch_be!(
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, false>(
          y, u, v, a, rgba_out, width, matrix, full_range
        ),
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<$bits, true>(
          y, u, v, a, rgba_out, width, matrix, full_range
        )
      );
    }

    /// LE-only wrapper preserving the pre-endian-aware public signature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    #[allow(clippy::too_many_arguments)]
    pub fn $le_u8(
      y: &[u16],
      u: &[u16],
      v: &[u16],
      a: &[u16],
      rgba_out: &mut [u8],
      width: usize,
      matrix: KernelMatrix,
      full_range: bool,
      use_simd: bool,
    ) {
      $endian_u8(
        y, u, v, a, rgba_out, width, matrix, full_range, use_simd, false,
      );
    }

    /// 4:4:4 YUVA high-bit (`BITS`) → native-depth u16 RGBA.
    /// Endian-aware variant.
    #[cfg_attr(not(tarpaulin), inline(always))]
    #[allow(clippy::too_many_arguments)]
    pub fn $endian_u16(
      y: &[u16],
      u: &[u16],
      v: &[u16],
      a: &[u16],
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
      assert!(a.len() >= width, "a row too short");
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
                unsafe {
                  arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::neon::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          target_arch = "x86_64" => {
            if avx512_available() {
              // SAFETY: AVX‑512BW verified.
              dispatch_be!(
                unsafe {
                  arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_avx512::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
            if avx2_available() {
              // SAFETY: AVX2 verified.
              dispatch_be!(
                unsafe {
                  arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_avx2::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
            if sse41_available() {
              // SAFETY: SSE4.1 verified.
              dispatch_be!(
                unsafe {
                  arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::x86_sse41::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          target_arch = "wasm32" => {
            if simd128_available() {
              // SAFETY: simd128 compile‑time verified.
              dispatch_be!(
                unsafe {
                  arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                },
                unsafe {
                  arch::wasm_simd128::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
                    y, u, v, a, rgba_out, width, matrix, full_range,
                  );
                }
              );
              return;
            }
          }
          _ => {}
        }
      }

      dispatch_be!(
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, false>(
          y, u, v, a, rgba_out, width, matrix, full_range
        ),
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<$bits, true>(
          y, u, v, a, rgba_out, width, matrix, full_range
        )
      );
    }

    /// LE-only wrapper preserving the pre-endian-aware public signature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    #[allow(clippy::too_many_arguments)]
    pub fn $le_u16(
      y: &[u16],
      u: &[u16],
      v: &[u16],
      a: &[u16],
      rgba_out: &mut [u16],
      width: usize,
      matrix: KernelMatrix,
      full_range: bool,
      use_simd: bool,
    ) {
      $endian_u16(
        y, u, v, a, rgba_out, width, matrix, full_range, use_simd, false,
      );
    }
  };
}

impl_yuva444p_n_endian_pair!(
  9,
  yuva444p9_to_rgba_row_endian,
  yuva444p9_to_rgba_row,
  yuva444p9_to_rgba_u16_row_endian,
  yuva444p9_to_rgba_u16_row
);
impl_yuva444p_n_endian_pair!(
  10,
  yuva444p10_to_rgba_row_endian,
  yuva444p10_to_rgba_row,
  yuva444p10_to_rgba_u16_row_endian,
  yuva444p10_to_rgba_u16_row
);
impl_yuva444p_n_endian_pair!(
  12,
  yuva444p12_to_rgba_row_endian,
  yuva444p12_to_rgba_row,
  yuva444p12_to_rgba_u16_row_endian,
  yuva444p12_to_rgba_u16_row
);
impl_yuva444p_n_endian_pair!(
  14,
  yuva444p14_to_rgba_row_endian,
  yuva444p14_to_rgba_row,
  yuva444p14_to_rgba_u16_row_endian,
  yuva444p14_to_rgba_u16_row
);

// ---- YUVA 4:4:4 16-bit RGBA dispatchers (Ship 8b-5a/b/c) -------------
//
// Yuva444p16 uses dedicated 16-bit kernels rather than the
// BITS-generic Q15 i32 template (which only covers {9,10,12,14}). The
// 8-bit RGBA path uses the i32 chroma pipeline (output-target scaling
// keeps `coeff × u_d` inside i32); the native-depth `u16` RGBA path
// uses the widened i64 chroma kernel family.

/// Converts one row of **16-bit** YUVA 4:4:4 to packed **8-bit**
/// **RGBA**. Endian-aware variant: `big_endian = true` selects the
/// BE-encoded `u16` plane contract for Y / U / V **and** the alpha
/// source plane.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
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
  assert!(a.len() >= width, "a row too short");
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
            unsafe {
              arch::neon::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
      y, u, v, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_444p16_to_rgba_with_alpha_src_row::<true>(
      y, u, v, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva444p16_to_rgba_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva444p16_to_rgba_row_endian(
    y, u, v, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}

/// Converts one row of **16-bit** YUVA 4:4:4 to **native-depth `u16`**
/// packed **RGBA**. Endian-aware variant. Full-range output in
/// `[0, 65535]`. Uses the i64 chroma kernel family.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_u16_row_endian(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
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
  assert!(a.len() >= width, "a row too short");
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
            unsafe {
              arch::neon::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::neon::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx512::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx512::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          dispatch_be!(
            unsafe {
              arch::x86_avx2::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_avx2::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          dispatch_be!(
            unsafe {
              arch::x86_sse41::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::x86_sse41::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          dispatch_be!(
            unsafe {
              arch::wasm_simd128::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            },
            unsafe {
              arch::wasm_simd128::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
                y, u, v, a, rgba_out, width, matrix, full_range,
              );
            }
          );
          return;
        }
      }
      _ => {}
    }
  }

  dispatch_be!(
    scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row::<false>(
      y, u, v, a, rgba_out, width, matrix, full_range
    ),
    scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row::<true>(
      y, u, v, a, rgba_out, width, matrix, full_range
    )
  );
}

/// LE-only wrapper around [`yuva444p16_to_rgba_u16_row_endian`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuva444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuva444p16_to_rgba_u16_row_endian(
    y, u, v, a, rgba_out, width, matrix, full_range, use_simd, false,
  );
}
