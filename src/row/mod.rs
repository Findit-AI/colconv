//! Crate-internal row-level primitives.
//!
//! These are the composable units that Sinks call on each row handed
//! to them by a source kernel. Source kernels are pure row walkers;
//! the actual arithmetic lives here.
//!
//! Backends (all crate‑private modules):
//! - `scalar` — always compiled, reference implementation.
//! - `arch::neon` — aarch64 NEON.
//! - `arch::x86_sse41`, `arch::x86_avx2`, `arch::x86_avx512` — x86_64
//!   tiers.
//! - `arch::wasm_simd128` — wasm32 simd128.
//!
//! Each is gated on the appropriate `target_arch` / `target_feature`
//! cfg.
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
//! `scalar` or differs by at most 1 LSB per channel (documented per
//! backend). Tests in `arch` enforce this contract.
//!
//! Dispatcher `cfg_select!` requires Rust 1.95+ (stable, in the core
//! prelude — no import needed). The crate's MSRV matches.

pub(crate) mod arch;
pub(crate) mod scalar;

use crate::ColorMatrix;

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
  matrix: ColorMatrix,
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

/// Converts one row of NV12 (semi‑planar 4:2:0) to packed RGB.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the only
/// difference is UV source — NV12 delivers U and V interleaved in a
/// single `width`‑byte row (`U0, V0, U1, V1, …`). See
/// `scalar::nv12_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  assert_eq!(width & 1, 0, "NV12 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (checked above).
          unsafe {
            arch::neon::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: `avx2_available()` verified AVX2 is present.
          unsafe {
            arch::x86_avx2::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: `sse41_available()` verified SSE4.1 is present.
          unsafe {
            arch::x86_sse41::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: `simd128_available()` verified simd128 is on at
          // compile time (WASM has no runtime CPU detection).
          unsafe {
            arch::wasm_simd128::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv12_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV21 (semi‑planar 4:2:0, VU-ordered) to
/// packed RGB.
///
/// Same numerical contract as [`nv12_to_rgb_row`]; the only
/// difference is chroma byte order — NV21 stores `V0, U0, V1, U1, …`
/// instead of NV12's `U0, V0, U1, V1, …`. See `scalar::nv21_to_rgb_row`
/// for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary.
  assert_eq!(width & 1, 0, "NV21 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(vu_half.len() >= width, "vu_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: `avx512_available()` verified AVX‑512BW is present.
          unsafe {
            arch::x86_avx512::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv21_to_rgb_row(y, vu_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV24 (semi‑planar 4:4:4, UV‑ordered) to packed
/// RGB. Dispatches to the best available SIMD backend for the current
/// target (NEON / SSE4.1 / AVX2 / AVX-512 / wasm simd128), falling
/// back to scalar when no backend is available.
///
/// Same numerical contract as [`yuv_420_to_rgb_row`]; the difference
/// from NV12 is 4:4:4 chroma — one UV pair per Y pixel, no chroma
/// upsampling, and no width parity constraint. See
/// `scalar::nv24_to_rgb_row` for the reference implementation.
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend. Benchmarks can flip this to compare scalar vs SIMD
/// directly on the same input; production code should pass `true`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  // NV24 chroma carries one UV pair per pixel = `2 * width` bytes.
  // Use `checked_mul` — on 32-bit targets, `2 * width` can overflow
  // `usize` at extreme widths and silently short-circuit the length
  // check before entering unsafe SIMD paths.
  let uv_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(uv.len() >= uv_min, "uv row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv24_to_rgb_row(y, uv, rgb_out, width, matrix, full_range);
}

/// Converts one row of NV42 (semi‑planar 4:4:4, VU‑ordered) to packed
/// RGB. Same as [`nv24_to_rgb_row`] but with swapped chroma byte order.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  let vu_min = match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize"),
  };
  assert!(y.len() >= width, "y row too short");
  assert!(vu.len() >= vu_min, "vu row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified at compile time.
          unsafe {
            arch::wasm_simd128::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD backend fall through to scalar.
      }
    }
  }

  scalar::nv42_to_rgb_row(y, vu, rgb_out, width, matrix, full_range);
}

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
  matrix: ColorMatrix,
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

/// YUV 4:4:4 planar 10/12/14-bit → **u8** RGB dispatcher. Const
/// generic over `BITS ∈ {10, 12, 14}`. Dispatches to the best
/// available backend for the current target (NEON / SSE4.1 / AVX2 /
/// AVX-512 / wasm simd128), falling back to scalar when no SIMD
/// backend is available or `use_simd` is false.
///
/// Crate-private — external callers use the concrete
/// [`yuv444p10_to_rgb_row`] / [`yuv444p12_to_rgb_row`] /
/// [`yuv444p14_to_rgb_row`] wrappers, which pin `BITS` to a
/// supported value. This avoids the 16-bit footgun (`(1 << 16) - 1`
/// truncates to `-1` when cast to `i16` in the SIMD clamp), and
/// matches the [`yuv420p10_to_rgb_row`] family's convention of
/// keeping the `<BITS>` generic internal.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgb_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar 10/12/14-bit → **native-depth u16** RGB dispatcher.
/// Const generic over `BITS ∈ {10, 12, 14}`. Low-bit-packed output.
/// Dispatches to the best available backend (NEON / SSE4.1 / AVX2 /
/// AVX-512 / wasm simd128), falling back to scalar when no SIMD
/// backend is available or `use_simd` is false.
///
/// Crate-private — see the note on [`yuv_444p_n_to_rgb_row`]. The
/// 16-bit path is [`yuv444p16_to_rgb_u16_row`], which uses a
/// dedicated i64-chroma kernel family.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar 9-bit → u8 RGB. Thin wrapper over the
/// crate-internal `yuv_444p_n_to_rgb_row::<9>`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_row::<9>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 9-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p9_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_u16_row::<9>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 10-bit → u8 RGB. Thin wrapper over the
/// crate-internal `yuv_444p_n_to_rgb_row::<10>`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_row::<10>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 10-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p10_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_u16_row::<10>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 12-bit → u8 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_row::<12>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 12-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p12_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_u16_row::<12>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 14-bit → u8 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p14_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_row::<14>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar 14-bit → native-depth u16 RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p14_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  yuv_444p_n_to_rgb_u16_row::<14>(y, u, v, rgb_out, width, matrix, full_range, use_simd);
}

/// YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Uses the
/// parallel 16-bit kernel family (same Q15 i32 output-range pipeline
/// as [`yuv_420p16_to_rgb_row`] but with 1:1 chroma per pixel).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgb_row(y, u, v, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar **16-bit** → packed **u16** RGB (full-range
/// output in `[0, 65535]`). Widens chroma multiply-add + Y scale to
/// i64 to avoid i32 overflow at 16-bit limited range.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u.len() >= width, "u row too short");
  assert!(v.len() >= width, "v row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified. Native 512-bit i64-chroma kernel.
          unsafe {
            arch::x86_avx512::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_444p16_to_rgb_u16_row(y, u, v, rgb_out, width, matrix, full_range);
}

/// Converts one row of **9‑bit** YUV 4:2:0 to packed **8‑bit** RGB.
///
/// Samples are `u16` with 9 active bits in the low bits of each
/// element. Niche format (AVC High 9 profile only). Reuses the same
/// `yuv_420p_n_to_rgb_row<BITS>` kernel family as 10/12/14-bit; the
/// only per-call difference is the const-generic `BITS = 9` which
/// fixes the AND-mask to `0x1FF` and the Q15 scale via
/// `range_params_n::<9, 8>`.
///
/// See `scalar::yuv_420p_n_to_rgb_row` for the full semantic
/// specification. `use_simd = false` forces the scalar reference
/// path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p9_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_row::<9>(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_row::<9>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **9‑bit** YUV 4:2:0 to **native‑depth** packed
/// `u16` RGB (9-bit values in the **low** 9 bits of each `u16`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p9_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_u16_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_u16_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_u16_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_u16_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_u16_row::<9>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_u16_row::<9>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **10‑bit** YUV 4:2:0 to packed **8‑bit** RGB.
///
/// Samples are `u16` with 10 active bits in the low bits of each
/// element. Output is packed `R, G, B` bytes (`3 * width` bytes),
/// with the conversion clamping to `[0, 255]` — the native‑depth
/// path is [`yuv420p10_to_rgb_u16_row`].
///
/// See `scalar::yuv_420p_n_to_rgb_row` for the full semantic
/// specification. `use_simd = false` forces the scalar reference
/// path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p10_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
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
          // SAFETY: NEON verified on this CPU; bounds / parity are
          // the caller's obligation (asserted above).
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_row::<10>(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_row::<10>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **10‑bit** YUV 4:2:0 to **native‑depth** packed
/// RGB `u16` (10‑bit values in the **low** 10 bits of each `u16`,
/// matching FFmpeg's `yuv420p10le` convention). Use this for lossless
/// downstream HDR processing when the consumer expects low‑bit‑packed
/// samples.
///
/// Output is packed `R, G, B` triples: `rgb_out[3 * width]` `u16`
/// elements, each in `[0, 1023]` with the upper 6 bits zero.
///
/// This is **not** the FFmpeg `p010` layout — `p010` stores samples
/// in the **high** 10 bits of each `u16` (`sample << 6`). Callers
/// feeding this output into a p010 consumer must shift left by 6
/// before handing off.
///
/// See `scalar::yuv_420p_n_to_rgb_u16_row` for the full semantic
/// specification. `use_simd = false` forces the scalar reference
/// path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p10_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_u16_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_u16_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_u16_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_u16_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_u16_row::<10>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_u16_row::<10>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P010** (semi‑planar 4:2:0, 10‑bit, high‑bit‑
/// packed — 10 active bits in the high 10 of each `u16`) to packed
/// **8‑bit** RGB.
///
/// This is the HDR hardware‑decode keystone format: VideoToolbox,
/// VA‑API, NVDEC, D3D11VA, and Intel QSV all emit P010 for 10‑bit
/// output. See `scalar::p_n_to_rgb_row::<10>` for the full semantic
/// specification. `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P010** to **native‑depth `u16`** packed RGB
/// (10 active bits in the **low** 10 of each output `u16`, matching
/// `yuv420p10le` convention — **not** the P010 high‑bit packing).
/// Callers feeding this output into a P010 consumer must shift left
/// by 6.
///
/// See `scalar::p_n_to_rgb_u16_row::<10>` for the full spec.
/// `use_simd = false` forces the scalar reference.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p010_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P010 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_u16_row::<10>(
              y, uv_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_u16_row::<10>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **12‑bit** YUV 4:2:0 to packed **8‑bit** RGB.
///
/// Samples are `u16` with 12 active bits in the low 12 bits of each
/// element (low‑bit‑packed `yuv420p12le` convention). Output is packed
/// `R, G, B` bytes (`3 * width` bytes), clamping to `[0, 255]`. The
/// native‑depth path is [`yuv420p12_to_rgb_u16_row`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p12_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
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
          // SAFETY: NEON verified.
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_row::<12>(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_row::<12>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **12‑bit** YUV 4:2:0 to **native‑depth** packed
/// `u16` RGB (12‑bit values in the **low** 12 of each `u16`, matching
/// `yuv420p12le` convention — upper 4 bits zero).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p12_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_u16_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_u16_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_u16_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_u16_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_u16_row::<12>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_u16_row::<12>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **14‑bit** YUV 4:2:0 to packed **8‑bit** RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
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
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **14‑bit** YUV 4:2:0 to **native‑depth** packed
/// `u16` RGB (14‑bit values in the low 14 of each `u16`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p14_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p_n_to_rgb_u16_row::<14>(
              y, u_half, v_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p_n_to_rgb_u16_row::<14>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P012** (semi‑planar 4:2:0, 12‑bit, high‑bit‑
/// packed — 12 active bits in the high 12 of each `u16`) to packed
/// **8‑bit** RGB.
///
/// P012 is the 12‑bit sibling of P010, emitted by HEVC Main 12 and
/// VP9 Profile 3 hardware decoders. Same shift semantics as P010 but
/// `>> 4` instead of `>> 6` at each `u16` load.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P012 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P012** to **native‑depth `u16`** packed RGB
/// (12 active bits in the low 12 of each output `u16` — low‑bit‑packed
/// `yuv420p12le` convention, **not** P012's high‑bit packing).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p012_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P012 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p_n_to_rgb_u16_row::<12>(
              y, uv_half, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p_n_to_rgb_u16_row::<12>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **16-bit** YUV 4:2:0 to packed **8-bit** RGB.
///
/// Samples are `u16` over the full 16-bit range (`[0, 65535]`). Runs
/// on the **i64 chroma** kernel family; see
/// [`scalar::yuv_420p16_to_rgb_row`] for the numerical contract.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
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
          unsafe {
            arch::neon::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p16_to_rgb_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **16-bit** YUV 4:2:0 to **native-depth**
/// packed `u16` RGB (full-range output in `[0, 65535]`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_half.len() >= width / 2, "u_half row too short");
  assert!(v_half.len() >= width / 2, "v_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_420p16_to_rgb_u16_row(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P016** (semi-planar 4:2:0, 16-bit) to
/// packed **8-bit** RGB. At 16 bits there is no high-bit-packed
/// vs. low-bit-packed distinction (all bits are active).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p016_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P016 requires even width");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p16_to_rgb_row(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of **P016** to **native-depth `u16`** packed RGB
/// (full-range output in `[0, 65535]`).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn p016_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 1, 0, "P016 requires even width");
  let rgb_min = rgb_row_elems(width);
  assert!(y.len() >= width, "y row too short");
  assert!(uv_half.len() >= width, "uv_half row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          unsafe {
            arch::neon::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          unsafe {
            arch::x86_avx512::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          unsafe {
            arch::x86_avx2::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          unsafe {
            arch::x86_sse41::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          unsafe {
            arch::wasm_simd128::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::p16_to_rgb_u16_row(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of packed RGB to planar HSV (OpenCV 8‑bit
/// encoding). See `scalar::rgb_to_hsv_row` for semantics.
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
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb.len() >= rgb_min, "rgb row too short");
  assert!(h_out.len() >= width, "h_out row too short");
  assert!(s_out.len() >= width, "s_out row too short");
  assert!(v_out.len() >= width, "v_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present.
          unsafe {
            arch::neon::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX‑512BW verified.
          unsafe {
            arch::x86_avx512::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe {
            arch::x86_avx2::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe {
            arch::x86_sse41::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile‑time verified.
          unsafe {
            arch::wasm_simd128::rgb_to_hsv_row(rgb, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      _ => {
        // Targets without a SIMD HSV backend fall through to scalar.
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
  // Runtime asserts at the dispatcher boundary (see
  // [`yuv_420_to_rgb_row`] for rationale, including the checked
  // `width × 3` multiplication).
  let rgb_min = rgb_row_bytes(width);
  assert!(input.len() >= rgb_min, "input row too short");
  assert!(output.len() >= rgb_min, "output row too short");

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

// ---- shared dispatcher helpers ---------------------------------------

/// Computes the byte length of one packed‑RGB row with overflow
/// checking. Panics if `width × 3` cannot be represented as `usize`
/// (only reachable on 32‑bit targets — wasm32, i686 — with extreme
/// widths). Callers use the returned length as the minimum buffer
/// size they hand to unsafe SIMD kernels, so an unchecked
/// multiplication here could admit an undersized buffer and trigger
/// out‑of‑bounds writes downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb_row_bytes(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) × 3 overflows usize"),
  }
}

/// Element count of one packed `u16`‑RGB row (`width × 3`). Identical
/// math to [`rgb_row_bytes`] — the returned value is in `u16`
/// elements, not bytes. Callers use it to size `&mut [u16]` buffers
/// for the `u16` output path. `width × 3` overflow still matters on
/// 32‑bit targets: the product names the number of elements the
/// caller allocates, and downstream SIMD kernels index with it
/// directly without re‑multiplying.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rgb_row_elems(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) × 3 overflows usize"),
  }
}

// ---- runtime CPU feature detection -----------------------------------
//
// Each `*_available` helper returns `true` iff the named feature is
// present. `feature = "std"` branches use std's cached
// `is_*_feature_detected!` macros (atomic load + branch after the
// first call). No‑std branches fall back to `cfg!(target_feature = ...)`
// which is resolved at compile time. Helpers are only compiled for
// targets where the corresponding feature exists.

// The `colconv_force_scalar` cfg, when set, short‑circuits every
// `*_available()` helper to `false` so the dispatcher always falls
// through to the scalar reference path. CI uses this via
// `RUSTFLAGS='--cfg colconv_force_scalar'` to benchmark / measure
// coverage of the scalar baseline. `colconv_disable_avx512` /
// `colconv_disable_avx2` similarly force lower‑tier x86 paths for
// per‑tier coverage on runners that would otherwise always pick
// AVX‑512.

/// NEON availability on aarch64.
#[cfg(all(target_arch = "aarch64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn neon_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// NEON availability on aarch64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn neon_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "neon")
}

/// AVX2 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx2_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx2) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx2")
}

/// AVX2 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn avx2_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx2) && cfg!(target_feature = "avx2")
}

/// SSE4.1 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn sse41_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("sse4.1")
}

/// SSE4.1 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn sse41_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "sse4.1")
}

/// AVX‑512 (F + BW) availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn avx512_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx512) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx512bw")
}

/// AVX‑512 (F + BW) availability on x86_64 — no‑std variant
/// (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn avx512_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx512) && cfg!(target_feature = "avx512bw")
}

/// simd128 availability on wasm32. WASM has no runtime CPU detection
/// (SIMD support is fixed at module produce time), so this is always
/// a compile‑time check regardless of the `std` feature.
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn simd128_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "simd128")
}

#[cfg(all(test, feature = "std"))]
mod overflow_tests {
  //! 32-bit RGB-row-bytes overflow regressions for the public
  //! dispatchers. `width × 3` can wrap `usize` on wasm32 / i686 for
  //! extreme widths; the shared [`rgb_row_bytes`] helper rejects
  //! these before any unsafe kernel sees them. Tests are gated on
  //! 32-bit because `u32 × 3` never wraps 64-bit `usize`.

  #[cfg(target_pointer_width = "32")]
  use super::*;
  #[cfg(target_pointer_width = "32")]
  use crate::ColorMatrix;

  /// The smallest even width greater than `usize::MAX / 3`, so
  /// `width * 3` overflows 32-bit `usize` without tripping the
  /// dispatchers' even-width precondition first. `(usize::MAX / 3)`
  /// is always odd on 32-bit (`(2^32 - 1) / 3 == 1431655765`), so
  /// `+ 1` produces an even number — the `+ (candidate & 1)` keeps
  /// this correct on hypothetical platforms where the parity
  /// differs.
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH: usize = {
    let candidate = (usize::MAX / 3) + 1;
    candidate + (candidate & 1)
  };

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_420_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let u: [u8; 0] = [];
    let v: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuv_420_to_rgb_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_444_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let u: [u8; 0] = [];
    let v: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuv_444_to_rgb_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn nv12_dispatcher_rejects_width_times_3_overflow() {
    let y: [u8; 0] = [];
    let uv: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    nv12_to_rgb_row(
      &y,
      &uv,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb_to_hsv_dispatcher_rejects_width_times_3_overflow() {
    let rgb: [u8; 0] = [];
    let mut h: [u8; 0] = [];
    let mut s: [u8; 0] = [];
    let mut v: [u8; 0] = [];
    rgb_to_hsv_row(&rgb, &mut h, &mut s, &mut v, OVERFLOW_WIDTH, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn bgr_to_rgb_dispatcher_rejects_width_times_3_overflow() {
    let input: [u8; 0] = [];
    let mut output: [u8; 0] = [];
    bgr_to_rgb_row(&input, &mut output, OVERFLOW_WIDTH, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv_444p_n_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let u: [u16; 0] = [];
    let v: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    yuv_444p_n_to_rgb_u16_row::<10>(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuv444p16_u16_dispatcher_rejects_width_times_3_overflow() {
    let y: [u16; 0] = [];
    let u: [u16; 0] = [];
    let v: [u16; 0] = [];
    let mut rgb: [u16; 0] = [];
    yuv444p16_to_rgb_u16_row(
      &y,
      &u,
      &v,
      &mut rgb,
      OVERFLOW_WIDTH,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }
}
