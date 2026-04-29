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
//!
//! # Submodule layout
//!
//! Public dispatchers are split across `dispatch::*` submodules by
//! source format family for readability — `yuv420` / `yuv444` / `nv` /
//! `pn` / `yuva` / `rgb_ops` / `bayer`. They are re-exported as
//! `pub use dispatch::*::*` here so the public API stays at
//! `crate::row::*` (e.g. `crate::row::yuv_420_to_rgb_row`). Callers
//! see no API change from the split.

pub(crate) mod arch;
mod dispatch;
pub(crate) mod scalar;

// Re-exported only when a caller is compiled. The `MixedSinker` Strategy A
// fan-out is the sole consumer, and it lives in `crate::sinker::mixed` which
// is gated on `feature = "std"` / `feature = "alloc"` (needs `Vec`). Without
// either feature both this re-export and the underlying scalar function would
// be unused, which is a hard error under `cargo clippy -- -D warnings`.
#[cfg(any(feature = "std", feature = "alloc"))]
pub(crate) use scalar::expand_rgb_to_rgba_row;
#[cfg(any(feature = "std", feature = "alloc"))]
pub(crate) use scalar::expand_rgb_u16_to_rgba_u16_row;

pub use dispatch::{
  bayer::*, nv::*, packed_yuv422::*, pn::*, rgb_ops::*, yuv420::*, yuv444::*, yuva::*,
};

// `yuv_444p_n_to_rgb_u16_row` is consumed by the 32-bit overflow test
// `yuv_444p_n_u16_dispatcher_rejects_width_times_3_overflow` below —
// the dispatch submodule keeps it as `pub(crate)`, so glob `pub use`
// doesn't pick it up. Gated on the same cfg the test uses to avoid
// `unused_imports` on builds that don't compile the test.
#[cfg(all(test, feature = "std", target_pointer_width = "32"))]
pub(crate) use dispatch::yuv444::yuv_444p_n_to_rgb_u16_row;

// ---- shared dispatcher helpers ---------------------------------------

/// Computes the byte length of one packed‑RGB row with overflow
/// checking. Panics if `width × 3` cannot be represented as `usize`
/// (only reachable on 32‑bit targets — wasm32, i686 — with extreme
/// widths). Callers use the returned length as the minimum buffer
/// size they hand to unsafe SIMD kernels, so an unchecked
/// multiplication here could admit an undersized buffer and trigger
/// out‑of‑bounds writes downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb_row_bytes(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) × 3 overflows usize"),
  }
}

/// Byte length of one packed‑RGBA row (`width × 4`) with overflow
/// checking. Same purpose as [`rgb_row_bytes`] for the 4-channel
/// path used by the RGBA dispatchers.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba_row_bytes(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize"),
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
pub(crate) fn rgb_row_elems(width: usize) -> usize {
  match width.checked_mul(3) {
    Some(n) => n,
    None => panic!("width ({width}) × 3 overflows usize"),
  }
}

/// Element count of one packed `u16`-RGBA row (`width × 4`). Identical
/// math to [`rgba_row_bytes`] — the returned value is in `u16`
/// elements, not bytes. Callers use it to size `&mut [u16]` buffers
/// for the high-bit-depth `u16` RGBA output path.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba_row_elems(width: usize) -> usize {
  match width.checked_mul(4) {
    Some(n) => n,
    None => panic!("width ({width}) × 4 overflows usize"),
  }
}

/// Maximum permitted magnitude of any element of a fused color
/// transform handed to a Bayer row dispatcher.
///
/// Set to `WhiteBalance::MAX_GAIN × ColorCorrectionMatrix::MAX_COEFFICIENT_ABS
/// = 1e6 × 1e6 = 1e12`, which is the largest absolute value any
/// fused entry can take when the upstream WB / CCM were
/// validated through [`crate::raw::WhiteBalance::try_new`] /
/// [`crate::raw::ColorCorrectionMatrix::try_new`]. The overflow
/// analysis (see those constructor docs) shows that with `|m[i][j]|
/// ≤ 1e12` and 16-bit samples, the largest per-channel sum stays
/// `~21` orders of magnitude under `f32::MAX`. So bounding here
/// at 1e12 closes the door on direct-row-API callers passing
/// extreme finite matrices that would silently overflow during
/// the matmul.
pub(crate) const MAX_FUSED_TRANSFORM_ABS: f32 = 1.0e12;

/// Asserts every element of a 3×3 fused color transform is
/// **finite and within the magnitude bound**
/// ([`MAX_FUSED_TRANSFORM_ABS`]).
///
/// Used by the Bayer row dispatchers in release builds before
/// invoking the kernel — once SIMD backends land they will rely on
/// this guarantee for branchless f32 arithmetic. A single Inf or
/// NaN would otherwise propagate through every pixel of the row
/// (Inf clamps to saturated white, NaN casts to 0, both producing
/// silently-wrong frames); finite-but-extreme entries (e.g. mixed
/// `±f32::MAX` from a direct row-API caller) likewise produce
/// `Inf + -Inf == NaN` during the matmul.
///
/// Validating WB / CCM upstream via
/// [`crate::raw::WhiteBalance::try_new`] /
/// [`crate::raw::ColorCorrectionMatrix::try_new`] catches the
/// common case; this is the kernel-boundary backstop for direct
/// row-API callers and the dispatcher-level guarantee that
/// matches what validated upstream inputs can produce.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn assert_color_transform_well_formed(m: &[[f32; 3]; 3]) {
  let mut row = 0;
  while row < 3 {
    let mut col = 0;
    while col < 3 {
      let v = m[row][col];
      assert!(
        v.is_finite(),
        "color transform m[{row}][{col}] is non-finite (NaN or ±∞)"
      );
      assert!(
        v.abs() <= MAX_FUSED_TRANSFORM_ABS,
        "color transform m[{row}][{col}] = {v} exceeds magnitude bound \
         (|coeff| ≤ {MAX_FUSED_TRANSFORM_ABS}); validated WB × CCM cannot \
         produce values past this bound"
      );
      col += 1;
    }
    row += 1;
  }
}

/// Element count of one full-width interleaved-UV row (`width × 2`)
/// for semi-planar 4:4:4 sources (`P410` / `P412` / `P416`). One
/// `(U, V)` pair per pixel = `2 * width` `u16` elements per row.
/// Same `checked_mul` rationale as [`rgb_row_bytes`] — the returned
/// length feeds into unsafe SIMD kernels' bounds via the dispatcher's
/// `assert!`, so an unchecked multiplication on 32-bit targets could
/// silently admit an undersized buffer.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uv_full_row_elems(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize (UV row)"),
  }
}

/// Byte length of one packed YUV 4:2:2 row (`width × 2`) for
/// `Yuyv422` / `Uyvy422` / `Yvyu422` sources. Two bytes per pixel
/// (one `Y` + one half of an interleaved `U`/`V` pair). Same
/// `checked_mul` rationale as [`rgb_row_bytes`] — the returned byte
/// count feeds into the packed dispatchers' input-side `assert!`,
/// which gates entry into unsafe SIMD loads. An unchecked
/// multiplication on 32-bit targets could silently admit an
/// undersized `packed` slice.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn packed_yuv422_row_bytes(width: usize) -> usize {
  match width.checked_mul(2) {
    Some(n) => n,
    None => panic!("width ({width}) × 2 overflows usize (packed YUV 4:2:2 row)"),
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
pub(crate) fn neon_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// NEON availability on aarch64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn neon_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "neon")
}

/// AVX2 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx2_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx2) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx2")
}

/// AVX2 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn avx2_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx2) && cfg!(target_feature = "avx2")
}

/// SSE4.1 availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn sse41_available() -> bool {
  if cfg!(colconv_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("sse4.1")
}

/// SSE4.1 availability on x86_64 — no‑std variant (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn sse41_available() -> bool {
  !cfg!(colconv_force_scalar) && cfg!(target_feature = "sse4.1")
}

/// AVX‑512 (F + BW) availability on x86_64.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx512_available() -> bool {
  if cfg!(colconv_force_scalar) || cfg!(colconv_disable_avx512) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx512bw")
}

/// AVX‑512 (F + BW) availability on x86_64 — no‑std variant
/// (compile‑time).
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn avx512_available() -> bool {
  !cfg!(colconv_force_scalar) && !cfg!(colconv_disable_avx512) && cfg!(target_feature = "avx512bw")
}

/// simd128 availability on wasm32. WASM has no runtime CPU detection
/// (SIMD support is fixed at module produce time), so this is always
/// a compile‑time check regardless of the `std` feature.
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn simd128_available() -> bool {
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

  // ---- Packed YUV 4:2:2 dispatchers — `width × 2` overflow ----
  //
  // The packed Tier 3 sources (Yuyv422 / Uyvy422 / Yvyu422) consume
  // `2 * width` bytes per row. Without the [`packed_yuv422_row_bytes`]
  // helper a 32-bit caller could overflow `width * 2` to a small
  // value, pass the input-side `assert!` with an undersized slice,
  // and reach unsafe SIMD loads. Each packed RGB / RGBA / luma
  // entry point gets its own regression so a future regression on
  // any one of them surfaces independently.

  /// Smallest even width whose `width × 2` overflows 32-bit `usize`
  /// without first failing the `width × 3` overflow guard the
  /// 3-channel-output dispatchers also enforce. On 32-bit
  /// `usize::MAX / 2 == 2^31 - 1` is odd, so `+ 1` produces an
  /// even value (`2^31`); the `+ (candidate & 1)` is a parity
  /// safety on hypothetical platforms where this differs.
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_TIMES_2: usize = {
    let candidate = (usize::MAX / 2) + 1;
    candidate + (candidate & 1)
  };

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuyv422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yuyv422_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn uyvy422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    uyvy422_to_rgba_row(
      &p,
      &mut rgba,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yvyu422_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    yvyu422_to_rgb_row(
      &p,
      &mut rgb,
      OVERFLOW_WIDTH_TIMES_2,
      ColorMatrix::Bt601,
      true,
      false,
    );
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yuyv422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    yuyv422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn uyvy422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    uyvy422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn yvyu422_luma_dispatcher_rejects_width_times_2_overflow() {
    let p: [u8; 0] = [];
    let mut luma: [u8; 0] = [];
    yvyu422_to_luma_row(&p, &mut luma, OVERFLOW_WIDTH_TIMES_2, false);
  }
}

#[cfg(all(test, feature = "std"))]
mod bayer_dispatcher_tests {
  //! Boundary-contract tests for the public Bayer row dispatchers.
  //! Walker / kernel correctness lives in `crate::raw::bayer*` and
  //! `crate::row::scalar`; these tests target the dispatcher's own
  //! preflight (notably the new `assert_color_transform_well_formed`
  //! check and the existing length / `BITS` / `rgb_out` checks)
  //! since that surface is what unsafe SIMD backends will rely on.
  use super::*;
  use crate::raw::{BayerDemosaic, BayerPattern};

  fn ident() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer_dispatcher_rejects_nan_in_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[1][1] = f32::NAN;
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer_dispatcher_rejects_infinity_in_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[0][2] = f32::INFINITY;
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer16_u8_dispatcher_rejects_neg_infinity_in_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u8; 12];
    let mut m = ident();
    m[2][1] = f32::NEG_INFINITY;
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  #[should_panic(expected = "non-finite")]
  fn bayer16_u16_dispatcher_rejects_nan_in_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u16; 12];
    let mut m = ident();
    m[2][2] = f32::NAN;
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  #[test]
  fn bayer_dispatcher_accepts_finite_m() {
    // Sanity: the assertion doesn't fire for ordinary finite
    // matrices. Realistic inputs (CCM with negative crosstalk,
    // WB > 1) all qualify.
    let above = [10u8; 4];
    let mid = [20u8; 4];
    let below = [30u8; 4];
    let mut rgb = [0u8; 12];
    let m: [[f32; 3]; 3] = [[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]];
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// Codex regression (round 8): a direct row-API caller that
  /// bypasses [`crate::raw::WhiteBalance::try_new`] /
  /// [`crate::raw::ColorCorrectionMatrix::try_new`] cannot inject
  /// finite-but-extreme matrices that would overflow during the
  /// per-pixel matmul. The dispatcher's
  /// `assert_color_transform_well_formed` enforces the same
  /// magnitude bound (1e12) that validated WB × CCM can produce.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer_dispatcher_rejects_finite_extreme_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[1][1] = f32::MAX; // finite but past the bound
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// Same regression for the Bayer16→u8 dispatcher.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer16_u8_dispatcher_rejects_finite_extreme_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u8; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[2][0] = -f32::MAX;
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// Same regression for the Bayer16→u16 dispatcher.
  #[test]
  #[should_panic(expected = "exceeds magnitude bound")]
  fn bayer16_u16_dispatcher_rejects_finite_extreme_m() {
    let above = [0u16; 4];
    let mid = [0u16; 4];
    let below = [0u16; 4];
    let mut rgb = [0u16; 12];
    let mut m = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    m[0][0] = 1e20; // finite but past the 1e12 bound
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  /// At-bound element passes (boundary inclusive, matches the
  /// constructor bounds).
  #[test]
  fn bayer_dispatcher_accepts_at_bound_m() {
    let above = [0u8; 4];
    let mid = [0u8; 4];
    let below = [0u8; 4];
    let mut rgb = [0u8; 12];
    let m = [
      [super::MAX_FUSED_TRANSFORM_ABS, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
    ];
    bayer_to_rgb_row(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
  }

  // ---- Direct Bayer16 row-API contract behavior --------------------------
  //
  // The walker path (`bayer16_to`) cannot reach the kernel with
  // out-of-range samples because `BayerFrame16::try_new` validates
  // every active sample at construction. The direct row API
  // (`bayer16_to_rgb_row`, `bayer16_to_rgb_u16_row`) takes raw
  // `&[u16]` slices and trusts the low-packed contract — out-of-
  // range samples are documented as "defined-but-saturated output,
  // no panic, no UB." These regressions pin that behavior so a
  // future change can't silently flip it (e.g., to a panic or to
  // masking) without updating the documented contract first.

  /// 12-bit dispatcher with MSB-aligned `0x8000` input
  /// (the classic packing-mismatch bug, where the caller forgot
  /// to right-shift before feeding the row API). Out-of-range
  /// per the low-packed contract; the kernel saturates the matmul
  /// output to `255` rather than panicking. Walker users get
  /// `Err(SampleOutOfRange)` from `try_new` instead.
  #[test]
  fn bayer16_u8_dispatcher_saturates_on_msb_aligned_input() {
    let above = [0x8000u16; 4];
    let mid = [0x8000u16; 4];
    let below = [0x8000u16; 4];
    let mut rgb = [0u8; 12];
    let m = ident();
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // 0x8000 = 32768 ≫ max_in (4095). All output channels clamp
    // to 255. No panic, no UB — defined behavior.
    assert!(
      rgb.iter().all(|&c| c == 255),
      "MSB-aligned 12-bit input expected to saturate to 255 across all channels; got {rgb:?}"
    );
  }

  /// Same regression for the u16 dispatcher: MSB-aligned 10-bit
  /// input saturates to the low-packed max (1023) rather than
  /// panicking.
  #[test]
  fn bayer16_u16_dispatcher_saturates_on_msb_aligned_input() {
    let above = [0xFFC0u16; 4]; // MSB-aligned 10-bit "white" (1023 << 6)
    let mid = [0xFFC0u16; 4];
    let below = [0xFFC0u16; 4];
    let mut rgb = [0u16; 12];
    let m = ident();
    bayer16_to_rgb_u16_row::<10>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // 0xFFC0 ≫ low-packed-10 max (1023). Output saturates to
    // 1023 (the u16 path's max_out). No panic.
    assert!(
      rgb.iter().all(|&c| c == 1023),
      "MSB-aligned 10-bit input expected to saturate to 1023 across all channels; got {rgb:?}"
    );
  }

  /// In-range Bayer16 input still works correctly through the
  /// direct row API (this protects the rest of the contract while
  /// the saturation tests pin the out-of-range behavior).
  #[test]
  fn bayer16_u8_dispatcher_in_range_input_correct() {
    let above = [4095u16; 4]; // 12-bit white, in range
    let mid = [4095u16; 4];
    let below = [4095u16; 4];
    let mut rgb = [0u8; 12];
    let m = ident();
    bayer16_to_rgb_row::<12>(
      &above,
      &mid,
      &below,
      0,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      &m,
      &mut rgb,
      false,
    );
    // Solid white (4095) at every site → output 255 on every
    // channel. Same final value as the saturated case, but the
    // path is correct (not a clamp).
    assert!(
      rgb.iter().all(|&c| c == 255),
      "in-range 12-bit white expected to map to 255; got {rgb:?}"
    );
  }
}
