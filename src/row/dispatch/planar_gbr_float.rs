//! Runtime SIMD dispatchers for planar GBR float sources.
//!
//! Covers `Gbrpf32` / `Gbrapf32` (f32 element type) and `Gbrpf16` /
//! `Gbrapf16` (half::f16 element type). SIMD backends will be wired in
//! Tasks 3–7; for now every entry calls the scalar kernel directly.
//!
//! `use_simd = false` bypasses any future SIMD cascade and calls scalar
//! directly. Lossless f32-output paths take `_use_simd` (ignored) because
//! they have no SIMD acceleration.
//!
//! # Overflow guards
//!
//! Output-buffer length checks use `rgb_row_bytes` / `rgba_row_bytes` /
//! `rgb_row_elems` / `rgba_row_elems` — the same checked-multiply helpers
//! used throughout the crate. These are hoisted BEFORE plane-bound assertions
//! so a 32-bit overflow surfaces as the documented "overflows usize" panic
//! rather than a passing plane-len check followed by a write past the end of
//! the buffer.
//!
//! # f16-source paths
//!
//! For f16-source → integer / luma / HSV outputs the dispatcher widens each
//! f16 plane to f32 in per-call stack scratch (up to 64 elements/plane,
//! chunked), then calls the corresponding `gbrpf32_to_*` scalar kernel.
//! For f16-source → f16 output the f16-native kernels in
//! [`super::scalar::planar_gbr_f16`] are called directly.

// Dispatchers in this module are not yet consumed by any sinker (Task 8 wires
// the MixedSinker impls). Allow dead_code until then.
#![allow(dead_code)]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, f16c_available, sse41_available};
#[cfg(target_arch = "aarch64")]
use crate::row::{fp16_available, neon_available};
use crate::{
  KernelMatrix,
  row::{
    rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems,
    scalar::{planar_gbr_f16 as scalar_f16, planar_gbr_float as scalar},
  },
};

/// `BE` value that makes the downstream `gbrpf32_to_*` kernels treat their
/// `f32` scratch input as **host-native** (a no-op byte-swap). After we widen
/// f16 → f32 via [`widen_f16_be_to_host_f32`] (which normalizes the source
/// f16 bits per the source `BE` and produces host-native f32), the resulting
/// scratch must be routed via `HOST_NATIVE_BE` so the downstream kernel's
/// `from_le` / `from_be` loaders no-op the swap.
///
/// Truth table — [`widen_f16_be_to_host_f32::<BE>`] + downstream
/// `gbrpf32_to_*::<HOST_NATIVE_BE>` route:
///
///   • LE host, `BE = false`: f16 bits `from_le`-loaded (no-op on LE) →
///     host-native f32; downstream `HOST_NATIVE_BE = false` → kernel's
///     `from_le` no-op on LE → correct.
///   • LE host, `BE = true`:  f16 bits `from_be`-loaded (swap on LE) →
///     host-native f32; downstream `HOST_NATIVE_BE = false` → kernel's
///     `from_le` no-op on LE → correct.
///   • BE host, `BE = false`: f16 bits `from_le`-loaded (swap on BE) →
///     host-native f32; downstream `HOST_NATIVE_BE = true` → kernel's
///     `from_be` no-op on BE → correct.
///   • BE host, `BE = true`:  f16 bits `from_be`-loaded (no-op on BE) →
///     host-native f32; downstream `HOST_NATIVE_BE = true` → kernel's
///     `from_be` no-op on BE → correct.
///
/// Without this routing the f16 → f32 widen path double-byte-swaps on
/// `BE`-source-on-LE-host (and symmetrically `LE`-source-on-BE-host),
/// corrupting every integer / luma / HSV output. This is the
/// **dispatch-layer** complement of the SIMD-backend-internal
/// `HOST_NATIVE_BE` used for the rgbf16 widen-then-convert paths.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Widen `n` `half::f16` values from `src[offset..offset + n]` into
/// `dst[..n]` (f32 elements), normalizing source f16 bits **before** the
// f16-widening helper is shared with per-backend SIMD scalar tails; the
// canonical definition lives in `crate::row::scalar::planar_gbr_f16` so
// every target that compiles either the dispatch fallback OR a SIMD
// backend exercises it (avoids dead-code on miri targets with no SIMD).
use crate::row::scalar::planar_gbr_f16::widen_f16_be_to_host_f32;

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// Dispatch `gbrpf32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgb_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → u8 RGBA ------------------------------------------------------

/// Dispatch `gbrpf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgba_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// Dispatch `gbrpf32_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgb_u16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → u16 RGBA -----------------------------------------------------

/// Dispatch `gbrpf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgba_u16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGB (lossless) -------------------------------------------

/// Dispatch `gbrpf32_to_rgb_f32_row` (lossless interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgb_f32_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      // SSE4.1 delegates to scalar for lossless f32 interleave (no vst3 equivalent).
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgb_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGBA (lossless) ------------------------------------------

/// Dispatch `gbrpf32_to_rgba_f32_row` (lossless).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_rgba_f32_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      // SSE4.1 delegates to scalar for lossless f32 interleave (no vst4 equivalent).
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgba_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGB (fused narrow) ----------------------------------------

/// Dispatch `gbrpf32_to_rgb_f16_row` (fused f32→f16 narrow + interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // fp16 feature needed for vcvt_f16_f32.
          if fp16_available() {
            // SAFETY: NEON + fp16 verified available.
            unsafe { arch::neon::gbrpf32_to_rgb_f16_row_fp16::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // F16C runtime detection for narrow.
          if f16c_available() {
            // SAFETY: AVX-512F + BW + F16C verified available.
            unsafe { arch::x86_avx512::gbrpf32_to_rgb_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
        if avx2_available() {
          // F16C runtime detection for narrow.
          if f16c_available() {
            // SAFETY: AVX2 + F16C verified available.
            unsafe { arch::x86_avx2::gbrpf32_to_rgb_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
        if sse41_available() {
          // F16C runtime detection for narrow.
          if f16c_available() {
            // SAFETY: SSE4.1 + F16C verified available.
            unsafe { arch::x86_sse41::gbrpf32_to_rgb_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // wasm32 has no native f16 narrowing — delegates to scalar narrow.
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGBA (fused narrow) ---------------------------------------

/// Dispatch `gbrpf32_to_rgba_f16_row` (fused f32→f16 narrow + interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // fp16 feature needed for vcvt_f16_f32.
          if fp16_available() {
            // SAFETY: NEON + fp16 verified available.
            unsafe { arch::neon::gbrpf32_to_rgba_f16_row_fp16::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          if f16c_available() {
            // SAFETY: AVX-512F + BW + F16C verified available.
            unsafe { arch::x86_avx512::gbrpf32_to_rgba_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
        if avx2_available() {
          if f16c_available() {
            // SAFETY: AVX2 + F16C verified available.
            unsafe { arch::x86_avx2::gbrpf32_to_rgba_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
        if sse41_available() {
          if f16c_available() {
            // SAFETY: SSE4.1 + F16C verified available.
            unsafe { arch::x86_sse41::gbrpf32_to_rgba_f16_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // wasm32 has no native f16 narrowing — delegates to scalar narrow.
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → u8 luma ------------------------------------------------------

/// Dispatch `gbrpf32_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= width, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_luma_row::<BE>(g, b, r, out, width, matrix, full_range);
}

// ---- Gbrpf32 → u16 luma -----------------------------------------------------

/// Dispatch `gbrpf32_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= width, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
          }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_luma_u16_row::<BE>(g, b, r, out, width, matrix, full_range);
}

// ---- Gbrpf32 → HSV ----------------------------------------------------------

/// Dispatch `gbrpf32_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_hsv_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(h_out.len() >= width, "h_out too short");
  assert!(s_out.len() >= width, "s_out too short");
  assert!(v_out.len() >= width, "v_out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
          }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrpf32_to_hsv_row::<BE>(g, b, r, h_out, s_out, v_out, width);
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

/// Dispatch `gbrapf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrapf32_to_rgba_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// Dispatch `gbrapf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe { arch::x86_avx512::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe { arch::x86_avx2::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe { arch::x86_sse41::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrapf32_to_rgba_u16_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// Dispatch `gbrapf32_to_rgba_f32_row` (lossless).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe { arch::neon::gbrapf32_to_rgba_f32_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      // SSE4.1 delegates to scalar for lossless f32 4-channel interleave.
      _ => {}
    }
  }
  scalar::gbrapf32_to_rgba_f32_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, source α) ---------------------------

/// Dispatch `gbrapf32_to_rgba_f16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // fp16 feature needed for vcvt_f16_f32.
          if fp16_available() {
            // SAFETY: NEON + fp16 verified available.
            unsafe { arch::neon::gbrapf32_to_rgba_f16_row_fp16::<BE>(g, b, r, a, out, width); }
          } else {
            scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
          }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          if f16c_available() {
            // SAFETY: AVX-512F + BW + F16C verified available.
            unsafe { arch::x86_avx512::gbrapf32_to_rgba_f16_row_f16c::<BE>(g, b, r, a, out, width); }
          } else {
            scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
          }
          return;
        }
        if avx2_available() {
          if f16c_available() {
            // SAFETY: AVX2 + F16C verified available.
            unsafe { arch::x86_avx2::gbrapf32_to_rgba_f16_row_f16c::<BE>(g, b, r, a, out, width); }
          } else {
            scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
          }
          return;
        }
        if sse41_available() {
          if f16c_available() {
            // SAFETY: SSE4.1 + F16C verified available.
            unsafe { arch::x86_sse41::gbrapf32_to_rgba_f16_row_f16c::<BE>(g, b, r, a, out, width); }
          } else {
            scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // wasm32 has no native f16 narrowing — delegates to scalar narrow.
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrpf16 → f16 RGB (lossless, f16-native) --------------------------------

/// Dispatch `gbrpf16_to_rgb_f16_row` (lossless f16 interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available (no fp16 needed — lossless u16 reinterpret).
          unsafe { arch::neon::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time (lossless, delegates scalar).
          unsafe { arch::wasm_simd128::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx512::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx2::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_sse41::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar_f16::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf16 → f16 RGBA (lossless, f16-native) ------------------------------

/// Dispatch `gbrpf16_to_rgba_f16_row` (lossless f16 interleave + α = f16(1.0)).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available (no fp16 needed — lossless u16 reinterpret).
          unsafe { arch::neon::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time (lossless, delegates scalar).
          unsafe { arch::wasm_simd128::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx512::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx2::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_sse41::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar_f16::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrapf16 → f16 RGBA (lossless, source α) --------------------------------

/// Dispatch `gbrapf16_to_rgba_f16_row` (lossless f16 interleave + source α).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available (no fp16 needed — lossless u16 reinterpret).
          unsafe { arch::neon::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time (lossless, delegates scalar).
          unsafe { arch::wasm_simd128::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx512::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_avx2::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available (no F16C needed — lossless u16 reinterpret).
          unsafe { arch::x86_sse41::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar_f16::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrpf16 → u16 RGB (fp16 NEON / F16C x86 widen / wasm simd128 / scalar) --

/// Dispatch `gbrpf16_to_rgb_u16_row`: NEON fp16 or F16C x86 widen+SIMD when
/// available, wasm-simd128 widen+SIMD on wasm32, else scalar widen fallback.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_u16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          // SAFETY: NEON + fp16 verified available.
          unsafe { arch::neon::gbrpf16_to_rgb_u16_row_fp16::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf16_to_rgb_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          // SAFETY: AVX-512F + BW + F16C verified available.
          unsafe { arch::x86_avx512::gbrpf16_to_rgb_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() && f16c_available() {
          // SAFETY: AVX2 + F16C verified available.
          unsafe { arch::x86_avx2::gbrpf16_to_rgb_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() && f16c_available() {
          // SAFETY: SSE4.1 + F16C verified available.
          unsafe { arch::x86_sse41::gbrpf16_to_rgb_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  // Scalar fallback: widen f16 → f32 (host-native after `from_be`/`from_le`),
  // then scalar f32 kernel routed via `HOST_NATIVE_BE` to avoid double swap.
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    scalar::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[offset * 3..],
      n,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → u16 RGBA (fp16 NEON / F16C x86 widen / wasm simd128 / scalar) -

/// Dispatch `gbrpf16_to_rgba_u16_row`: NEON fp16 or F16C x86 widen+SIMD when
/// available, wasm-simd128 widen+SIMD on wasm32, else scalar widen fallback.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_u16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && fp16_available() {
          // SAFETY: NEON + fp16 verified available.
          unsafe { arch::neon::gbrpf16_to_rgba_u16_row_fp16::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf16_to_rgba_u16_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() && f16c_available() {
          // SAFETY: AVX-512F + BW + F16C verified available.
          unsafe { arch::x86_avx512::gbrpf16_to_rgba_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
        if avx2_available() && f16c_available() {
          // SAFETY: AVX2 + F16C verified available.
          unsafe { arch::x86_avx2::gbrpf16_to_rgba_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
        if sse41_available() && f16c_available() {
          // SAFETY: SSE4.1 + F16C verified available.
          unsafe { arch::x86_sse41::gbrpf16_to_rgba_u16_row_f16c::<BE>(g, b, r, out, width); }
          return;
        }
      },
      _ => {}
    }
  }
  // Scalar fallback: widen f16 → f32 (host-native after `from_be`/`from_le`),
  // then scalar f32 kernel routed via `HOST_NATIVE_BE` to avoid double swap.
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    scalar::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[offset * 4..],
      n,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → u8 RGB (fp16 NEON widen / F16C SSE4.1 widen / wasm / scalar) -

/// Dispatch `gbrpf16_to_rgb_row`: NEON fp16 or SSE4.1+F16C widening when
/// available, wasm-simd128 widen+SIMD on wasm32, else scalar fallback.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          if fp16_available() {
            // SAFETY: NEON + fp16 verified available.
            unsafe { arch::neon::gbrpf16_to_rgb_row_fp16::<BE>(g, b, r, out, width); }
          } else {
            // NEON available but no fp16 — widen scalar (host-native after
            // `from_be`/`from_le`), then NEON f32→u8 routed via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: NEON verified available.
              unsafe {
                arch::neon::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(&gf[..n], &bf[..n], &rf[..n], &mut out[offset * 3..], n);
              }
              offset += n;
            }
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf16_to_rgb_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          if f16c_available() {
            // SAFETY: AVX-512F + BW + F16C verified available.
            unsafe { arch::x86_avx512::gbrpf16_to_rgb_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // AVX-512 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then AVX-512 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: AVX-512F + BW verified available.
              unsafe {
                arch::x86_avx512::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 3..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
        if avx2_available() {
          if f16c_available() {
            // SAFETY: AVX2 + F16C verified available.
            unsafe { arch::x86_avx2::gbrpf16_to_rgb_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // AVX2 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then AVX2 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: AVX2 verified available.
              unsafe {
                arch::x86_avx2::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 3..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
        if sse41_available() {
          if f16c_available() {
            // SAFETY: SSE4.1 + F16C verified available.
            unsafe { arch::x86_sse41::gbrpf16_to_rgb_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // SSE4.1 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then SSE4.1 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: SSE4.1 verified available.
              unsafe {
                arch::x86_sse41::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 3..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
      },
      _ => {}
    }
  }
  // Scalar fallback: widen f16 → f32 (host-native after `from_be`/`from_le`),
  // then scalar f32 kernel routed via `HOST_NATIVE_BE` to avoid double swap.
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    scalar::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[offset * 3..],
      n,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → u8 RGBA (fp16 NEON widen / F16C SSE4.1 widen / wasm / scalar) -

/// Dispatch `gbrpf16_to_rgba_row`: NEON fp16 or SSE4.1+F16C widening when
/// available, wasm-simd128 widen+SIMD on wasm32, else scalar fallback.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          if fp16_available() {
            // SAFETY: NEON + fp16 verified available.
            unsafe { arch::neon::gbrpf16_to_rgba_row_fp16::<BE>(g, b, r, out, width); }
          } else {
            // NEON available but no fp16 — widen scalar (host-native after
            // `from_be`/`from_le`), then NEON f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: NEON verified available.
              unsafe {
                arch::neon::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 4..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe { arch::wasm_simd128::gbrpf16_to_rgba_row::<BE>(g, b, r, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          if f16c_available() {
            // SAFETY: AVX-512F + BW + F16C verified available.
            unsafe { arch::x86_avx512::gbrpf16_to_rgba_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // AVX-512 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then AVX-512 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: AVX-512F + BW verified available.
              unsafe {
                arch::x86_avx512::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 4..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
        if avx2_available() {
          if f16c_available() {
            // SAFETY: AVX2 + F16C verified available.
            unsafe { arch::x86_avx2::gbrpf16_to_rgba_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // AVX2 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then AVX2 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: AVX2 verified available.
              unsafe {
                arch::x86_avx2::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 4..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
        if sse41_available() {
          if f16c_available() {
            // SAFETY: SSE4.1 + F16C verified available.
            unsafe { arch::x86_sse41::gbrpf16_to_rgba_row_f16c::<BE>(g, b, r, out, width); }
          } else {
            // SSE4.1 available but no F16C — widen scalar (host-native after
            // `from_be`/`from_le`), then SSE4.1 f32→u8 via `HOST_NATIVE_BE`.
            const CHUNK: usize = 64;
            let mut gf = [0.0f32; CHUNK];
            let mut bf = [0.0f32; CHUNK];
            let mut rf = [0.0f32; CHUNK];
            let mut offset = 0;
            while offset < width {
              let n = (width - offset).min(CHUNK);
              widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
              widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
              widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
              // SAFETY: SSE4.1 verified available.
              unsafe {
                arch::x86_sse41::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
                  &gf[..n],
                  &bf[..n],
                  &rf[..n],
                  &mut out[offset * 4..],
                  n,
                );
              }
              offset += n;
            }
          }
          return;
        }
      },
      _ => {}
    }
  }
  // Scalar fallback: widen f16 → f32 (host-native after `from_be`/`from_le`),
  // then scalar f32 kernel routed via `HOST_NATIVE_BE` to avoid double swap.
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    scalar::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[offset * 4..],
      n,
    );
    offset += n;
  }
}

// ---- 32-bit overflow guard tests --------------------------------------------

#[cfg(all(test, feature = "std", target_pointer_width = "32"))]
mod tests {
  use super::*;

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgb_panics_on_width_overflow() {
    // Use empty slices so no allocation happens before the dispatcher's
    // checked-multiply helper fires. On i686 (usize=u32), width*3 overflows,
    // and rgb_row_bytes(width) panics with "overflows usize" before any slice
    // bounds check is reached.
    let g: &[f32] = &[];
    let b: &[f32] = &[];
    let r: &[f32] = &[];
    let mut out: [u8; 0] = [];
    let w = usize::MAX / 2 + 1;
    gbrpf32_to_rgb_row::<false>(g, b, r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgba_panics_on_width_overflow() {
    // Use empty slices so no allocation happens before the dispatcher's
    // checked-multiply helper fires. On i686 (usize=u32), width*4 overflows,
    // and rgba_row_bytes(width) panics with "overflows usize" before any slice
    // bounds check is reached.
    let g: &[f32] = &[];
    let b: &[f32] = &[];
    let r: &[f32] = &[];
    let mut out: [u8; 0] = [];
    let w = usize::MAX / 2 + 1;
    gbrpf32_to_rgba_row::<false>(g, b, r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgb_u16_panics_on_width_overflow() {
    // Use empty slices so no allocation happens before the dispatcher's
    // checked-multiply helper fires. On i686 (usize=u32), width*3 overflows,
    // and rgb_row_elems(width) panics with "overflows usize" before any slice
    // bounds check is reached.
    let g: &[f32] = &[];
    let b: &[f32] = &[];
    let r: &[f32] = &[];
    let mut out: [u16; 0] = [];
    let w = usize::MAX / 2 + 1;
    gbrpf32_to_rgb_u16_row::<false>(g, b, r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgba_u16_panics_on_width_overflow() {
    // Use empty slices so no allocation happens before the dispatcher's
    // checked-multiply helper fires. On i686 (usize=u32), width*4 overflows,
    // and rgba_row_elems(width) panics with "overflows usize" before any slice
    // bounds check is reached.
    let g: &[f32] = &[];
    let b: &[f32] = &[];
    let r: &[f32] = &[];
    let mut out: [u16; 0] = [];
    let w = usize::MAX / 2 + 1;
    gbrpf32_to_rgba_u16_row::<false>(g, b, r, &mut out, w, false);
  }
}
