//! Runtime SIMD dispatcher for Strategy A+ α-extract helpers. Selects
//! the highest-available SIMD backend per platform; falls back to scalar.
//!
//! Each dispatcher function has the same signature as its scalar
//! counterpart plus a `use_simd: bool` flag, and is `pub(crate)` so the
//! sinker impls can call it without going through the public row API.
//! The `use_simd` flag mirrors `MixedSinker::with_simd(false)` — when
//! `false`, the dispatcher skips feature detection and calls scalar
//! directly. This is required for benchmarking, fuzzing, and
//! differential-testing parity with the rest of the kernel call sites
//! that already accept `use_simd`.
//!
//! Dispatch follows the standard `cfg_select!` pattern used everywhere
//! in `dispatch::*`: the platform arm is selected at compile time, and
//! the best available backend is selected at runtime via the
//! `*_available()` helpers in [`crate::row`].

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::alpha_extract as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

// Helper 1: VUYA u8 → u8 RGBA  (α at packed slot 3).
/// Runtime-dispatched α-extract for VUYA: gather α from
/// `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false` (`MixedSinker::with_simd(false)`), the
/// SIMD cascade is bypassed and scalar runs directly.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u8x4_at_3(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime.
        unsafe {
          arch::neon::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
}

// Helper 2: AYUV64 u16 → u8 RGBA  (α at packed slot 0, depth >> 8).
/// Runtime-dispatched α-extract for AYUV64 → u8 RGBA: gather α from
/// `packed[0 + 4*n]` (u16) into `rgba_out[3 + 4*n]` (u8) via `>> 8`.
///
/// `BE` selects the source `packed` plane byte order (`false` = LE on
/// disk/wire — matching the LE-encoded `Ayuv64Frame` contract;
/// `true` = BE). Like [`copy_alpha_plane_u16_to_u8`], the existing SIMD
/// helpers use host-native u16 loads with no `from_le` / `from_be`
/// normalisation, so SIMD is only correct on LE host processing LE
/// source. The dispatcher computes
/// `safe_for_simd = !BE && cfg!(target_endian = "little")` and falls
/// back to the target-endian-aware scalar in every other quadrant.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_0<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  // SIMD α-extract helpers use host-native u16 loads. Force scalar in
  // any quadrant where source byte order doesn't match host byte order.
  let safe_for_simd = !BE && cfg!(target_endian = "little");
  if !safe_for_simd || !use_simd {
    return scalar::copy_alpha_packed_u16x4_to_u8_at_0::<BE>(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe {
          arch::neon::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_packed_u16x4_to_u8_at_0::<BE>(packed, rgba_out, width);
}

// Helper 3: AYUV64 u16 → u16 RGBA  (α at packed slot 0, no depth conv).
/// Runtime-dispatched α-extract for AYUV64 → u16 RGBA: gather α from
/// `packed[0 + 4*n]` (u16) into `rgba_out[3 + 4*n]` (u16). No depth
/// conversion.
///
/// `BE` selects the source `packed` plane byte order. See
/// [`copy_alpha_packed_u16x4_to_u8_at_0`] for the rationale: SIMD is
/// only correct on LE host with LE source; scalar is target-endian-aware.
#[cfg(feature = "yuv-444-packed")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u16x4_at_0<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let safe_for_simd = !BE && cfg!(target_endian = "little");
  if !safe_for_simd || !use_simd {
    return scalar::copy_alpha_packed_u16x4_at_0::<BE>(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe {
          arch::neon::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_packed_u16x4_at_0::<BE>(packed, rgba_out, width);
}

// Helper 4: α plane u8 → u8 RGBA.
/// Runtime-dispatched α-extract for planar Yuva u8: scatter α plane
/// into `rgba_out[3 + 4*n]`.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false`, calls scalar directly.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  if !use_simd {
    return scalar::copy_alpha_plane_u8(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe {
          arch::neon::copy_alpha_plane_u8(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_plane_u8(alpha, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_plane_u8(alpha, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_plane_u8(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_plane_u8(alpha, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_plane_u8(alpha, rgba_out, width);
}

// Helper 5: α plane u16 → u8 RGBA  (depth-conv >> (BITS-8)).
/// Runtime-dispatched α-extract for planar Yuva*p high-bit → u8 RGBA:
/// scatter α plane (u16) into `rgba_out[3 + 4*n]` (u8) with
/// depth-conv `>> (BITS - 8)`.
///
/// `BE` selects the source α plane byte order (`false` = LE on disk/wire,
/// `true` = BE on disk/wire). The SIMD α-extract helpers use host-native
/// `u16` loads (`vld1q_u16` / `_mm_loadu_si128` / `v128_load64_zero`) AND
/// hardcode their scalar tail to `scalar::<BITS, false>`. So SIMD is only
/// correct when BOTH the host CPU is little-endian AND the source data is
/// little-endian — any other quadrant either loads the wrong byte order in
/// the vector body (LE-data on BE-host / BE-data on LE-host) or feeds
/// already-native u16 samples through `u16::from_le` in the scalar tail
/// (BE-data on BE-host), corrupting the tail at non-multiple widths.
///
/// The dispatcher computes
/// `safe_for_simd = !BE && cfg!(target_endian = "little")` and routes to
/// scalar in every other quadrant. The scalar helper is target-endian-aware
/// via `u16::from_be` / `u16::from_le`, so this scalar fallback emits the
/// correct α plane on every host. Phase 4 will plumb BE through the SIMD
/// helpers if a BE-input sinker hot-path lands.
///
/// Truth table (`safe_for_simd = !BE && target_endian == "little"`):
/// - LE data, LE host: `!false && true  = true`  → SIMD (host-native LE u16 loads correct, tail `from_le` is no-op)
/// - LE data, BE host: `!false && false = false` → scalar (handles via `from_le`)
/// - BE data, LE host: `!true  && true  = false` → scalar (handles via `from_be`)
/// - BE data, BE host: `!true  && false = false` → scalar (handles via `from_be`; SIMD vector body would be correct but the tail's `from_le` would corrupt non-multiple widths)
///
/// Selects the highest available SIMD backend on LE-host with LE-data;
/// falls back to scalar otherwise. When `use_simd` is `false`, calls
/// scalar directly.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u16_to_u8<const BITS: u32, const BE: bool>(
  alpha: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  // SIMD α-extract helpers use host-native u16 loads + a scalar tail
  // hardcoded to BE=false. They are only correct on LE host with LE
  // source data. Force scalar in every other quadrant.
  let safe_for_simd = !BE && cfg!(target_endian = "little");
  if !safe_for_simd || !use_simd {
    return scalar::copy_alpha_plane_u16_to_u8::<BITS, BE>(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe {
          arch::neon::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_plane_u16_to_u8::<BITS, BE>(alpha, rgba_out, width);
}

// Helper 6: α plane u16 → u16 RGBA  (no depth conv).
/// Runtime-dispatched α-extract for planar Yuva*p high-bit → u16 RGBA:
/// scatter α plane (u16) into `rgba_out[3 + 4*n]` (u16). No depth
/// conversion.
///
/// `BE` selects the source α plane byte order (`false` = LE on disk/wire,
/// `true` = BE on disk/wire). The dispatcher computes
/// `safe_for_simd = !BE && cfg!(target_endian = "little")` and routes to
/// scalar in every other quadrant: see `copy_alpha_plane_u16_to_u8` above
/// for the truth table and rationale (SIMD α-extract uses host-native u16
/// loads AND hardcodes its scalar tail to `BE=false`, so it only handles
/// the LE-host/LE-data quadrant correctly; scalar is target-endian-aware).
///
/// Selects the highest available SIMD backend on LE-host with LE-data;
/// falls back to scalar otherwise. When `use_simd` is `false`, calls
/// scalar directly.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u16<const BITS: u32, const BE: bool>(
  alpha: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  // SIMD α-extract helpers use host-native u16 loads + a scalar tail
  // hardcoded to BE=false. They are only correct on LE host with LE
  // source data. Force scalar in every other quadrant.
  let safe_for_simd = !BE && cfg!(target_endian = "little");
  if !safe_for_simd || !use_simd {
    return scalar::copy_alpha_plane_u16::<BITS, BE>(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe {
          arch::neon::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe {
          arch::x86_avx512::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe {
          arch::x86_avx2::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe {
          arch::x86_sse41::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe {
          arch::wasm_simd128::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::copy_alpha_plane_u16::<BITS, BE>(alpha, rgba_out, width);
}
