//! Tier 12 — packed CIE XYZ 12-bit (`Xyz12`) source-side row
//! dispatchers.
//!
//! Each entry point converts one row of packed `X, Y, Z` `u16` input
//! (high-bit-packed per FFmpeg `AV_PIX_FMT_XYZ12LE/BE`: active 12 bits
//! in `[15:4]`, low 4 bits zero) to the requested output format. Every
//! kernel takes:
//!
//! - `BE: const bool` — wire-format endianness of the source `u16`s.
//! - `target_gamut: KernelGamut` — runtime choice of XYZ → RGB
//!   matrix (DCI-P3 / Rec.709 / Rec.2020).
//!
//! Pipeline (per pixel): SMPTE ST 428-1 §8 inverse-OETF → 3×3 matmul
//! → sRGB-shape OETF (skipped for f32 outputs) → range scale + integer
//! narrow (only for u8 / u16 outputs).
//!
//! SIMD backends: NEON (aarch64), SSE4.1 / AVX2 / AVX-512 (x86_64),
//! and wasm-simd128 — runtime-selected on x86_64 with the standard
//! AVX-512 → AVX2 → SSE4.1 priority order; compile-time-selected on
//! wasm32. Each backend follows the established `cfg_select!` pattern
//! from `dispatch::planar_gbr_*`.

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
  KernelGamut,
  row::{
    rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::xyz12 as scalar_xyz12,
  },
};

/// XYZ12 → packed `R, G, B` `u8` row dispatcher.
///
/// `use_simd = false` forces scalar; SIMD backends pick up the call
/// when their architecture is detected at runtime (NEON / SSE4.1 /
/// AVX2 / AVX-512) or compile-time (wasm-simd128).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_bytes(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `u8` row dispatcher (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_bytes(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B` `u16` row dispatcher (full-range scaling).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgb_u16_row::<BE>(
              xyz,
              rgb_out,
              width,
              target_gamut,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `u16` row dispatcher (alpha = `0xFFFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgba_u16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgba_u16_row::<BE>(
              xyz,
              rgba_out,
              width,
              target_gamut,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgba_u16_row::<BE>(
              xyz,
              rgba_out,
              width,
              target_gamut,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgba_u16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgba_u16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgba_u16_row::<BE>(xyz, rgba_out, width, target_gamut);
}

/// XYZ12 → packed linear `R, G, B` `f32` row dispatcher.
///
/// **Lossless** linear-RGB output — no OETF, no clamp. Out-of-gamut
/// negative R/G/B and HDR > 1 values are emitted bit-exact.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_f32_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [f32],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgb_f32_row::<BE>(
              xyz,
              rgb_out,
              width,
              target_gamut,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed linear `X, Y, Z` `f32` row dispatcher (lossless XYZ
/// pass-through after step-1 inverse-OETF).
///
/// No matrix, no gamma, no clamp — useful for callers that do their
/// own gamut conversion downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_xyz_f32_row<const BE: bool>(
  xyz: &[u16],
  xyz_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let xyz_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(xyz_out.len() >= xyz_out_min, "xyz_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
}

/// XYZ12 → packed `R, G, B` `f16` row dispatcher (gamma-encoded,
/// clamped to `[0, 1]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_f16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgb_f16_row::<BE>(
              xyz,
              rgb_out,
              width,
              target_gamut,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `f16` row dispatcher (alpha = `1.0`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_f16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
  use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified available.
          unsafe {
            arch::neon::xyz12::xyz12_to_rgba_f16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 verified available at compile time.
          unsafe {
            arch::wasm_simd128::xyz12::xyz12_to_rgba_f16_row::<BE>(
              xyz,
              rgba_out,
              width,
              target_gamut,
            );
          }
          return;
        }
      }
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F + BW verified available.
          unsafe {
            arch::x86_avx512::xyz12::xyz12_to_rgba_f16_row::<BE>(
              xyz,
              rgba_out,
              width,
              target_gamut,
            );
          }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified available.
          unsafe {
            arch::x86_avx2::xyz12::xyz12_to_rgba_f16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified available.
          unsafe {
            arch::x86_sse41::xyz12::xyz12_to_rgba_f16_row::<BE>(xyz, rgba_out, width, target_gamut);
          }
          return;
        }
      }
      _ => {}
    }
  }

  scalar_xyz12::xyz12_to_rgba_f16_row::<BE>(xyz, rgba_out, width, target_gamut);
}

/// XYZ12 staged-RGB → `u8` luma row dispatcher.
///
/// Inputs the staged `u8` packed RGB row produced by
/// [`xyz12_to_rgb_row`] and emits a single-channel `u8` luma plane
/// using the gamut-derived Q15 weights from
/// [`crate::source::luma_weights_q15_for_gamut`] (carried on
/// [`crate::source::Xyz12Row::luma_q15`]).
///
/// **No SIMD** path: per-pixel cost (one Q15 dot product) is dwarfed
/// by the upstream 6× scalar `powf` work in the matmul + OETF stages.
/// `use_simd` is accepted for sinker-API uniformity and ignored.
///
/// DciP3 must use its own Q15 weights (derived from the DCI-white-pointed
/// RGB→XYZ matrix Y row); the BT.709 triple biases luma for saturated
/// content under DCI-P3.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_rgb_to_luma_row(
  rgb: &[u8],
  luma_out: &mut [u8],
  width: usize,
  luma_q15: (i32, i32, i32),
  _use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb.len() >= rgb_min, "rgb row too short");
  assert!(luma_out.len() >= width, "luma row too short");
  scalar_xyz12::xyz12_rgb_to_luma_row(rgb, luma_out, width, luma_q15);
}

/// XYZ12 staged-RGB → `u16` luma row dispatcher.
///
/// `u16` carrier preserves the `[0, 255]` dynamic range from the u8
/// luma path (zero-extended), matching every other `*_to_luma_u16_row`
/// kernel in pixon. `use_simd` is currently a no-op (see
/// [`xyz12_rgb_to_luma_row`] for rationale).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_rgb_to_luma_u16_row(
  rgb: &[u8],
  luma_out: &mut [u16],
  width: usize,
  luma_q15: (i32, i32, i32),
  _use_simd: bool,
) {
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb.len() >= rgb_min, "rgb row too short");
  assert!(luma_out.len() >= width, "luma row too short");
  scalar_xyz12::xyz12_rgb_to_luma_u16_row(rgb, luma_out, width, luma_q15);
}
