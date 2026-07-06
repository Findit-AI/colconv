//! Architecture‑specific SIMD backends for the row primitives.
//!
//! Each submodule here is gated on the target architecture it targets.
//! The public dispatcher in [`super`] selects among them at call
//! boundaries.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx512;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_common;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_sse41;

// This whole backend is the `#[target_feature(enable = "simd128")]` SIMD tier.
// When the crate is built for a wasm target *without* `+simd128` (e.g. the
// generic `cross` CI leg), the dispatcher's `simd128_available()` gate is
// const-false so every kernel is unreachable — the intrinsic `use` globs read
// as unused and the private helpers as dead. Both are required for the
// `+simd128` build, so allow the dead-code-family lints in the no-simd128
// configuration only (the `+simd128` build keeps them live and checked).
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(target_feature = "simd128"), allow(unused_imports, dead_code))]
pub(crate) mod wasm_simd128;
