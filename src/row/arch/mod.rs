//! Architecture‑specific SIMD backends for the row primitives.
//!
//! Each submodule here is gated on the target architecture it targets.
//! The public dispatcher in [`super`] selects among them at call
//! boundaries.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;
