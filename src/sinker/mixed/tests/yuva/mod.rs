//! YUVA MixedSinker integration tests, split by sub-family.
//!
//! `pub(super) use super::*;` forwards `tests/mod.rs`'s glob (which
//! brings in `crate::{ColorMatrix, frame::*, raw::*, yuv::*}` plus
//! the cross-cutting `pseudo_random_u8` etc. helpers) into this
//! module so the per-sub-family submodules can pick them up via
//! their own `use super::*;`.

#[allow(unused_imports)]
pub(super) use super::*;

mod sub_4_2_0;
mod sub_4_2_2;
mod sub_4_4_4;
