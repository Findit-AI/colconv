//! YUV source kernels.
//!
//! One sub-module and kernel per YUV pixel-format family. v0.1 ships
//! [`Yuv420p`](crate::yuv::Yuv420p) — the mainline 4:2:0 planar layout
//! (H.264 / HEVC / AV1 / VP9 default); other families land in follow-
//! up commits.

mod yuv420p;

pub use yuv420p::{Yuv420p, Yuv420pRow, Yuv420pSink, yuv420p_to};
