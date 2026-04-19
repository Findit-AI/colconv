//! YUV source kernels.
//!
//! One sub-module and kernel per YUV pixel-format family:
//! - [`Yuv420p`](crate::yuv::Yuv420p) — the mainline 4:2:0 **planar**
//!   layout (H.264 / HEVC / AV1 / VP9 software‑decode default).
//! - [`Nv12`](crate::yuv::Nv12) — 4:2:0 **semi‑planar** with interleaved
//!   UV (VideoToolbox / VA‑API / NVDEC / D3D11VA hardware‑decode
//!   default).
//!
//! Other families land in follow-up commits.

mod nv12;
mod yuv420p;

pub use nv12::{Nv12, Nv12Row, Nv12Sink, nv12_to};
pub use yuv420p::{Yuv420p, Yuv420pRow, Yuv420pSink, yuv420p_to};
