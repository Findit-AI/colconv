//! YUV source kernels.
//!
//! One sub-module and kernel per YUV pixel-format family:
//! - [`Yuv420p`](crate::yuv::Yuv420p) — the mainline 4:2:0 **planar**
//!   layout (H.264 / HEVC / AV1 / VP9 software‑decode default).
//! - [`Nv12`](crate::yuv::Nv12) — 4:2:0 **semi‑planar** with interleaved
//!   UV (VideoToolbox / VA‑API / NVDEC / D3D11VA hardware‑decode
//!   default).
//! - [`Nv21`](crate::yuv::Nv21) — 4:2:0 semi‑planar with **VU**-ordered
//!   chroma (Android MediaCodec default).
//! - [`Yuv420p10`](crate::yuv::Yuv420p10) — 4:2:0 planar at 10 bits
//!   per sample (HDR10 / 10‑bit SDR software decode).
//! - [`Yuv420p12`](crate::yuv::Yuv420p12) — 4:2:0 planar at 12 bits
//!   per sample (HEVC Main 12 / VP9 Profile 3 software decode).
//! - [`Yuv420p14`](crate::yuv::Yuv420p14) — 4:2:0 planar at 14 bits
//!   per sample (grading / mastering pipelines).
//! - [`P010`](crate::yuv::P010) — 4:2:0 semi‑planar at 10 bits per
//!   sample, high‑bit‑packed (HDR hardware decode: VideoToolbox,
//!   VA‑API, NVDEC, D3D11VA, Intel QSV).
//! - [`P012`](crate::yuv::P012) — 4:2:0 semi‑planar at 12 bits per
//!   sample, high‑bit‑packed (HEVC Main 12 / VP9 Profile 3 hardware
//!   decode).
//!
//! Other families land in follow-up commits.

mod nv12;
mod nv21;
mod p010;
mod p012;
mod yuv420p;
mod yuv420p10;
mod yuv420p12;
mod yuv420p14;

pub use nv12::{Nv12, Nv12Row, Nv12Sink, nv12_to};
pub use nv21::{Nv21, Nv21Row, Nv21Sink, nv21_to};
pub use p010::{P010, P010Row, P010Sink, p010_to};
pub use p012::{P012, P012Row, P012Sink, p012_to};
pub use yuv420p::{Yuv420p, Yuv420pRow, Yuv420pSink, yuv420p_to};
pub use yuv420p10::{Yuv420p10, Yuv420p10Row, Yuv420p10Sink, yuv420p10_to};
pub use yuv420p12::{Yuv420p12, Yuv420p12Row, Yuv420p12Sink, yuv420p12_to};
pub use yuv420p14::{Yuv420p14, Yuv420p14Row, Yuv420p14Sink, yuv420p14_to};
