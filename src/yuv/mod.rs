//! YUV source kernels.
//!
//! One sub-module and kernel per YUV pixel-format family.
//!
//! # Shipped (8-bit 4:2:0)
//!
//! - [`Yuv420p`](crate::yuv::Yuv420p) — the mainline 4:2:0 **planar**
//!   layout (H.264 / HEVC / AV1 / VP9 software‑decode default).
//! - [`Nv12`](crate::yuv::Nv12) — 4:2:0 **semi‑planar** with interleaved
//!   UV (VideoToolbox / VA‑API / NVDEC / D3D11VA hardware‑decode
//!   default).
//! - [`Nv21`](crate::yuv::Nv21) — 4:2:0 semi‑planar with **VU**-ordered
//!   chroma (Android MediaCodec default).
//!
//! # Shipped (8-bit 4:2:2 / 4:4:4)
//!
//! - [`Nv16`](crate::yuv::Nv16) — 4:2:2 semi‑planar, UV‑ordered.
//!   Reuses [`Nv12`](crate::yuv::Nv12)'s per‑row kernel; the 4:2:0
//!   vs 4:2:2 difference is purely in the vertical walker.
//! - [`Nv24`](crate::yuv::Nv24) — 4:4:4 semi‑planar, UV‑ordered.
//!   Dedicated kernel family (chroma is 1:1 with Y, no
//!   duplication step).
//! - [`Nv42`](crate::yuv::Nv42) — 4:4:4 semi‑planar, **VU**‑ordered.
//!   Shares kernels with [`Nv24`](crate::yuv::Nv24) via a `SWAP_UV`
//!   const generic, the same way [`Nv21`](crate::yuv::Nv21) pairs
//!   with [`Nv12`](crate::yuv::Nv12).
//!
//! # Shipped (high-bit-depth 4:2:0, low-bit-packed planar)
//!
//! - [`Yuv420p10`](crate::yuv::Yuv420p10) — 4:2:0 planar at 10 bits
//!   per sample (HDR10 / 10‑bit SDR software decode).
//! - [`Yuv420p12`](crate::yuv::Yuv420p12) — 4:2:0 planar at 12 bits
//!   per sample (HEVC Main 12 / VP9 Profile 3 software decode).
//! - [`Yuv420p14`](crate::yuv::Yuv420p14) — 4:2:0 planar at 14 bits
//!   per sample (grading / mastering pipelines).
//! - [`Yuv420p16`](crate::yuv::Yuv420p16) — 4:2:0 planar at 16 bits
//!   per sample (reference / intermediate HDR, runs on the parallel
//!   i64 kernel family).
//!
//! # Shipped (high-bit-depth 4:2:0, high-bit-packed semi-planar)
//!
//! - [`P010`](crate::yuv::P010) — 4:2:0 semi‑planar at 10 bits per
//!   sample, high‑bit‑packed (HDR hardware decode: VideoToolbox,
//!   VA‑API, NVDEC, D3D11VA, Intel QSV).
//! - [`P012`](crate::yuv::P012) — 4:2:0 semi‑planar at 12 bits per
//!   sample, high‑bit‑packed (HEVC Main 12 / VP9 Profile 3 hardware
//!   decode).
//! - [`P016`](crate::yuv::P016) — 4:2:0 semi‑planar at 16 bits per
//!   sample (reference). At 16 bits the high‑vs‑low packing
//!   distinction degenerates — every bit is active.
//!
//! # Kernel families
//!
//! - **Q15 i32 family** covers 8‑bit (non-generic `yuv_420_to_rgb_row`
//!   + siblings) and 10/12/14‑bit (const-generic `yuv_420p_n_to_rgb_*
//!   <BITS>` + siblings). Hot path for SDR + most HDR workflows.
//! - **i64 chroma-widened family** covers 16‑bit
//!   (`yuv_420p16_to_rgb_*` + `p16_to_rgb_*`). The chroma matrix
//!   multiply `c_u * u_d + c_v * v_d` overflows i32 at 16 bits, so
//!   the 16‑bit kernels widen that specific step to i64 and narrow
//!   back after the `>> 15`. Scalar stays free; SIMD pays a ~2×
//!   chroma compute tax in exchange for i32 overflow safety.
//!
//! # Not yet shipped
//!
//! - **Planar 4:2:2 / 4:4:4** (`Yuv422p`, `Yuv444p`) — semi‑planar
//!   4:2:2 and 4:4:4 now ship as [`Nv16`](crate::yuv::Nv16) /
//!   [`Nv24`](crate::yuv::Nv24) / [`Nv42`](crate::yuv::Nv42). The
//!   planar equivalents would share the same row math but need their
//!   own frame types and walkers.
//! - **u16 semi‑planar 4:2:2 / 4:4:4** (`P210`, `P216`, `P410`,
//!   `P416`) — follow‑up. Would reuse the 16‑bit u16 kernel family
//!   from Ship 4b with 4:2:2 / 4:4:4 chroma strides.
//! - **Packed RGB sources** (`Rgb24`, `Bgr24`, `Rgba`, `Bgra`,
//!   `Rgba1010102`, etc.) — follow‑up. Will land as their own
//!   family of `*_to` kernels feeding a new row‑shape subtrait.

mod nv12;
mod nv16;
mod nv21;
mod nv24;
mod nv42;
mod p010;
mod p012;
mod p016;
mod yuv420p;
mod yuv420p10;
mod yuv420p12;
mod yuv420p14;
mod yuv420p16;
mod yuv422p;
mod yuv444p;

pub use nv12::{Nv12, Nv12Row, Nv12Sink, nv12_to};
pub use nv16::{Nv16, Nv16Row, Nv16Sink, nv16_to};
pub use nv21::{Nv21, Nv21Row, Nv21Sink, nv21_to};
pub use nv24::{Nv24, Nv24Row, Nv24Sink, nv24_to};
pub use nv42::{Nv42, Nv42Row, Nv42Sink, nv42_to};
pub use p010::{P010, P010Row, P010Sink, p010_to};
pub use p012::{P012, P012Row, P012Sink, p012_to};
pub use p016::{P016, P016Row, P016Sink, p016_to};
pub use yuv420p::{Yuv420p, Yuv420pRow, Yuv420pSink, yuv420p_to};
pub use yuv420p10::{Yuv420p10, Yuv420p10Row, Yuv420p10Sink, yuv420p10_to};
pub use yuv420p12::{Yuv420p12, Yuv420p12Row, Yuv420p12Sink, yuv420p12_to};
pub use yuv420p14::{Yuv420p14, Yuv420p14Row, Yuv420p14Sink, yuv420p14_to};
pub use yuv420p16::{Yuv420p16, Yuv420p16Row, Yuv420p16Sink, yuv420p16_to};
pub use yuv422p::{Yuv422p, Yuv422pRow, Yuv422pSink, yuv422p_to};
pub use yuv444p::{Yuv444p, Yuv444pRow, Yuv444pSink, yuv444p_to};
