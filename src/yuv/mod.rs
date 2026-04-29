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
//! # Shipped (8-bit 4:2:2 / 4:4:0 / 4:4:4)
//!
//! - [`Yuv422p`](crate::yuv::Yuv422p) — 4:2:2 **planar** (libx264
//!   default for chroma‑rich captures, ProRes Proxy at 8 bits).
//!   Reuses the 4:2:0 per‑row kernel; differs only in the vertical
//!   walker.
//! - [`Nv16`](crate::yuv::Nv16) — 4:2:2 semi‑planar, UV‑ordered.
//!   Reuses [`Nv12`](crate::yuv::Nv12)'s per‑row kernel; the 4:2:0
//!   vs 4:2:2 difference is purely in the vertical walker.
//! - [`Yuv440p`](crate::yuv::Yuv440p) — 4:4:0 planar (full‑width
//!   chroma × half‑height — axis‑flipped 4:2:2). Mostly seen from
//!   JPEG decoders that subsample vertically only. Reuses
//!   [`Yuv444p`](crate::yuv::Yuv444p)'s per‑row kernel; only the
//!   walker reads chroma row `r / 2`.
//! - [`Yuv444p`](crate::yuv::Yuv444p) — 4:4:4 **planar** (libx264
//!   default for screen capture / RGB‑source re‑encodes). Dedicated
//!   kernel family — chroma is 1:1 with Y, no duplication step.
//! - [`Nv24`](crate::yuv::Nv24) — 4:4:4 semi‑planar, UV‑ordered.
//!   Dedicated kernel family (chroma is 1:1 with Y, no
//!   duplication step).
//! - [`Nv42`](crate::yuv::Nv42) — 4:4:4 semi‑planar, **VU**‑ordered.
//!   Shares kernels with [`Nv24`](crate::yuv::Nv24) via a `SWAP_UV`
//!   const generic, the same way [`Nv21`](crate::yuv::Nv21) pairs
//!   with [`Nv12`](crate::yuv::Nv12).
//!
//! # Shipped (high-bit-depth 4:2:0 / 4:2:2 / 4:4:0 / 4:4:4, low-bit-packed planar)
//!
//! - [`Yuv420p9`](crate::yuv::Yuv420p9) /
//!   [`Yuv422p9`](crate::yuv::Yuv422p9) /
//!   [`Yuv444p9`](crate::yuv::Yuv444p9) — 9 bits per sample (AVC High
//!   9 profile only — niche; HEVC / VP9 / AV1 don't produce 9‑bit).
//!   Const‑generic kernel reuse at `BITS = 9`.
//! - [`Yuv420p10`](crate::yuv::Yuv420p10) — 4:2:0 planar at 10 bits
//!   per sample (HDR10 / 10‑bit SDR software decode).
//! - [`Yuv420p12`](crate::yuv::Yuv420p12) — 4:2:0 planar at 12 bits
//!   per sample (HEVC Main 12 / VP9 Profile 3 software decode).
//! - [`Yuv420p14`](crate::yuv::Yuv420p14) — 4:2:0 planar at 14 bits
//!   per sample (grading / mastering pipelines).
//! - [`Yuv420p16`](crate::yuv::Yuv420p16) — 4:2:0 planar at 16 bits
//!   per sample (reference / intermediate HDR, runs on the parallel
//!   i64 kernel family).
//! - [`Yuv422p10`](crate::yuv::Yuv422p10) /
//!   [`Yuv422p12`](crate::yuv::Yuv422p12) /
//!   [`Yuv422p14`](crate::yuv::Yuv422p14) /
//!   [`Yuv422p16`](crate::yuv::Yuv422p16) — 4:2:2 planar at 10 / 12 /
//!   14 / 16 bits (ProRes 422 LT/HQ, DNxHD/HR). Reuses the 4:2:0
//!   per‑row kernels at the corresponding `BITS`.
//! - [`Yuv440p10`](crate::yuv::Yuv440p10) /
//!   [`Yuv440p12`](crate::yuv::Yuv440p12) — 4:4:0 planar at 10 / 12
//!   bits. Reuses the 4:4:4 const‑generic kernel family; only the
//!   walker reads chroma row `r / 2`. (No 9 / 14 / 16‑bit variants
//!   exist in FFmpeg.)
//! - [`Yuv444p10`](crate::yuv::Yuv444p10) /
//!   [`Yuv444p12`](crate::yuv::Yuv444p12) /
//!   [`Yuv444p14`](crate::yuv::Yuv444p14) — 4:4:4 planar at 10 / 12 /
//!   14 bits (ProRes 4444 / 4444 XQ, mastering pipelines).
//! - [`Yuv444p16`](crate::yuv::Yuv444p16) — 4:4:4 planar at 16 bits
//!   per sample (NVDEC / CUDA 4:4:4 HDR download target). Runs on
//!   the parallel i64 kernel family.
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
//! # Shipped (high-bit-depth 4:2:2 / 4:4:4, high-bit-packed semi-planar)
//!
//! - [`P210`](crate::yuv::P210) /
//!   [`P212`](crate::yuv::P212) /
//!   [`P216`](crate::yuv::P216) — 4:2:2 semi‑planar at 10 / 12 / 16
//!   bits. Reuses the 4:2:0 P‑family per‑row kernels verbatim
//!   (half‑width interleaved UV layout is identical); only the walker
//!   reads chroma row `r` instead of `r / 2`. NVDEC / CUDA HDR 4:2:2
//!   download targets.
//! - [`P410`](crate::yuv::P410) /
//!   [`P412`](crate::yuv::P412) /
//!   [`P416`](crate::yuv::P416) — 4:4:4 semi‑planar at 10 / 12 / 16
//!   bits. Full‑width interleaved UV (`2 * width` u16 elements per
//!   row, one `U, V` pair per pixel). Dedicated row‑kernel family
//!   `p_n_444_to_rgb_*<BITS>` + `p_n_444_16_to_rgb_*`. NVDEC / CUDA
//!   HDR 4:4:4 download target.
//!
//! # Kernel families
//!
//! - **Q15 i32 family** covers 8‑bit (non-generic `yuv_420_to_rgb_row` + siblings)
//!   and 9/10/12/14‑bit (const-generic `yuv_420p_n_to_rgb_*<BITS>`
//!   and `yuv_444p_n_to_rgb_*<BITS>` + siblings). Hot path for SDR + most HDR workflows.
//! - **i64 chroma-widened family** covers 16‑bit
//!   (`yuv_420p16_to_rgb_*` + `yuv_444p16_to_rgb_*` +
//!   `p16_to_rgb_*`). The chroma matrix multiply
//!   `c_u * u_d + c_v * v_d` overflows i32 at 16 bits, so the 16‑bit
//!   kernels widen that specific step to i64 and narrow back after
//!   the `>> 15`. Scalar stays free; SIMD pays a ~2× chroma compute
//!   tax in exchange for i32 overflow safety.
//!
//! # Shipped (packed RGB sources)
//!
//! - [`Rgb24`](crate::yuv::Rgb24) — packed `R, G, B` 8‑bit (3 bytes
//!   per pixel), single plane. Source-side feed for callers that
//!   already hold packed RGB and want HSV / luma / RGBA via the
//!   standard `MixedSinker` channels (Ship 9a).
//! - [`Bgr24`](crate::yuv::Bgr24) — packed `B, G, R` 8‑bit. Reuses
//!   [`Rgb24`](crate::yuv::Rgb24)'s sink pipeline behind a
//!   `bgr_to_rgb_row` swap into the existing `rgb_scratch` buffer.
//! - [`Rgba`] — packed `R, G, B, A` 8‑bit (4 bytes per pixel), single
//!   plane; alpha is real (not padding) and is passed through to RGBA
//!   output (Ship 9b).
//! - [`Bgra`] — packed `B, G, R, A` 8‑bit. Channel order swapped on
//!   the first three bytes vs [`Rgba`]; alpha lane preserved (Ship 9b).
//! - [`Argb`] — packed `A, R, G, B` 8‑bit. Same payload as [`Rgba`]
//!   with alpha at the **leading** position; sinker rotates alpha to
//!   trailing for `with_rgba` output (Ship 9c).
//! - [`Abgr`] — packed `A, B, G, R` 8‑bit. Leading alpha + reversed
//!   RGB order vs [`Argb`]; sinker performs a full byte reverse for
//!   `with_rgba` output (Ship 9c).
//! - [`Xrgb`] / [`Rgbx`] / [`Xbgr`] / [`Bgrx`] — 4-byte packed RGB
//!   with one **ignored padding byte** at the leading or trailing
//!   position (FFmpeg `0rgb` / `rgb0` / `0bgr` / `bgr0`). The padding
//!   byte's value is undefined on read; `with_rgba` output forces
//!   alpha to `0xFF` rather than passing through (Ship 9d).
//! - [`X2Rgb10`] / [`X2Bgr10`] — 10-bit packed RGB (FFmpeg
//!   `X2RGB10LE` / `X2BGR10LE`). Each pixel is a 32-bit
//!   little-endian word with `(MSB) 2X | 10R | 10G | 10B (LSB)`
//!   (or BGR-ordered). The 2-bit field is **padding**, not real
//!   alpha — `with_rgba` forces alpha to `0xFF`. Both u8 outputs
//!   (down-shifted 10→8) and native u16 outputs (`with_rgb_u16`,
//!   value range `[0, 1023]`) are supported (Ship 9e — closes
//!   Tier 6).
//!
//! # Not yet shipped
//!
//! - **Legacy planar** (`Yuv411p`, `Yuv410p`) — DV / Cinepak only;
//!   uncommon enough that adding them would be speculative.
//! - **Big-endian 10-bit packed RGB** (`X2RGB10BE` / `X2BGR10BE`).
//!   Most modern systems are LE; BE can be added as a thin wrapper
//!   over the LE kernel (byte-swap on read) when a caller needs it.
//!
//! # Tracked refactor (no behavior change)
//!
//! Every walker module below follows the same per‑format pattern:
//! marker → `Row` struct → `Sink` subtrait → `*_to` walker fn. The
//! walker bodies are ~85% duplication across the ~30 modules; only
//! the per‑row chroma slice length, the chroma‑row index expression,
//! and the `Row::new(...)` call vary. A `walker!` macro expanding
//! the boilerplate from a small spec would consolidate the family.
//!
//! Deferred because doing it incrementally creates asymmetry
//! (some walkers macro‑expanded, others hand‑written). The right
//! shape is a single all‑walkers‑refactored PR with zero behavioral
//! change — easy to review on its own merits, unrelated to any
//! pending format‑shipping work. See `docs/color-conversion-functions.md`
//! § "Cleanup follow‑ups → Walker module deduplication" for the full
//! discussion (originated from PR #14 review).

mod abgr;
mod argb;
mod bgr24;
mod bgra;
mod bgrx;
mod nv12;
mod nv16;
mod nv21;
mod nv24;
mod nv42;
mod p010;
mod p012;
mod p016;
mod p210;
mod p212;
mod p216;
mod p410;
mod p412;
mod p416;
mod rgb24;
mod rgba;
mod rgbx;
mod uyvy422;
mod x2bgr10;
mod x2rgb10;
mod xbgr;
mod xrgb;
mod yuv420p;
mod yuv420p10;
mod yuv420p12;
mod yuv420p14;
mod yuv420p16;
mod yuv420p9;
mod yuv422p;
mod yuv422p10;
mod yuv422p12;
mod yuv422p14;
mod yuv422p16;
mod yuv422p9;
mod yuv440p;
mod yuv440p10;
mod yuv440p12;
mod yuv444p;
mod yuv444p10;
mod yuv444p12;
mod yuv444p14;
mod yuv444p16;
mod yuv444p9;
mod yuva420p;
mod yuva420p10;
mod yuva420p16;
mod yuva420p9;
mod yuva422p;
mod yuva422p10;
mod yuva422p12;
mod yuva422p16;
mod yuva422p9;
mod yuva444p;
mod yuva444p10;
mod yuva444p12;
mod yuva444p14;
mod yuva444p16;
mod yuva444p9;
mod yuyv422;
mod yvyu422;

pub use abgr::{Abgr, AbgrRow, AbgrSink, abgr_to};
pub use argb::{Argb, ArgbRow, ArgbSink, argb_to};
pub use bgr24::{Bgr24, Bgr24Row, Bgr24Sink, bgr24_to};
pub use bgra::{Bgra, BgraRow, BgraSink, bgra_to};
pub use bgrx::{Bgrx, BgrxRow, BgrxSink, bgrx_to};
pub use nv12::{Nv12, Nv12Row, Nv12Sink, nv12_to};
pub use nv16::{Nv16, Nv16Row, Nv16Sink, nv16_to};
pub use nv21::{Nv21, Nv21Row, Nv21Sink, nv21_to};
pub use nv24::{Nv24, Nv24Row, Nv24Sink, nv24_to};
pub use nv42::{Nv42, Nv42Row, Nv42Sink, nv42_to};
pub use p010::{P010, P010Row, P010Sink, p010_to};
pub use p012::{P012, P012Row, P012Sink, p012_to};
pub use p016::{P016, P016Row, P016Sink, p016_to};
pub use p210::{P210, P210Row, P210Sink, p210_to};
pub use p212::{P212, P212Row, P212Sink, p212_to};
pub use p216::{P216, P216Row, P216Sink, p216_to};
pub use p410::{P410, P410Row, P410Sink, p410_to};
pub use p412::{P412, P412Row, P412Sink, p412_to};
pub use p416::{P416, P416Row, P416Sink, p416_to};
pub use rgb24::{Rgb24, Rgb24Row, Rgb24Sink, rgb24_to};
pub use rgba::{Rgba, RgbaRow, RgbaSink, rgba_to};
pub use rgbx::{Rgbx, RgbxRow, RgbxSink, rgbx_to};
pub use uyvy422::{Uyvy422, Uyvy422Row, Uyvy422Sink, uyvy422_to};
pub use x2bgr10::{X2Bgr10, X2Bgr10Row, X2Bgr10Sink, x2bgr10_to};
pub use x2rgb10::{X2Rgb10, X2Rgb10Row, X2Rgb10Sink, x2rgb10_to};
pub use xbgr::{Xbgr, XbgrRow, XbgrSink, xbgr_to};
pub use xrgb::{Xrgb, XrgbRow, XrgbSink, xrgb_to};
pub use yuv420p::{Yuv420p, Yuv420pRow, Yuv420pSink, yuv420p_to};
pub use yuv420p9::{Yuv420p9, Yuv420p9Row, Yuv420p9Sink, yuv420p9_to};
pub use yuv420p10::{Yuv420p10, Yuv420p10Row, Yuv420p10Sink, yuv420p10_to};
pub use yuv420p12::{Yuv420p12, Yuv420p12Row, Yuv420p12Sink, yuv420p12_to};
pub use yuv420p14::{Yuv420p14, Yuv420p14Row, Yuv420p14Sink, yuv420p14_to};
pub use yuv420p16::{Yuv420p16, Yuv420p16Row, Yuv420p16Sink, yuv420p16_to};
pub use yuv422p::{Yuv422p, Yuv422pRow, Yuv422pSink, yuv422p_to};
pub use yuv422p9::{Yuv422p9, Yuv422p9Row, Yuv422p9Sink, yuv422p9_to};
pub use yuv422p10::{Yuv422p10, Yuv422p10Row, Yuv422p10Sink, yuv422p10_to};
pub use yuv422p12::{Yuv422p12, Yuv422p12Row, Yuv422p12Sink, yuv422p12_to};
pub use yuv422p14::{Yuv422p14, Yuv422p14Row, Yuv422p14Sink, yuv422p14_to};
pub use yuv422p16::{Yuv422p16, Yuv422p16Row, Yuv422p16Sink, yuv422p16_to};
pub use yuv440p::{Yuv440p, Yuv440pRow, Yuv440pSink, yuv440p_to};
pub use yuv440p10::{Yuv440p10, Yuv440p10Row, Yuv440p10Sink, yuv440p10_to};
pub use yuv440p12::{Yuv440p12, Yuv440p12Row, Yuv440p12Sink, yuv440p12_to};
pub use yuv444p::{Yuv444p, Yuv444pRow, Yuv444pSink, yuv444p_to};
pub use yuv444p9::{Yuv444p9, Yuv444p9Row, Yuv444p9Sink, yuv444p9_to};
pub use yuv444p10::{Yuv444p10, Yuv444p10Row, Yuv444p10Sink, yuv444p10_to};
pub use yuv444p12::{Yuv444p12, Yuv444p12Row, Yuv444p12Sink, yuv444p12_to};
pub use yuv444p14::{Yuv444p14, Yuv444p14Row, Yuv444p14Sink, yuv444p14_to};
pub use yuv444p16::{Yuv444p16, Yuv444p16Row, Yuv444p16Sink, yuv444p16_to};
pub use yuva420p::{Yuva420p, Yuva420pRow, Yuva420pSink, yuva420p_to};
pub use yuva420p9::{Yuva420p9, Yuva420p9Row, Yuva420p9Sink, yuva420p9_to};
pub use yuva420p10::{Yuva420p10, Yuva420p10Row, Yuva420p10Sink, yuva420p10_to};
pub use yuva420p16::{Yuva420p16, Yuva420p16Row, Yuva420p16Sink, yuva420p16_to};
pub use yuva422p::{Yuva422p, Yuva422pRow, Yuva422pSink, yuva422p_to};
pub use yuva422p9::{Yuva422p9, Yuva422p9Row, Yuva422p9Sink, yuva422p9_to};
pub use yuva422p10::{Yuva422p10, Yuva422p10Row, Yuva422p10Sink, yuva422p10_to};
pub use yuva422p12::{Yuva422p12, Yuva422p12Row, Yuva422p12Sink, yuva422p12_to};
pub use yuva422p16::{Yuva422p16, Yuva422p16Row, Yuva422p16Sink, yuva422p16_to};
pub use yuva444p::{Yuva444p, Yuva444pRow, Yuva444pSink, yuva444p_to};
pub use yuva444p9::{Yuva444p9, Yuva444p9Row, Yuva444p9Sink, yuva444p9_to};
pub use yuva444p10::{Yuva444p10, Yuva444p10Row, Yuva444p10Sink, yuva444p10_to};
pub use yuva444p12::{Yuva444p12, Yuva444p12Row, Yuva444p12Sink, yuva444p12_to};
pub use yuva444p14::{Yuva444p14, Yuva444p14Row, Yuva444p14Sink, yuva444p14_to};
pub use yuva444p16::{Yuva444p16, Yuva444p16Row, Yuva444p16Sink, yuva444p16_to};
pub use yuyv422::{Yuyv422, Yuyv422Row, Yuyv422Sink, yuyv422_to};
pub use yvyu422::{Yvyu422, Yvyu422Row, Yvyu422Sink, yvyu422_to};
