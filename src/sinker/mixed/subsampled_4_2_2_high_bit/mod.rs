//! High-bit-depth 4:2:2 / 4:4:0 `MixedSinker` impls, split per
//! sub-format so no single source exceeds ~1.5 KLoC: Yuv422p9/10/12/14/16
//! + Yuv440p10/12 + P210/P212/P216.

mod p2xx;
mod yuv422p;
mod yuv440p;
