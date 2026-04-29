//! High-bit-depth 4:2:0 `MixedSinker` impls, split per sub-format
//! so no single source exceeds ~1.5 KLoC: Yuv420p9/10/12/14/16 +
//! P010/P012/P016.

mod p0xx;
mod yuv420p;
