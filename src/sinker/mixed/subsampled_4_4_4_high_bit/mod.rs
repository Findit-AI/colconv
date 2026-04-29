//! High-bit-depth 4:4:4 `MixedSinker` impls, split per sub-format
//! so no single source exceeds ~1.5 KLoC: Yuv444p9/10/12/14/16 +
//! P410/P412/P416.

mod p4xx;
mod yuv444p;
