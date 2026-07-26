// Tests reference `mediaframe::frame::{WhiteBalance, ColorCorrectionMatrix,
// fuse_wb_ccm, ...}` which only exist when mediaframe's `bayer` feature is
// enabled (pixon's `bayer` feature passes through to it). Without the gate,
// `cargo test --no-default-features` would try to import names that don't
// exist in the mediaframe build it's linking against.
#[cfg(all(test, feature = "bayer"))]
mod tests;
