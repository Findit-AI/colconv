use std::env;

fn main() {
  // Don't rerun this on changes other than build.rs.
  println!("cargo:rerun-if-changed=build.rs");

  // Detect cargo-tarpaulin by the env vars it sets at coverage-collection
  // time. There is no `tarpaulin` Cargo feature defined in `Cargo.toml`
  // (the `cfg(tarpaulin)` annotations throughout the crate are set HERE,
  // not via a feature toggle), so the env-var check is the active
  // mechanism.
  println!("cargo:rerun-if-env-changed=CARGO_TARPAULIN");
  println!("cargo:rerun-if-env-changed=CARGO_CFG_TARPAULIN");

  if env::var("CARGO_TARPAULIN").is_ok() || env::var("CARGO_CFG_TARPAULIN").is_ok() {
    println!("cargo:rustc-cfg=tarpaulin");
  }
}
