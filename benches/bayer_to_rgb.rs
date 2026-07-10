//! Per‑row 8‑bit Bayer → packed RGB throughput baseline.
//!
//! Bayer demosaic is the 3-row-window stencil — kernel reads `above`,
//! `mid`, `below` rows, applies bilinear (or future) demosaic, then
//! multiplies by `m` (precomputed `CCM · diag(wb)`) and writes 8-bit
//! packed RGB.
//!
//! **Note:** `use_simd` is currently a **no-op** for all Bayer paths —
//! the dispatcher always routes to scalar today; per-arch SIMD backends
//! ship in a follow-up. We still toggle the parameter so the bench
//! shape stays consistent with the rest of the suite once SIMD lands.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  bench_internals::bayer_to_rgb_row,
  raw::{BayerDemosaic, BayerPattern},
};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const PATTERN: BayerPattern = BayerPattern::Rggb;
  const DEMOSAIC: BayerDemosaic = BayerDemosaic::Bilinear;
  // Identity matrix — neutral CCM x neutral WB.
  const M: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

  let mut group = c.benchmark_group("bayer_to_rgb_row");

  for &w in WIDTHS {
    let mut above = std::vec![0u8; w];
    let mut mid = std::vec![0u8; w];
    let mut below = std::vec![0u8; w];
    fill_pseudo_random(&mut above, 0x5555);
    fill_pseudo_random(&mut mid, 0x6666);
    fill_pseudo_random(&mut below, 0x7777);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        b.iter(|| {
          bayer_to_rgb_row(
            black_box(&above),
            black_box(&mid),
            black_box(&below),
            // Even row parity (first Bayer pixel position).
            0,
            PATTERN,
            DEMOSAIC,
            black_box(&M),
            black_box(&mut rgb),
            use_simd,
          );
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
