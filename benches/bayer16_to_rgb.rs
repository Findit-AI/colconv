//! Per‑row Bayer16 (low‑packed 10/12/14/16‑bit Bayer) → packed RGB
//! throughput baseline.
//!
//! Same 3-row demosaic stencil as `bayer_to_rgb_row`, with u16 inputs.
//! We bench `BITS = 12` as a representative — kernel shape is identical
//! across the 10/12/14/16 specializations (only the input-to-u8 rescale
//! constant differs).
//!
//! **Note:** `use_simd` is currently a **no-op** for all Bayer paths.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  bench_internals::bayer16_to_rgb_row,
  raw::{BayerDemosaic, BayerPattern},
};

/// Fills a `u16` buffer with deterministic low-packed 12-bit pseudo-random
/// samples (low 12 bits set, high 4 bits zero — matches `bayer_bggr12le`,
/// `bayer_rggb12le`, etc.).
fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) & 0xFFF) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const PATTERN: BayerPattern = BayerPattern::Rggb;
  const DEMOSAIC: BayerDemosaic = BayerDemosaic::Bilinear;
  const M: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

  let mut group = c.benchmark_group("bayer16_to_rgb_row");

  for &w in WIDTHS {
    let mut above = std::vec![0u16; w];
    let mut mid = std::vec![0u16; w];
    let mut below = std::vec![0u16; w];
    fill_pseudo_random_u16(&mut above, 0x8888);
    fill_pseudo_random_u16(&mut mid, 0x9999);
    fill_pseudo_random_u16(&mut below, 0xAAAA);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        b.iter(|| {
          // BITS = 12 — representative of the 10/12/14/16 family.
          // LE wire order — the historical bench path uses the
          // single-const LE public dispatcher.
          bayer16_to_rgb_row::<12>(
            black_box(&above),
            black_box(&mid),
            black_box(&below),
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
