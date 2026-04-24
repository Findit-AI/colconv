//! Per‑row NV42 (semi‑planar 4:4:4, VU‑ordered) → packed RGB
//! throughput baseline.
//!
//! Shares per‑row math with NV24 via the `SWAP_UV` const generic.
//! Two variants per width — `simd=true` / `simd=false` — so the SIMD
//! speedup reads directly off adjacent Criterion lines.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::nv42_to_rgb_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("nv42_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    let mut vu = std::vec![0u8; 2 * w];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut vu, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          nv42_to_rgb_row(
            black_box(&y),
            black_box(&vu),
            black_box(&mut rgb),
            w,
            MATRIX,
            FULL_RANGE,
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
