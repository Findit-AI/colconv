//! Per‑row RGB → planar HSV throughput baseline.
//!
//! Two variants per width — `simd=true` (NEON on aarch64; falls back
//! to scalar on targets without an HSV SIMD backend yet) and
//! `simd=false` (forced scalar).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::row::rgb_to_hsv_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgb_to_hsv_row");

  for &w in WIDTHS {
    let mut rgb = std::vec![0u8; w * 3];
    fill_pseudo_random(&mut rgb, 0x4444);
    let mut h = std::vec![0u8; w];
    let mut s = std::vec![0u8; w];
    let mut v = std::vec![0u8; w];

    // Throughput in HSV output bytes (3 planes × width) — matches the
    // YUV→RGB bench so MB/s figures are apples to apples.
    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgb_to_hsv_row(
            black_box(&rgb),
            black_box(&mut h),
            black_box(&mut s),
            black_box(&mut v),
            w,
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
