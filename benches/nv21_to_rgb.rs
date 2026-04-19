//! Per‑row NV21 (semi‑planar 4:2:0, VU-ordered) → packed RGB bench.
//!
//! Structurally identical to the NV12 bench — same input fixture,
//! same widths, same tier layout. The diff versus `nv12_to_rgb`
//! isolates the cost of the one-line U/V lane swap in each SIMD
//! backend (which should be ~zero — the const generic makes it
//! compile-time-resolved).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::nv21_to_rgb_row};

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

  let mut group = c.benchmark_group("nv21_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    // VU row payload is `width` bytes — same length as NV12's UV row.
    let mut vu = std::vec![0u8; w];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut vu, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          nv21_to_rgb_row(
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
