//! Per‑row packed RGB → planar luma throughput baseline.
//!
//! The public dispatcher is `rgb_to_luma_row` (also covers `RGB24`,
//! FFmpeg `AV_PIX_FMT_RGB24`). Bench mirrors `rgb_to_hsv` — packed-in
//! / planar-out shape — but writes 1 luma byte/pixel instead of 3
//! HSV bytes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{KernelMatrix, bench_internals::rgb_to_luma_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("rgb24_to_luma_row");

  for &w in WIDTHS {
    let mut rgb = std::vec![0u8; w * 3];
    fill_pseudo_random(&mut rgb, 0x1111);
    let mut luma = std::vec![0u8; w];

    // Throughput in output bytes (1 byte/pixel).
    group.throughput(Throughput::Bytes(w as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgb_to_luma_row(
            black_box(&rgb),
            black_box(&mut luma),
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
