//! Per‑row packed RGBA64 (16‑bit RGBA) → packed RGBA8 throughput baseline.
//!
//! RGBA64 stores R/G/B/A as 16‑bit `u16` elements; the LE wrapper
//! `rgba64_to_rgba_row` clamps + narrows each channel to 8‑bit u8.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::rgba64_to_rgba_row;

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgba64_to_rgba_row");

  for &w in WIDTHS {
    // RGBA64: 4 u16 per pixel.
    let mut rgba64 = std::vec![0u16; w * 4];
    fill_pseudo_random_u16(&mut rgba64, 0x5555);
    let mut rgba = std::vec![0u8; w * 4];

    group.throughput(Throughput::Bytes((w * 4) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgba64_to_rgba_row(black_box(&rgba64), black_box(&mut rgba), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
