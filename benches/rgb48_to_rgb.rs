//! Per‑row packed RGB48 (16‑bit RGB) → packed RGB8 throughput baseline.
//!
//! RGB48 stores R/G/B as 16‑bit `u16` elements; the LE wrapper
//! `rgb48_to_rgb_row` clamps + narrows to 8‑bit u8. Two variants per
//! width — `simd=true` and `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::bench_internals::rgb48_to_rgb_row;

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgb48_to_rgb_row");

  for &w in WIDTHS {
    // RGB48: 3 u16 per pixel.
    let mut rgb48 = std::vec![0u16; w * 3];
    fill_pseudo_random_u16(&mut rgb48, 0x4444);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgb48_to_rgb_row(black_box(&rgb48), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
