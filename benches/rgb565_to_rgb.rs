//! Per‑row RGB565 (legacy 5/6/5‑bit packed RGB) → packed RGB8 throughput
//! baseline.
//!
//! RGB565 is FFmpeg `AV_PIX_FMT_RGB565`: 2 bytes per pixel, fields
//! `RRRRRGGG GGGBBBBB`. The kernel unpacks each 16-bit unit, widens
//! the 5/6/5 fields to 8 bits, and writes packed `R, G, B` (3 bytes
//! per pixel).
//!
//! Two variants per width — `simd=true` and `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::rgb565_to_rgb_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgb565_to_rgb_row");

  for &w in WIDTHS {
    // RGB565: 2 bytes per pixel.
    let mut src = std::vec![0u8; w * 2];
    fill_pseudo_random(&mut src, 0x9999);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgb565_to_rgb_row(black_box(&src), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
