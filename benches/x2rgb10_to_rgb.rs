//! Per‑row X2RGB10 (10-bit RGB in 32-bit packed words) → packed RGB8
//! throughput baseline.
//!
//! X2RGB10 stores 10 bits each of R/G/B plus 2 padding bits in one
//! 32-bit LE word per pixel (FFmpeg `AV_PIX_FMT_X2RGB10LE`). The LE
//! wrapper `x2rgb10_to_rgb_row` reads `[u8]` (4 bytes/pixel) and writes
//! packed u8 RGB. Two variants per width — `simd=true` and
//! `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::bench_internals::x2rgb10_to_rgb_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("x2rgb10_to_rgb_row");

  for &w in WIDTHS {
    // X2RGB10: 4 bytes per pixel (one LE u32).
    let mut packed = std::vec![0u8; w * 4];
    fill_pseudo_random(&mut packed, 0x6666);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          x2rgb10_to_rgb_row(black_box(&packed), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
