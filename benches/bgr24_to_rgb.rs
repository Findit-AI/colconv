//! Per‑row packed BGR → packed RGB throughput baseline.
//!
//! BGR24 is the FFmpeg `AV_PIX_FMT_BGR24` channel-swap sibling of RGB24.
//! The public dispatcher is `bgr_to_rgb_row` (R/B swap). Two variants
//! per width — `simd=true` and `simd=false` — reveal the byte-shuffle
//! win on each backend.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::bgr_to_rgb_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("bgr24_to_rgb_row");

  for &w in WIDTHS {
    let mut bgr = std::vec![0u8; w * 3];
    fill_pseudo_random(&mut bgr, 0x3333);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          bgr_to_rgb_row(black_box(&bgr), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
