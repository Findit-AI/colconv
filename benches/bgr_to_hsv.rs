//! Per‑row BGR → planar HSV throughput baseline.
//!
//! HSV has no SIMD backend yet, so there is only a scalar path for
//! now. The bench is structured to match
//! [`yuv_420_to_bgr`](./yuv_420_to_bgr.rs): when an HSV SIMD backend
//! lands, flip to a two‑variant loop (`scalar` / `simd`) and
//! regression numbers stay comparable to today's baseline.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::row::bgr_to_hsv_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("bgr_to_hsv_row");

  for &w in WIDTHS {
    let mut bgr = std::vec![0u8; w * 3];
    fill_pseudo_random(&mut bgr, 0x4444);
    let mut h = std::vec![0u8; w];
    let mut s = std::vec![0u8; w];
    let mut v = std::vec![0u8; w];

    // Throughput in HSV output bytes (3 planes × width) — matches the
    // YUV→BGR bench so MB/s figures are apples to apples.
    group.throughput(Throughput::Bytes((w * 3) as u64));

    group.bench_with_input(BenchmarkId::new("scalar", w), &w, |b, &w| {
      b.iter(|| {
        bgr_to_hsv_row(
          black_box(&bgr),
          black_box(&mut h),
          black_box(&mut s),
          black_box(&mut v),
          w,
        );
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
