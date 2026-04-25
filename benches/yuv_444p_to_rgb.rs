//! Per-row YUV 4:4:4 planar → packed RGB throughput baseline.
//!
//! 4:4:4 has its own per-row kernel family — chroma is 1:1 with Y
//! (no horizontal duplication step), so the main loop is a touch
//! simpler than 4:2:0. Native SIMD on every backend (NEON on
//! aarch64; SSE4.1 / AVX2 / AVX-512 on x86_64; wasm simd128).
//! Scalar fallback when SIMD is unavailable or disabled via
//! `use_simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::yuv_444_to_rgb_row};

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

  let mut group = c.benchmark_group("yuv_444p_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    let mut u = std::vec![0u8; w];
    let mut v = std::vec![0u8; w];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut u, 0x2222);
    fill_pseudo_random(&mut v, 0x3333);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv_444_to_rgb_row(
            black_box(&y),
            black_box(&u),
            black_box(&v),
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
