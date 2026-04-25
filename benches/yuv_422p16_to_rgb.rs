//! Per-row YUV 4:2:2 16-bit planar → packed RGB throughput baseline.
//!
//! Yuv422p16 shares its per-row kernel with Yuv420p16 (parallel i64
//! family — see `yuv/mod.rs` kernel-families notes). 4:2:2 only
//! changes the vertical walker.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{yuv420p16_to_rgb_row, yuv420p16_to_rgb_u16_row},
};

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group_u8 = c.benchmark_group("yuv422p16_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut u = std::vec![0u16; w / 2];
    let mut v = std::vec![0u16; w / 2];
    fill_pseudo_random_u16(&mut y, 0x1111);
    fill_pseudo_random_u16(&mut u, 0x2222);
    fill_pseudo_random_u16(&mut v, 0x3333);
    let mut rgb = std::vec![0u8; w * 3];

    group_u8.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "u8_simd" } else { "u8_scalar" };
      group_u8.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv420p16_to_rgb_row(
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
  group_u8.finish();

  let mut group_u16 = c.benchmark_group("yuv422p16_to_rgb_u16_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut u = std::vec![0u16; w / 2];
    let mut v = std::vec![0u16; w / 2];
    fill_pseudo_random_u16(&mut y, 0x1111);
    fill_pseudo_random_u16(&mut u, 0x2222);
    fill_pseudo_random_u16(&mut v, 0x3333);
    let mut rgb = std::vec![0u16; w * 3];

    group_u16.throughput(Throughput::Bytes((w * 3 * 2) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "u16_simd" } else { "u16_scalar" };
      group_u16.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv420p16_to_rgb_u16_row(
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
  group_u16.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
