//! Per‑row P210 (semi‑planar 4:2:2, 10‑bit, high‑bit‑packed) → RGB
//! throughput baseline. Mirrors `p010_to_rgb` — same kernel family,
//! the 4:2:2 walker just reads chroma row `r` instead of `r / 2`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{p010_to_rgb_row, p010_to_rgb_u16_row},
};

fn fill_pseudo_random_p210(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (((state >> 8) & 0x3FF) as u16) << 6;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group_u8 = c.benchmark_group("p210_to_rgb_row");
  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut uv = std::vec![0u16; w];
    fill_pseudo_random_p210(&mut y, 0x1111);
    fill_pseudo_random_p210(&mut uv, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];
    group_u8.throughput(Throughput::Bytes((w * 3) as u64));
    for use_simd in [false, true] {
      let label = if use_simd { "u8_simd" } else { "u8_scalar" };
      group_u8.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          p010_to_rgb_row(
            black_box(&y),
            black_box(&uv),
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

  let mut group_u16 = c.benchmark_group("p210_to_rgb_u16_row");
  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut uv = std::vec![0u16; w];
    fill_pseudo_random_p210(&mut y, 0x1111);
    fill_pseudo_random_p210(&mut uv, 0x2222);
    let mut rgb = std::vec![0u16; w * 3];
    group_u16.throughput(Throughput::Bytes((w * 3 * 2) as u64));
    for use_simd in [false, true] {
      let label = if use_simd { "u16_simd" } else { "u16_scalar" };
      group_u16.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          p010_to_rgb_u16_row(
            black_box(&y),
            black_box(&uv),
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
