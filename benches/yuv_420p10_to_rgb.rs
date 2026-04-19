//! Per‑row YUV 4:2:0 10‑bit → packed RGB throughput baseline.
//!
//! Three variants per width:
//! - `u8_simd` / `u8_scalar` — native‑SIMD vs scalar on the u8 output
//!   path (analogous to the 8‑bit bench).
//! - `u16_simd` / `u16_scalar` — same pair for the native‑depth u16
//!   output path. The u16 path writes 2× the bytes so the MB/s
//!   figure is comparable only within the u16 column.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{yuv420p10_to_rgb_row, yuv420p10_to_rgb_u16_row},
};

/// Fills a `u16` buffer with a deterministic 10‑bit pseudo‑random
/// sequence — values occupy the low 10 bits of each `u16`, matching
/// the storage layout of `yuv420p10le`.
fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) & 0x3FF) as u16;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K — multiples of 64 so the widest backend
  // (AVX‑512, 64 pixels per iteration) covers each fully without tail
  // work. Avoids skewing comparisons across targets.
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  // ---- u8 output ------------------------------------------------------
  let mut group_u8 = c.benchmark_group("yuv420p10_to_rgb_row");

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
          yuv420p10_to_rgb_row(
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

  // ---- u16 native-depth output ----------------------------------------
  let mut group_u16 = c.benchmark_group("yuv420p10_to_rgb_u16_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut u = std::vec![0u16; w / 2];
    let mut v = std::vec![0u16; w / 2];
    fill_pseudo_random_u16(&mut y, 0x1111);
    fill_pseudo_random_u16(&mut u, 0x2222);
    fill_pseudo_random_u16(&mut v, 0x3333);
    let mut rgb = std::vec![0u16; w * 3];

    // u16 output writes 2× the bytes of u8.
    group_u16.throughput(Throughput::Bytes((w * 3 * 2) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "u16_simd" } else { "u16_scalar" };
      group_u16.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv420p10_to_rgb_u16_row(
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
