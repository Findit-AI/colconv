//! Per‑row P010 (semi‑planar 4:2:0, 10‑bit, high‑bit‑packed) → RGB
//! throughput baseline.
//!
//! Mirrors [`yuv_420p10_to_rgb`] — two output paths per width:
//! - `u8_*` — P010 → packed 8‑bit RGB (hot path for scene / keyframe
//!   detection).
//! - `u16_*` — P010 → native‑depth 10‑bit RGB in `u16` storage
//!   (lossless, for HDR tone mapping).
//!
//! Each width gets a `scalar` vs `simd` pair so the SIMD speedup on
//! whichever backend the dispatcher selects is a two‑line comparison
//! in the Criterion report.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{p010_to_rgb_row, p010_to_rgb_u16_row},
};

/// Fills a `u16` buffer with a deterministic P010‑packed pseudo‑random
/// sequence — 10‑bit values shifted into the high 10 bits of each
/// `u16` (low 6 bits zero), matching the real P010 storage layout.
fn fill_pseudo_random_p010(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (((state >> 8) & 0x3FF) as u16) << 6;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K widths — multiples of 64 so the widest SIMD
  // tier (AVX‑512, 64 pixels per iteration) covers each block fully.
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  // ---- u8 output ------------------------------------------------------
  let mut group_u8 = c.benchmark_group("p010_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    // UV row payload is `width` u16 elements (w / 2 interleaved pairs).
    let mut uv = std::vec![0u16; w];
    fill_pseudo_random_p010(&mut y, 0x1111);
    fill_pseudo_random_p010(&mut uv, 0x2222);
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

  // ---- u16 native-depth output ----------------------------------------
  let mut group_u16 = c.benchmark_group("p010_to_rgb_u16_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u16; w];
    let mut uv = std::vec![0u16; w];
    fill_pseudo_random_p010(&mut y, 0x1111);
    fill_pseudo_random_p010(&mut uv, 0x2222);
    let mut rgb = std::vec![0u16; w * 3];

    // u16 output writes 2× the bytes of u8.
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
