//! Per‑row v410 (packed YUV 4:4:4 10‑bit, one `u32` per pixel —
//! Apple/QuickTime variant) → packed RGB throughput baseline.
//!
//! v410 stores 10‑bit U/Y/V samples in one `u32` per pixel (with 2
//! padding bits), similar to v30x but with a different field order
//! and endian-aware unpacking.
//!
//! Two variants per width — `simd=true` and `simd=false`. We benchmark
//! the LE wire path (`be_input = false`) which matches the common
//! Apple/QuickTime input.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{KernelMatrix, bench_internals::v410_to_rgb_row};

fn fill_pseudo_random_u32(buf: &mut [u32], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = state;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;
  // LE wire — matches FFmpeg `AV_PIX_FMT_Y410` / QuickTime `v410` default
  // (packed 4:4:4 10-bit YUV, one u32 per pixel).
  const BE_INPUT: bool = false;

  let mut group = c.benchmark_group("v410_to_rgb_row");

  for &w in WIDTHS {
    let mut packed = std::vec![0u32; w];
    fill_pseudo_random_u32(&mut packed, 0x6666);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          v410_to_rgb_row(
            black_box(&packed),
            black_box(&mut rgb),
            w,
            MATRIX,
            FULL_RANGE,
            use_simd,
            BE_INPUT,
          );
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
