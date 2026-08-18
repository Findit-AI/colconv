//! Per‑row XV36 (packed YUV 4:4:4 12‑bit, four `u16` elements per pixel
//! `X U Y V` with 12 bits active in each u16) → packed RGB throughput
//! baseline.
//!
//! XV36 reserves one channel as padding (the leading X slot) plus full
//! 12‑bit U/Y/V — so 8 bytes per pixel (4 u16 elements). Two variants
//! per width — `simd=true` (NEON on aarch64; native SSE4.1 / AVX2 /
//! AVX‑512 on x86_64; wasm‑simd128) and `simd=false`. We benchmark the
//! LE wire path (`be_input = false`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{KernelMatrix, bench_internals::xv36_to_rgb_row};

/// Fills a `u16` buffer with deterministic XV36‑packed pseudo‑random
/// 12‑bit samples (low 12 bits set, high 4 bits zero — kernel masks
/// at load time so any pattern is valid but staying in-range matches
/// real-world streams).
fn fill_pseudo_random_xv36(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) & 0xFFF) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;
  const BE_INPUT: bool = false;

  let mut group = c.benchmark_group("xv36_to_rgb_row");

  for &w in WIDTHS {
    // XV36: 4 u16 elements per pixel (X, U, Y, V).
    let mut packed = std::vec![0u16; w * 4];
    fill_pseudo_random_xv36(&mut packed, 0x7777);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          xv36_to_rgb_row(
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
