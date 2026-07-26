//! Per‑row v30x (packed YUV 4:4:4 10‑bit, one `u32` per pixel)
//! → packed RGB throughput baseline.
//!
//! v30x carries 4:4:4 10‑bit chroma in a single 32‑bit word per pixel
//! (10 bits each for U/Y/V plus 2 padding bits). `width` u32 elements
//! per row. Two variants per width — `simd=true` (NEON on aarch64;
//! native SSE4.1 / AVX2 / AVX‑512 on x86_64; wasm‑simd128) and
//! `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{ColorMatrix, bench_internals::v30x_to_rgb_row};

/// Fills a `u32` buffer with deterministic v30x‑packed pseudo‑random
/// values. The kernel masks each 10‑bit field at load time, so any
/// 32‑bit pattern is a valid input.
fn fill_pseudo_random_u32(buf: &mut [u32], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = state;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("v30x_to_rgb_row");

  for &w in WIDTHS {
    // v30x: one u32 per pixel.
    let mut packed = std::vec![0u32; w];
    fill_pseudo_random_u32(&mut packed, 0x5555);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          v30x_to_rgb_row(
            black_box(&packed),
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
