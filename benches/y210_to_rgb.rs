//! Per‑row Y210 (packed YUV 4:2:2 10‑bit, MSB‑aligned `u16` quadruples)
//! → packed RGB throughput baseline.
//!
//! Y210 carries the 10 active bits in the high 10 bits of each `u16`
//! sample (low 6 bits zero); the kernel loads 4 `u16` elements per
//! pixel pair `(Y0 U Y1 V)`.
//!
//! Two variants per width — `simd=true` (NEON on aarch64; native
//! SSE4.1 / AVX2 / AVX‑512 on x86_64; wasm‑simd128) and `simd=false`
//! (forced scalar reference).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{KernelMatrix, bench_internals::y210_to_rgb_row};

/// Fills a `u16` buffer with deterministic Y210‑packed pseudo‑random
/// samples — 10‑bit values shifted into the high 10 bits of each
/// `u16` (low 6 bits zero), matching the real Y210 storage layout.
fn fill_pseudo_random_y210(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (((state >> 8) & 0x3FF) as u16) << 6;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K — all even (Y210 requires even width).
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("y210_to_rgb_row");

  for &w in WIDTHS {
    // Y210 packs 4 u16 elements per 2 pixels — `width * 2` u16 per row.
    let mut packed = std::vec![0u16; w * 2];
    fill_pseudo_random_y210(&mut packed, 0x1111);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          y210_to_rgb_row(
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
