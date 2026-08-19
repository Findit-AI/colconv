//! Per‑row VUYX (packed YUV 4:4:4 8‑bit, `V U Y X` quadruples — α is
//! padding) → packed RGBA throughput baseline.
//!
//! VUYX is the FFmpeg `AV_PIX_FMT_VUYX` layout — siblings VUYA (with
//! source α) already has an A+ combo bench; VUYX is the α-as-padding
//! variant where the high byte is ignored on the input side and the
//! output α is forced opaque. Public per-row dispatcher is
//! `vuyx_to_rgba_row` (no plain RGB path — the kernel is built to
//! write 4 bytes/pixel with α = `0xFF`).
//!
//! Two variants per width — `simd=true` (NEON on aarch64; native
//! SSE4.1 / AVX2 / AVX‑512 on x86_64; wasm‑simd128) and `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{KernelMatrix, bench_internals::vuyx_to_rgba_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("vuyx_to_rgba_row");

  for &w in WIDTHS {
    // VUYX: 4 bytes per pixel (V, U, Y, padding).
    let mut packed = std::vec![0u8; w * 4];
    fill_pseudo_random(&mut packed, 0x8888);
    let mut rgba = std::vec![0u8; w * 4];

    group.throughput(Throughput::Bytes((w * 4) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          vuyx_to_rgba_row(
            black_box(&packed),
            black_box(&mut rgba),
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
