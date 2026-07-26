//! Per‑row packed YUV 4:1:1 (UYYVYY411) → packed RGB throughput baseline.
//!
//! UYYVYY411 is the FFmpeg `AV_PIX_FMT_UYYVYY411` layout — `U Y Y V Y Y`
//! repeating, four pixels per 6 bytes (1.5 bytes/pixel). Two variants per
//! width — `simd=true` (NEON on aarch64; native SSE4.1 / AVX2 / AVX‑512 on
//! x86_64; wasm‑simd128 where available) and `simd=false` (forced scalar
//! reference).
//!
//! All three widths are multiples of 4 (the format's chroma block size).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{ColorMatrix, bench_internals::uyyvyy411_to_rgb_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K row widths — all multiples of 4 (4:1:1 quad).
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("uyyvyy411_to_rgb_row");

  for &w in WIDTHS {
    // Packed YUV 4:1:1 row is `width * 3 / 2` bytes (1.5 bytes/pixel).
    let mut packed = std::vec![0u8; w * 3 / 2];
    fill_pseudo_random(&mut packed, 0x4444);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          uyyvyy411_to_rgb_row(
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
