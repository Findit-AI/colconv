//! Per‑row YUV 4:1:0 planar → packed RGB throughput baseline.
//!
//! 4:1:0 has quarter-width and quarter-height chroma — chroma plane width
//! is `width.div_ceil(4)`. The dispatcher's public entry is
//! `yuv_410_to_rgb_row`. Two variants per width — `simd=true` (NEON on
//! aarch64; native SSE4.1 / AVX2 / AVX‑512 on x86_64; wasm‑simd128) and
//! `simd=false`.
//!
//! All widths are multiples of 4 (4:1:0 chroma stride is `div_ceil(w, 4)`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{ColorMatrix, bench_internals::yuv_410_to_rgb_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuv_410p_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    // 4:1:0 chroma is `width.div_ceil(4)` (quarter-width chroma).
    let cw = w.div_ceil(4);
    let mut u = std::vec![0u8; cw];
    let mut v = std::vec![0u8; cw];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut u, 0x2222);
    fill_pseudo_random(&mut v, 0x3333);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv_410_to_rgb_row(
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

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
