//! Per‑row v210 (packed YUV 4:2:2 10‑bit, 6 pixels per 16‑byte word)
//! → packed RGB throughput baseline.
//!
//! v210 packs 6 pixels (3 chroma pairs) into 4 LE u32s = 16 bytes — the
//! widest "fold" of any 4:2:2 layout. The row byte count is
//! `div_ceil(width, 6) * 16`. Two variants per width — `simd=true`
//! (NEON on aarch64; native SSE4.1 / AVX2 / AVX‑512 on x86_64;
//! wasm‑simd128) and `simd=false` (forced scalar reference).
//!
//! All widths are even (4:2:2 chroma pair constraint).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{ColorMatrix, bench_internals::v210_to_rgb_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K — all even. v210 packed row stride is
  // `div_ceil(width, 6) * 16` bytes.
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("v210_to_rgb_row");

  for &w in WIDTHS {
    let packed_bytes = w.div_ceil(6) * 16;
    let mut packed = std::vec![0u8; packed_bytes];
    fill_pseudo_random(&mut packed, 0x4444);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          v210_to_rgb_row(
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
