//! Per‑row GBRP (planar GBR 8‑bit) → packed RGB throughput baseline.
//!
//! GBRP stores G, B, R in three separate full-width planes (FFmpeg
//! `AV_PIX_FMT_GBRP`); kernel reads `(g, b, r)` planes, interleaves
//! into packed RGB output. Two variants per width — `simd=true` and
//! `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::gbr_to_rgb_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("gbrp_to_rgb_row");

  for &w in WIDTHS {
    let mut g = std::vec![0u8; w];
    let mut b_plane = std::vec![0u8; w];
    let mut r = std::vec![0u8; w];
    fill_pseudo_random(&mut g, 0xAAAA);
    fill_pseudo_random(&mut b_plane, 0xBBBB);
    fill_pseudo_random(&mut r, 0xCCCC);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          gbr_to_rgb_row(
            black_box(&g),
            black_box(&b_plane),
            black_box(&r),
            black_box(&mut rgb),
            w,
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
