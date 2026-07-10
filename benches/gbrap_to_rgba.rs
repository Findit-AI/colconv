//! Per‑row GBRAP (planar GBRA 8‑bit) → packed RGBA throughput baseline.
//!
//! GBRAP stores G, B, R, A in four separate full-width planes (FFmpeg
//! `AV_PIX_FMT_GBRAP`); kernel reads `(g, b, r, a)` planes, interleaves
//! into packed RGBA output. This is the α‑pass‑through sibling of
//! `gbrp_to_rgb`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::bench_internals::gbra_to_rgba_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("gbrap_to_rgba_row");

  for &w in WIDTHS {
    let mut g = std::vec![0u8; w];
    let mut b_plane = std::vec![0u8; w];
    let mut r = std::vec![0u8; w];
    let mut a = std::vec![0u8; w];
    // Widely-spaced 32-bit seeds so per-plane LCG streams are
    // independent from the first iteration (the LCG diverges
    // quickly even with close seeds, but spaced seeds keep
    // early-iter outputs visibly different across planes).
    fill_pseudo_random(&mut g, 0xDEAD_BEEF);
    fill_pseudo_random(&mut b_plane, 0xCAFE_F00D);
    fill_pseudo_random(&mut r, 0x1234_5678);
    fill_pseudo_random(&mut a, 0xA5A5_5A5A);
    let mut rgba = std::vec![0u8; w * 4];

    group.throughput(Throughput::Bytes((w * 4) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          gbra_to_rgba_row(
            black_box(&g),
            black_box(&b_plane),
            black_box(&r),
            black_box(&a),
            black_box(&mut rgba),
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
