//! YUV 4:4:0 planar (8‑bit) → packed RGB throughput baseline.
//!
//! 4:4:0 is full-width chroma + half-height chroma — vertically subsampled
//! sibling of 4:2:0. It has no per-row dispatcher of its own; the walker
//! routes each Y row through `yuv_444_to_rgb_row` while reading the
//! corresponding half-height chroma row. We bench through the public
//! `yuv440p_to` walker via `MixedSinker` so the bench reflects the
//! production code path (including the chroma-row indexing logic).
//!
//! Multi-row frames (height = `FRAME_HEIGHT`) so per-frame `MixedSinker`
//! setup is amortized across several row dispatches.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::Yuv440pFrame,
  sinker::MixedSinker,
  source::{Yuv440p, yuv440p_to},
};

const FRAME_HEIGHT: u32 = 8;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuv440p_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    // 4:4:0: full-width chroma but half-height (`h.div_ceil(2)`).
    let ch_us = h_us.div_ceil(2);

    let mut y = std::vec![0u8; w_us * h_us];
    let mut u = std::vec![0u8; w_us * ch_us];
    let mut v = std::vec![0u8; w_us * ch_us];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut u, 0x2222);
    fill_pseudo_random(&mut v, 0x3333);

    // Throughput in output bytes (3 bytes/pixel x rows).
    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Yuv440pFrame::new(
            black_box(&y),
            black_box(&u),
            black_box(&v),
            w,
            FRAME_HEIGHT,
            w,
            w,
            w,
          );
          let mut sinker = MixedSinker::<Yuv440p>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          yuv440p_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
