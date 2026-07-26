//! Per‑frame Grayf32 → packed RGB throughput baseline.
//!
//! `Grayf32` carries f32 luma in a single plane (`AV_PIX_FMT_GRAYF32LE`).
//! The per-row dispatcher is `pub(crate)`; we bench through the public
//! walker `grayf32_to` + `MixedSinker<Grayf32>` on a multi-row frame.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  ColorMatrix,
  frame::Grayf32Frame,
  sinker::MixedSinker,
  source::{Grayf32, grayf32_to},
};

const FRAME_HEIGHT: u32 = 8;

fn fill_pseudo_random_f32(buf: &mut [f32], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state as f32) / (u32::MAX as f32);
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("grayf32_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let mut y = std::vec![0f32; w_us * h_us];
    fill_pseudo_random_f32(&mut y, 0x3333);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Grayf32Frame::new(black_box(&y), w, FRAME_HEIGHT, w);
          let mut sinker = MixedSinker::<Grayf32>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          grayf32_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
