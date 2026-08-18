//! Per‑frame Gray8 → packed RGB throughput baseline.
//!
//! The per-row dispatcher for `Gray8` is `pub(crate)`; the public
//! surface is the walker `gray8_to` + a `MixedSinker<Gray8>`. We bench
//! through that walker on a multi-row frame so per-frame sinker setup
//! is amortized.
//!
//! Two variants per width — `simd=true` and `simd=false`. Gray sources
//! broadcast luma → R = G = B (and α = `0xFF` for RGBA).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::Gray8Frame,
  sinker::MixedSinker,
  source::{Gray8, gray8_to},
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

  let mut group = c.benchmark_group("gray8_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let mut y = std::vec![0u8; w_us * h_us];
    fill_pseudo_random(&mut y, 0x1111);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Gray8Frame::new(black_box(&y), w, FRAME_HEIGHT, w);
          let mut sinker = MixedSinker::<Gray8>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          gray8_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
