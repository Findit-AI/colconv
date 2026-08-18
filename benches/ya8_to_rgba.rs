//! Per‑frame YA8 (luma + α, 8-bit packed) → packed RGBA throughput
//! baseline.
//!
//! `Ya8` packs `[Y, A, Y, A, ...]` in a single u8 plane (2 bytes/pixel,
//! FFmpeg `AV_PIX_FMT_YA8`). The per-row dispatcher is `pub(crate)`;
//! we bench through the public walker `ya8_to` + `MixedSinker<Ya8>`
//! attached as RGBA.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::Ya8Frame,
  sinker::MixedSinker,
  source::{Ya8, ya8_to},
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

  let mut group = c.benchmark_group("ya8_to_rgba");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    // YA8: 2 bytes per pixel (Y, A).
    let mut packed = std::vec![0u8; w_us * 2 * h_us];
    fill_pseudo_random(&mut packed, 0x4444);

    group.throughput(Throughput::Bytes((w_us * 4 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgba = std::vec![0u8; w_us * 4 * h_us];
        b.iter(|| {
          let frame = Ya8Frame::new(black_box(&packed), w, FRAME_HEIGHT, w * 2);
          let mut sinker = MixedSinker::<Ya8>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgba(&mut rgba)
            .unwrap();
          ya8_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgba);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
