//! Per‑frame Gray16 → packed RGB throughput baseline.
//!
//! `Gray16` carries 16-bit luma in a single u16 plane (`AV_PIX_FMT_GRAY16LE`).
//! The per-row dispatcher is `pub(crate)`; we bench through the public
//! walker `gray16_to` + `MixedSinker<Gray16>` on a multi-row frame.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::Gray16Frame,
  sinker::MixedSinker,
  source::{Gray16, gray16_to},
};

const FRAME_HEIGHT: u32 = 8;

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("gray16_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let mut y = std::vec![0u16; w_us * h_us];
    fill_pseudo_random_u16(&mut y, 0x2222);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Gray16Frame::new(black_box(&y), w, FRAME_HEIGHT, w);
          let mut sinker = MixedSinker::<Gray16>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          gray16_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
