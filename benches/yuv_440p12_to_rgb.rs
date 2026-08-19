//! YUV 4:4:0 planar 12‑bit → packed RGB throughput baseline.
//!
//! 4:4:0 12‑bit reuses the 4:4:4 12‑bit per‑row dispatcher
//! (`yuv444p12_to_rgb_row`); walker indexing supplies the half-height
//! chroma row for each pair of Y rows. We bench through the public
//! `yuv440p12_to` walker via `MixedSinker`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::Yuv440p12Frame,
  sinker::MixedSinker,
  source::{Yuv440p12, yuv440p12_to},
};

const FRAME_HEIGHT: u32 = 8;

/// Fills a `u16` buffer with deterministic 12‑bit pseudo‑random samples —
/// values occupy the low 12 bits of each `u16`.
fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) & 0xFFF) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuv440p12_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let ch_us = h_us.div_ceil(2);

    let mut y = std::vec![0u16; w_us * h_us];
    let mut u = std::vec![0u16; w_us * ch_us];
    let mut v = std::vec![0u16; w_us * ch_us];
    fill_pseudo_random_u16(&mut y, 0x1111);
    fill_pseudo_random_u16(&mut u, 0x2222);
    fill_pseudo_random_u16(&mut v, 0x3333);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Yuv440p12Frame::new(
            black_box(&y),
            black_box(&u),
            black_box(&v),
            w,
            FRAME_HEIGHT,
            w,
            w,
            w,
          );
          let mut sinker = MixedSinker::<Yuv440p12>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          yuv440p12_to(
            &frame,
            FULL_RANGE,
            sinker
              .set_color_spec(&pixon::ColorSpec::of_matrix(MATRIX))
              .unwrap(),
          )
          .unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
