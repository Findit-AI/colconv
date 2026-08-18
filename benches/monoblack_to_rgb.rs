//! Per‑frame Monoblack (1‑bit packed, MSB‑first, bit=0 → black) →
//! packed RGB throughput baseline.
//!
//! `Monoblack` carries 8 pixels per byte, MSB-first (FFmpeg
//! `AV_PIX_FMT_MONOBLACK`). The per-row dispatcher is `pub(crate)`;
//! we bench through the public walker `monoblack_to` +
//! `MixedSinker<Monoblack>` on a multi-row frame.
//!
//! All widths are multiples of 8 so the trailing-byte mask path doesn't
//! dominate.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  frame::MonoblackFrame,
  sinker::MixedSinker,
  source::{Monoblack, monoblack_to},
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

  let mut group = c.benchmark_group("monoblack_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    // 8 pixels per byte — `width / 8` bytes per row (all widths are
    // multiples of 8 here).
    let stride_bytes = (w / 8) as usize;
    let mut data = std::vec![0u8; stride_bytes * h_us];
    fill_pseudo_random(&mut data, 0xCCCC);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame =
            MonoblackFrame::try_new(black_box(&data), w, FRAME_HEIGHT, stride_bytes as u32)
              .unwrap();
          let mut sinker = MixedSinker::<Monoblack>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          monoblack_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
