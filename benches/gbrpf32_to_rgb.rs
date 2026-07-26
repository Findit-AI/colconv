//! Per‑frame Gbrpf32 (planar GBR f32) → packed RGB throughput baseline.
//!
//! The per-row dispatcher for f32 GBR planes is `pub(crate)`; the
//! public surface is the walker `gbrpf32_to` + a `MixedSinker<Gbrpf32>`.
//! We bench through that walker on a multi-row frame (height =
//! `FRAME_HEIGHT`) so per-frame sinker setup is amortized across rows.
//!
//! Two variants per width — `simd=true` (NEON / SSE4.1 / AVX2 / AVX‑512
//! / wasm‑simd128 where wired up) and `simd=false`. GBR doesn't need
//! a YUV→RGB matrix; the walker signature reflects that.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  frame::Gbrpf32LeFrame,
  sinker::MixedSinker,
  source::{Gbrpf32, gbrpf32_to},
};

const FRAME_HEIGHT: u32 = 8;

/// Fills an `f32` buffer with deterministic pseudo‑random values in the
/// `[0.0, 1.0]` valid range.
fn fill_pseudo_random_f32(buf: &mut [f32], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state as f32) / (u32::MAX as f32);
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("gbrpf32_to_rgb");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;

    let mut g = std::vec![0f32; w_us * h_us];
    let mut b_plane = std::vec![0f32; w_us * h_us];
    let mut r = std::vec![0f32; w_us * h_us];
    fill_pseudo_random_f32(&mut g, 0x1357);
    fill_pseudo_random_f32(&mut b_plane, 0x2468);
    fill_pseudo_random_f32(&mut r, 0x369C);

    group.throughput(Throughput::Bytes((w_us * 3 * h_us) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &_w| {
        let mut rgb = std::vec![0u8; w_us * 3 * h_us];
        b.iter(|| {
          let frame = Gbrpf32LeFrame::try_new(
            black_box(&g),
            black_box(&b_plane),
            black_box(&r),
            w,
            FRAME_HEIGHT,
            w,
            w,
            w,
          )
          .unwrap();
          let mut sinker = MixedSinker::<Gbrpf32>::new(w_us, h_us)
            .with_simd(use_simd)
            .with_rgb(&mut rgb)
            .unwrap();
          gbrpf32_to(&frame, &mut sinker).unwrap();
          black_box(&rgb);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
