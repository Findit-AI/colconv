//! VUYA `with_rgb + with_rgba` combo benchmarks: Strategy A+ vs simulated 2-kernel.
//!
//! Compares the v0.18+ Strategy A+ flow (single chroma kernel → `expand_rgb_to_rgba_row`
//! → α-overwrite from source) against the simulated v0.17 baseline (two independent
//! chroma kernels — runs chroma math TWICE). Both flows are measured at:
//! - `use_simd = true`  — best available SIMD backend on the host
//! - `use_simd = false` — forced scalar reference path
//!
//! **Approach: only public APIs.** The A+ side runs through the real `MixedSinker`
//! flow (`vuya_to(&frame, .., &mut sinker)` with `with_rgb + with_rgba` attached).
//! The 2-kernel side hits the public per-row dispatchers directly. Trade-off: the
//! A+ side now includes constant per-frame sinker setup/dispatch overhead (~1-2 µs
//! at 1080p), but the per-row chroma + α work dominates from 1080p upward, so the
//! relative speedup vs the 2-kernel path stays meaningful.
//!
//! Multi-row frames (height = `FRAME_HEIGHT`) are used so the per-frame setup
//! cost is amortized across multiple row dispatches.
//!
//! Throughput metric: combined RGB + RGBA output bytes (`width * 7 * height`).
//!
//! Bench paths:
//! - `a_plus_simd`      — `MixedSinker::with_rgb + with_rgba`, simd dispatch  ← real v0.18+
//! - `a_plus_scalar`    — same, with `with_simd(false)`
//! - `two_kernel_simd`  — direct `vuya_to_rgb_row + vuya_to_rgba_row`         ← pre-A+ baseline
//! - `two_kernel_scalar`— same, with `use_simd = false`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  bench_internals::{vuya_to_rgb_row, vuya_to_rgba_row},
  frame::VuyaFrame,
  sinker::MixedSinker,
  source::{Vuya, vuya_to},
};

/// A small but >1 row count so per-frame `MixedSinker` setup cost is amortized
/// across several `process` calls. Single-row frames would inflate the A+
/// numbers by giving the constant setup overhead full weight on every iter.
const FRAME_HEIGHT: u32 = 4;

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

  let mut group = c.benchmark_group("vuya_a_plus_combo");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let row_bytes = w_us * 4;
    let stride = w * 4;

    // VUYA frame: width x height x 4 bytes.
    let mut packed = std::vec![0u8; row_bytes * h_us];
    fill_pseudo_random(&mut packed, 0x1111);

    // Throughput: combined RGB + RGBA output bytes for the whole frame.
    group.throughput(Throughput::Bytes((w_us * 7 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ via the public `MixedSinker` API -----------------------------
      // Mirrors the v0.18+ user-visible Strategy A+ path: chroma kernel runs
      // ONCE; expand + α-overwrite derive RGBA in-place. Includes per-frame
      // sinker overhead, amortized across `FRAME_HEIGHT` rows.
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            // Re-build the sinker each iter so the borrow on rgb/rgba is
            // released between iters and Criterion can keep them.
            let frame = VuyaFrame::new(black_box(&packed), w, FRAME_HEIGHT, stride);
            let mut sinker = MixedSinker::<Vuya>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap();
            vuya_to(&frame, FULL_RANGE, MATRIX, &mut sinker).unwrap();
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: direct per-row kernel calls ---------------------
      // Simulates the pre-A+ v0.17 path where each output format ran its own
      // full chroma kernel. Chroma math runs TWICE per row.
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let p_off = row * row_bytes;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              vuya_to_rgb_row(
                black_box(&packed[p_off..p_off + row_bytes]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              vuya_to_rgba_row(
                black_box(&packed[p_off..p_off + row_bytes]),
                black_box(&mut rgba[rgba_off..rgba_off + w_us * 4]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
            }
            black_box((&rgb, &rgba));
          });
        },
      );
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
