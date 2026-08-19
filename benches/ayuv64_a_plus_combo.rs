//! AYUV64 `with_rgb + with_rgba` combo benchmarks: Strategy A+ vs simulated 2-kernel.
//!
//! AYUV64 (FFmpeg `AV_PIX_FMT_AYUV64LE`) carries a 16-bit source α channel, so the
//! A+ combo must handle two output depths:
//! - **u8 RGBA**: α depth-converted u16 → u8 via `>> 8`
//! - **u16 RGBA**: α written direct at native 16-bit depth
//!
//! Two Criterion groups are used — `ayuv64_a_plus_combo_u8` and
//! `ayuv64_a_plus_combo_u16` — so Criterion's report separates the two paths.
//!
//! Compares the v0.18+ Strategy A+ flow (single chroma kernel → expand → α-overwrite
//! from source) against the simulated v0.17 baseline (two independent chroma kernels
//! — runs chroma math TWICE). Both paths at `use_simd = true/false`.
//!
//! **Approach: only public APIs.** The A+ side runs through the real `MixedSinker`
//! flow (`ayuv64_to(&frame, .., &mut sinker)` with `with_rgb + with_rgba` attached
//! for u8, or `with_rgb_u16 + with_rgba_u16` for u16). The 2-kernel side hits the
//! public per-row dispatchers directly. Trade-off: the A+ side now includes constant
//! per-frame sinker setup/dispatch overhead (~1-2 µs at 1080p), but the per-row
//! chroma + α work dominates from 1080p upward.
//!
//! Multi-row frames (height = `FRAME_HEIGHT`) amortize the per-frame setup across
//! several row dispatches.
//!
//! Bench paths (per group):
//! - `a_plus_simd`      — `MixedSinker` with both buffers, simd dispatch
//! - `a_plus_scalar`    — same, with `with_simd(false)`
//! - `two_kernel_simd`  — direct `ayuv64_to_rgb_(u16_)row + ayuv64_to_rgba_(u16_)row`
//! - `two_kernel_scalar`— same, with `use_simd = false`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  Ayuv64, Ayuv64Frame, KernelMatrix, ayuv64_to,
  bench_internals::{
    ayuv64_to_rgb_row, ayuv64_to_rgb_u16_row, ayuv64_to_rgba_row, ayuv64_to_rgba_u16_row,
  },
  sinker::MixedSinker,
};

/// Multi-row frame so per-frame `MixedSinker` setup is amortized across
/// several row dispatches.
const FRAME_HEIGHT: u32 = 4;

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Produce values in 0..=65535 (full u16 range) — AYUV64 is 16-bit native.
    *b = (state >> 16) as u16;
  }
}

// ---- u8 RGBA group ----------------------------------------------------------

fn bench_u8(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("ayuv64_a_plus_combo_u8");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let row_elems = w_us * 4; // u16 elements per row
    let stride = w * 4;

    // AYUV64 frame: width x height x 4 u16 elements.
    let mut packed = std::vec![0u16; row_elems * h_us];
    fill_pseudo_random_u16(&mut packed, 0x1111);

    // Throughput: u8 RGB (3 bytes/px) + u8 RGBA (4 bytes/px) = 7 bytes/px,
    // multiplied by rows.
    group.throughput(Throughput::Bytes((w_us * 7 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ via the public `MixedSinker` API (u8 RGBA) -------------------
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Ayuv64Frame::new(black_box(&packed), w, FRAME_HEIGHT, stride);
            let mut sinker = MixedSinker::<Ayuv64>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap();
            ayuv64_to(
              &frame,
              FULL_RANGE,
              sinker
                .set_color_spec(&pixon::ColorSpec::of_matrix(MATRIX))
                .unwrap(),
            )
            .unwrap();
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: direct per-row dispatchers (u8 RGBA) ------------
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let p_off = row * row_elems;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              ayuv64_to_rgb_row(
                black_box(&packed[p_off..p_off + row_elems]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
                false,
              );
              ayuv64_to_rgba_row(
                black_box(&packed[p_off..p_off + row_elems]),
                black_box(&mut rgba[rgba_off..rgba_off + w_us * 4]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
                false,
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

// ---- u16 RGBA group ---------------------------------------------------------

fn bench_u16(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("ayuv64_a_plus_combo_u16");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let row_elems = w_us * 4;
    let stride = w * 4;

    let mut packed = std::vec![0u16; row_elems * h_us];
    fill_pseudo_random_u16(&mut packed, 0x2222);

    // Throughput: u16 RGB (6 bytes/px) + u16 RGBA (8 bytes/px) = 14 bytes/px,
    // multiplied by rows.
    group.throughput(Throughput::Bytes((w_us * 14 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ via the public `MixedSinker` API (u16 RGBA) ------------------
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Ayuv64Frame::new(black_box(&packed), w, FRAME_HEIGHT, stride);
            let mut sinker = MixedSinker::<Ayuv64>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb_u16(&mut rgb)
              .unwrap()
              .with_rgba_u16(&mut rgba)
              .unwrap();
            ayuv64_to(
              &frame,
              FULL_RANGE,
              sinker
                .set_color_spec(&pixon::ColorSpec::of_matrix(MATRIX))
                .unwrap(),
            )
            .unwrap();
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: direct per-row dispatchers (u16 RGBA) -----------
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let p_off = row * row_elems;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              ayuv64_to_rgb_u16_row(
                black_box(&packed[p_off..p_off + row_elems]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
                false,
              );
              ayuv64_to_rgba_u16_row(
                black_box(&packed[p_off..p_off + row_elems]),
                black_box(&mut rgba[rgba_off..rgba_off + w_us * 4]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
                false,
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

criterion_group!(benches_u8, bench_u8);
criterion_group!(benches_u16, bench_u16);
criterion_main!(benches_u8, benches_u16);
