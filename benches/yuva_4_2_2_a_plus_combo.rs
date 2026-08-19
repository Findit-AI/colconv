//! YUVA 4:2:2 `with_rgb + with_rgba` combo benchmarks: Strategy A+ vs simulated 2-kernel.
//!
//! Compares the v0.18+ Strategy A+ flow (single chroma kernel → `expand_rgb_to_rgba_row`
//! → α-overwrite from source) against the simulated v0.17 baseline (two independent
//! chroma kernels — runs chroma math TWICE). Covers:
//! - `Yuva422p`   (8-bit, u8 RGBA output)
//! - `Yuva422p10` (10-bit, both u8 RGBA + u16 RGBA outputs)
//!
//! **Scope / BITS choice**: BITS=10 and BITS=16 are both benchmarked for the high-bit
//! family. BITS=9 is niche; BITS=12 also exists for Yuva422p (unlike Yuva420p where
//! FFmpeg has no 12/14-bit variants) and exercises the Q15 i32 path, not expected to
//! differ materially from BITS=10. BITS=10 is the most common path (ProRes 422
//! LT/HQ, DNxHD/HR, HDR10 4:2:2) and fully exercises the BITS-generic Q15 i32
//! chroma kernel. BITS=16 exercises the dedicated i64 chroma-widened family — the
//! Yuva422p16 walkers and RGBA dispatchers are fully exported and SIMD-backed
//! (NEON, AVX2, AVX-512, wasm). The i64 chroma path is where the A+ win should be
//! largest, since the i64 kernel is the slowest chroma variant and A+ eliminates one
//! full invocation per row.
//!
//! **Per-row kernel note**: Yuva422p reuses the same per-row kernel family as
//! Yuva420p — `yuv_420_to_rgb_row` / `yuva420p_to_rgba_row` (8-bit) and
//! `yuv420p10_to_rgb_row` / `yuva420p10_to_rgba_row` (10-bit). The only
//! difference from Yuva420p is in the vertical walker: 4:2:2 chroma rows are
//! full-height (not half-height), so the UV planes have `(w/2) x h` elements.
//!
//! **Approach: only public APIs.** The A+ side runs through the real `MixedSinker`
//! flow. The 2-kernel side hits the public per-row dispatchers directly.
//! Multi-row frames (height = `FRAME_HEIGHT`) amortize per-frame setup overhead.
//!
//! Bench paths per group:
//! - `a_plus_scalar`    — `MixedSinker::with_rgb + with_rgba`, forced scalar
//! - `a_plus_simd`      — same, best available SIMD backend
//! - `two_kernel_scalar`— direct RGB + RGBA per-row kernel calls, forced scalar
//! - `two_kernel_simd`  — same, best available SIMD backend
//!
//! Throughput metric: combined RGB + RGBA output bytes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  KernelMatrix,
  bench_internals::{
    yuv_420_to_rgb_row, yuv420p10_to_rgb_row, yuv420p10_to_rgb_u16_row, yuv420p16_to_rgb_row,
    yuv420p16_to_rgb_u16_row, yuva420p_to_rgba_row, yuva420p10_to_rgba_row,
    yuva420p10_to_rgba_u16_row, yuva420p16_to_rgba_row, yuva420p16_to_rgba_u16_row,
  },
  frame::{Yuva422pFrame, Yuva422pFrame16},
  sinker::MixedSinker,
  source::{Yuva422p, Yuva422p10, Yuva422p16, yuva422p_to, yuva422p10_to, yuva422p16_to},
};

/// Multi-row frame height — amortizes per-frame `MixedSinker` setup across
/// several row dispatches.
const FRAME_HEIGHT: u32 = 4;

fn fill_pseudo_random_u8(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Low-bit-packed 10-bit samples in [0, 1023].
    *b = ((state >> 8) & 0x3FF) as u16;
  }
}

fn fill_pseudo_random_u16_full(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Full 16-bit samples in [0, 65535].
    *b = (state >> 16) as u16;
  }
}

// ---- Yuva422p (8-bit) u8 RGBA group ----------------------------------------

fn bench_yuva422p(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuva422p_a_plus_combo");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    // 4:2:2: Y plane = wxh, UV planes = (w/2)xh (full-height chroma), A = wxh.
    let y_len = w_us * h_us;
    let uv_len = (w_us / 2) * h_us;
    let a_len = w_us * h_us;

    let mut y_plane = std::vec![0u8; y_len];
    let mut u_plane = std::vec![0u8; uv_len];
    let mut v_plane = std::vec![0u8; uv_len];
    let mut a_plane = std::vec![0u8; a_len];
    fill_pseudo_random_u8(&mut y_plane, 0x5511);
    fill_pseudo_random_u8(&mut u_plane, 0x6622);
    fill_pseudo_random_u8(&mut v_plane, 0x7733);
    fill_pseudo_random_u8(&mut a_plane, 0x8844);

    // Throughput: u8 RGB (3 bytes/px) + u8 RGBA (4 bytes/px) = 7 bytes/px x w x h.
    group.throughput(Throughput::Bytes((w_us * 7 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ via the public `MixedSinker` API --------------------------------
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Yuva422pFrame::new(
              black_box(&y_plane),
              black_box(&u_plane),
              black_box(&v_plane),
              black_box(&a_plane),
              w,
              FRAME_HEIGHT,
              w,
              w / 2,
              w / 2,
              w,
            );
            let mut sinker = MixedSinker::<Yuva422p>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap();
            yuva422p_to(
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

      // ---- 2-kernel combo: direct per-row kernel calls ------------------------
      // 4:2:2: each Y row has its own chroma row (no r/2 sharing).
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let y_off = row * w_us;
              // 4:2:2: chroma row index == Y row index (full-height UV).
              let uv_off = row * (w_us / 2);
              let a_off = row * w_us;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              yuv_420_to_rgb_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              yuva420p_to_rgba_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&a_plane[a_off..a_off + w_us]),
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

// ---- Yuva422p10 (10-bit) u8 RGBA group -------------------------------------

fn bench_yuva422p10_u8(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuva422p10_a_plus_combo_u8");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let y_len = w_us * h_us;
    let uv_len = (w_us / 2) * h_us;
    let a_len = w_us * h_us;

    let mut y_plane = std::vec![0u16; y_len];
    let mut u_plane = std::vec![0u16; uv_len];
    let mut v_plane = std::vec![0u16; uv_len];
    let mut a_plane = std::vec![0u16; a_len];
    fill_pseudo_random_u16(&mut y_plane, 0x5511);
    fill_pseudo_random_u16(&mut u_plane, 0x6622);
    fill_pseudo_random_u16(&mut v_plane, 0x7733);
    fill_pseudo_random_u16(&mut a_plane, 0x8844);

    // Throughput: u8 RGB (3 bytes/px) + u8 RGBA (4 bytes/px) = 7 bytes/px.
    group.throughput(Throughput::Bytes((w_us * 7 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Yuva422pFrame16::<10>::new(
              black_box(&y_plane),
              black_box(&u_plane),
              black_box(&v_plane),
              black_box(&a_plane),
              w,
              FRAME_HEIGHT,
              w,
              w / 2,
              w / 2,
              w,
            );
            let mut sinker = MixedSinker::<Yuva422p10>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap();
            yuva422p10_to(
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

      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let y_off = row * w_us;
              let uv_off = row * (w_us / 2);
              let a_off = row * w_us;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              yuv420p10_to_rgb_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              yuva420p10_to_rgba_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&a_plane[a_off..a_off + w_us]),
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

// ---- Yuva422p10 (10-bit) u16 RGBA group ------------------------------------

fn bench_yuva422p10_u16(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuva422p10_a_plus_combo_u16");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let y_len = w_us * h_us;
    let uv_len = (w_us / 2) * h_us;
    let a_len = w_us * h_us;

    let mut y_plane = std::vec![0u16; y_len];
    let mut u_plane = std::vec![0u16; uv_len];
    let mut v_plane = std::vec![0u16; uv_len];
    let mut a_plane = std::vec![0u16; a_len];
    fill_pseudo_random_u16(&mut y_plane, 0x5512);
    fill_pseudo_random_u16(&mut u_plane, 0x6623);
    fill_pseudo_random_u16(&mut v_plane, 0x7734);
    fill_pseudo_random_u16(&mut a_plane, 0x8845);

    // Throughput: u16 RGB (6 bytes/px) + u16 RGBA (8 bytes/px) = 14 bytes/px.
    group.throughput(Throughput::Bytes((w_us * 14 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Yuva422pFrame16::<10>::new(
              black_box(&y_plane),
              black_box(&u_plane),
              black_box(&v_plane),
              black_box(&a_plane),
              w,
              FRAME_HEIGHT,
              w,
              w / 2,
              w / 2,
              w,
            );
            let mut sinker = MixedSinker::<Yuva422p10>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb_u16(&mut rgb)
              .unwrap()
              .with_rgba_u16(&mut rgba)
              .unwrap();
            yuva422p10_to(
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

      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let y_off = row * w_us;
              let uv_off = row * (w_us / 2);
              let a_off = row * w_us;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              yuv420p10_to_rgb_u16_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              yuva420p10_to_rgba_u16_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&a_plane[a_off..a_off + w_us]),
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

// ---- Yuva422p16 (16-bit) u8 RGBA group -------------------------------------

fn bench_yuva422p16_u8(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuva422p16_a_plus_combo_u8");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    // 4:2:2: Y plane = wxh, UV planes = (w/2)xh (full-height chroma), A = wxh.
    let y_len = w_us * h_us;
    let uv_len = (w_us / 2) * h_us;
    let a_len = w_us * h_us;

    let mut y_plane = std::vec![0u16; y_len];
    let mut u_plane = std::vec![0u16; uv_len];
    let mut v_plane = std::vec![0u16; uv_len];
    let mut a_plane = std::vec![0u16; a_len];
    fill_pseudo_random_u16_full(&mut y_plane, 0x5513);
    fill_pseudo_random_u16_full(&mut u_plane, 0x6624);
    fill_pseudo_random_u16_full(&mut v_plane, 0x7735);
    fill_pseudo_random_u16_full(&mut a_plane, 0x8846);

    // Throughput: u8 RGB (3 bytes/px) + u8 RGBA (4 bytes/px) = 7 bytes/px x w x h.
    group.throughput(Throughput::Bytes((w_us * 7 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Yuva422pFrame16::<16>::new(
              black_box(&y_plane),
              black_box(&u_plane),
              black_box(&v_plane),
              black_box(&a_plane),
              w,
              FRAME_HEIGHT,
              w,
              w / 2,
              w / 2,
              w,
            );
            let mut sinker = MixedSinker::<Yuva422p16>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb(&mut rgb)
              .unwrap()
              .with_rgba(&mut rgba)
              .unwrap();
            yuva422p16_to(
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

      // 4:2:2: each Y row has its own chroma row (full-height UV).
      // Per-row kernel for 422 16-bit reuses the 420p16 family
      // (chroma layout is identical row-by-row; only the vertical walker differs).
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u8; w_us * 3 * h_us];
          let mut rgba = std::vec![0u8; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let y_off = row * w_us;
              // 4:2:2: chroma row index == Y row index (full-height UV).
              let uv_off = row * (w_us / 2);
              let a_off = row * w_us;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              yuv420p16_to_rgb_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              yuva420p16_to_rgba_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&a_plane[a_off..a_off + w_us]),
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

// ---- Yuva422p16 (16-bit) u16 RGBA group — largest expected A+ delta --------

fn bench_yuva422p16_u16(c: &mut Criterion) {
  const WIDTHS: &[u32] = &[1280, 1920, 3840];
  const MATRIX: KernelMatrix = KernelMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuva422p16_a_plus_combo_u16");

  for &w in WIDTHS {
    let w_us = w as usize;
    let h_us = FRAME_HEIGHT as usize;
    let y_len = w_us * h_us;
    let uv_len = (w_us / 2) * h_us;
    let a_len = w_us * h_us;

    let mut y_plane = std::vec![0u16; y_len];
    let mut u_plane = std::vec![0u16; uv_len];
    let mut v_plane = std::vec![0u16; uv_len];
    let mut a_plane = std::vec![0u16; a_len];
    fill_pseudo_random_u16_full(&mut y_plane, 0x5514);
    fill_pseudo_random_u16_full(&mut u_plane, 0x6625);
    fill_pseudo_random_u16_full(&mut v_plane, 0x7736);
    fill_pseudo_random_u16_full(&mut a_plane, 0x8847);

    // Throughput: u16 RGB (6 bytes/px) + u16 RGBA (8 bytes/px) = 14 bytes/px.
    group.throughput(Throughput::Bytes((w_us * 14 * h_us) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            let frame = Yuva422pFrame16::<16>::new(
              black_box(&y_plane),
              black_box(&u_plane),
              black_box(&v_plane),
              black_box(&a_plane),
              w,
              FRAME_HEIGHT,
              w,
              w / 2,
              w / 2,
              w,
            );
            let mut sinker = MixedSinker::<Yuva422p16>::new(w_us, h_us)
              .with_simd(use_simd)
              .with_rgb_u16(&mut rgb)
              .unwrap()
              .with_rgba_u16(&mut rgba)
              .unwrap();
            yuva422p16_to(
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

      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &_w| {
          let mut rgb = std::vec![0u16; w_us * 3 * h_us];
          let mut rgba = std::vec![0u16; w_us * 4 * h_us];
          b.iter(|| {
            for row in 0..h_us {
              let y_off = row * w_us;
              let uv_off = row * (w_us / 2);
              let a_off = row * w_us;
              let rgb_off = row * w_us * 3;
              let rgba_off = row * w_us * 4;
              yuv420p16_to_rgb_u16_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&mut rgb[rgb_off..rgb_off + w_us * 3]),
                w_us,
                MATRIX,
                FULL_RANGE,
                use_simd,
              );
              yuva420p16_to_rgba_u16_row(
                black_box(&y_plane[y_off..y_off + w_us]),
                black_box(&u_plane[uv_off..uv_off + w_us / 2]),
                black_box(&v_plane[uv_off..uv_off + w_us / 2]),
                black_box(&a_plane[a_off..a_off + w_us]),
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

criterion_group!(benches_8bit, bench_yuva422p);
criterion_group!(benches_10bit_u8, bench_yuva422p10_u8);
criterion_group!(benches_10bit_u16, bench_yuva422p10_u16);
criterion_group!(benches_16bit_u8, bench_yuva422p16_u8);
criterion_group!(benches_16bit_u16, bench_yuva422p16_u16);
criterion_main!(
  benches_8bit,
  benches_10bit_u8,
  benches_10bit_u16,
  benches_16bit_u8,
  benches_16bit_u16
);
