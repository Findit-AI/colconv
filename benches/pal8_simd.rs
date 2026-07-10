//! Criterion benchmark: `AV_PIX_FMT_PAL8` scalar vs. NEON SIMD throughput.
//!
//! Measures all four pal8 row kernels across three production widths
//! (256, 1280, 1920) with `simd=false` (scalar) and `simd=true` (NEON
//! on aarch64, scalar fallback elsewhere). The `simd=true / simd=false`
//! ratio from adjacent rows in the Criterion output is the honest speedup
//! figure for the benchmark report.
//!
//! # Palette / index data
//!
//! Both the palette and index arrays are filled with a simple LCG PRNG so
//! the benchmark is not inflated by cache-friendly uniform data. The palette
//! uses all 256 entries with pseudo-random BGRA values; indices are uniform
//! across [0, 255]. This mimics a real-world indexed-color image with a
//! fully-populated palette.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::bench_internals::{
  pal8_to_rgb_row, pal8_to_rgb_u16_row, pal8_to_rgba_row, pal8_to_rgba_u16_row,
};

/// Fills `buf` with pseudo-random bytes using a simple LCG.
fn fill_lcg(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 24) as u8;
  }
}

/// Builds a 256-entry test palette from pseudo-random BGRA values.
fn build_palette(seed: u32) -> [[u8; 4]; 256] {
  let mut raw = [0u8; 1024];
  fill_lcg(&mut raw, seed);
  let mut pal = [[0u8; 4]; 256];
  for (i, entry) in pal.iter_mut().enumerate() {
    entry[0] = raw[i * 4];
    entry[1] = raw[i * 4 + 1];
    entry[2] = raw[i * 4 + 2];
    entry[3] = raw[i * 4 + 3];
  }
  pal
}

/// Builds `width` pseudo-random index bytes in [0, 255].
fn build_indices(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut buf = std::vec![0u8; width];
  fill_lcg(&mut buf, seed);
  buf
}

fn bench_pal8_to_rgb(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[256, 1280, 1920];
  let mut group = c.benchmark_group("pal8_to_rgb_row");

  for &w in WIDTHS {
    let palette = build_palette(0x1111_1111);
    let indices = build_indices(w, 0x2222_2222);
    let mut rgb_out = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, _| {
        b.iter(|| {
          pal8_to_rgb_row(
            black_box(&indices),
            black_box(&palette),
            black_box(&mut rgb_out),
            use_simd,
          );
        });
      });
    }
  }
  group.finish();
}

fn bench_pal8_to_rgba(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[256, 1280, 1920];
  let mut group = c.benchmark_group("pal8_to_rgba_row");

  for &w in WIDTHS {
    let palette = build_palette(0x3333_3333);
    let indices = build_indices(w, 0x4444_4444);
    let mut rgba_out = std::vec![0u8; w * 4];

    group.throughput(Throughput::Bytes((w * 4) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, _| {
        b.iter(|| {
          pal8_to_rgba_row(
            black_box(&indices),
            black_box(&palette),
            black_box(&mut rgba_out),
            use_simd,
          );
        });
      });
    }
  }
  group.finish();
}

fn bench_pal8_to_rgb_u16(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[256, 1280, 1920];
  let mut group = c.benchmark_group("pal8_to_rgb_u16_row");

  for &w in WIDTHS {
    let palette = build_palette(0x5555_5555);
    let indices = build_indices(w, 0x6666_6666);
    let mut rgb_u16_out = std::vec![0u16; w * 3];

    // Throughput in output bytes (u16 = 2 bytes each, 3 channels).
    group.throughput(Throughput::Bytes((w * 3 * 2) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, _| {
        b.iter(|| {
          pal8_to_rgb_u16_row(
            black_box(&indices),
            black_box(&palette),
            black_box(&mut rgb_u16_out),
            use_simd,
          );
        });
      });
    }
  }
  group.finish();
}

fn bench_pal8_to_rgba_u16(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[256, 1280, 1920];
  let mut group = c.benchmark_group("pal8_to_rgba_u16_row");

  for &w in WIDTHS {
    let palette = build_palette(0x7777_7777);
    let indices = build_indices(w, 0x8888_8888);
    let mut rgba_u16_out = std::vec![0u16; w * 4];

    group.throughput(Throughput::Bytes((w * 4 * 2) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, _| {
        b.iter(|| {
          pal8_to_rgba_u16_row(
            black_box(&indices),
            black_box(&palette),
            black_box(&mut rgba_u16_out),
            use_simd,
          );
        });
      });
    }
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_pal8_to_rgb,
  bench_pal8_to_rgba,
  bench_pal8_to_rgb_u16,
  bench_pal8_to_rgba_u16
);
criterion_main!(benches);
