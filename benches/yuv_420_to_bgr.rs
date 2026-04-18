//! Per‑row YUV 4:2:0 → packed BGR throughput baseline.
//!
//! Each iteration converts one row of the given width. Two variants
//! per width — `simd=true` (NEON on aarch64, scalar elsewhere) and
//! `simd=false` (forced scalar reference) — so we can read the NEON
//! speedup directly from adjacent lines in the Criterion report.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::yuv_420_to_bgr_row};

/// Fills a buffer with a deterministic pseudo‑random byte sequence so
/// the measurement isn't inflated by cache‑friendly uniform data.
fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  // 720p / 1080p / 4K row widths — all multiples of 16 so the NEON
  // loop covers them fully; picking non‑multiples here would spend
  // measurable time in the scalar tail and skew the comparison.
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("yuv_420_to_bgr_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    let mut u = std::vec![0u8; w / 2];
    let mut v = std::vec![0u8; w / 2];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut u, 0x2222);
    fill_pseudo_random(&mut v, 0x3333);
    let mut bgr = std::vec![0u8; w * 3];

    // Throughput reported in output bytes so `MB/s` numbers are
    // comparable across widths.
    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          yuv_420_to_bgr_row(
            black_box(&y),
            black_box(&u),
            black_box(&v),
            black_box(&mut bgr),
            w,
            MATRIX,
            FULL_RANGE,
            use_simd,
          );
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
