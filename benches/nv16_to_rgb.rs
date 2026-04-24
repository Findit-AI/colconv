//! Per‑row NV16 (semi‑planar 4:2:2) → packed RGB throughput baseline.
//!
//! NV16 shares its per‑row kernel with NV12 — the 4:2:0 vs 4:2:2
//! difference is purely in the vertical walker (one UV row per Y
//! row for NV16 vs one per two Y rows for NV12). This bench calls
//! [`nv12_to_rgb_row`] directly, which is what
//! [`MixedSinker<Nv16>`](colconv::sinker::MixedSinker) does as well.
//! Numerically identical output to the NV12 bench at the same width.
//!
//! Two variants per width — `simd=true` (best available backend) and
//! `simd=false` (forced scalar reference) — so the SIMD speedup can
//! be read directly from adjacent lines in the Criterion report.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::nv12_to_rgb_row};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("nv16_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    // 4:2:2: UV row is `width` bytes (`width/2` interleaved UV pairs).
    let mut uv = std::vec![0u8; w];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut uv, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          nv12_to_rgb_row(
            black_box(&y),
            black_box(&uv),
            black_box(&mut rgb),
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
