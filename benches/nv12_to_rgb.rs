//! Per‑row NV12 (semi‑planar 4:2:0) → packed RGB throughput baseline.
//!
//! Each iteration converts one row of the given width. Two variants
//! per width — `simd=true` (best available backend for the current
//! target) and `simd=false` (forced scalar reference) — so the SIMD
//! speedup can be read directly from adjacent lines in the Criterion
//! report.
//!
//! Numerically identical output to the matching `yuv_420_to_rgb_row`
//! bench, so the SIMD speedup delta between the two benches isolates
//! the cost of the per‑iteration UV deinterleave.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{ColorMatrix, row::nv12_to_rgb_row};

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
  // 720p / 1080p / 4K row widths — all multiples of 16 (and 32 / 64
  // for the wider SIMD tiers) so the main loop covers them fully.
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("nv12_to_rgb_row");

  for &w in WIDTHS {
    let mut y = std::vec![0u8; w];
    // UV row is `width` bytes (`width/2` interleaved UV pairs).
    let mut uv = std::vec![0u8; w];
    fill_pseudo_random(&mut y, 0x1111);
    fill_pseudo_random(&mut uv, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];

    // Throughput reported in output bytes so `MB/s` numbers are
    // comparable across widths and against the yuv_420 bench.
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
