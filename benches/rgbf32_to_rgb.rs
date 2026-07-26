//! Per‑row packed RGB float32 → packed RGB8 throughput baseline.
//!
//! RGBF32 stores R/G/B as f32 elements (FFmpeg `AV_PIX_FMT_GBRPF32LE`'s
//! packed cousin). The kernel clamps each f32 to `[0.0, 1.0]`, scales
//! by 255, and rounds to u8. Generic on `BE`; benched at LE (the
//! common wire encoding).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::rgbf32_to_rgb_row;

/// Fills an `f32` buffer with deterministic pseudo‑random values in the
/// `[0.0, 1.0]` valid range (the clamp/scale path's hot region).
fn fill_pseudo_random_f32(buf: &mut [f32], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Map u32 → [0.0, 1.0]: divide by u32::MAX as f32.
    *b = (state as f32) / (u32::MAX as f32);
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgbf32_to_rgb_row");

  for &w in WIDTHS {
    // RGBF32: 3 f32 per pixel.
    let mut rgb_in = std::vec![0f32; w * 3];
    fill_pseudo_random_f32(&mut rgb_in, 0x7777);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          // LE wire encoding (the FFmpeg default for packed RGB float).
          rgbf32_to_rgb_row::<false>(black_box(&rgb_in), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
