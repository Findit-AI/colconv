//! Per‑row packed RGB float16 (half-precision) → packed RGB8 throughput
//! baseline.
//!
//! RGBF16 stores R/G/B as IEEE 754 half-precision `f16` elements (FFmpeg
//! `AV_PIX_FMT_GBRPF16LE`'s packed cousin). Each pixel is 6 bytes —
//! kernel widens f16 → f32, clamps `[0.0, 1.0]`, scales x255, rounds
//! to u8. Generic on `BE`; benched at LE.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::bench_internals::rgbf16_to_rgb_row;
use half::f16;

/// Fills an `f16` buffer with deterministic pseudo‑random values in the
/// `[0.0, 1.0]` valid range.
fn fill_pseudo_random_f16(buf: &mut [f16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let f = (state as f32) / (u32::MAX as f32);
    *b = f16::from_f32(f);
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("rgbf16_to_rgb_row");

  for &w in WIDTHS {
    // RGBF16: 3 f16 per pixel.
    let mut rgb_in = std::vec![f16::ZERO; w * 3];
    fill_pseudo_random_f16(&mut rgb_in, 0x8888);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgbf16_to_rgb_row::<false>(black_box(&rgb_in), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
