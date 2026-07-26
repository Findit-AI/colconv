//! Per‑row Y216 (packed YUV 4:2:2 16‑bit, native `u16` quadruples)
//! → packed RGB throughput baseline.
//!
//! Y216 stores 16 active bits per `u16` sample (no MSB-alignment shift);
//! kernel loads 4 `u16` elements per pixel pair `(Y0 U Y1 V)`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{ColorMatrix, bench_internals::y216_to_rgb_row};

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt2020Ncl;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("y216_to_rgb_row");

  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    fill_pseudo_random_u16(&mut packed, 0x3333);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          y216_to_rgb_row(
            black_box(&packed),
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
