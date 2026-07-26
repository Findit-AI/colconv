//! Per‑row GBRP12 (planar GBR 12‑bit) → packed RGB throughput baseline.
//!
//! GBRP12 stores G, B, R as 12-bit samples in three separate u16 planes
//! (low-bit packed). Representative of the high-bit GBR family —
//! GBRP9/10/12/14/16 all share the same `gbr_to_rgb_high_bit_row`
//! kernel shape (templated on `BITS` and `BE`).
//!
//! Two variants per width — `simd=true` and `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::gbr_to_rgb_high_bit_row;

/// Fills a `u16` buffer with deterministic pseudo‑random 12-bit samples
/// (low 12 bits set, high 4 bits zero — matches `gbrp12le`).
fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) & 0xFFF) as u16;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  let mut group = c.benchmark_group("gbrp12_to_rgb_row");

  for &w in WIDTHS {
    let mut g = std::vec![0u16; w];
    let mut b_plane = std::vec![0u16; w];
    let mut r = std::vec![0u16; w];
    // Widely-spaced 32-bit seeds so per-plane LCG streams are
    // independent from the first iteration.
    fill_pseudo_random_u16(&mut g, 0xDEAD_BEEF);
    fill_pseudo_random_u16(&mut b_plane, 0xCAFE_F00D);
    fill_pseudo_random_u16(&mut r, 0x1234_5678);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          // BITS = 12, BE = false (LE wire).
          gbr_to_rgb_high_bit_row::<12, false>(
            black_box(&g),
            black_box(&b_plane),
            black_box(&r),
            black_box(&mut rgb),
            w,
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
