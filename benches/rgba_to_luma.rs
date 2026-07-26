//! Per‑row packed RGBA → packed RGB (α‑drop) throughput baseline.
//!
//! Note: there is no `rgba_to_luma_row` direct kernel in the public API;
//! the RGBA → luma path is composed in the sinker as
//! `rgba_to_rgb_row` (α-drop, scratch RGB) then `rgb_to_luma_row` on the
//! scratch. This bench measures the α-drop half — the part that's
//! unique to RGBA sources and shares no SIMD lanes with `rgb_to_luma`.
//! Pair it with `rgb24_to_luma.rs` to estimate the full chain.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::bench_internals::rgba_to_rgb_row;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];

  // Group name reflects the bench-file name so `cargo bench --bench
  // rgba_to_luma` puts its results under a matching Criterion path.
  // What we actually measure is the α-drop primitive (`rgba_to_rgb_row`)
  // — the RGBA-specific half of the RGBA→luma chain; the RGB→luma half
  // lives in `rgb24_to_luma`.
  let mut group = c.benchmark_group("rgba_to_luma_alpha_drop");

  for &w in WIDTHS {
    let mut rgba = std::vec![0u8; w * 4];
    fill_pseudo_random(&mut rgba, 0x2222);
    let mut rgb = std::vec![0u8; w * 3];

    // Throughput in RGB output bytes (3 bytes/pixel).
    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          rgba_to_rgb_row(black_box(&rgba), black_box(&mut rgb), w, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
