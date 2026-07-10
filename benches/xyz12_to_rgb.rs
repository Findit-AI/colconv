//! Per‑row XYZ12 (packed CIE XYZ 12‑bit) → packed RGB throughput
//! baseline.
//!
//! XYZ12 is FFmpeg `AV_PIX_FMT_XYZ12{LE,BE}`: packed X/Y/Z `u16` triples
//! with the 12 active bits in the high 12 bits (`[15:4]`). The kernel
//! applies SMPTE ST 428-1 inverse-OETF, a 3×3 XYZ→target-RGB matmul, the
//! sRGB-shape OETF, and finally `[0, 255]` rescale + u8 narrow. We
//! benchmark the LE path (`BE = false`) with `DcpTargetGamut::Rec709`
//! as a representative; the matrix dim sits behind a runtime argument
//! so the kernel shape doesn't change across gamuts.
//!
//! Two variants per width — `simd=true` (NEON / SSE4.1 / AVX2 /
//! AVX‑512 / wasm‑simd128 where wired) and `simd=false`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{DcpTargetGamut, bench_internals::xyz12_to_rgb_row};

/// Fills a `u16` buffer with deterministic XYZ12-packed pseudo-random
/// samples — 12-bit values shifted into the high 12 bits (low 4 bits
/// zero), matching the real XYZ12 storage layout.
fn fill_pseudo_random_xyz12(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (((state >> 8) & 0xFFF) as u16) << 4;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const GAMUT: DcpTargetGamut = DcpTargetGamut::Rec709;

  let mut group = c.benchmark_group("xyz12_to_rgb_row");

  for &w in WIDTHS {
    // XYZ12: 3 u16 elements per pixel (X, Y, Z).
    let mut xyz = std::vec![0u16; w * 3];
    fill_pseudo_random_xyz12(&mut xyz, 0xBBBB);
    let mut rgb = std::vec![0u8; w * 3];

    group.throughput(Throughput::Bytes((w * 3) as u64));

    for use_simd in [false, true] {
      let label = if use_simd { "simd" } else { "scalar" };
      group.bench_with_input(BenchmarkId::new(label, w), &w, |b, &w| {
        b.iter(|| {
          // LE wire encoding (`BE = false`).
          xyz12_to_rgb_row::<false>(black_box(&xyz), black_box(&mut rgb), w, GAMUT, use_simd);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
