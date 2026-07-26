//! Full-pipeline fused-downscale gate: 1920x1080 -> 336x189 (the
//! SigLIP2-NaFlex analysis geometry from the design issues).
//!
//! Compares, per frame:
//! - the NATIVE tier (bin Y/U/V at output res, convert once per
//!   output row),
//! - the ROW-STAGE tier (convert each row at source res, then bin),
//! - the full-res conversion baseline (what convert-then-resize
//!   pipelines pay before they even start resizing),
//!
//! plus the packed-RGB fused path, and luma-only variants (which
//! never read chroma).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use pixon::{
  ColorMatrix,
  frame::{Nv12Frame, Nv21Frame, P010LeFrame, Rgb24Frame, Yuv420p16LeFrame, Yuv420pFrame},
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{
    Nv12, Nv21, P010, Rgb24, Yuv420p, Yuv420p16, nv12_to, nv21_to, p010_to, rgb24_to, yuv420p_to,
    yuv420p16_to,
  },
};

const SRC_W: usize = 1920;
const SRC_H: usize = 1080;
const OUT_W: usize = 336;
const OUT_H: usize = 189;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  let mut y = std::vec![0u8; SRC_W * SRC_H];
  let mut u = std::vec![0u8; SRC_W * SRC_H / 4];
  let mut v = std::vec![0u8; SRC_W * SRC_H / 4];
  fill_pseudo_random(&mut y, 0x1111);
  fill_pseudo_random(&mut u, 0x2222);
  fill_pseudo_random(&mut v, 0x3333);

  // Interleaved chroma planes for the semi-planar twins (same logical
  // chroma as the planar U / V above): NV12 is `U V U V …`, NV21 swaps
  // to `V U V U …`. The semi-planar native tier de-interleaves these
  // back into U / V scratch and bins through the same 4:2:0 join.
  let mut uv_nv12 = std::vec![0u8; SRC_W * SRC_H / 2];
  let mut uv_nv21 = std::vec![0u8; SRC_W * SRC_H / 2];
  for (i, (&uu, &vv)) in u.iter().zip(v.iter()).enumerate() {
    uv_nv12[i * 2] = uu;
    uv_nv12[i * 2 + 1] = vv;
    uv_nv21[i * 2] = vv;
    uv_nv21[i * 2 + 1] = uu;
  }

  let mut group = c.benchmark_group("fused_downscale_1080p_to_336x189");
  group.sample_size(20);

  let n = OUT_W * OUT_H;
  let mut rgb = std::vec![0u8; n * 3];
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut vv = std::vec![0u8; n];
  let mut luma = std::vec![0u8; n];

  for (name, native) in [
    ("yuv420p_rgb_hsv_native", true),
    ("yuv420p_rgb_hsv_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420pFrame::new(
          &y,
          &u,
          &v,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        );
        let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut vv)
        .unwrap();
        yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb);
      });
    });
  }

  // Semi-planar (NV12 / NV21) native vs row-stage — the P2 fast tier
  // de-interleaves the interleaved chroma row into U / V scratch and
  // bins through the same 4:2:0 join as the planar twin.
  for (name, native) in [
    ("nv12_rgb_hsv_native", true),
    ("nv12_rgb_hsv_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Nv12Frame::new(
          &y,
          &uv_nv12,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          SRC_W as u32,
        );
        let mut sink = MixedSinker::<Nv12, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut vv)
        .unwrap();
        nv12_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb);
      });
    });
  }

  for (name, native) in [
    ("nv21_rgb_hsv_native", true),
    ("nv21_rgb_hsv_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Nv21Frame::new(
          &y,
          &uv_nv21,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          SRC_W as u32,
        );
        let mut sink = MixedSinker::<Nv21, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut vv)
        .unwrap();
        nv21_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb);
      });
    });
  }

  // High-bit planar 4:2:0 (Yuv420p16 LE) native vs row-stage — the P2
  // u16 fast tier. Two output flavors: the native-depth u16 colour path
  // (`with_rgb_u16`, which exercises the independent u16 4:4:4 kernel) and
  // the u8 colour + HSV path (the u16-input → u8-output 4:4:4 kernel). The
  // 16-bit Y plane is the low-packed `u16` codes the kernels actually
  // consume; reuse the 8-bit pseudo-random bytes widened to `u16`.
  let mut y16 = std::vec![0u16; SRC_W * SRC_H];
  let mut u16p = std::vec![0u16; SRC_W * SRC_H / 4];
  let mut v16p = std::vec![0u16; SRC_W * SRC_H / 4];
  for (d, &s) in y16.iter_mut().zip(y.iter()) {
    *d = ((s as u16) << 8) | s as u16;
  }
  for (d, &s) in u16p.iter_mut().zip(u.iter()) {
    *d = ((s as u16) << 8) | s as u16;
  }
  for (d, &s) in v16p.iter_mut().zip(v.iter()) {
    *d = ((s as u16) << 8) | s as u16;
  }
  let mut rgb_u16 = std::vec![0u16; n * 3];

  for (name, native) in [
    ("yuv420p16_rgb_u16_native", true),
    ("yuv420p16_rgb_u16_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420p16LeFrame::try_new(
          &y16,
          &u16p,
          &v16p,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        )
        .unwrap();
        let mut sink = MixedSinker::<Yuv420p16, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb_u16);
      });
    });
  }

  for (name, native) in [
    ("yuv420p16_rgb_hsv_native", true),
    ("yuv420p16_rgb_hsv_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420p16LeFrame::try_new(
          &y16,
          &u16p,
          &v16p,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        )
        .unwrap();
        let mut sink = MixedSinker::<Yuv420p16, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut vv)
        .unwrap();
        yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb);
      });
    });
  }

  // High-bit SEMI-planar 4:2:0 (P010 LE) native vs row-stage — the P2
  // u16 fast tier for the high-bit semi-planar P-format family. The native
  // wrapper de-interleaves + DE-PACKS the high-bit-packed Y and interleaved
  // UV into host-native LOGICAL u16 scratch, then reuses the planar high-bit
  // join. P010 packs the 10-bit logical value in the HIGH 10 bits
  // (`logical << 6`); build a packed Y plane and interleaved UV plane from
  // the 8-bit pseudo-random source (`logical = byte << 2` → packed
  // `byte << 8`). The per-row triple de-pack is the cost the bench measures
  // against the row-stage tier (the hard bench-gate).
  let mut y_p010 = std::vec![0u16; SRC_W * SRC_H];
  let mut uv_p010 = std::vec![0u16; SRC_W * SRC_H / 2];
  for (d, &s) in y_p010.iter_mut().zip(y.iter()) {
    *d = (s as u16) << 8;
  }
  for (i, (&uu, &vv)) in u.iter().zip(v.iter()).enumerate() {
    uv_p010[i * 2] = (uu as u16) << 8;
    uv_p010[i * 2 + 1] = (vv as u16) << 8;
  }

  for (name, native) in [
    ("p010_rgb_u16_native", true),
    ("p010_rgb_u16_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = P010LeFrame::new(
          &y_p010,
          &uv_p010,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          SRC_W as u32,
        );
        let mut sink = MixedSinker::<P010, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
        p010_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb_u16);
      });
    });
  }

  for (name, native) in [
    ("yuv420p_luma_only_native", true),
    ("yuv420p_luma_only_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420pFrame::new(
          &y,
          &u,
          &v,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        );
        let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_luma(&mut luma)
        .unwrap();
        yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&luma);
      });
    });
  }

  // What a convert-then-resize pipeline pays for the convert step
  // alone, before any resizing.
  let mut full_rgb = std::vec![0u8; SRC_W * SRC_H * 3];
  group.bench_function("yuv420p_fullres_convert_baseline", |b| {
    b.iter(|| {
      let src = Yuv420pFrame::new(
        &y,
        &u,
        &v,
        SRC_W as u32,
        SRC_H as u32,
        SRC_W as u32,
        (SRC_W / 2) as u32,
        (SRC_W / 2) as u32,
      );
      let mut sink = MixedSinker::<Yuv420p>::new(SRC_W, SRC_H)
        .with_rgb(&mut full_rgb)
        .unwrap();
      yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
      black_box(&full_rgb);
    });
  });

  let mut packed = std::vec![0u8; SRC_W * SRC_H * 3];
  fill_pseudo_random(&mut packed, 0x4444);
  group.bench_function("rgb24_rgb_hsv_fused", |b| {
    b.iter(|| {
      let src = Rgb24Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3) as u32);
      let mut sink = MixedSinker::<Rgb24, AreaResampler>::with_resampler(
        SRC_W,
        SRC_H,
        AreaResampler::to(OUT_W, OUT_H),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut vv)
      .unwrap();
      rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
      black_box(&rgb);
    });
  });

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
