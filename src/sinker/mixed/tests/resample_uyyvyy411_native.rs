//! Fused-downscale coverage for the 8-bit **packed** 4:1:1 YUV NATIVE fast tier
//! (issue #123) — `Uyyvyy411` (`AV_PIX_FMT_UYYVYY411`, DV legacy;
//! `U Y Y V Y Y` per 6-byte / 4-pixel block, one `(U, V)` chroma pair shared by
//! FOUR luma samples). The 4:1:1 analog of the packed 4:2:2 8-bit native tier
//! (`Yuyv422` / `Uyvy422` / `Yvyu422`); it differs only in the de-pack stride
//! and the chroma width (`w / 4` vs `w / 2`).
//!
//! The wrapper de-PACKS the fully-interleaved source row into separate
//! Y (`w`) / U (`w / 4`) / V (`w / 4`) scratch planes at the fixed byte offsets
//! (`U0 @ 0, Y0 @ 1, Y1 @ 2, V0 @ 3, Y2 @ 4, Y3 @ 5` per group), then reuses the
//! planar twin's non-4:2:0 join
//! ([`yuv_planar_process_native`](crate::sinker::mixed::planar_8bit::yuv_planar_process_native))
//! at `Yuv411p` geometry (`chroma_vsub = 1`, chroma plan a plain `area` over the
//! `w / 4`-wide source chroma). The native tier bins the planes to the output
//! grid and converts ONCE per output row at output width, vs the packed
//! row-stage tier (`packed_yuv_dual_resample`), which converts each source row at
//! source width then bins in RGB. The tiers differ in colour SEMANTICS (native
//! averages in YUV then converts; row-stage converts then averages in RGB), so
//! native is NOT byte-identical to row-stage — only within a small tolerance
//! in-gamut. Luma is bit-identical (both bin the same native Y).
//!
//! There is no `Yuv411p` NATIVE tier (Yuv411p is row-stage only), so unlike the
//! 4:2:2 / 4:4:4 packed suites there is no planar-native twin to cross-check the
//! de-pack against. Instead the de-pack correctness is pinned EXACTLY by an
//! INDEPENDENT bin-then-convert oracle: extract the logical Y / U / V planes
//! straight from the wire bytes (a separate de-pack), area-bin each plane to
//! OUTPUT resolution (Y from `w x h`, chroma from `w / 4 x h` — horizontal-only
//! subsample), then convert the full-output-width planes ONCE through an
//! identity-resolution `Yuv444p` sink (chroma binned to full output width).
//! A wrong de-pack byte offset / chroma ratio would diverge.
//!
//! The suite asserts:
//! - `native_equals_bin_then_convert_oracle`: the GROUND-TRUTH check — native
//!   output (RGB, RGBA, luma, luma_u16) EXACTLY equals the oracle.
//! - `native_within_tolerance_of_row_stage`: the cv2 INTER_AREA parity bound +
//!   the `with_native(true)` vs `with_native(false)` differential, swept over
//!   matrices and the range flag (luma bit-identical, in-gamut RGB within
//!   `TOL_U8`).
//! - `native_solid_frame_exact`: constant planes bin exactly on both grids, so
//!   native reproduces the full-resolution direct conversion EXACTLY (0-LSB).
//! - the luma-only lazy-chroma carry-through (no chroma planning/alloc), the
//!   default-on flag, and the native/row-stage route-freeze (#186) contracts.

use crate::{
  KernelMatrix, PixelSink,
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{Uyyvyy411, Uyyvyy411Row, Yuv444p, uyyvyy411_to, yuv444p_to},
};
use mediaframe::frame::{Uyyvyy411Frame, Yuv444pFrame};

/// In-gamut per-channel tolerance between the native and packed row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round independently per
/// output pixel; matches the packed 4:2:2 native suite's bound. Native
/// correctness itself is pinned EXACTLY by the bin-then-convert oracle below;
/// this bound only documents the row-stage semantic gap.
const TOL_U8: u8 = 5;

const M: KernelMatrix = KernelMatrix::Bt601;

/// Byte stride of a `w`-wide UYYVYY411 row: `w * 3 / 2` (12 bpp, no padding).
const fn stride(w: usize) -> u32 {
  (w * 3 / 2) as u32
}

/// Builds a UYYVYY411 packed plane from separate Y (`w x h`) and U / V
/// (`w / 4 x h`) planes. Layout per 6-byte / 4-pixel block:
/// `U0, Y0, Y1, V0, Y2, Y3` (4:1:1 — one chroma pair per 4 luma). `w` must be a
/// multiple of 4.
fn uyyvyy411_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  assert_eq!(w & 3, 0, "uyyvyy411 width must be a multiple of 4");
  let cw = w / 4;
  let mut buf = vec![0u8; w * 3 / 2 * h];
  for row in 0..h {
    let base = row * w * 3 / 2;
    for cx in 0..cw {
      let blk = base + cx * 6;
      buf[blk] = u[row * cw + cx];
      buf[blk + 1] = y[row * w + cx * 4];
      buf[blk + 2] = y[row * w + cx * 4 + 1];
      buf[blk + 3] = v[row * cw + cx];
      buf[blk + 4] = y[row * w + cx * 4 + 2];
      buf[blk + 5] = y[row * w + cx * 4 + 3];
    }
  }
  buf
}

/// A gamut-INTERIOR `(Y, U, V)` ramp (`w x h` luma, `w / 4 x h` chroma): every
/// code stays well inside the limited-range gamut, so the native (average-in-
/// YUV) and row-stage (convert-then-average) tiers diverge only by per-pixel
/// rounding, not by an RGB clamp. Used by the tolerance comparison and the
/// oracle / route tests.
fn textured(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w / 4;
  let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
  let u: Vec<u8> = (0..cw * h).map(|i| 110 + (i % 24) as u8).collect();
  let v: Vec<u8> = (0..cw * h).map(|i| 120 + (i % 24) as u8).collect();
  (y, u, v)
}

// ---- Independent de-pack (mirrors the wrapper's `chunks_exact(6)` de-pack) ----
//
// Extracts the logical planes straight from the wire bytes, INDEPENDENTLY of the
// native wrapper's de-pack — so a wrong byte offset / chroma ratio in the
// wrapper is caught by the oracle comparison.

/// De-pack a UYYVYY411 plane into the logical Y plane (`w x h`).
fn logical_y(packed: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 4;
  let mut y = vec![0u8; w * h];
  for row in 0..h {
    let base = row * w * 3 / 2;
    for cx in 0..cw {
      let blk = base + cx * 6;
      let o = row * w + cx * 4;
      y[o] = packed[blk + 1];
      y[o + 1] = packed[blk + 2];
      y[o + 2] = packed[blk + 4];
      y[o + 3] = packed[blk + 5];
    }
  }
  y
}

/// De-pack a UYYVYY411 plane into the logical U / V planes (`w / 4 x h`).
fn logical_uv(packed: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
  let cw = w / 4;
  let mut u = vec![0u8; cw * h];
  let mut v = vec![0u8; cw * h];
  for row in 0..h {
    let base = row * w * 3 / 2;
    for cx in 0..cw {
      let blk = base + cx * 6;
      u[row * cw + cx] = packed[blk];
      v[row * cw + cx] = packed[blk + 3];
    }
  }
  (u, v)
}

/// Exact box-coverage area mean (round-half-up) of an `in_w x in_h` u8 plane to
/// `out_w x out_h`, INDEPENDENTLY reimplementing `AxisSpans::area`: each output
/// cell covers the source interval `[j * in / out, (j + 1) * in / out)` in
/// `1 / out` units, weighting each touched source cell by its `1 / out`-unit
/// overlap; the separable 2D weight is the product of the per-axis overlaps and
/// the normalization denominator is `in_w * in_h` (= the per-axis `in_w` and
/// `in_h` denominators). Handles BOTH downscale and upscale (the 4:1:1 chroma
/// `w / 4 -> out_w` may upsample), so it cannot divide by zero like an
/// integer-ratio block mean. The native join's [`AreaResampler`] computes the
/// identical coverage, so the oracle is exact.
fn bin_to(plane: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
  // Per-axis overlap weights for output index `j`: `(start, weights)` over the
  // source cells, in `1 / out`-unit coverage (each weight in `0..=out`).
  let axis = |src: usize, out: usize| -> Vec<(usize, Vec<u64>)> {
    let (src64, out64) = (src as u64, out as u64);
    (0..out as u64)
      .map(|j| {
        let lo = j * src64;
        let hi = lo + src64;
        let start = (lo / out64) as usize;
        let mut ws = Vec::new();
        let mut i = start as u64;
        while i < hi.div_ceil(out64) {
          let cell_lo = i * out64;
          let cell_hi = cell_lo + out64;
          ws.push(cell_hi.min(hi) - cell_lo.max(lo));
          i += 1;
        }
        (start, ws)
      })
      .collect()
  };
  let hx = axis(in_w, out_w);
  let vy = axis(in_h, out_h);
  let denom = (in_w * in_h) as u64;
  let mut out = vec![0u8; out_w * out_h];
  for (oy, (ys, yw)) in vy.iter().enumerate() {
    for (ox, (xs, xw)) in hx.iter().enumerate() {
      let mut s = 0u64;
      for (dy, &wy) in yw.iter().enumerate() {
        for (dx, &wx) in xw.iter().enumerate() {
          s += plane[(ys + dy) * in_w + xs + dx] as u64 * wy * wx;
        }
      }
      out[oy * out_w + ox] = ((s + denom / 2) / denom) as u8;
    }
  }
  out
}

/// `(rgb, rgba, luma, luma_u16)` of one downscale through a tier. `native`
/// toggles the bin-then-convert native fast tier vs the convert-then-bin
/// row-stage tier.
#[allow(clippy::too_many_arguments)]
fn run(
  packed: &[u8],
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  matrix: KernelMatrix,
  native: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>) {
  let n = ow * oh;
  let mut rgb = vec![0u8; n * 3];
  let mut rgba = vec![0u8; n * 4];
  let mut luma = vec![0u8; n];
  let mut luma_u16 = vec![0u16; n];
  {
    let frame = Uyyvyy411Frame::new(packed, w as u32, h as u32, stride(w));
    let mut sink =
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    uyyvyy411_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
  }
  (rgb, rgba, luma, luma_u16)
}

/// The bin-then-convert oracle: de-pack the wire, area-bin every plane to OUTPUT
/// resolution (Y from `w x h`, chroma from `w / 4 x h`), then convert the
/// full-output-width planes ONCE through an identity-resolution `Yuv444p` sink
/// (chroma binned to full output width, so a 4:4:4 sink). Returns
/// `(rgb, rgba, luma, luma_u16)` to match [`run`].
fn oracle(
  packed: &[u8],
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  matrix: KernelMatrix,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>) {
  let cw = w / 4;
  let yl = logical_y(packed, w, h);
  let (u, v) = logical_uv(packed, w, h);
  let yb = bin_to(&yl, w, h, ow, oh);
  let ub = bin_to(&u, cw, h, ow, oh);
  let vb = bin_to(&v, cw, h, ow, oh);
  let n = ow * oh;
  let mut rgb = vec![0u8; n * 3];
  let mut rgba = vec![0u8; n * 4];
  {
    let f = Yuv444pFrame::new(
      &yb, &ub, &vb, ow as u32, oh as u32, ow as u32, ow as u32, ow as u32,
    );
    let mut sink = MixedSinker::<Yuv444p>::new(ow, oh)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv444p_to(&f, full_range, sink.set_kernel_matrix(matrix)).unwrap();
  }
  let luma_u16: Vec<u16> = yb.iter().map(|&by| by as u16).collect();
  (rgb, rgba, yb, luma_u16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_equals_bin_then_convert_oracle() {
  // Several geometries, including odd output dims (non-integer ratio) — the
  // join bins both planes against the SAME output grid, so the oracle must
  // bin identically. Widths are multiples of 4 (the UYYVYY411 constraint).
  for (w, h, ow, oh) in [
    (8usize, 8usize, 4usize, 4usize),
    (16, 10, 4, 5),
    (16, 10, 7, 6),
    (8, 10, 3, 4),
  ] {
    let (y, u, v) = textured(w, h);
    let packed = uyyvyy411_from(&y, &u, &v, w, h);

    let (n_rgb, n_rgba, n_luma, n_lu16) = run(&packed, w, h, ow, oh, true, M, true);
    let (o_rgb, o_rgba, o_luma, o_lu16) = oracle(&packed, w, h, ow, oh, true, M);

    assert_eq!(
      n_rgb, o_rgb,
      "native rgb must equal the bin-then-convert oracle ({w}x{h}->{ow}x{oh})"
    );
    assert_eq!(
      n_rgba, o_rgba,
      "native rgba must equal the oracle ({w}x{h}->{ow}x{oh})"
    );
    assert_eq!(
      n_luma, o_luma,
      "native luma must equal the binned Y ({w}x{h}->{ow}x{oh})"
    );
    assert_eq!(
      n_lu16, o_lu16,
      "native luma_u16 must equal the binned Y zero-extended ({w}x{h}->{ow}x{oh})"
    );
    // RGBA colour mirrors RGB with an opaque-alpha pad.
    for (px, rgb_px) in n_rgba.chunks_exact(4).zip(n_rgb.chunks_exact(3)) {
      assert_eq!(&px[..3], rgb_px, "native rgba colour == rgb");
      assert_eq!(px[3], 0xFF, "native rgba alpha opaque");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_within_tolerance_of_row_stage() {
  let (w, h) = (16, 10);
  let (y, u, v) = textured(w, h);
  let packed = uyyvyy411_from(&y, &u, &v, w, h);
  for (ow, oh) in [(8, 5), (4, 4), (7, 6), (5, 3)] {
    for full_range in [false, true] {
      for matrix in [
        KernelMatrix::Bt601,
        KernelMatrix::Bt709,
        KernelMatrix::Bt2020Ncl,
      ] {
        let native = run(&packed, w, h, ow, oh, full_range, matrix, true);
        let row = run(&packed, w, h, ow, oh, full_range, matrix, false);
        assert_eq!(
          native.2, row.2,
          "luma bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
        );
        assert_eq!(
          native.3, row.3,
          "luma_u16 bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
        );
        for (name, a, b) in [("rgb", &native.0, &row.0), ("rgba", &native.1, &row.1)] {
          for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
              x.abs_diff(*y) <= TOL_U8,
              "{name} {ow}x{oh} fr={full_range} {matrix:?} idx {i}: native {x} vs row {y}"
            );
          }
        }
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_solid_frame_exact() {
  // Constant planes bin exactly on both grids, so native reproduces the
  // full-resolution direct conversion EXACTLY — the true 0-LSB case.
  let (w, h) = (8, 8);
  let cw = w / 4;
  let y = vec![120u8; w * h];
  let u = vec![90u8; cw * h];
  let v = vec![170u8; cw * h];
  let packed = uyyvyy411_from(&y, &u, &v, w, h);

  // Full-resolution direct conversion (identity sink).
  let mut full_rgb = vec![0u8; w * h * 3];
  {
    let frame = Uyyvyy411Frame::new(&packed, w as u32, h as u32, stride(w));
    let mut sink = MixedSinker::<Uyyvyy411>::new(w, h)
      .with_rgb(&mut full_rgb)
      .unwrap();
    uyyvyy411_to(&frame, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let out = run(&packed, w, h, 4, 4, false, KernelMatrix::Bt709, true);
  for px in out.0.chunks_exact(3) {
    assert_eq!(
      (px[0], px[1], px[2]),
      (full_rgb[0], full_rgb[1], full_rgb[2]),
      "native solid rgb == full-res conversion"
    );
  }
  assert!(out.2.iter().all(|&l| l == 120), "native solid luma == Y");
}

/// `with_native(true)` is the builder default for this format.
#[test]
fn native_is_default_on() {
  let sink =
    MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4)).unwrap();
  assert!(sink.native(), "with_native must default to true");
  assert!(
    !sink.with_native(false).native(),
    "with_native(false) must disable the tier"
  );
}

/// A mid-frame `set_native` flip splits one frame across two independent stream
/// machines and must reject as the deterministic `NativeRouteChanged` (the #186
/// CHECK-before / SET-after guard).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_to_rowstage_route_flip_mid_frame_rejected() {
  let (w, h) = (8, 8);
  let (y, u, v) = textured(w, h);
  let packed = uyyvyy411_from(&y, &u, &v, w, h);
  let row_bytes = w * 3 / 2;
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(w as u32, h as u32).unwrap();
  // Row 0 freezes the route = native.
  sink
    .process(Uyyvyy411Row::for_tests(&packed[0..row_bytes], 0, M, true))
    .expect("native row 0 freezes the route and succeeds");
  sink.set_native(false);
  let err = sink
    .process(Uyyvyy411Row::for_tests(
      &packed[row_bytes..row_bytes * 2],
      1,
      M,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::NativeRouteChanged(_)),
    "a native -> row-stage mid-frame route flip must reject as \
     NativeRouteChanged, got {err:?}"
  );
}

/// Native survives a frame restart on a reused sink: `begin_frame` resets the
/// join + the frozen route, so a second frame (the OTHER tier) downscales its
/// own planes correctly.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_reuses_join_and_resets_route_across_frames() {
  let (w, h) = (8, 8);
  let (y, u, v) = textured(w, h);
  let packed = uyyvyy411_from(&y, &u, &v, w, h);
  let mut luma = vec![0u8; 4 * 4];
  let mut sink =
    MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  let frame = Uyyvyy411Frame::new(&packed, w as u32, h as u32, stride(w));
  // Frame 1: native, route constant across every row — no false reject.
  uyyvyy411_to(&frame, true, sink.set_kernel_matrix(M)).unwrap();
  // Frame 2: flip to row-stage for the WHOLE frame; the per-frame reset (in
  // `begin_frame`) cleared the frozen route, so this is allowed.
  sink.set_native(false);
  uyyvyy411_to(&frame, true, sink.set_kernel_matrix(M))
    .expect("a new frame may pick the other tier; the route reset per frame");
}

/// A luma-only packed native sink must NOT plan or allocate any chroma state —
/// else luma-only Uyyvyy411 resampling depends on an unused chroma allocation
/// and can fail under memory pressure before producing luma (a regression vs the
/// Y-only row-stage path). Armed with the planar-native chroma-planning
/// failpoint (the join is shared with the planar twin): a luma-only row leaves
/// it unconsumed (so the run succeeds), while a colour row reaches chroma
/// planning and the failpoint fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_packed_native_skips_chroma_planning() {
  let (w, h) = (8, 8);
  let (y, u, v) = textured(w, h);
  let packed = uyyvyy411_from(&y, &u, &v, w, h);
  let frame = Uyyvyy411Frame::new(&packed, w as u32, h as u32, stride(w));

  // 2x2-block area mean of the Y plane — the luma reference.
  let y_ref = bin_to(&logical_y(&packed, w, h), w, h, 4, 4);

  crate::sinker::mixed::arm_planar_native_chroma_failure();

  // Luma-only: the armed chroma failpoint is never reached -> Ok.
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    uyyvyy411_to(&frame, true, sink.set_kernel_matrix(M))
      .expect("luma-only native must not plan chroma");
  }
  assert_eq!(luma, y_ref, "luma-only native == area-downscaled Y");

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This both
  // proves the failpoint is wired to chroma planning and consumes the arm so it
  // cannot leak to another test on this thread.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Uyyvyy411, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    uyyvyy411_to(&frame, true, sink.set_kernel_matrix(M)).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}
