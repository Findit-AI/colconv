//! Fused-downscale coverage for the 8-bit **packed 4:4:4** YUV NATIVE fast tier
//! (issue #123) — [`Vuyx`](crate::source::Vuyx) (bytes `V U Y X` per pixel,
//! `X` padding). The 4:4:4 analog of the packed 4:2:2 native tier
//! (`Yuyv422` / `Uyvy422` / `Yvyu422`).
//!
//! The wrapper de-PACKS each fully-interleaved source row into separate
//! Y (`w`) / U (`w`) / V (`w`) scratch planes — all FULL width, since 4:4:4 has
//! no chroma subsampling — at the per-format byte offsets, then reuses the
//! planar twin's non-4:2:0 join
//! ([`yuv_planar_process_native`](crate::sinker::mixed::planar_8bit::yuv_planar_process_native))
//! at [`Yuv444p`](crate::source::Yuv444p) geometry (`chroma_vsub = 1`,
//! `chroma_w = w`). The native tier bins the planes to the output grid and
//! converts ONCE per output row at output width, vs the packed row-stage tier
//! ([`packed_yuv444_triple_resample`](crate::sinker::mixed::packed_yuv444_triple_resample)),
//! which converts each source row at source width then bins in RGB. The tiers
//! differ in colour SEMANTICS (native averages in YUV then converts; row-stage
//! converts then averages in RGB), so native is NOT byte-identical to row-stage
//! — only within a small tolerance in-gamut. Luma is bit-identical (both bin the
//! same native Y).
//!
//! The suite asserts (the #227 bar, re-pointed at the packed 4:4:4 de-pack):
//! - (a) the strongest check — packed native is BYTE-IDENTICAL to a `Yuv444p`
//!   NATIVE conversion of the de-packed planes (the de-pack-then-reuse claim; a
//!   wrong byte offset would diverge). This is the twin-parity oracle.
//! - (b) native vs the packed ROW-STAGE tier (the cv2 INTER_AREA oracle): luma
//!   bit-identical, in-gamut colour within `TOL_U8`, swept over matrices and the
//!   range flag.
//! - (c) constant planes bin exactly on both grids, so native reproduces the
//!   full-resolution direct conversion EXACTLY (the true 0-LSB case).
//! - the luma-only lazy-chroma carry-through (no chroma planning / alloc), the
//!   default-on flag, and the native/row-stage route-freeze (#186) contracts.

use crate::{
  KernelMatrix, PixelSink,
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{Vuyx, VuyxRow, Yuv444p, vuyx_to, yuv444p_to},
};
use mediaframe::frame::{VuyxFrame, Yuv444pFrame};

/// In-gamut per-channel tolerance between the native and packed row-stage tiers.
/// The two average in different domains (YUV vs RGB) and round independently per
/// output pixel; matches the packed 4:2:2 twin's bound. Native correctness itself
/// is pinned EXACTLY by the byte-identical `Yuv444p`-native twin-parity oracle
/// below; this bound only documents the row-stage semantic gap.
const TOL_U8: u8 = 5;

/// Per-pixel `(Y, U, V)` ramp (`w x h` each — 4:4:4, every pixel its own
/// chroma): a wide-swinging Y over chroma ramps. Used by the byte-exact
/// twin-parity oracle (a pure de-pack-then-reuse identity, valid at any code)
/// and the constant-plane / route tests — NOT by the row-stage tolerance
/// comparison, which needs a gamut-interior fixture (see [`textured`]).
fn yuv_ramp(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let y: Vec<u8> = (0..w * h)
    .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
    .collect();
  let u: Vec<u8> = (0..w * h).map(|i| 110 + (i % 24) as u8).collect();
  let v: Vec<u8> = (0..w * h).map(|i| 120 + (i % 24) as u8).collect();
  (y, u, v)
}

/// A gamut-INTERIOR `(Y, U, V)` ramp (`w x h` each): every code stays well inside
/// the limited-range gamut, so the native (average-in-YUV) and row-stage
/// (convert-then-average) tiers diverge only by per-pixel rounding, not by an RGB
/// clamp.
fn textured(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
  let u: Vec<u8> = (0..w * h).map(|i| 110 + (i % 24) as u8).collect();
  let v: Vec<u8> = (0..w * h).map(|i| 120 + (i % 24) as u8).collect();
  (y, u, v)
}

/// Builds a Vuyx packed plane (`V U Y X` per pixel; `X` padding = 0). 4:4:4 —
/// every pixel carries its own V / U / Y. `w x 4` bytes per row.
fn vuyx_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let mut buf = vec![0u8; 4 * w * h];
  for i in 0..w * h {
    let base = i * 4;
    buf[base] = v[i]; // V
    buf[base + 1] = u[i]; // U
    buf[base + 2] = y[i]; // Y
    buf[base + 3] = 0; // X padding
  }
  buf
}

/// `(rgb, rgba, luma, luma_u16, hsv_h, hsv_s, hsv_v)` of one downscale.
type Outs = (
  Vec<u8>,
  Vec<u8>,
  Vec<u8>,
  Vec<u16>,
  Vec<u8>,
  Vec<u8>,
  Vec<u8>,
);

/// All-outputs `Yuv444p` downscale of the de-packed planes (the byte-exact twin
/// reference). Only RGB + luma are oracled against, but every output is produced
/// so the helper doubles as the colour/luma source.
#[allow(clippy::too_many_arguments)]
fn run_yuv444p(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  w: usize,
  h: usize,
  ow: usize,
  oh: usize,
  full_range: bool,
  matrix: KernelMatrix,
  native: bool,
) -> (Vec<u8>, Vec<u8>) {
  let n = ow * oh;
  let mut rgb = vec![0u8; n * 3];
  let mut luma = vec![0u8; n];
  {
    let frame = Yuv444pFrame::new(y, u, v, w as u32, h as u32, w as u32, w as u32, w as u32);
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv444p_to(&frame, full_range, matrix, &mut sink).unwrap();
  }
  (rgb, luma)
}

/// One all-outputs native-or-row-stage downscale of the packed Vuyx frame.
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
) -> Outs {
  let n = ow * oh;
  let mut rgb = vec![0u8; n * 3];
  let mut rgba = vec![0u8; n * 4];
  let mut luma = vec![0u8; n];
  let mut luma_u16 = vec![0u16; n];
  let (mut hh, mut ss, mut vv) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
  {
    let frame = VuyxFrame::try_new(packed, w as u32, h as u32, (4 * w) as u32).unwrap();
    let mut sink =
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    vuyx_to(&frame, full_range, matrix, &mut sink).unwrap();
  }
  (rgb, rgba, luma, luma_u16, hh, ss, vv)
}

/// (a) The strongest check: packed native is byte-identical to a `Yuv444p`
/// NATIVE conversion of the de-packed planes, for RGB and luma, at several
/// geometries. A wrong de-pack byte offset diverges.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_equals_yuv444p_native_on_depacked_planes() {
  for (w, h, ow, oh) in [
    (8usize, 8usize, 4usize, 4usize),
    (12, 10, 5, 4),
    (12, 10, 7, 6),
    (8, 10, 3, 4),
  ] {
    let (y, u, v) = yuv_ramp(w, h);
    let packed = vuyx_from(&y, &u, &v, w, h);

    let nv = run(&packed, w, h, ow, oh, true, KernelMatrix::Bt601, true);
    let (p_rgb, p_luma) = run_yuv444p(&y, &u, &v, w, h, ow, oh, true, KernelMatrix::Bt601, true);

    assert_eq!(
      nv.0, p_rgb,
      "packed native rgb == yuv444p native rgb ({w}x{h}->{ow}x{oh})"
    );
    assert_eq!(
      nv.2, p_luma,
      "packed native luma == yuv444p native luma ({w}x{h}->{ow}x{oh})"
    );
    // RGBA colour mirrors RGB with an opaque-alpha pad.
    for (px, rgb_px) in nv.1.chunks_exact(4).zip(p_rgb.chunks_exact(3)) {
      assert_eq!(&px[..3], rgb_px, "native rgba colour == rgb");
      assert_eq!(px[3], 0xFF, "native rgba alpha opaque");
    }
    // luma_u16 is the binned Y zero-extended.
    let lu16: Vec<u16> = p_luma.iter().map(|&b| b as u16).collect();
    assert_eq!(nv.3, lu16, "native luma_u16 == binned Y zero-extended");
  }
}

/// (b) Native vs the packed ROW-STAGE tier (the cv2 INTER_AREA oracle): luma
/// bit-identical (both bin the same native Y), in-gamut colour within `TOL_U8`.
/// The `with_native(true)` vs `with_native(false)` differential, swept over
/// matrices and the range flag.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_within_tolerance_of_row_stage() {
  let (w, h) = (12, 10);
  let (y, u, v) = textured(w, h);
  let packed = vuyx_from(&y, &u, &v, w, h);
  for (ow, oh) in [(6, 5), (4, 4), (7, 6), (5, 3)] {
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

/// (c) Constant planes bin exactly on both grids, so native reproduces the
/// full-resolution direct conversion EXACTLY — the true 0-LSB case.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_solid_frame_exact() {
  let (w, h) = (8, 8);
  let y = vec![120u8; w * h];
  let u = vec![90u8; w * h];
  let v = vec![170u8; w * h];
  let packed = vuyx_from(&y, &u, &v, w, h);

  // Full-resolution direct conversion (identity sink).
  let mut full_rgb = vec![0u8; w * h * 3];
  {
    let frame = VuyxFrame::try_new(&packed, w as u32, h as u32, (4 * w) as u32).unwrap();
    let mut sink = MixedSinker::<Vuyx>::new(w, h)
      .with_rgb(&mut full_rgb)
      .unwrap();
    vuyx_to(&frame, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
    MixedSinker::<Vuyx, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4)).unwrap();
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
  let (y, u, v) = yuv_ramp(w, h);
  let packed = vuyx_from(&y, &u, &v, w, h);
  let mut luma = vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Vuyx, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
    .unwrap()
    .with_native(true)
    .with_luma(&mut luma)
    .unwrap();
  sink.begin_frame(w as u32, h as u32).unwrap();
  // Row 0 freezes the route = native.
  sink
    .process(VuyxRow::new(
      &packed[0..4 * w],
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("native row 0 freezes the route and succeeds");
  sink.set_native(false);
  let err = sink
    .process(VuyxRow::new(
      &packed[4 * w..8 * w],
      1,
      KernelMatrix::Bt601,
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
  let (y, u, v) = yuv_ramp(w, h);
  let packed = vuyx_from(&y, &u, &v, w, h);
  let mut luma = vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Vuyx, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
    .unwrap()
    .with_native(true)
    .with_luma(&mut luma)
    .unwrap();
  let frame = VuyxFrame::try_new(&packed, w as u32, h as u32, (4 * w) as u32).unwrap();
  // Frame 1: native, route constant across every row — no false reject.
  vuyx_to(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
  // Frame 2: flip to row-stage for the WHOLE frame; the per-frame reset (in
  // `begin_frame`) cleared the frozen route, so this is allowed.
  sink.set_native(false);
  vuyx_to(&frame, true, KernelMatrix::Bt601, &mut sink)
    .expect("a new frame may pick the other tier; the route reset per frame");
}

/// A luma-only packed 4:4:4 native sink must NOT plan or allocate any chroma
/// state — else luma-only Vuyx resampling depends on an unused chroma allocation
/// and can fail under memory pressure before producing luma. Armed with the
/// planar-native chroma-planning failpoint (the join is shared with the planar
/// twin): a luma-only row leaves it unconsumed (the run succeeds), while a colour
/// row reaches chroma planning and fires.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_native_skips_chroma_planning() {
  let (w, h) = (8, 8);
  let (y, u, v) = yuv_ramp(w, h);
  let packed = vuyx_from(&y, &u, &v, w, h);
  let frame = VuyxFrame::try_new(&packed, w as u32, h as u32, (4 * w) as u32).unwrap();

  // 2x2-block area mean of the Y plane — the luma reference.
  let mut y_ref = vec![0u8; 4 * 4];
  for oy in 0..4 {
    for ox in 0..4 {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += y[(oy * 2 + dy) * w + ox * 2 + dx] as u32;
        }
      }
      y_ref[oy * 4 + ox] = ((s + 2) / 4) as u8;
    }
  }

  crate::sinker::mixed::arm_planar_native_chroma_failure();

  // Luma-only: the armed chroma failpoint is never reached -> Ok.
  let mut luma = vec![0u8; 4 * 4];
  {
    let mut sink =
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    vuyx_to(&frame, true, KernelMatrix::Bt601, &mut sink)
      .expect("luma-only native must not plan chroma");
  }
  assert_eq!(luma, y_ref, "luma-only native == area-downscaled Y");

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This both
  // proves the failpoint is wired to chroma planning and consumes the arm.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink = MixedSinker::<Vuyx, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
    .unwrap()
    .with_native(true)
    .with_rgb(&mut rgb)
    .unwrap();
  assert!(
    vuyx_to(&frame, true, KernelMatrix::Bt601, &mut sink).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}
