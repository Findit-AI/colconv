//! Fused-downscale coverage for the 8-bit **packed** 4:2:2 YUV NATIVE fast
//! tier (issue #123) — `Yuyv422` (YUYV `Y U Y V …`), `Uyvy422` (UYVY
//! `U Y V Y …`), `Yvyu422` (YVYU `Y V Y U …`). The packed analog of the
//! semi-planar non-4:2:0 native tier (`Nv16` / `Nv24` / `Nv42`).
//!
//! The wrapper de-PACKS each fully-interleaved source row into separate
//! Y (`w`) / U (`w / 2`) / V (`w / 2`) scratch planes at the per-format byte
//! offsets, then reuses the planar twin's non-4:2:0 join
//! ([`yuv_planar_process_native`](crate::sinker::mixed::planar_8bit::yuv_planar_process_native))
//! at `Yuv422p` geometry. The native tier bins the planes to the output grid
//! and converts ONCE per output row at output width, vs the packed row-stage
//! tier (`packed_yuv422_dual_resample`), which converts each source row at
//! source width then bins in RGB. The tiers differ in colour SEMANTICS
//! (native averages in YUV then converts; row-stage converts then averages in
//! RGB), so native is NOT byte-identical to row-stage — only within a small
//! tolerance in-gamut. Luma is bit-identical (both bin the same native Y).
//!
//! Per format the suite asserts (the #227 bar, re-pointed at the packed
//! de-pack instead of a semi-planar de-interleave):
//! - (a) the strongest check — packed native is BYTE-IDENTICAL to a `Yuv422p`
//!   NATIVE conversion of the de-packed planes (the de-pack-then-reuse claim;
//!   a wrong byte offset would diverge). This is the twin-parity oracle.
//! - (b) native vs the packed ROW-STAGE tier (the cv2 INTER_AREA oracle):
//!   luma bit-identical, in-gamut colour within `TOL_U8`, swept over matrices
//!   and the range flag.
//! - (c) constant planes bin exactly on both grids, so native reproduces the
//!   full-resolution direct conversion EXACTLY (the true 0-LSB case).
//! - the luma-only lazy-chroma carry-through (no chroma planning/alloc), the
//!   default-on flag, and the native/row-stage route-freeze (#186) contracts.

use crate::{
  KernelMatrix, PixelSink,
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Uyvy422, Uyvy422Row, Yuv422p, Yuyv422, Yuyv422Row, Yvyu422, Yvyu422Row, uyvy422_to, yuv422p_to,
    yuyv422_to, yvyu422_to,
  },
};
use mediaframe::frame::{Uyvy422Frame, Yuv422pFrame, Yuyv422Frame, Yvyu422Frame};

/// In-gamut per-channel tolerance between the native and packed row-stage
/// tiers. The two average in different domains (YUV vs RGB) and round
/// independently per output pixel; matches the planar non-4:2:0 twin's bound
/// (`resample_semi_planar.rs`). Native correctness itself is pinned EXACTLY by
/// the byte-identical `Yuv422p`-native twin-parity oracle below; this bound
/// only documents the row-stage semantic gap.
const TOL_U8: u8 = 5;

/// Per-pixel `(Y, U, V)` ramp: a wide-swinging Y (the full code range, via a
/// wrapping step) over a chroma ramp, chroma per chroma column (`w / 2 x h`,
/// a chroma row per Y row — 4:2:2). Used by the byte-exact twin-parity oracle
/// (which is a pure de-pack-then-reuse identity, valid at any code) and the
/// constant-plane / route tests — NOT by the row-stage tolerance comparison,
/// which needs a gamut-interior fixture (see [`textured`]).
fn yuv_ramp(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w / 2;
  let y: Vec<u8> = (0..w * h)
    .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
    .collect();
  let u: Vec<u8> = (0..cw * h).map(|i| 110 + (i % 24) as u8).collect();
  let v: Vec<u8> = (0..cw * h).map(|i| 120 + (i % 24) as u8).collect();
  (y, u, v)
}

/// A gamut-INTERIOR `(Y, U, V)` ramp (`w x h` luma, `w / 2 x h` chroma): every
/// code stays well inside the limited-range gamut, so the native (average-in-
/// YUV) and row-stage (convert-then-average) tiers diverge only by per-pixel
/// rounding, not by an RGB clamp. Mirrors the semi-planar non-4:2:0 twin's
/// `textured` fixture. Near the clamp the two genuinely diverge by more than
/// `TOL_U8` (the documented out-of-gamut caveat), which would mask a genuine
/// regression — hence the tolerance comparison uses this, not [`yuv_ramp`].
fn textured(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = w / 2;
  let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
  let u: Vec<u8> = (0..cw * h).map(|i| 110 + (i % 24) as u8).collect();
  let v: Vec<u8> = (0..cw * h).map(|i| 120 + (i % 24) as u8).collect();
  (y, u, v)
}

/// Builds a YUYV422 packed plane (`Y0 U0 Y1 V0` per 2-pixel group).
fn yuyv_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut buf = vec![0u8; 2 * w * h];
  for row in 0..h {
    for cx in 0..cw {
      let base = row * 2 * w + cx * 4;
      buf[base] = y[row * w + cx * 2];
      buf[base + 1] = u[row * cw + cx];
      buf[base + 2] = y[row * w + cx * 2 + 1];
      buf[base + 3] = v[row * cw + cx];
    }
  }
  buf
}

/// Builds a UYVY422 packed plane (`U0 Y0 V0 Y1` per group).
fn uyvy_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut buf = vec![0u8; 2 * w * h];
  for row in 0..h {
    for cx in 0..cw {
      let base = row * 2 * w + cx * 4;
      buf[base] = u[row * cw + cx];
      buf[base + 1] = y[row * w + cx * 2];
      buf[base + 2] = v[row * cw + cx];
      buf[base + 3] = y[row * w + cx * 2 + 1];
    }
  }
  buf
}

/// Builds a YVYU422 packed plane (`Y0 V0 Y1 U0` per group — V/U swapped).
fn yvyu_from(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut buf = vec![0u8; 2 * w * h];
  for row in 0..h {
    for cx in 0..cw {
      let base = row * 2 * w + cx * 4;
      buf[base] = y[row * w + cx * 2];
      buf[base + 1] = v[row * cw + cx];
      buf[base + 2] = y[row * w + cx * 2 + 1];
      buf[base + 3] = u[row * cw + cx];
    }
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

/// All-outputs `Yuv422p` downscale of the de-packed planes (the byte-exact
/// twin reference). Only RGB + luma are oracled against, but every output is
/// produced so the helper doubles as the colour/luma source.
#[allow(clippy::too_many_arguments)]
fn run_yuv422p(
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
  let cw = w / 2;
  let mut rgb = vec![0u8; n * 3];
  let mut luma = vec![0u8; n];
  {
    let frame = Yuv422pFrame::new(y, u, v, w as u32, h as u32, w as u32, cw as u32, cw as u32);
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv422p_to(&frame, full_range, matrix, &mut sink).unwrap();
  }
  (rgb, luma)
}

/// Generates the per-format native suite. `$build` packs the shared ramp at
/// the format's byte order; `$frame` / `$drv` are the frame builder + walker;
/// `$row` is the source row (for the route-flip test); `$y0..$v` the de-pack
/// byte offsets within each 4-byte group (documented for the reader).
macro_rules! packed_yuv_native_suite {
  (
    $mod:ident, $fmt:ident, $row:ident, $drv:ident, $frame:ident, $build:ident,
  ) => {
    mod $mod {
      use super::*;

      /// One all-outputs native-or-row-stage downscale of the packed frame.
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
          let frame = $frame::new(packed, w as u32, h as u32, (2 * w) as u32);
          let mut sink =
            MixedSinker::<$fmt, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
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
          $drv(&frame, full_range, matrix, &mut sink).unwrap();
        }
        (rgb, rgba, luma, luma_u16, hh, ss, vv)
      }

      /// (a) The strongest check: packed native is byte-identical to a
      /// `Yuv422p` NATIVE conversion of the de-packed planes, for RGB and
      /// luma, at several geometries. A wrong de-pack byte offset diverges.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_equals_yuv422p_native_on_depacked_planes() {
        for (w, h, ow, oh) in [
          (8usize, 8usize, 4usize, 4usize),
          (12, 10, 5, 4),
          (12, 10, 7, 6),
          (8, 10, 3, 4),
        ] {
          let (y, u, v) = yuv_ramp(w, h);
          let packed = $build(&y, &u, &v, w, h);

          let nv = run(&packed, w, h, ow, oh, true, KernelMatrix::Bt601, true);
          let (p_rgb, p_luma) =
            run_yuv422p(&y, &u, &v, w, h, ow, oh, true, KernelMatrix::Bt601, true);

          assert_eq!(
            nv.0, p_rgb,
            "packed native rgb == yuv422p native rgb ({w}x{h}->{ow}x{oh})"
          );
          assert_eq!(
            nv.2, p_luma,
            "packed native luma == yuv422p native luma ({w}x{h}->{ow}x{oh})"
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

      /// (b) Native vs the packed ROW-STAGE tier (the cv2 INTER_AREA oracle):
      /// luma bit-identical (both bin the same native Y), in-gamut colour
      /// within `TOL_U8`. The `with_native(true)` vs `with_native(false)`
      /// differential, swept over matrices and the range flag.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_within_tolerance_of_row_stage() {
        let (w, h) = (12, 10);
        let (y, u, v) = textured(w, h);
        let packed = $build(&y, &u, &v, w, h);
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

      /// (c) Constant planes bin exactly on both grids, so native reproduces
      /// the full-resolution direct conversion EXACTLY — the true 0-LSB case.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_solid_frame_exact() {
        let (w, h) = (8, 8);
        let cw = w / 2;
        let y = vec![120u8; w * h];
        let u = vec![90u8; cw * h];
        let v = vec![170u8; cw * h];
        let packed = $build(&y, &u, &v, w, h);

        // Full-resolution direct conversion (identity sink).
        let mut full_rgb = vec![0u8; w * h * 3];
        {
          let frame = $frame::new(&packed, w as u32, h as u32, (2 * w) as u32);
          let mut sink = MixedSinker::<$fmt>::new(w, h)
            .with_rgb(&mut full_rgb)
            .unwrap();
          $drv(&frame, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
          MixedSinker::<$fmt, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
            .unwrap();
        assert!(sink.native(), "with_native must default to true");
        assert!(
          !sink.with_native(false).native(),
          "with_native(false) must disable the tier"
        );
      }

      /// A mid-frame `set_native` flip splits one frame across two
      /// independent stream machines and must reject as the deterministic
      /// `NativeRouteChanged` (the #186 CHECK-before / SET-after guard).
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_to_rowstage_route_flip_mid_frame_rejected() {
        let (w, h) = (8, 8);
        let (y, u, v) = yuv_ramp(w, h);
        let packed = $build(&y, &u, &v, w, h);
        let mut luma = vec![0u8; 4 * 4];
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_luma(&mut luma)
            .unwrap();
        sink.begin_frame(w as u32, h as u32).unwrap();
        // Row 0 freezes the route = native.
        sink
          .process($row::new(&packed[0..2 * w], 0, KernelMatrix::Bt601, true))
          .expect("native row 0 freezes the route and succeeds");
        sink.set_native(false);
        let err = sink
          .process($row::new(
            &packed[2 * w..4 * w],
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

      /// Native survives a frame restart on a reused sink: `begin_frame`
      /// resets the join + the frozen route, so a second frame (the OTHER
      /// tier) downscales its own planes correctly.
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn native_reuses_join_and_resets_route_across_frames() {
        let (w, h) = (8, 8);
        let (y, u, v) = yuv_ramp(w, h);
        let packed = $build(&y, &u, &v, w, h);
        let mut luma = vec![0u8; 4 * 4];
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
            .unwrap()
            .with_native(true)
            .with_luma(&mut luma)
            .unwrap();
        let frame = $frame::new(&packed, w as u32, h as u32, (2 * w) as u32);
        // Frame 1: native, route constant across every row — no false reject.
        $drv(&frame, true, KernelMatrix::Bt601, &mut sink).unwrap();
        // Frame 2: flip to row-stage for the WHOLE frame; the per-frame reset
        // (in `begin_frame`) cleared the frozen route, so this is allowed.
        sink.set_native(false);
        $drv(&frame, true, KernelMatrix::Bt601, &mut sink)
          .expect("a new frame may pick the other tier; the route reset per frame");
      }
    }
  };
}

packed_yuv_native_suite!(
  yuyv422,
  Yuyv422,
  Yuyv422Row,
  yuyv422_to,
  Yuyv422Frame,
  yuyv_from,
);
packed_yuv_native_suite!(
  uyvy422,
  Uyvy422,
  Uyvy422Row,
  uyvy422_to,
  Uyvy422Frame,
  uyvy_from,
);
packed_yuv_native_suite!(
  yvyu422,
  Yvyu422,
  Yvyu422Row,
  yvyu422_to,
  Yvyu422Frame,
  yvyu_from,
);

/// A luma-only packed native sink must NOT plan or allocate any chroma state
/// — else luma-only Yuyv/Uyvy/Yvyu resampling depends on an unused chroma
/// allocation and can fail under memory pressure before producing luma (a
/// regression vs the Y-only row-stage path). Armed with the planar-native
/// chroma-planning failpoint (the join is shared with the planar twin): a
/// luma-only row leaves it unconsumed (so the run succeeds), while a colour
/// row reaches chroma planning and the failpoint fires. (Tested on Yuyv422 —
/// the de-pack is identical across the three formats.)
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_packed_native_skips_chroma_planning() {
  let (w, h) = (8, 8);
  let (y, u, v) = yuv_ramp(w, h);
  let packed = yuyv_from(&y, &u, &v, w, h);
  let frame = Yuyv422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

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
      MixedSinker::<Yuyv422, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    yuyv422_to(&frame, true, KernelMatrix::Bt601, &mut sink)
      .expect("luma-only native must not plan chroma");
  }
  assert_eq!(luma, y_ref, "luma-only native == area-downscaled Y");

  // Colour: the still-armed failpoint fires at chroma planning -> Err. This
  // both proves the failpoint is wired to chroma planning and consumes the
  // arm so it cannot leak to another test on this thread.
  let mut rgb = vec![0u8; 4 * 4 * 3];
  let mut sink =
    MixedSinker::<Yuyv422, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
      .unwrap()
      .with_native(true)
      .with_rgb(&mut rgb)
      .unwrap();
  assert!(
    yuyv422_to(&frame, true, KernelMatrix::Bt601, &mut sink).is_err(),
    "colour native must reach chroma planning (the armed failpoint fires)"
  );
}
