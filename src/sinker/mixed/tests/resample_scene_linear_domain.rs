//! RFC #238 #244 — scene-referred ([`LinearMode::SceneReferred`])
//! [`AveragingDomain::Linear`] coverage for the planar 8-bit YUV family
//! (`Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p`).
//!
//! Scene-referred linear averaging decodes each source pixel through the
//! **same affine YUV→RGB matrix** as the production Q15 kernel, but in
//! unclamped real-valued `f32` (preserving out-of-gamut super-black /
//! super-white / saturated-chroma excursions the 8-bit convert clamp would
//! discard), lifts that to linear light via the [`TransferFunction`] EOTF
//! (whose odd-symmetric extrapolation handles out-of-`[0, 1]` inputs),
//! area-bins in linear light, re-encodes via the OETF, and clamps ONLY at the
//! output. These tests pin:
//!
//! - **`*_scene_referred_equals_independent_unclamped_oracle`** — the
//!   scene-referred RGB output equals a from-scratch oracle that decodes the
//!   same full-resolution RGB with an INDEPENDENT real-valued matrix (the
//!   same coefficients, re-derived in the test as `q15 / 32768`), EOTFs,
//!   2x2-block-means in linear light, OETFs, and clamps — within a documented
//!   1-LSB f32-rounding tolerance.
//! - **`scene_and_display_referred_differ_on_out_of_gamut`** — on content with
//!   out-of-gamut excursions (saturated chroma driving a channel `< 0` or
//!   `> 1` before clamp), the scene-referred and display-referred averages
//!   differ materially, proving the out-of-gamut preservation is real and
//!   observable. (On in-gamut content they coincide to within rounding — also
//!   pinned.)
//! - **`display_referred_default_*`** — without `with_linear_mode` (default
//!   [`LinearMode::DisplayReferred`]) the output is bit-identical to an
//!   explicit `DisplayReferred`, i.e. the RFC #238 Phase 2 behaviour is
//!   unchanged. (The Phase 2 suite in `resample_linear_domain` itself runs
//!   unchanged.)
//! - the per-frame freeze / atomicity contract holds in scene mode too
//!   (mid-frame transfer / domain / mode change rejected; a final-row
//!   allocation failure leaves the frame retryable).
//!
//! [`LinearMode`]: crate::resample::LinearMode
//! [`LinearMode::SceneReferred`]: crate::resample::LinearMode::SceneReferred
//! [`LinearMode::DisplayReferred`]: crate::resample::LinearMode::DisplayReferred

use crate::{
  KernelMatrix, PixelSink,
  resample::{
    AreaResampler, AveragingDomain, FilteredResampler, LinearMode, ResampleError, TransferFunction,
    Triangle,
  },
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Yuv420p, Yuv420pRow, Yuv422p, Yuv440p, Yuv444p, yuv420p_to, yuv422p_to, yuv440p_to, yuv444p_to,
  },
};
use mediaframe::frame::{Yuv420pFrame, Yuv422pFrame, Yuv440pFrame, Yuv444pFrame};

const SRC: usize = 8;
const OUT: usize = 4;

// ---- shared fixtures -----------------------------------------------------

fn y_ramp() -> Vec<u8> {
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 24 + (i as u8) % 200;
  }
  y
}

fn chroma(cw: usize, ch: usize, base: u8, step: u8) -> Vec<u8> {
  let mut c = vec![0u8; cw * ch];
  for (i, p) in c.iter_mut().enumerate() {
    *p = base.wrapping_add(((i % cw) as u8).wrapping_mul(step));
  }
  c
}

// ---- independent unclamped real-valued decode oracle ---------------------
//
// The oracle decodes YUV→RGB with a from-scratch real-valued matrix, using
// the SAME coefficients as the production decode but re-derived here as
// `q15 / 32768` from the published Q15 table that `Coefficients::for_matrix`
// holds (BT.709). The ONLY difference from the production display-referred
// decode is the absent intermediate Q15 rounding and the absent final
// clamp+round — exactly what scene-referred does. The result is a `[0, 1]`
// scale (the real 8-bit code value / 255) that MAY leave `[0, 1]`.

/// BT.709 real coefficients, re-derived as `q15 / 32768` from the production
/// `Coefficients::for_matrix(Bt709)` table — `r_v` 51606, `g_u` -6136,
/// `g_v` -15339, `b_u` 60808 (`r_u = b_v = 0`) — paired with the
/// `range_params_n::<8, 8>(full_range)` offset / scales, also re-derived here.
/// The tests all decode `full_range = true` (the `true` passed to `yuv*_to`),
/// for which the scales are exactly `1.0` and `y_off` is `0`; the
/// `full_range = false` branch mirrors the limited-range derivation for
/// completeness.
struct Bt709Real {
  y_off: f32,
  y_scale: f32,
  c_scale: f32,
  r_v: f32,
  g_u: f32,
  g_v: f32,
  b_u: f32,
}

impl Bt709Real {
  fn new(full_range: bool) -> Self {
    let q15 = 32768.0f32;
    // Mirrors `range_params_n::<8, 8>(full_range)`: rounded integer Q15
    // scales, then taken to real (`/ 32768`). Full range maps `[0, 255]`
    // directly (`y_off = 0`, both scales `1.0`); limited range maps
    // `[16, 235]` luma / `[16, 240]` chroma to `[0, 255]`.
    let (y_off, y_scale_q15, c_scale_q15) = if full_range {
      (
        0.0,
        (((255i64 << 15) + 255 / 2) / 255) as f32,
        (((255i64 << 15) + 255 / 2) / 255) as f32,
      )
    } else {
      (
        16.0,
        (((255i64 << 15) + 219 / 2) / 219) as f32,
        (((255i64 << 15) + 224 / 2) / 224) as f32,
      )
    };
    Self {
      y_off,
      y_scale: y_scale_q15 / q15,
      c_scale: c_scale_q15 / q15,
      r_v: 51606.0 / q15,
      g_u: -6136.0 / q15,
      g_v: -15339.0 / q15,
      b_u: 60808.0 / q15,
    }
  }

  /// One unclamped pixel → normalized `[0, 1]`-scale `(R, G, B)` (may leave
  /// `[0, 1]`).
  fn decode(&self, y: u8, u: u8, v: u8) -> [f32; 3] {
    let yl = (y as f32 - self.y_off) * self.y_scale;
    let u_d = (u as f32 - 128.0) * self.c_scale;
    let v_d = (v as f32 - 128.0) * self.c_scale;
    let r = yl + self.r_v * v_d;
    let g = yl + self.g_u * u_d + self.g_v * v_d;
    let b = yl + self.b_u * u_d;
    [r / 255.0, g / 255.0, b / 255.0]
  }
}

/// Build the full-resolution unclamped normalized RGB (`SRC x SRC x 3` `f32`)
/// for a frame, given a per-pixel `(u, v)` chroma lookup that encodes the
/// subsampling. `chroma_at(x, y)` returns the chroma sample covering luma
/// pixel `(x, y)` — nearest-neighbor, matching the production kernels'
/// in-register upsample.
fn full_res_unclamped_rgb(
  y_plane: &[u8],
  chroma_at: impl Fn(usize, usize) -> (u8, u8),
) -> Vec<f32> {
  // Every test decodes full-range (`true` passed to `yuv*_to`).
  let c = Bt709Real::new(true);
  let mut rgb = vec![0.0f32; SRC * SRC * 3];
  for py in 0..SRC {
    for px in 0..SRC {
      let (u, v) = chroma_at(px, py);
      let [r, g, b] = c.decode(y_plane[py * SRC + px], u, v);
      let i = (py * SRC + px) * 3;
      rgb[i] = r;
      rgb[i + 1] = g;
      rgb[i + 2] = b;
    }
  }
  rgb
}

/// The independent scene-referred oracle: from the full-resolution UNCLAMPED
/// normalized RGB, EOTF each pixel (the odd-symmetric extrapolation handling
/// out-of-`[0, 1]`), take each 2x2-block linear mean, OETF back, and clamp to
/// `[0, 255]` at the output. The decode→linearise→bin→encode→clamp reference,
/// computed without touching the production binning stream.
fn scene_oracle(full_unclamped_rgb: &[f32], tf: TransferFunction) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for ch in 0..3 {
        let mut acc = 0.0f32;
        for dy in 0..2 {
          for dx in 0..2 {
            let sy = oy * 2 + dy;
            let sx = ox * 2 + dx;
            let e = full_unclamped_rgb[(sy * SRC + sx) * 3 + ch];
            acc += tf.eotf(e);
          }
        }
        let mean = acc / 4.0;
        let enc = (tf.oetf(mean) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        out[(oy * OUT + ox) * 3 + ch] = enc;
      }
    }
  }
  out
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
  a.iter()
    .zip(b.iter())
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}

// Chroma-lookup helpers per format (nearest-neighbor, matching the kernels).
fn at_420<'a>(u: &'a [u8], v: &'a [u8], cw: usize) -> impl Fn(usize, usize) -> (u8, u8) + 'a {
  move |x, y| {
    let cx = x / 2;
    let cy = y / 2;
    (u[cy * cw + cx], v[cy * cw + cx])
  }
}
fn at_422<'a>(u: &'a [u8], v: &'a [u8], cw: usize) -> impl Fn(usize, usize) -> (u8, u8) + 'a {
  move |x, y| {
    let cx = x / 2;
    (u[y * cw + cx], v[y * cw + cx])
  }
}
fn at_444<'a>(u: &'a [u8], v: &'a [u8]) -> impl Fn(usize, usize) -> (u8, u8) + 'a {
  move |x, y| (u[y * SRC + x], v[y * SRC + x])
}
fn at_440<'a>(u: &'a [u8], v: &'a [u8]) -> impl Fn(usize, usize) -> (u8, u8) + 'a {
  move |x, y| {
    let cy = y / 2;
    (u[cy * SRC + x], v[cy * SRC + x])
  }
}

// ---- scene == independent unclamped oracle (per format) ------------------
//
// The scene-referred RGB output matches the from-scratch unclamped oracle
// within 1 LSB. Both decode the same unclamped real-valued RGB and
// box-average in linear light; the only difference is the f32 accumulation
// order (the production `AreaStream<f32>` H-reduce-then-V-accumulate vs the
// oracle's flat 2x2 sum), which can perturb the re-encoded byte by at most 1
// — pinned as `<= 1`, the same tolerance the Phase 2 display-referred oracle
// carries.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_scene_referred_equals_independent_unclamped_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  let oracle = scene_oracle(&full_res_unclamped_rgb(&y, at_420(&u, &v, cw)), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv420pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv420p scene vs unclamped oracle: max diff {} (rgb={rgb:?} oracle={oracle:?})",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_scene_referred_equals_independent_unclamped_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, SRC, 200, 6);
  let v = chroma(cw, SRC, 40, 7);

  let oracle = scene_oracle(&full_res_unclamped_rgb(&y, at_422(&u, &v, cw)), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv422pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv422p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv422p scene vs unclamped oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_scene_referred_equals_independent_unclamped_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let u = chroma(SRC, SRC, 200, 6);
  let v = chroma(SRC, SRC, 40, 7);

  let oracle = scene_oracle(&full_res_unclamped_rgb(&y, at_444(&u, &v)), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv444pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv444p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv444p scene vs unclamped oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_scene_referred_equals_independent_unclamped_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let ch = SRC / 2;
  let u = chroma(SRC, ch, 200, 6);
  let v = chroma(SRC, ch, 40, 7);

  let oracle = scene_oracle(&full_res_unclamped_rgb(&y, at_440(&u, &v)), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv440pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv440p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv440p scene vs unclamped oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

// ---- scene vs display: differ on out-of-gamut, coincide in-gamut ----------

/// Runs a `Yuv420p` area downscale to RGB under the Linear domain in the given
/// [`LinearMode`], with the row-stage tier pinned (Linear is row-stage anyway,
/// but keep it explicit for parity with the Phase 2 tests).
fn run_420_mode(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix, mode: LinearMode) -> Vec<u8> {
  let cw = SRC / 2;
  let src = Yuv420pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(mode)
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_and_display_referred_differ_on_out_of_gamut() {
  // Saturated chroma far from neutral drives the affine decode out of gamut:
  // several channels go below 0 or above 1 BEFORE the clamp. The 2x2 blocks
  // mix a deeply-out-of-gamut chroma pair with a near-neutral one, so the
  // display-referred average (which clamps each source pixel to [0, 1] first,
  // throwing the excursion away) lands materially apart from the scene-referred
  // average (which preserves the excursion and clamps only at the output).
  let matrix = KernelMatrix::Bt709;
  // High-contrast luma so the out-of-gamut chroma actually pushes channels
  // past the cube on the bright pixels.
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    let (r, c) = (i / SRC, i % SRC);
    *p = if (r + c) % 2 == 0 { 235 } else { 40 };
  }
  let cw = SRC / 2;
  // A checkerboard of extreme vs neutral chroma at the BLOCK scale: each 2x2
  // output footprint averages an extreme (0 / 255) chroma sample with a
  // neutral (128) one.
  let mut u = vec![128u8; cw * cw];
  let mut v = vec![128u8; cw * cw];
  for cy in 0..cw {
    for cx in 0..cw {
      if (cx + cy) % 2 == 0 {
        u[cy * cw + cx] = 255;
        v[cy * cw + cx] = 0;
      }
    }
  }

  let display = run_420_mode(&y, &u, &v, matrix, LinearMode::DisplayReferred);
  let scene = run_420_mode(&y, &u, &v, matrix, LinearMode::SceneReferred);
  assert!(
    max_abs_diff(&display, &scene) > 4,
    "scene-referred must differ from display-referred on out-of-gamut content \
     (max diff {}, display={display:?} scene={scene:?})",
    max_abs_diff(&display, &scene),
  );

  // The scene-referred result is what its independent unclamped oracle
  // predicts — proving the divergence is the preserved out-of-gamut signal,
  // not noise.
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let oracle = scene_oracle(&full_res_unclamped_rgb(&y, at_420(&u, &v, cw)), tf);
  assert!(
    max_abs_diff(&scene, &oracle) <= 1,
    "scene-referred out-of-gamut output must match its unclamped oracle (max diff {})",
    max_abs_diff(&scene, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_and_display_referred_coincide_in_gamut() {
  // In-gamut content (neutral chroma, mid-tone luma): the unclamped decode
  // never leaves [0, 1], so the clamp the display path applies is a no-op and
  // the two modes agree to within f32 rounding. This documents that scene mode
  // is a strict superset — it changes the answer ONLY where the clamp would
  // have discarded information.
  let matrix = KernelMatrix::Bt709;
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    let (r, c) = (i / SRC, i % SRC);
    *p = if (r + c) % 2 == 0 { 90 } else { 160 };
  }
  let cw = SRC / 2;
  let u = vec![128u8; cw * cw];
  let v = vec![128u8; cw * cw];

  let display = run_420_mode(&y, &u, &v, matrix, LinearMode::DisplayReferred);
  let scene = run_420_mode(&y, &u, &v, matrix, LinearMode::SceneReferred);
  assert!(
    max_abs_diff(&display, &scene) <= 1,
    "scene and display must coincide on in-gamut content (max diff {})",
    max_abs_diff(&display, &scene),
  );
}

// ---- display-referred default is byte-identical to Phase 2 ----------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn display_referred_default_is_byte_identical_to_explicit() {
  // Leaving the mode at its default (DisplayReferred) must produce
  // byte-identical Linear output to explicitly setting DisplayReferred — i.e.
  // the new mode field is inert on the default path, and the default IS the
  // RFC #238 Phase 2 display-referred linear average. Cover all four formats
  // and an RGBA output.
  let matrix = KernelMatrix::Bt709;

  // Yuv420p (RGBA output).
  {
    let y = y_ramp();
    let cw = SRC / 2;
    let u = chroma(cw, cw, 200, 6);
    let v = chroma(cw, cw, 40, 7);
    let render = |set_display: bool| {
      let src = Yuv420pFrame::new(
        &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
      );
      let mut rgba = vec![0u8; OUT * OUT * 4];
      {
        let base = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear);
        let base = if set_display {
          base.with_linear_mode(LinearMode::DisplayReferred)
        } else {
          base
        };
        let mut sink = base.with_rgba(&mut rgba).unwrap();
        yuv420p_to(&src, true, matrix, &mut sink).unwrap();
      }
      rgba
    };
    assert_eq!(
      render(false),
      render(true),
      "Yuv420p default (unset) LinearMode must equal explicit DisplayReferred",
    );
  }

  // Yuv444p (RGB output) — the other decode kernel family.
  {
    let y = y_ramp();
    let u = chroma(SRC, SRC, 200, 6);
    let v = chroma(SRC, SRC, 40, 7);
    let render = |set_display: bool| {
      let src = Yuv444pFrame::new(
        &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
      );
      let mut rgb = vec![0u8; OUT * OUT * 3];
      {
        let base = MixedSinker::<Yuv444p, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear);
        let base = if set_display {
          base.with_linear_mode(LinearMode::DisplayReferred)
        } else {
          base
        };
        let mut sink = base.with_rgb(&mut rgb).unwrap();
        yuv444p_to(&src, true, matrix, &mut sink).unwrap();
      }
      rgb
    };
    assert_eq!(
      render(false),
      render(true),
      "Yuv444p default (unset) LinearMode must equal explicit DisplayReferred",
    );
  }
}

/// The default `LinearMode` getter is `DisplayReferred`, and the builder
/// round-trips both variants through the getter.
#[test]
fn linear_mode_default_and_round_trip() {
  let sink = MixedSinker::<Yuv420p>::new(SRC, SRC);
  assert_eq!(sink.linear_mode(), LinearMode::DisplayReferred);
  assert_eq!(
    sink
      .with_linear_mode(LinearMode::SceneReferred)
      .linear_mode(),
    LinearMode::SceneReferred,
  );
}

// ---- scene mode inherits the freeze / atomicity contract ------------------
//
// Scene-referred is a decode SWAP-IN inside the existing linear-light tail, so
// it inherits the per-frame freeze (transfer / domain / output) and the
// recoverable-allocation contract unchanged. These pin that the contract still
// holds with `LinearMode::SceneReferred` set — the same scenarios the Phase 2
// `resample_linear_domain` suite pins for display mode.

/// A mid-frame transfer-function change is rejected in scene mode too, and the
/// rejected row leaves the frame retryable (the freeze is the tail's, shared
/// by both modes).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_mode_mid_frame_transfer_change_is_rejected() {
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_transfer_function(TransferFunction::Srgb)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Row 0 fixes the frozen transfer (Srgb).
    let yr = &y[0..SRC];
    let ur = &u[0..cw];
    let vr = &v[0..cw];
    sink
      .process(Yuv420pRow::new(yr, ur, vr, 0, KernelMatrix::Bt709, true))
      .unwrap();
    // Row 1 with a different transfer override → rejected.
    sink.set_transfer_function(TransferFunction::Bt1886);
    let yr = &y[SRC..2 * SRC];
    let ur = &u[cw..2 * cw];
    let vr = &v[cw..2 * cw];
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::TransferFunctionChanged(_)),
      "mid-frame transfer change in scene mode must reject with TransferFunctionChanged, got {err:?}",
    );
    // The retry with the ORIGINAL transfer on the SAME row succeeds (the
    // rejected row left the frame unpoisoned).
    sink.set_transfer_function(TransferFunction::Srgb);
    sink
      .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
      .unwrap();
  }
}

/// A mid-frame averaging-domain change is rejected in scene mode too.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_mode_mid_frame_domain_change_is_rejected() {
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    let yr = &y[0..SRC];
    let ur = &u[0..cw];
    let vr = &v[0..cw];
    sink
      .process(Yuv420pRow::new(yr, ur, vr, 0, KernelMatrix::Bt709, true))
      .unwrap();
    // Flip the domain to Encoded mid-frame → rejected.
    sink.set_averaging_domain(AveragingDomain::Encoded);
    let yr = &y[SRC..2 * SRC];
    let ur = &u[cw..2 * cw];
    let vr = &v[cw..2 * cw];
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::AveragingDomainChanged(_)),
      "mid-frame domain change in scene mode must reject with AveragingDomainChanged, got {err:?}",
    );
  }
}

/// The [`LinearMode`] resolved on the first output-bearing Linear row is frozen
/// for the frame, parallel to the frozen transfer / domain / output set: a
/// caller flipping [`MixedSinker::set_linear_mode`] mid-frame must be rejected
/// with the specific [`MixedSinkerError::LinearModeChanged`] BEFORE any state
/// mutation (every buffered row is already decoded under the first mode's
/// referent, so mixing display- and scene-decoded rows in one frame is a
/// silent corruption, NOT a tolerable result), and the accumulator must stay
/// retryable — restoring the mode lets the SAME sink resume the row and run the
/// frame to completion with no `begin_frame`. Both flip directions are pinned:
/// Scene→Display (the frame freezes Scene) and Display→Scene (freezes Display).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_referred_mid_frame_mode_change_is_rejected() {
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let row = |r: usize| {
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    (yr, &u[cr * cw..(cr + 1) * cw], &v[cr * cw..(cr + 1) * cw])
  };

  // `frozen` is the mode row 0 fixes; `flipped` is the opposing mode fed on
  // row 1 (which must reject) and then restored to `frozen` to resume. Run
  // both directions so neither the Scene- nor the Display-frozen frame accepts
  // a mid-frame swap to the other referent.
  for (frozen, flipped) in [
    (LinearMode::SceneReferred, LinearMode::DisplayReferred),
    (LinearMode::DisplayReferred, LinearMode::SceneReferred),
  ] {
    const SENTINEL: u8 = 0xEF;
    let mut rgb = vec![SENTINEL; OUT * OUT * 3];
    {
      let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_averaging_domain(AveragingDomain::Linear)
      .with_linear_mode(frozen)
      .with_rgb(&mut rgb)
      .unwrap();
      sink.begin_frame(SRC as u32, SRC as u32).unwrap();

      // Row 0 freezes the linear mode (`frozen`) on the lazily-created frame.
      let (yr, ur, vr) = row(0);
      sink
        .process(Yuv420pRow::new(yr, ur, vr, 0, KernelMatrix::Bt709, true))
        .unwrap();

      // Flip the mode mid-frame, then feed row 1 — the freeze guards the mode
      // BEFORE any state mutation, so it must reject with the SPECIFIC
      // LinearModeChanged (NOT a silently-mixed display/scene result) and leave
      // `next_y` unadvanced.
      sink.set_linear_mode(flipped);
      let (yr, ur, vr) = row(1);
      let err = sink
        .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
        .unwrap_err();
      assert!(
        matches!(err, MixedSinkerError::LinearModeChanged(_)),
        "a mid-frame linear-mode change ({frozen:?} -> {flipped:?}) must reject \
         with the specific LinearModeChanged, got {err:?}",
      );

      // Restore the frozen mode: the SAME sink resumes row 1 and runs the frame
      // to completion (proving the rejected call left `next_y` unadvanced — no
      // poisoning, no `begin_frame`).
      sink.set_linear_mode(frozen);
      for r in 1..SRC {
        let (yr, ur, vr) = row(r);
        sink
          .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
          .unwrap();
      }
    }
    assert!(
      rgb.iter().any(|&b| b != SENTINEL),
      "the resumed frame ({frozen:?} frozen) must produce real output once completed",
    );
  }
}

/// A final-row allocation failure in scene mode leaves the frame retryable:
/// the failed final row does not advance `next_y` or consume the buffered
/// frame, so the SAME final row retries cleanly and produces the correct
/// binned result. Mirrors the Phase 2 display-mode test, exercising the
/// scene-referred f32 scratch / unclamped decode path on every row.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_mode_final_row_alloc_failure_leaves_frame_retryable() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  let expected = scene_oracle(&full_res_unclamped_rgb(&y, at_420(&u, &v, cw)), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_linear_mode(LinearMode::SceneReferred)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Rows 0..SRC-1 buffer cleanly.
    for r in 0..SRC - 1 {
      let yr = &y[r * SRC..(r + 1) * SRC];
      let cr = r / 2;
      let ur = &u[cr * cw..(cr + 1) * cw];
      let vr = &v[cr * cw..(cr + 1) * cw];
      sink
        .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
        .unwrap();
    }
    // Arm the final-row tail allocation failpoint.
    crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure();
    let r = SRC - 1;
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    let ur = &u[cr * cw..(cr + 1) * cw];
    let vr = &v[cr * cw..(cr + 1) * cw];
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "scene-mode final-row alloc failure must surface AllocationFailed, got {err:?}",
    );
    // The failpoint is one-shot; retry the SAME final row → succeeds, frame
    // not poisoned.
    sink
      .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
      .unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &expected) <= 1,
    "scene-mode retried final row must produce the correct binned result (max diff {})",
    max_abs_diff(&rgb, &expected),
  );
}

/// Scene mode is area-only too: a filter plan is rejected with the typed
/// `UnsupportedFilter` (the in-tail backstop fires regardless of mode), and
/// the pre-seeded output is left untouched.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn scene_mode_rejects_filter_plan() {
  const SENTINEL: u8 = 0x5A;
  let matrix = KernelMatrix::Bt709;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let src = Yuv420pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );

  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(OUT, OUT, Triangle),
    )
    .unwrap()
    .with_averaging_domain(AveragingDomain::Linear)
    .with_linear_mode(LinearMode::SceneReferred)
    .with_rgb(&mut rgb)
    .unwrap();
    let err = yuv420p_to(&src, true, matrix, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
      ),
      "scene mode + a filter plan must reject with UnsupportedFilter, got {err:?}",
    );
  }
  assert!(
    rgb.iter().all(|&b| b == SENTINEL),
    "a rejected filter plan must leave the output unmutated",
  );
}
