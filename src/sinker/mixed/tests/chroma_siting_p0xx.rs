//! Chroma-siting-aware **high-bit semi-planar** 4:2:0 upsampling for the
//! MSB-aligned P-format family `P010` / `P012` / `P016` (#302) — the
//! high-bit-AND-semi-planar combination of `chroma_siting_hibit_420` (high-bit
//! planar `Yuv420p9…16`) and `chroma_siting_nv` (8-bit semi-planar
//! `Nv12` / `Nv21`).
//!
//! Covers, per format (10 / 12 / 16, via the macro below): the default /
//! co-sited path staying byte-identical to the pre-#302 fused decode (the
//! regression guard, plus its negative control that the centered phase actually
//! moves chroma); the centered RGB / RGBA / HSV decodes — and their `u16`
//! twins — matching an independent "upsample-then-P4xx-4:4:4" reference; SIMD-vs-
//! scalar parity of the centered path; the preflight-ordering atomicity (a
//! centered chroma-scratch alloc failure leaves luma AND colour untouched); the
//! dirty-bit sanitization (per the MSB-aligned packing — the IGNORED LOW bits,
//! not the high bits, are scrubbed before the blend); and the `ChromaDerivedNcl`
//! consistency invariant (the P-formats are NOT primaries-wired, so BOTH the
//! default and centered paths resolve it via the BT.709 matrix-tag fallback).
//! The MSB-aligned `u16` upsample kernel is also checked directly against a
//! hand-computed oracle, including the big-endian wire path.
//!
//! The macro instantiates each format with its **little-endian** marker, so a
//! sample's wire `u16` equals its MSB-aligned value on the (little-endian) test
//! host; the references encode in that same MSB-aligned convention. The
//! endianness re-encode is exercised host-independently by the kernel-level BE
//! oracle and the BE dirty-bit test.

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// MSB-aligns a logical sample into the wire `u16` for a P-format of `BITS`
/// active bits: `value << (16 - BITS)` (P010 `<< 6`, P012 `<< 4`, P016 `<< 0`).
fn pack(value: u16, bits: u32) -> u16 {
  value << (16 - bits)
}

/// Builds a high-bit P-format frame's LOGICAL planes: flat mid-gray luma plus a
/// per-column chroma ramp (distinct adjacent columns so the horizontal phase is
/// observable; the small `+ r` term keeps chroma rows from being identical so a
/// vertical mistake would surface). Values are clamped to `maxv =
/// (1 << BITS) - 1`. Planar (half-width) U / V — the interleave step packs them
/// into the semi-planar wire form.
fn ramp_planes_logical(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let step = (maxv / 16).max(1);
  let y = std::vec![(maxv / 2) as u16; w * h];
  let mut u = std::vec![0u16; cw * ch];
  let mut v = std::vec![0u16; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(maxv) as u16;
      v[r * cw + c] = maxv.saturating_sub(step * c as u32).max(step) as u16;
    }
  }
  (y, u, v)
}

/// Packs the flat-luma + planar-chroma logical frame into the MSB-aligned
/// semi-planar wire form: Y is `width` MSB-aligned u16 per row; the interleaved
/// half-width UV plane is `U V U V…` (U at the even element), `width` u16 per
/// chroma row, height / 2 rows — all `value << (16 - bits)`.
fn pack_p0xx(yp: &[u16], up: &[u16], vp: &[u16], bits: u32) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let cw = w / 2;
  let ch = (H / 2) as usize;
  let y_wire: Vec<u16> = yp.iter().map(|&x| pack(x, bits)).collect();
  let mut uv = std::vec![0u16; w * ch];
  for r in 0..ch {
    for c in 0..cw {
      uv[r * w + 2 * c] = pack(up[r * cw + c], bits);
      uv[r * w + 2 * c + 1] = pack(vp[r * cw + c], bits);
    }
  }
  (y_wire, uv)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp, on LOGICAL
/// `u16`. Written separately from the production kernel so it is a real oracle.
fn ref_upsample_center_h(c_half: &[u16], width: usize) -> Vec<u16> {
  let half = width / 2;
  let mut out = std::vec![0u16; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)] as u32;
    let m = c_half[j] as u32;
    let r = c_half[if j + 1 < half { j + 1 } else { j }] as u32;
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u16;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u16;
  }
  out
}

/// Builds the full-resolution MSB-aligned **interleaved** UV the centered
/// P-format decode reconstructs: each luma row `r` takes chroma row `r / 2` (the
/// walker's vertical replication, unchanged by #302) horizontally upsampled with
/// the centered weights, then U / V re-interleaved and MSB-packed. Feeding this
/// to the matching `P4xx` 4:4:4 conversion is the end-to-end oracle.
fn ref_full_uv_interleaved(up: &[u16], vp: &[u16], bits: u32) -> Vec<u16> {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut uv444 = std::vec![0u16; 2 * w * h];
  for r in 0..h {
    let cr = r / 2;
    let urow = ref_upsample_center_h(&up[cr * cw..cr * cw + cw], w);
    let vrow = ref_upsample_center_h(&vp[cr * cw..cr * cw + cw], w);
    for c in 0..w {
      uv444[2 * (r * w + c)] = pack(urow[c], bits);
      uv444[2 * (r * w + c) + 1] = pack(vrow[c], bits);
    }
  }
  uv444
}

// ---- MSB-aligned u16 kernel oracle (endianness-explicit) -------------------

#[test]
fn center_upsample_p0xx_kernel_matches_hand_computed() {
  // Interleaved U,V half-row: U = [0, 0, 400, 400], V = [400, 400, 0, 0]
  // (half = 4, width = 8), MSB-aligned at BITS=10 (`<< 6`), little-endian wire.
  //   even 2j   = (c[j-1] + 3·c[j] + 2) >> 2
  //   odd  2j+1 = (3·c[j] + c[j+1] + 2) >> 2
  let u = [0u16, 0, 400, 400];
  let v = [400u16, 400, 0, 0];
  let mut uv_half = [0u16; 8];
  for j in 0..4 {
    uv_half[2 * j] = pack(u[j], 10).to_le();
    uv_half[2 * j + 1] = pack(v[j], 10).to_le();
  }
  let mut uv_full = [0u16; 16];
  crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<10, false>(
    &uv_half,
    &mut uv_full,
    8,
    false,
  );

  // Decode back to logical (>> 6) and split U / V.
  let dec: Vec<u16> = uv_full.iter().map(|&x| u16::from_le(x) >> 6).collect();
  let u_out: Vec<u16> = (0..8).map(|i| dec[2 * i]).collect();
  let v_out: Vec<u16> = (0..8).map(|i| dec[2 * i + 1]).collect();
  assert_eq!(u_out, std::vec![0, 0, 0, 100, 300, 400, 400, 400]);
  assert_eq!(v_out, std::vec![400, 400, 400, 300, 100, 0, 0, 0]);
}

#[test]
fn center_upsample_p0xx_kernel_big_endian_matches_le_logical() {
  // Same LOGICAL input, MSB-packed, wire-encoded big-endian: the kernel
  // interpolates in the logical domain and re-encodes to the same MSB-aligned
  // wire order, so decoding the BE output back yields the SAME logical result as
  // the LE path. Host-independent.
  let u = [0u16, 0, 400, 400];
  let v = [400u16, 400, 0, 0];
  let mut half_le = [0u16; 8];
  let mut half_be = [0u16; 8];
  for j in 0..4 {
    half_le[2 * j] = pack(u[j], 10).to_le();
    half_le[2 * j + 1] = pack(v[j], 10).to_le();
    half_be[2 * j] = pack(u[j], 10).to_be();
    half_be[2 * j + 1] = pack(v[j], 10).to_be();
  }
  let mut out_le = [0u16; 16];
  let mut out_be = [0u16; 16];
  crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<10, false>(
    &half_le,
    &mut out_le,
    8,
    false,
  );
  crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<10, false>(
    &half_be,
    &mut out_be,
    8,
    true,
  );
  let dec_le: Vec<u16> = out_le.iter().map(|&x| u16::from_le(x) >> 6).collect();
  let dec_be: Vec<u16> = out_be.iter().map(|&x| u16::from_be(x) >> 6).collect();
  assert_eq!(
    dec_be, dec_le,
    "BE wire path must equal the LE logical interpolation"
  );
}

#[test]
fn center_upsample_p0xx_kernel_sanitizes_dirty_low_bits() {
  // P-format is MSB-aligned: the IGNORED LOW `16 - BITS` bits are scrubbed by
  // the de-pack (`>> (16 - BITS)`) BEFORE the 1/4-3/4 blend, exactly as the
  // fused P-format decode does. For every sub-16-bit depth and both wire endians
  // a frame with ALL the ignored low bits set must blend identically to the
  // clean (low-bits-zero) frame. (At BITS = 16 there are no ignored bits, so
  // this is the clean == clean identity.)
  fn check<const BITS: u32>() {
    let low_dirty = (1u16 << (16 - BITS)).wrapping_sub(1); // the ignored low bits
    // Logical ramp U,V (half = 4): distinct columns so the blend is non-trivial.
    let u = [0u16, 1, 2, 3];
    let v = [3u16, 2, 1, 0];
    for &be in &[false, true] {
      let enc = |v: u16| if be { v.to_be() } else { v.to_le() };
      let dec = |v: u16| if be { u16::from_be(v) } else { u16::from_le(v) };
      let mut clean = [0u16; 8];
      let mut dirty = [0u16; 8];
      for j in 0..4 {
        let up = pack(u[j], BITS);
        let vp = pack(v[j], BITS);
        clean[2 * j] = enc(up);
        clean[2 * j + 1] = enc(vp);
        dirty[2 * j] = enc(up | low_dirty);
        dirty[2 * j + 1] = enc(vp | low_dirty);
      }
      let mut out_clean = [0u16; 16];
      let mut out_dirty = [0u16; 16];
      crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, false>(
        &clean,
        &mut out_clean,
        8,
        be,
      );
      crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<BITS, false>(
        &dirty,
        &mut out_dirty,
        8,
        be,
      );
      let dec_clean: Vec<u16> = out_clean.iter().map(|&v| dec(v)).collect();
      let dec_dirty: Vec<u16> = out_dirty.iter().map(|&v| dec(v)).collect();
      assert_eq!(
        dec_dirty, dec_clean,
        "BITS={BITS} be={be}: dirty IGNORED-LOW bits must be scrubbed before the blend"
      );
    }
  }
  check::<10>();
  check::<12>();
  // 16-bit has no ignored low bits: a clean frame round-trips unchanged.
  check::<16>();
}

// ---- per-format suite ------------------------------------------------------

// The suite is identical bar the format, so generate it once per member. Each
// lands in its own `mod` so the names don't collide.
macro_rules! p0xx_chroma_tests {
  ($mod:ident, $bits:expr, $Marker:ident, $Frame:ident, $walker:ident,
   $Ref:ident, $RefFrame:ident, $ref_walker:ident,
   $MarkerBe:ty, $FrameBe:ident, $walker_be:ident) => {
    mod $mod {
      use super::*;

      const BITS: u32 = $bits;
      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Centered/default identity-decode RGB for a siting + SIMD toggle.
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_simd(simd);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        rgb
      }

      // ---- default / co-sited path is byte-identical (regression guard) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_and_cosited_sitings_are_byte_identical() {
        // `TopLeft` (`v = 0`) now folds the forward vertical triangle (RFC #238
        // Top), so it LEAVES the co-sited byte-identity group.
        let baseline = convert_rgb(ChromaLocation::Unspecified, true);
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::other("unassigned-99"),
          ChromaLocation::Left,
        ] {
          assert_eq!(
            convert_rgb(loc.clone(), true),
            baseline,
            "siting {loc:?} must keep the byte-identical default decode"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_path_does_not_allocate_chroma_scratch() {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(chroma_len, 0, "co-sited path must not grow the u16 chroma scratch");
      }

      // ---- centered path correctness ---------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_grows_chroma_scratch_to_full_width() {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(
          chroma_len,
          2 * W as usize,
          "centered path stages the full-width interleaved chroma (U+V)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, _) = pack_p0xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);
        let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb_ref,
          "centered P-format RGB must equal upsample-then-P4xx-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_u16_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);

        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

        let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();

        assert_eq!(
          rgb16, rgb16_ref,
          "centered P-format RGB(u16) must equal upsample-then-P4xx-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_rgba_u16_and_hsv_match_444_reference() {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);

        // RGBA (u8).
        {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(rgba, rgba_ref, "centered RGBA must equal upsample-then-P4xx");
        }

        // RGBA (u16).
        {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgba16 = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(
            rgba16, rgba16_ref,
            "centered RGBA(u16) must equal upsample-then-P4xx"
          );
        }

        // HSV-direct (no RGB / RGBA attached).
        {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let (mut h, mut s, mut v) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_hsv(&mut h, &mut s, &mut v)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!((h, s, v), (hr, sr, vr), "centered HSV must equal upsample-then-P4xx");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_forward_vblend_differs_from_center_and_bottom() {
        // RFC #238 Top (`v = 0`) folds the FORWARD vertical triangle: an ODD output
        // row box-blends its chroma row with the NEXT one (a one-row delay), so on a
        // vertically-varying chroma ramp Top diverges from both the vertical-co-sited
        // Center (`v` box) and the BACKWARD-folding Bottom (`v = 1`). A strong
        // per-ROW chroma step keeps the divergence visible through 8-bit RGB
        // quantization even at 16-bit source depth. The exact forward-delay values +
        // cross-tier consistency are pinned by the resample suite (`vramp`) and the
        // cross-format `p0xx_direct_top_matches_yuv420p_top` check.
        let w = W as usize;
        let h = H as usize;
        let (cw, ch) = (w / 2, h / 2);
        let step = (MAXV / 8).max(1);
        let yp = std::vec![(MAXV / 2) as u16; w * h];
        let mut up = std::vec![0u16; cw * ch];
        let mut vp = std::vec![0u16; cw * ch];
        for r in 0..ch {
          for c in 0..cw {
            up[r * cw + c] = (step + r as u32 * step).min(MAXV) as u16;
            vp[r * cw + c] = MAXV.saturating_sub(r as u32 * step).max(step) as u16;
          }
        }
        let decode = |loc: ChromaLocation, simd: bool| -> Vec<u8> {
          let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgb = std::vec![0u8; w * h * 3];
          let mut sink = MixedSinker::<$Marker>::new(w, h)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone())
            .with_simd(simd);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let top = decode(ChromaLocation::Top, true);
        assert_ne!(
          top,
          decode(ChromaLocation::Center, true),
          "Top's forward vertical box blend must differ from Center on a vertical ramp"
        );
        assert_ne!(
          top,
          decode(ChromaLocation::Bottom, true),
          "Top (forward, odd rows) must differ from Bottom (backward, even rows)"
        );
        assert_eq!(
          top,
          decode(ChromaLocation::Top, false),
          "Top identity path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_phase_differs_from_default() {
        // Negative control: on a chroma ramp the centered phase must move chroma
        // relative to the co-sited / nearest-neighbor default — otherwise the
        // byte-identity assertions above would be vacuous.
        assert_ne!(
          convert_rgb(ChromaLocation::Center, true),
          convert_rgb(ChromaLocation::Left, true),
          "centered siting must shift chroma vs the co-sited default"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_path_simd_matches_scalar() {
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          convert_rgb(ChromaLocation::Center, false),
          "centered path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottomleft_differs_from_bottom_and_simd_matches_scalar() {
        // BottomLeft (h=0 co-sited + v=1) keeps the co-sited horizontal phase, so it
        // differs from Bottom (h=0.5) on the ramp's strong horizontal variation, and
        // is SIMD == scalar. Its vertical `v=1` fold and its exact value are pinned
        // across all depths against the planar Yuv420pN BottomLeft by the resample
        // suite's cross-format check + vramp tests (a vertical ramp strong enough to
        // survive 8-bit RGB quantization even at 16-bit source depth — this direct
        // ramp's per-row step is not, exactly as the `Bottom` direct decode).
        let bl = convert_rgb(ChromaLocation::BottomLeft, true);
        assert_ne!(
          bl,
          convert_rgb(ChromaLocation::Bottom, true),
          "BottomLeft (h=0) must differ from Bottom (h=0.5)"
        );
        assert_eq!(
          bl,
          convert_rgb(ChromaLocation::BottomLeft, false),
          "bottom-left path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      // ---- dirty-bit sanitization (MSB-aligned: scrub the ignored low bits) -

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_low_bits_le() {
        // A malformed-but-accepted MSB-aligned frame with the IGNORED LOW
        // `16 - BITS` bits set must decode (centered) identically to the clean
        // frame: the centered upsample de-packs (`>> (16 - BITS)`) each sample
        // BEFORE the 1/4-3/4 blend, exactly as the fused decode does, so a dirty
        // sample's ignored low bits never leak. (At BITS = 16 there are no
        // ignored bits, so this is the clean == clean identity.)
        let low_dirty = (1u16 << (16 - BITS)).wrapping_sub(1);
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let decode = |y: &[u16], uv: &[u16]| -> Vec<u8> {
          let src = $Frame::new(y, uv, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let y_dirty: Vec<u16> = y_wire.iter().map(|&x| x | low_dirty).collect();
        let uv_dirty: Vec<u16> = uv_wire.iter().map(|&x| x | low_dirty).collect();
        assert_eq!(
          decode(&y_dirty, &uv_dirty),
          decode(&y_wire, &uv_wire),
          "centered LE decode must scrub the ignored low bits (de-pack before blend)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_low_bits_be() {
        // Same invariant on the big-endian wire path: the de-pack runs in the
        // logical domain (after the endian load), so the ignored low bits are
        // scrubbed for BE inputs too. Planes are BE-encoded and decoded via the
        // BE marker / frame / walker.
        let low_dirty = (1u16 << (16 - BITS)).wrapping_sub(1);
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let y_be: Vec<u16> = y_wire.iter().map(|&x| x.to_be()).collect();
        let uv_be: Vec<u16> = uv_wire.iter().map(|&x| x.to_be()).collect();
        let decode = |y: &[u16], uv: &[u16]| -> Vec<u8> {
          let src = $FrameBe::try_new(y, uv, W, H, W, W).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let y_dirty: Vec<u16> = y_be.iter().map(|&x| x | low_dirty.to_be()).collect();
        let uv_dirty: Vec<u16> = uv_be.iter().map(|&x| x | low_dirty.to_be()).collect();
        assert_eq!(
          decode(&y_dirty, &uv_dirty),
          decode(&y_be, &uv_be),
          "centered BE decode must scrub the ignored low bits (de-pack before blend)"
        );
      }

      // ---- preflight-ordering atomicity (#302, cf. #180 / #308) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);

        // Negative control: unarmed, the SAME luma + centered-RGB config DOES
        // write luma — so the armed "untouched" assertion below is non-vacuous.
        {
          let src_ok = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut luma_ok = std::vec![0xABu8; (W * H) as usize];
          let mut rgb_ok = std::vec![0xCDu8; (W * H * 3) as usize];
          let mut sink_ok = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_luma(&mut luma_ok)
            .unwrap()
            .with_rgb(&mut rgb_ok)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src_ok, false, KernelMatrix::Bt601, &mut sink_ok).unwrap();
          drop(sink_ok);
          assert!(
            luma_ok.iter().any(|&b| b != 0xAB),
            "control: the centered path writes luma when the scratch alloc is not armed"
          );
        }

        // Armed: a centered RGB decode whose u16 chroma-scratch allocation fails
        // must leave EVERY output — luma included — untouched, because the
        // scratch is reserved (fallibly) BEFORE any output row is written.
        let mut luma = std::vec![0xABu8; (W * H) as usize];
        let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_luma(&mut luma)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);

        super::super::super::arm_chroma_full_alloc_failure();
        let err = $walker(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
        drop(sink);

        assert!(
          matches!(err, MixedSinkerError::Resample(ResampleError::AllocationFailed(_))),
          "centered chroma-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
        );
        assert!(
          luma.iter().all(|&b| b == 0xAB),
          "luma must be untouched on the centered alloc-failure path"
        );
        assert!(
          rgb.iter().all(|&b| b == 0xCD),
          "rgb must be untouched on the centered alloc-failure path"
        );
      }

      // ---- ChromaDerivedNcl consistency (#302 / #303 cross-feature seam) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
        // The P-formats are NOT ChromaDerivedNcl-primaries-wired (only 8-bit
        // Yuv420p got #316). BOTH paths — the default fused P-format kernel AND
        // the centered P4xx 4:4:4 kernel — resolve ChromaDerivedNcl via the
        // shared BT.709 matrix-tag fallback (`Coefficients::for_matrix`),
        // IGNORING the ColorSpec primaries, so default and centered stay
        // internally consistent (the centered phase shift is the ONLY difference
        // between them). Full primaries-derived support is a documented
        // Yuv420p-8bit-only follow-up. Guards that consistency AND that the
        // centered path did not accidentally half-adopt primaries on one tier.
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuv420p,
            ColorInfo::new(
              Primaries::Bt2020,
              Transfer::Bt709,
              ColorMatrix::ChromaDerivedNcl,
              DynamicRange::Limited,
              loc,
            ),
          )
        };
        let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(&spec(loc));
          $walker(&src, false, KernelMatrix::ChromaDerivedNcl, &mut sink).unwrap();
          rgb
        };
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone());
          $walker(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
          rgb
        };

        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered P-format ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
        );
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default P-format ChromaDerivedNcl must resolve via the same BT.709 fallback"
        );
      }

      // ---- end-to-end ColorSpec flow (no manual with_chroma_location) ------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn color_spec_center_drives_decode_without_manual_chroma_call() {
        use crate::{
          ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer, YuvOptions,
        };

        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);

        let info = ColorInfo::new(
          Primaries::Bt709,
          Transfer::Bt709,
          ColorMatrix::Bt601,
          DynamicRange::Limited,
          ChromaLocation::Center,
        );
        let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
        let opts = YuvOptions::from_color_spec(&spec).unwrap();
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_color_spec(&spec);
        $walker(&src, opts.full_range(), opts.matrix(), &mut sink).unwrap();
        drop(sink);

        assert_ne!(
          rgb,
          convert_rgb(ChromaLocation::Unspecified, true),
          "ColorSpec ChromaLocation::Center must change the decode via the options path"
        );
        assert_eq!(
          rgb,
          convert_rgb(ChromaLocation::Center, true),
          "ColorSpec-driven centered decode must equal the explicit centered path"
        );
      }
    }
  };
}

p0xx_chroma_tests!(
  p010,
  10,
  P010,
  P010Frame,
  p010_to,
  P410,
  P410Frame,
  p410_to,
  P010<true>,
  P010BeFrame,
  p010_to_endian
);
p0xx_chroma_tests!(
  p012,
  12,
  P012,
  P012Frame,
  p012_to,
  P412,
  P412Frame,
  p412_to,
  P012<true>,
  P012BeFrame,
  p012_to_endian
);
p0xx_chroma_tests!(
  p016,
  16,
  P016,
  P016Frame,
  p016_to,
  P416,
  P416Frame,
  p416_to,
  P016<true>,
  P016BeFrame,
  p016_to_endian
);

// ---- mid-frame siting-flip rejection (identity path freeze) -----------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_direct_path_mid_frame_siting_flip_is_rejected() {
  // The identity (no-resample) high-bit semi-planar 4:2:0 `P010` decode freezes the
  // effective phase — the horizontal centered flag and the vertical `Bottom` / `Top`
  // flags — on its first output-bearing row. Flipping `Bottom` / `Top` ⇆ co-sited /
  // `Center` mid-frame must reject the next in-sequence row with `ChromaSitingChanged`,
  // WITHOUT growing the chroma scratch, advancing the stateful `chroma_prev_u16`
  // vertical lookback (its validity tag included), OR touching a held `Top` odd row
  // (RFC #238). Flipping back and retrying then matches a clean single-phase decode
  // byte-for-byte.
  use super::super::MixedSinkerError;
  const BITS: u32 = 10;
  let maxv = (1u32 << BITS) - 1;
  let (yp, up, vp) = ramp_planes_logical(maxv);
  let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, BITS);
  let w = W as usize;
  let h = H as usize;
  for (loc1, loc2) in [
    (ChromaLocation::Bottom, ChromaLocation::Left),
    (ChromaLocation::Left, ChromaLocation::Bottom),
    (ChromaLocation::Bottom, ChromaLocation::Center),
    (ChromaLocation::Top, ChromaLocation::Center),
    (ChromaLocation::Center, ChromaLocation::Top),
    (ChromaLocation::TopLeft, ChromaLocation::Left),
  ] {
    // Reference: a clean whole-frame decode at the held siting.
    let mut want = std::vec![0u8; w * h * 3];
    {
      let src = P010Frame::new(&y_wire, &uv_wire, W, H, W, W);
      let mut ref_sink = MixedSinker::<P010>::new(w, h)
        .with_rgb(&mut want)
        .unwrap()
        .with_chroma_location(loc1.clone())
        .with_simd(true);
      p010_to(&src, false, KernelMatrix::Bt601, &mut ref_sink).unwrap();
    }

    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<P010>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc1.clone())
      .with_simd(true);
    crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
    for r in 0..2 {
      let cr = r / 2;
      let row = P010Row::new(
        &y_wire[r * w..(r + 1) * w],
        &uv_wire[cr * w..(cr + 1) * w],
        r,
        KernelMatrix::Bt601,
        false,
      );
      crate::PixelSink::process(&mut sink, row).unwrap();
    }
    let scratch_len = sink.chroma_full_u16.len();
    let prev_len = sink.chroma_prev_u16.len();
    let prev_tag = sink.chroma_prev_row;
    let pending_before = sink.chroma_top_pending;

    sink.set_chroma_location(loc2.clone());
    let row2 = P010Row::new(
      &y_wire[2 * w..3 * w],
      &uv_wire[w..2 * w],
      2,
      KernelMatrix::Bt601,
      false,
    );
    let err = crate::PixelSink::process(&mut sink, row2).unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
      "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
    );
    assert_eq!(
      sink.chroma_full_u16.len(),
      scratch_len,
      "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
    );
    assert_eq!(
      sink.chroma_prev_u16.len(),
      prev_len,
      "{loc1:?}->{loc2:?}: a rejected flip must not grow the vertical lookback"
    );
    assert_eq!(
      sink.chroma_prev_row, prev_tag,
      "{loc1:?}->{loc2:?}: a rejected flip must not advance the lookback tag"
    );
    assert_eq!(
      sink.chroma_top_pending, pending_before,
      "{loc1:?}->{loc2:?}: a rejected flip must not touch a held Top odd row"
    );

    sink.set_chroma_location(loc1.clone());
    for r in 2..h {
      let cr = r / 2;
      let row = P010Row::new(
        &y_wire[r * w..(r + 1) * w],
        &uv_wire[cr * w..(cr + 1) * w],
        r,
        KernelMatrix::Bt601,
        false,
      );
      crate::PixelSink::process(&mut sink, row).unwrap();
    }
    drop(sink);
    assert_eq!(
      rgb, want,
      "{loc1:?}: retry after a rejected flip must match a clean in-order decode"
    );
  }
}

#[test]
fn p0xx_begin_frame_drops_held_top_row() {
  // RFC #238 Top forward delay: a `P0xx` odd row is HELD (`chroma_top_pending`) to
  // emit at the following even row. If a Top frame is interrupted after a held odd
  // non-last row, `begin_frame` MUST drop that held state (and the frozen Top
  // phase) so frame N's deferred colour row can never flush into frame N+1 — for
  // EVERY P0xx impl (the Nv21 begin_frame-omission regression class). All three
  // route through the shared `reset_high_bit_yuv_streams`.
  macro_rules! check {
    ($Marker:ty, $Row:ident, $bits:expr) => {{
      let maxv = (1u32 << $bits) - 1;
      let (yp, up, vp) = ramp_planes_logical(maxv);
      let (y_wire, uv_wire) = pack_p0xx(&yp, &up, &vp, $bits);
      let w = W as usize;
      let mut rgb = std::vec![0u8; w * H as usize * 3];
      let mut sink = MixedSinker::<$Marker>::new(w, H as usize)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_chroma_location(ChromaLocation::Top)
        .with_simd(true);
      crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
      // Row 0 (even, co-sited, emitted) then row 1 (odd, HELD).
      for r in 0..2 {
        let cr = r / 2;
        crate::PixelSink::process(
          &mut sink,
          $Row::new(
            &y_wire[r * w..(r + 1) * w],
            &uv_wire[cr * w..(cr + 1) * w],
            r,
            KernelMatrix::Bt601,
            false,
          ),
        )
        .unwrap();
      }
      assert!(
        sink.chroma_top_pending.is_some(),
        "an odd Top row must be held before the next even row (bits={})",
        $bits
      );
      crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
      assert!(
        sink.chroma_top_pending.is_none(),
        "begin_frame must drop the held Top odd row so it can't flush into frame N+1 (bits={})",
        $bits
      );
      assert!(
        sink.frozen_chroma_top_v.is_none(),
        "begin_frame must clear the frozen Top phase so the next frame may re-pick siting (bits={})",
        $bits
      );
    }};
  }
  check!(P010, P010Row, 10u32);
  check!(P012, P012Row, 12u32);
  check!(P016, P016Row, 16u32);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p0xx_direct_top_matches_yuv420p_hibit_top() {
  // The identity FORWARD one-row-delay `Top` / `TopLeft` decode (RFC #238) must
  // reproduce the validated high-bit planar `Yuv420pN` `Top` / `TopLeft` decode of
  // the same de-interleaved logical planes byte-for-byte (P010 vs Yuv420p10, etc.)
  // — the strongest catch for a U/V swap in the delayed semi-planar path. EVEN
  // heights end on the trailing-odd co-sited clamp; ODD heights end on the final
  // even-row flush of the held odd row — both exercised. Bt601 keeps this on the
  // shared matrix-tag path.
  macro_rules! xcheck {
    ($P:ty, $PFrame:ty, $p_to:ident, $Y:ty, $YFrame:ty, $y_to:ident, $bits:expr) => {{
      let maxv = (1u32 << $bits) - 1;
      for (w, h) in [(16usize, 8usize), (16, 6), (16, 5), (8, 3)] {
        let (cw, ch) = (w / 2, h.div_ceil(2));
        // Two-axis chroma ramp + varying luma so the forward vertical fold and the
        // co-sited even rows are both non-trivial. Logical (low-packed) values.
        let yl: Vec<u16> = (0..w * h)
          .map(|i| ((maxv / 8) + (i as u32 * 37) % (maxv - maxv / 8)) as u16)
          .collect();
        let step = (maxv / 16).max(1);
        let mut ul = std::vec![0u16; cw * ch];
        let mut vl = std::vec![0u16; cw * ch];
        for r in 0..ch {
          for c in 0..cw {
            ul[r * cw + c] = (step + c as u32 * step + r as u32 * step * 2).min(maxv) as u16;
            vl[r * cw + c] = maxv
              .saturating_sub(c as u32 * step + r as u32 * step)
              .max(step) as u16;
          }
        }
        // P0xx wire: MSB-aligned Y + interleaved MSB-aligned UV (`ch` rows).
        let y_wire: Vec<u16> = yl.iter().map(|&x| pack(x, $bits)).collect();
        let mut uv_wire = std::vec![0u16; w * ch];
        for r in 0..ch {
          for c in 0..cw {
            uv_wire[r * w + 2 * c] = pack(ul[r * cw + c], $bits);
            uv_wire[r * w + 2 * c + 1] = pack(vl[r * cw + c], $bits);
          }
        }
        for loc in [ChromaLocation::Top, ChromaLocation::TopLeft] {
          // High-bit planar `Yuv420pN` reference (validated Top forward decode, #384).
          let mut want = std::vec![0u8; w * h * 3];
          {
            let src = <$YFrame>::new(
              &yl, &ul, &vl, w as u32, h as u32, w as u32, cw as u32, cw as u32,
            );
            let mut sink = MixedSinker::<$Y>::new(w, h)
              .with_rgb(&mut want)
              .unwrap()
              .with_chroma_location(loc.clone());
            $y_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
          }
          for simd in [true, false] {
            let mut got = std::vec![0u8; w * h * 3];
            {
              let src =
                <$PFrame>::new(&y_wire, &uv_wire, w as u32, h as u32, w as u32, w as u32);
              let mut sink = MixedSinker::<$P>::new(w, h)
                .with_rgb(&mut got)
                .unwrap()
                .with_chroma_location(loc.clone())
                .with_simd(simd);
              $p_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap();
            }
            assert_eq!(
              got, want,
              "P0xx {loc:?} identity {w}x{h} simd={simd} bits={} must equal Yuv420pN {loc:?}",
              $bits
            );
          }
        }
      }
    }};
  }
  xcheck!(
    P010,
    P010Frame<'_>,
    p010_to,
    Yuv420p10,
    mediaframe::frame::Yuv420p10Frame<'_>,
    yuv420p10_to,
    10u32
  );
  xcheck!(
    P012,
    P012Frame<'_>,
    p012_to,
    Yuv420p12,
    mediaframe::frame::Yuv420p12Frame<'_>,
    yuv420p12_to,
    12u32
  );
  xcheck!(
    P016,
    P016Frame<'_>,
    p016_to,
    Yuv420p16,
    mediaframe::frame::Yuv420p16Frame<'_>,
    yuv420p16_to,
    16u32
  );
}
