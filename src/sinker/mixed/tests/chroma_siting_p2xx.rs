//! Chroma-siting-aware **high-bit semi-planar** 4:2:2 upsampling for the
//! MSB-aligned P-format family `P210` / `P212` / `P216` (#302) — the 4:2:2
//! sibling of `chroma_siting_p0xx` (semi-planar 4:2:0 `P010` / `P012` / `P016`)
//! and the semi-planar twin of `chroma_siting_hibit_422` (planar 4:2:2).
//!
//! 4:2:2 subsamples chroma 2:1 horizontally ONLY (one chroma row per luma row,
//! no vertical subsampling), so the centered reconstruction reuses the SAME
//! MSB-aligned `u16` interleaved phase-0.5 upsample kernel 4:2:0 uses
//! ([`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx)) —
//! whose endianness / dirty-low-bit masking behaviour is oracle-tested in
//! `chroma_siting_p0xx`. Covers here, per format (10 / 12 / 16, via the macro):
//! the default / co-sited path staying byte-identical to the pre-#302 fused
//! decode (the regression guard + its negative control); the centered RGB / RGBA
//! / HSV decodes — and their `u16` twins — matching an independent
//! "upsample-then-P4xx-4:4:4" reference; SIMD-vs-scalar parity; that every
//! centered siting (Center / Top / Bottom) agrees horizontally; the dirty-low-bit
//! sanitization end-to-end on BOTH wire endians; the preflight-ordering
//! atomicity (a centered chroma-scratch alloc failure leaves luma AND colour
//! untouched); the `ChromaDerivedNcl` consistency invariant (the P-formats are
//! NOT primaries-wired); and the no-output overflow guard (per depth).
//!
//! The macro instantiates each format with its **little-endian** marker, so a
//! sample's wire `u16` equals its MSB-aligned value on the (little-endian) test
//! host; the references encode in that same MSB-aligned convention. The
//! endianness re-encode is exercised by the BE dirty-bit test (and host-
//! independently by `chroma_siting_p0xx`'s kernel-level BE oracle).

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// MSB-aligns a logical sample into the wire `u16` for a P-format of `BITS`
/// active bits: `value << (16 - BITS)` (P210 `<< 6`, P212 `<< 4`, P216 `<< 0`).
fn pack(value: u16, bits: u32) -> u16 {
  value << (16 - bits)
}

/// Independent reference for the centered horizontal upsample — phase-0.5
/// `1/4`–`3/4` with edge clamp, on LOGICAL `u16`. Written separately from the
/// production kernel so it is a real oracle.
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

/// Builds a high-bit 4:2:2 frame's LOGICAL planes: flat mid-gray luma plus a
/// per-column chroma ramp on a half-width, **full-height** chroma plane (one
/// chroma row per luma row). Distinct adjacent columns make the horizontal phase
/// observable; the `+ r` term keeps chroma rows distinct. Clamped to `maxv =
/// (1 << BITS) - 1`. Planar (half-width) U / V — the interleave step packs them
/// into the semi-planar wire form.
fn ramp_planes_logical(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let step = (maxv / 16).max(1);
  let y = std::vec![(maxv / 2) as u16; w * h];
  let mut u = std::vec![0u16; cw * h];
  let mut v = std::vec![0u16; cw * h];
  for r in 0..h {
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
/// chroma row, **`height`** rows (4:2:2 — one chroma row per luma row).
fn pack_p2xx(yp: &[u16], up: &[u16], vp: &[u16], bits: u32) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let y_wire: Vec<u16> = yp.iter().map(|&x| pack(x, bits)).collect();
  let mut uv = std::vec![0u16; w * h];
  for r in 0..h {
    for c in 0..cw {
      uv[r * w + 2 * c] = pack(up[r * cw + c], bits);
      uv[r * w + 2 * c + 1] = pack(vp[r * cw + c], bits);
    }
  }
  (y_wire, uv)
}

/// Builds the full-resolution MSB-aligned **interleaved** UV the centered
/// P-format decode reconstructs: each luma row `r` takes chroma row `r` (1:1 — NO
/// vertical subsampling, unlike 4:2:0's `r / 2`) horizontally upsampled with the
/// centered weights, then U / V re-interleaved and MSB-packed. Feeding this to
/// the matching `P4xx` 4:4:4 conversion is the end-to-end oracle.
fn ref_full_uv_interleaved(up: &[u16], vp: &[u16], bits: u32) -> Vec<u16> {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut uv444 = std::vec![0u16; 2 * w * h];
  for r in 0..h {
    let urow = ref_upsample_center_h(&up[r * cw..r * cw + cw], w);
    let vrow = ref_upsample_center_h(&vp[r * cw..r * cw + cw], w);
    for c in 0..w {
      uv444[2 * (r * w + c)] = pack(urow[c], bits);
      uv444[2 * (r * w + c) + 1] = pack(vrow[c], bits);
    }
  }
  uv444
}

// ---- shared MSB-aligned u16 kernel oracle ----------------------------------

#[test]
fn center_upsample_2to1_kernel_matches_hand_computed() {
  // 4:2:2 reconstructs centered chroma with the SAME `u16` interleaved kernel
  // 4:2:0 uses (the horizontal 2:1 upsample is identical); its endianness +
  // dirty-low-bit behaviour is oracle-tested per depth in `chroma_siting_p0xx`.
  // Here a single hand-computed case documents that shared kernel.
  // Interleaved U,V half-row: U = [0, 0, 400, 400], V = [400, 400, 0, 0]
  // (half = 4, width = 8), MSB-aligned at BITS=10 (`<< 6`), little-endian wire.
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

  let dec: Vec<u16> = uv_full.iter().map(|&x| u16::from_le(x) >> 6).collect();
  let u_out: Vec<u16> = (0..8).map(|i| dec[2 * i]).collect();
  let v_out: Vec<u16> = (0..8).map(|i| dec[2 * i + 1]).collect();
  assert_eq!(u_out, std::vec![0, 0, 0, 100, 300, 400, 400, 400]);
  assert_eq!(v_out, std::vec![400, 400, 400, 300, 100, 0, 0, 0]);
}

// ---- per-format suite ------------------------------------------------------

// The suite is identical bar the format, so generate it once per member. Each
// lands in its own `mod` so the names don't collide.
macro_rules! p2xx_chroma_tests {
  ($mod:ident, $bits:expr, $Marker:ident, $Frame:ident, $walker:ident,
   $Ref:ident, $RefFrame:ident, $ref_walker:ident,
   $MarkerBe:ty, $FrameBe:ident, $walker_be:ident, $Row:ident) => {
    mod $mod {
      use super::*;

      const BITS: u32 = $bits;
      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Centered/default identity-decode RGB for a siting + SIMD toggle.
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(loc.clone())
          .with_simd(simd);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        rgb
      }

      // ---- default / co-sited path is byte-identical (regression guard) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn default_and_cosited_sitings_are_byte_identical() {
        let baseline = convert_rgb(ChromaLocation::Unspecified, true);
        for loc in [
          ChromaLocation::Unspecified,
          ChromaLocation::other("unassigned-99"),
          ChromaLocation::Left,
          ChromaLocation::TopLeft,
          ChromaLocation::BottomLeft,
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let (y_wire, _) = pack_p2xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);
        let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);

        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

        let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let uv444 = ref_full_uv_interleaved(&up, &vp, BITS);

        // RGBA (u8).
        {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

          let ref_src = $RefFrame::new(&y_wire, &uv444, W, H, W, 2 * W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          assert_eq!((h, s, v), (hr, sr, vr), "centered HSV must equal upsample-then-P4xx");
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_and_bottom_route_like_center_horizontally() {
        // Top / Bottom share Center's horizontal (centered) phase; 4:2:2 has no
        // vertical axis to drive, so all three agree.
        let center = convert_rgb(ChromaLocation::Center, true);
        assert_eq!(convert_rgb(ChromaLocation::Top, true), center);
        assert_eq!(convert_rgb(ChromaLocation::Bottom, true), center);
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
        // BEFORE the 1/4-3/4 blend, exactly as the fused decode does. (At
        // BITS = 16 there are no ignored bits, so this is clean == clean.)
        let low_dirty = (1u16 << (16 - BITS)).wrapping_sub(1);
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let decode = |y: &[u16], uv: &[u16]| -> Vec<u8> {
          let src = $Frame::new(y, uv, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let y_be: Vec<u16> = y_wire.iter().map(|&x| x.to_be()).collect();
        let uv_be: Vec<u16> = uv_wire.iter().map(|&x| x.to_be()).collect();
        let decode = |y: &[u16], uv: &[u16]| -> Vec<u8> {
          let src = $FrameBe::try_new(y, uv, W, H, W, W).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
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
          $walker(&src_ok, false, sink_ok.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
        let err = $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
        // The P-formats are NOT ChromaDerivedNcl-primaries-wired. BOTH paths —
        // the default fused P-format kernel AND the centered P4xx 4:4:4 kernel —
        // resolve ChromaDerivedNcl via the shared BT.709 matrix-tag fallback
        // (`Coefficients::for_matrix`), IGNORING the ColorSpec primaries, so
        // default and centered stay internally consistent (the centered phase
        // shift is the ONLY difference between them).
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuv422p,
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
            .with_color_spec(&spec(loc)).unwrap();
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::ChromaDerivedNcl)).unwrap();
          rgb
        };
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone());
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
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
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let src = $Frame::new(&y_wire, &uv_wire, W, H, W, W);

        let info = ColorInfo::new(
          Primaries::Bt709,
          Transfer::Bt709,
          ColorMatrix::Bt601,
          DynamicRange::Limited,
          ChromaLocation::Center,
        );
        let spec = ColorSpec::from_info(PixelFormat::Yuv422p, info);
        let opts = YuvOptions::from_color_spec(&spec);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_color_spec(&spec).unwrap();
        $walker(&src, opts.full_range(), &mut sink).unwrap();
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

      // ---- no-output invariant: guard runs before the row-offset math ------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "constructs an absurd geometry; the no-op contract is the point, not Miri"
      )]
      fn no_output_row_large_geometry_does_not_overflow() {
        // The no-output guard must run BEFORE the `idx * w` single-plane offset
        // arithmetic. A no-output `process` call never ran an attach-time
        // `w x h x 1` validation, so on a 32-bit target (`usize == u32`) an absurd
        // geometry where `idx * w` exceeds `u32::MAX` would overflow that offset
        // and panic under overflow checks. With no outputs attached, `process`
        // must return `Ok(())` having done NO row math and NO allocation.
        //
        // w = 4, idx = 2^30 -> idx * w = 2^32 = u32::MAX + 1 (overflows u32).
        let w: usize = 4;
        let idx: usize = 1 << 30;
        let h: usize = idx + 1; // idx < height so the row-index check passes
        assert!(
          (idx as u64) * (w as u64) > u32::MAX as u64,
          "test geometry must exceed u32::MAX to exercise the 32-bit offset overflow"
        );

        let y = std::vec![pack((MAXV / 2) as u16, BITS); w];
        let uv = std::vec![pack((MAXV / 2) as u16, BITS); w]; // w/2 interleaved pairs
        let mut sink =
          MixedSinker::<$Marker>::new(w, h).with_chroma_location(ChromaLocation::Center);
        // No outputs attached: the guard returns before `idx * w` (no overflow
        // panic) and before the centered preflight (no allocation).
        let row = crate::source::$Row::for_tests(&y, &uv, idx, KernelMatrix::Bt601, false);
        crate::PixelSink::process(&mut sink, row).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(
          chroma_len, 0,
          "a no-output large-geometry high-bit row must allocate nothing"
        );
      }

      // ---- mid-frame siting-flip rejection (identity path freeze) ---------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_path_mid_frame_siting_flip_is_rejected() {
        // The identity (no-resample) high-bit semi-planar 4:2:2 decode freezes the
        // horizontal siting on its first output-bearing row, mirroring the already-
        // frozen Nv20 sibling. Flipping co-sited ⇆ centered mid-frame must reject the
        // next in-sequence row with `ChromaSitingChanged` (leaving the centered
        // chroma scratch untouched); flipping back and retrying then matches a clean
        // single-phase decode byte-for-byte.
        let (yp, up, vp) = ramp_planes_logical(MAXV);
        let (y_wire, uv_wire) = pack_p2xx(&yp, &up, &vp, BITS);
        let w = W as usize;
        let h = H as usize;
        for (loc1, loc2) in [
          (ChromaLocation::Left, ChromaLocation::Center),
          (ChromaLocation::Center, ChromaLocation::Left),
        ] {
          let want = convert_rgb(loc1.clone(), true);
          let mut rgb = std::vec![0u8; w * h * 3];
          let mut sink = MixedSinker::<$Marker>::new(w, h)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc1.clone())
            .with_simd(true);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          let row0 =
            crate::source::$Row::for_tests(&y_wire[0..w], &uv_wire[0..w], 0, KernelMatrix::Bt601, false);
          crate::PixelSink::process(&mut sink, row0).unwrap();
          let scratch_len = sink.chroma_full_u16.len();

          sink.set_chroma_location(loc2.clone());
          let row1 = crate::source::$Row::for_tests(
            &y_wire[w..2 * w],
            &uv_wire[w..2 * w],
            1,
            KernelMatrix::Bt601,
            false,
          );
          let err = crate::PixelSink::process(&mut sink, row1).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "direct path {loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
          assert_eq!(
            sink.chroma_full_u16.len(),
            scratch_len,
            "{loc1:?}->{loc2:?}: a rejected flip must not grow the chroma scratch"
          );

          sink.set_chroma_location(loc1.clone());
          for r in 1..h {
            let row = crate::source::$Row::for_tests(
              &y_wire[r * w..(r + 1) * w],
              &uv_wire[r * w..(r + 1) * w],
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
    }
  };
}

p2xx_chroma_tests!(
  p210,
  10,
  P210,
  P210Frame,
  p210_to,
  P410,
  P410Frame,
  p410_to,
  P210<true>,
  P210BeFrame,
  p210_to_endian,
  P210Row
);
p2xx_chroma_tests!(
  p212,
  12,
  P212,
  P212Frame,
  p212_to,
  P412,
  P412Frame,
  p412_to,
  P212<true>,
  P212BeFrame,
  p212_to_endian,
  P212Row
);
p2xx_chroma_tests!(
  p216,
  16,
  P216,
  P216Frame,
  p216_to,
  P416,
  P416Frame,
  p416_to,
  P216<true>,
  P216BeFrame,
  p216_to_endian,
  P216Row
);
