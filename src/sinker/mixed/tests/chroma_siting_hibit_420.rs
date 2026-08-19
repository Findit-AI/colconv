//! Chroma-siting-aware **high-bit** 4:2:0 upsampling for `Yuv420p9` …
//! `Yuv420p16` (#302).
//!
//! Covers, per bit depth (9 / 10 / 12 / 14 / 16, via the macro below): the
//! default / co-sited path staying byte-identical to the pre-#302
//! nearest-neighbor decode (the regression guard, plus its negative control
//! that the centered phase actually moves chroma); the centered RGB / RGBA /
//! HSV decodes — and their `u16` twins — matching an independent
//! "upsample-then-4:4:4" reference; SIMD-vs-scalar parity of the centered
//! path; the preflight-ordering atomicity (a centered chroma-scratch alloc
//! failure leaves luma AND colour untouched); and the `ChromaDerivedNcl`
//! consistency invariant (the high-bit formats are NOT primaries-wired, so
//! BOTH the default and centered paths resolve it via the BT.709 matrix-tag
//! fallback). The bit-exact `u16` upsample kernel is also checked directly
//! against a hand-computed oracle, including the big-endian wire path.
//!
//! The macro instantiates each bit depth with its **little-endian** marker, so
//! a sample's wire `u16` equals its logical value on the (little-endian) test
//! host; the references compute in that logical domain. The endianness
//! re-encode is exercised host-independently by the kernel-level BE oracle.

use super::*;
use crate::{ChromaLocation, ColorMatrix};

const W: u32 = 16;
const H: u32 = 8;

/// Builds a high-bit 4:2:0 frame's logical planes: flat mid-gray luma plus a
/// per-column chroma ramp (distinct adjacent columns so the horizontal phase
/// is observable; the small `+ r` term keeps chroma rows from being identical
/// so a vertical mistake would surface). Values are clamped to `maxv =
/// (1 << BITS) - 1`.
fn ramp_planes_n(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp, on logical
/// `u16`. Written separately from the production kernel so it is a real oracle.
fn ref_upsample_center_h_u16(c_half: &[u16], width: usize) -> Vec<u16> {
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

/// Builds the full-resolution U / V a centered-siting high-bit 4:2:0 decode
/// reconstructs: each luma row `r` takes chroma row `r / 2` (the walker's
/// vertical replication, unchanged by #302) horizontally upsampled with the
/// centered weights. Feeding these to the matching `Yuv444pN` conversion is the
/// end-to-end oracle for the centered path.
fn ref_full_chroma_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  for r in 0..h {
    let cr = r / 2;
    let urow = ref_upsample_center_h_u16(&u420[cr * cw..cr * cw + cw], w);
    let vrow = ref_upsample_center_h_u16(&v420[cr * cw..cr * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

/// Independent reference for the bottom-sited (`v = 1`) full reconstruction on
/// logical `u16` (RFC #238 S6d): per luma row `r`, the EVEN rows take the
/// vertical box average `(prev + cur + 1) >> 1` of chroma rows `r/2 - 1`
/// (clamped to `r/2` at the top edge) and `r/2`; the ODD rows take chroma row
/// `r/2` directly. Each half-row is then horizontally upsampled with the SAME
/// centered `1/4`–`3/4` weights (Bottom is `h = 0.5`). Written separately from
/// the production kernel so it is a true oracle.
fn ref_full_chroma_bottom_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u16> {
    (0..cw)
      .map(|c| {
        let a = plane[prev * cw + c] as u32;
        let b = plane[cr * cw + c] as u32;
        ((a + b + 1) >> 1) as u16
      })
      .collect::<Vec<u16>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u420, cr, prev), vblend(v420, cr, prev))
    } else {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    let urow = ref_upsample_center_h_u16(&uhalf, w);
    let vrow = ref_upsample_center_h_u16(&vhalf, w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

/// The full-resolution U / V a **`BottomLeft`** (`h = 0`, `v = 1`) high-bit 4:2:0
/// decode reconstructs: [`ref_full_chroma_bottom_u16`]'s even-row vertical box
/// blend, but fed to the CO-SITED horizontal 2× replicate instead of the centered
/// kernel. The direct-decode ground truth for the co-sited-h + bottom-v siting.
fn ref_full_chroma_bottomleft_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  let vblend = |plane: &[u16], cr: usize, prev: usize| -> Vec<u16> {
    (0..cw)
      .map(|c| {
        let a = plane[prev * cw + c] as u32;
        let b = plane[cr * cw + c] as u32;
        ((a + b + 1) >> 1) as u16
      })
      .collect::<Vec<u16>>()
  };
  let replicate = |half: &[u16]| -> Vec<u16> {
    let mut out = std::vec![0u16; w];
    for j in 0..cw {
      out[2 * j] = half[j];
      out[2 * j + 1] = half[j];
    }
    out
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 0 {
      let prev = cr.saturating_sub(1);
      (vblend(u420, cr, prev), vblend(v420, cr, prev))
    } else {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    u444[r * w..r * w + w].copy_from_slice(&replicate(&uhalf));
    v444[r * w..r * w + w].copy_from_slice(&replicate(&vhalf));
  }
  (u444, v444)
}

/// Independent reference for the top-sited (`v = 0`, FORWARD fold) full
/// reconstruction on logical `u16` (RFC #238 Top) — the vertical MIRROR of
/// [`ref_full_chroma_bottom_u16`]: per luma row `r`, the EVEN rows take chroma
/// row `r/2` directly (co-sited on the even row); the ODD rows take the vertical
/// box average `(cur + next + 1) >> 1` of chroma rows `r/2` and `r/2 + 1`
/// (clamped to `r/2` at the BOTTOM edge). Each half-row is then horizontally
/// upsampled with the SAME centered `1/4`–`3/4` weights (Top is `h = 0.5`).
fn ref_full_chroma_top_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h.div_ceil(2);
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  let vblend = |plane: &[u16], cr: usize, next: usize| -> Vec<u16> {
    (0..cw)
      .map(|c| {
        let a = plane[cr * cw + c] as u32;
        let b = plane[next * cw + c] as u32;
        ((a + b + 1) >> 1) as u16
      })
      .collect::<Vec<u16>>()
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 1 {
      let next = (cr + 1).min(ch - 1);
      (vblend(u420, cr, next), vblend(v420, cr, next))
    } else {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    let urow = ref_upsample_center_h_u16(&uhalf, w);
    let vrow = ref_upsample_center_h_u16(&vhalf, w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

/// The full-resolution U / V a **`TopLeft`** (`h = 0`, `v = 0`) high-bit 4:2:0
/// decode reconstructs: [`ref_full_chroma_top_u16`]'s odd-row forward vertical
/// box blend, but fed to the CO-SITED horizontal 2× replicate instead of the
/// centered kernel. The direct-decode ground truth for the co-sited-h + top-v
/// siting.
fn ref_full_chroma_topleft_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h.div_ceil(2);
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  let vblend = |plane: &[u16], cr: usize, next: usize| -> Vec<u16> {
    (0..cw)
      .map(|c| {
        let a = plane[cr * cw + c] as u32;
        let b = plane[next * cw + c] as u32;
        ((a + b + 1) >> 1) as u16
      })
      .collect::<Vec<u16>>()
  };
  let replicate = |half: &[u16]| -> Vec<u16> {
    let mut out = std::vec![0u16; w];
    for j in 0..cw {
      out[2 * j] = half[j];
      out[2 * j + 1] = half[j];
    }
    out
  };
  for r in 0..h {
    let cr = r / 2;
    let (uhalf, vhalf) = if r & 1 == 1 {
      let next = (cr + 1).min(ch - 1);
      (vblend(u420, cr, next), vblend(v420, cr, next))
    } else {
      (
        u420[cr * cw..cr * cw + cw].to_vec(),
        v420[cr * cw..cr * cw + cw].to_vec(),
      )
    };
    u444[r * w..r * w + w].copy_from_slice(&replicate(&uhalf));
    v444[r * w..r * w + w].copy_from_slice(&replicate(&vhalf));
  }
  (u444, v444)
}

// ---- u16 kernel oracle (endianness-explicit) -------------------------------

#[test]
fn center_upsample_u16_kernel_matches_hand_computed() {
  // c = [0, 0, 400, 400] (half = 4, width = 8), little-endian wire.
  //   even 2j   = (c[j-1] + 3·c[j] + 2) >> 2
  //   odd  2j+1 = (3·c[j] + c[j+1] + 2) >> 2
  // Values < 512 fit every depth, so the `BITS` mask is a no-op here; the
  // dirty-upper-bit masking is exercised by
  // `center_upsample_u16_kernel_masks_dirty_upper_bits`.
  let c_half = [0u16, 0, 400, 400].map(u16::to_le);
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<10>(&c_half, &mut out, 8, false);
  assert_eq!(out.map(u16::from_le), [0, 0, 0, 100, 300, 400, 400, 400]);
}

#[test]
fn center_upsample_u16_kernel_clamps_edges() {
  // Width 4: left edge even = c[0] exactly, right edge odd = c[last] exactly.
  let c_half = [1000u16, 2000].map(u16::to_le);
  let mut out = [0u16; 4];
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<12>(&c_half, &mut out, 4, false);
  let dec = out.map(u16::from_le);
  assert_eq!(dec, [1000, 1250, 1750, 2000]);
  assert_eq!(dec[0], 1000, "left edge even column is co-sited");
  assert_eq!(dec[3], 2000, "right edge odd column is co-sited");
}

#[test]
fn center_upsample_u16_kernel_big_endian_matches_le_logical() {
  // Same LOGICAL input, wire-encoded big-endian: the kernel interpolates in the
  // logical domain and re-encodes to the same wire order, so decoding the BE
  // output back yields the SAME logical result as the LE path. Host-independent.
  let logical = [0u16, 0, 400, 400];
  let le: Vec<u16> = logical.iter().map(|&x| x.to_le()).collect();
  let be: Vec<u16> = logical.iter().map(|&x| x.to_be()).collect();
  let mut out_le = [0u16; 8];
  let mut out_be = [0u16; 8];
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<10>(&le, &mut out_le, 8, false);
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<10>(&be, &mut out_be, 8, true);
  let dec_le: Vec<u16> = out_le.iter().map(|&x| u16::from_le(x)).collect();
  let dec_be: Vec<u16> = out_be.iter().map(|&x| u16::from_be(x)).collect();
  assert_eq!(
    dec_be, dec_le,
    "BE wire path must equal the LE logical interpolation"
  );
  assert_eq!(dec_be, std::vec![0u16, 0, 0, 100, 300, 400, 400, 400]);
}

#[test]
fn center_upsample_u16_kernel_masks_dirty_upper_bits() {
  // The fused high-bit decode kernels mask low-packed samples to BITS
  // (`& bits_mask::<BITS>()`) before use, sanitizing dirty upper bits in a
  // malformed-but-accepted frame. The centered upsample must do the SAME BEFORE
  // its 1/4-3/4 blend — otherwise a dirty sample's high bits leak into a
  // neighbour's low bits. For every sub-16-bit depth and both wire endians, a
  // frame with ALL bits above BITS set must blend identically to the masked
  // (clean) frame, and stay within `[0, (1 << BITS) - 1]`.
  fn check<const BITS: u32>() {
    let mask = ((1u32 << BITS) - 1) as u16;
    let upper = !mask; // every bit above BITS
    let clean = [0u16, 0, mask, mask]; // half = 4, width = 8; a non-constant ramp
    let dirty = [
      clean[0] | upper,
      clean[1] | upper,
      clean[2] | upper,
      clean[3] | upper,
    ];
    for &be in &[false, true] {
      let enc = |v: u16| if be { v.to_be() } else { v.to_le() };
      let dec = |v: u16| if be { u16::from_be(v) } else { u16::from_le(v) };
      let dirty_wire: Vec<u16> = dirty.iter().map(|&v| enc(v)).collect();
      let clean_wire: Vec<u16> = clean.iter().map(|&v| enc(v)).collect();
      let mut out_dirty = [0u16; 8];
      let mut out_clean = [0u16; 8];
      crate::row::scalar::chroma_upsample_2to1_center_h_u16::<BITS>(
        &dirty_wire,
        &mut out_dirty,
        8,
        be,
      );
      crate::row::scalar::chroma_upsample_2to1_center_h_u16::<BITS>(
        &clean_wire,
        &mut out_clean,
        8,
        be,
      );
      let dec_dirty: Vec<u16> = out_dirty.iter().map(|&v| dec(v)).collect();
      let dec_clean: Vec<u16> = out_clean.iter().map(|&v| dec(v)).collect();
      assert_eq!(
        dec_dirty, dec_clean,
        "BITS={BITS} be={be}: dirty upper bits must be masked before the blend"
      );
      assert!(
        dec_dirty.iter().all(|&v| v <= mask),
        "BITS={BITS} be={be}: blended output must stay within the bit depth"
      );
    }
  }
  check::<9>();
  check::<10>();
  check::<12>();
  check::<14>();

  // 16-bit has no spare bits: the mask is `u16::MAX` (a no-op), so a
  // top-of-range sample is preserved through the blend.
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<16>(
    &[0u16, 0, 65535, 65535].map(u16::to_le),
    &mut out,
    8,
    false,
  );
  assert_eq!(
    out.map(u16::from_le),
    [0, 0, 0, 16384, 49151, 65535, 65535, 65535]
  );
}

// ---- u16 bottom (v = 1) kernel oracle (RFC #238 S6d) -----------------------

#[test]
fn bottom_even_upsample_u16_kernel_matches_hand_computed() {
  // prev = [0, 0, 100, 100], cur = [40, 40, 60, 60] (half = 4, width = 8), LE.
  //   e = (prev + cur + 1) >> 1 = [20, 20, 80, 80], then the centered horizontal
  //   1/4-3/4 reconstruction: 2j = (e[j-1] + 3e[j] + 2) >> 2, 2j+1 = (3e[j] +
  //   e[j+1] + 2) >> 2. Values < 512 fit every depth, so the BITS mask is a no-op.
  let prev = [0u16, 0, 100, 100].map(u16::to_le);
  let cur = [40u16, 40, 60, 60].map(u16::to_le);
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<10>(&prev, &cur, &mut out, 8, false);
  assert_eq!(out.map(u16::from_le), [20, 20, 20, 35, 65, 80, 80, 80]);
}

#[test]
fn bottom_even_upsample_u16_kernel_equals_center_when_rows_match() {
  // prev == cur => the vertical box blend is a no-op, so the bottom-even kernel
  // reproduces the plain horizontal centered upsample exactly.
  let cur = [10u16, 400, 900, 300];
  let mut bottom = [0u16; 8];
  let mut center = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<10>(
    &cur,
    &cur,
    &mut bottom,
    8,
    false,
  );
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<10>(&cur, &mut center, 8, false);
  assert_eq!(
    bottom, center,
    "prev == cur must collapse the vertical blend to the horizontal centered path"
  );
}

#[test]
fn bottom_even_upsample_u16_kernel_big_endian_matches_le_logical() {
  // Same LOGICAL input, wire-encoded big-endian: the kernel blends in the logical
  // domain and re-encodes to BE, so decoding the BE output back yields the SAME
  // logical result as the LE path. Host-independent.
  let prev = [0u16, 0, 400, 400];
  let cur = [100u16, 100, 200, 200];
  let enc = |s: &[u16], be: bool| -> Vec<u16> {
    s.iter()
      .map(|&x| if be { x.to_be() } else { x.to_le() })
      .collect()
  };
  let mut out_le = [0u16; 8];
  let mut out_be = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<10>(
    &enc(&prev, false),
    &enc(&cur, false),
    &mut out_le,
    8,
    false,
  );
  crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<10>(
    &enc(&prev, true),
    &enc(&cur, true),
    &mut out_be,
    8,
    true,
  );
  let dec_le: Vec<u16> = out_le.iter().map(|&x| u16::from_le(x)).collect();
  let dec_be: Vec<u16> = out_be.iter().map(|&x| u16::from_be(x)).collect();
  assert_eq!(
    dec_be, dec_le,
    "BE bottom kernel must equal LE for the same logical planes"
  );
}

#[test]
fn bottom_even_upsample_u16_kernel_masks_dirty_upper_bits() {
  // A malformed low-packed input with bits set ABOVE BITS must blend identically
  // to the masked clean input: the kernel masks each sample to BITS AFTER the
  // endian load and BEFORE the blend, so dirty high bits never leak into a
  // neighbour's low bits, and the output stays within `[0, (1 << BITS) - 1]`.
  fn check<const BITS: u32>() {
    let mask = ((1u32 << BITS) - 1) as u16;
    let upper = !mask;
    let prev = [10u16 & mask, 300 & mask, 200 & mask, 50 & mask];
    let cur = [80u16 & mask, 40 & mask, 260 & mask, 120 & mask];
    for be in [false, true] {
      let enc = |s: &[u16]| -> Vec<u16> {
        s.iter()
          .map(|&x| if be { x.to_be() } else { x.to_le() })
          .collect()
      };
      let dirty = |s: &[u16]| -> Vec<u16> {
        s.iter()
          .map(|&x| {
            let d = x | upper;
            if be { d.to_be() } else { d.to_le() }
          })
          .collect()
      };
      let mut clean_out = [0u16; 8];
      let mut dirty_out = [0u16; 8];
      crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
        &enc(&prev),
        &enc(&cur),
        &mut clean_out,
        8,
        be,
      );
      crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
        &dirty(&prev),
        &dirty(&cur),
        &mut dirty_out,
        8,
        be,
      );
      assert_eq!(
        clean_out, dirty_out,
        "BITS={BITS} be={be}: dirty upper bits must be masked before the blend"
      );
      let dec_max = clean_out
        .iter()
        .map(|&x| u32::from(if be { u16::from_be(x) } else { u16::from_le(x) }))
        .max()
        .unwrap();
      assert!(
        dec_max <= mask as u32,
        "BITS={BITS} be={be}: blended output must stay within the bit depth"
      );
    }
  }
  check::<10>();
  check::<12>();
}

// ---- u16 bottom-LEFT (h = 0, v = 1) kernel oracle --------------------------

#[test]
fn cosited_upsample_u16_kernel_matches_hand_computed_and_be_matches_le() {
  // Co-sited horizontal reconstruction is a plain 2× replicate (masked to BITS).
  let c = [10u16, 400, 900, 300];
  let enc = |s: &[u16], be: bool| -> Vec<u16> {
    s.iter()
      .map(|&x| if be { x.to_be() } else { x.to_le() })
      .collect()
  };
  let mut out_le = [0u16; 8];
  crate::row::scalar::chroma_upsample_2to1_cosited_h_u16::<10>(
    &enc(&c, false),
    &mut out_le,
    8,
    false,
  );
  assert_eq!(
    out_le.map(u16::from_le),
    [10, 10, 400, 400, 900, 900, 300, 300]
  );
  let mut out_be = [0u16; 8];
  crate::row::scalar::chroma_upsample_2to1_cosited_h_u16::<10>(
    &enc(&c, true),
    &mut out_be,
    8,
    true,
  );
  let dec_be: Vec<u16> = out_be.iter().map(|&x| u16::from_be(x)).collect();
  assert_eq!(
    dec_be,
    out_le.map(u16::from_le).to_vec(),
    "BE co-sited kernel must equal LE for the same logical samples"
  );
}

#[test]
fn bottomleft_even_upsample_u16_kernel_matches_hand_computed() {
  // prev = [0, 0, 100, 100], cur = [40, 40, 60, 60]; e = (prev + cur + 1) >> 1 =
  // [20, 20, 80, 80], then the CO-SITED 2× replicate.
  let prev = [0u16, 0, 100, 100].map(u16::to_le);
  let cur = [40u16, 40, 60, 60].map(u16::to_le);
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<10>(
    &prev, &cur, &mut out, 8, false,
  );
  assert_eq!(out.map(u16::from_le), [20, 20, 20, 20, 80, 80, 80, 80]);
}

#[test]
fn bottomleft_even_upsample_u16_kernel_equals_cosited_when_rows_match() {
  let cur = [10u16, 400, 900, 300];
  let mut bl = [0u16; 8];
  let mut cosited = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<10>(
    &cur, &cur, &mut bl, 8, false,
  );
  crate::row::scalar::chroma_upsample_2to1_cosited_h_u16::<10>(&cur, &mut cosited, 8, false);
  assert_eq!(
    bl, cosited,
    "prev == cur must collapse the vertical blend to the co-sited replicate"
  );
}

#[test]
fn bottomleft_even_upsample_u16_kernel_be_matches_le_and_masks_dirty_bits() {
  fn check<const BITS: u32>() {
    let mask = ((1u32 << BITS) - 1) as u16;
    let upper = !mask;
    let prev = [10u16 & mask, 300 & mask, 200 & mask, 50 & mask];
    let cur = [80u16 & mask, 40 & mask, 260 & mask, 120 & mask];
    for be in [false, true] {
      let enc = |s: &[u16]| -> Vec<u16> {
        s.iter()
          .map(|&x| if be { x.to_be() } else { x.to_le() })
          .collect()
      };
      let dirty = |s: &[u16]| -> Vec<u16> {
        s.iter()
          .map(|&x| {
            let d = x | upper;
            if be { d.to_be() } else { d.to_le() }
          })
          .collect()
      };
      let mut clean = [0u16; 8];
      let mut dirtied = [0u16; 8];
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
        &enc(&prev),
        &enc(&cur),
        &mut clean,
        8,
        be,
      );
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
        &dirty(&prev),
        &dirty(&cur),
        &mut dirtied,
        8,
        be,
      );
      assert_eq!(
        clean, dirtied,
        "BITS={BITS} be={be}: dirty upper bits must be masked before the blend"
      );
      let dec_max = clean
        .iter()
        .map(|&x| u32::from(if be { u16::from_be(x) } else { u16::from_le(x) }))
        .max()
        .unwrap();
      assert!(
        dec_max <= mask as u32,
        "BITS={BITS} be={be}: blended output must stay within the bit depth"
      );
    }
  }
  check::<10>();
  check::<12>();
}

/// Vertical chroma ramp: flat mid-gray luma with chroma CONSTANT across columns
/// but stepping strongly per ROW, so the `Bottom` vertical blend is observable in
/// isolation (a horizontal-only siting leaves it untouched). Values clamp to
/// `maxv`.
fn vramp_planes_n(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let (cw, ch) = (w / 2, h / 2);
  let step = (maxv / 8).max(1);
  let y = std::vec![(maxv / 2) as u16; w * h];
  let mut u = std::vec![0u16; cw * ch];
  let mut v = std::vec![0u16; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (step + r as u32 * step).min(maxv) as u16;
      v[r * cw + c] = maxv.saturating_sub(r as u32 * step).max(step) as u16;
    }
  }
  (y, u, v)
}

// ---- per-bit-depth suite ---------------------------------------------------

// The suite is identical bar the bit depth, format marker, frame type, and
// walker, so generate it once per depth. Each lands in its own `mod` so the
// names don't collide.
macro_rules! hibit_420_chroma_tests {
  ($mod:ident, $bits:expr, $Marker:ident, $Frame:ident, $walker:ident, $Ref:ident, $RefFrame:ident, $ref_walker:ident, $MarkerBe:ty, $FrameBe:ident, $walker_be:ident, $Row:ident) => {
    mod $mod {
      use super::*;

      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Centered/default identity-decode RGB for a siting + SIMD toggle (over
      /// the horizontal `ramp_planes_n` fixture).
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        convert_rgb_with(loc, simd, &yp, &up, &vp)
      }

      /// Identity-decode RGB for a siting + SIMD toggle over EXPLICIT planes (so
      /// the bottom-sited tests can drive the vertical `vramp_planes_n` fixture).
      fn convert_rgb_with(loc: ChromaLocation, simd: bool, yp: &[u16], up: &[u16], vp: &[u16]) -> Vec<u8> {
        let src = $Frame::new(yp, up, vp, W, H, W, W / 2, W / 2);
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
        // `TopLeft` (`v = 0`) now folds the forward vertical triangle (RFC #238
        // Top), so it LEAVES the co-sited byte-identity group.
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
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
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
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
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
          "centered path stages U+V at full width"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb_ref,
          "centered high-bit 4:2:0 RGB must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_u16_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

        assert_eq!(
          rgb16, rgb16_ref,
          "centered high-bit 4:2:0 RGB(u16) must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_rgba_u16_and_hsv_match_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        // RGBA (u8).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          assert_eq!(rgba, rgba_ref, "centered RGBA must equal upsample-then-4:4:4");
        }

        // RGBA (u16).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba16 = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          assert_eq!(
            rgba16, rgba16_ref,
            "centered RGBA(u16) must equal upsample-then-4:4:4"
          );
        }

        // HSV-direct (no RGB / RGBA attached).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
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

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          assert_eq!(
            (h, s, v),
            (hr, sr, vr),
            "centered HSV must equal upsample-then-4:4:4"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_differs_from_center_and_bottom_folding_vertically() {
        // RFC #238 Top folds the `v = 0` FORWARD vertical triangle, so on a
        // vertical chroma ramp (strong per-row step) it must DIVERGE from BOTH the
        // vertically co-sited Center and the backward-folding Bottom.
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let top = convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp);
        let center = convert_rgb_with(ChromaLocation::Center, true, &yp, &up, &vp);
        let bottom = convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp);
        assert_ne!(top, center, "Top must fold the v=0 forward phase (differs from Center)");
        assert_ne!(top, bottom, "Top (v=0) must differ from Bottom (v=1)");
        assert_ne!(bottom, center, "Bottom must fold the vertical phase (differs from Center)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_rgb_and_u16_match_forward_vblend_then_444_reference() {
        // Top RGB (u8 and native u16) must equal the independent forward-vblend +
        // centered-h reconstruction fed to an identity 4:4:4 sink.
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_top_u16(&up, &vp);

        // RGB (u8).
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
          rgb_ref,
          "top-sited high-bit 4:2:0 RGB must equal forward-vblend then 4:4:4"
        );

        // RGB (u16).
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Top);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(rgb16, rgb16_ref, "top-sited RGB u16 must equal forward-vblend-then-4:4:4");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_path_simd_matches_scalar() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        assert_eq!(
          convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::Top, false, &yp, &up, &vp),
          "top path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn topleft_matches_cosited_forward_vblend_and_differs_from_top() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_topleft_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
          rgb_ref,
          "top-left high-bit 4:2:0 RGB must equal co-sited-replicate + forward-vblend then 4:4:4"
        );
        assert_eq!(
          convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::TopLeft, false, &yp, &up, &vp),
          "top-left path must be bit-identical across SIMD and scalar"
        );
        // TopLeft (h=0) must differ from Top (h=0.5) on a horizontal ramp.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        assert_ne!(
          convert_rgb_with(ChromaLocation::TopLeft, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::Top, true, &yp, &up, &vp),
          "TopLeft (h=0) must differ from Top (h=0.5) on a horizontal ramp"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_mid_frame_flip_is_rejected_and_begin_frame_clears_held_row() {
        // RFC #238 Top identity decode: an ODD `Top` row DEFERS its colour output
        // into the forward one-row delay (`chroma_top_pending` set + the Top phase
        // frozen). A mid-frame flip TO a non-Top siting before the following even
        // row must reject with `ChromaSitingChanged` (leaving the held row + the
        // lookback untouched), and `begin_frame` must CLEAR both the held row and
        // the frozen Top phase (the Nv21 cross-frame-corruption regression class).
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let w = W as usize;
        let cw = w / 2;
        for (loc1, loc2) in [
          (ChromaLocation::Top, ChromaLocation::Center),
          (ChromaLocation::TopLeft, ChromaLocation::Left),
        ] {
          let mut rgb = std::vec![0u8; w * H as usize * 3];
          let mut sink = MixedSinker::<$Marker>::new(w, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc1.clone())
            .with_simd(true);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          let row_at = |r: usize| {
            let cr = r / 2;
            $Row::for_tests(
              &yp[r * w..r * w + w],
              &up[cr * cw..cr * cw + cw],
              &vp[cr * cw..cr * cw + cw],
              r,
              KernelMatrix::Bt601,
              false,
            )
          };
          // Row 0 (even Top) decodes immediately and freezes the Top phase; row 1
          // (odd Top) DEFERS into the held delay line.
          crate::PixelSink::process(&mut sink, row_at(0)).unwrap();
          crate::PixelSink::process(&mut sink, row_at(1)).unwrap();
          assert!(
            sink.chroma_top_pending.is_some(),
            "{loc1:?}: odd Top row must be HELD in the forward delay line"
          );
          assert_eq!(
            sink.frozen_chroma_top_v,
            Some(true),
            "{loc1:?}: the Top vertical phase must be frozen"
          );
          // Mid-frame flip → reject; held row survives.
          sink.set_chroma_location(loc2.clone());
          let err = crate::PixelSink::process(&mut sink, row_at(2)).unwrap_err();
          assert!(
            matches!(err, MixedSinkerError::ChromaSitingChanged(_)),
            "{loc1:?}->{loc2:?}: want ChromaSitingChanged, got {err:?}"
          );
          assert!(
            sink.chroma_top_pending.is_some(),
            "{loc1:?}->{loc2:?}: a rejected flip must leave the held row untouched"
          );
          // begin_frame CLEARS the held row + the frozen Top phase.
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          assert!(
            sink.chroma_top_pending.is_none(),
            "{loc1:?}: begin_frame must clear the held Top row (cross-frame corruption guard)"
          );
          assert!(
            sink.frozen_chroma_top_v.is_none(),
            "{loc1:?}: begin_frame must clear the frozen Top phase"
          );
        }
      }

      // ---- bottom-sited (v = 1) vertical fold (RFC #238 S6d) ---------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_rgb_matches_vblend_then_444_reference() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_bottom_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
          rgb_ref,
          "bottom-sited high-bit 4:2:0 RGB must equal vblend + horizontal-upsample then 4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_rgb_u16_matches_vblend_then_444_reference() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_bottom_u16(&up, &vp);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Bottom);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          rgb16, rgb16_ref,
          "bottom-sited high-bit 4:2:0 RGB u16 must equal vblend-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_path_simd_matches_scalar() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        assert_eq!(
          convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::Bottom, false, &yp, &up, &vp),
          "bottom path must be bit-identical across the SIMD and scalar tiers"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottomleft_rgb_and_u16_match_cosited_vblend_reference() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_bottomleft_u16(&up, &vp);

        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(
          convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
          rgb_ref,
          "bottom-left high-bit 4:2:0 RGB must equal co-sited-replicate + vblend then 4:4:4"
        );

        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::BottomLeft);
        $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        assert_eq!(rgb16, rgb16_ref, "bottom-left RGB u16 must equal cosited-vblend-then-4:4:4");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottomleft_folds_vertically_simd_matches_scalar_and_differs_from_bottom() {
        let (yp, up, vp) = vramp_planes_n(MAXV);
        // v=1 fold: differs from the co-sited default on a vertical ramp.
        assert_ne!(
          convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::Left, true, &yp, &up, &vp),
          "BottomLeft must fold the vertical phase (differ from co-sited)"
        );
        assert_eq!(
          convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::BottomLeft, false, &yp, &up, &vp),
          "bottom-left path must be bit-identical across SIMD and scalar"
        );
        // Co-sited-h difference from Bottom shows on a horizontal ramp.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        assert_ne!(
          convert_rgb_with(ChromaLocation::BottomLeft, true, &yp, &up, &vp),
          convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp),
          "BottomLeft (h=0) must differ from Bottom (h=0.5) on a horizontal ramp"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_luma_only_then_late_color_box_blends() {
        // Always-maintained lookback: rows 0, 1 are processed LUMA-ONLY (no
        // colour), so the colour upsample never runs — yet the bottom lookback
        // must still be staged through them. After a late `set_rgb`, row 2 (Bottom
        // even, pair 1) must box-blend chroma rows 0 and 1 (== the all-output
        // reference), NOT clamp to chroma row 1.
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        let (u444, v444) = ref_full_chroma_bottom_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; w * h * 3];
        let mut ref_sink = MixedSinker::<$Ref>::new(w, h).with_rgb(&mut rgb_ref).unwrap();
        $ref_walker(&ref_src, false, ref_sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
        drop(ref_sink);
        let mut luma = std::vec![0u8; w * h];
        let mut rgb = std::vec![0u8; w * h * 3];
        {
          let mut sink = MixedSinker::<$Marker>::new(w, h)
            .with_luma(&mut luma)
            .unwrap()
            .with_chroma_location(ChromaLocation::Bottom);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          let feed = |sink: &mut MixedSinker<'_, $Marker>, r: usize| {
            let cr = r / 2;
            let row = $Row::for_tests(
              &yp[r * w..r * w + w],
              &up[cr * cw..cr * cw + cw],
              &vp[cr * cw..cr * cw + cw],
              r,
              KernelMatrix::Bt601,
              false,
            );
            crate::PixelSink::process(sink, row).unwrap();
          };
          feed(&mut sink, 0);
          feed(&mut sink, 1);
          sink.set_rgb(&mut rgb).unwrap();
          feed(&mut sink, 2);
          feed(&mut sink, 3);
        }
        let got_row2 = &rgb[2 * w * 3..3 * w * 3];
        assert_eq!(
          got_row2,
          &rgb_ref[2 * w * 3..3 * w * 3],
          "a luma-only-then-late-colour row 2 must box-blend chroma rows 0,1 (all-output reference)"
        );
        // Must NOT be the clamp (= Center's row 2, the centered upsample of chroma
        // row 1 only), which is what an unmaintained lookback would produce.
        let clamp = convert_rgb_with(ChromaLocation::Center, true, &yp, &up, &vp);
        assert_ne!(
          got_row2,
          &clamp[2 * w * 3..3 * w * 3],
          "row 2 must NOT clamp — the lookback was maintained through the luma-only rows"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn bottom_no_output_rows_do_not_enable_late_color_blend() {
        // The no-output invariant's correctness twin: rows 0, 1 are processed with
        // NO outputs (invisible — they return before the preflight and must not
        // prime the lookback). After a late `set_rgb`, row 2 (Bottom even) must
        // CLAMP (== Center's row 2), NOT box-blend through those invisible rows.
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let w = W as usize;
        let cw = w / 2;
        let mut rgb = std::vec![0u8; w * H as usize * 3];
        {
          let mut sink =
            MixedSinker::<$Marker>::new(w, H as usize).with_chroma_location(ChromaLocation::Bottom);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          let feed = |sink: &mut MixedSinker<'_, $Marker>, r: usize| {
            let cr = r / 2;
            let row = $Row::for_tests(
              &yp[r * w..r * w + w],
              &up[cr * cw..cr * cw + cw],
              &vp[cr * cw..cr * cw + cw],
              r,
              KernelMatrix::Bt601,
              false,
            );
            crate::PixelSink::process(sink, row).unwrap();
          };
          feed(&mut sink, 0);
          feed(&mut sink, 1);
          sink.set_rgb(&mut rgb).unwrap();
          feed(&mut sink, 2);
          feed(&mut sink, 3);
        }
        let got_row2 = &rgb[2 * w * 3..3 * w * 3];
        let clamp = convert_rgb_with(ChromaLocation::Center, true, &yp, &up, &vp);
        assert_eq!(
          got_row2,
          &clamp[2 * w * 3..3 * w * 3],
          "a no-output predecessor row must not enable a later colour even row to box-blend"
        );
        // Guard: the box-blend (the all-output bottom decode) is a DIFFERENT
        // value, so the clamp is observably the no-output-invisible behaviour.
        let bottom_full = convert_rgb_with(ChromaLocation::Bottom, true, &yp, &up, &vp);
        assert_ne!(
          got_row2,
          &bottom_full[2 * w * 3..3 * w * 3],
          "guard: the clamp must differ from the box-blend, so the no-output rows were invisible"
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

      // ---- dirty-upper-bit sanitization (mask before the blend) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_le() {
        // A malformed-but-accepted low-packed frame with bits set ABOVE BITS must
        // decode (centered) identically to the masked clean frame: the centered
        // upsample masks each sample to BITS BEFORE the 1/4-3/4 blend, exactly as
        // the fused decode kernels do, so a dirty sample's high bits never leak
        // into a neighbour's low bits. (At BITS = 16 `upper` is 0, so this is the
        // clean == clean identity — 16-bit has no spare bits.)
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let decode = |u: &[u16], v: &[u16]| -> Vec<u8> {
          let src = $Frame::new(&yp, u, v, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered LE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_be() {
        // Same invariant on the big-endian wire path: the mask is applied in the
        // logical domain (after the endian load), so dirty bits are stripped for
        // BE inputs too. Planes are BE-encoded and decoded via the BE marker /
        // frame / walker.
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let y_be: Vec<u16> = yp.iter().map(|&x| x.to_be()).collect();
        let decode = |u_logical: &[u16], v_logical: &[u16]| -> Vec<u8> {
          let u_be: Vec<u16> = u_logical.iter().map(|&x| x.to_be()).collect();
          let v_be: Vec<u16> = v_logical.iter().map(|&x| x.to_be()).collect();
          let src = $FrameBe::try_new(&y_be, &u_be, &v_be, W, H, W, W / 2, W / 2).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered BE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      // ---- preflight-ordering atomicity (#302 / #314, cf. #180) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        // luma PLUS a centered RGB decode whose u16 chroma-scratch allocation
        // fails must leave EVERY output buffer — luma included — untouched: the
        // centered scratch is reserved (fallibly) BEFORE any output row is
        // written, so a refusal can't half-update the frame.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
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
          matches!(
            err,
            MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
          ),
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
        // The high-bit Yuv420p formats are NOT ChromaDerivedNcl-primaries-wired
        // (only 8-bit Yuv420p got #316). BOTH paths — the default fused 4:2:0
        // kernel AND the centered 4:4:4 kernel — resolve ChromaDerivedNcl via the
        // shared BT.709 matrix-tag fallback (`Coefficients::for_matrix`), IGNORING
        // the ColorSpec primaries, so default and centered stay internally
        // consistent (the centered phase shift is the ONLY difference between
        // them). Full primaries-derived support is a documented Yuv420p-8bit-only
        // follow-up. This guards that consistency AND that the centered path did
        // not accidentally half-adopt primaries on one tier.
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes_n(MAXV);
        // ChromaDerivedNcl + Bt2020 primaries: were the decode to honour the
        // primaries (it must NOT here), it would diverge from BT.709. The
        // PixelFormat in the spec is cosmetic — the sink consumes only
        // chroma_location + primaries.
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
        // ChromaDerivedNcl(Bt2020) decode via the ColorSpec path.
        let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(&spec(loc)).unwrap();
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::ChromaDerivedNcl)).unwrap();
          rgb
        };
        // The BT.709 reference the matrix-tag fallback must equal.
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc.clone());
          $walker(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
          rgb
        };

        // Centered: ChromaDerivedNcl(Bt2020) resolves to the BT.709 fallback, NOT
        // the Bt2020-primaries-derived coefficients.
        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered high-bit ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
        );
        // Default (co-sited): same fallback → default and centered agree on the
        // coefficient path (neither half-adopts primaries).
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default high-bit ChromaDerivedNcl must resolve via the same BT.709 fallback"
        );
      }

      // ---- mid-frame siting-flip rejection (identity path freeze) ---------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn direct_path_mid_frame_siting_flip_is_rejected() {
        // The identity (no-resample) high-bit planar 4:2:0 decode freezes the
        // effective phase — BOTH the horizontal centered flag and the vertical
        // `Bottom` flag — on its first output-bearing row. Flipping `Bottom` ⇆
        // co-sited mid-frame must reject the next in-sequence row with
        // `ChromaSitingChanged`, WITHOUT growing the chroma scratch OR advancing the
        // stateful `chroma_prev_u16` vertical lookback (its validity tag included).
        // Flipping back and retrying then matches a clean single-phase decode.
        let (yp, up, vp) = vramp_planes_n(MAXV);
        let w = W as usize;
        let h = H as usize;
        let cw = w / 2;
        for (loc1, loc2) in [
          (ChromaLocation::Bottom, ChromaLocation::Left),
          (ChromaLocation::Left, ChromaLocation::Bottom),
          (ChromaLocation::Bottom, ChromaLocation::Center),
        ] {
          let want = convert_rgb_with(loc1.clone(), true, &yp, &up, &vp);
          let mut rgb = std::vec![0u8; w * h * 3];
          let mut sink = MixedSinker::<$Marker>::new(w, h)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc1.clone())
            .with_simd(true);
          crate::PixelSink::begin_frame(&mut sink, W, H).unwrap();
          for r in 0..2 {
            let cr = r / 2;
            let row = $Row::for_tests(
              &yp[r * w..r * w + w],
              &up[cr * cw..cr * cw + cw],
              &vp[cr * cw..cr * cw + cw],
              r,
              KernelMatrix::Bt601,
              false,
            );
            crate::PixelSink::process(&mut sink, row).unwrap();
          }
          let scratch_len = sink.chroma_full_u16.len();
          let prev_len = sink.chroma_prev_u16.len();
          let prev_tag = sink.chroma_prev_row;

          sink.set_chroma_location(loc2.clone());
          let row2 = $Row::for_tests(
            &yp[2 * w..3 * w],
            &up[cw..2 * cw],
            &vp[cw..2 * cw],
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

          sink.set_chroma_location(loc1.clone());
          for r in 2..h {
            let cr = r / 2;
            let row = $Row::for_tests(
              &yp[r * w..r * w + w],
              &up[cr * cw..cr * cw + cw],
              &vp[cr * cw..cr * cw + cw],
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

hibit_420_chroma_tests!(
  p9,
  9,
  Yuv420p9,
  Yuv420p9Frame,
  yuv420p9_to,
  Yuv444p9,
  Yuv444p9Frame,
  yuv444p9_to,
  Yuv420p9<true>,
  Yuv420p9BeFrame,
  yuv420p9_to_endian,
  Yuv420p9Row
);
hibit_420_chroma_tests!(
  p10,
  10,
  Yuv420p10,
  Yuv420p10Frame,
  yuv420p10_to,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to,
  Yuv420p10<true>,
  Yuv420p10BeFrame,
  yuv420p10_to_endian,
  Yuv420p10Row
);
hibit_420_chroma_tests!(
  p12,
  12,
  Yuv420p12,
  Yuv420p12Frame,
  yuv420p12_to,
  Yuv444p12,
  Yuv444p12Frame,
  yuv444p12_to,
  Yuv420p12<true>,
  Yuv420p12BeFrame,
  yuv420p12_to_endian,
  Yuv420p12Row
);
hibit_420_chroma_tests!(
  p14,
  14,
  Yuv420p14,
  Yuv420p14Frame,
  yuv420p14_to,
  Yuv444p14,
  Yuv444p14Frame,
  yuv444p14_to,
  Yuv420p14<true>,
  Yuv420p14BeFrame,
  yuv420p14_to_endian,
  Yuv420p14Row
);
hibit_420_chroma_tests!(
  p16,
  16,
  Yuv420p16,
  Yuv420p16Frame,
  yuv420p16_to,
  Yuv444p16,
  Yuv444p16Frame,
  yuv444p16_to,
  Yuv420p16<true>,
  Yuv420p16BeFrame,
  yuv420p16_to_endian,
  Yuv420p16Row
);
