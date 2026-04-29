use super::*;

// ---- Yuv420pFrame16 / Yuv420p10Frame ----------------------------------
//
// Storage is `&[u16]` with sample-indexed strides. Validation mirrors
// the 8-bit [`Yuv420pFrame`] with the addition of the `BITS` guard.

fn p10_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 16×8 frame, chroma 8×4. Y plane solid black (Y=0); UV planes
  // neutral (UV=512 = 10‑bit chroma center). Exact sample values
  // don't matter for the constructor tests that use this helper —
  // they only look at shape, geometry errors, and the reported
  // bits.
  (
    std::vec![0u16; 16 * 8],
    std::vec![512u16; 8 * 4],
    std::vec![512u16; 8 * 4],
  )
}

#[test]
fn yuv420p10_try_new_accepts_valid_tight() {
  let (y, u, v) = p10_planes();
  let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv420p10_try_new_accepts_odd_height() {
  // 16x9 → chroma_height = 5. Y plane 16*9 = 144 samples, U/V 8*5 = 40.
  let y = std::vec![0u16; 16 * 9];
  let u = std::vec![512u16; 8 * 5];
  let v = std::vec![512u16; 8 * 5];
  let f = Yuv420p10Frame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
  assert_eq!(f.height(), 9);
}

#[test]
fn yuv420p10_try_new_rejects_odd_width() {
  let (y, u, v) = p10_planes();
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::OddWidth { width: 15 }));
}

#[test]
fn yuv420p10_try_new_rejects_zero_dim() {
  let (y, u, v) = p10_planes();
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::ZeroDimension { .. }));
}

#[test]
fn yuv420p10_try_new_rejects_short_y_plane() {
  let y = std::vec![0u16; 10];
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
}

#[test]
fn yuv420p10_try_new_rejects_short_u_plane() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::UPlaneTooShort { .. }));
}

#[test]
fn yuv420p_frame16_try_new_rejects_unsupported_bits() {
  // BITS must be in {9, 10, 12, 14, 16}. 11, 15, etc. are rejected
  // before any plane math runs.
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![128u16; 8 * 4];
  let v = std::vec![128u16; 8 * 4];
  let e = Yuv420pFrame16::<11>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::UnsupportedBits { bits: 11 }
  ));
  let e15 = Yuv420pFrame16::<15>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e15,
    Yuv420pFrame16Error::UnsupportedBits { bits: 15 }
  ));
}

#[test]
fn yuv420p16_try_new_accepts_12_14_and_16() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![2048u16; 8 * 4];
  let v = std::vec![2048u16; 8 * 4];
  let f12 = Yuv420pFrame16::<12>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("12-bit valid");
  assert_eq!(f12.bits(), 12);
  let f14 = Yuv420pFrame16::<14>::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("14-bit valid");
  assert_eq!(f14.bits(), 14);
  let f16 = Yuv420p16Frame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("16-bit valid");
  assert_eq!(f16.bits(), 16);
}

#[test]
fn yuv420p16_try_new_checked_accepts_full_u16_range() {
  // At 16 bits the full u16 range is valid — max sample = 65535.
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 8 * 4];
  let v = std::vec![32768u16; 8 * 4];
  Yuv420p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8)
    .expect("every u16 value is in range at 16 bits");
}

#[test]
fn p016_try_new_accepts_16bit() {
  let y = std::vec![0xFFFFu16; 16 * 8];
  let uv = std::vec![0x8000u16; 16 * 4];
  let f = P016Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("P016 valid");
  assert_eq!(f.bits(), 16);
}

#[test]
fn p016_try_new_checked_is_a_noop() {
  // At BITS == 16 there are zero "low" bits to check — every u16
  // value is a valid P016 sample because `16 - BITS == 0`. The
  // checked constructor therefore accepts everything. This pins
  // that behavior in a test: at 16 bits the semantic distinction
  // between P016 and yuv420p16le **cannot be detected** from
  // sample values at all (no bit pattern is packing-specific).
  let y = std::vec![0x1234u16; 16 * 8];
  let uv = std::vec![0x5678u16; 16 * 4];
  P016Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("every u16 passes the low-bits check at BITS == 16");
}

#[test]
fn pn_try_new_rejects_bits_other_than_10_12_16() {
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0u16; 16 * 4];
  let e14 = PnFrame::<14>::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e14, PnFrameError::UnsupportedBits { bits: 14 }));
  let e11 = PnFrame::<11>::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e11, PnFrameError::UnsupportedBits { bits: 11 }));
}

#[test]
#[should_panic(expected = "invalid Yuv420pFrame16")]
fn yuv420p10_new_panics_on_invalid() {
  let y = std::vec![0u16; 10];
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let _ = Yuv420p10Frame::new(&y, &u, &v, 16, 8, 16, 8, 8);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn yuv420p10_try_new_rejects_geometry_overflow() {
  // Sample count overflow on 32-bit. Same rationale as the 8-bit
  // version — strides are in `u16` elements here, so the same
  // `0x1_0000 * 0x1_0000` product overflows `usize`.
  let big: u32 = 0x1_0000;
  let y: [u16; 0] = [];
  let u: [u16; 0] = [];
  let v: [u16; 0] = [];
  let e = Yuv420p10Frame::try_new(&y, &u, &v, big, big, big, big / 2, big / 2).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::GeometryOverflow { .. }));
}

#[test]
fn yuv420p10_try_new_checked_accepts_in_range_samples() {
  // Same valid frame as `yuv420p10_try_new_accepts_valid_tight`,
  // but run through the checked constructor. All samples live in
  // the 10‑bit range.
  let (y, u, v) = p10_planes();
  let f = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv420p10_try_new_checked_rejects_y_high_bit_set() {
  // A Y sample with bit 15 set — typical of `p010` packing where
  // the 10 active bits sit in the high bits. `try_new` would
  // accept this and let the SIMD kernels produce arch‑dependent
  // garbage; `try_new_checked` catches it up front.
  let mut y = std::vec![0u16; 16 * 8];
  y[3 * 16 + 5] = 0x8000; // bit 15 set → way above 1023
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  match e {
    Yuv420pFrame16Error::SampleOutOfRange {
      plane,
      value,
      max_valid,
      ..
    } => {
      assert_eq!(plane, Yuv420pFrame16Plane::Y);
      assert_eq!(value, 0x8000);
      assert_eq!(max_valid, 1023);
    }
    other => panic!("expected SampleOutOfRange, got {other:?}"),
  }
}

#[test]
fn yuv420p10_try_new_checked_rejects_u_plane_sample() {
  // Offending sample in the U plane — error must name U, not Y or V.
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 8 * 4];
  u[2 * 8 + 3] = 1024; // just above max
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::U,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv420p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 8 * 4];
  let mut v = std::vec![512u16; 8 * 4];
  v[8 + 7] = 0xFFFF; // all bits set
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv420p10_try_new_checked_accepts_exact_max_sample() {
  // Boundary: sample value == (1 << BITS) - 1 is valid.
  let mut y = std::vec![0u16; 16 * 8];
  y[0] = 1023;
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("1023 is in range");
}

#[test]
fn yuv420p10_try_new_checked_reports_geometry_errors_first() {
  // If geometry is invalid, we never get to the sample scan — the
  // same errors as `try_new` surface first. Prevents the checked
  // path from doing unnecessary O(N) work on inputs that would
  // fail for a simpler reason.
  let y = std::vec![0u16; 10]; // Too small.
  let u = std::vec![512u16; 8 * 4];
  let v = std::vec![512u16; 8 * 4];
  let e = Yuv420p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrame16Error::YPlaneTooShort { .. }));
}

// ---- P010Frame ---------------------------------------------------------
//
// Semi‑planar 10‑bit. Plane shape mirrors Nv12Frame (Y + interleaved
// UV) but sample width is `u16` with the 10 active bits in the
// **high** 10 of each element (`value << 6`). Strides are in
// samples, not bytes.

fn p010_planes() -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 16×8 frame — UV plane carries 16 u16 × 4 chroma rows = 64 u16.
  // P010 white Y = 1023 << 6 = 0xFFC0; neutral UV = 512 << 6 = 0x8000.
  (std::vec![0xFFC0u16; 16 * 8], std::vec![0x8000u16; 16 * 4])
}

#[test]
fn p010_try_new_accepts_valid_tight() {
  let (y, uv) = p010_planes();
  let f = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.uv_stride(), 16);
}

#[test]
fn p010_try_new_accepts_odd_height() {
  // 640×481 — same concrete odd‑height case covered by NV12 / NV21.
  let y = std::vec![0u16; 640 * 481];
  let uv = std::vec![0x8000u16; 640 * 241];
  let f = P010Frame::try_new(&y, &uv, 640, 481, 640, 640).expect("odd height valid");
  assert_eq!(f.height(), 481);
}

#[test]
fn p010_try_new_rejects_odd_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 15, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::OddWidth { width: 15 }));
}

#[test]
fn p010_try_new_rejects_zero_dim() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 0, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::ZeroDimension { .. }));
}

#[test]
fn p010_try_new_rejects_y_stride_under_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YStrideTooSmall { .. }));
}

#[test]
fn p010_try_new_rejects_uv_stride_under_width() {
  let (y, uv) = p010_planes();
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 8).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideTooSmall { .. }));
}

#[test]
fn p010_try_new_rejects_odd_uv_stride() {
  // uv_stride = 17 passes the size check (>= width = 16) but is
  // odd, which would mis-align the (U, V) pair on every other row.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 17 * 4];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 17).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 17 }));
}

#[test]
fn p210_try_new_rejects_odd_uv_stride() {
  // PnFrame422 chroma is half-width × full-height with 2 u16 per
  // pair → uv_row_elems = width. Same odd-stride bug as P010.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 17 * 8];
  let e = P210Frame::try_new(&y, &uv, 16, 8, 16, 17).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 17 }));
}

#[test]
fn p410_try_new_rejects_odd_uv_stride() {
  // PnFrame444 chroma is full-width × full-height with 2 u16 per
  // pair → uv_row_elems = 2 * width = 32. uv_stride = 33 passes
  // the size check but is odd.
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 33 * 8];
  let e = P410Frame::try_new(&y, &uv, 16, 8, 16, 33).unwrap_err();
  assert!(matches!(e, PnFrameError::UvStrideOdd { uv_stride: 33 }));
}

#[test]
fn p010_try_new_rejects_short_y_plane() {
  let y = std::vec![0u16; 10];
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YPlaneTooShort { .. }));
}

#[test]
fn p010_try_new_rejects_short_uv_plane() {
  let y = std::vec![0u16; 16 * 8];
  let uv = std::vec![0x8000u16; 8];
  let e = P010Frame::try_new(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::UvPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid PnFrame")]
fn p010_new_panics_on_invalid() {
  let y = std::vec![0u16; 10];
  let uv = std::vec![0x8000u16; 16 * 4];
  let _ = P010Frame::new(&y, &uv, 16, 8, 16, 16);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn p010_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u16; 0] = [];
  let uv: [u16; 0] = [];
  let e = P010Frame::try_new(&y, &uv, big, big, big, big).unwrap_err();
  assert!(matches!(e, PnFrameError::GeometryOverflow { .. }));
}

#[test]
fn p010_try_new_checked_accepts_shifted_samples() {
  // Valid P010 samples: low 6 bits zero.
  let (y, uv) = p010_planes();
  P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).expect("shifted samples valid");
}

#[test]
fn p010_try_new_checked_rejects_y_low_bits_set() {
  // A Y sample with low 6 bits set — characteristic of yuv420p10le
  // packing (value in low 10 bits) accidentally handed to the P010
  // constructor. `try_new_checked` catches this; plain `try_new`
  // would let the kernel mask it down and produce wrong colors.
  let mut y = std::vec![0xFFC0u16; 16 * 8];
  y[3 * 16 + 5] = 0x03FF; // 10-bit value in low bits — wrong packing
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  match e {
    PnFrameError::SampleLowBitsSet { plane, value, .. } => {
      assert_eq!(plane, P010FramePlane::Y);
      assert_eq!(value, 0x03FF);
    }
    other => panic!("expected SampleLowBitsSet, got {other:?}"),
  }
}

#[test]
fn p010_try_new_checked_rejects_uv_plane_sample() {
  let y = std::vec![0xFFC0u16; 16 * 8];
  let mut uv = std::vec![0x8000u16; 16 * 4];
  uv[2 * 16 + 3] = 0x0001; // low bit set
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    PnFrameError::SampleLowBitsSet {
      plane: P010FramePlane::Uv,
      value: 0x0001,
      ..
    }
  ));
}

#[test]
fn p010_try_new_checked_reports_geometry_errors_first() {
  let y = std::vec![0u16; 10]; // Too small.
  let uv = std::vec![0x8000u16; 16 * 4];
  let e = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(e, PnFrameError::YPlaneTooShort { .. }));
}

/// Regression documenting a **known limitation** of
/// [`P010Frame::try_new_checked`]: the low‑6‑bits‑zero check is a
/// packing sanity check, not a provenance validator. A
/// `yuv420p10le` buffer whose samples all happen to be multiples
/// of 64 — e.g. `Y = 64` (limited‑range black, `0x0040`) and
/// `UV = 512` (neutral chroma, `0x0200`) — passes the check
/// silently, even though the layout is wrong and downstream P010
/// kernels will produce incorrect output.
///
/// The test asserts the check accepts these values so the limit
/// is visible in the test log; any future attempt to tighten the
/// constructor into a real provenance validator will need to
/// update or replace this test.
#[test]
fn p010_try_new_checked_accepts_ambiguous_yuv420p10le_samples() {
  // `yuv420p10le`-style samples, all multiples of 64: low 6 bits
  // are zero, so they pass the P010 sanity check even though this
  // is wrong data for a P010 frame.
  let y = std::vec![0x0040u16; 16 * 8]; // limited-range black in 10-bit low-packed
  let uv = std::vec![0x0200u16; 16 * 4]; // neutral chroma in 10-bit low-packed
  let f = P010Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("known limitation: low-6-bits-zero check cannot tell yuv420p10le from P010");
  assert_eq!(f.width(), 16);
  // Downstream decoding of this frame would produce wrong colors
  // (every `>> 6` extracts 1 from Y=0x0040 and 8 from UV=0x0200,
  // which P010 kernels then bias/scale as if those were the 10-bit
  // source values). That's accepted behavior — the type system,
  // not `try_new_checked`, is what keeps yuv420p10le out of P010.
}

#[test]
fn p012_try_new_checked_accepts_shifted_samples() {
  // Valid P012 samples: low 4 bits zero (12-bit value << 4).
  let y = std::vec![(2048u16) << 4; 16 * 8]; // 12-bit mid-gray shifted up
  let uv = std::vec![(2048u16) << 4; 16 * 4];
  P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).expect("shifted samples valid");
}

#[test]
fn p012_try_new_checked_rejects_low_bits_set() {
  // A Y sample with any of the low 4 bits set — e.g. yuv420p12le
  // value 0x0ABC landing where P012 expects `value << 4`. The check
  // catches samples like this that are obviously mispacked.
  let mut y = std::vec![(2048u16) << 4; 16 * 8];
  y[3 * 16 + 5] = 0x0ABC; // low 4 bits = 0xC ≠ 0
  let uv = std::vec![(2048u16) << 4; 16 * 4];
  let e = P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  match e {
    PnFrameError::SampleLowBitsSet {
      plane,
      value,
      low_bits,
      ..
    } => {
      assert_eq!(plane, PnFramePlane::Y);
      assert_eq!(value, 0x0ABC);
      assert_eq!(low_bits, 4);
    }
    other => panic!("expected SampleLowBitsSet, got {other:?}"),
  }
}

/// Regression documenting a **worse known limitation** of
/// [`P012Frame::try_new_checked`] compared to P010: because the
/// low‑bits check only has 4 bits to work with at `BITS == 12`,
/// every multiple‑of‑16 `yuv420p12le` value passes silently. The
/// practical impact is that common limited‑range flat‑region
/// content in real decoder output — `Y = 256` (limited‑range
/// black), `UV = 2048` (neutral chroma), `Y = 1024` (full black)
/// — is entirely invisible to this check.
///
/// This test pins the limitation with a reproducible input so
/// that:
/// 1. Users reading the test suite can see the exact failure
///    mode for `try_new_checked` on 12‑bit data.
/// 2. Any future attempt to strengthen `try_new_checked` (e.g.,
///    into a statistical provenance heuristic) has a concrete
///    input to validate against.
/// 3. The `PnFrame` docs' warning about this limitation has a
///    named test to point to.
///
/// For P012, the type system (choosing [`P012Frame`] vs
/// [`Yuv420p12Frame`] at construction based on decoder metadata)
/// is the only reliable provenance guarantee.
#[test]
fn p012_try_new_checked_accepts_low_packed_flat_content_by_design() {
  // All values are multiples of 16 — exactly the set that slips
  // through a 4-low-bits-zero check. `yuv420p12le` limited-range
  // black and neutral chroma both satisfy this.
  let y = std::vec![0x0100u16; 16 * 8]; // Y = 256 (limited-range black), multiple of 16
  let uv = std::vec![0x0800u16; 16 * 4]; // UV = 2048 (neutral chroma), multiple of 16
  let f = P012Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("known limitation: 4-low-bits-zero check cannot tell yuv420p12le from P012");
  assert_eq!(f.width(), 16);
  // Downstream P012 kernels would extract `>> 4` — giving Y=16 and
  // UV=128 instead of the intended Y=256 and UV=2048. Silent color
  // corruption. The type system, not `try_new_checked`, must
  // guarantee provenance for 12-bit.
}

