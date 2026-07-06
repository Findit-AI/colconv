//! Tests for `crate::row::scalar::xyz12`.

use super::*;

/// Tolerance for comparing f32 values derived via the same algorithm
/// the kernel implements. Round-trip through `f32::powf` is platform
/// stable but the f64 -> f32 narrow at our derived-fixture step
/// introduces a ~4e-7 noise floor, so `4e-6` is comfortably above
/// any platform variation.
const EPSILON_F32: f32 = 4e-6;

fn assert_close(a: f32, b: f32, tag: &str) {
  let diff = (a - b).abs();
  assert!(diff <= EPSILON_F32, "{tag}: {a} vs {b} (diff {diff})");
}

// ---- OETF / inverse-OETF spot checks ----

#[test]
fn smpte428_inverse_oetf_zero_is_zero() {
  assert_eq!(smpte428_inverse_oetf(0), 0.0);
}

#[test]
fn smpte428_inverse_oetf_max_is_normalised() {
  // (4095/4095)^2.6 / 0.91653 = 1.0 / 0.91653 ≈ 1.0911
  let actual = smpte428_inverse_oetf(4095);
  assert!((actual - 1.0_f32 / 0.91653_f32).abs() < EPSILON_F32);
}

#[test]
#[cfg_attr(miri, ignore = "f32::powf is non-deterministic under Miri")]
fn smpte428_inverse_oetf_masks_upper_bits() {
  // Defensive: `smpte428_inverse_oetf` masks its argument with
  // `SAMPLE_MASK`, so passing a non-shifted dirty value still
  // produces the same result as the clean 12-bit code. Callers
  // (`read_xyz12_sample` / SIMD shift path) provide the shifted
  // value directly.
  let clean = smpte428_inverse_oetf(0x0800);
  let dirty = smpte428_inverse_oetf(0xF800);
  assert_eq!(clean, dirty);
}

#[test]
fn read_xyz12_sample_extracts_high_bit_packed_code_le() {
  // FFmpeg `AV_PIX_FMT_XYZ12LE`: 12-bit code in `[15:4]`, low 4 bits
  // zero. Mid-gray sample on the 16-bit scale is `0x8000`, which
  // decodes as `0x800` (mid-gray on the 12-bit scale = 2048).
  assert_eq!(read_xyz12_sample::<false>(pack12_le(0x800)), 0x800);
  assert_eq!(read_xyz12_sample::<false>(pack12_le(0xFFF)), 0x0FFF);
  assert_eq!(read_xyz12_sample::<false>(pack12_le(0x000)), 0x0000);
}

#[test]
fn read_xyz12_sample_extracts_high_bit_packed_code_be() {
  // BE wire: bytes stored big-endian on disk; `pack12_be` produces a
  // host-native u16 whose bytes match BE encoding on every host.
  assert_eq!(read_xyz12_sample::<true>(pack12_be(0x800)), 0x800);
}

#[test]
fn smpte428_mid_gray_high_bit_packed_is_nonzero() {
  // Pre-fix regression: `0x8000` (real mid-gray sample) was decoded
  // as `0x000` because `read_xyz12_sample` masked the *low* 12 bits
  // instead of the high-bit-packed payload. Post-fix it is `0x800`
  // and produces the mid-gray linear-XYZ value.
  let mid_gray = read_xyz12_sample::<false>(pack12_le(0x800));
  assert_eq!(mid_gray, 0x800);
  let xyz_lin = smpte428_inverse_oetf(mid_gray);
  assert!(xyz_lin > 0.1, "expected mid-gray > 0.1, got {xyz_lin}");
}

#[test]
fn oetf_srgb_zero_is_zero() {
  assert_eq!(oetf_srgb(0.0), 0.0);
}

#[test]
fn oetf_srgb_uses_linear_below_threshold() {
  let c = 0.001_f32;
  let expected = 12.92_f32 * c;
  assert_eq!(oetf_srgb(c), expected);
}

#[test]
fn oetf_srgb_one_is_one() {
  let v = oetf_srgb(1.0);
  assert!((v - 1.0).abs() < EPSILON_F32);
}

#[test]
fn oetf_srgb_continuous_at_threshold() {
  let lo = oetf_srgb(0.0031307);
  let hi = oetf_srgb(0.0031309);
  // Should be close — function is continuous at the segment boundary.
  assert!((hi - lo).abs() < 1e-5);
}

// ---- Polynomial-OETF parity vs f64-narrowed reference oracle ----

/// f64-narrowed sRGB OETF reference oracle — every operation in
/// f64, narrowed once at the end. Matches the form chased by the
/// derivation tool (`examples/derive_oetf_polynomial.rs`) and is
/// the canonical reference both `oetf_srgb` (polynomial) and SIMD
/// backends must come within ≤ 2 ULP of (B' decision).
///
/// Lives in the test module only — production code uses the
/// polynomial via `oetf_srgb` directly, which is closer to truth
/// than `f32::powf` *and* avoids per-pixel `powf` cost.
fn oetf_srgb_reference_f64(c: f32) -> f32 {
  if c < 0.0031308_f32 {
    12.92_f32 * c
  } else {
    let c64 = c as f64;
    (1.055_f64 * powf64(c64, 1.0_f64 / 2.4_f64) - 0.055_f64) as f32
  }
}

/// IEEE-754 monotone bijection: returns `i64` such that
/// `(f32_to_sortable(a) - f32_to_sortable(b)).abs()` equals the
/// f32-ULP distance between `a` and `b` (signed-aware so that
/// crossing zero / sign change still gives a positive ULP count).
fn f32_to_sortable(x: f32) -> i64 {
  let bits = x.to_bits() as i32;
  if bits >= 0 {
    bits as i64
  } else {
    (i32::MIN as i64) - (bits as i64)
  }
}

fn f32_ulps(a: f32, b: f32) -> u64 {
  let a_b = f32_to_sortable(a);
  let b_b = f32_to_sortable(b);
  a_b.abs_diff(b_b)
}

#[test]
fn oetf_srgb_below_threshold_uses_linear() {
  // Same exact 12.92 * c as the reference oracle (linear toe is
  // bit-exact between polynomial and reference).
  let c = 0.001_f32;
  assert_eq!(oetf_srgb(c), 12.92_f32 * c);
  assert_eq!(oetf_srgb(c), oetf_srgb_reference_f64(c));
}

#[test]
fn oetf_srgb_polynomial_within_2_ulp_of_reference() {
  // 65 536 sample points across [0.0031308, 1.0]. Asserts the
  // production polynomial OETF (`oetf_srgb`) is within ≤ 2 ULP of
  // the f64-narrowed reference oracle everywhere — the SIMD backends
  // rely on this contract when they vectorize the same polynomial
  // via Horner across f32 lanes (giving 0-ULP scalar↔SIMD parity).
  const SAMPLES: usize = 65_536;
  let lo = 0.003_130_8_f64;
  let hi = 1.0_f64;
  let mut max_ulp = 0_u64;
  let mut max_x = 0.0_f32;
  for i in 0..SAMPLES {
    let t = (i as f64) / ((SAMPLES - 1) as f64);
    let c = (lo + (hi - lo) * t) as f32;
    let poly = oetf_srgb(c);
    let reference = oetf_srgb_reference_f64(c);
    let u = f32_ulps(poly, reference);
    if u > max_ulp {
      max_ulp = u;
      max_x = c;
    }
  }
  assert!(
    max_ulp <= 2,
    "polynomial OETF exceeded 2 ULP vs f64-narrowed reference: max = {} at x = {}",
    max_ulp,
    max_x,
  );
}

#[test]
fn oetf_srgb_at_segment_boundary_within_2_ulp() {
  // Just above the linear/upper-segment threshold — polynomial
  // matches f64-narrowed reference within ≤ 2 ULP.
  let c = 0.003_131_f32;
  let poly = oetf_srgb(c);
  let reference = oetf_srgb_reference_f64(c);
  assert!(
    f32_ulps(poly, reference) <= 2,
    "polynomial vs reference at boundary: {} vs {}",
    poly,
    reference,
  );
}

#[test]
fn narrow_unit_to_u8_round_half_up() {
  assert_eq!(narrow_unit_to_u8(0.0), 0);
  assert_eq!(narrow_unit_to_u8(1.0), 255);
  // 0.5 / 255 = 0.00196… → narrow_unit_to_u8(0.00196) ≈ 1.
  assert_eq!(narrow_unit_to_u8(0.5_f32), 128);
  assert_eq!(narrow_unit_to_u8(-1.0), 0);
  assert_eq!(narrow_unit_to_u8(2.0), 255);
}

#[test]
fn narrow_unit_to_u16_round_half_up() {
  assert_eq!(narrow_unit_to_u16(0.0), 0);
  assert_eq!(narrow_unit_to_u16(1.0), 65535);
  assert_eq!(narrow_unit_to_u16(-1.0), 0);
  assert_eq!(narrow_unit_to_u16(2.0), 65535);
}

// ---- Derived-fixture parity tests (per gamut) ----
//
// Expected values produced by `examples/derive_xyz_matrices.rs` (run
// 2026-05-08). Hardcoded as f32 literals below; the same algorithm
// is implemented in the kernel, so this test is a regression lock
// against accidental drift in the per-pixel math.

#[test]
fn xyz12_to_rgb_f32_rec709_zero_input() {
  let xyz = [0_u16; 3];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert_eq!(out, [0.0; 3]);
}

/// Encodes a 12-bit code in the high-bit-packed wire layout
/// (`code << 4`) for FFmpeg `AV_PIX_FMT_XYZ12LE` fixtures.
/// `(code << 4).to_le_bytes()` reinterpreted as a host-native `u16`
/// produces a value whose **byte storage** is LE-encoded on every
/// host. The `<BE = false>` kernel applies `u16::from_le` internally
/// to recover the intended logical sample (no-op on LE; byte-swap on
/// BE). Mirrors `pack12_be` below — host-independent by construction.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_le(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_le_bytes())
}

/// Encodes a 12-bit code in the high-bit-packed BE wire layout —
/// `(code << 4).to_be_bytes()` reinterpreted as a host-native `u16`.
/// Host-independent: produces the same logical wire value on LE and
/// BE hosts (the `<BE = true>` kernel applies `from_be` internally).
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_be(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_be_bytes())
}

/// LE wire fixture variant of `pack12_le` that also stuffs the low
/// 4 wire bits with `low_bits` (`0..=0xF`). The reserved-low-bits
/// invariant is set on the **logical** wire value before LE byte
/// re-encoding, so the dirty bits land at the LE low byte's low
/// nibble on every host. ORing `low_bits` *after* `pack12_le` would
/// only work on LE: on BE, `pack12_le(code)` returns the byte-swapped
/// host word, so the OR clobbers the high byte instead.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_le_dirty(code: u16, low_bits: u16) -> u16 {
  u16::from_ne_bytes(((code << 4) | (low_bits & 0xF)).to_le_bytes())
}

#[test]
fn xyz12_to_rgb_f32_dci_p3_mid_gray() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  // DCI-P3 (theatrical, DCI white) expected at u12 mid-gray:
  //   linear XYZ = ((0x800 / 4095)^2.6) / 0.91653 ≈ 0.158 (per channel)
  //   RGB_lin    = M_DCI_P3 · (0.158, 0.158, 0.158)
  //              ≈ (0.228194803, 0.165165901, 0.189893857)
  // Generated by `examples/derive_xyz_matrices.rs` using DCI white
  // (NOT D65 — D65 would give the P3-D65 values (0.2088, 0.1723, 0.1650)).
  assert_close(out[0], 0.228_194_8, "R");
  assert_close(out[1], 0.165_165_9, "G");
  assert_close(out[2], 0.189_893_85, "B");
}

#[test]
fn xyz12_to_rgb_f32_rec709_mid_gray() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert_close(out[0], 0.216_984_87, "R");
  assert_close(out[1], 0.170_760_4, "G");
  assert_close(out[2], 0.163_619_68, "B");
}

#[test]
fn xyz12_to_rgb_f32_rec2020_three_quarter() {
  let xyz: [u16; 3] = [pack12_le(0xC00), pack12_le(0xC00), pack12_le(0xC00)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec2020);
  assert_close(out[0], 0.572_369_93, "R");
  assert_close(out[1], 0.498_964_94, "G");
  assert_close(out[2], 0.473_854_f32, "B");
}

#[test]
fn xyz12_to_rgb_f32_preserves_negative_after_matrix() {
  // y_only_max under Rec.709 → R = -1.677, G = +2.05, B = -0.222.
  let xyz: [u16; 3] = [pack12_le(0), pack12_le(0xFFF), pack12_le(0)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert!(out[0] < 0.0, "expected negative R, got {}", out[0]);
  assert!(out[2] < 0.0, "expected negative B, got {}", out[2]);
  assert_close(out[0], -1.677_395_3, "R");
  assert_close(out[1], 2.046_815_2, "G");
  assert_close(out[2], -0.222_553_5, "B");
}

#[test]
fn xyz12_to_rgb_clamps_at_u8() {
  // x_only_max under Rec.709 → R = +3.5, G = -1.0 → after OETF +
  // clamp + x255 → R = 255, G = 0.
  let xyz: [u16; 3] = [pack12_le(0xFFF), pack12_le(0), pack12_le(0)];
  let mut out = [0_u8; 3];
  xyz12_to_rgb_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert_eq!(out[0], 255);
  assert_eq!(out[1], 0);
}

#[test]
fn xyz12_to_rgba_fills_alpha_max() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0_u8; 4];
  xyz12_to_rgba_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_eq!(out[3], 0xFF);
}

#[test]
fn xyz12_to_rgba_u16_fills_alpha_max() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0_u16; 4];
  xyz12_to_rgba_u16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_eq!(out[3], 0xFFFF);
}

#[test]
fn xyz12_to_xyz_f32_lossless_round_trip() {
  // Pass-through: input -> step-1 inverse-OETF -> output. For u12 =
  // (0x800, 0x800, 0x800) the linear value is the same in all three
  // channels.
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_xyz_f32_row::<false>(&xyz, &mut out, 1);
  let expected = powf32(0x800_u16 as f32 * INV_4095, 2.6_f32) * SMPTE428_INV_NORM;
  assert_close(out[0], expected, "X");
  assert_close(out[1], expected, "Y");
  assert_close(out[2], expected, "Z");
}

#[test]
#[cfg_attr(miri, ignore = "f32::powf is non-deterministic under Miri")]
fn xyz12_be_byte_swap_matches_le() {
  // Same logical 12-bit code (`0x800`) encoded with both LE and BE
  // wire conventions; both kernel paths must produce identical
  // output. Host-independent: `pack12_be` builds the BE bytes and
  // reinterprets via `from_ne_bytes`, so the test runs on BE hosts
  // (s390x miri) without further `cfg` gates.
  let xyz_le: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let xyz_be: [u16; 3] = [pack12_be(0x800), pack12_be(0x800), pack12_be(0x800)];
  let mut out_le = [0.0_f32; 3];
  let mut out_be = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz_le, &mut out_le, 1, DcpTargetGamut::DciP3);
  xyz12_to_rgb_f32_row::<true>(&xyz_be, &mut out_be, 1, DcpTargetGamut::DciP3);
  assert_eq!(out_le, out_be);
}

#[test]
fn xyz12_to_rgb_u16_full_range_scaling() {
  let xyz: [u16; 3] = [pack12_le(0xFFF), pack12_le(0xFFF), pack12_le(0xFFF)];
  let mut out = [0_u16; 3];
  xyz12_to_rgb_u16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  // Per derivation: rgb_linear = (1.265, 1.044, 1.0) → after OETF +
  // clamp [0,1] x 65535 → (65535, 65535, 65535).
  assert_eq!(out[0], 65535);
  assert_eq!(out[1], 65535);
  assert_eq!(out[2], 65535);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn xyz12_to_rgb_f16_clamps_to_unit_range() {
  let xyz: [u16; 3] = [pack12_le(0xFFF), pack12_le(0), pack12_le(0)];
  let mut out = [half::f16::from_f32(0.0); 3];
  xyz12_to_rgb_f16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert_eq!(out[0].to_f32(), 1.0);
  assert_eq!(out[1].to_f32(), 0.0);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn xyz12_to_rgba_f16_alpha_one() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [half::f16::from_f32(0.0); 4];
  xyz12_to_rgba_f16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_eq!(out[3].to_f32(), 1.0);
}

#[test]
fn xyz12_to_rgb_target_gamut_changes_output() {
  let xyz: [u16; 3] = [pack12_le(0xC00), pack12_le(0xC00), pack12_le(0xC00)];
  let mut out_p3 = [0.0_f32; 3];
  let mut out_709 = [0.0_f32; 3];
  let mut out_2020 = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_p3, 1, DcpTargetGamut::DciP3);
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_709, 1, DcpTargetGamut::Rec709);
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_2020, 1, DcpTargetGamut::Rec2020);
  // All three should differ on R (different matrix scales).
  assert!(
    (out_p3[0] - out_709[0]).abs() > 1e-3,
    "DCI-P3 vs Rec.709 R: {} vs {}",
    out_p3[0],
    out_709[0],
  );
  assert!(
    (out_p3[0] - out_2020[0]).abs() > 1e-3,
    "DCI-P3 vs Rec.2020 R: {} vs {}",
    out_p3[0],
    out_2020[0],
  );
}

#[test]
fn xyz12_to_rgb_low_4_bits_ignored() {
  // FFmpeg spec: low 4 bits of each `u16` are zero. A producer that
  // sets them anyway must not change the output (the `>> 4` shift
  // discards them before the OETF).
  let xyz_clean: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let xyz_dirty: [u16; 3] = [
    pack12_le_dirty(0x800, 0xF),
    pack12_le_dirty(0x800, 0xA),
    pack12_le_dirty(0x800, 0x7),
  ];
  let mut out_clean = [0_u8; 3];
  let mut out_dirty = [0_u8; 3];
  xyz12_to_rgb_row::<false>(&xyz_clean, &mut out_clean, 1, DcpTargetGamut::DciP3);
  xyz12_to_rgb_row::<false>(&xyz_dirty, &mut out_dirty, 1, DcpTargetGamut::DciP3);
  assert_eq!(out_clean, out_dirty);
}

#[test]
fn xyz12_to_rgb_multi_pixel_independence() {
  let xyz: [u16; 6] = [
    pack12_le(0x800),
    pack12_le(0x800),
    pack12_le(0x800), // pixel 0
    pack12_le(0xFFF),
    pack12_le(0),
    pack12_le(0), // pixel 1
  ];
  let mut out = [0_u8; 6];
  xyz12_to_rgb_row::<false>(&xyz, &mut out, 2, DcpTargetGamut::Rec709);

  let mut single = [0_u8; 3];
  xyz12_to_rgb_row::<false>(&xyz[..3], &mut single, 1, DcpTargetGamut::Rec709);
  assert_eq!(&out[..3], &single);

  let mut single1 = [0_u8; 3];
  xyz12_to_rgb_row::<false>(&xyz[3..], &mut single1, 1, DcpTargetGamut::Rec709);
  assert_eq!(&out[3..], &single1);
}

// Independent reference-vector tests.
//
// These tests do NOT compare scalar↔SIMD parity (already covered
// upstream). Instead, they pin colorimetric correctness against
// expected values computed from independent authoritative references:
//
// - XYZ → RGB matrices: closed-form primary-scaling derivation in
//   f64 (matches `examples/derive_xyz_matrices.rs`) with DCI white
//   `(0.314, 0.351)` for the `DciP3` target — NOT D65.
// - Luma weights: Y row of each gamut's RGB→XYZ matrix, normalised
//   so a unit RGB triple maps to the gamut's white-point Y.
//
// Inputs are canonical points (zero / peak / mid-gray) and per-axis
// unit XYZ stimuli that are easy to audit by inspection. Tolerance
// is `EPSILON_F32 = 4e-6` to absorb the f64→f32 narrow at the kernel
// boundary.
/// DCI-P3 (theatrical, DCI white) zero-input: SMPTE ST 428 inverse-
/// OETF of `0` is `0`, and `M · (0, 0, 0) = (0, 0, 0)` for every
/// matrix. This locks the inverse-OETF math AND the matrix's zero
/// handling.
#[test]
fn xyz12_dci_p3_zero_input_zero_output_reference() {
  let xyz: [u16; 3] = [pack12_le(0), pack12_le(0), pack12_le(0)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_eq!(out, [0.0, 0.0, 0.0]);
}

/// DCI-P3 mid-gray reference vector. Independently-computed
/// expected RGB:
///
/// 1. inverse-OETF: `(0x800 / 4095)^2.6 / 0.91653 = 0.158064...`
///    (per channel, since X=Y=Z).
/// 2. matrix · `(0.158064, 0.158064, 0.158064)` =
///    `(0.228195, 0.165166, 0.189894)` to 6 decimals
///    (DCI-white-pointed, per the canonical primary-scaling
///    derivation; see `examples/derive_xyz_matrices.rs` 2026-05-09).
///
/// Pre-fix (D65, "Display-P3") this would produce
/// `(0.217, 0.171, 0.164)` — different by ~0.011 / 0.006 / -0.026
/// per channel, well above any rounding tolerance, so this test
/// independently catches a regression to the wrong white point.
#[test]
fn xyz12_dci_p3_mid_gray_reference_dci_white() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_close(out[0], 0.228_194_8, "DciP3 mid-gray R");
  assert_close(out[1], 0.165_165_9, "DciP3 mid-gray G");
  assert_close(out[2], 0.189_893_85, "DciP3 mid-gray B");
  // Sanity: R, B both clearly distinct from BT.709 / Rec.2020 mid-
  // gray for the same input — guards against silent fall-through to
  // the wrong matrix at the dispatch site.
  assert!(
    (out[0] - 0.216_984_87_f32).abs() > 1e-3,
    "must differ from Rec.709 R"
  );
  assert!(
    (out[2] - 0.163_619_68_f32).abs() > 1e-3,
    "must differ from Rec.709 B"
  );
}

/// DCI-P3 peak white reference. Inverse-OETF of `0xFFF` is `1.0 /
/// 0.91653 ≈ 1.09109`. Multiplied by the M_DCI_P3 row sums:
///   row 0 sum = 2.7253940 - 1.0180030 - 0.4401632 = 1.2672278
///   → R = 1.0911 · 1.2672278 ≈ 1.382637
///   row 1 sum = -0.7951680 + 1.6897321 + 0.0226472 = 0.9172113
///   → G = 1.0911 · 0.9172113 ≈ 1.000743
///   row 2 sum =  0.0412419 - 0.0876390 + 1.1009294 = 1.0545323
///   → B = 1.0911 · 1.0545323 ≈ 1.150570
///
/// All three are slightly > 1 (peak white in DCI XYZ exceeds DCI-P3
/// gamut white-point luminance because the SMPTE 428 normalisation
/// is per-channel, not per-luminance) — the f32 path preserves this
/// excursion losslessly; the u8 / u16 paths clamp.
#[test]
fn xyz12_dci_p3_peak_white_reference() {
  let xyz: [u16; 3] = [pack12_le(0xFFF), pack12_le(0xFFF), pack12_le(0xFFF)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_close(out[0], 1.382_636_5, "DciP3 peak R");
  assert_close(out[1], 1.000_743_3, "DciP3 peak G");
  assert_close(out[2], 1.150_570_4, "DciP3 peak B");
}

/// DCI-P3 axis-isolation reference: input `(0xFFF, 0, 0)` activates
/// only the matrix's first column. Expected RGB is therefore exactly
/// `inverse_oetf(0xFFF) · M_DCI_P3[:, 0]` =
///   `1.09109 · (2.7253940, -0.7951680, 0.0412419)`
/// = `(2.973600, -0.867585, 0.044998)`.
///
/// The negative G value is a real out-of-DCI-P3-gamut excursion (X
/// alone has no green component at all in the matrix); the f32 path
/// preserves it. This test independently catches column ordering
/// regressions.
#[test]
fn xyz12_dci_p3_x_only_axis_reference() {
  let xyz: [u16; 3] = [pack12_le(0xFFF), pack12_le(0), pack12_le(0)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
  assert_close(out[0], 2.973_600_4, "DciP3 X-only R");
  assert_close(out[1], -0.867_585_36, "DciP3 X-only G");
  assert_close(out[2], 0.044_997_863, "DciP3 X-only B");
  assert!(out[1] < 0.0, "X-only must produce negative G under DciP3");
}

/// Rec.709 mid-gray reference: independent verification that
/// non-DciP3 gamuts continue to use their own M_XYZ_TO_RGB matrix.
/// `(0.158064, 0.158064, 0.158064)` · M_REC709 =
/// `(0.216984, 0.170760, 0.163620)` (Rec.709 / sRGB white = D65).
#[test]
fn xyz12_rec709_mid_gray_reference() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
  assert_close(out[0], 0.216_984_87, "Rec.709 mid-gray R");
  assert_close(out[1], 0.170_760_4, "Rec.709 mid-gray G");
  assert_close(out[2], 0.163_619_68, "Rec.709 mid-gray B");
}

/// Rec.2020 mid-gray reference: same canonical mid-gray input,
/// expected RGB derived from Rec.2020's primary-scaling matrix.
#[test]
fn xyz12_rec2020_mid_gray_reference() {
  let xyz: [u16; 3] = [pack12_le(0x800), pack12_le(0x800), pack12_le(0x800)];
  let mut out = [0.0_f32; 3];
  xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec2020);
  assert_close(out[0], 0.199_452_52, "Rec.2020 mid-gray R");
  assert_close(out[1], 0.173_873_25, "Rec.2020 mid-gray G");
  assert_close(out[2], 0.165_122_9, "Rec.2020 mid-gray B");
}

// -- Luma reference-vector tests --

/// Pure-red u8 RGB `(255, 0, 0)` under DCI-P3 luma weights:
/// `Y = 0.2094917 · 255 ≈ 53.42` → round-half-up → `53`.
/// Under BT.709 weights: `Y = 0.2126 · 255 ≈ 54.21` → `54`.
/// This test independently confirms the P3-specific luma triple
/// is wired through.
#[test]
fn xyz12_rgb_to_luma_dci_p3_pure_red_reference() {
  let rgb: [u8; 3] = [255, 0, 0];
  let mut out_p3 = [0_u8; 1];
  let mut out_709 = [0_u8; 1];
  let p3 = (6865_i32, 23645_i32, 2258_i32); // DCI-P3 Q15.
  let bt709 = (6966_i32, 23436_i32, 2366_i32); // BT.709 Q15.
  xyz12_rgb_to_luma_row(&rgb, &mut out_p3, 1, p3);
  xyz12_rgb_to_luma_row(&rgb, &mut out_709, 1, bt709);
  // Q15 path: `(6865 · 255 + 0 + 0 + 16384) >> 15` = `(1750575 +
  // 16384) >> 15` = `1766959 >> 15` = `53`.
  assert_eq!(out_p3[0], 53, "DciP3 luma(255,0,0) must be 53");
  assert_eq!(out_709[0], 54, "Bt709 luma(255,0,0) must be 54 (control)");
  assert_ne!(
    out_p3[0], out_709[0],
    "DciP3 and Bt709 luma weights must differ for saturated red",
  );
}

/// Pure-blue u8 RGB `(0, 0, 255)` under DCI-P3 vs BT.709:
/// DCI-P3 `Y = 0.0689 · 255 ≈ 17.57` → `18`.
/// BT.709 `Y = 0.0722 · 255 ≈ 18.41` → `18`.
/// They round to the same integer at saturated blue, but the Q15
/// dot product still produces distinct values pre-narrow — this
/// test exercises a u16 luma channel where the difference is
/// preserved (zero-extended u8 path is identical, so we use a
/// known-distinct stimulus instead: pure green dominates).
#[test]
fn xyz12_rgb_to_luma_u16_dci_p3_pure_green_reference() {
  let rgb: [u8; 3] = [0, 255, 0];
  let mut out_p3 = [0_u16; 1];
  let mut out_709 = [0_u16; 1];
  let p3 = (6865_i32, 23645_i32, 2258_i32);
  let bt709 = (6966_i32, 23436_i32, 2366_i32);
  xyz12_rgb_to_luma_u16_row(&rgb, &mut out_p3, 1, p3);
  xyz12_rgb_to_luma_u16_row(&rgb, &mut out_709, 1, bt709);
  // DCI-P3: `(0 + 23645 · 255 + 0 + 16384) >> 15` = `184` u8.
  // BT.709: `(0 + 23436 · 255 + 0 + 16384) >> 15` = `182` u8.
  assert_eq!(out_p3[0], 184, "DciP3 luma_u16(0,255,0) must be 184");
  assert_eq!(
    out_709[0], 182,
    "Bt709 luma_u16(0,255,0) must be 182 (control)"
  );
  assert_ne!(
    out_p3[0], out_709[0],
    "DciP3 and Bt709 luma weights must differ for saturated green",
  );
}

/// Luma weights sum to 32768 (`1.0` in Q15) — round-half-up on a
/// uniform-gray RGB triple must reproduce the input grey exactly,
/// independent of the weight choice.
#[test]
fn xyz12_rgb_to_luma_uniform_gray_reproduces_input_all_gamuts() {
  let rgb: [u8; 3] = [128, 128, 128];
  let mut out = [0_u8; 1];
  for triple in [
    (6865_i32, 23645, 2258), // DciP3
    (6966_i32, 23436, 2366), // Bt709
    (8607_i32, 22217, 1944), // Bt2020Ncl
  ] {
    out[0] = 0;
    xyz12_rgb_to_luma_row(&rgb, &mut out, 1, triple);
    assert_eq!(
      out[0], 128,
      "uniform gray must reproduce input under {:?}",
      triple
    );
  }
}
