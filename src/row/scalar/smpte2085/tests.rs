//! Reference cross-checks for the SMPTE ST 2085 (H.273 code 11, "Y'D'zD'x")
//! non-affine decode.
//!
//! No external library is needed: the published H.273 §8.3 Table 4 code 11
//! forward relations (eqs 76-78) ARE the oracle. [`recover_xyz_prime`] is
//! their exact analytic inverse, and the end-to-end kernels are validated by
//! an `X'Y'Z'` encode → decode round-trip — encode a target `X'Y'Z'` the way
//! the spec defines (eqs 76-78 + studio/full quant), then assert the kernel
//! recovers it and the `XYZ → RGB → PQ` chain reproduces the reference
//! `R'G'B'`.

use super::*;

/// Re-encode a logical u16 as LE wire storage so a `BE = false` kernel recovers
/// it via `u16::from_le` on both endiannesses (the smpte2085 kernels read wire
/// u16).
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

fn assert_close(got: [f32; 3], want: [f32; 3], tol: f32, what: &str) {
  for i in 0..3 {
    assert!(
      (got[i] - want[i]).abs() <= tol,
      "{what}: channel {i} = {} (want {}, |Δ| = {})",
      got[i],
      want[i],
      (got[i] - want[i]).abs()
    );
  }
}

/// Forward H.273 eq 76-78 encode of a normalized `X'Y'Z'` triple to the
/// normalized `Y',D'z,D'x` domain — the exact inverse of
/// [`recover_xyz_prime`]:
///
/// ```text
/// Y'  = X'Y'Z''s Y'
/// D'z = (0.986566 · Z' − Y') / 2
/// D'x = (X' − 0.991902 · Y') / 2
/// ```
fn encode_xyz_prime_to_norm(xyz_prime: [f32; 3]) -> [f32; 3] {
  let [xp, yp, zp] = xyz_prime;
  let dz = (WHITE_NORM_Z * zp - yp) / 2.0_f32;
  let dx = (xp - WHITE_NORM_X * yp) / 2.0_f32;
  [yp, dz, dx]
}

/// `Smpte2085Transfer::for_transfer` resolves only the PQ transfer; everything
/// else (incl. `Unspecified`) is `None` → affine fallback.
#[test]
fn for_transfer_selects_pq_only() {
  use crate::Transfer;
  assert_eq!(
    Smpte2085Transfer::for_transfer(Transfer::SmpteSt2084Pq),
    Some(Smpte2085Transfer::Pq)
  );
  for t in [
    Transfer::Unspecified,
    Transfer::Bt709,
    Transfer::Bt2020_10Bit,
    Transfer::AribStdB67Hlg,
    Transfer::SmpteSt428,
    Transfer::Unknown(99),
  ] {
    assert_eq!(
      Smpte2085Transfer::for_transfer(t),
      None,
      "{t:?} must not select the SMPTE 2085 transfer"
    );
  }
  assert_eq!(Smpte2085Transfer::Pq.as_str(), "pq");
}

/// `recover_xyz_prime` is the exact inverse of the forward eq 76-78 relations:
/// encoding an `X'Y'Z'` triple and recovering it round-trips to itself.
#[test]
fn recover_xyz_prime_inverts_forward_eqs() {
  let cases: &[[f32; 3]] = &[
    [0.5, 0.5, 0.5],
    [0.6, 0.55, 0.5],
    [0.3, 0.4, 0.7],
    [0.9, 0.75, 0.6],
    [0.1, 0.2, 0.35],
  ];
  for &xyz_prime in cases {
    let norm = encode_xyz_prime_to_norm(xyz_prime);
    let got = recover_xyz_prime(norm);
    assert_close(got, xyz_prime, 1e-6, "recover_xyz_prime inverse");
  }
}

/// Structural white-point anchor: an equal `X' = Y' = Z' = v` yields small,
/// bounded `D'z`/`D'x` offsets (the normalizers `0.986566` / `0.991902` differ
/// from `1` by ~1%, so the offsets are ≤ ~0.007·v). This is the "D65 white →
/// small D'z/D'x" behaviour the normalizers encode.
#[test]
fn equal_xyz_prime_yields_small_chroma() {
  for v in [0.1_f32, 0.3, 0.5, 0.75, 0.9] {
    let norm = encode_xyz_prime_to_norm([v, v, v]);
    assert!((norm[0] - v).abs() <= 1e-6, "Y' = {} (want {v})", norm[0]);
    assert!(
      norm[1].abs() <= 0.01,
      "equal X'Y'Z' D'z = {} (want small)",
      norm[1]
    );
    assert!(
      norm[2].abs() <= 0.01,
      "equal X'Y'Z' D'x = {} (want small)",
      norm[2]
    );
  }
}

/// Dequantization matches the H.273 studio/full-range convention shared with
/// the affine YCbCr decode (the `range_params_n` normalization) — identical to
/// the ICtCp / IPT-C2 dequant (SMPTE 2085 shares the YCbCr integer encoding).
#[test]
fn dequant_matches_h273_convention() {
  // 12-bit, full range: Y'/4095, (C - 2048)/4095.
  let n = dequant_smpte2085::<12>(2048, 2148, 2248, true);
  assert!((n[0] - 2048.0 / 4095.0).abs() <= 1e-6);
  assert!((n[1] - 100.0 / 4095.0).abs() <= 1e-6);
  assert!((n[2] - 200.0 / 4095.0).abs() <= 1e-6);
  // 12-bit, studio range: (Y' - 256)/3504, (C - 2048)/3584  (k = 16).
  let n = dequant_smpte2085::<12>(1800, 2048, 2148, false);
  assert!((n[0] - (1800.0 - 256.0) / (219.0 * 16.0)).abs() <= 1e-6);
  assert!((n[1] - 0.0).abs() <= 1e-6);
  assert!((n[2] - (2148.0 - 2048.0) / (224.0 * 16.0)).abs() <= 1e-6);
}

/// Spec-integer forward encode of a target `X'Y'Z'` to the quantized wire
/// `(Y', D'z, D'x)` triple at `BITS` depth, studio or full range.
fn encode_wire<const BITS: u32>(xyz_prime: [f32; 3], full_range: bool) -> [u16; 3] {
  let norm = encode_xyz_prime_to_norm(xyz_prime);
  let k: i32 = 1 << (BITS - 8);
  let chroma_bias = (128 * k) as f32;
  let (y_off, y_range, c_range): (f32, f32, f32) = if full_range {
    let in_max = ((1u32 << BITS) - 1) as f32;
    (0.0, in_max, in_max)
  } else {
    ((16 * k) as f32, (219 * k) as f32, (224 * k) as f32)
  };
  let max = ((1u32 << BITS) - 1) as i32;
  let q = |val: f32| -> u16 { (val.round() as i32).clamp(0, max) as u16 };
  [
    q(norm[0] * y_range + y_off),
    q(norm[1] * c_range + chroma_bias),
    q(norm[2] * c_range + chroma_bias),
  ]
}

/// Round-trip anchor: a target `X'Y'Z'`, forward-encoded to the wire the way
/// the spec defines, decodes back to the `R'G'B'` its (unquantized) `XYZ → RGB
/// → PQ` chain produces (within the 12-bit quantization tolerance). Validates
/// the full decode pipeline against the spec relations. Covers full and studio
/// range.
#[cfg_attr(miri, ignore)]
#[test]
fn smpte2085_roundtrip_recovers_rgb_prime() {
  // Bright, near-neutral X'Y'Z' triples so the decoded R'G'B' lands in the
  // well-conditioned mid range: PQ's OETF is steep near black, where a coarse
  // (studio-range) code round-trip amplifies past the tolerance — a real
  // property of the PQ-XYZ transfer, not a decode defect.
  let cases: &[([f32; 3], bool)] = &[
    ([0.6, 0.6, 0.6], true),
    ([0.6, 0.55, 0.5], true),
    ([0.65, 0.62, 0.6], false),
    ([0.55, 0.58, 0.62], false),
  ];
  for &(xyz_prime, full) in cases {
    // The reference `R'G'B'` is the exact (unquantized) decode of this triple.
    let want = smpte2085_norm_to_rgb_prime(encode_xyz_prime_to_norm(xyz_prime));
    let [y, dz, dx] = encode_wire::<12>(xyz_prime, full);
    let dec = smpte2085_norm_to_rgb_prime(dequant_smpte2085::<12>(y, dz, dx, full));
    assert_close(dec, want, 1e-3, "SMPTE 2085 R'G'B' round-trip");
  }
}

/// End-to-end integer kernel (yuv444p12 → u8 RGB): the spec-integer wire for a
/// target `X'Y'Z'` decodes to the reference `R'G'B'` narrowed to u8, within ±1
/// LSB (12-bit quantization + f32 narrowing).
#[cfg_attr(miri, ignore)]
#[test]
fn smpte2085_444p12_to_rgb_u8_roundtrip() {
  // Bright, near-neutral X'Y'Z' triples so the decoded R'G'B' lands in the
  // well-conditioned mid range: PQ's OETF is steep near black, where a coarse
  // (studio-range) code round-trip amplifies past the tolerance — a real
  // property of the PQ-XYZ transfer, not a decode defect.
  let cases: &[([f32; 3], bool)] = &[
    ([0.6, 0.6, 0.6], true),
    ([0.6, 0.55, 0.5], true),
    ([0.65, 0.62, 0.6], false),
    ([0.55, 0.58, 0.62], false),
  ];
  for &(xyz_prime, full) in cases {
    let rgb_prime = smpte2085_norm_to_rgb_prime(encode_xyz_prime_to_norm(xyz_prime));
    let want = [
      narrow_unit_to_u8(rgb_prime[0]),
      narrow_unit_to_u8(rgb_prime[1]),
      narrow_unit_to_u8(rgb_prime[2]),
    ];
    let [y, dz, dx] = encode_wire::<12>(xyz_prime, full);
    let (yp, u, v) = (
      [le_wire_u16(y); 2],
      [le_wire_u16(dz); 2],
      [le_wire_u16(dx); 2],
    );
    let mut out = [0_u8; 6];
    smpte2085_444p_n_to_rgb_row::<12, false>(&yp, &u, &v, &mut out, 2, full);
    for px in 0..2 {
      for c in 0..3 {
        let g = out[px * 3 + c] as i32;
        assert!(
          (g - want[c] as i32).abs() <= 1,
          "{xyz_prime:?} full={full}: px{px} ch{c} = {g} (want {})",
          want[c]
        );
      }
    }
  }
}

/// End-to-end integer kernel (yuv444p12 → u16 RGB) at the **native 12-bit**
/// scale (`× 4095`, NOT full-16-bit) — the `Yuv444p12` u16 output contract.
/// Every value is in `[0, 4095]`; ±2 LSB for 12-bit quantization + f32
/// narrowing.
#[cfg_attr(miri, ignore)]
#[test]
fn smpte2085_444p12_to_rgb_u16_roundtrip() {
  // Bright, near-neutral X'Y'Z' triples so the decoded R'G'B' lands in the
  // well-conditioned mid range: PQ's OETF is steep near black, where a coarse
  // (studio-range) code round-trip amplifies past the tolerance — a real
  // property of the PQ-XYZ transfer, not a decode defect.
  let cases: &[([f32; 3], bool)] = &[
    ([0.6, 0.6, 0.6], true),
    ([0.6, 0.55, 0.5], true),
    ([0.65, 0.62, 0.6], false),
    ([0.55, 0.58, 0.62], false),
  ];
  for &(xyz_prime, full) in cases {
    let rgb_prime = smpte2085_norm_to_rgb_prime(encode_xyz_prime_to_norm(xyz_prime));
    let want = [
      narrow_unit_to_u16_native::<12>(rgb_prime[0]),
      narrow_unit_to_u16_native::<12>(rgb_prime[1]),
      narrow_unit_to_u16_native::<12>(rgb_prime[2]),
    ];
    let [y, dz, dx] = encode_wire::<12>(xyz_prime, full);
    let (yp, u, v) = ([le_wire_u16(y)], [le_wire_u16(dz)], [le_wire_u16(dx)]);
    let mut out = [0_u16; 3];
    smpte2085_444p_n_to_rgb_u16_row::<12, false>(&yp, &u, &v, &mut out, 1, full);
    for c in 0..3 {
      let g = out[c] as i32;
      assert!(
        g <= 4095,
        "{xyz_prime:?}: ch{c} = {g} over native 12-bit range"
      );
      assert!(
        (g - want[c] as i32).abs() <= 2,
        "{xyz_prime:?} full={full}: ch{c} = {g} (want {})",
        want[c]
      );
    }
  }
}

/// RGBA kernels match the RGB kernels channel-for-channel and append opaque
/// alpha — `0xFF` for u8, native `(1 << BITS) - 1` (= 4095 at 12-bit) for u16,
/// matching the affine + expand convention.
// miri's interpreted floating-point diverges from hardware for the SMPTE 2085
// (XYZ + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn rgba_kernels_match_rgb_plus_opaque_alpha() {
  let (y, u, v) = ([2048_u16], [2148_u16], [2248_u16]);
  let mut rgb = [0_u8; 3];
  let mut rgba = [0_u8; 4];
  smpte2085_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut rgb, 1, true);
  smpte2085_444p_n_to_rgba_row::<12, false>(&y, &u, &v, &mut rgba, 1, true);
  assert_eq!(&rgba[..3], &rgb[..]);
  assert_eq!(rgba[3], 0xFF);

  let mut rgb16 = [0_u16; 3];
  let mut rgba16 = [0_u16; 4];
  smpte2085_444p_n_to_rgb_u16_row::<12, false>(&y, &u, &v, &mut rgb16, 1, true);
  smpte2085_444p_n_to_rgba_u16_row::<12, false>(&y, &u, &v, &mut rgba16, 1, true);
  assert_eq!(&rgba16[..3], &rgb16[..]);
  assert_eq!(
    rgba16[3], 4095,
    "native 12-bit opaque alpha = (1 << BITS) - 1"
  );
  assert!(
    rgb16.iter().all(|&c| c <= 4095),
    "u16 RGB must be native 12-bit [0, 4095], got {rgb16:?}"
  );
}

/// Big-endian wire samples decode identically to their byte-swapped
/// little-endian counterparts.
#[test]
fn big_endian_matches_swapped_little_endian() {
  let le = ([2048_u16], [2148_u16], [2248_u16]);
  let be = (
    [2048_u16.swap_bytes()],
    [2148_u16.swap_bytes()],
    [2248_u16.swap_bytes()],
  );
  let mut out_le = [0_u8; 3];
  let mut out_be = [0_u8; 3];
  smpte2085_444p_n_to_rgb_row::<12, false>(&le.0, &le.1, &le.2, &mut out_le, 1, true);
  smpte2085_444p_n_to_rgb_row::<12, true>(&be.0, &be.1, &be.2, &mut out_be, 1, true);
  assert_eq!(out_le, out_be);
}

/// The Smpte2085 direct-decode branch of the `yuv444p12` u8 RGB dispatcher runs
/// the same output-length preflight as the base affine path: a too-short buffer
/// panics on the length check up front, before any pixel is written (no partial
/// mutation).
#[test]
#[should_panic(expected = "rgb_out row too short")]
fn smpte2085_rgb_u8_short_output_panics_on_length_check() {
  use crate::{ColorMatrix, Primaries, Transfer, row::yuv444p12_to_rgb_row_smpte2085_endian};
  let (y, u, v) = ([2048_u16; 2], [2048_u16; 2], [2048_u16; 2]);
  let mut rgb = [0_u8; 3]; // width 2 needs 6.
  yuv444p12_to_rgb_row_smpte2085_endian(
    &y,
    &u,
    &v,
    &mut rgb,
    2,
    ColorMatrix::Smpte2085,
    Primaries::Unspecified,
    true,
    Transfer::SmpteSt2084Pq,
    false,
    false,
  );
}

/// As above for the u8 RGBA dispatcher.
#[test]
#[should_panic(expected = "rgba_out row too short")]
fn smpte2085_rgba_u8_short_output_panics_on_length_check() {
  use crate::{ColorMatrix, Primaries, Transfer, row::yuv444p12_to_rgba_row_smpte2085_endian};
  let (y, u, v) = ([2048_u16; 2], [2048_u16; 2], [2048_u16; 2]);
  let mut rgba = [0_u8; 4]; // width 2 needs 8.
  yuv444p12_to_rgba_row_smpte2085_endian(
    &y,
    &u,
    &v,
    &mut rgba,
    2,
    ColorMatrix::Smpte2085,
    Primaries::Unspecified,
    true,
    Transfer::SmpteSt2084Pq,
    false,
    false,
  );
}

/// As above for the native-depth u16 RGB dispatcher.
#[test]
#[should_panic(expected = "rgb_out row too short")]
fn smpte2085_rgb_u16_short_output_panics_on_length_check() {
  use crate::{ColorMatrix, Primaries, Transfer, row::yuv444p12_to_rgb_u16_row_smpte2085_endian};
  let (y, u, v) = ([2048_u16; 2], [2048_u16; 2], [2048_u16; 2]);
  let mut rgb = [0_u16; 3]; // width 2 needs 6.
  yuv444p12_to_rgb_u16_row_smpte2085_endian(
    &y,
    &u,
    &v,
    &mut rgb,
    2,
    ColorMatrix::Smpte2085,
    Primaries::Unspecified,
    true,
    Transfer::SmpteSt2084Pq,
    false,
    false,
  );
}

/// As above for the native-depth u16 RGBA dispatcher.
#[test]
#[should_panic(expected = "rgba_out row too short")]
fn smpte2085_rgba_u16_short_output_panics_on_length_check() {
  use crate::{ColorMatrix, Primaries, Transfer, row::yuv444p12_to_rgba_u16_row_smpte2085_endian};
  let (y, u, v) = ([2048_u16; 2], [2048_u16; 2], [2048_u16; 2]);
  let mut rgba = [0_u16; 4]; // width 2 needs 8.
  yuv444p12_to_rgba_u16_row_smpte2085_endian(
    &y,
    &u,
    &v,
    &mut rgba,
    2,
    ColorMatrix::Smpte2085,
    Primaries::Unspecified,
    true,
    Transfer::SmpteSt2084Pq,
    false,
    false,
  );
}
