//! Reference cross-checks for the IPT-C2 (H.273 code 15) non-affine decode.
//!
//! No external library is needed: the published H.273 §8.3 Table 4 code 15
//! integer forward matrices ARE the oracle. The decode matrices are their
//! exact rational inverses, and the end-to-end kernels are validated by a
//! spec-integer encode → decode round-trip (encode a target `R'G'B'` the way
//! the spec defines — `RGB→LMS` via eq 17-19, PQ, rotation via eq 85-87,
//! studio/full quant — then assert the kernel recovers it).

use super::*;

/// Re-encode a logical u16 as LE wire storage so a `BE = false` kernel recovers
/// it via `u16::from_le` on both endiannesses (the iptc2 kernels read wire u16).
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// H.273 §8.3 Table 4 code 15 published forward matrices (`×4096` integer
/// form). The decode constants invert these.
const RGB_TO_LMS: [[f32; 3]; 3] = [
  [1747.0 / 4096.0, 2169.0 / 4096.0, 180.0 / 4096.0],
  [673.0 / 4096.0, 3029.0 / 4096.0, 394.0 / 4096.0],
  [50.0 / 4096.0, 207.0 / 4096.0, 3839.0 / 4096.0],
];
const LMSP_TO_IPT: [[f32; 3]; 3] = [
  [1638.0 / 4096.0, 1638.0 / 4096.0, 820.0 / 4096.0],
  [18248.0 / 4096.0, -19870.0 / 4096.0, 1622.0 / 4096.0],
  [3300.0 / 4096.0, 1463.0 / 4096.0, -4763.0 / 4096.0],
];

fn matmul_mm(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
  let mut out = [[0.0_f32; 3]; 3];
  for (i, row) in out.iter_mut().enumerate() {
    for (j, cell) in row.iter_mut().enumerate() {
      *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
    }
  }
  out
}

fn assert_identity(m: &[[f32; 3]; 3], tol: f32, what: &str) {
  for (i, row) in m.iter().enumerate() {
    for (j, &cell) in row.iter().enumerate() {
      let want = if i == j { 1.0 } else { 0.0 };
      assert!(
        (cell - want).abs() <= tol,
        "{what}: M[{i}][{j}] = {cell} (want {want})"
      );
    }
  }
}

/// The decode matrices are genuine inverses of the published H.273 forward
/// matrices: `M_fwd · M_inv = I`.
#[test]
fn decode_matrices_are_exact_inverses() {
  assert_identity(
    &matmul_mm(&RGB_TO_LMS, &IPTC2_LMS_TO_RGB),
    1e-4,
    "RGB→LMS · LMS→RGB",
  );
  assert_identity(
    &matmul_mm(&LMSP_TO_IPT, &IPTC2_IPT_TO_LMSP),
    1e-4,
    "L'M'S'→IPT · IPT→L'M'S'",
  );
}

/// `IptC2Transfer::for_transfer` resolves only the PQ transfer; everything
/// else (incl. `Unspecified`) is `None` → affine fallback.
#[test]
fn for_transfer_selects_pq_only() {
  use crate::Transfer;
  assert_eq!(
    IptC2Transfer::for_transfer(Transfer::SmpteSt2084Pq),
    Some(IptC2Transfer::Pq)
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
      IptC2Transfer::for_transfer(t),
      None,
      "{t:?} must not select the IPT-C2 transfer"
    );
  }
  assert_eq!(IptC2Transfer::Pq.as_str(), "pq");
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

/// Structural neutral-axis anchor (library-independent): an equal
/// `L' = M' = S' = v` maps to `I = v, P = T = 0` under the forward eq 85-87
/// rotation. The P and T rows of `L'M'S'→IPT` each sum to `0`, and the I row
/// sums to `1`.
#[test]
fn equal_lmsp_maps_to_i_only() {
  for v in [0.1_f32, 0.3, 0.5, 0.75, 0.9] {
    let ipt = matmul3(&LMSP_TO_IPT, [v, v, v]);
    assert_close(ipt, [v, 0.0, 0.0], 2e-4, "equal L'M'S' → [I,0,0]");
  }
}

/// Structural neutral-axis anchor (the decode direction): a neutral `IPT`
/// (`P = T = 0`) decodes to a neutral `R'G'B'` with `R' = G' = B' = I`.
/// `I = v` ⇒ `L' = M' = S' = v` (each inverse row is `[1, x, y]`), the
/// `LMS→RGB` rows sum to `1` so gray stays gray, and `OETF(EOTF(v)) = v`.
#[test]
fn neutral_ipt_decodes_to_neutral_grey() {
  for v in [0.1_f32, 0.3, 0.5, 0.75, 0.9] {
    let rgb = iptc2_norm_to_rgb_prime([v, 0.0, 0.0]);
    assert_close(rgb, [v, v, v], 2e-4, "neutral grey");
  }
}

/// A gray `R'G'B'` input, forward-encoded the way the spec defines, yields
/// `P = T = 0` in the normalized IPT domain: `RGB→LMS` maps gray to gray (its
/// rows sum to 1), PQ leaves it equal, and the rotation's P/T rows sum to 0.
#[test]
fn grey_rgb_encodes_to_zero_chroma() {
  for g in [0.1_f32, 0.35, 0.6, 0.85] {
    let ipt = encode_ipt_norm([g, g, g]);
    assert!(ipt[1].abs() <= 2e-4, "grey P = {} (want 0)", ipt[1]);
    assert!(ipt[2].abs() <= 2e-4, "grey T = {} (want 0)", ipt[2]);
  }
}

/// Dequantization matches the H.273 studio/full-range convention shared with
/// the affine YCbCr decode (the `range_params_n` normalization) — identical to
/// the ICtCp dequant (IPT-C2 shares the YCbCr integer encoding).
#[test]
fn dequant_matches_h273_convention() {
  // 12-bit, full range: I/4095, (C - 2048)/4095.
  let n = dequant_iptc2::<12>(2048, 2148, 2248, true);
  assert!((n[0] - 2048.0 / 4095.0).abs() <= 1e-6);
  assert!((n[1] - 100.0 / 4095.0).abs() <= 1e-6);
  assert!((n[2] - 200.0 / 4095.0).abs() <= 1e-6);
  // 12-bit, studio range: (I - 256)/3504, (C - 2048)/3584  (k = 16).
  let n = dequant_iptc2::<12>(1800, 2048, 2148, false);
  assert!((n[0] - (1800.0 - 256.0) / (219.0 * 16.0)).abs() <= 1e-6);
  assert!((n[1] - 0.0).abs() <= 1e-6);
  assert!((n[2] - (2148.0 - 2048.0) / (224.0 * 16.0)).abs() <= 1e-6);
}

/// Spec-integer forward encode of a target `R'G'B'` to normalized `IPT`:
/// `R'G'B' → linear` (PQ EOTF), `→ LMS` (eq 17-19), `→ L'M'S'` (PQ OETF),
/// `→ IPT` (eq 85-87). The exact inverse of the kernel's decode.
fn encode_ipt_norm(rgb_prime: [f32; 3]) -> [f32; 3] {
  let lin = [
    crate::resample::pq_hlg::pq_eotf(rgb_prime[0]),
    crate::resample::pq_hlg::pq_eotf(rgb_prime[1]),
    crate::resample::pq_hlg::pq_eotf(rgb_prime[2]),
  ];
  let lms = matmul3(&RGB_TO_LMS, lin);
  let lms_p = [
    crate::resample::pq_hlg::pq_oetf(lms[0]),
    crate::resample::pq_hlg::pq_oetf(lms[1]),
    crate::resample::pq_hlg::pq_oetf(lms[2]),
  ];
  matmul3(&LMSP_TO_IPT, lms_p)
}

/// Spec-integer forward encode of a target `R'G'B'` all the way to the quantized
/// wire `(I, P, T)` triple at `BITS` depth, studio or full range.
fn encode_wire<const BITS: u32>(rgb_prime: [f32; 3], full_range: bool) -> [u16; 3] {
  let ipt = encode_ipt_norm(rgb_prime);
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
    q(ipt[0] * y_range + y_off),
    q(ipt[1] * c_range + chroma_bias),
    q(ipt[2] * c_range + chroma_bias),
  ]
}

/// Round-trip anchor: a target `R'G'B'`, forward-encoded to the wire the way
/// the spec defines, decodes back to the same `R'G'B'` (within the 12-bit
/// quantization tolerance). Validates the full decode pipeline against the
/// spec integer matrices. Covers full and studio range.
#[cfg_attr(miri, ignore)]
#[test]
fn iptc2_roundtrip_recovers_rgb_prime() {
  let cases: &[([f32; 3], bool)] = &[
    ([0.5, 0.4, 0.45], true),
    ([0.6, 0.2, 0.3], true),
    ([0.3, 0.35, 0.5], false),
    ([0.7, 0.6, 0.55], false),
  ];
  for &(rgb_prime, full) in cases {
    let [i, p, t] = encode_wire::<12>(rgb_prime, full);
    let dec = iptc2_norm_to_rgb_prime(dequant_iptc2::<12>(i, p, t, full));
    assert_close(dec, rgb_prime, 1e-3, "IPT-C2 R'G'B' round-trip");
  }
}

/// End-to-end integer kernel (yuv444p12 → u8 RGB): the spec-integer wire for a
/// target `R'G'B'` decodes back to that `R'G'B'` narrowed to u8, within ±1 LSB
/// (12-bit quantization + f32 narrowing).
#[cfg_attr(miri, ignore)]
#[test]
fn iptc2_444p12_to_rgb_u8_roundtrip() {
  let cases: &[([f32; 3], bool)] = &[
    ([0.5, 0.4, 0.45], true),
    ([0.6, 0.2, 0.3], true),
    ([0.3, 0.35, 0.5], false),
    ([0.7, 0.6, 0.55], false),
  ];
  for &(rgb_prime, full) in cases {
    let [i, p, t] = encode_wire::<12>(rgb_prime, full);
    let want = [
      narrow_unit_to_u8(rgb_prime[0]),
      narrow_unit_to_u8(rgb_prime[1]),
      narrow_unit_to_u8(rgb_prime[2]),
    ];
    let (y, u, v) = (
      [le_wire_u16(i); 2],
      [le_wire_u16(p); 2],
      [le_wire_u16(t); 2],
    );
    let mut out = [0_u8; 6];
    iptc2_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut out, 2, full);
    for px in 0..2 {
      for c in 0..3 {
        let g = out[px * 3 + c] as i32;
        assert!(
          (g - want[c] as i32).abs() <= 1,
          "{rgb_prime:?} full={full}: px{px} ch{c} = {g} (want {})",
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
fn iptc2_444p12_to_rgb_u16_roundtrip() {
  let cases: &[([f32; 3], bool)] = &[
    ([0.5, 0.4, 0.45], true),
    ([0.6, 0.2, 0.3], true),
    ([0.3, 0.35, 0.5], false),
    ([0.7, 0.6, 0.55], false),
  ];
  for &(rgb_prime, full) in cases {
    let [i, p, t] = encode_wire::<12>(rgb_prime, full);
    let want = [
      narrow_unit_to_u16_native::<12>(rgb_prime[0]),
      narrow_unit_to_u16_native::<12>(rgb_prime[1]),
      narrow_unit_to_u16_native::<12>(rgb_prime[2]),
    ];
    let (y, u, v) = ([le_wire_u16(i)], [le_wire_u16(p)], [le_wire_u16(t)]);
    let mut out = [0_u16; 3];
    iptc2_444p_n_to_rgb_u16_row::<12, false>(&y, &u, &v, &mut out, 1, full);
    for c in 0..3 {
      let g = out[c] as i32;
      assert!(
        g <= 4095,
        "{rgb_prime:?}: ch{c} = {g} over native 12-bit range"
      );
      assert!(
        (g - want[c] as i32).abs() <= 2,
        "{rgb_prime:?} full={full}: ch{c} = {g} (want {})",
        want[c]
      );
    }
  }
}

/// RGBA kernels match the RGB kernels channel-for-channel and append opaque
/// alpha — `0xFF` for u8, native `(1 << BITS) - 1` (= 4095 at 12-bit) for u16,
/// matching the affine + expand convention.
// miri's interpreted floating-point diverges from hardware for the IPT-C2
// (LMS + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn rgba_kernels_match_rgb_plus_opaque_alpha() {
  let (y, u, v) = ([2048_u16], [2148_u16], [2248_u16]);
  let mut rgb = [0_u8; 3];
  let mut rgba = [0_u8; 4];
  iptc2_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut rgb, 1, true);
  iptc2_444p_n_to_rgba_row::<12, false>(&y, &u, &v, &mut rgba, 1, true);
  assert_eq!(&rgba[..3], &rgb[..]);
  assert_eq!(rgba[3], 0xFF);

  let mut rgb16 = [0_u16; 3];
  let mut rgba16 = [0_u16; 4];
  iptc2_444p_n_to_rgb_u16_row::<12, false>(&y, &u, &v, &mut rgb16, 1, true);
  iptc2_444p_n_to_rgba_u16_row::<12, false>(&y, &u, &v, &mut rgba16, 1, true);
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
  iptc2_444p_n_to_rgb_row::<12, false>(&le.0, &le.1, &le.2, &mut out_le, 1, true);
  iptc2_444p_n_to_rgb_row::<12, true>(&be.0, &be.1, &be.2, &mut out_be, 1, true);
  assert_eq!(out_le, out_be);
}
