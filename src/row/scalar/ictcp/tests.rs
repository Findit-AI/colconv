//! Reference cross-checks for the ICtCp (BT.2100) non-affine decode.
//!
//! The decode is pinned against `colour-science` 0.4.7 — the authoritative
//! BT.2100 ICtCp implementation (`colour.ICtCp_to_RGB`, PQ and HLG) — plus
//! BT.2100-2 structural anchors that hold independently of any library.
//!
//! Domain convention (matching pixon's normalized transfer, where signal
//! `1.0` = 10 000 cd/m² for PQ): a `colour-science` PQ result is in absolute
//! `cd/m²` with `L_p = 10000`, so `pixon linear RGB = colour / 10000`; HLG
//! has no `L_p` (scene-linear `[0, 1]`), so they coincide directly. This is
//! the same `÷10000` relationship the #313 transfer tests already establish
//! (`pq_oetf(0.01)` = `colour eotf_inverse_ST2084(100)`).

use super::*;

/// Re-encode a logical u16 as LE wire storage so a `BE = false` kernel recovers
/// it via `u16::from_le` on both endiannesses (the ictcp kernels read wire u16).
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// BT.2100-2 published forward matrices (`×4096` integer form, #313).
const RGB_TO_LMS: [[f32; 3]; 3] = [
  [1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0],
  [683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0],
  [99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0],
];
const LMSP_TO_ICTCP_PQ: [[f32; 3]; 3] = [
  [2048.0 / 4096.0, 2048.0 / 4096.0, 0.0],
  [6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0],
  [17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0],
];
const LMSP_TO_ICTCP_HLG: [[f32; 3]; 3] = [
  [2048.0 / 4096.0, 2048.0 / 4096.0, 0.0],
  [3625.0 / 4096.0, -7465.0 / 4096.0, 3840.0 / 4096.0],
  [9500.0 / 4096.0, -9212.0 / 4096.0, -288.0 / 4096.0],
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

/// The decode matrices are genuine inverses of the published BT.2100 forward
/// matrices: `M_fwd · M_inv = I`.
#[test]
fn decode_matrices_are_exact_inverses() {
  assert_identity(
    &matmul_mm(&RGB_TO_LMS, &LMS_TO_RGB),
    2e-6,
    "RGB→LMS · LMS→RGB",
  );
  assert_identity(
    &matmul_mm(&LMSP_TO_ICTCP_PQ, &ICTCP_TO_LMSP_PQ),
    2e-6,
    "PQ L'M'S'→ICtCp · ICtCp→L'M'S'",
  );
  assert_identity(
    &matmul_mm(&LMSP_TO_ICTCP_HLG, &ICTCP_TO_LMSP_HLG),
    2e-6,
    "HLG L'M'S'→ICtCp · ICtCp→L'M'S'",
  );
}

/// `IctcpTransfer::for_transfer` resolves only the two BT.2100 ICtCp
/// transfers; everything else (incl. `Unspecified`) is `None` → affine
/// fallback.
#[test]
fn for_transfer_selects_pq_and_hlg_only() {
  use crate::Transfer;
  assert_eq!(
    IctcpTransfer::for_transfer(&Transfer::SmpteSt2084Pq),
    Some(IctcpTransfer::Pq)
  );
  assert_eq!(
    IctcpTransfer::for_transfer(&Transfer::AribStdB67Hlg),
    Some(IctcpTransfer::Hlg)
  );
  for t in [
    Transfer::Unspecified,
    Transfer::Bt709,
    Transfer::Bt2020_10Bit,
    Transfer::SmpteSt428,
    Transfer::other("unassigned-99"),
  ] {
    assert_eq!(
      IctcpTransfer::for_transfer(&t),
      None,
      "{t:?} must not select an ICtCp transfer"
    );
  }
  assert_eq!(IctcpTransfer::Pq.as_str(), "pq");
  assert_eq!(IctcpTransfer::Hlg.as_str(), "hlg");
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

/// PQ decode pinned against `colour.ICtCp_to_RGB` (default `ITU-R BT.2100-1`,
/// `L_p = 10000`). Linear RGB = `colour / 10000`; `R'G'B'` = that re-encoded
/// through the PQ OETF.
#[test]
fn pq_decode_matches_colour_science() {
  let tf = IctcpTransfer::Pq;
  // colour ICtCp_to_RGB([0.5, 0.05, 0.10]) = [155.9897, 64.0029, 88.8101]
  assert_close(
    ictcp_norm_to_rgb_linear([0.5, 0.05, 0.10], tf),
    [1.559_897_1e-2, 6.400_287e-3, 8.881_01e-3],
    5e-5,
    "PQ linear [0.5,0.05,0.10]",
  );
  assert_close(
    ictcp_norm_to_rgb_prime([0.5, 0.05, 0.10], tf),
    [0.553_335_65, 0.464_005_47, 0.496_216_85],
    1e-3,
    "PQ R'G'B' [0.5,0.05,0.10]",
  );
  // colour ICtCp_to_RGB([0.4, 0.1, 0.2]) = [82.2899, 11.1387, 29.8106]
  assert_close(
    ictcp_norm_to_rgb_linear([0.4, 0.1, 0.2], tf),
    [8.228_991e-3, 1.113_868_4e-3, 2.981_062_3e-3],
    5e-5,
    "PQ linear [0.4,0.1,0.2]",
  );
  // The canonical colour-science docstring example, inverted:
  // ICtCp_to_RGB([0.07351364, 0.00475253, 0.09351596])
  //   = [0.45620521, 0.03081070, 0.04091952]  (÷10000 in our domain)
  assert_close(
    ictcp_norm_to_rgb_linear([0.073_513_64, 0.004_752_53, 0.093_515_96], tf),
    [4.562_052e-5, 3.081_070_3e-6, 4.091_952e-6],
    1e-7,
    "PQ docstring anchor",
  );
}

/// HLG decode pinned against `colour.ICtCp_to_RGB(method="ITU-R BT.2100-2
/// HLG")`. No `L_p`; pixon linear RGB equals the `colour` result directly.
#[test]
fn hlg_decode_matches_colour_science() {
  let tf = IctcpTransfer::Hlg;
  // colour ICtCp_to_RGB([0.6, -0.04, 0.02], HLG) = [0.13850, 0.12739, 0.09808]
  assert_close(
    ictcp_norm_to_rgb_linear([0.6, -0.04, 0.02], tf),
    [1.384_960_1e-1, 1.273_943_6e-1, 9.808_244e-2],
    5e-5,
    "HLG linear [0.6,-0.04,0.02]",
  );
  assert_close(
    ictcp_norm_to_rgb_prime([0.6, -0.04, 0.02], tf),
    [0.617_157_4, 0.598_964_73, 0.539_536_3],
    1e-3,
    "HLG R'G'B' [0.6,-0.04,0.02]",
  );
  // colour ICtCp_to_RGB([0.4, 0.1, 0.2], HLG) = [0.12250, 0.02285, 0.04856]
  assert_close(
    ictcp_norm_to_rgb_linear([0.4, 0.1, 0.2], tf),
    [1.224_960_9e-1, 2.285_209_4e-2, 4.855_745e-2],
    5e-5,
    "HLG linear [0.4,0.1,0.2]",
  );
}

/// BT.2100 structural anchor (library-independent): a neutral `ICtCp`
/// (`Ct = Cp = 0`) decodes to a neutral `R'G'B'` with `R' = G' = B' = I`.
/// `I = v` ⇒ `L' = M' = S' = v` (each inverse row is `[1, x, y]`), the
/// `LMS→RGB` rows sum to `1` so the gray stays gray, and `OETF(EOTF(v)) = v`.
/// Holds for **both** transfers.
#[test]
fn neutral_ictcp_decodes_to_neutral_grey() {
  for tf in [IctcpTransfer::Pq, IctcpTransfer::Hlg] {
    for v in [0.1_f32, 0.3, 0.5, 0.75, 0.9] {
      let rgb = ictcp_norm_to_rgb_prime([v, 0.0, 0.0], tf);
      assert_close(rgb, [v, v, v], 2e-4, "neutral grey");
    }
  }
}

/// The PQ-vs-HLG selection genuinely matters: the same `ICtCp` triple
/// decodes to substantially different RGB under PQ vs HLG (different inverse
/// matrix *and* different transfer). Guards against the transfer-dependent
/// selection being dropped.
#[test]
fn pq_and_hlg_selection_differ() {
  let norm = [0.4_f32, 0.1, 0.2];
  let pq = ictcp_norm_to_rgb_linear(norm, IctcpTransfer::Pq);
  let hlg = ictcp_norm_to_rgb_linear(norm, IctcpTransfer::Hlg);
  let max_diff = (0..3)
    .map(|i| (pq[i] - hlg[i]).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_diff > 1e-2,
    "PQ and HLG decode must differ (max |Δ| = {max_diff})"
  );
}

/// Dequantization matches the H.273 studio/full-range convention shared with
/// the affine YCbCr decode (the `range_params_n` normalization).
#[test]
fn dequant_matches_h273_convention() {
  // 12-bit, full range: I/4095, (C - 2048)/4095.
  let n = dequant_ictcp::<12>(2048, 2148, 2248, true);
  assert!((n[0] - 2048.0 / 4095.0).abs() <= 1e-6);
  assert!((n[1] - 100.0 / 4095.0).abs() <= 1e-6);
  assert!((n[2] - 200.0 / 4095.0).abs() <= 1e-6);
  // 12-bit, studio range: (I - 256)/3504, (C - 2048)/3584  (k = 16).
  let n = dequant_ictcp::<12>(1800, 2048, 2148, false);
  assert!((n[0] - (1800.0 - 256.0) / (219.0 * 16.0)).abs() <= 1e-6);
  assert!((n[1] - 0.0).abs() <= 1e-6);
  assert!((n[2] - (2148.0 - 2048.0) / (224.0 * 16.0)).abs() <= 1e-6);
}

/// End-to-end integer kernel (yuv444p12 → u8 RGB) pinned against the
/// `colour-science`-derived integer outputs for the exact integer samples.
/// f32-vs-f64 narrowing differences stay within ±1 LSB.
#[test]
fn ictcp_444p12_to_rgb_u8_matches_reference() {
  // (I, Ct, Cp, full_range, transfer, expected u8 RGB) from colour-science.
  let cases: &[([u16; 3], bool, IctcpTransfer, [u8; 3])] = &[
    ([2048, 2148, 2248], true, IctcpTransfer::Pq, [135, 123, 127]),
    (
      [2048, 2148, 2248],
      true,
      IctcpTransfer::Hlg,
      [141, 120, 126],
    ),
    (
      [1800, 2048, 2148],
      false,
      IctcpTransfer::Pq,
      [117, 111, 110],
    ),
    (
      [2200, 1948, 2148],
      false,
      IctcpTransfer::Hlg,
      [148, 140, 128],
    ),
  ];
  for &([i, ct, cp], full, tf, want) in cases {
    let (y, u, v) = (
      [le_wire_u16(i); 2],
      [le_wire_u16(ct); 2],
      [le_wire_u16(cp); 2],
    );
    let mut out = [0_u8; 6];
    ictcp_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut out, 2, full, tf);
    for px in 0..2 {
      for c in 0..3 {
        let g = out[px * 3 + c] as i32;
        assert!(
          (g - want[c] as i32).abs() <= 1,
          "{:?} {tf:?} full={full}: px{px} ch{c} = {g} (want {})",
          [i, ct, cp],
          want[c]
        );
      }
    }
  }
}

/// End-to-end integer kernel (yuv444p12 → u16 RGB) against the
/// `colour-science`-derived outputs at the **native 12-bit** scale
/// (`× 4095`, NOT full-16-bit) — the `Yuv444p12` u16 output contract. Every
/// value is in `[0, 4095]`; ±2 LSB for f32 narrowing.
#[test]
fn ictcp_444p12_to_rgb_u16_matches_reference() {
  let cases: &[([u16; 3], bool, IctcpTransfer, [u16; 3])] = &[
    (
      [2048, 2148, 2248],
      true,
      IctcpTransfer::Pq,
      [2167, 1981, 2040],
    ),
    (
      [2048, 2148, 2248],
      true,
      IctcpTransfer::Hlg,
      [2271, 1927, 2030],
    ),
    (
      [1800, 2048, 2148],
      false,
      IctcpTransfer::Pq,
      [1872, 1775, 1764],
    ),
    (
      [2200, 1948, 2148],
      false,
      IctcpTransfer::Hlg,
      [2383, 2242, 2061],
    ),
  ];
  for &([i, ct, cp], full, tf, want) in cases {
    let (y, u, v) = ([le_wire_u16(i)], [le_wire_u16(ct)], [le_wire_u16(cp)]);
    let mut out = [0_u16; 3];
    ictcp_444p_n_to_rgb_u16_row::<12, false>(&y, &u, &v, &mut out, 1, full, tf);
    for c in 0..3 {
      let g = out[c] as i32;
      assert!(
        g <= 4095,
        "{:?} {tf:?}: ch{c} = {g} over native 12-bit range",
        [i, ct, cp]
      );
      assert!(
        (g - want[c] as i32).abs() <= 2,
        "{:?} {tf:?} full={full}: ch{c} = {g} (want {})",
        [i, ct, cp],
        want[c]
      );
    }
  }
}

/// RGBA kernels match the RGB kernels channel-for-channel and append opaque
/// alpha — `0xFF` for u8, native `(1 << BITS) - 1` (= 4095 at 12-bit) for
/// u16, matching the affine + expand convention.
// miri's interpreted floating-point diverges from hardware for the ICtCp
// (LMS + PQ) transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn rgba_kernels_match_rgb_plus_opaque_alpha() {
  let (y, u, v) = ([2048_u16], [2148_u16], [2248_u16]);
  let mut rgb = [0_u8; 3];
  let mut rgba = [0_u8; 4];
  ictcp_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut rgb, 1, true, IctcpTransfer::Pq);
  ictcp_444p_n_to_rgba_row::<12, false>(&y, &u, &v, &mut rgba, 1, true, IctcpTransfer::Pq);
  assert_eq!(&rgba[..3], &rgb[..]);
  assert_eq!(rgba[3], 0xFF);

  let mut rgb16 = [0_u16; 3];
  let mut rgba16 = [0_u16; 4];
  ictcp_444p_n_to_rgb_u16_row::<12, false>(&y, &u, &v, &mut rgb16, 1, true, IctcpTransfer::Pq);
  ictcp_444p_n_to_rgba_u16_row::<12, false>(&y, &u, &v, &mut rgba16, 1, true, IctcpTransfer::Pq);
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
  ictcp_444p_n_to_rgb_row::<12, false>(
    &le.0,
    &le.1,
    &le.2,
    &mut out_le,
    1,
    true,
    IctcpTransfer::Pq,
  );
  ictcp_444p_n_to_rgb_row::<12, true>(&be.0, &be.1, &be.2, &mut out_be, 1, true, IctcpTransfer::Pq);
  assert_eq!(out_le, out_be);
}
