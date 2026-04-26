use super::*;

#[test]
fn white_balance_neutral_is_default() {
  assert_eq!(WhiteBalance::default(), WhiteBalance::neutral());
  assert_eq!(WhiteBalance::neutral().r(), 1.0);
  assert_eq!(WhiteBalance::neutral().g(), 1.0);
  assert_eq!(WhiteBalance::neutral().b(), 1.0);
}

#[test]
fn ccm_identity_is_default() {
  assert_eq!(
    ColorCorrectionMatrix::default(),
    ColorCorrectionMatrix::identity()
  );
  let id = ColorCorrectionMatrix::identity();
  let m = id.as_array();
  assert_eq!(m[0], [1.0, 0.0, 0.0]);
  assert_eq!(m[1], [0.0, 1.0, 0.0]);
  assert_eq!(m[2], [0.0, 0.0, 1.0]);
}

#[test]
fn fuse_wb_ccm_with_neutral_wb_returns_ccm() {
  let ccm = ColorCorrectionMatrix::new([[1.0, 0.5, 0.25], [0.0, 0.8, 0.2], [0.1, 0.1, 0.7]]);
  let m = fuse_wb_ccm(&WhiteBalance::neutral(), &ccm);
  assert_eq!(&m, ccm.as_array());
}

#[test]
fn fuse_wb_ccm_with_identity_ccm_returns_diag_wb() {
  let wb = WhiteBalance::new(1.5, 1.0, 2.0);
  let m = fuse_wb_ccm(&wb, &ColorCorrectionMatrix::identity());
  assert_eq!(m, [[1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
}

#[test]
fn fuse_wb_ccm_scales_columns_by_wb() {
  // M = CCM · diag(wb) ⇒ column j of M is column j of CCM × wb_j.
  let ccm = ColorCorrectionMatrix::new([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0], [64.0, 128.0, 256.0]]);
  let wb = WhiteBalance::new(0.5, 1.0, 0.25);
  let m = fuse_wb_ccm(&wb, &ccm);
  assert_eq!(m[0], [0.5, 2.0, 1.0]);
  assert_eq!(m[1], [4.0, 16.0, 8.0]);
  assert_eq!(m[2], [32.0, 128.0, 64.0]);
}

// ---- WhiteBalance validation ------------------------------------------

#[test]
fn wb_try_new_rejects_nan() {
  let e = WhiteBalance::try_new(f32::NAN, 1.0, 1.0).unwrap_err();
  assert!(matches!(
    e,
    WhiteBalanceError::NonFinite {
      channel: WbChannel::R,
      ..
    }
  ));
  let e = WhiteBalance::try_new(1.0, f32::NAN, 1.0).unwrap_err();
  assert!(matches!(
    e,
    WhiteBalanceError::NonFinite {
      channel: WbChannel::G,
      ..
    }
  ));
  let e = WhiteBalance::try_new(1.0, 1.0, f32::NAN).unwrap_err();
  assert!(matches!(
    e,
    WhiteBalanceError::NonFinite {
      channel: WbChannel::B,
      ..
    }
  ));
}

#[test]
fn wb_try_new_rejects_infinity() {
  let e = WhiteBalance::try_new(f32::INFINITY, 1.0, 1.0).unwrap_err();
  assert!(matches!(e, WhiteBalanceError::NonFinite { .. }));
  let e = WhiteBalance::try_new(1.0, f32::NEG_INFINITY, 1.0).unwrap_err();
  assert!(matches!(e, WhiteBalanceError::NonFinite { .. }));
}

#[test]
fn wb_try_new_rejects_negative() {
  let e = WhiteBalance::try_new(-0.1, 1.0, 1.0).unwrap_err();
  assert!(matches!(
    e,
    WhiteBalanceError::Negative {
      channel: WbChannel::R,
      ..
    }
  ));
}

#[test]
fn wb_try_new_accepts_zero_gain() {
  // Zero gain zeroes the channel — degenerate but well-defined.
  let wb = WhiteBalance::try_new(0.0, 1.0, 0.0).expect("zero gain valid");
  assert_eq!(wb.r(), 0.0);
}

#[test]
fn wb_try_new_accepts_typical_gains() {
  let wb = WhiteBalance::try_new(1.95, 1.0, 1.55).expect("typical");
  assert_eq!((wb.r(), wb.g(), wb.b()), (1.95, 1.0, 1.55));
}

#[test]
#[should_panic(expected = "invalid WhiteBalance")]
fn wb_new_panics_on_nan() {
  let _ = WhiteBalance::new(f32::NAN, 1.0, 1.0);
}

// ---- ColorCorrectionMatrix validation ---------------------------------

#[test]
fn ccm_try_new_rejects_nan_off_diagonal() {
  let e = ColorCorrectionMatrix::try_new([[1.0, f32::NAN, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    .unwrap_err();
  assert!(matches!(
    e,
    ColorCorrectionMatrixError::NonFinite { row: 0, col: 1, .. }
  ));
}

#[test]
fn ccm_try_new_rejects_infinity_diagonal() {
  let e =
    ColorCorrectionMatrix::try_new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, f32::INFINITY]])
      .unwrap_err();
  assert!(matches!(
    e,
    ColorCorrectionMatrixError::NonFinite { row: 2, col: 2, .. }
  ));
}

#[test]
fn ccm_try_new_accepts_negative_off_diagonal() {
  // Real CCMs subtract crosstalk → negative off-diagonal entries
  // are normal. Only non-finite values should fail.
  let ccm =
    ColorCorrectionMatrix::try_new([[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]])
      .expect("negative entries valid");
  assert_eq!(ccm.as_array()[0][1], -0.3);
}

#[test]
#[should_panic(expected = "invalid ColorCorrectionMatrix")]
fn ccm_new_panics_on_nan() {
  let _ = ColorCorrectionMatrix::new([[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
}

#[test]
fn fuse_wb_ccm_with_validated_inputs_is_finite() {
  // Sanity: validated inputs always produce a finite fused matrix.
  let wb = WhiteBalance::new(1.95, 1.0, 1.55);
  let ccm = ColorCorrectionMatrix::new([[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]]);
  let m = fuse_wb_ccm(&wb, &ccm);
  for row in m.iter() {
    for &v in row.iter() {
      assert!(v.is_finite(), "fused matrix has non-finite value: {v}");
    }
  }
}

// ---- WhiteBalance / ColorCorrectionMatrix magnitude bounds -------------

#[test]
fn wb_try_new_rejects_extreme_finite_gain() {
  // A finite gain above the magnitude bound is rejected even
  // though it would pass the NaN / Inf / negative checks. Real
  // camera WB gains are O(1–10); 1e10 is well past the bound
  // and would risk overflowing the per-pixel matmul.
  let e = WhiteBalance::try_new(1e10, 1.0, 1.0).unwrap_err();
  assert!(matches!(
    e,
    WhiteBalanceError::OutOfBounds {
      channel: WbChannel::R,
      ..
    }
  ));
}

#[test]
fn wb_try_new_accepts_value_at_bound() {
  // Exactly at the bound is permitted; the bound itself doesn't
  // overflow downstream arithmetic.
  let wb = WhiteBalance::try_new(WhiteBalance::MAX_GAIN, 1.0, 1.0).expect("at-bound valid");
  assert_eq!(wb.r(), WhiteBalance::MAX_GAIN);
}

#[test]
fn ccm_try_new_rejects_extreme_finite_coefficient() {
  // Same principle for CCM elements — finite-but-extreme values
  // that pass the is_finite check but would overflow per-pixel
  // matmul are rejected via OutOfBounds.
  let e = ColorCorrectionMatrix::try_new([[1.0, 0.0, 1e30], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    .unwrap_err();
  assert!(matches!(
    e,
    ColorCorrectionMatrixError::OutOfBounds { row: 0, col: 2, .. }
  ));
}

#[test]
fn ccm_try_new_rejects_extreme_negative_coefficient() {
  // Symmetric negative bound: real CCMs have negative
  // off-diagonals, but only in the realistic ~[-5, 5] range.
  let e = ColorCorrectionMatrix::try_new([[1.0, -1e10, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    .unwrap_err();
  assert!(matches!(
    e,
    ColorCorrectionMatrixError::OutOfBounds { row: 0, col: 1, .. }
  ));
}

#[test]
fn ccm_try_new_accepts_typical_negative_off_diagonal() {
  // Real-world CCM with crosstalk subtraction stays well within
  // the bound and validates cleanly.
  ColorCorrectionMatrix::try_new([[1.5, -0.3, -0.2], [-0.1, 1.2, -0.1], [-0.05, -0.15, 1.2]])
    .expect("typical CCM valid");
}

/// Codex regression: even at the bound, fusion + per-pixel
/// matmul stays finite for the maximum-stress 16-bit input.
/// `WB.MAX_GAIN * CCM.MAX_COEFFICIENT_ABS * 65535 ≈ 6.55e16`,
/// well under `f32::MAX ≈ 3.4e38`.
#[test]
fn fuse_wb_ccm_at_bounds_with_max_sample_stays_finite() {
  let wb = WhiteBalance::try_new(
    WhiteBalance::MAX_GAIN,
    WhiteBalance::MAX_GAIN,
    WhiteBalance::MAX_GAIN,
  )
  .unwrap();
  let max = ColorCorrectionMatrix::MAX_COEFFICIENT_ABS;
  let ccm =
    ColorCorrectionMatrix::try_new([[max, max, max], [max, max, max], [max, max, max]]).unwrap();
  let m = fuse_wb_ccm(&wb, &ccm);
  // Worst-case per-pixel sum: 3 channels * fused_max * 65535.
  let sample = 65535.0f32;
  for row in m.iter() {
    let s = (row[0] + row[1] + row[2]) * sample;
    assert!(s.is_finite(), "per-pixel sum overflowed at bound: {s}");
  }
}
