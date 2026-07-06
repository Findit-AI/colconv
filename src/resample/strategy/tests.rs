use super::*;

/// The tier an output-bearing `Yuv420p` row runs through, as the
/// **pre-RFC-#238 inline dispatch** decided it (see `planar_8bit.rs`'s
/// `Yuv420p` `process`). This is the reference oracle the selector must
/// reproduce: the filter-first branch, then the `with_native` boolean,
/// reading exactly the same inputs the production dispatch read. The
/// route-rejection (`frozen` / `need_output`) machinery does **not**
/// participate in *which tier* runs — it gates rejection and freezing —
/// so it is intentionally absent here and exercised independently below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OldTier {
  /// `yuv420p_process_native` — the native fast tier.
  Native,
  /// `yuv420p_process_resampled` — the row-stage tier.
  RowStage,
  /// `planar_dual_filter_resample` — the filter resampler, taken before
  /// the native/row-stage route machinery on a filter plan.
  Filter,
}

/// Reproduces the old inline tier choice verbatim. `Yuv420p` is always
/// native-eligible, so eligibility is implicit in the original code; it
/// is threaded through here so the equivalence is asserted across the
/// eligibility axis too.
fn old_tier(native_eligible: bool, with_native: bool, area_plan: bool) -> OldTier {
  // `if plan.kind().is_filter()` — the filter resampler runs first,
  // before any native/row-stage decision.
  if !area_plan {
    return OldTier::Filter;
  }
  // `if *native { yuv420p_process_native } else { yuv420p_process_resampled }`.
  // The original Yuv420p code has no `native_eligible` guard because the
  // format is statically eligible; an ineligible format would never reach
  // the native tier, i.e. behaves as the row-stage branch.
  if native_eligible && with_native {
    OldTier::Native
  } else {
    OldTier::RowStage
  }
}

/// Map the old tier onto the splice stage it corresponds to, for the
/// equivalence assertion. A filter plan is not a native-vs-row-stage
/// choice at all; the native tier never runs on it, so the selector must
/// keep it off the native codes (`EncodedOutput`).
fn expected_point(tier: OldTier) -> InsertionPoint {
  match tier {
    OldTier::Native => InsertionPoint::NativeCodes,
    OldTier::RowStage | OldTier::Filter => InsertionPoint::EncodedOutput,
  }
}

#[test]
fn selector_reproduces_old_take_native_decision_exhaustively() {
  // Every combination of the inputs the old dispatch branched on, plus
  // the orthogonal route-machinery axes (`need_output`, frozen-route
  // state) that must NOT perturb the splice choice. `frozen_route`:
  // `None` = unfrozen, `Some(true)` = frozen-to-native, `Some(false)` =
  // frozen-to-row-stage.
  for native_eligible in [false, true] {
    for with_native in [false, true] {
      for area_plan in [false, true] {
        for need_output in [false, true] {
          for frozen_route in [None, Some(true), Some(false)] {
            let ctx = InsertionContext {
              native_eligible,
              with_native,
              area_plan,
            };
            let got = select_insertion_point(AveragingDomain::Encoded, ctx);
            let expected = expected_point(old_tier(native_eligible, with_native, area_plan));
            assert_eq!(
              got, expected,
              "splice mismatch for native_eligible={native_eligible} \
               with_native={with_native} area_plan={area_plan} \
               need_output={need_output} frozen_route={frozen_route:?}",
            );
            // The route-machinery axes are pure pass-through: they change
            // nothing about which stage the resample splices at, which is
            // exactly why re-expressing the dispatch is byte-identical.
            let _ = (need_output, frozen_route);
          }
        }
      }
    }
  }
}

#[test]
fn phase1_rollout_native_eligible_formats_reproduce_if_native_boolean() {
  // RFC #238 Phase 1 routed every native-bearing format (8-bit + high-bit
  // planar 4:2:0 / 4:2:2 / 4:4:4 / 4:4:0) onto this selector with a
  // `*_NATIVE_ELIGIBLE = true` const, replacing an inline
  // `if *native { native } else { row-stage }` on an area plan (a filter plan
  // already returned before the dispatch, so the routed call sites always
  // pass `area_plan: true`). The re-expression is byte-identical iff, for
  // those `native_eligible: true, area_plan: true` call sites, the selector
  // returns `NativeCodes` EXACTLY when the old `*native` boolean was true —
  // i.e. it equals `with_native`. This pins that Phase 1 contract directly,
  // distinct from the generic Yuv420p oracle above.
  for with_native in [false, true] {
    let ctx = InsertionContext {
      native_eligible: true,
      with_native,
      area_plan: true,
    };
    let got = select_insertion_point(AveragingDomain::Encoded, ctx);
    let expected_native = with_native; // the former `if *native` branch
    assert_eq!(
      got == InsertionPoint::NativeCodes,
      expected_native,
      "Phase 1 splice mismatch: native_eligible=true area_plan=true \
       with_native={with_native} must map NativeCodes<=>with_native",
    );
  }
}

#[test]
fn encoded_area_native_eligible_with_native_selects_native_codes() {
  // The one combination that must splice at the native codes — the
  // affine-format, area-downscale, native-enabled-and-eligible case.
  let ctx = InsertionContext {
    native_eligible: true,
    with_native: true,
    area_plan: true,
  };
  assert_eq!(
    select_insertion_point(AveragingDomain::Encoded, ctx),
    InsertionPoint::NativeCodes,
  );
}

#[test]
fn encoded_disabling_native_forces_encoded_output() {
  let ctx = InsertionContext {
    native_eligible: true,
    with_native: false,
    area_plan: true,
  };
  assert_eq!(
    select_insertion_point(AveragingDomain::Encoded, ctx),
    InsertionPoint::EncodedOutput,
  );
}

#[test]
fn encoded_filter_plan_never_splices_at_native_codes() {
  // A filter plan routes to the filter resampler before the route
  // machinery; the native tier never runs, so the splice stays at the
  // output regardless of `with_native`.
  for with_native in [false, true] {
    let ctx = InsertionContext {
      native_eligible: true,
      with_native,
      area_plan: false,
    };
    assert_eq!(
      select_insertion_point(AveragingDomain::Encoded, ctx),
      InsertionPoint::EncodedOutput,
    );
  }
}

#[test]
fn ineligible_format_never_splices_at_native_codes() {
  for with_native in [false, true] {
    for area_plan in [false, true] {
      let ctx = InsertionContext {
        native_eligible: false,
        with_native,
        area_plan,
      };
      assert_eq!(
        select_insertion_point(AveragingDomain::Encoded, ctx),
        InsertionPoint::EncodedOutput,
      );
    }
  }
}

#[test]
fn linear_domain_resolves_to_linear_light() {
  // RFC #238 Phase 2: the Linear domain splices at the linear-light stage,
  // independent of the native-tier inputs (the linear average is its own
  // splice, not a native fast-tier variant).
  for native_eligible in [false, true] {
    for with_native in [false, true] {
      for area_plan in [false, true] {
        let ctx = InsertionContext {
          native_eligible,
          with_native,
          area_plan,
        };
        assert_eq!(
          select_insertion_point(AveragingDomain::Linear, ctx),
          InsertionPoint::LinearLight,
          "Linear must splice at LinearLight regardless of native inputs",
        );
      }
    }
  }
}

#[test]
#[should_panic(expected = "Premultiplied is rejected at dispatch")]
fn premultiplied_domain_has_no_splice_and_is_unreachable() {
  // Premultiplied is a reserved future-phase domain with no valid insertion
  // point yet. The selector must NOT silently resolve it to the encoded
  // output (a different domain) — every caller rejects it before the
  // selector, so reaching this arm is a routing bug. Asserting the panic
  // (rather than a returned splice) pins the honesty contract: there is no
  // legitimate Premultiplied→Encoded route here.
  let ctx = InsertionContext {
    native_eligible: true,
    with_native: true,
    area_plan: true,
  };
  let _ = select_insertion_point(AveragingDomain::Premultiplied, ctx);
}

#[test]
fn averaging_domain_as_str_round_trips_variants() {
  assert_eq!(AveragingDomain::Encoded.as_str(), "encoded");
  assert_eq!(AveragingDomain::Linear.as_str(), "linear");
  assert_eq!(AveragingDomain::Premultiplied.as_str(), "premultiplied");
}

#[test]
fn resample_strategy_default_is_encoded_area() {
  let strat = ResampleStrategy::default();
  assert_eq!(strat.domain(), AveragingDomain::Encoded);
  assert_eq!(strat.filter(), FilterSpec::Area);
}

#[test]
fn transfer_function_default_is_bt1886() {
  assert_eq!(TransferFunction::default(), TransferFunction::Bt1886);
}

#[test]
fn transfer_function_as_str_round_trips_variants() {
  assert_eq!(TransferFunction::LinearPassthrough.as_str(), "linear");
  assert_eq!(TransferFunction::Srgb.as_str(), "srgb");
  assert_eq!(TransferFunction::Bt1886.as_str(), "bt1886");
  assert_eq!(TransferFunction::Gamma22.as_str(), "gamma22");
}

#[test]
fn transfer_function_eotf_oetf_are_inverses() {
  // EOTF∘OETF and OETF∘EOTF round-trip to the identity within f32 epsilon
  // across the unit interval — the property the Linear domain relies on to
  // stay close to the encoded path when chroma is flat.
  for tf in [
    TransferFunction::LinearPassthrough,
    TransferFunction::Srgb,
    TransferFunction::Bt1886,
    TransferFunction::Gamma22,
  ] {
    for i in 0..=256 {
      let c = i as f32 / 256.0;
      let round = tf.oetf(tf.eotf(c));
      assert!(
        (round - c).abs() <= 2e-4,
        "{}: oetf(eotf({c})) = {round}, want {c}",
        tf.as_str(),
      );
      let round2 = tf.eotf(tf.oetf(c));
      assert!(
        (round2 - c).abs() <= 2e-4,
        "{}: eotf(oetf({c})) = {round2}, want {c}",
        tf.as_str(),
      );
    }
  }
}

#[test]
fn transfer_function_endpoints_are_fixed() {
  // 0 and 1 map to themselves under every curve (the gamut endpoints must
  // not drift).
  for tf in [
    TransferFunction::LinearPassthrough,
    TransferFunction::Srgb,
    TransferFunction::Bt1886,
    TransferFunction::Gamma22,
  ] {
    assert!(tf.eotf(0.0).abs() <= 1e-6, "{}: eotf(0)", tf.as_str());
    assert!(
      (tf.eotf(1.0) - 1.0).abs() <= 1e-6,
      "{}: eotf(1)",
      tf.as_str()
    );
    assert!(tf.oetf(0.0).abs() <= 1e-6, "{}: oetf(0)", tf.as_str());
    assert!(
      (tf.oetf(1.0) - 1.0).abs() <= 1e-6,
      "{}: oetf(1)",
      tf.as_str()
    );
  }
}

#[test]
fn transfer_function_curves_are_distinct() {
  // A mid-tone linearises differently under each curve, so the caller's
  // choice is observable (the property `transfer_function_caller_override`
  // exercises end-to-end).
  let c = 0.5_f32;
  let srgb = TransferFunction::Srgb.eotf(c);
  let bt1886 = TransferFunction::Bt1886.eotf(c);
  let g22 = TransferFunction::Gamma22.eotf(c);
  let lin = TransferFunction::LinearPassthrough.eotf(c);
  assert!((srgb - bt1886).abs() > 1e-3, "sRGB vs BT.1886 must differ");
  assert!(
    (bt1886 - g22).abs() > 1e-3,
    "BT.1886 vs gamma2.2 must differ"
  );
  assert!(
    (lin - bt1886).abs() > 1e-3,
    "passthrough vs BT.1886 must differ"
  );
}

#[test]
fn transfer_function_for_matrix_default_mapping() {
  use crate::ColorMatrix;
  // The sRGB identity (GBR) pairs with the sRGB curve.
  assert_eq!(
    TransferFunction::for_matrix(ColorMatrix::Rgb),
    TransferFunction::Srgb,
  );
  // Every YCbCr video matrix resolves to the BT.1886 display EOTF.
  for matrix in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Bt2020Cl,
    ColorMatrix::Smpte170M,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::Bt470Bg,
    ColorMatrix::YCgCo,
    ColorMatrix::Unspecified,
    ColorMatrix::Unknown(99),
  ] {
    assert_eq!(
      TransferFunction::for_matrix(matrix),
      TransferFunction::Bt1886,
      "{} must resolve to BT.1886",
      matrix.as_str(),
    );
  }
}

// ---- ITU-R BT.2100 PQ / HLG transfer functions --------------------------
//
// These exercise the private `transfer::pq_hlg` free functions (the BT.2100
// inverse-EOTF math the deferred ICtCp matrix-wiring (#303) will consume).
// The reference values are taken from the `colour-science` Python library
// docstring examples (`eotf_ST2084` / `eotf_inverse_ST2084`,
// `oetf_ARIBSTDB67`) and the exact construction anchors of ITU-R BT.2100-2
// (Tables 4 / 5). They independently pin every PQ constant (m1, m2, c1, c2,
// c3) and HLG constant (a, b, c) — a transcription error in any of them
// moves these points well outside the stated tolerances.

#[test]
fn transfer_function_pq_matches_st2084_reference() {
  use super::transfer::pq_hlg::{pq_eotf, pq_oetf};

  // BT.2100 anchors: signal 1.0 ↔ linear 1.0 (= 10 000 cd/m²), signal 0 ↔
  // linear 0. The signal-1.0 fixed point holds iff c1 = c3 − c2 + 1.
  assert!(
    (pq_eotf(1.0) - 1.0).abs() <= 1e-6,
    "eotf(1.0)={}",
    pq_eotf(1.0)
  );
  assert!(pq_eotf(0.0).abs() <= 1e-9, "eotf(0.0)={}", pq_eotf(0.0));
  assert!(
    (pq_oetf(1.0) - 1.0).abs() <= 1e-5,
    "oetf(1.0)={}",
    pq_oetf(1.0)
  );

  // colour-science `eotf_inverse_ST2084(100)` = 0.5080784: 100 cd/m² is the
  // normalised linear 0.01, encoded to signal 0.5080784. This single point
  // pins all five PQ constants in the encode direction.
  const SIGNAL_100_NITS: f32 = 0.508_078_4;
  assert!(
    (pq_oetf(0.01) - SIGNAL_100_NITS).abs() <= 5e-4,
    "oetf(0.01)={} want {SIGNAL_100_NITS}",
    pq_oetf(0.01),
  );
  // colour-science `eotf_ST2084(0.508078421517399)` = 100 cd/m²: the decode
  // direction of the same anchor (normalised 0.01).
  assert!(
    (pq_eotf(SIGNAL_100_NITS) - 0.01).abs() <= 1e-4,
    "eotf({SIGNAL_100_NITS})={} want 0.01",
    pq_eotf(SIGNAL_100_NITS),
  );

  // PQ's inverse-EOTF maps linear 0 to a small non-zero signal `c1^m2`
  // (≈ 7.3e-7), a known property of the curve — assert it stays negligible.
  assert!(pq_oetf(0.0).abs() <= 1e-5, "oetf(0.0)={}", pq_oetf(0.0));
}

#[test]
fn transfer_function_eotf_super_white_saturates() {
  use super::transfer::pq_hlg::{hlg_eotf, pq_eotf};

  // PQ signal E' is defined on [0, 1]; a super-white magnitude saturates at
  // the peak (linear 1.0) — it must NOT cross the den=0 pole at |c| ~= 1.99
  // (which overflows toward +inf just below it and folds to black just above
  // it). Pins values straddling the pole.
  for &c in &[1.0_f32, 1.5, 1.99, 2.0, 10.0] {
    let y = pq_eotf(c);
    assert!(y.is_finite(), "pq_eotf({c})={y} must be finite");
    assert!(
      (y - 1.0).abs() <= 1e-6,
      "pq_eotf({c})={y} must saturate at 1.0 (not inf/black)"
    );
  }
  // Odd extension: super-black saturates at -1.0.
  assert!(
    (pq_eotf(-2.0) + 1.0).abs() <= 1e-6,
    "pq_eotf(-2.0)={}",
    pq_eotf(-2.0)
  );

  // HLG likewise clamps the [0, 1] signal domain (its log segment would grow
  // unbounded for E' > 1); a super-white input equals the peak hlg_eotf(1.0).
  let hlg_peak = hlg_eotf(1.0);
  for &c in &[1.0_f32, 2.0, 10.0] {
    let e = hlg_eotf(c);
    assert!(
      e.is_finite() && (e - hlg_peak).abs() <= 1e-6,
      "hlg_eotf({c})={e} must saturate at hlg_eotf(1.0)={hlg_peak}"
    );
  }
}

#[test]
fn transfer_function_hlg_matches_bt2100_reference() {
  use super::transfer::pq_hlg::{hlg_eotf, hlg_oetf};

  // BT.2100 breakpoint: scene-linear 1/12 (HLG reference white) encodes to
  // signal 0.5, and the lower (gamma) segment is exact there.
  assert!(
    (hlg_oetf(1.0 / 12.0) - 0.5).abs() <= 1e-6,
    "oetf(1/12)={}",
    hlg_oetf(1.0 / 12.0),
  );
  // Inverse breakpoint: signal 0.5 decodes back to scene-linear 1/12.
  assert!(
    (hlg_eotf(0.5) - 1.0 / 12.0).abs() <= 1e-6,
    "eotf(0.5)={}",
    hlg_eotf(0.5),
  );

  // colour-science `oetf_ARIBSTDB67(0.18)` = 0.2121320. Under the BT.2100
  // normalisation E = E_arib / 12, that is `oetf(0.015)` — a lower (gamma)
  // segment point: sqrt(3·0.015) = 0.2121320.
  assert!(
    (hlg_oetf(0.015) - 0.212_132).abs() <= 1e-5,
    "oetf(0.015)={}",
    hlg_oetf(0.015),
  );
  assert!(
    (hlg_eotf(0.212_132) - 0.015).abs() <= 1e-5,
    "eotf(0.2121320)={}",
    hlg_eotf(0.212_132),
  );

  // Upper (log) segment construction anchors — these exercise a, b and c:
  // scene-linear 1.0 ↔ signal 1.0 (c is defined to make this hold), and
  // scene-linear 0 ↔ signal 0.
  assert!(
    (hlg_oetf(1.0) - 1.0).abs() <= 1e-4,
    "oetf(1.0)={}",
    hlg_oetf(1.0)
  );
  assert!(
    (hlg_eotf(1.0) - 1.0).abs() <= 1e-4,
    "eotf(1.0)={}",
    hlg_eotf(1.0)
  );
  assert!(hlg_oetf(0.0).abs() <= 1e-9, "oetf(0.0)={}", hlg_oetf(0.0));
  assert!(hlg_eotf(0.0).abs() <= 1e-9, "eotf(0.0)={}", hlg_eotf(0.0));
}

#[test]
fn transfer_function_pq_hlg_round_trip() {
  use super::transfer::pq_hlg::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};

  // EOTF and OETF are exact analytic inverses; verify the f32 round-trip
  // over the unit signal interval. (The round-trip is insensitive to the
  // constants' absolute values — the `*_matches_*_reference` tests pin
  // those — so this guards only the inverse relationship.)
  for i in 0..=256 {
    let c = i as f32 / 256.0;
    let pq = pq_oetf(pq_eotf(c));
    assert!((pq - c).abs() <= 2e-3, "pq oetf(eotf({c}))={pq}");
    let pq2 = pq_eotf(pq_oetf(c));
    assert!((pq2 - c).abs() <= 2e-3, "pq eotf(oetf({c}))={pq2}");
    let hlg = hlg_oetf(hlg_eotf(c));
    assert!((hlg - c).abs() <= 1e-3, "hlg oetf(eotf({c}))={hlg}");
    let hlg2 = hlg_eotf(hlg_oetf(c));
    assert!((hlg2 - c).abs() <= 1e-3, "hlg eotf(oetf({c}))={hlg2}");
  }
}

// miri's interpreted floating-point diverges from hardware for the PQ/HLG
// transcendentals past this test's tolerance.
#[cfg(not(miri))]
#[test]
fn transfer_function_pq_hlg_odd_extension() {
  use super::transfer::pq_hlg::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};

  // The per-channel curves are odd (`f(-c) == -f(c)`) so super-black
  // excursions linearise symmetrically rather than folding — the same
  // contract the SDR curves honour.
  for &c in &[0.05_f32, 0.25, 0.6, 0.9] {
    assert!(
      (pq_eotf(-c) + pq_eotf(c)).abs() <= 1e-6,
      "pq eotf odd at {c}"
    );
    assert!(
      (pq_oetf(-c) + pq_oetf(c)).abs() <= 1e-6,
      "pq oetf odd at {c}"
    );
    assert!(
      (hlg_eotf(-c) + hlg_eotf(c)).abs() <= 1e-6,
      "hlg eotf odd at {c}"
    );
    assert!(
      (hlg_oetf(-c) + hlg_oetf(c)).abs() <= 1e-6,
      "hlg oetf odd at {c}"
    );
  }
}

#[test]
fn transfer_function_pq_hlg_distinct_from_sdr() {
  use super::transfer::pq_hlg::{hlg_eotf, pq_eotf};

  // A mid-tone linearises differently under the HDR curves than the SDR
  // BT.1886 default, so the BT.2100 decode is observably distinct.
  let c = 0.5_f32;
  let bt1886 = TransferFunction::Bt1886.eotf(c);
  let pq = pq_eotf(c);
  let hlg = hlg_eotf(c);
  assert!((pq - bt1886).abs() > 1e-3, "PQ vs BT.1886 must differ");
  assert!((hlg - bt1886).abs() > 1e-3, "HLG vs BT.1886 must differ");
  assert!((pq - hlg).abs() > 1e-3, "PQ vs HLG must differ");
}
