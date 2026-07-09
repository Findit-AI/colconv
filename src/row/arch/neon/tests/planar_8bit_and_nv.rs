use super::super::*;

/// Deterministic scalar‑equivalence fixture. Fills Y/U/V with a
/// hash‑like sequence so every byte varies, then compares byte‑exact.
#[cfg(feature = "yuv-planar")]
fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_width_32() {
  check_equivalence(32, ColorMatrix::Bt601, true);
  check_equivalence(32, ColorMatrix::Bt709, false);
  check_equivalence(32, ColorMatrix::YCgCo, true);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_width_1920() {
  check_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_odd_tail_widths() {
  // Widths that leave a non‑trivial scalar tail (non‑multiple of 16).
  for w in [18usize, 30, 34, 1922] {
    check_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- yuv_420_to_rgba_row equivalence --------------------------------
//
// Direct backend test for the new RGBA path: bypasses the public
// dispatcher so the NEON `vst4q_u8` write is exercised regardless
// of what tier the dispatcher would pick on the current runner.
// Catches lane-order or alpha-splat corruption in `vst4q_u8` that
// a dispatcher-routed test on a different host would miss.

#[cfg(feature = "yuv-planar")]
fn check_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_width_32() {
  check_rgba_equivalence(32, ColorMatrix::Bt601, true);
  check_rgba_equivalence(32, ColorMatrix::Bt709, false);
  check_rgba_equivalence(32, ColorMatrix::YCgCo, true);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_width_1920() {
  check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- nv12_to_rgb_row equivalence ------------------------------------

#[cfg(feature = "yuv-semi-planar")]
/// Scalar‑equivalence fixture for NV12. Builds an interleaved UV row
/// from the same U/V byte sequences used by the yuv420p fixture so a
/// single NV12 call should produce byte‑identical output to the
/// scalar NV12 reference.
fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| {
      [
        ((i * 53 + 23) & 0xFF) as u8, // U_i
        ((i * 71 + 91) & 0xFF) as u8, // V_i
      ]
    })
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON NV12 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
/// Cross-format equivalence: the NV12 output must match the YUV420P
/// output when fed the same U / V bytes interleaved. Guards against
/// any stray deinterleave bug.
fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

  let mut rgb_yuv420p = std::vec![0u8; width * 3];
  let mut rgb_nv12 = std::vec![0u8; width * 3];
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
    nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
  }
  assert_eq!(
    rgb_yuv420p, rgb_nv12,
    "NV12 and YUV420P must produce byte-identical output for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv12_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_scalar_width_1920() {
  check_nv12_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_yuv420p_neon() {
  for w in [16usize, 30, 64, 1920] {
    check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
    check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv21_to_rgb_row equivalence ------------------------------------

#[cfg(feature = "yuv-semi-planar")]
/// Scalar-equivalence for NV21. Same pseudo-random byte stream as
/// the NV12 fixture, just handed to the VU-ordered kernel.
fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| {
      [
        ((i * 53 + 23) & 0xFF) as u8, // V_i
        ((i * 71 + 91) & 0xFF) as u8, // U_i
      ]
    })
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgb_row(&y, &vu, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON NV21 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
/// Cross-format invariant: NV21 kernel on a VU-swapped byte stream
/// must produce byte-identical output to the NV12 kernel on the
/// UV-ordered original — proves the const-generic `SWAP_UV` path
/// actually inverts the byte order.
fn check_nv21_matches_nv12_with_swapped_uv(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  // Build the UV stream (NV12 order), then the VU stream as the
  // same pairs byte-swapped.
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| {
      [
        ((i * 53 + 23) & 0xFF) as u8, // U_i
        ((i * 71 + 91) & 0xFF) as u8, // V_i
      ]
    })
    .collect();
  let mut vu = std::vec![0u8; width];
  for i in 0..width / 2 {
    vu[2 * i] = uv[2 * i + 1]; // V_i
    vu[2 * i + 1] = uv[2 * i]; // U_i
  }

  let mut rgb_nv12 = std::vec![0u8; width * 3];
  let mut rgb_nv21 = std::vec![0u8; width * 3];
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
  }
  assert_eq!(
    rgb_nv12, rgb_nv21,
    "NV21 should produce identical output to NV12 with byte-swapped chroma (width={width}, matrix={matrix:?})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv21_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_matches_scalar_widths() {
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_matches_nv12_swapped() {
  for w in [16usize, 30, 64, 1920] {
    check_nv21_matches_nv12_with_swapped_uv(w, ColorMatrix::Bt709, false);
    check_nv21_matches_nv12_with_swapped_uv(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv12_to_rgba_row / nv21_to_rgba_row equivalence ----------------
//
// Direct backend tests for the new RGBA path, mirroring the RGB
// pattern above. Bypasses the dispatcher so the NEON `vst4q_u8`
// store is exercised regardless of what tier the dispatcher picks.

#[cfg(feature = "yuv-semi-planar")]
fn check_nv12_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::nv12_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgba_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON NV12 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
fn check_nv21_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::nv21_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgba_row(&y, &vu, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON NV21 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
/// Cross-format invariant: NV12 RGBA must match Yuv420p RGBA on
/// equivalent UV bytes. Catches U/V swap regressions specific to
/// the new RGBA store path.
fn check_nv12_rgba_matches_yuv420p_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

  let mut rgba_yuv420p = std::vec![0u8; width * 4];
  let mut rgba_nv12 = std::vec![0u8; width * 4];
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_yuv420p, width, matrix, full_range);
    nv12_to_rgba_row(&y, &uv, &mut rgba_nv12, width, matrix, full_range);
  }
  assert_eq!(
    rgba_yuv420p, rgba_nv12,
    "NEON NV12 RGBA must match Yuv420p RGBA for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv12_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_rgba_matches_yuv420p_rgba_neon() {
  for w in [16usize, 30, 64, 1920] {
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::Bt709, false);
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::YCgCo, true);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv21_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_nv21_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

#[cfg(feature = "yuv-semi-planar")]
fn check_nv24_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  // NV24: 1 UV pair per Y pixel → 2*width bytes.
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| {
      [
        ((i * 53 + 23) & 0xFF) as u8, // U_i
        ((i * 71 + 91) & 0xFF) as u8, // V_i
      ]
    })
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgb_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON NV24 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  // NV42: V first, then U (byte-swapped).
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| {
      [
        ((i * 53 + 23) & 0xFF) as u8, // V_i
        ((i * 71 + 91) & 0xFF) as u8, // U_i
      ]
    })
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgb_row(&y, &vu, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON NV42 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
/// NV42 kernel on a byte-swapped UV stream must match NV24 on the
/// original — validates the `SWAP_UV` const generic.
fn check_nv42_matches_nv24_with_swapped_uv(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut vu = std::vec![0u8; 2 * width];
  for i in 0..width {
    vu[2 * i] = uv[2 * i + 1];
    vu[2 * i + 1] = uv[2 * i];
  }

  let mut rgb_nv24 = std::vec![0u8; width * 3];
  let mut rgb_nv42 = std::vec![0u8; width * 3];
  unsafe {
    nv24_to_rgb_row(&y, &uv, &mut rgb_nv24, width, matrix, full_range);
    nv42_to_rgb_row(&y, &vu, &mut rgb_nv42, width, matrix, full_range);
  }
  assert_eq!(
    rgb_nv24, rgb_nv42,
    "NV42 should produce identical output to NV24 with byte-swapped chroma (width={width}, matrix={matrix:?})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv24_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv24_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv24_neon_matches_scalar_widths() {
  // Odd widths validate the no-parity-constraint contract (NV24 is
  // 4:4:4, no chroma pairing) and force non-multiple-of-16 scalar
  // tails.
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv24_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv42_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv42_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_matches_nv24_swapped() {
  for w in [16usize, 17, 33, 64, 1920] {
    check_nv42_matches_nv24_with_swapped_uv(w, ColorMatrix::Bt709, false);
    check_nv42_matches_nv24_with_swapped_uv(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv24_to_rgba_row / nv42_to_rgba_row equivalence ----------------

#[cfg(feature = "yuv-semi-planar")]
fn check_nv24_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::nv24_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgba_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON NV24 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
fn check_nv42_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::nv42_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgba_row(&y, &vu, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON NV42 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv24_neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv24_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv24_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv42_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv42_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgb_row equivalence ---------------------------------

#[cfg(feature = "yuv-planar")]
fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON yuv_444 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_444_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_444_neon_matches_scalar_widths() {
  // Odd widths validate the no-parity-constraint contract (4:4:4
  // chroma is 1:1 with Y, not paired) and force non-multiple-of-16
  // scalar tails.
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgba_row equivalence --------------------------------

#[cfg(feature = "yuv-planar")]
fn check_yuv_444_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "NEON yuv_444 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_444_neon_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_rgba_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_444_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuva")]
fn check_yuv_444_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let a_src: std::vec::Vec<u8> = (0..width)
    .map(|i| ((i * alpha_seed + 17) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_with_alpha_src_row(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444_to_rgba_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_neon,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Yuva444p → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_rgba_matches_scalar_widths_and_alpha() {
  for w in [16usize, 17, 31, 47, 1920, 1922] {
    check_yuv_444_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv_444_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, false, seed);
  }
}

// ---- rgb_to_hsv_row equivalence ------------------------------------
//
// The NEON HSV kernel uses `vmaxq_f32` / `vminq_f32` / `vdivq_f32`
// (true f32 ops). Miri's interpreter does not currently implement
// these aarch64 NEON f32 intrinsics — under
// `cargo miri test --target aarch64-unknown-linux-gnu` the calls
// raise `unsupported operation: can't call foreign function
// "llvm.aarch64.neon.fmax.v4f32"`. The previous
// `#[cfg_attr(miri, ignore = ...)]` annotations didn't suffice in
// CI (Miri tried to evaluate them anyway). Compiling the helper
// and the tests *out* under `cfg(miri)` removes the f32
// intrinsics from the binary entirely so Miri can't trip on them.
// The other backends (wasm / x86) are tested by their respective
// arch modules; correctness of the NEON HSV path is still covered
// by host-arch CI runs that don't go through Miri.

#[cfg(not(miri))]
fn check_hsv_equivalence(rgb: &[u8], width: usize) {
  let mut h_scalar = std::vec![0u8; width];
  let mut s_scalar = std::vec![0u8; width];
  let mut v_scalar = std::vec![0u8; width];
  let mut h_neon = std::vec![0u8; width];
  let mut s_neon = std::vec![0u8; width];
  let mut v_neon = std::vec![0u8; width];

  scalar::rgb_to_hsv_row(rgb, &mut h_scalar, &mut s_scalar, &mut v_scalar, width);
  unsafe {
    rgb_to_hsv_row(rgb, &mut h_neon, &mut s_neon, &mut v_neon, width);
  }

  // Scalar uses integer LUT (matches OpenCV byte-exact), NEON uses
  // true f32 division. They can disagree by ±1 LSB at boundary
  // pixels — identical tolerance to what OpenCV reports between
  // their own scalar and SIMD HSV paths. Hue uses *circular*
  // distance since 0 and 179 are neighbors on the hue wheel: a pixel
  // at 360°≈0 in one path can land at 358°≈179 in the other due to
  // sign flips in delta with tiny f32 rounding.
  for (i, (a, b)) in h_scalar.iter().zip(h_neon.iter()).enumerate() {
    let d = a.abs_diff(*b);
    let circ = d.min(180 - d);
    assert!(circ <= 1, "H divergence at pixel {i}: scalar={a} neon={b}");
  }
  for (i, (a, b)) in s_scalar.iter().zip(s_neon.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "S divergence at pixel {i}: scalar={a} neon={b}"
    );
  }
  for (i, (a, b)) in v_scalar.iter().zip(v_neon.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "V divergence at pixel {i}: scalar={a} neon={b}"
    );
  }
}

fn pseudo_random_bgr(width: usize) -> std::vec::Vec<u8> {
  let n = width * 3;
  let mut out = std::vec::Vec::with_capacity(n);
  let mut state: u32 = 0x9E37_79B9;
  for _ in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((state >> 8) as u8);
  }
  out
}

#[test]
#[cfg(not(miri))]
fn hsv_neon_matches_scalar_pseudo_random_16() {
  let rgb = pseudo_random_bgr(16);
  check_hsv_equivalence(&rgb, 16);
}

#[test]
#[cfg(not(miri))]
fn hsv_neon_matches_scalar_pseudo_random_1920() {
  let rgb = pseudo_random_bgr(1920);
  check_hsv_equivalence(&rgb, 1920);
}

#[test]
#[cfg(not(miri))]
fn hsv_neon_matches_scalar_tail_widths() {
  // Widths that force a non‑trivial scalar tail (non‑multiple of 16).
  for w in [1usize, 7, 15, 17, 31, 1921] {
    let rgb = pseudo_random_bgr(w);
    check_hsv_equivalence(&rgb, w);
  }
}

#[test]
#[cfg(not(miri))]
fn hsv_neon_matches_scalar_primaries_and_edges() {
  // Primary colors, grays, near‑saturation — exercise each hue branch
  // and the v==0, delta==0, h<0 wrap paths.
  let rgb: std::vec::Vec<u8> = [
    (0, 0, 0),       // black: v = 0 → s = 0, h = 0
    (255, 255, 255), // white: delta = 0 → s = 0, h = 0
    (128, 128, 128), // gray: delta = 0
    (255, 0, 0),     // pure red: v == r path
    (0, 255, 0),     // pure green: v == g path
    (0, 0, 255),     // pure blue: v == b path
    (255, 127, 0),   // red→yellow transition
    (0, 127, 255),   // blue→cyan
    (255, 0, 127),   // red→magenta
    (1, 2, 3),       // near black: small delta
    (254, 253, 252), // near white
    (150, 200, 10),  // arbitrary: v == g path, h > 0
    (150, 10, 200),  // arbitrary: v == b path
    (10, 200, 150),  // arbitrary: v == g
    (200, 100, 50),  // arbitrary: v == r
    (0, 64, 128),    // arbitrary: v == b
  ]
  .iter()
  .flat_map(|&(r, g, b)| [r, g, b])
  .collect();
  check_hsv_equivalence(&rgb, 16);
}

// ---- bgr_rgb_swap_row equivalence -----------------------------------

#[cfg(feature = "rgb")]
fn check_swap_equivalence(width: usize) {
  let input = pseudo_random_bgr(width);
  let mut out_scalar = std::vec![0u8; width * 3];
  let mut out_neon = std::vec![0u8; width * 3];

  scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
  unsafe {
    bgr_rgb_swap_row(&input, &mut out_neon, width);
  }

  assert_eq!(out_scalar, out_neon, "NEON swap diverges from scalar");

  // Byte 0 ↔ byte 2 should be swapped, byte 1 unchanged. Verify
  // the semantic directly.
  for x in 0..width {
    assert_eq!(
      out_scalar[x * 3],
      input[x * 3 + 2],
      "byte 0 != input byte 2"
    );
    assert_eq!(
      out_scalar[x * 3 + 1],
      input[x * 3 + 1],
      "middle byte changed"
    );
    assert_eq!(
      out_scalar[x * 3 + 2],
      input[x * 3],
      "byte 2 != input byte 0"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn swap_neon_matches_scalar_widths() {
  for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
    check_swap_equivalence(w);
  }
}

#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn swap_is_self_inverse() {
  let input = pseudo_random_bgr(64);
  let mut round_trip = std::vec![0u8; 64 * 3];
  let mut back = std::vec![0u8; 64 * 3];

  scalar::bgr_rgb_swap_row(&input, &mut round_trip, 64);
  scalar::bgr_rgb_swap_row(&round_trip, &mut back, 64);

  assert_eq!(input, back, "swap is not self-inverse");
}

// ---- yuv_410_to_rgb_row equivalence ---------------------------------
//
// 4:1:0 NEON parity with scalar — same Q15 sequence, just 4× chroma
// duplication instead of 2×. Width must be a multiple of 4 (so the
// scalar tail receives an aligned input).

#[cfg(feature = "yuv-planar")]
fn check_yuv_410_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let cw = width / 4;
  let u: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON yuv_410 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_410_neon_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_410_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_410_neon_matches_scalar_widths() {
  // Multiples of 4 — including pure scalar (4, 8, 12), exactly one
  // SIMD iter (16), SIMD + tail (20, 28), SIMD + multi-iter (32),
  // larger frames (64, 1920).
  for &w in &[4usize, 8, 12, 16, 20, 28, 32, 64, 128, 1920] {
    check_yuv_410_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv_410_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-planar")]
fn check_yuv_410_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let cw = width / 4;
  let u: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::yuv_410_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_410_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }

  if rgba_scalar != rgba_neon {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON yuv_410 RGBA diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgba_scalar[first_diff], rgba_neon[first_diff]
    );
  }

  // Alpha must always be 0xFF (Yuv410p has no alpha plane).
  for (i, px) in rgba_neon.chunks(4).enumerate() {
    assert_eq!(px[3], 0xFF, "alpha at pixel {i} must be 0xFF");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_410_rgba_neon_matches_scalar_widths() {
  for &w in &[4usize, 8, 16, 20, 32, 64, 128] {
    check_yuv_410_rgba_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv_410_rgba_equivalence(w, ColorMatrix::YCgCo, false);
  }
}

// ---- yuv_410 chroma byte-lane-order parity --------------------------
//
// Targets the BE-safety of the chroma load: the kernel reads four
// chroma bytes per plane and fans each across four Y columns. If the
// load were native-endian dependent (the pre-fix `(*const u32).read +
// vdup_n_u32` pattern), a big-endian aarch64 host would reorder the
// four samples before the fan-out, producing different chroma at
// horizontal positions 0..4, 4..8, 8..12, 12..16.
//
// We pick four very different chroma values so any reordering would
// surface as a byte-level RGB mismatch vs the scalar reference. The
// test runs on every host (no `cfg(target_endian)` guard) so it
// exercises the BE-host decode path under Miri-style emulation; on
// LE x86/aarch64 hosts it provides additional functional coverage of
// the byte-lane load.

#[cfg(feature = "yuv-planar")]
fn check_yuv_410_chroma_lane_order_parity(matrix: ColorMatrix, full_range: bool) {
  // 16 pixels = exactly one NEON main-loop iteration (no scalar tail).
  let width = 16usize;
  let y: std::vec::Vec<u8> = (0..width).map(|_| 128u8).collect();
  // Distinct values across all 4 chroma lanes, well-spaced over 0..255
  // so a swap of any two would change the reconstructed RGB.
  let u: std::vec::Vec<u8> = std::vec![10, 60, 110, 200];
  let v: std::vec::Vec<u8> = std::vec![240, 30, 170, 90];

  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON yuv_410 chroma lane order diverged from scalar (matrix={matrix:?}, full_range={full_range}); \
     this typically indicates the chroma load reordered the 4 samples"
  );

  // Also check the columns 0..4 / 4..8 / 8..12 / 12..16 use distinct
  // chroma — defends against a regression where the fan-out collapses
  // multiple lanes onto the same group.
  let px = |i: usize| {
    (
      rgb_scalar[i * 3],
      rgb_scalar[i * 3 + 1],
      rgb_scalar[i * 3 + 2],
    )
  };
  assert_ne!(
    px(0),
    px(4),
    "chroma lane 0 and 1 must produce different RGB"
  );
  assert_ne!(
    px(4),
    px(8),
    "chroma lane 1 and 2 must produce different RGB"
  );
  assert_ne!(
    px(8),
    px(12),
    "chroma lane 2 and 3 must produce different RGB"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_410_neon_chroma_lane_order_parity() {
  for &(m, full) in &[
    (ColorMatrix::Bt601, true),
    (ColorMatrix::Bt709, false),
    (ColorMatrix::Bt2020Ncl, true),
  ] {
    check_yuv_410_chroma_lane_order_parity(m, full);
  }
}

// ---- rgb_to_luma_row equivalence ------------------------------------
//
// The NEON luma kernel uses pure integer Q15 arithmetic (no f32 ops),
// so output is byte‑identical to the scalar reference for every
// input. Sweeps cover both main loop (mult‑of‑16 widths) and scalar
// tail (non‑multiples), all 5 ColorMatrix variants, and both ranges.

fn check_luma_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let rgb = pseudo_random_bgr(width);
  let mut y_scalar = std::vec![0u8; width];
  let mut y_neon = std::vec![0u8; width];

  scalar::rgb_to_luma_row(&rgb, &mut y_scalar, width, matrix, full_range);
  unsafe {
    rgb_to_luma_row(&rgb, &mut y_neon, width, matrix, full_range);
  }

  assert_eq!(
    y_scalar, y_neon,
    "NEON rgb_to_luma diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb_to_luma_row_matches_scalar_widths() {
  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Fcc,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::YCgCo,
  ] {
    for full_range in [true, false] {
      for &w in &[1usize, 7, 8, 15, 16, 17, 31, 32, 33, 47, 64, 128, 130] {
        check_luma_equivalence(w, matrix, full_range);
      }
    }
  }
}

// ---- yuv_411_to_rgb_row equivalence (NEON ↔ scalar) ------------------
//
// Direct backend test for the 4:1:1 path: bypasses the public dispatcher
// so the NEON 1→4 chroma upsample is exercised regardless of what tier
// the dispatcher would pick on the current runner.

#[cfg(feature = "yuv-planar")]
fn check_yuv411_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  // FFmpeg `AV_PIX_FMT_YUV411P`: chroma row = `width.div_ceil(4)`.
  // Test fixture accepts any non-zero width; non-4-aligned widths
  // exercise the partial-chroma scalar tail.
  assert!(width > 0);
  let chroma_w = width.div_ceil(4);
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON yuv_411 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_matches_scalar_all_matrices_16() {
  // Width 16 = exactly one SIMD iteration with no scalar tail.
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv411_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_matches_scalar_tail_widths() {
  // Widths that leave a non-trivial scalar tail (not multiple of 16
  // but multiple of 4).
  for w in [4usize, 8, 12, 20, 24, 28, 36, 60, 100, 132] {
    check_yuv411_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_matches_scalar_width_1920() {
  check_yuv411_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_matches_scalar_non_4_aligned_widths() {
  // FFmpeg `AV_PIX_FMT_YUV411P` accepts any width via
  // `chroma_width = width.div_ceil(4)`. Widths < 16 stay entirely in the
  // scalar tail; larger non-4-aligned widths exercise the SIMD body +
  // partial-chroma scalar tail boundary.
  for w in [1usize, 2, 3, 5, 6, 7, 17, 31, 33, 47, 641] {
    check_yuv411_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv411_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_rgba_matches_scalar_non_4_aligned_widths() {
  for w in [1usize, 2, 3, 5, 6, 7, 17, 641] {
    check_yuv411_rgba_equivalence(w, ColorMatrix::Bt601, true);
  }
}

#[cfg(feature = "yuv-planar")]
fn check_yuv411_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  assert!(width > 0);
  let chroma_w = width.div_ceil(4);
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];

  scalar::yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }

  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON yuv_411 RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_rgba_matches_scalar_widths() {
  for &m in &[ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
    for full in [true, false] {
      for &w in &[16usize, 32, 64, 128, 1920] {
        check_yuv411_rgba_equivalence(w, m, full);
      }
    }
  }
}

// ---- yuv_411 chroma byte-lane-order parity --------------------------
//
// Targets the BE-safety of the 4:1:1 chroma load: the kernel reads
// four chroma bytes per plane and fans each across four Y columns. If
// the load were native-endian dependent (the pre-fix `(*const
// u32).read_unaligned + vcreate_u8` pattern), a big-endian aarch64
// host would reorder the four samples before the 1→4 fan-out,
// producing different chroma at horizontal positions 0..4, 4..8,
// 8..12, 12..16. The 4:1:0 NEON kernel has the analogous parity test
// (`yuv_410_neon_chroma_lane_order_parity`).
//
// We pick four very different chroma values so any reordering would
// surface as a byte-level RGB mismatch vs the scalar reference. The
// test is host-independent (no `cfg(target_endian)` guard) so it
// exercises the BE-host decode path under emulation; on LE hosts it
// provides additional functional coverage of the byte-lane load.

#[cfg(feature = "yuv-planar")]
fn check_yuv_411_chroma_lane_order_parity(matrix: ColorMatrix, full_range: bool) {
  // 16 pixels = exactly one NEON main-loop iteration (no scalar tail).
  let width = 16usize;
  let y: std::vec::Vec<u8> = (0..width).map(|_| 128u8).collect();
  // Distinct values across all 4 chroma lanes, well-spaced over 0..255
  // so a swap of any two would change the reconstructed RGB.
  let u: std::vec::Vec<u8> = std::vec![10, 60, 110, 200];
  let v: std::vec::Vec<u8> = std::vec![240, 30, 170, 90];

  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON yuv_411 chroma lane order diverged from scalar (matrix={matrix:?}, full_range={full_range}); \
     this typically indicates the chroma load reordered the 4 samples"
  );

  // Also check the four 4-pixel groups (columns 0..4 / 4..8 / 8..12 /
  // 12..16) reconstruct distinct RGB — defends against a regression
  // where the fan-out collapses multiple lanes onto the same group.
  let px = |i: usize| {
    (
      rgb_scalar[i * 3],
      rgb_scalar[i * 3 + 1],
      rgb_scalar[i * 3 + 2],
    )
  };
  assert_ne!(
    px(0),
    px(4),
    "chroma lane 0 and 1 must produce different RGB"
  );
  assert_ne!(
    px(4),
    px(8),
    "chroma lane 1 and 2 must produce different RGB"
  );
  assert_ne!(
    px(8),
    px(12),
    "chroma lane 2 and 3 must produce different RGB"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv411_chroma_lane_order_parity() {
  for &(m, full) in &[
    (ColorMatrix::Bt601, true),
    (ColorMatrix::Bt709, false),
    (ColorMatrix::Bt2020Ncl, true),
  ] {
    check_yuv_411_chroma_lane_order_parity(m, full);
  }
}
