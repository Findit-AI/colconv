use super::*;

/// Deterministic scalar‑equivalence fixture. Fills Y/U/V with a
/// hash‑like sequence so every byte varies, then compares byte‑exact.
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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_width_32() {
  check_equivalence(32, ColorMatrix::Bt601, true);
  check_equivalence(32, ColorMatrix::Bt709, false);
  check_equivalence(32, ColorMatrix::YCgCo, true);
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_matches_scalar_width_1920() {
  check_equivalence(1920, ColorMatrix::Bt709, false);
}

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_width_32() {
  check_rgba_equivalence(32, ColorMatrix::Bt601, true);
  check_rgba_equivalence(32, ColorMatrix::Bt709, false);
  check_rgba_equivalence(32, ColorMatrix::YCgCo, true);
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_width_1920() {
  check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgba_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- nv12_to_rgb_row equivalence ------------------------------------

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_scalar_width_1920() {
  check_nv12_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_matches_yuv420p_neon() {
  for w in [16usize, 30, 64, 1920] {
    check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
    check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv21_to_rgb_row equivalence ------------------------------------

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_matches_scalar_widths() {
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv12_neon_rgba_matches_yuv420p_rgba_neon() {
  for w in [16usize, 30, 64, 1920] {
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::Bt709, false);
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::YCgCo, true);
  }
}

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv21_neon_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_nv21_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv42_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_matches_nv24_swapped() {
  for w in [16usize, 17, 33, 64, 1920] {
    check_nv42_matches_nv24_with_swapped_uv(w, ColorMatrix::Bt709, false);
    check_nv42_matches_nv24_with_swapped_uv(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv24_to_rgba_row / nv42_to_rgba_row equivalence ----------------

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv24_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn nv42_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv42_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgb_row equivalence ---------------------------------

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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn yuv_444_neon_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
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

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn swap_neon_matches_scalar_widths() {
  for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
    check_swap_equivalence(w);
  }
}

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

// ---- yuv420p10 scalar-equivalence -----------------------------------

/// Deterministic pseudo‑random `u16` samples in `[0, 1023]` — the
/// 10‑bit range. Upper 6 bits always zero, so the generator matches
/// real `yuv420p10le` bit patterns.
fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

fn check_p10_u8_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON 10→u8 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

fn check_p10_u16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];

  scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }

  if rgb_scalar != rgb_neon {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON 10→u16 diverges from scalar at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[first_diff], rgb_neon[first_diff]
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p10_u8_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p10_u16_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p10_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p10_u8_equivalence(w, ColorMatrix::Bt601, false);
    check_p10_u16_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p10_matches_scalar_1920() {
  check_p10_u8_equivalence(1920, ColorMatrix::Bt709, false);
  check_p10_u16_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> scalar-equivalence (BITS=9 coverage) -------------
//
// Const-generic siblings of the BITS=10 helpers above. Used to pin
// the BITS=9 4:2:0 SIMD path against scalar — Yuv420p9 / Yuv422p9
// both dispatch into the same `yuv_420p_n_to_rgb_*<9>` kernels.

fn p_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

fn check_p_n_u8_equivalence<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p_n_plane::<BITS>(width, 37);
  let u = p_n_plane::<BITS>(width / 2, 53);
  let v = p_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_u16_equivalence<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p_n_plane::<BITS>(width, 37);
  let u = p_n_plane::<BITS>(width / 2, 53);
  let v = p_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];

  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_equivalence::<9>(16, m, full);
      check_p_n_u16_equivalence::<9>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p9_matches_scalar_tail_and_large_widths() {
  // Tail widths force scalar fallback past the SIMD main loop;
  // 1920 is one full HD luma row.
  for w in [18usize, 30, 34, 1922] {
    check_p_n_u8_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_p_n_u16_equivalence::<9>(w, ColorMatrix::Bt709, true);
  }
  check_p_n_u8_equivalence::<9>(1920, ColorMatrix::Bt709, false);
  check_p_n_u16_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
}

/// Out‑of‑range regression: every kernel AND‑masks each `u16` load
/// to the low `BITS` bits, so **arbitrary** upper‑bit corruption
/// (not just p010 packing) produces scalar/NEON bit‑identical
/// output. This test sweeps three adversarial input shapes:
///
/// - `p010`: 10 active bits in the high 10 of each `u16`
///   (`sample << 6`) — the canonical mispacking mistake.
/// - `ycgco_worst`: `Y=[0x8000; W]`, `U=[0; W/2]`, `V=[0x8000; W/2]`
///   — the specific Codex‑identified case that used to produce
///   `(1023, 0, 0)` on scalar vs `(0, 0, 0)` on NEON before the
///   load‑time mask was added.
/// - `random`: arbitrary upper‑bit flips with no particular pattern.
///
/// Each variant runs through every color matrix × range × both
/// output paths (u8 + native‑depth u16) and asserts byte equality.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p10_matches_scalar_on_out_of_range_samples() {
  let width = 32;

  let p010_variant =
    |i: usize, seed: u16| 0xFC00u16.wrapping_add(((i as u16).wrapping_mul(seed)) << 6);
  let random_variant = |i: usize, seed: u16| {
    let x = (i as u32)
      .wrapping_mul(seed as u32)
      .wrapping_add(0xDEAD_BEEF) as u16;
    x ^ 0xA5A5
  };

  for variant_name in ["p010", "ycgco_worst", "random"] {
    let y: std::vec::Vec<u16> = match variant_name {
      "ycgco_worst" => std::vec![0x8000u16; width],
      "p010" => (0..width).map(|i| p010_variant(i, 37)).collect(),
      _ => (0..width).map(|i| random_variant(i, 37)).collect(),
    };
    let u: std::vec::Vec<u16> = match variant_name {
      "ycgco_worst" => std::vec![0x0u16; width / 2],
      "p010" => (0..width / 2).map(|i| p010_variant(i, 53)).collect(),
      _ => (0..width / 2).map(|i| random_variant(i, 53)).collect(),
    };
    let v: std::vec::Vec<u16> = match variant_name {
      "ycgco_worst" => std::vec![0x8000u16; width / 2],
      "p010" => (0..width / 2).map(|i| p010_variant(i, 71)).collect(),
      _ => (0..width / 2).map(|i| random_variant(i, 71)).collect(),
    };

    for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
      for full_range in [true, false] {
        let mut rgb_scalar = std::vec![0u8; width * 3];
        let mut rgb_neon = std::vec![0u8; width * 3];
        scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
        unsafe {
          yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
        }
        assert_eq!(
          rgb_scalar, rgb_neon,
          "scalar and NEON diverge on {variant_name} input (matrix={matrix:?}, full_range={full_range})"
        );

        let mut rgb16_scalar = std::vec![0u16; width * 3];
        let mut rgb16_neon = std::vec![0u16; width * 3];
        scalar::yuv_420p_n_to_rgb_u16_row::<10>(
          &y,
          &u,
          &v,
          &mut rgb16_scalar,
          width,
          matrix,
          full_range,
        );
        unsafe {
          yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb16_neon, width, matrix, full_range);
        }
        assert_eq!(
          rgb16_scalar, rgb16_neon,
          "scalar and NEON diverge on {variant_name} u16 output (matrix={matrix:?}, full_range={full_range})"
        );
      }
    }
  }
}

// ---- P010 NEON scalar-equivalence --------------------------------------

/// P010 test samples: 10‑bit values shifted into the high 10 bits
/// (`value << 6`). Deterministic pseudo‑random generator keyed by
/// index × seed so U, V, Y vectors are mutually distinct.
fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
    .collect()
}

/// Interleaves per‑pair U, V samples into P010's semi‑planar UV
/// layout: `[U0, V0, U1, V1, …]`.
fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
  let pairs = u.len();
  debug_assert_eq!(u.len(), v.len());
  let mut out = std::vec::Vec::with_capacity(pairs * 2);
  for i in 0..pairs {
    out.push(u[i]);
    out.push(v[i]);
  }
  out
}

fn check_p010_u8_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p010_plane(width, 37);
  let u_plane = p010_plane(width / 2, 53);
  let v_plane = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u_plane, &v_plane);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];

  scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  if rgb_scalar != rgb_neon {
    let diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON P010→u8 diverges at byte {diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[diff], rgb_neon[diff]
    );
  }
}

fn check_p010_u16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p010_plane(width, 37);
  let u_plane = p010_plane(width / 2, 53);
  let v_plane = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u_plane, &v_plane);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];

  scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  if rgb_scalar != rgb_neon {
    let diff = rgb_scalar
      .iter()
      .zip(rgb_neon.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "NEON P010→u16 diverges at elem {diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
      rgb_scalar[diff], rgb_neon[diff]
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_u8_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_u16_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_matches_scalar_odd_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p010_u8_equivalence(w, ColorMatrix::Bt601, false);
    check_p010_u16_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_matches_scalar_1920() {
  check_p010_u8_equivalence(1920, ColorMatrix::Bt709, false);
  check_p010_u16_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

/// Adversarial regression: mispacked input — `yuv420p10le` values
/// (10 bits in low 10) accidentally handed to the P010 kernel, or
/// arbitrary bit corruption — must still produce bit‑identical
/// output on scalar and NEON. The kernel's `>> 6` load extracts
/// only the high 10 bits, so any low‑6‑bits data gets deterministically
/// discarded in both paths.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_matches_scalar_on_mispacked_input() {
  let width = 32;

  // Three input variants:
  //   - `yuv420p10le_style`: values in low 10 bits (wrong packing
  //     for P010 — `>> 6` drops the actual data, producing near‑black).
  //   - `noise`: arbitrary 16‑bit noise, no particular pattern.
  //   - `every_bit`: each sample has every bit set (0xFFFF).
  for variant in ["yuv420p10le_style", "noise", "every_bit"] {
    let y: std::vec::Vec<u16> = match variant {
      "every_bit" => std::vec![0xFFFFu16; width],
      "yuv420p10le_style" => (0..width).map(|i| ((i * 37 + 11) & 0x3FF) as u16).collect(),
      _ => (0..width)
        .map(|i| ((i as u32 * 53 + 0xDEAD) as u16) ^ 0xA5A5)
        .collect(),
    };
    let uv: std::vec::Vec<u16> = match variant {
      "every_bit" => std::vec![0xFFFFu16; width],
      "yuv420p10le_style" => (0..width).map(|i| ((i * 71 + 23) & 0x3FF) as u16).collect(),
      _ => (0..width)
        .map(|i| ((i as u32 * 91 + 0xBEEF) as u16) ^ 0x5A5A)
        .collect(),
    };

    for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
      for full_range in [true, false] {
        let mut rgb_scalar = std::vec![0u8; width * 3];
        let mut rgb_neon = std::vec![0u8; width * 3];
        scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
        unsafe {
          p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
        }
        assert_eq!(
          rgb_scalar, rgb_neon,
          "scalar and NEON diverge on {variant} P010 input (matrix={matrix:?}, full_range={full_range})"
        );

        let mut rgb16_scalar = std::vec![0u16; width * 3];
        let mut rgb16_neon = std::vec![0u16; width * 3];
        scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb16_scalar, width, matrix, full_range);
        unsafe {
          p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb16_neon, width, matrix, full_range);
        }
        assert_eq!(
          rgb16_scalar, rgb16_neon,
          "scalar and NEON diverge on {variant} P010 u16 output (matrix={matrix:?}, full_range={full_range})"
        );
      }
    }
  }
}

// ---- Generic BITS equivalence (12/14-bit coverage) ------------------

fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  (0..n)
    .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
    .collect()
}

fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
    .collect()
}

fn check_planar_u8_neon_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_neon, "NEON planar {BITS}-bit → u8 diverges");
}

fn check_planar_u16_neon_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON planar {BITS}-bit → u16 diverges"
  );
}

fn check_pn_u8_neon_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_neon, "NEON Pn {BITS}-bit → u8 diverges");
}

fn check_pn_u16_neon_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_neon, "NEON Pn {BITS}-bit → u16 diverges");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_neon_equivalence_n::<12>(16, m, full);
      check_planar_u16_neon_equivalence_n::<12>(16, m, full);
      check_pn_u8_neon_equivalence_n::<12>(16, m, full);
      check_pn_u16_neon_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_neon_equivalence_n::<14>(16, m, full);
      check_planar_u16_neon_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p12_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_neon_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_planar_u16_neon_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
    check_pn_u8_neon_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_pn_u16_neon_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p14_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_neon_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
    check_planar_u16_neon_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
  }
}

// ---- High-bit 4:2:0 RGBA equivalence (Ship 8 Tranche 5a) ----------
//
// RGBA wrappers share the math of their RGB siblings — only the store
// (and tail dispatch) branches on `ALPHA`. These tests pin that the
// SIMD RGBA path produces byte-identical output to the scalar RGBA
// reference, which already encodes the alpha = 0xFF contract.

fn check_planar_u8_neon_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::yuv_420p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON yuv_420p_n<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u8_neon_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::p_n_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_row::<BITS>(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Pn<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p_n_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_neon_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u8_neon_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u8_neon_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u8_neon_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u8_neon_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u8_neon_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u8_neon_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u8_neon_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u8_neon_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u8_neon_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_rgba_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u8_neon_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u8_neon_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

fn check_yuv420p16_u8_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width / 2, 53);
  let v = p16_plane_neon(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON yuv_420p16→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p016_u8_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width / 2, 53);
  let v = p16_plane_neon(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::p16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON P016→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u8_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p016_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p016_u8_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p016_u8_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- High-bit 4:2:0 native-depth `u16` RGBA equivalence (Ship 8 Tranche 5b) ----
//
// u16 RGBA wrappers share the math of their u16 RGB siblings — only
// the store (and tail dispatch) branches on `ALPHA`, with alpha set to
// `(1 << BITS) - 1` for BITS-generic kernels and `0xFFFF` for 16-bit
// kernels. Tests pin byte-identical output against the scalar RGBA
// reference.

fn check_planar_u16_neon_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_u16_row::<BITS>(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON yuv_420p_n<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u16_neon_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::p_n_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Pn<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p_n_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u16_neon_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u16_neon_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u16_neon_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u16_neon_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p_n_rgba_u16_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u16_neon_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u16_neon_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u16_neon_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u16_neon_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u16_neon_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u16_neon_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_rgba_u16_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u16_neon_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u16_neon_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

fn check_yuv420p16_u16_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width / 2, 53);
  let v = p16_plane_neon(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON yuv_420p16→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p016_u16_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width / 2, 53);
  let v = p16_plane_neon(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::p16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_u16_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON P016→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv420p16_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u16_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u16_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p016_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p016_u16_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p016_u16_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- Yuv444p_n NEON equivalence (10/12/14) --------------------------

fn check_yuv444p_n_u8_neon_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // 4:4:4 — chroma is full-width, 1:1 with Y.
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Yuv444p {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p_n_u16_neon_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Yuv444p {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p9_matches_scalar_all_matrices() {
  // BITS=9 reuses the same const-generic kernel as 10/12/14; this
  // test pins the AND-mask + Q15 scale path at the lowest legal depth.
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_equivalence::<9>(16, m, full);
      check_yuv444p_n_u16_neon_equivalence::<9>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p10_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_equivalence::<10>(16, m, full);
      check_yuv444p_n_u16_neon_equivalence::<10>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_equivalence::<12>(16, m, full);
      check_yuv444p_n_u16_neon_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_equivalence::<14>(16, m, full);
      check_yuv444p_n_u16_neon_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p_n_matches_scalar_widths() {
  // Odd widths validate the 4:4:4 no-parity contract and force
  // non-trivial scalar tails.
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv444p_n_u8_neon_equivalence::<10>(w, ColorMatrix::Bt709, false);
    check_yuv444p_n_u16_neon_equivalence::<10>(w, ColorMatrix::Bt2020Ncl, true);
  }
}

// ---- Yuv444p16 NEON equivalence -------------------------------------

fn p16_plane_neon(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_yuv444p16_u8_neon_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Yuv444p16 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_neon_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Yuv444p16 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_neon_equivalence(16, m, full);
      check_yuv444p16_u16_neon_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p16_matches_scalar_widths() {
  for w in [1usize, 3, 7, 8, 9, 15, 16, 17, 32, 33, 1920, 1921] {
    check_yuv444p16_u8_neon_equivalence(w, ColorMatrix::Bt709, false);
    check_yuv444p16_u16_neon_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) NEON equivalence -----------------

/// Generates a high-bit-packed `u16` plane: random `BITS`-bit values
/// shifted left by `16 - BITS` (P410/P412 convention).
fn high_bit_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

fn interleave_uv(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}

fn check_p_n_444_u8_neon_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane::<BITS>(width, 37);
  let u = high_bit_plane::<BITS>(width, 53);
  let v = high_bit_plane::<BITS>(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_u16_neon_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane::<BITS>(width, 37);
  let u = high_bit_plane::<BITS>(width, 53);
  let v = high_bit_plane::<BITS>(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_neon_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_neon = std::vec![0u8; width * 3];
  scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_neon_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_neon = std::vec![0u16; width * 3];
  scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_neon,
    "NEON P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p410_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_u8_neon_equivalence::<10>(16, m, full);
      check_p_n_444_u16_neon_equivalence::<10>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p412_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_p_n_444_u8_neon_equivalence::<12>(16, m, full);
      check_p_n_444_u16_neon_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p410_p412_matches_scalar_tail_widths() {
  // Tail widths force scalar fallback past the SIMD main loop.
  // 4:4:4 has no width-parity constraint.
  for w in [1usize, 3, 7, 15, 17, 31, 33, 1920, 1921] {
    check_p_n_444_u8_neon_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_p_n_444_u16_neon_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_p_n_444_u8_neon_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p416_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_neon_equivalence(16, m, full);
      check_p_n_444_16_u16_neon_equivalence(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p416_matches_scalar_tail_widths() {
  for w in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 33, 1920, 1921] {
    check_p_n_444_16_u8_neon_equivalence(w, ColorMatrix::Bt709, false);
    check_p_n_444_16_u16_neon_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}

// ---- High-bit 4:4:4 u8 RGBA equivalence (Ship 8 Tranche 7b) ---------
//
// Mirrors the 4:2:0 RGBA pattern in PR #25 (Tranche 5a). Each kernel
// family — Yuv444p_n (BITS-generic), Yuv444p16, Pn_444 (BITS-generic),
// Pn_444_16 — has its NEON RGBA kernel byte-pinned against the scalar
// reference at the natural width and a sweep of tail widths.

fn check_yuv444p_n_u8_neon_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Yuv444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u8_neon_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane::<BITS>(width, 37);
  let u = high_bit_plane::<BITS>(width, 53);
  let v = high_bit_plane::<BITS>(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Pn4:4:4<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u8_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Yuv444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON P416 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p_n_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_rgba_equivalence::<9>(16, m, full);
      check_yuv444p_n_u8_neon_rgba_equivalence::<10>(16, m, full);
      check_yuv444p_n_u8_neon_rgba_equivalence::<12>(16, m, full);
      check_yuv444p_n_u8_neon_rgba_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p_n_rgba_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_neon_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u8_neon_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u8_neon_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u8_neon_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_444_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_444_u8_neon_rgba_equivalence::<10>(16, m, full);
      check_pn_444_u8_neon_rgba_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_444_rgba_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u8_neon_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u8_neon_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p416_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u8_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- YUVA 4:4:4 u8 RGBA equivalence (Ship 8b‑1b) --------------------
//
// Mirrors the no-alpha 4:4:4 RGBA pattern above for the alpha-source
// path: per-pixel alpha byte is loaded from the source plane, masked
// with `bits_mask::<10>()`, and depth-converted via `>> 2`. Pseudo-
// random alpha is used to flush out lane-order corruption that a
// solid-alpha buffer would mask.

fn check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_neon = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
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
    yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
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
    "NEON Yuva444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_matches_scalar_widths() {
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_matches_scalar_random_alpha() {
  // Different alpha seeds — ensures the alpha lane order through
  // `vst4q_u8` is not confused with R/G/B.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<10>(
      16,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<10>(
      31,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_n_rgba_matches_scalar_all_bits() {
  // BITS = 9, 12, 14 (BITS = 10 is covered above with full matrix
  // sweep). Confirms the variable shift count `BITS - 8` resolves
  // correctly across the supported bit depths.
  for full in [true, false] {
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<9>(16, ColorMatrix::Bt601, full, 53);
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<12>(16, ColorMatrix::Bt709, full, 53);
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<14>(
      16,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_n_rgba_matches_scalar_all_bits_widths() {
  // BITS = 9, 12, 14 across tail widths — the variable-shift alpha
  // path applies across both SIMD body and scalar tail.
  for w in [17usize, 47, 1922] {
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u8_neon_rgba_with_alpha_src_equivalence::<14>(w, ColorMatrix::YCgCo, false, 89);
  }
}

// ---- High-bit 4:4:4 native-depth `u16` RGBA equivalence (Ship 8 Tranche 7c) ----
//
// u16 RGBA wrappers share the math of their u16 RGB siblings — only
// the store (and tail dispatch) branches on `ALPHA`, with alpha set to
// `(1 << BITS) - 1` for BITS-generic kernels and `0xFFFF` for 16-bit
// kernels. Tests pin byte-identical output against the scalar RGBA
// reference.

fn check_yuv444p_n_u16_neon_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgba_u16_row::<BITS>(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Yuv444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u16_neon_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane::<BITS>(width, 37);
  let u = high_bit_plane::<BITS>(width, 53);
  let v = high_bit_plane::<BITS>(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Pn4:4:4<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON Yuv444p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_neon_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let uv = interleave_uv(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_neon, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_neon,
    "NEON P416 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p_n_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u16_neon_rgba_equivalence::<9>(16, m, full);
      check_yuv444p_n_u16_neon_rgba_equivalence::<10>(16, m, full);
      check_yuv444p_n_u16_neon_rgba_equivalence::<12>(16, m, full);
      check_yuv444p_n_u16_neon_rgba_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p_n_rgba_u16_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_neon_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u16_neon_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u16_neon_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u16_neon_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_444_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_444_u16_neon_rgba_equivalence::<10>(16, m, full);
      check_pn_444_u16_neon_rgba_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_pn_444_rgba_u16_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u16_neon_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u16_neon_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv444p16_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u16_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u16_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p416_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u16_neon_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u16_neon_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- YUVA 4:4:4 native-depth `u16` RGBA equivalence (Ship 8b‑1c) ----
//
// Mirrors the u8 RGBA alpha-source tests above for the u16 output
// path: per-pixel alpha element is loaded from the source plane,
// AND-masked with `bits_mask::<10>()`, and stored at native depth (no
// `>> (BITS - 8)` since both source alpha and output element are at
// the same bit depth). Pseudo-random alpha flushes lane-order
// corruption that a solid-alpha buffer would mask.

fn check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_neon = std::vec![0u16; width * 4];
  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
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
    yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
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
    "NEON Yuva444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_u16_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_u16_matches_scalar_widths() {
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p10_rgba_u16_matches_scalar_random_alpha() {
  // Different alpha seeds — ensures the alpha lane order through
  // `vst4q_u16` is not confused with R/G/B.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<10>(
      16,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<10>(
      31,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_n_rgba_u16_matches_scalar_all_bits() {
  // BITS = 9, 12, 14 (BITS = 10 covered above). Confirms the
  // AND-mask `mask_v` resolves correctly across the supported bit
  // depths (no shift count to vary in the u16 path).
  for full in [true, false] {
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<9>(16, ColorMatrix::Bt601, full, 53);
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<12>(
      16,
      ColorMatrix::Bt709,
      full,
      53,
    );
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<14>(
      16,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p_n_rgba_u16_matches_scalar_all_bits_widths() {
  // BITS = 9, 12, 14 across tail widths.
  for w in [17usize, 47, 1922] {
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u16_neon_rgba_with_alpha_src_equivalence::<14>(
      w,
      ColorMatrix::YCgCo,
      false,
      89,
    );
  }
}
