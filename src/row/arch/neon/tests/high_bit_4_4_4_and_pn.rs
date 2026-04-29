use super::super::*;

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

fn check_yuv444p16_u8_neon_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_neon(width, 37);
  let u = p16_plane_neon(width, 53);
  let v = p16_plane_neon(width, 71);
  let a_src = p16_plane_neon(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_with_alpha_src_row(
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
    yuv_444p16_to_rgba_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "NEON Yuva444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_neon_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuva444p16_rgba_matches_scalar_widths_and_alpha() {
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_neon_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u8_neon_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, false, seed);
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
