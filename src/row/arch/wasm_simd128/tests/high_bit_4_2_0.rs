use super::{
  super::*, high_bit_plane_wasm, interleave_uv_wasm, p_n_packed_plane, p010_uv_interleave,
  p16_plane_wasm, planar_n_plane,
};

// ---- rgb_to_hsv_row equivalence --------------------------------------

fn check_hsv_equivalence(rgb: &[u8], width: usize) {
  let mut h_s = std::vec![0u8; width];
  let mut s_s = std::vec![0u8; width];
  let mut v_s = std::vec![0u8; width];
  let mut h_k = std::vec![0u8; width];
  let mut s_k = std::vec![0u8; width];
  let mut v_k = std::vec![0u8; width];
  scalar::rgb_to_hsv_row(rgb, &mut h_s, &mut s_s, &mut v_s, width);
  unsafe {
    rgb_to_hsv_row(rgb, &mut h_k, &mut s_k, &mut v_k, width);
  }
  for (i, (a, b)) in h_s.iter().zip(h_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "H divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
  for (i, (a, b)) in s_s.iter().zip(s_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "S divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
  for (i, (a, b)) in v_s.iter().zip(v_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "V divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
}

#[test]
fn simd128_hsv_matches_scalar() {
  let rgb: std::vec::Vec<u8> = (0..1921 * 3)
    .map(|i| ((i * 37 + 11) & 0xFF) as u8)
    .collect();
  for w in [1usize, 15, 16, 17, 31, 1920, 1921] {
    check_hsv_equivalence(&rgb[..w * 3], w);
  }
}

// ---- yuv420p10 scalar-equivalence -----------------------------------
//
// These tests compile only for `target_arch = "wasm32"` (via the
// outer `target_feature = "simd128"` gate on the module). CI
// executes them under wasmtime in the `test-wasm-simd128` job
// (see `.github/workflows/ci.yml`): the lib is compiled for
// `wasm32-wasip1` with `-C target-feature=+simd128` and
// `CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime run --` passes each
// compiled `.wasm` test binary to wasmtime. Every scalar‑
// equivalence check below runs on real SIMD instructions, not
// just a compile check.

fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

fn check_p10_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];

  scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }

  if rgb_scalar != rgb_simd {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "simd128 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

fn check_p10_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];

  scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }

  if rgb_scalar != rgb_simd {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "simd128 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

#[test]
fn simd128_p10_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p10_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p10_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p10_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p10_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p10_matches_scalar_1920() {
  check_p10_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p10_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> simd128 scalar-equivalence (BITS=9 coverage) ---

fn p_n_plane_simd128<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

fn check_p_n_u8_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_plane_simd128::<BITS>(width, 37);
  let u = p_n_plane_simd128::<BITS>(width / 2, 53);
  let v = p_n_plane_simd128::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_u16_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_plane_simd128::<BITS>(width, 37);
  let u = p_n_plane_simd128::<BITS>(width / 2, 53);
  let v = p_n_plane_simd128::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_simd128_equivalence::<9>(16, m, full);
      check_p_n_u16_simd128_equivalence::<9>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p9_matches_scalar_tail_and_large_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p_n_u8_simd128_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_p_n_u16_simd128_equivalence::<9>(w, ColorMatrix::Bt709, true);
  }
  check_p_n_u8_simd128_equivalence::<9>(1920, ColorMatrix::Bt709, false);
  check_p_n_u16_simd128_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- P010 simd128 scalar-equivalence --------------------------------

fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
    .collect()
}

fn check_p010_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u8 diverges");
}

fn check_p010_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u16 diverges");
}

#[test]
fn simd128_p010_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p010_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p010_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p010_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p010_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p010_matches_scalar_1920() {
  check_p010_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p010_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- Generic BITS equivalence (12/14-bit coverage) ------------------

fn check_planar_u8_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 planar {BITS}-bit → u8 diverges"
  );
}

fn check_planar_u16_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 planar {BITS}-bit → u16 diverges"
  );
}

fn check_pn_u8_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u8 diverges");
}

fn check_pn_u16_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u16 diverges");
}

#[test]
fn simd128_p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_simd128_equivalence_n::<12>(16, m, full);
      check_planar_u16_simd128_equivalence_n::<12>(16, m, full);
      check_pn_u8_simd128_equivalence_n::<12>(16, m, full);
      check_pn_u16_simd128_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_simd128_equivalence_n::<14>(16, m, full);
      check_planar_u16_simd128_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_p12_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_planar_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
    check_pn_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_pn_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn simd128_p14_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_simd128_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
    check_planar_u16_simd128_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
  }
}

// ---- 16-bit (full-range u16 samples) simd128 equivalence ------------

fn check_yuv420p16_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

/// u16-output equivalence for the 4:2:0 16-bit kernel. Uses
/// **exact-sized** chroma planes (`width / 2` samples) to catch
/// any over-read past the end of tight chroma rows — the
/// specific failure mode that originally shipped as a `v128_load`
/// (16-byte) read for only 4 u16 needed, now fixed via
/// `v128_load64_zero`.
fn check_yuv420p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv444p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_simd128_equivalence(16, m, full);
      check_p16_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p16_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_yuv420p16_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p16_u8_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p16_matches_scalar_1920() {
  check_yuv420p16_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p16_u8_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

#[test]
fn simd128_16bit_u16_matches_scalar_all_matrices() {
  // Every new 16-bit u16-output path at the smallest block-aligned
  // width (8 pixels for 420/p16, which matches the native SIMD
  // iteration; 4:4:4 uses 8 too).
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u16_simd128_equivalence(16, m, full);
      check_p16_u16_simd128_equivalence(16, m, full);
      check_yuv444p16_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p16_u16_matches_scalar_tight_widths() {
  // Tight widths explicitly stress over-read of half-width chroma
  // planes (8 samples at width=16 → second 8-pixel SIMD iter reads
  // chroma from offset 4, formerly `v128_load` of 8 samples = 8
  // bytes past plane-end).
  for w in [8usize, 10, 16, 18, 24, 26, 1920, 1922] {
    check_yuv420p16_u16_simd128_equivalence(w, ColorMatrix::Bt709, false);
    check_p16_u16_simd128_equivalence(w, ColorMatrix::Bt2020Ncl, true);
    check_yuv444p16_u16_simd128_equivalence(w, ColorMatrix::Bt601, false);
  }
}
