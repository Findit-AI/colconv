use super::super::*;

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
