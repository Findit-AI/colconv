//! Dispatch-level BE/LE parity tests for the high-bit YUV planar and
//! P-format row dispatchers.
//!
//! Each test asserts that `<format>_to_<output>_row_endian(.., true)`
//! on a BE-encoded fixture produces byte-identical output to
//! `<format>_to_<output>_row_endian(.., false)` on the corresponding
//! LE-encoded fixture. Fixtures are built byte-wise via
//! `to_le_bytes` / `to_be_bytes` and reinterpreted with `from_ne_bytes`,
//! so the test is host-independent — the LE buffer / `BE=false` pair
//! exercises the no-swap kernel path while the BE buffer / `BE=true`
//! pair exercises the swap path, regardless of whether the host is LE
//! or BE.
//!
//! Tests run with SIMD active where the host CPU supports it; the
//! `#[cfg_attr(miri, ignore)]` guard avoids exercising SIMD intrinsics
//! under Miri.
//!
//! Coverage (one representative per family):
//! - `yuv420p10` — already covered inline in `dispatch/yuv420/yuv420p10.rs`
//! - `yuv444p10` — full-width planar, BITS-generic helper path
//! - `p010`     — 4:2:0 P-format, low-packed scalar / SIMD kernels
//! - `p410`     — 4:4:4 P-format, BITS-generic helper path
//! - `yuv420p16` / `p016` / `yuv444p16` / `p416` — dedicated 16-bit
//!   kernels (i64 chroma multiply path).

use crate::{KernelMatrix, row::*};

/// Build LE / BE host-native u16 buffers from a slice of intended u16
/// samples. Returns `(le, be)` where each slice contains `u16` elements
/// such that `to_ne_bytes` reproduces the LE/BE wire bytes for the
/// intended values. Identical pattern to the per-arch fixtures (see
/// `src/row/arch/*/tests/ayuv64.rs`).
fn split_le_be(intended: &[u16]) -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  let le: std::vec::Vec<u16> = le_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  let be: std::vec::Vec<u16> = be_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  (le, be)
}

fn pseudo_plane(len: usize, seed: u32, mask: u16) -> std::vec::Vec<u16> {
  (0..len)
    .map(|i| ((seed.wrapping_mul(i as u32 + 1).wrapping_add(0x55_u32)) & mask as u32) as u16)
    .collect()
}

fn pseudo_uv_interleaved(half_pairs: usize, seed: u32, mask: u16) -> std::vec::Vec<u16> {
  (0..half_pairs * 2)
    .map(|i| ((seed.wrapping_mul(i as u32 + 7).wrapping_add(0x123_u32)) & mask as u32) as u16)
    .collect()
}

// ---- yuv444p10 dispatch parity ------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_dispatch_be_le_parity_simd_and_scalar() {
  for w in [8usize, 16, 24] {
    let y_int = pseudo_plane(w, 0x111, 0x3FF);
    let u_int = pseudo_plane(w, 0x222, 0x3FF);
    let v_int = pseudo_plane(w, 0x333, 0x3FF);
    let (y_le, y_be) = split_le_be(&y_int);
    let (u_le, u_be) = split_le_be(&u_int);
    let (v_le, v_be) = split_le_be(&v_int);

    for &use_simd in &[false, true] {
      // u8 RGB — exercises BITS-generic `yuv_444p_n_to_rgb_row<10, BE>`.
      let mut out_le = std::vec![0u8; w * 3];
      let mut out_be = std::vec![0u8; w * 3];
      yuv444p10_to_rgb_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &mut out_le,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuv444p10_to_rgb_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &mut out_be,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(
        out_le, out_be,
        "yuv444p10 rgb BE/LE parity (w={w}, simd={use_simd})"
      );

      // u16 RGB
      let mut out_le16 = std::vec![0u16; w * 3];
      let mut out_be16 = std::vec![0u16; w * 3];
      yuv444p10_to_rgb_u16_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &mut out_le16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuv444p10_to_rgb_u16_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &mut out_be16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le16, out_be16, "yuv444p10 rgb_u16 BE/LE parity");

      // u8 RGBA
      let mut out_le4 = std::vec![0u8; w * 4];
      let mut out_be4 = std::vec![0u8; w * 4];
      yuv444p10_to_rgba_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &mut out_le4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuv444p10_to_rgba_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &mut out_be4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4, out_be4, "yuv444p10 rgba BE/LE parity");

      // u16 RGBA
      let mut out_le4u = std::vec![0u16; w * 4];
      let mut out_be4u = std::vec![0u16; w * 4];
      yuv444p10_to_rgba_u16_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &mut out_le4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuv444p10_to_rgba_u16_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &mut out_be4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4u, out_be4u, "yuv444p10 rgba_u16 BE/LE parity");
    }
  }
}

// ---- p010 dispatch parity (semi-planar 4:2:0, 10-bit high-packed) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_dispatch_be_le_parity_simd_and_scalar() {
  for w in [8usize, 16, 24] {
    // P010 stores active bits in the high 10 of each u16 (sample << 6),
    // so build samples already shifted into MSB-aligned form.
    let y_int: std::vec::Vec<u16> = pseudo_plane(w, 0x440, 0x3FF)
      .into_iter()
      .map(|v| v << 6)
      .collect();
    let uv_int: std::vec::Vec<u16> = pseudo_uv_interleaved(w / 2, 0x55C, 0x3FF)
      .into_iter()
      .map(|v| v << 6)
      .collect();
    let (y_le, y_be) = split_le_be(&y_int);
    let (uv_le, uv_be) = split_le_be(&uv_int);

    for &use_simd in &[false, true] {
      let mut out_le = std::vec![0u8; w * 3];
      let mut out_be = std::vec![0u8; w * 3];
      p010_to_rgb_row_endian(
        &y_le,
        &uv_le,
        &mut out_le,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p010_to_rgb_row_endian(
        &y_be,
        &uv_be,
        &mut out_be,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le, out_be, "p010 rgb BE/LE parity (w={w})");

      let mut out_le16 = std::vec![0u16; w * 3];
      let mut out_be16 = std::vec![0u16; w * 3];
      p010_to_rgb_u16_row_endian(
        &y_le,
        &uv_le,
        &mut out_le16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p010_to_rgb_u16_row_endian(
        &y_be,
        &uv_be,
        &mut out_be16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le16, out_be16, "p010 rgb_u16 BE/LE parity");

      let mut out_le4 = std::vec![0u8; w * 4];
      let mut out_be4 = std::vec![0u8; w * 4];
      p010_to_rgba_row_endian(
        &y_le,
        &uv_le,
        &mut out_le4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p010_to_rgba_row_endian(
        &y_be,
        &uv_be,
        &mut out_be4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4, out_be4, "p010 rgba BE/LE parity");

      let mut out_le4u = std::vec![0u16; w * 4];
      let mut out_be4u = std::vec![0u16; w * 4];
      p010_to_rgba_u16_row_endian(
        &y_le,
        &uv_le,
        &mut out_le4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p010_to_rgba_u16_row_endian(
        &y_be,
        &uv_be,
        &mut out_be4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4u, out_be4u, "p010 rgba_u16 BE/LE parity");
    }
  }
}

// ---- p410 dispatch parity (semi-planar 4:4:4, 10-bit high-packed) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_dispatch_be_le_parity_simd_and_scalar() {
  for w in [8usize, 16, 24] {
    let y_int: std::vec::Vec<u16> = pseudo_plane(w, 0x710, 0x3FF)
      .into_iter()
      .map(|v| v << 6)
      .collect();
    // P4xx UV is full-width interleaved (one (U,V) pair per pixel).
    let uv_int: std::vec::Vec<u16> = pseudo_uv_interleaved(w, 0x842, 0x3FF)
      .into_iter()
      .map(|v| v << 6)
      .collect();
    let (y_le, y_be) = split_le_be(&y_int);
    let (uv_le, uv_be) = split_le_be(&uv_int);

    for &use_simd in &[false, true] {
      let mut out_le = std::vec![0u8; w * 3];
      let mut out_be = std::vec![0u8; w * 3];
      p410_to_rgb_row_endian(
        &y_le,
        &uv_le,
        &mut out_le,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p410_to_rgb_row_endian(
        &y_be,
        &uv_be,
        &mut out_be,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le, out_be, "p410 rgb BE/LE parity (w={w})");

      let mut out_le16 = std::vec![0u16; w * 3];
      let mut out_be16 = std::vec![0u16; w * 3];
      p410_to_rgb_u16_row_endian(
        &y_le,
        &uv_le,
        &mut out_le16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p410_to_rgb_u16_row_endian(
        &y_be,
        &uv_be,
        &mut out_be16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le16, out_be16, "p410 rgb_u16 BE/LE parity");

      let mut out_le4 = std::vec![0u8; w * 4];
      let mut out_be4 = std::vec![0u8; w * 4];
      p410_to_rgba_row_endian(
        &y_le,
        &uv_le,
        &mut out_le4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p410_to_rgba_row_endian(
        &y_be,
        &uv_be,
        &mut out_be4,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4, out_be4, "p410 rgba BE/LE parity");

      let mut out_le4u = std::vec![0u16; w * 4];
      let mut out_be4u = std::vec![0u16; w * 4];
      p410_to_rgba_u16_row_endian(
        &y_le,
        &uv_le,
        &mut out_le4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      p410_to_rgba_u16_row_endian(
        &y_be,
        &uv_be,
        &mut out_be4u,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(out_le4u, out_be4u, "p410 rgba_u16 BE/LE parity");
    }
  }
}

// ---- 16-bit families: yuv420p16 / p016 / yuv444p16 / p416 --------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0xAAAA, 0xFFFF);
  let u_int = pseudo_plane(w / 2, 0xBBBB, 0xFFFF);
  let v_int = pseudo_plane(w / 2, 0xCCCC, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (u_le, u_be) = split_le_be(&u_int);
  let (v_le, v_be) = split_le_be(&v_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    yuv420p16_to_rgb_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuv420p16_to_rgb_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "yuv420p16 rgb BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 3];
    let mut out_be16 = std::vec![0u16; w * 3];
    yuv420p16_to_rgb_u16_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuv420p16_to_rgb_u16_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "yuv420p16 rgb_u16 BE/LE parity");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0xD0D0, 0xFFFF);
  let uv_int = pseudo_uv_interleaved(w / 2, 0xE0E0, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (uv_le, uv_be) = split_le_be(&uv_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    p016_to_rgb_row_endian(
      &y_le,
      &uv_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    p016_to_rgb_row_endian(
      &y_be,
      &uv_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "p016 rgb BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 3];
    let mut out_be16 = std::vec![0u16; w * 3];
    p016_to_rgb_u16_row_endian(
      &y_le,
      &uv_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    p016_to_rgb_u16_row_endian(
      &y_be,
      &uv_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "p016 rgb_u16 BE/LE parity");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p16_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0x4444, 0xFFFF);
  let u_int = pseudo_plane(w, 0x5555, 0xFFFF);
  let v_int = pseudo_plane(w, 0x6666, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (u_le, u_be) = split_le_be(&u_int);
  let (v_le, v_be) = split_le_be(&v_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    yuv444p16_to_rgb_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuv444p16_to_rgb_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "yuv444p16 rgb BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 3];
    let mut out_be16 = std::vec![0u16; w * 3];
    yuv444p16_to_rgb_u16_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuv444p16_to_rgb_u16_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "yuv444p16 rgb_u16 BE/LE parity");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0x7070, 0xFFFF);
  let uv_int = pseudo_uv_interleaved(w, 0x8080, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (uv_le, uv_be) = split_le_be(&uv_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    p416_to_rgb_row_endian(
      &y_le,
      &uv_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    p416_to_rgb_row_endian(
      &y_be,
      &uv_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "p416 rgb BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 3];
    let mut out_be16 = std::vec![0u16; w * 3];
    p416_to_rgb_u16_row_endian(
      &y_le,
      &uv_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    p416_to_rgb_u16_row_endian(
      &y_be,
      &uv_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "p416 rgb_u16 BE/LE parity");

    // Also exercise the i64 chroma RGBA path.
    let mut out_le4u = std::vec![0u16; w * 4];
    let mut out_be4u = std::vec![0u16; w * 4];
    p416_to_rgba_u16_row_endian(
      &y_le,
      &uv_le,
      &mut out_le4u,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    p416_to_rgba_u16_row_endian(
      &y_be,
      &uv_be,
      &mut out_be4u,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le4u, out_be4u, "p416 rgba_u16 BE/LE parity");
  }
}

// YUVA dispatch parity.
//
// Mirrors the non-alpha YUV high-bit dispatcher tests above. Adds
// dispatch-level BE/LE parity coverage for the YUVA 4:2:0 and 4:4:4
// families (otherwise the YUVA dispatch path goes through
// `BE = false` regardless of the source contract). Uses the same
// `to_le_bytes` / `to_be_bytes` host-independent fixture pattern;
// asserts byte-identical output between
// `_endian(LE_buf, false)` and `_endian(BE_buf, true)` for both
// `use_simd = true` and `use_simd = false`.

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_dispatch_be_le_parity_simd_and_scalar() {
  for w in [8usize, 16, 24] {
    let y_int = pseudo_plane(w, 0x1010, 0x3FF);
    let u_int = pseudo_plane(w / 2, 0x2020, 0x3FF);
    let v_int = pseudo_plane(w / 2, 0x3030, 0x3FF);
    let a_int = pseudo_plane(w, 0x4040, 0x3FF);
    let (y_le, y_be) = split_le_be(&y_int);
    let (u_le, u_be) = split_le_be(&u_int);
    let (v_le, v_be) = split_le_be(&v_int);
    let (a_le, a_be) = split_le_be(&a_int);

    for &use_simd in &[false, true] {
      // u8 RGBA — exercises BITS-generic
      // `yuv_420p_n_to_rgba_with_alpha_src_row<10, BE>` across all backends.
      let mut out_le = std::vec![0u8; w * 4];
      let mut out_be = std::vec![0u8; w * 4];
      yuva420p10_to_rgba_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &a_le,
        &mut out_le,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuva420p10_to_rgba_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &a_be,
        &mut out_be,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(
        out_le, out_be,
        "yuva420p10 rgba BE/LE parity (w={w}, simd={use_simd})"
      );

      // u16 RGBA — native-depth path, alpha sourced at full BITS.
      let mut out_le16 = std::vec![0u16; w * 4];
      let mut out_be16 = std::vec![0u16; w * 4];
      yuva420p10_to_rgba_u16_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &a_le,
        &mut out_le16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuva420p10_to_rgba_u16_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &a_be,
        &mut out_be16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(
        out_le16, out_be16,
        "yuva420p10 rgba_u16 BE/LE parity (w={w}, simd={use_simd})"
      );
    }
  }
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_dispatch_be_le_parity_simd_and_scalar() {
  for w in [8usize, 16, 24] {
    let y_int = pseudo_plane(w, 0x1111, 0x3FF);
    let u_int = pseudo_plane(w, 0x2222, 0x3FF);
    let v_int = pseudo_plane(w, 0x3333, 0x3FF);
    let a_int = pseudo_plane(w, 0x4444, 0x3FF);
    let (y_le, y_be) = split_le_be(&y_int);
    let (u_le, u_be) = split_le_be(&u_int);
    let (v_le, v_be) = split_le_be(&v_int);
    let (a_le, a_be) = split_le_be(&a_int);

    for &use_simd in &[false, true] {
      let mut out_le = std::vec![0u8; w * 4];
      let mut out_be = std::vec![0u8; w * 4];
      yuva444p10_to_rgba_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &a_le,
        &mut out_le,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuva444p10_to_rgba_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &a_be,
        &mut out_be,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(
        out_le, out_be,
        "yuva444p10 rgba BE/LE parity (w={w}, simd={use_simd})"
      );

      let mut out_le16 = std::vec![0u16; w * 4];
      let mut out_be16 = std::vec![0u16; w * 4];
      yuva444p10_to_rgba_u16_row_endian(
        &y_le,
        &u_le,
        &v_le,
        &a_le,
        &mut out_le16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        false,
      );
      yuva444p10_to_rgba_u16_row_endian(
        &y_be,
        &u_be,
        &v_be,
        &a_be,
        &mut out_be16,
        w,
        KernelMatrix::Bt709,
        false,
        use_simd,
        true,
      );
      assert_eq!(
        out_le16, out_be16,
        "yuva444p10 rgba_u16 BE/LE parity (w={w}, simd={use_simd})"
      );
    }
  }
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0xA1A1, 0xFFFF);
  let u_int = pseudo_plane(w / 2, 0xB2B2, 0xFFFF);
  let v_int = pseudo_plane(w / 2, 0xC3C3, 0xFFFF);
  let a_int = pseudo_plane(w, 0xD4D4, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (u_le, u_be) = split_le_be(&u_int);
  let (v_le, v_be) = split_le_be(&v_int);
  let (a_le, a_be) = split_le_be(&a_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    yuva420p16_to_rgba_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &a_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuva420p16_to_rgba_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &a_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "yuva420p16 rgba BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 4];
    let mut out_be16 = std::vec![0u16; w * 4];
    yuva420p16_to_rgba_u16_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &a_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuva420p16_to_rgba_u16_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &a_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "yuva420p16 rgba_u16 BE/LE parity");
  }
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p16_dispatch_be_le_parity() {
  let w = 16usize;
  let y_int = pseudo_plane(w, 0xA5A5, 0xFFFF);
  let u_int = pseudo_plane(w, 0xB6B6, 0xFFFF);
  let v_int = pseudo_plane(w, 0xC7C7, 0xFFFF);
  let a_int = pseudo_plane(w, 0xD8D8, 0xFFFF);
  let (y_le, y_be) = split_le_be(&y_int);
  let (u_le, u_be) = split_le_be(&u_int);
  let (v_le, v_be) = split_le_be(&v_int);
  let (a_le, a_be) = split_le_be(&a_int);

  for &use_simd in &[false, true] {
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    yuva444p16_to_rgba_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &a_le,
      &mut out_le,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuva444p16_to_rgba_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &a_be,
      &mut out_be,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le, out_be, "yuva444p16 rgba BE/LE parity");

    let mut out_le16 = std::vec![0u16; w * 4];
    let mut out_be16 = std::vec![0u16; w * 4];
    yuva444p16_to_rgba_u16_row_endian(
      &y_le,
      &u_le,
      &v_le,
      &a_le,
      &mut out_le16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      false,
    );
    yuva444p16_to_rgba_u16_row_endian(
      &y_be,
      &u_be,
      &v_be,
      &a_be,
      &mut out_be16,
      w,
      KernelMatrix::Bt709,
      false,
      use_simd,
      true,
    );
    assert_eq!(out_le16, out_be16, "yuva444p16 rgba_u16 BE/LE parity");
  }
}
