//! #263 PR 6 — direct **4:4:4-packed** YUV→HSV kernels + RGB-free
//! routing (ayuv64 / vuya / vuyx / xv36).
//!
//! Two concerns, mirroring the planar / semi-planar HSV suites:
//!
//! 1. **Row-kernel parity** — each `{fmt}_to_hsv_row` dispatcher must be
//!    byte-identical to `rgb_to_hsv_row({fmt}_to_rgb_row(...))` within a
//!    tier (they share the same **8-bit** RGB intermediate the existing
//!    packed→RGB→HSV path uses, and the same per-tier HSV), across the
//!    three colour matrices, full/limited range, both endiannesses where
//!    the format carries a `BE` wire variant (ayuv64 / xv36), and a
//!    spread of widths — for both the scalar path (`use_simd = false`)
//!    and the host SIMD path (`use_simd = true`). The α / padding slot is
//!    independent of HSV (HSV derives from the Y/U/V colour only), so the
//!    reference RGB kernel and the HSV kernel both drop it identically.
//! 2. **Structural** — a 4:4:4-packed-YUV sink with ONLY `with_hsv()`
//!    (no RGB / RGBA) must not grow the source-width RGB scratch.

use super::*;
use crate::row::{
  ayuv64_to_hsv_row, ayuv64_to_rgb_row, rgb_to_hsv_row, vuya_to_hsv_row, vuya_to_rgb_row,
  vuyx_to_hsv_row, vuyx_to_rgb_row, xv36_to_hsv_row, xv36_to_rgb_row,
};

const MATRICES: [KernelMatrix; 3] = [
  KernelMatrix::Bt601,
  KernelMatrix::Bt709,
  KernelMatrix::Bt2020Ncl,
];

/// A non-trivial, non-gray pseudo-random `u8` sample at a given position,
/// so the HSV hue / saturation branches are all exercised rather than the
/// degenerate gray fast-path.
fn samp8(i: usize, salt: usize) -> u8 {
  (i.wrapping_mul(2_654_435_761)
    .wrapping_add(salt.wrapping_mul(40_503))
    .wrapping_add(0x9E37) as u32
    & 0xFF) as u8
}

/// A non-trivial pseudo-random `u16` sample (full 16-bit range) at a
/// given position — used for AYUV64 (16-bit native) and, MSB-aligned, for
/// XV36 (12-bit).
fn samp16(i: usize, salt: usize) -> u16 {
  (i.wrapping_mul(2_654_435_761)
    .wrapping_add(salt.wrapping_mul(40_503))
    .wrapping_add(0x9E37) as u32
    & 0xFFFF) as u16
}

fn ref_hsv_from_rgb(rgb: &[u8], w: usize, use_simd: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut h = std::vec![0u8; w];
  let mut s = std::vec![0u8; w];
  let mut v = std::vec![0u8; w];
  rgb_to_hsv_row(rgb, &mut h, &mut s, &mut v, w, use_simd);
  (h, s, v)
}

macro_rules! assert_planes_eq {
  ($got:expr, $want:expr, $ctx:expr) => {{
    let (gh, gs, gv) = &$got;
    let (wh, ws, wv) = &$want;
    assert_eq!(gh, wh, "H mismatch ({})", $ctx);
    assert_eq!(gs, ws, "S mismatch ({})", $ctx);
    assert_eq!(gv, wv, "V mismatch ({})", $ctx);
  }};
}

// ---- AYUV64 (16-bit, BE-threaded) --------------------------------------

/// Build host-native LE/BE packed AYUV64 buffers (slot order `[A, Y, U,
/// V]`) for `width` pixels. Returns `(le, be)` where each `u16`, when
/// serialized via `to_ne_bytes`, reproduces the LE/BE wire bytes — so the
/// `be_input = false` / `be_input = true` kernel paths are exercised
/// regardless of host endianness.
fn ayuv64_le_be(width: usize) -> (Vec<u16>, Vec<u16>) {
  let mut intended = Vec::with_capacity(width * 4);
  for x in 0..width {
    intended.push(samp16(x, 0)); // A (dropped by HSV)
    intended.push(samp16(x, 1)); // Y
    intended.push(samp16(x, 2)); // U
    intended.push(samp16(x, 3)); // V
  }
  let le = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  let be = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect();
  (le, be)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_hsv_row_matches_rgb_then_hsv() {
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let (le, be) = ayuv64_le_be(w);
    for (be_input, buf) in [(false, &le), (true, &be)] {
      for &matrix in &MATRICES {
        for &full_range in &[true, false] {
          for &use_simd in &[false, true] {
            let mut rgb = std::vec![0u8; w * 3];
            ayuv64_to_rgb_row(buf, &mut rgb, w, matrix, full_range, use_simd, be_input);
            let want = ref_hsv_from_rgb(&rgb, w, use_simd);

            let mut h = std::vec![0u8; w];
            let mut s = std::vec![0u8; w];
            let mut v = std::vec![0u8; w];
            ayuv64_to_hsv_row(
              buf, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, be_input,
            );
            assert_planes_eq!(
              (h, s, v),
              want,
              std::format!(
                "ayuv64 be={be_input} w={w} {matrix:?} full={full_range} simd={use_simd}"
              )
            );
          }
        }
      }
    }
  }
}

// ---- XV36 (12-bit MSB-aligned, BE-threaded) ----------------------------

/// Build host-native LE/BE packed XV36 buffers (slot order `[U, Y, V, A]`,
/// each MSB-aligned at 12 bits — low 4 bits zero) for `width` pixels.
fn xv36_le_be(width: usize) -> (Vec<u16>, Vec<u16>) {
  let mut intended = Vec::with_capacity(width * 4);
  for x in 0..width {
    // Mask to 12 bits then shift up 4 to MSB-align (matches the XV36 wire
    // layout; the kernels `>> 4` to recover the 12-bit value).
    intended.push((samp16(x, 5) & 0x0FFF) << 4); // U
    intended.push((samp16(x, 6) & 0x0FFF) << 4); // Y
    intended.push((samp16(x, 7) & 0x0FFF) << 4); // V
    intended.push((samp16(x, 8) & 0x0FFF) << 4); // A (padding)
  }
  let le = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  let be = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect();
  (le, be)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_hsv_row_matches_rgb_then_hsv() {
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let (le, be) = xv36_le_be(w);
    for (be_input, buf) in [(false, &le), (true, &be)] {
      for &matrix in &MATRICES {
        for &full_range in &[true, false] {
          for &use_simd in &[false, true] {
            let mut rgb = std::vec![0u8; w * 3];
            xv36_to_rgb_row(buf, &mut rgb, w, matrix, full_range, use_simd, be_input);
            let want = ref_hsv_from_rgb(&rgb, w, use_simd);

            let mut h = std::vec![0u8; w];
            let mut s = std::vec![0u8; w];
            let mut v = std::vec![0u8; w];
            xv36_to_hsv_row(
              buf, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd, be_input,
            );
            assert_planes_eq!(
              (h, s, v),
              want,
              std::format!("xv36 be={be_input} w={w} {matrix:?} full={full_range} simd={use_simd}")
            );
          }
        }
      }
    }
  }
}

// ---- VUYA / VUYX (8-bit, no endian variant) ----------------------------

/// Build a packed VUYA / VUYX buffer (byte order `[V, U, Y, A]`) for
/// `width` pixels.
fn vuya_packed(width: usize) -> Vec<u8> {
  let mut out = Vec::with_capacity(width * 4);
  for x in 0..width {
    out.push(samp8(x, 11)); // V
    out.push(samp8(x, 12)); // U
    out.push(samp8(x, 13)); // Y
    out.push(samp8(x, 14)); // A / X (dropped by HSV)
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_hsv_row_matches_rgb_then_hsv() {
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let buf = vuya_packed(w);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          vuya_to_rgb_row(&buf, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          vuya_to_hsv_row(
            &buf, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("vuya w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_hsv_row_matches_rgb_then_hsv() {
  // VUYX shares VUYA's byte stream; the padding byte is dropped by both
  // the RGB and HSV kernels, so the parity oracle is the VUYX RGB kernel.
  for &w in &[1usize, 3, 5, 15, 16, 31, 64, 65] {
    let buf = vuya_packed(w);
    for &matrix in &MATRICES {
      for &full_range in &[true, false] {
        for &use_simd in &[false, true] {
          let mut rgb = std::vec![0u8; w * 3];
          vuyx_to_rgb_row(&buf, &mut rgb, w, matrix, full_range, use_simd);
          let want = ref_hsv_from_rgb(&rgb, w, use_simd);

          let mut h = std::vec![0u8; w];
          let mut s = std::vec![0u8; w];
          let mut v = std::vec![0u8; w];
          vuyx_to_hsv_row(
            &buf, &mut h, &mut s, &mut v, w, matrix, full_range, use_simd,
          );
          assert_planes_eq!(
            (h, s, v),
            want,
            std::format!("vuyx w={w} {matrix:?} full={full_range} simd={use_simd}")
          );
        }
      }
    }
  }
}

// ---- Structural: HSV-only attaches no source-width RGB scratch ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn packed_yuv444_hsv_only_grows_no_rgb_scratch_all_formats() {
  let (w, h) = (16usize, 8usize);

  // AYUV64 — 16-bit packed, slot order [A, Y, U, V].
  {
    let mut buf = Vec::with_capacity(w * h * 4);
    for i in 0..w * h {
      buf.push(samp16(i, 0));
      buf.push(samp16(i, 1));
      buf.push(samp16(i, 2));
      buf.push(samp16(i, 3));
    }
    let src = Ayuv64Frame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Ayuv64>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      ayuv64_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(
      scratch_len, 0,
      "Ayuv64 HSV-only must not grow the RGB scratch"
    );

    // Cross-check row 0 against the explicit YUV→RGB→HSV reference (LE).
    let mut rgb0 = std::vec![0u8; w * 3];
    ayuv64_to_rgb_row(
      &buf[..w * 4],
      &mut rgb0,
      w,
      KernelMatrix::Bt709,
      true,
      true,
      false,
    );
    let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
    assert_eq!(&hh[..w], &rh[..], "ayuv64 row 0 H");
    assert_eq!(&ss[..w], &rs[..], "ayuv64 row 0 S");
    assert_eq!(&vv[..w], &rv[..], "ayuv64 row 0 V");
  }

  // XV36 — 12-bit MSB-aligned packed, slot order [U, Y, V, A].
  {
    let mut buf = Vec::with_capacity(w * h * 4);
    for i in 0..w * h {
      buf.push((samp16(i, 5) & 0x0FFF) << 4);
      buf.push((samp16(i, 6) & 0x0FFF) << 4);
      buf.push((samp16(i, 7) & 0x0FFF) << 4);
      buf.push((samp16(i, 8) & 0x0FFF) << 4);
    }
    let src = Xv36Frame::new(&buf, w as u32, h as u32, (w * 4) as u32);
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Xv36>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      xv36_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(
      scratch_len, 0,
      "Xv36 HSV-only must not grow the RGB scratch"
    );
  }

  // VUYA — 8-bit packed, byte order [V, U, Y, A].
  {
    let buf = vuya_packed(w * h);
    let src = VuyaFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Vuya>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      vuya_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(
      scratch_len, 0,
      "Vuya HSV-only must not grow the RGB scratch"
    );
  }

  // VUYX — 8-bit packed, byte order [V, U, Y, X] (X padding).
  {
    let buf = vuya_packed(w * h);
    let src = VuyxFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();
    let mut hh = std::vec![0u8; w * h];
    let mut ss = std::vec![0u8; w * h];
    let mut vv = std::vec![0u8; w * h];
    let scratch_len = {
      let mut sink = MixedSinker::<Vuyx>::new(w, h)
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
      vuyx_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      sink.rgb_scratch.len()
    };
    assert_eq!(
      scratch_len, 0,
      "Vuyx HSV-only must not grow the RGB scratch"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_luma_plus_hsv_only_is_correct_and_rgb_free() {
  // with_luma() + with_hsv(), no RGB: native Y luma AND direct HSV with
  // no source-width RGB scratch.
  let (w, h) = (16usize, 8usize);
  let buf = vuya_packed(w * h);
  let src = VuyaFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut luma = std::vec![0u8; w * h];
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<Vuya>::new(w, h)
      .with_luma(&mut luma)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    vuya_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    sink.rgb_scratch.len()
  };

  assert_eq!(
    scratch_len, 0,
    "luma + HSV (no RGB) must not grow the RGB scratch"
  );
  // Native Y luma: Y is at byte offset 2 of each quadruple, copied verbatim.
  for (i, &got) in luma.iter().enumerate() {
    assert_eq!(got, buf[i * 4 + 2], "vuya luma row pixel {i}");
  }
  // HSV row 0 matches the two-step reference.
  let mut rgb0 = std::vec![0u8; w * 3];
  vuya_to_rgb_row(
    &buf[..w * 4],
    &mut rgb0,
    w,
    KernelMatrix::Bt709,
    true,
    false,
  );
  let (rh, rs, rv) = ref_hsv_from_rgb(&rgb0, w, true);
  assert_eq!(&hh[..w], &rh[..], "vuya row 0 H");
  assert_eq!(&ss[..w], &rs[..], "vuya row 0 S");
  assert_eq!(&vv[..w], &rv[..], "vuya row 0 V");
}
