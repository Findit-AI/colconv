//! Phase 4 — YUV-HB megatier LE/BE round-trip parity tests.
//!
//! Mirrors the Tier 8 (`packed_rgb_16bit`) pattern: encode the same
//! logical samples as LE bytes and BE bytes, walk both through their
//! `<const BE>` sinker monomorphizations, assert byte-identical
//! output. This catches:
//!
//!   - missing `<BE>` propagation in walker → sinker call sites,
//!   - regressions in the `*_endian` row-kernel byte-swap path,
//!   - mismatches between `MixedSinker<Marker<true>>` and the BE row
//!     kernels.
//!
//! The tests cover all 30+ migrated formats via a small helper-macro
//! pattern that reduces each format to a 5-line test body. Per-format
//! helpers (`as_le_u16` / `as_be_u16` / matching intended-sample
//! generation) live at module scope.

use crate::{
  KernelMatrix,
  frame::{
    Yuv420p9BeFrame, Yuv420p9LeFrame, Yuv420p10BeFrame, Yuv420p10LeFrame, Yuv420p12BeFrame,
    Yuv420p12LeFrame, Yuv420p14BeFrame, Yuv420p14LeFrame, Yuv420p16BeFrame, Yuv420p16LeFrame,
    Yuv422p9BeFrame, Yuv422p9LeFrame, Yuv422p10BeFrame, Yuv422p10LeFrame, Yuv422p12BeFrame,
    Yuv422p12LeFrame, Yuv422p14BeFrame, Yuv422p14LeFrame, Yuv422p16BeFrame, Yuv422p16LeFrame,
    Yuv440p10BeFrame, Yuv440p10LeFrame, Yuv440p12BeFrame, Yuv440p12LeFrame, Yuv444p9BeFrame,
    Yuv444p9LeFrame, Yuv444p10BeFrame, Yuv444p10LeFrame, Yuv444p12BeFrame, Yuv444p12LeFrame,
    Yuv444p14BeFrame, Yuv444p14LeFrame, Yuv444p16BeFrame, Yuv444p16LeFrame,
  },
  sinker::mixed::MixedSinker,
  source::{
    Yuv420p9, Yuv420p10, Yuv420p12, Yuv420p14, Yuv420p16, Yuv422p9, Yuv422p10, Yuv422p12,
    Yuv422p14, Yuv422p16, Yuv440p10, Yuv440p12, Yuv444p9, Yuv444p10, Yuv444p12, Yuv444p14,
    Yuv444p16, yuv420p9_to, yuv420p9_to_endian, yuv420p10_to, yuv420p10_to_endian, yuv420p12_to,
    yuv420p12_to_endian, yuv420p14_to, yuv420p14_to_endian, yuv420p16_to, yuv420p16_to_endian,
    yuv422p9_to, yuv422p9_to_endian, yuv422p10_to, yuv422p10_to_endian, yuv422p12_to,
    yuv422p12_to_endian, yuv422p14_to, yuv422p14_to_endian, yuv422p16_to, yuv422p16_to_endian,
    yuv440p10_to, yuv440p10_to_endian, yuv440p12_to, yuv440p12_to_endian, yuv444p9_to,
    yuv444p9_to_endian, yuv444p10_to, yuv444p10_to_endian, yuv444p12_to, yuv444p12_to_endian,
    yuv444p14_to, yuv444p14_to_endian, yuv444p16_to, yuv444p16_to_endian,
  },
};
// Semi-planar (`P0xx` / `P2xx` / `P4xx`) frame / source / walker imports —
// used only by the `yuv-semi-planar`-gated `pn_be_roundtrip_test!`
// invocations below.
#[cfg(feature = "yuv-semi-planar")]
use crate::{
  frame::{
    P010BeFrame, P010LeFrame, P012BeFrame, P012LeFrame, P016BeFrame, P016LeFrame, P210BeFrame,
    P210LeFrame, P212BeFrame, P212LeFrame, P216BeFrame, P216LeFrame, P410BeFrame, P410LeFrame,
    P412BeFrame, P412LeFrame, P416BeFrame, P416LeFrame,
  },
  source::{
    P010, P012, P016, P210, P212, P216, P410, P412, P416, p010_to, p010_to_endian, p012_to,
    p012_to_endian, p016_to, p016_to_endian, p210_to, p210_to_endian, p212_to, p212_to_endian,
    p216_to, p216_to_endian, p410_to, p410_to_endian, p412_to, p412_to_endian, p416_to,
    p416_to_endian,
  },
};
// YUVA frame / source / walker imports — used only by the `yuva`-gated
// `planar4_be_roundtrip_test!` invocations below.
#[cfg(feature = "yuva")]
use crate::{
  frame::{
    Yuva420p9BeFrame, Yuva420p9LeFrame, Yuva420p10BeFrame, Yuva420p10LeFrame, Yuva420p12BeFrame,
    Yuva420p12LeFrame, Yuva420p16BeFrame, Yuva420p16LeFrame, Yuva422p9BeFrame, Yuva422p9LeFrame,
    Yuva422p10BeFrame, Yuva422p10LeFrame, Yuva422p12BeFrame, Yuva422p12LeFrame, Yuva422p16BeFrame,
    Yuva422p16LeFrame, Yuva444p9BeFrame, Yuva444p9LeFrame, Yuva444p10BeFrame, Yuva444p10LeFrame,
    Yuva444p12BeFrame, Yuva444p12LeFrame, Yuva444p14BeFrame, Yuva444p14LeFrame, Yuva444p16BeFrame,
    Yuva444p16LeFrame,
  },
  source::{
    Yuva420p9, Yuva420p10, Yuva420p12, Yuva420p16, Yuva422p9, Yuva422p10, Yuva422p12, Yuva422p16,
    Yuva444p9, Yuva444p10, Yuva444p12, Yuva444p14, Yuva444p16, yuva420p9_to, yuva420p9_to_endian,
    yuva420p10_to, yuva420p10_to_endian, yuva420p12_to, yuva420p12_to_endian, yuva420p16_to,
    yuva420p16_to_endian, yuva422p9_to, yuva422p9_to_endian, yuva422p10_to, yuva422p10_to_endian,
    yuva422p12_to, yuva422p12_to_endian, yuva422p16_to, yuva422p16_to_endian, yuva444p9_to,
    yuva444p9_to_endian, yuva444p10_to, yuva444p10_to_endian, yuva444p12_to, yuva444p12_to_endian,
    yuva444p14_to, yuva444p14_to_endian, yuva444p16_to, yuva444p16_to_endian,
  },
};

// ---- shared encoding helpers ----------------------------------------------

/// Re-encode a host-native `u16` slice as LE byte storage. On LE hosts
/// `from_le` is a no-op; on BE hosts the kernels byte-swap before use.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native `u16` slice as BE byte storage. The mirror
/// of [`as_le_u16`].
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Generate `n` deterministic `u16` samples bounded to `max_value`
/// (inclusive). The pattern is a low-rate cyclic mix that exercises
/// both low and high byte positions in each `u16` to surface
/// byte-swap regressions.
fn intended_samples(n: usize, max_value: u16) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| {
      let v = match i % 5 {
        0 => 0x12u16,
        1 => 0xABu16,
        2 => 0x7Fu16,
        3 => 0x01u16,
        _ => 0xFEu16,
      } as u32
        | ((i as u32 * 0x0101) & 0xFF00);
      // Mask down to format depth so input is a valid sample (no
      // upper-bit garbage that would change with byte order).
      (v as u16) & max_value
    })
    .collect()
}

// ---- planar 4:2:0 / 4:2:2 / 4:4:4 / 4:4:0 high-bit YUV --------------------

/// Macro: generate a single LE/BE round-trip test for a 3-plane planar
/// high-bit YUV format. `$frame_le` / `$frame_be` are the per-format
/// `*LeFrame` / `*BeFrame` aliases; `$marker_be_ty` is the marker's
/// BE specialization (e.g. `Yuv420p10<true>`); `$walker` is the
/// walker fn; `$max_value` bounds the per-sample value to the
/// format's `(1 << BITS) - 1`.
///
/// 4:2:2 and 4:4:0 use the same chroma-half-width/full-height /
/// full-width-half-height layouts as 4:2:0, with the macro
/// substituting the per-axis chroma sizes.
macro_rules! planar3_be_roundtrip_test {
  (
    $name:ident,
    frame_le = $frame_le:ident,
    frame_be = $frame_be:ident,
    marker_le = $marker_le:ty,
    marker_be = $marker_be:ty,
    walker = $walker:expr,
    walker_endian = $walker_endian:ident,
    max_value = $max_value:expr,
    chroma_w_div = $chroma_w_div:expr,
    chroma_h_div = $chroma_h_div:expr,
  ) => {
    #[test]
    fn $name() {
      let w: u32 = 16;
      let h: u32 = 4;
      let cw = (w as usize) / $chroma_w_div;
      let ch = (h as usize).div_ceil($chroma_h_div);
      let max_v: u16 = $max_value;

      let y_intended = intended_samples((w * h) as usize, max_v);
      let u_intended = intended_samples(cw * ch, max_v);
      let v_intended = intended_samples(cw * ch, max_v);

      let y_le = as_le_u16(&y_intended);
      let u_le = as_le_u16(&u_intended);
      let v_le = as_le_u16(&v_intended);
      let y_be = as_be_u16(&y_intended);
      let u_be = as_be_u16(&u_intended);
      let v_be = as_be_u16(&v_intended);

      let frame_le =
        $frame_le::try_new(&y_le, &u_le, &v_le, w, h, w, cw as u32, cw as u32).unwrap();
      let mut out_le = vec![0u8; (w * h * 4) as usize];
      let mut luma_le = vec![0u8; (w * h) as usize];
      let mut sink_le = MixedSinker::<$marker_le>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_le)
        .unwrap()
        .with_luma(&mut luma_le)
        .unwrap();
      $walker(
        &frame_le,
        true,
        sink_le.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      let frame_be =
        $frame_be::try_new(&y_be, &u_be, &v_be, w, h, w, cw as u32, cw as u32).unwrap();
      let mut out_be = vec![0u8; (w * h * 4) as usize];
      let mut luma_be = vec![0u8; (w * h) as usize];
      let mut sink_be = MixedSinker::<$marker_be>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_be)
        .unwrap()
        .with_luma(&mut luma_be)
        .unwrap();
      $walker_endian(
        &frame_be,
        true,
        sink_be.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      assert_eq!(
        out_le,
        out_be,
        "{} LE/BE outputs diverge — `<const BE>` propagation broken",
        stringify!($name),
      );
      assert_eq!(
        luma_le,
        luma_be,
        "{} LE/BE luma diverge — BE luma fast-path normalization broken",
        stringify!($name),
      );
    }
  };
}

planar3_be_roundtrip_test!(
  yuv420p9_le_be_roundtrip_byte_identical,
  frame_le = Yuv420p9LeFrame, frame_be = Yuv420p9BeFrame,
  marker_le = Yuv420p9, marker_be = Yuv420p9<true>, walker = yuv420p9_to, walker_endian = yuv420p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 2, chroma_h_div = 2,
);
planar3_be_roundtrip_test!(
  yuv420p10_le_be_roundtrip_byte_identical,
  frame_le = Yuv420p10LeFrame, frame_be = Yuv420p10BeFrame,
  marker_le = Yuv420p10, marker_be = Yuv420p10<true>, walker = yuv420p10_to, walker_endian = yuv420p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 2, chroma_h_div = 2,
);
planar3_be_roundtrip_test!(
  yuv420p12_le_be_roundtrip_byte_identical,
  frame_le = Yuv420p12LeFrame, frame_be = Yuv420p12BeFrame,
  marker_le = Yuv420p12, marker_be = Yuv420p12<true>, walker = yuv420p12_to, walker_endian = yuv420p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 2, chroma_h_div = 2,
);
planar3_be_roundtrip_test!(
  yuv420p14_le_be_roundtrip_byte_identical,
  frame_le = Yuv420p14LeFrame, frame_be = Yuv420p14BeFrame,
  marker_le = Yuv420p14, marker_be = Yuv420p14<true>, walker = yuv420p14_to, walker_endian = yuv420p14_to_endian,
  max_value = 0x3FFF, chroma_w_div = 2, chroma_h_div = 2,
);
planar3_be_roundtrip_test!(
  yuv420p16_le_be_roundtrip_byte_identical,
  frame_le = Yuv420p16LeFrame, frame_be = Yuv420p16BeFrame,
  marker_le = Yuv420p16, marker_be = Yuv420p16<true>, walker = yuv420p16_to, walker_endian = yuv420p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 2, chroma_h_div = 2,
);

planar3_be_roundtrip_test!(
  yuv422p9_le_be_roundtrip_byte_identical,
  frame_le = Yuv422p9LeFrame, frame_be = Yuv422p9BeFrame,
  marker_le = Yuv422p9, marker_be = Yuv422p9<true>, walker = yuv422p9_to, walker_endian = yuv422p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 2, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv422p10_le_be_roundtrip_byte_identical,
  frame_le = Yuv422p10LeFrame, frame_be = Yuv422p10BeFrame,
  marker_le = Yuv422p10, marker_be = Yuv422p10<true>, walker = yuv422p10_to, walker_endian = yuv422p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 2, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv422p12_le_be_roundtrip_byte_identical,
  frame_le = Yuv422p12LeFrame, frame_be = Yuv422p12BeFrame,
  marker_le = Yuv422p12, marker_be = Yuv422p12<true>, walker = yuv422p12_to, walker_endian = yuv422p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 2, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv422p14_le_be_roundtrip_byte_identical,
  frame_le = Yuv422p14LeFrame, frame_be = Yuv422p14BeFrame,
  marker_le = Yuv422p14, marker_be = Yuv422p14<true>, walker = yuv422p14_to, walker_endian = yuv422p14_to_endian,
  max_value = 0x3FFF, chroma_w_div = 2, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv422p16_le_be_roundtrip_byte_identical,
  frame_le = Yuv422p16LeFrame, frame_be = Yuv422p16BeFrame,
  marker_le = Yuv422p16, marker_be = Yuv422p16<true>, walker = yuv422p16_to, walker_endian = yuv422p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 2, chroma_h_div = 1,
);

planar3_be_roundtrip_test!(
  yuv444p9_le_be_roundtrip_byte_identical,
  frame_le = Yuv444p9LeFrame, frame_be = Yuv444p9BeFrame,
  marker_le = Yuv444p9, marker_be = Yuv444p9<true>, walker = yuv444p9_to, walker_endian = yuv444p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 1, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv444p10_le_be_roundtrip_byte_identical,
  frame_le = Yuv444p10LeFrame, frame_be = Yuv444p10BeFrame,
  marker_le = Yuv444p10, marker_be = Yuv444p10<true>, walker = yuv444p10_to, walker_endian = yuv444p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 1, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv444p12_le_be_roundtrip_byte_identical,
  frame_le = Yuv444p12LeFrame, frame_be = Yuv444p12BeFrame,
  marker_le = Yuv444p12, marker_be = Yuv444p12<true>, walker = yuv444p12_to, walker_endian = yuv444p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 1, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv444p14_le_be_roundtrip_byte_identical,
  frame_le = Yuv444p14LeFrame, frame_be = Yuv444p14BeFrame,
  marker_le = Yuv444p14, marker_be = Yuv444p14<true>, walker = yuv444p14_to, walker_endian = yuv444p14_to_endian,
  max_value = 0x3FFF, chroma_w_div = 1, chroma_h_div = 1,
);
planar3_be_roundtrip_test!(
  yuv444p16_le_be_roundtrip_byte_identical,
  frame_le = Yuv444p16LeFrame, frame_be = Yuv444p16BeFrame,
  marker_le = Yuv444p16, marker_be = Yuv444p16<true>, walker = yuv444p16_to, walker_endian = yuv444p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 1, chroma_h_div = 1,
);

planar3_be_roundtrip_test!(
  yuv440p10_le_be_roundtrip_byte_identical,
  frame_le = Yuv440p10LeFrame, frame_be = Yuv440p10BeFrame,
  marker_le = Yuv440p10, marker_be = Yuv440p10<true>, walker = yuv440p10_to, walker_endian = yuv440p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 1, chroma_h_div = 2,
);
planar3_be_roundtrip_test!(
  yuv440p12_le_be_roundtrip_byte_identical,
  frame_le = Yuv440p12LeFrame, frame_be = Yuv440p12BeFrame,
  marker_le = Yuv440p12, marker_be = Yuv440p12<true>, walker = yuv440p12_to, walker_endian = yuv440p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 1, chroma_h_div = 2,
);

// ---- planar4 (YUVA) high-bit ---------------------------------------------
//
// The generated test fns are `yuva`-gated (inside the expansion) so the
// invocations below stay unconditional while a `yuv-planar`-without-`yuva`
// build configures out every YUVA fn (and its YUVA-type references) before
// name resolution.

macro_rules! planar4_be_roundtrip_test {
  (
    $name:ident,
    frame_le = $frame_le:ident,
    frame_be = $frame_be:ident,
    marker_le = $marker_le:ty,
    marker_be = $marker_be:ty,
    walker = $walker:expr,
    walker_endian = $walker_endian:ident,
    max_value = $max_value:expr,
    chroma_w_div = $chroma_w_div:expr,
    chroma_h_div = $chroma_h_div:expr,
  ) => {
    #[cfg(feature = "yuva")]
    #[test]
    fn $name() {
      let w: u32 = 16;
      let h: u32 = 4;
      let cw = (w as usize) / $chroma_w_div;
      let ch = (h as usize).div_ceil($chroma_h_div);
      let max_v: u16 = $max_value;

      let y_intended = intended_samples((w * h) as usize, max_v);
      let u_intended = intended_samples(cw * ch, max_v);
      let v_intended = intended_samples(cw * ch, max_v);
      let a_intended = intended_samples((w * h) as usize, max_v);

      let y_le = as_le_u16(&y_intended);
      let u_le = as_le_u16(&u_intended);
      let v_le = as_le_u16(&v_intended);
      let a_le = as_le_u16(&a_intended);
      let y_be = as_be_u16(&y_intended);
      let u_be = as_be_u16(&u_intended);
      let v_be = as_be_u16(&v_intended);
      let a_be = as_be_u16(&a_intended);

      let frame_le =
        $frame_le::try_new(&y_le, &u_le, &v_le, &a_le, w, h, w, cw as u32, cw as u32, w).unwrap();
      let mut out_le = vec![0u8; (w * h * 4) as usize];
      let mut luma_le = vec![0u8; (w * h) as usize];
      let mut sink_le = MixedSinker::<$marker_le>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_le)
        .unwrap()
        .with_luma(&mut luma_le)
        .unwrap();
      $walker(
        &frame_le,
        true,
        sink_le.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      let frame_be =
        $frame_be::try_new(&y_be, &u_be, &v_be, &a_be, w, h, w, cw as u32, cw as u32, w).unwrap();
      let mut out_be = vec![0u8; (w * h * 4) as usize];
      let mut luma_be = vec![0u8; (w * h) as usize];
      let mut sink_be = MixedSinker::<$marker_be>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_be)
        .unwrap()
        .with_luma(&mut luma_be)
        .unwrap();
      $walker_endian(
        &frame_be,
        true,
        sink_be.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      assert_eq!(
        out_le,
        out_be,
        "{} LE/BE outputs diverge — `<const BE>` propagation broken",
        stringify!($name),
      );
      assert_eq!(
        luma_le,
        luma_be,
        "{} LE/BE luma diverge — BE luma fast-path normalization broken",
        stringify!($name),
      );
    }
  };
}

planar4_be_roundtrip_test!(
  yuva420p9_le_be_roundtrip_byte_identical,
  frame_le = Yuva420p9LeFrame, frame_be = Yuva420p9BeFrame,
  marker_le = Yuva420p9, marker_be = Yuva420p9<true>, walker = yuva420p9_to, walker_endian = yuva420p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 2, chroma_h_div = 2,
);
planar4_be_roundtrip_test!(
  yuva420p10_le_be_roundtrip_byte_identical,
  frame_le = Yuva420p10LeFrame, frame_be = Yuva420p10BeFrame,
  marker_le = Yuva420p10, marker_be = Yuva420p10<true>, walker = yuva420p10_to, walker_endian = yuva420p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 2, chroma_h_div = 2,
);
planar4_be_roundtrip_test!(
  yuva420p12_le_be_roundtrip_byte_identical,
  frame_le = Yuva420p12LeFrame, frame_be = Yuva420p12BeFrame,
  marker_le = Yuva420p12, marker_be = Yuva420p12<true>, walker = yuva420p12_to, walker_endian = yuva420p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 2, chroma_h_div = 2,
);
planar4_be_roundtrip_test!(
  yuva420p16_le_be_roundtrip_byte_identical,
  frame_le = Yuva420p16LeFrame, frame_be = Yuva420p16BeFrame,
  marker_le = Yuva420p16, marker_be = Yuva420p16<true>, walker = yuva420p16_to, walker_endian = yuva420p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 2, chroma_h_div = 2,
);

planar4_be_roundtrip_test!(
  yuva422p9_le_be_roundtrip_byte_identical,
  frame_le = Yuva422p9LeFrame, frame_be = Yuva422p9BeFrame,
  marker_le = Yuva422p9, marker_be = Yuva422p9<true>, walker = yuva422p9_to, walker_endian = yuva422p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 2, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva422p10_le_be_roundtrip_byte_identical,
  frame_le = Yuva422p10LeFrame, frame_be = Yuva422p10BeFrame,
  marker_le = Yuva422p10, marker_be = Yuva422p10<true>, walker = yuva422p10_to, walker_endian = yuva422p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 2, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva422p12_le_be_roundtrip_byte_identical,
  frame_le = Yuva422p12LeFrame, frame_be = Yuva422p12BeFrame,
  marker_le = Yuva422p12, marker_be = Yuva422p12<true>, walker = yuva422p12_to, walker_endian = yuva422p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 2, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva422p16_le_be_roundtrip_byte_identical,
  frame_le = Yuva422p16LeFrame, frame_be = Yuva422p16BeFrame,
  marker_le = Yuva422p16, marker_be = Yuva422p16<true>, walker = yuva422p16_to, walker_endian = yuva422p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 2, chroma_h_div = 1,
);

planar4_be_roundtrip_test!(
  yuva444p9_le_be_roundtrip_byte_identical,
  frame_le = Yuva444p9LeFrame, frame_be = Yuva444p9BeFrame,
  marker_le = Yuva444p9, marker_be = Yuva444p9<true>, walker = yuva444p9_to, walker_endian = yuva444p9_to_endian,
  max_value = 0x01FF, chroma_w_div = 1, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva444p10_le_be_roundtrip_byte_identical,
  frame_le = Yuva444p10LeFrame, frame_be = Yuva444p10BeFrame,
  marker_le = Yuva444p10, marker_be = Yuva444p10<true>, walker = yuva444p10_to, walker_endian = yuva444p10_to_endian,
  max_value = 0x03FF, chroma_w_div = 1, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva444p12_le_be_roundtrip_byte_identical,
  frame_le = Yuva444p12LeFrame, frame_be = Yuva444p12BeFrame,
  marker_le = Yuva444p12, marker_be = Yuva444p12<true>, walker = yuva444p12_to, walker_endian = yuva444p12_to_endian,
  max_value = 0x0FFF, chroma_w_div = 1, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva444p14_le_be_roundtrip_byte_identical,
  frame_le = Yuva444p14LeFrame, frame_be = Yuva444p14BeFrame,
  marker_le = Yuva444p14, marker_be = Yuva444p14<true>, walker = yuva444p14_to, walker_endian = yuva444p14_to_endian,
  max_value = 0x3FFF, chroma_w_div = 1, chroma_h_div = 1,
);
planar4_be_roundtrip_test!(
  yuva444p16_le_be_roundtrip_byte_identical,
  frame_le = Yuva444p16LeFrame, frame_be = Yuva444p16BeFrame,
  marker_le = Yuva444p16, marker_be = Yuva444p16<true>, walker = yuva444p16_to, walker_endian = yuva444p16_to_endian,
  max_value = 0xFFFF, chroma_w_div = 1, chroma_h_div = 1,
);

// ---- semi-planar (Pn) ----------------------------------------------------

/// `chroma_w_factor`: u16 elements per row of UV plane = `factor * width`.
/// 4:2:0 / 4:2:2: 1 (half-width x {two, one} U,V pairs interleaved =
/// `width` elements). 4:4:4: 2 (full-width pairs = `2 * width`).
macro_rules! pn_be_roundtrip_test {
  (
    $name:ident,
    frame_le = $frame_le:ident,
    frame_be = $frame_be:ident,
    marker_le = $marker_le:ty,
    marker_be = $marker_be:ty,
    walker = $walker:expr,
    walker_endian = $walker_endian:ident,
    max_value = $max_value:expr,
    chroma_w_factor = $chroma_w_factor:expr,
    chroma_h_div = $chroma_h_div:expr,
  ) => {
    #[cfg(feature = "yuv-semi-planar")]
    #[test]
    fn $name() {
      let w: u32 = 16;
      let h: u32 = 4;
      let uv_row_elems = ($chroma_w_factor as usize) * (w as usize);
      let ch = (h as usize).div_ceil($chroma_h_div);
      let max_v: u16 = $max_value;

      let y_intended = intended_samples((w * h) as usize, max_v);
      let uv_intended = intended_samples(uv_row_elems * ch, max_v);

      let y_le = as_le_u16(&y_intended);
      let uv_le = as_le_u16(&uv_intended);
      let y_be = as_be_u16(&y_intended);
      let uv_be = as_be_u16(&uv_intended);

      let frame_le = $frame_le::try_new(&y_le, &uv_le, w, h, w, uv_row_elems as u32).unwrap();
      let mut out_le = vec![0u8; (w * h * 4) as usize];
      let mut luma_le = vec![0u8; (w * h) as usize];
      let mut sink_le = MixedSinker::<$marker_le>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_le)
        .unwrap()
        .with_luma(&mut luma_le)
        .unwrap();
      $walker(
        &frame_le,
        true,
        sink_le.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      let frame_be = $frame_be::try_new(&y_be, &uv_be, w, h, w, uv_row_elems as u32).unwrap();
      let mut out_be = vec![0u8; (w * h * 4) as usize];
      let mut luma_be = vec![0u8; (w * h) as usize];
      let mut sink_be = MixedSinker::<$marker_be>::new(w as usize, h as usize)
        .with_simd(false)
        .with_rgba(&mut out_be)
        .unwrap()
        .with_luma(&mut luma_be)
        .unwrap();
      $walker_endian(
        &frame_be,
        true,
        sink_be.set_kernel_matrix(KernelMatrix::Bt709),
      )
      .unwrap();

      assert_eq!(
        out_le,
        out_be,
        "{} LE/BE outputs diverge — `<const BE>` propagation broken",
        stringify!($name),
      );
      assert_eq!(
        luma_le,
        luma_be,
        "{} LE/BE luma diverge — BE luma fast-path normalization broken",
        stringify!($name),
      );
    }
  };
}

pn_be_roundtrip_test!(
  p010_le_be_roundtrip_byte_identical,
  frame_le = P010LeFrame, frame_be = P010BeFrame,
  marker_le = P010, marker_be = P010<true>, walker = p010_to, walker_endian = p010_to_endian,
  max_value = 0xFFC0, chroma_w_factor = 1, chroma_h_div = 2,
);
pn_be_roundtrip_test!(
  p012_le_be_roundtrip_byte_identical,
  frame_le = P012LeFrame, frame_be = P012BeFrame,
  marker_le = P012, marker_be = P012<true>, walker = p012_to, walker_endian = p012_to_endian,
  max_value = 0xFFF0, chroma_w_factor = 1, chroma_h_div = 2,
);
pn_be_roundtrip_test!(
  p016_le_be_roundtrip_byte_identical,
  frame_le = P016LeFrame, frame_be = P016BeFrame,
  marker_le = P016, marker_be = P016<true>, walker = p016_to, walker_endian = p016_to_endian,
  max_value = 0xFFFF, chroma_w_factor = 1, chroma_h_div = 2,
);
pn_be_roundtrip_test!(
  p210_le_be_roundtrip_byte_identical,
  frame_le = P210LeFrame, frame_be = P210BeFrame,
  marker_le = P210, marker_be = P210<true>, walker = p210_to, walker_endian = p210_to_endian,
  max_value = 0xFFC0, chroma_w_factor = 1, chroma_h_div = 1,
);
pn_be_roundtrip_test!(
  p212_le_be_roundtrip_byte_identical,
  frame_le = P212LeFrame, frame_be = P212BeFrame,
  marker_le = P212, marker_be = P212<true>, walker = p212_to, walker_endian = p212_to_endian,
  max_value = 0xFFF0, chroma_w_factor = 1, chroma_h_div = 1,
);
pn_be_roundtrip_test!(
  p216_le_be_roundtrip_byte_identical,
  frame_le = P216LeFrame, frame_be = P216BeFrame,
  marker_le = P216, marker_be = P216<true>, walker = p216_to, walker_endian = p216_to_endian,
  max_value = 0xFFFF, chroma_w_factor = 1, chroma_h_div = 1,
);
pn_be_roundtrip_test!(
  p410_le_be_roundtrip_byte_identical,
  frame_le = P410LeFrame, frame_be = P410BeFrame,
  marker_le = P410, marker_be = P410<true>, walker = p410_to, walker_endian = p410_to_endian,
  max_value = 0xFFC0, chroma_w_factor = 2, chroma_h_div = 1,
);
pn_be_roundtrip_test!(
  p412_le_be_roundtrip_byte_identical,
  frame_le = P412LeFrame, frame_be = P412BeFrame,
  marker_le = P412, marker_be = P412<true>, walker = p412_to, walker_endian = p412_to_endian,
  max_value = 0xFFF0, chroma_w_factor = 2, chroma_h_div = 1,
);
pn_be_roundtrip_test!(
  p416_le_be_roundtrip_byte_identical,
  frame_le = P416LeFrame, frame_be = P416BeFrame,
  marker_le = P416, marker_be = P416<true>, walker = p416_to, walker_endian = p416_to_endian,
  max_value = 0xFFFF, chroma_w_factor = 2, chroma_h_div = 1,
);
