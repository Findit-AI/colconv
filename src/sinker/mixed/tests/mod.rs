#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use super::*;
#[cfg(any(
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::ColorMatrix;
// `frame::*` glob is consumed only by the families whose test files
// reference unqualified `*Frame` types; the Bayer tests import their
// frame types locally, so `bayer` is not in this set.
#[cfg(any(
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::frame::*;
// `source::*` carries the source markers (`Bayer` / `Bayer16` / …) the
// test files name unqualified — Bayer included.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::source::*;

// Per-format-family submodules. Each houses tests + format-local
// helpers (`solid_*_frame` builders); cross-cutting helpers
// (`pseudo_random_u8`, `pseudo_random_u16_low_n_bits`) live at
// module scope below and are re-exported as `pub(super)` so the
// submodules can pull them via `use super::*;`.
#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(feature = "bayer")]
mod bayer;
#[cfg(feature = "yuv-planar")]
mod chroma_derived_cl;
#[cfg(feature = "yuv-planar")]
mod chroma_siting_420;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_420_resample;
#[cfg(feature = "yuv-planar")]
mod chroma_siting_422;
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
mod chroma_siting_422_packed;
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_422_packed_resample;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_422_resample;
#[cfg(feature = "yuva")]
mod chroma_siting_422_yuva;
#[cfg(feature = "yuv-planar")]
mod chroma_siting_hibit_420;
#[cfg(feature = "yuv-planar")]
mod chroma_siting_hibit_422;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_hibit_422_resample;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_hibit_422_semiplanar_resample;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
mod chroma_siting_nv;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_nv12_resample;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
mod chroma_siting_nv16;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar", feature = "rgb"))]
mod chroma_siting_nv16_resample;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
mod chroma_siting_p0xx;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
mod chroma_siting_p2xx;
#[cfg(feature = "yuva")]
mod chroma_siting_yuva;
#[cfg(all(feature = "yuva", feature = "rgb"))]
mod chroma_siting_yuva420p_resample;
#[cfg(feature = "gray")]
mod gray;
#[cfg(feature = "rgb-legacy")]
mod legacy_rgb;
#[cfg(feature = "mono")]
mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
mod packed_rgb_32bit;
#[cfg(feature = "rgb")]
mod packed_rgb_8bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_f16;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "rgb-float")]
mod packed_rgba_f16;
#[cfg(feature = "rgb-float")]
mod packed_rgba_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_hsv_direct;
#[cfg(feature = "mono")]
mod pal8;
#[cfg(feature = "yuv-planar")]
mod phase4_yuv_hb_be_roundtrip;
#[cfg(feature = "gbr")]
mod planar_gbr;
#[cfg(feature = "gbr")]
mod planar_gbr_32bit;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "gbr")]
mod planar_gbr_msb;
#[cfg(feature = "yuv-planar")]
mod planar_other_8bit_9bit;
#[cfg(feature = "yuv-444-packed")]
mod resample_ayuv64;
#[cfg(feature = "bayer")]
mod resample_bayer8;
#[cfg(feature = "rgb")]
mod resample_bgr24;
#[cfg(feature = "gbr")]
mod resample_gbr_high_bit;
#[cfg(feature = "gbr")]
mod resample_gbrap_32bit;
#[cfg(feature = "gbr")]
mod resample_gbrap_8bit;
#[cfg(feature = "gbr")]
mod resample_gbrap_high_bit;
#[cfg(feature = "gbr")]
mod resample_gbrapf16;
#[cfg(feature = "gbr")]
mod resample_gbrapf16_filter;
#[cfg(feature = "gbr")]
mod resample_gbrapf32;
#[cfg(feature = "gbr")]
mod resample_gbrapf32_filter;
#[cfg(feature = "gbr")]
mod resample_gbrp;
#[cfg(feature = "gbr")]
mod resample_gbrpf16;
#[cfg(feature = "gbr")]
mod resample_gbrpf16_filter;
#[cfg(feature = "gbr")]
mod resample_gbrpf32;
#[cfg(feature = "gbr")]
mod resample_gbrpf32_filter;
#[cfg(feature = "yuv-planar")]
mod resample_geometry;
#[cfg(feature = "gray")]
mod resample_gray16;
#[cfg(feature = "gray")]
mod resample_gray32;
#[cfg(feature = "gray")]
mod resample_gray8;
#[cfg(feature = "gray")]
mod resample_gray_n;
#[cfg(feature = "gray")]
mod resample_grayf16;
#[cfg(feature = "gray")]
mod resample_grayf32;
#[cfg(feature = "rgb-legacy")]
mod resample_legacy_rgb;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_linear_domain;
#[cfg(all(feature = "yuv-planar", not(feature = "rgb")))]
mod resample_linear_domain_no_rgb;
#[cfg(feature = "mono")]
mod resample_mono;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_scene_linear_domain;
// The high-bit semi-planar P-format sinks live under `yuv-semi-planar`
// (the `subsampled_4_*_high_bit` parents now compile under either family);
// these area + filter suites pin `with_native(false)` and oracle the
// P-format sink directly, so they run under yuv-semi-planar-solo.
#[cfg(feature = "yuv-semi-planar")]
mod resample_p0xx_high_bit;
// The native fast tier reuses the planar high-bit join, so the native
// suite needs BOTH families.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p0xx_high_bit_native;
#[cfg(feature = "yuv-semi-planar")]
mod resample_p2xx_high_bit;
// The native fast tier reuses the planar high-bit join, so the native
// suite needs BOTH families.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p2xx_high_bit_native;
#[cfg(feature = "yuv-semi-planar")]
mod resample_p4xx_high_bit;
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p4xx_high_bit_native;
#[cfg(feature = "rgb")]
mod resample_packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgb_32bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_16bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_8bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_u16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_packed_rgbf16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_packed_rgbf32;
#[cfg(feature = "yuv-packed")]
mod resample_packed_yuv_8bit;
#[cfg(feature = "yuv-packed")]
mod resample_packed_yuv_8bit_filter;
// The native fast tier reuses the planar non-4:2:0 join, so it (and these
// twin-parity / oracle suites) only exist when `yuv-planar` is also compiled.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
mod resample_packed_yuv_8bit_native;
// Packed 4:4:4 native fast-tier suites (Vuyx 8-bit; V410 / Xv36 high-bit). The
// native tier reuses the planar join, so it only exists under `yuv-planar` too.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
mod resample_packed_yuv444_8bit_native;
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
mod resample_packed_yuv444_hb_native;
#[cfg(feature = "rgb")]
mod resample_padding_byte;
#[cfg(feature = "mono")]
mod resample_pal8;
#[cfg(feature = "mono")]
mod resample_pal8_filter;
#[cfg(feature = "rgb")]
mod resample_rgb24;
#[cfg(feature = "rgb")]
mod resample_rgb48;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbaf16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbaf32;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbf16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbf32;
#[cfg(all(feature = "yuv-semi-planar", feature = "rgb"))]
mod resample_semi_planar;
#[cfg(feature = "yuv-semi-planar")]
mod resample_semi_planar_8bit_filter;
// Pure P-format filter coverage (`rgb` oracles gated inside); runs under
// yuv-semi-planar-solo.
#[cfg(feature = "yuv-semi-planar")]
mod resample_subsampled_high_bit_p_filter;
#[cfg(feature = "yuv-packed")]
mod resample_uyyvyy411;
// The native fast tier reuses the planar non-4:2:0 join, so the 4:1:1 native
// suite (its bin-then-convert oracle / route-freeze tests) only exists when
// `yuv-planar` is also compiled.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
mod resample_uyyvyy411_native;
#[cfg(feature = "v210")]
mod resample_v210;
// The native fast tier reuses the high-bit non-4:2:0 planar join, so the V210
// native suite (its de-pack twin-parity / bin-then-convert oracle suites) only
// exists when `yuv-planar` is also compiled.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
mod resample_v210_native;
#[cfg(feature = "yuv-444-packed")]
mod resample_v30x;
#[cfg(feature = "yuv-444-packed")]
mod resample_v410;
#[cfg(feature = "yuv-444-packed")]
mod resample_v410_v30x_filter;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuya;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuya_vuyx_filter;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuyx;
#[cfg(feature = "yuv-444-packed")]
mod resample_xv36;
#[cfg(feature = "yuv-444-packed")]
mod resample_xv48;
#[cfg(feature = "xyz")]
mod resample_xyz12;
#[cfg(feature = "xyz")]
mod resample_xyz12_filter;
#[cfg(feature = "y2xx")]
mod resample_y2xx;
#[cfg(feature = "y2xx")]
mod resample_y2xx_filter;
// The native fast tier reuses the high-bit non-4:2:0 planar join, so the native
// suite (and its twin-parity / oracle suites) only exists when `yuv-planar` is
// also compiled.
#[cfg(feature = "yuv-semi-planar")]
mod nv20;
// NV20's native fast tier reuses the high-bit non-4:2:0 planar join (like the
// P2xx native suite), and the cross-packing equivalence pins the native tier
// against P210, so the resample suite needs BOTH families.
#[cfg(feature = "yuv-planar")]
mod ictcp;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
mod resample_nv20;
#[cfg(all(feature = "y2xx", feature = "yuv-planar"))]
mod resample_y2xx_native;
#[cfg(feature = "gray")]
mod resample_ya16;
#[cfg(feature = "gray")]
mod resample_ya16_filter;
#[cfg(feature = "gray")]
mod resample_ya8;
#[cfg(feature = "gray")]
mod resample_ya8_filter;
#[cfg(feature = "gray")]
mod resample_yaf32;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv410_440p;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv411p;
#[cfg(feature = "yuv-planar")]
mod resample_yuv420p_bicublin;
#[cfg(feature = "yuv-planar")]
mod resample_yuv420p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv420p_high_bit_native;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv422_444p;
#[cfg(feature = "yuv-planar")]
mod resample_yuv422p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv440p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv444p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv_planar_8bit_filter;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv_planar_8bit_native;
#[cfg(feature = "yuv-planar")]
mod resample_yuv_planar_high_bit_filter;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv_planar_high_bit_native;
#[cfg(feature = "yuva")]
mod resample_yuva420p;
#[cfg(feature = "yuva")]
mod resample_yuva420p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva420p_native;
#[cfg(feature = "yuva")]
mod resample_yuva422p;
#[cfg(feature = "yuva")]
mod resample_yuva422p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva444p;
#[cfg(feature = "yuva")]
mod resample_yuva444p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva_planar_8bit_filter;
#[cfg(feature = "yuva")]
mod resample_yuva_planar_high_bit_filter;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_hsv_direct;
#[cfg(feature = "yuv-planar")]
mod subsampled_4_2_0_high_bit;
#[cfg(feature = "yuv-planar")]
mod yuv444p_msb;
// Exercises the high-bit semi-planar P-format `MixedSinker` impls (the
// P210/P410 sanity + walker-SIMD suites run under yuv-semi-planar-solo);
// the planar Yuv4*p cross-check tests + their frame helper are gated on
// `yuv-planar` inside.
#[cfg(feature = "yuv-444-packed")]
mod ayuv;
#[cfg(feature = "yuv-planar")]
mod resample_yuv_planar_hsv_direct;
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn;
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_4_2_2_hsv_direct;
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_4_4_4_hsv_direct;
#[cfg(feature = "yuv-semi-planar")]
mod subsampled_high_bit_pn_hsv_direct;
#[cfg(feature = "yuv-444-packed")]
mod uyva;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod vuyx;
#[cfg(feature = "yuv-444-packed")]
mod vyu444;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(feature = "yuv-444-packed")]
mod xv48;
#[cfg(feature = "xyz")]
mod xyz12;
#[cfg(feature = "y2xx")]
mod y210;
#[cfg(feature = "y2xx")]
mod y212;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "y2xx")]
mod y2xx_hsv_direct;
#[cfg(feature = "yuv-planar")]
mod yuv410p_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv411p_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv420p_8bit;
#[cfg(feature = "yuv-444-packed")]
mod yuv_444_packed_hsv_direct;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_hsv_direct;
#[cfg(feature = "yuv-planar")]
mod yuv_planar_hsv_direct_high_bit;
#[cfg(feature = "yuva")]
mod yuva;
#[cfg(feature = "yuva")]
mod yuva_hsv_direct;

// `v210` is intentionally absent: V210 only consumes this helper inside
// its `yuv-planar`-gated planar-parity oracles (in `v210.rs` /
// `resample_v210.rs`), so the `yuv-planar` arm already covers it and a
// `v210`-solo `--tests` build would otherwise see it as dead.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuva",
))]
pub(super) fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
  let mask = ((1u32 << bits) - 1) as u16;
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) as u16) & mask;
  }
}

#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "v210",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(super) fn pseudo_random_u8(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}

// Host-independent **wire-byte** encoders for `&[u16]` / `&[u32]` test
// fixtures. Frames carry bytes, not numbers — so a sinker test that
// wants to feed an LE-wire `0x1234` u16 needs the underlying byte view
// to be `[0x34, 0x12]` regardless of host endianness; ditto BE wants
// `[0x12, 0x34]`. The pattern `T::from_ne_bytes(v.to_{le,be}_bytes())`
// achieves exactly that: on LE hosts `as_le_*` is identity and `as_be_*`
// byte-swaps; on BE hosts (e.g. s390x) the polarity flips. Centralising
// these here matches the `le_encoded_u16_buf` convention from the
// `frame/tests/` fixture builders and keeps the call sites in
// xv36/v410/ayuv64 sinker tests self-documenting.

/// Encode a logical `u16` as host-independent **LE-wire** byte storage.
#[cfg(all(
  feature = "std",
  any(feature = "gbr", feature = "rgb", feature = "yuv-444-packed")
))]
#[inline]
pub(super) fn as_le_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u16` as host-independent **BE-wire** byte storage.
#[cfg(all(
  feature = "std",
  any(feature = "gbr", feature = "rgb", feature = "yuv-444-packed")
))]
#[inline]
pub(super) fn as_be_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_be_bytes())
}

/// Encode a logical `u32` as host-independent **LE-wire** byte storage.
#[cfg(all(feature = "std", feature = "yuv-444-packed"))]
#[inline]
pub(super) fn as_le_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u32` as host-independent **BE-wire** byte storage.
#[cfg(all(feature = "std", feature = "yuv-444-packed"))]
#[inline]
pub(super) fn as_be_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_be_bytes())
}

// Pins the **row-stage** tier where the native decimator exists, and is
// the identity otherwise. [`MixedSinker::with_native`] is gated on
// `any(yuv-planar, yuv-semi-planar)` (the native tier needs the planar
// join), so a packed-format sink built SOLO — its own feature without
// either planar feature — has no `with_native` method at all. The
// packed-format resample oracles assert the row-stage convert-then-bin
// SEMANTICS exactly, so they must pin the row-stage tier whenever it is
// togglable; under a packed-solo build there is no native tier, so the
// row-stage tail is already the only path and the identity is correct.
//
// The two arms mirror `with_native`'s gate EXACTLY: the `with_native`
// arm fires for every build where the method is compiled (so the pin is
// never silently skipped while a native tier is reachable), and the
// identity arm only for the genuinely native-free packed-solo builds.
// The outer gate is the union of the consuming packed families so the
// helper is not dead code in a planar-only build (where no consumer
// module compiles).
#[cfg(all(
  any(feature = "yuv-planar", feature = "yuv-semi-planar"),
  any(
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed"
  )
))]
pub(super) fn force_row_stage<F: mediaframe::SourceFormat, R>(
  sink: MixedSinker<'_, F, R>,
) -> MixedSinker<'_, F, R> {
  sink.with_native(false)
}
#[cfg(all(
  not(any(feature = "yuv-planar", feature = "yuv-semi-planar")),
  any(
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed"
  )
))]
pub(super) fn force_row_stage<F: mediaframe::SourceFormat, R>(
  sink: MixedSinker<'_, F, R>,
) -> MixedSinker<'_, F, R> {
  sink
}
