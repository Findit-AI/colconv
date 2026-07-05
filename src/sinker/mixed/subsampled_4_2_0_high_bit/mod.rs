//! High-bit-depth 4:2:0 `MixedSinker` impls, split per sub-format
//! so no single source exceeds ~1.5 KLoC: Yuv420p9/10/12/14/16
//! (`yuv-planar`) + P010/P012/P016 (`yuv-semi-planar`). The `native`
//! fast-tier join is `yuv-planar` (the P-format native tier reuses it,
//! so it is `all(yuv-semi-planar, yuv-planar)` inside `p0xx`); the
//! semi-planar P-format sinks fall back to the shared row-stage tail
//! when `yuv-planar` is absent.

#[cfg(feature = "yuv-planar")]
mod native;
#[cfg(feature = "yuv-semi-planar")]
mod p0xx;
#[cfg(feature = "yuv-planar")]
mod yuv420p;

#[cfg(feature = "yuv-planar")]
pub(crate) use native::{NativeYuv420U16, yuv420p16_process_native};
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) use native::{
  arm_native_u16_alloc_failure, arm_native_u16_chroma_failure, arm_native_u16_rgb_scratch_failure,
  arm_native_u16_rgb_u16_scratch_failure,
};
// Shared by the 4:2:2 high-bit centered-siting path (#302): the same u16
// half-width->full-width phase-0.5 INTERLEAVED chroma staging (4:2:0 and 4:2:2
// subsample chroma 2:1 horizontally identically), exported under format-neutral
// `_pn` names so `p2xx` reuses the `p0xx` wrappers verbatim rather than
// duplicating them. `reserve_pn_chroma_full_u16` is the semi-planar interleaved
// twin of the planar `yuv420p::reserve_420_chroma_full_u16` re-exported below;
// the alias also avoids a name collision between the two.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) use p0xx::arm_p0xx_alloc_failure;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
pub(crate) use p0xx::{
  reserve_420_chroma_full_u16 as reserve_pn_chroma_full_u16,
  upsample_420_chroma_center_h_p0xx as upsample_pn_chroma_center_h,
};
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv420p::reserve_420_chroma_full_u16;
// `upsample_420_chroma_center_h_u16`'s only remaining external consumer is the
// YUVA 4:2:2 centered path: the planar `Yuv422p` twin now routes through the
// shared L2 `reconstruct_chroma` stage (RFC #238 PR-F2), so gate the re-export
// to that sole consumer — a `yuv-planar`-without-`yuva` build would otherwise
// see it as unused. (PR-S7 repoints YUVA too and retires this re-export.)
#[cfg(feature = "yuva")]
pub(crate) use yuv420p::upsample_420_chroma_center_h_u16;
// RFC #238 S6f: the high-bit YUVA 4:2:0 `Bottom` (v = 1) vertical fold reuses the
// planar `Yuv420pN` u16 sited reconstruction ([`yuv420p::upsample_420_chroma_sited_u16`])
// and its one-row chroma lookback ([`yuv420p::reserve_420_chroma_prev_u16`] /
// [`yuv420p::stage_420_chroma_prev_u16`]) rather than duplicating the kernel —
// gated to the sole `yuva` consumer (`yuva` implies `yuv-planar`, so `yuv420p`
// exists). `Yuv420pN` itself uses them in-module, so they are live under
// `yuv-planar` without this re-export.
#[cfg(feature = "yuva")]
pub(crate) use yuv420p::{
  reserve_420_chroma_prev_u16, stage_420_chroma_prev_u16, upsample_420_chroma_sited_u16,
};
