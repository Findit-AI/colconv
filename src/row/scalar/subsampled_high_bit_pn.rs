use super::{load_u16, *};

/// Extracts the `BITS`-bit logical sample from one wire `u16` for the
/// semi-planar P-format / NV20 family.
///
/// `LOW_PACKED = false` (P010/P012/P210/P212 — high-bit-packed) shifts
/// the active high `BITS` down by `16 - BITS`; `LOW_PACKED = true` (NV20
/// — low-bit-packed) masks the active low `BITS` (`& ((1 << BITS) - 1)`).
/// `value` must already be host-native (apply [`load_u16`] first). Both
/// `BITS` and `LOW_PACKED` are const, so the branch folds and the unused
/// path is eliminated per monomorphization. Mispacked input has the
/// wrong bits discarded (the deliberate, SIMD-matching failure mode) —
/// e.g. a high-bit-packed buffer handed to the low-packed path keeps only
/// its zero low bits.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn depack_pn<const BITS: u32, const LOW_PACKED: bool>(value: u16) -> u16 {
  if LOW_PACKED {
    value & ((1u16 << BITS) - 1)
  } else {
    value >> (16 - BITS)
  }
}

/// Chroma-siting-aware horizontal 2:1 upsample for the **high-bit-packed
/// semi-planar** P-format family — the 4:2:0 `P010` / `P012` / `P016` AND the
/// 4:2:2 `P210` / `P212` / `P216` (#302). 4:2:0 and 4:2:2 subsample chroma 2:1
/// **horizontally** identically (4:2:2 only drops the vertical subsample), so
/// both reconstruct centered chroma with this one kernel — the MSB-aligned
/// `u16` twin of
/// [`chroma_upsample_2to1_center_h_u16`](super::yuv_planar_8bit::chroma_upsample_2to1_center_h_u16).
///
/// Reconstructs **center-sited** chroma (MPEG-1 / JPEG horizontal phase,
/// FFmpeg `AVCHROMA_LOC_CENTER` / `Top` / `Bottom`) with the standard
/// `1/4`–`3/4` phase-0.5 weights and edge clamp, then **re-interleaves**
/// the result so the output is a single full-width `U, V, U, V…` plane the
/// existing P-format 4:4:4 decode kernels
/// ([`p_n_444_to_rgb_row`] / [`p_n_444_16_to_rgb_row`] and siblings)
/// consume directly — so the centered path reuses their full SIMD backends
/// and stays bit-identical per tier.
///
/// **Packing seam (the difference from the planar `_u16` kernel).** P010 /
/// P012 / P016 are **MSB-aligned**: the active value sits in the **high**
/// `BITS` of each wire `u16`, the low `16 - BITS` bits ignored
/// (`depack_pn::<BITS, false>` = `>> (16 - BITS)`). The interleaved
/// half-row `uv_half` and the full-row output therefore both carry samples
/// in that MSB-aligned wire convention (and in the source `big_endian`
/// order). Each input is normalized wire → host-native then **de-packed**
/// by `>> (16 - BITS)` — discarding the ignored low bits, the SIMD-matching
/// sanitization the fused decode performs — BEFORE the blend; without it a
/// malformed sample's dirty low bits would leak into a neighbour through
/// the `1/4`–`3/4` blend, decoding the centered path differently from the
/// default. The blend runs in the logical domain (a `BITS`-bit value, so
/// `left + 3 * mid` ≤ `4 * 65535` never overflows the `u32` accumulator),
/// then each output is **re-packed** `<< (16 - BITS)` back to MSB-aligned
/// and re-encoded to the same wire order — so the 4:4:4 kernel's own
/// `>> (16 - BITS)` de-pack round-trips it exactly. At `BITS = 16` both
/// shifts are `>> 0` / `<< 0` (no-ops): the 16-bit sample already is its
/// logical value.
///
/// `uv_half` is the half-width interleaved chroma (`U` at even element);
/// `uv_full` receives `2 * width` `u16` (`U, V` per pixel). The
/// de-interleave, the phase-0.5 upsample, and the re-interleave all fuse
/// into this single pass — there is no separate planar U / V scratch
/// (unlike the NV 8-bit path's de-interleave-then-`chroma_upsample`
/// staging), because the P-format 4:4:4 kernels want the interleaved form.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `uv_half.len() >= width` (a half-row is `width` `u16`: `width / 2`
///   `U, V` pairs), `uv_full.len() >= 2 * width`.
// Reachable only through the high-bit P-format sinks' centered paths (4:2:0
// `p0xx` and 4:2:2 `p2xx`), which stage the full-width chroma in a `Vec`
// scratch — so the kernel is live exactly where the sink (hence heap
// allocation) is. Those paths are `all(yuv-semi-planar, yuv-planar)` (their
// `chroma_420_center_sited_h` / `chroma_422_center_sited_h` predicates + the
// 4:4:4 P-format decode kernels need `yuv-planar`); this module already
// requires `yuv-semi-planar`, so gating the kernel on `yuv-planar` matches its
// callers and keeps it from being dead code in a semi-planar-only build.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_center_h_p0xx<const BITS: u32, const LOW_PACKED: bool>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(uv_half.len() >= width, "uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let shift = 16 - BITS;
  // De-pack wire → logical via [`depack_pn`] (`>> (16 - BITS)` for the
  // high-bit-packed P-formats, `& ((1 << BITS) - 1)` for low-bit-packed
  // NV20) — discarding the ignored bits, the SIMD-matching sanitization
  // the fused decode performs, so a mispacked neighbour's dirty bits never
  // leak into the `1/4`–`3/4` blend. `c` indexes the chroma sample,
  // `comp ∈ {0, 1}` selects U (even) / V (odd) within the interleaved pair.
  let load = |c: usize, comp: usize| -> u32 {
    let raw = uv_half[2 * c + comp];
    let host = if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    depack_pn::<BITS, LOW_PACKED>(host) as u32
  };
  // Re-pack logical → wire + re-encode, symmetric with `load`: the
  // high-bit-packed store re-aligns to MSB (`<< (16 - BITS)`, `<< 0` at
  // BITS = 16); the low-bit-packed store keeps the value in the low `BITS`
  // (no shift) so the 4:4:4 kernel's own `& ((1 << BITS) - 1)` de-pack
  // round-trips it exactly.
  let store = |logical: u32| -> u16 {
    let v = if LOW_PACKED {
      logical as u16
    } else {
      (logical as u16) << shift
    };
    if big_endian { v.to_be() } else { v.to_le() }
  };

  let half = width / 2;
  for j in 0..half {
    // `c[j-1]` clamps to `c[0]` at the left edge; `c[j+1]` clamps to the
    // last sample at the right edge — boundary replication, so the edge
    // columns collapse to `c[0]` / `c[half-1]` exactly. U and V each get
    // the independent phase-0.5 blend, then interleave back into uv_full.
    let lj = j.saturating_sub(1);
    let rj = if j + 1 < half { j + 1 } else { j };
    let (ul, um, ur) = (load(lj, 0), load(j, 0), load(rj, 0));
    let (vl, vm, vr) = (load(lj, 1), load(j, 1), load(rj, 1));
    // even column 2j → 1/4 left + 3/4 mid; odd column 2j+1 → 3/4 mid + 1/4 right.
    uv_full[2 * (2 * j)] = store((ul + 3 * um + 2) >> 2);
    uv_full[2 * (2 * j) + 1] = store((vl + 3 * vm + 2) >> 2);
    uv_full[2 * (2 * j + 1)] = store((3 * um + ur + 2) >> 2);
    uv_full[2 * (2 * j + 1) + 1] = store((3 * vm + vr + 2) >> 2);
  }
}

/// Reconstructs the four output elements (`U`/`V` × even/odd column) for a
/// single interleaved chroma sample `j` of the centered (#302 phase-0.5) 2:1
/// horizontal upsample — the per-sample body of
/// [`chroma_upsample_2to1_center_h_p0xx`], factored out so the SIMD backends
/// share this exact de-pack / blend / re-pack / re-interleave math for their
/// edge / tail columns and stay byte-identical to this reference. `c[j-1]`
/// clamps to `c[0]` and `c[j+1]` to `c[half-1]`, applied independently to `U`
/// (even element) and `V` (odd element).
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
// Consumed only by the semi-planar SIMD arch kernels (`arch::{neon, x86_sse41,
// wasm_simd128}`), which compile solely on aarch64 / x86_64 / wasm32 under
// `yuv-semi-planar`. Without that feature, or on any other target (e.g. the
// s390x / i686 miri jobs), there is no caller and the helper is genuinely dead;
// allow it there while the SIMD builds keep it live and `-D warnings`-checked.
#[cfg_attr(
  not(all(
    feature = "yuv-semi-planar",
    any(
      target_arch = "aarch64",
      target_arch = "x86_64",
      target_arch = "wasm32"
    )
  )),
  allow(dead_code)
)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_center_h_p0xx_pair<const BITS: u32, const LOW_PACKED: bool>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  j: usize,
  half: usize,
  big_endian: bool,
) {
  let shift = 16 - BITS;
  let load = |c: usize, comp: usize| -> u32 {
    let raw = uv_half[2 * c + comp];
    let host = if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    depack_pn::<BITS, LOW_PACKED>(host) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = if LOW_PACKED {
      logical as u16
    } else {
      (logical as u16) << shift
    };
    if big_endian { v.to_be() } else { v.to_le() }
  };
  let lj = j.saturating_sub(1);
  let rj = if j + 1 < half { j + 1 } else { j };
  let (ul, um, ur) = (load(lj, 0), load(j, 0), load(rj, 0));
  let (vl, vm, vr) = (load(lj, 1), load(j, 1), load(rj, 1));
  uv_full[2 * (2 * j)] = store((ul + 3 * um + 2) >> 2);
  uv_full[2 * (2 * j) + 1] = store((vl + 3 * vm + 2) >> 2);
  uv_full[2 * (2 * j + 1)] = store((3 * um + ur + 2) >> 2);
  uv_full[2 * (2 * j + 1) + 1] = store((3 * vm + vr + 2) >> 2);
}

/// Bottom-sited (`AVCHROMA_LOC_BOTTOM`, `v = 1`) vertical + horizontal 2:1
/// chroma reconstruction for the **high-bit-packed semi-planar** 4:2:0 P-format
/// family — `P010` / `P012` / `P016` (RFC #238 S6e), the interleaved MSB-aligned
/// twin of the planar
/// [`chroma_upsample_420_bottom_even_h_u16`](super::yuv_planar_8bit::chroma_upsample_420_bottom_even_h_u16).
/// It is the `Bottom` sibling of [`chroma_upsample_2to1_center_h_p0xx`]: same
/// de-pack / re-pack / re-interleave seam, but the EVEN output luma row `2i`
/// first vertically box-blends the previous (`prev_uv_half`) and current
/// (`cur_uv_half`) interleaved chroma rows before the horizontal centered fold.
///
/// ```text
///   e[c] = (prev[c] + cur[c] + 1) >> 1                 (vertical box blend)
///   even col 2j → (e[j-1] + 3·e[j] + 2) >> 2   (e[-1]    clamped to e[0])
///   odd  col 2j+1 → (3·e[j] + e[j+1] + 2) >> 2  (e[half] clamped to e[half-1])
/// ```
///
/// applied INDEPENDENTLY to `U` (even element) and `V` (odd element) of each
/// interleaved pair. Both rows carry samples in the source's MSB-aligned wire
/// convention (active value in the high `BITS`) + `big_endian` order; each input
/// is normalized wire → host-native then de-packed `>> (16 - BITS)` (the
/// SIMD-matching sanitization, so a mispacked neighbour's dirty low bits never
/// leak through the blend) BEFORE the arithmetic, and each output is re-packed
/// `<< (16 - BITS)` + re-encoded to the same wire order — so the result feeds
/// straight to the existing `p_n_444_*` decode kernels, bit-identical per tier
/// (the exact logical chroma the planar `_u16` twin produces, so a `P0xx`
/// `Bottom` decode stays byte-identical to the `Yuv420pN` one). At `BITS = 16`
/// both shifts are no-ops. The blend runs in the logical domain (`left + 3·mid`
/// ≤ `4 · 65535` never overflows the `u32` accumulator).
///
/// The **odd** luma row `2i+1` is co-sited with chroma row `i` (`v = 1`), so it
/// needs no vertical blend and reuses [`chroma_upsample_2to1_center_h_p0xx`] on
/// `cur_uv_half` directly — only the even row routes here.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `prev_uv_half.len() >= width`, `cur_uv_half.len() >= width`,
///   `uv_full.len() >= 2 * width`.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottom_even_h_p0xx<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let shift = 16 - BITS;
  let load = |row: &[u16], c: usize, comp: usize| -> u32 {
    let raw = row[2 * c + comp];
    (if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    } >> shift) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = (logical as u16) << shift;
    if big_endian { v.to_be() } else { v.to_le() }
  };
  // Vertical box blend of the previous and current chroma rows, per chroma
  // column `c` and component `comp ∈ {0, 1}` (U / V), then the horizontal center
  // phase across it — the fused `v = 1` reconstruction.
  let vblend = |c: usize, comp: usize| -> u32 {
    (load(prev_uv_half, c, comp) + load(cur_uv_half, c, comp) + 1) >> 1
  };

  let half = width / 2;
  for j in 0..half {
    // `e[j-1]` clamps to `e[0]` at the left edge; `e[j+1]` clamps to the last
    // sample at the right edge — boundary replication, matching the horizontal
    // sibling. U and V each get the independent phase, then interleave back.
    let lj = j.saturating_sub(1);
    let rj = if j + 1 < half { j + 1 } else { j };
    let (ul, um, ur) = (vblend(lj, 0), vblend(j, 0), vblend(rj, 0));
    let (vl, vm, vr) = (vblend(lj, 1), vblend(j, 1), vblend(rj, 1));
    uv_full[2 * (2 * j)] = store((ul + 3 * um + 2) >> 2);
    uv_full[2 * (2 * j) + 1] = store((vl + 3 * vm + 2) >> 2);
    uv_full[2 * (2 * j + 1)] = store((3 * um + ur + 2) >> 2);
    uv_full[2 * (2 * j + 1) + 1] = store((3 * vm + vr + 2) >> 2);
  }
}

/// Reconstructs the four output elements (`U`/`V` × even/odd column) for a
/// single interleaved chroma sample `j` of the bottom-sited even-row upsample —
/// the per-sample body of [`chroma_upsample_420_bottom_even_h_p0xx`], factored
/// out so the SIMD backends share this exact de-pack / vertical-blend / centered
/// fold / re-pack / re-interleave math for their edge / tail columns and stay
/// byte-identical to this reference. `e[j-1]` clamps to `e[0]` and `e[j+1]` to
/// `e[half-1]`, applied independently to `U` (even element) and `V` (odd).
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
// Consumed only by the semi-planar SIMD arch kernels (`arch::{neon, x86_sse41,
// wasm_simd128}`), which compile solely on aarch64 / x86_64 / wasm32 under
// `yuv-semi-planar`. Without that feature, or on any other target (e.g. the
// s390x / i686 miri jobs), there is no caller and the helper is genuinely dead;
// allow it there while the SIMD builds keep it live and `-D warnings`-checked.
#[cfg_attr(
  not(all(
    feature = "yuv-semi-planar",
    any(
      target_arch = "aarch64",
      target_arch = "x86_64",
      target_arch = "wasm32"
    )
  )),
  allow(dead_code)
)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottom_even_h_p0xx_pair<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  j: usize,
  half: usize,
  big_endian: bool,
) {
  let shift = 16 - BITS;
  let load = |row: &[u16], c: usize, comp: usize| -> u32 {
    let raw = row[2 * c + comp];
    (if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    } >> shift) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = (logical as u16) << shift;
    if big_endian { v.to_be() } else { v.to_le() }
  };
  let vblend = |c: usize, comp: usize| -> u32 {
    (load(prev_uv_half, c, comp) + load(cur_uv_half, c, comp) + 1) >> 1
  };
  let lj = j.saturating_sub(1);
  let rj = if j + 1 < half { j + 1 } else { j };
  let (ul, um, ur) = (vblend(lj, 0), vblend(j, 0), vblend(rj, 0));
  let (vl, vm, vr) = (vblend(lj, 1), vblend(j, 1), vblend(rj, 1));
  uv_full[2 * (2 * j)] = store((ul + 3 * um + 2) >> 2);
  uv_full[2 * (2 * j) + 1] = store((vl + 3 * vm + 2) >> 2);
  uv_full[2 * (2 * j + 1)] = store((3 * um + ur + 2) >> 2);
  uv_full[2 * (2 * j + 1) + 1] = store((3 * vm + vr + 2) >> 2);
}

/// Co-sited (`h = 0`) horizontal 2:1 chroma reconstruction for the interleaved
/// high-bit semi-planar 4:2:0 P-format family — the co-sited (nearest-neighbor)
/// twin of [`chroma_upsample_2to1_center_h_p0xx`] used by `BottomLeft`. Each
/// interleaved `(U, V)` chroma sample covers its two luma columns unchanged, so
/// reconstruction is a plain 2× replicate; the de-pack / re-pack / re-interleave
/// seam is identical to the centered sibling (so a mispacked neighbour's dirty
/// bits are discarded), just with no horizontal interpolation:
///
/// ```text
///   uv_full[2·(2j)..] = uv_full[2·(2j+1)..] = (U[j], V[j])
/// ```
///
/// The P-format sibling of the planar
/// [`chroma_upsample_2to1_cosited_h_u16`](super::yuv_planar_8bit::chroma_upsample_2to1_cosited_h_u16).
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `uv_half.len() >= width`, `uv_full.len() >= 2 * width`.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_2to1_cosited_h_p0xx<const BITS: u32, const LOW_PACKED: bool>(
  uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(uv_half.len() >= width, "uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let shift = 16 - BITS;
  let load = |c: usize, comp: usize| -> u32 {
    let raw = uv_half[2 * c + comp];
    let host = if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    depack_pn::<BITS, LOW_PACKED>(host) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = if LOW_PACKED {
      logical as u16
    } else {
      (logical as u16) << shift
    };
    if big_endian { v.to_be() } else { v.to_le() }
  };

  let half = width / 2;
  for j in 0..half {
    let u = store(load(j, 0));
    let v = store(load(j, 1));
    uv_full[2 * (2 * j)] = u;
    uv_full[2 * (2 * j) + 1] = v;
    uv_full[2 * (2 * j + 1)] = u;
    uv_full[2 * (2 * j + 1) + 1] = v;
  }
}

/// Bottom-LEFT-sited (`AVCHROMA_LOC_BOTTOMLEFT`, `h = 0`, `v = 1`) vertical + 2:1
/// chroma reconstruction for the high-bit-packed semi-planar 4:2:0 P-format
/// family — the interleaved MSB-aligned twin of the planar
/// [`chroma_upsample_420_bottomleft_even_h_u16`](super::yuv_planar_8bit::chroma_upsample_420_bottomleft_even_h_u16),
/// and the co-sited-`h` sibling of [`chroma_upsample_420_bottom_even_h_p0xx`]:
/// same de-pack / re-pack / re-interleave seam and vertical box blend of the EVEN
/// output luma row, but fed to the co-sited (`h = 0`) horizontal 2× replicate
/// rather than the centered `1/4`–`3/4` weights.
///
/// ```text
///   e[c]       = (prev[c] + cur[c] + 1) >> 1        (vertical box blend)
///   uv_full[2·(2j)..] = uv_full[2·(2j+1)..] = (e_U[j], e_V[j])
/// ```
///
/// applied INDEPENDENTLY to `U` / `V`, in the logical (de-packed) domain, then
/// re-packed `<< (16 - BITS)` (a no-op at `BITS = 16`). The **odd** luma row
/// `2i+1` is co-sited with chroma row `i` (`v = 1`), so it needs no vertical blend
/// and reuses [`chroma_upsample_2to1_cosited_h_p0xx`] on `cur_uv_half` directly.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `prev_uv_half.len() >= width`, `cur_uv_half.len() >= width`,
///   `uv_full.len() >= 2 * width`.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottomleft_even_h_p0xx<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  width: usize,
  big_endian: bool,
) {
  debug_assert_eq!(width & 1, 0, "P-format 4:2:0 requires even width");
  debug_assert!(prev_uv_half.len() >= width, "prev_uv_half row too short");
  debug_assert!(cur_uv_half.len() >= width, "cur_uv_half row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");

  let shift = 16 - BITS;
  let load = |row: &[u16], c: usize, comp: usize| -> u32 {
    let raw = row[2 * c + comp];
    (if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    } >> shift) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = (logical as u16) << shift;
    if big_endian { v.to_be() } else { v.to_le() }
  };
  let vblend = |c: usize, comp: usize| -> u32 {
    (load(prev_uv_half, c, comp) + load(cur_uv_half, c, comp) + 1) >> 1
  };

  let half = width / 2;
  for j in 0..half {
    let u = store(vblend(j, 0));
    let v = store(vblend(j, 1));
    uv_full[2 * (2 * j)] = u;
    uv_full[2 * (2 * j) + 1] = v;
    uv_full[2 * (2 * j + 1)] = u;
    uv_full[2 * (2 * j + 1) + 1] = v;
  }
}

/// Reconstructs the four output elements for a single interleaved chroma sample
/// `j` of the bottom-left-sited even-row upsample — the per-sample body of
/// [`chroma_upsample_420_bottomleft_even_h_p0xx`], shared with the SIMD backends
/// for their tail columns so those stay byte-identical to this reference. The
/// co-sited (`h = 0`) horizontal phase is a plain replicate with no neighbour, so
/// no edge clamp is needed; `U` (even element) and `V` (odd) each get the
/// independent vertical box blend in the de-packed domain, then re-packed.
#[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
// Consumed only by the semi-planar SIMD arch kernels (`arch::{neon, x86_sse41,
// wasm_simd128}`), which compile solely on aarch64 / x86_64 / wasm32 under
// `yuv-semi-planar`. Without that feature, or on any other target (e.g. the
// s390x / i686 miri jobs), there is no caller and the helper is genuinely dead;
// allow it there while the SIMD builds keep it live and `-D warnings`-checked.
#[cfg_attr(
  not(all(
    feature = "yuv-semi-planar",
    any(
      target_arch = "aarch64",
      target_arch = "x86_64",
      target_arch = "wasm32"
    )
  )),
  allow(dead_code)
)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn chroma_upsample_420_bottomleft_even_h_p0xx_pair<const BITS: u32>(
  prev_uv_half: &[u16],
  cur_uv_half: &[u16],
  uv_full: &mut [u16],
  j: usize,
  big_endian: bool,
) {
  let shift = 16 - BITS;
  let load = |row: &[u16], c: usize, comp: usize| -> u32 {
    let raw = row[2 * c + comp];
    (if big_endian {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    } >> shift) as u32
  };
  let store = |logical: u32| -> u16 {
    let v = (logical as u16) << shift;
    if big_endian { v.to_be() } else { v.to_le() }
  };
  let vblend =
    |comp: usize| -> u32 { (load(prev_uv_half, j, comp) + load(cur_uv_half, j, comp) + 1) >> 1 };
  let u = store(vblend(0));
  let v = store(vblend(1));
  uv_full[2 * (2 * j)] = u;
  uv_full[2 * (2 * j) + 1] = v;
  uv_full[2 * (2 * j + 1)] = u;
  uv_full[2 * (2 * j + 1) + 1] = v;
}

// ---- P010 (semi-planar 10-bit, high-bit-packed) → RGB ------------------

/// Converts one row of P010 (semi‑planar 4:2:0 with UV interleaved,
/// `BITS` active bits in the **high** `BITS` of each `u16`) to
/// **8‑bit** packed RGB.
///
/// Structurally identical to [`nv12_to_rgb_row`] plus the per‑sample
/// shift: each `u16` load is extracted to its `BITS`‑bit value via
/// `sample >> (16 - BITS)`, then the same Q15 pipeline as
/// [`yuv_420p_n_to_rgb_row`] runs with the same `BITS`. For `BITS ==
/// 10` this is P010 (`>> 6`); for `BITS == 12` it's P012 (`>> 4`).
/// Mispacked input — e.g. a low‑bit‑packed buffer handed to this
/// kernel — has its active low bits discarded (producing near‑black
/// output), matching every SIMD backend.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<BITS, false, BE, false>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0 (P010/P012)
/// to **8‑bit** packed **RGBA**. Same numerical contract as
/// [`p_n_to_rgb_row`]; the only differences are the per-pixel stride
/// (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5a: the public dispatchers
// `row::p010_to_rgba_row` and `row::p012_to_rgba_row` land in the
// follow-up SIMD/dispatcher PR. Until then this thin wrapper has no
// caller. P016 has its own kernel family
// ([`p16_to_rgb_or_rgba_row`]) — never routed here.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<BITS, true, BE, false>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) → packed
/// **8-bit** RGB. The low-bit twin of [`p_n_to_rgb_row`]: identical Q15
/// pipeline and interleaved-UV shape, but each `u16` is de-packed via
/// `& 0x03FF` (active bits in the **low** 10) instead of `>> 6`. Pinned
/// to `BITS = 10` (the only NV20 depth). Thin wrapper over
/// [`p_n_to_rgb_or_rgba_row`] with `LOW_PACKED = true`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv20_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// NV20 → packed **8-bit** **RGBA** (`R, G, B, 0xFF`). The low-bit twin
/// of [`p_n_to_rgba_row`]. Thin wrapper over [`p_n_to_rgb_or_rgba_row`]
/// with `LOW_PACKED = true`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv20_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_to_rgb_row`] (`ALPHA = false`, 3 bpp store)
/// and [`p_n_to_rgba_row`] (`ALPHA = true`, 4 bpp store with constant
/// `0xFF` alpha).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // High-bit-packed Pn kernels are only defined for BITS in {10, 12}.
  // Outside that set, `16 - BITS` could under/overflow and the Q15
  // coefficient table has no corresponding entry. P016 (BITS=16) has
  // its own dedicated kernel family with i64 chroma multiply — using
  // this i32 path at BITS=16 would silently overflow on high chroma
  // values. The compile-time assertion fails monomorphization for any
  // BITS outside {10, 12}, eliminating that release-build corruption
  // trap.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  // Each `u16` load is converted to its `BITS`-bit sample via
  // [`depack_pn`]: `>> (16 - BITS)` for the high-bit-packed P-formats
  // (6 for P010/P210, 4 for P012/P212), or `& ((1 << BITS) - 1)` for the
  // low-bit-packed NV20 (`LOW_PACKED = true`). The BE byte-swap is applied
  // first (on the raw wire format), then the de-pack extracts the active
  // bits. If mispacked input is handed to the kernel, the wrong bits are
  // discarded rather than recovering the intended value (matching every
  // SIMD backend). No hot-path cost: one swap + one shift-or-mask per load.
  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    let y1 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x + 1])) as i32 - y_off,
      y_scale,
    );
    out[(x + 1) * bpp] = clamp_u8(y1 + r_chroma);
    out[(x + 1) * bpp + 1] = clamp_u8(y1 + g_chroma);
    out[(x + 1) * bpp + 2] = clamp_u8(y1 + b_chroma);
    if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0
/// (`BITS` ∈ {10, 12}: P010, P012) to **native‑depth `u16`**
/// packed RGB — samples are **low‑bit‑packed** on output
/// (`[0, (1 << BITS) - 1]` in the low bits of each `u16`, upper bits
/// zero), matching the `yuv420p10le` / `yuv420p12le` convention —
/// **not** the P010/P012 high‑bit packing. Callers feeding a P010/
/// P012 consumer must shift the output left by `16 - BITS`.
///
/// Mirrors [`yuv_420p_n_to_rgb_u16_row`] on the math side; the only
/// differences are the input shift (`sample >> (16 - BITS)` to
/// extract the `BITS`-bit value from the high-bit packing) and the
/// interleaved UV layout.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
    y, uv_half, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of high‑bit‑packed semi‑planar 4:2:0 (P010/P012)
/// to **native‑depth `u16`** packed **RGBA** — output is low‑bit‑packed
/// to match [`p_n_to_rgb_u16_row`]. Alpha is `(1 << BITS) - 1` (opaque
/// maximum at the input bit depth).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
//
// Scalar prep for Ship 8 Tranche 5b: the public dispatchers
// `row::p010_to_rgba_u16_row` and `row::p012_to_rgba_u16_row` land in
// the follow-up SIMD/dispatcher PR. Until then this thin wrapper has
// no caller. P016 has its own u16 kernel family
// ([`p16_to_rgb_or_rgba_u16_row`]) — never routed here.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
    y, uv_half, rgba_out, width, matrix, full_range,
  );
}

/// NV20 → **native-depth `u16`** packed RGB — low-bit-packed output
/// (`[0, 1023]`). The low-bit twin of [`p_n_to_rgb_u16_row`]. Thin
/// wrapper over [`p_n_to_rgb_or_rgba_u16_row`] with `LOW_PACKED = true`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv20_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
}

/// NV20 → **native-depth `u16`** packed **RGBA** — low-bit-packed
/// output; alpha element is `1023`. The low-bit twin of
/// [`p_n_to_rgba_u16_row`]. Thin wrapper over
/// [`p_n_to_rgb_or_rgba_u16_row`] with `LOW_PACKED = true`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv20_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_rgb_or_rgba_u16_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
}

/// Shared kernel for [`p_n_to_rgb_u16_row`] (`ALPHA = false`, 3 bpp
/// store) and [`p_n_to_rgba_u16_row`] (`ALPHA = true`, 4 bpp store
/// with opaque alpha = `(1 << BITS) - 1`).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // See `p_n_to_rgb_or_rgba_row` for the BITS range rationale. The
  // P016 u16 path lives in [`p16_to_rgb_or_rgba_u16_row`] (i64 chroma
  // multiply); this i32 path would overflow before clamp at 16-bit
  // chroma. Compile-time assertion eliminates the release-build
  // corruption trap.
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let alpha_max: u16 = out_max as u16;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }

    let y1 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x + 1])) as i32 - y_off,
      y_scale,
    );
    out[(x + 1) * bpp] = (y1 + r_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 1] = (y1 + g_chroma).clamp(0, out_max) as u16;
    out[(x + 1) * bpp + 2] = (y1 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[(x + 1) * bpp + 3] = alpha_max;
    }

    x += 2;
  }
}

// ---- Pn 4:4:4 (semi-planar high-bit-packed) → RGB ----------------------
//
// Mirrors `p_n_to_rgb_*<BITS>` but with full-width interleaved UV: one
// `U, V` pair per pixel (= `2 * width` u16 elements per row), no
// horizontal duplication. Same `>> (16 - BITS)` extraction at load
// time. BITS ∈ {10, 12} on the i32 Q15 pipeline; BITS = 16 lives in
// `p_n_444_16_to_rgb_*` because the chroma multiply-add overflows
// i32 at u16 output (same rationale as p16 / yuv_444p16).

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **8-bit** packed RGB. `BITS ∈ {10, 12}`. Each `u16` load is
/// shifted right by `16 - BITS` to extract the active value before
/// running the standard Q15 i32 pipeline.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_row::<BITS, false, BE, false>(
    y, uv_full, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **8-bit** packed **RGBA**. Same numerical contract as
/// [`p_n_444_to_rgb_row`]; the only differences are the per-pixel
/// stride (4 vs 3) and the alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_row::<BITS, true, BE, false>(
    y, uv_full, rgba_out, width, matrix, full_range,
  );
}

/// Shared kernel for [`p_n_444_to_rgb_row`] (`ALPHA = false`, 3 bpp
/// store) and [`p_n_444_to_rgba_row`] (`ALPHA = true`, 4 bpp store
/// with constant `0xFF` alpha).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  for x in 0..width {
    // 4:4:4: one UV pair per pixel — uv_full[x*2] = U, uv_full[x*2+1] = V.
    // `depack_pn` extracts the active bits (`>> (16 - BITS)` high-packed,
    // `& ((1 << BITS) - 1)` low-packed NV20).
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **native-depth `u16`** packed RGB — low-bit-packed output (the
/// `BITS` active bits in the **low** bits of each `u16`, upper bits
/// zero), matching the [`yuv_444p_n_to_rgb_u16_row`] convention.
/// `BITS ∈ {10, 12}`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
    y, uv_full, rgb_out, width, matrix, full_range,
  );
}

/// Converts one row of high-bit-packed semi-planar 4:4:4 (P410, P412)
/// to **native-depth `u16`** packed **RGBA** — low-bit-packed output;
/// alpha element is `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
    y, uv_full, rgba_out, width, matrix, full_range,
  );
}

/// Shared kernel for [`p_n_444_to_rgb_u16_row`] (`ALPHA = false`,
/// 3 bpp store) and [`p_n_444_to_rgba_u16_row`] (`ALPHA = true`,
/// 4 bpp store with opaque alpha = `(1 << BITS) - 1`).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `out.len() >= width * if ALPHA { 4 } else { 3 }` (`u16` elements).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, BITS>(full_range);
  let bias = chroma_bias::<BITS>();
  let out_max: i32 = (1i32 << BITS) - 1;
  let alpha_max: u16 = out_max as u16;

  for x in 0..width {
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = alpha_max;
    }
  }
}

/// Converts one row of P416 (semi-planar 4:4:4, 16-bit, full UV) to
/// **8-bit** packed RGB. Y and chroma both stay on i32 — same logic
/// as `p16_to_rgb_row` plus the full-width UV layout.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_row::<false, BE>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of P416 to **8-bit** packed **RGBA**. Same
/// numerical contract as [`p_n_444_16_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_row::<true, BE>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared P416 → 8-bit RGB / RGBA kernel. `ALPHA = false` emits 3 bpp;
/// `ALPHA = true` emits 4 bpp with constant `0xFF` alpha.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_sample = load_u16::<BE>(uv_full[x * 2]);
    let v_sample = load_u16::<BE>(uv_full[x * 2 + 1]);
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(load_u16::<BE>(y[x]) as i32 - y_off, y_scale);
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

/// Converts one row of P416 to **native-depth `u16`** packed RGB —
/// full-range output in `[0, 65535]`. Chroma multiply-add runs in i64
/// (same rationale as `p16_to_rgb_u16_row` and
/// `yuv_444p16_to_rgb_u16_row`: `coeff x u_d` overflows i32 at 16
/// bits for the BT.2020 blue coefficient).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_full, rgb_out, width, matrix, full_range);
}

/// Converts one row of P416 to **native-depth `u16`** packed
/// **RGBA** — full-range output `[0, 65535]`; alpha element is
/// `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_444_16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_full, rgba_out, width, matrix, full_range);
}

/// Shared P416 → native-depth `u16` RGB / RGBA kernel. `ALPHA = false`
/// emits 3 bpp; `ALPHA = true` emits 4 bpp with constant `0xFFFF`
/// alpha. Uses i64 chroma multiply (same rationale as
/// [`p_n_444_16_to_rgb_u16_row`]).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  for x in 0..width {
    let u_sample = load_u16::<BE>(uv_full[x * 2]);
    let v_sample = load_u16::<BE>(uv_full[x * 2 + 1]);
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale64(load_u16::<BE>(y[x]) as i32 - y_off, y_scale);
    out[x * bpp] = (y0 + r_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 1] = (y0 + g_chroma).clamp(0, out_max) as u16;
    out[x * bpp + 2] = (y0 + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[x * bpp + 3] = 0xFFFF;
    }
  }
}

// ---- Pn 4:4:4 (semi-planar high-bit-packed) → HSV (direct) ------------
//
// The display-referred twins of [`p_n_444_to_rgb_row`] /
// [`p_n_444_16_to_rgb_row`], fused with the OpenCV HSV quantizer. Each
// shares the EXACT per-pixel **8-bit-output** Q15 decode (the
// `range_params_n::<BITS, 8>` scaling, the `>> (16 - BITS)` high-bit
// de-pack — none for the already-16-bit P416 — and the full-width 1:1 UV
// layout) as its `_to_rgb` sibling, then feeds the decoded `(r, g, b)`
// straight into [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes —
// never materializing a packed-RGB row. The HSV output is 8-bit
// (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`) regardless of source depth, exactly
// as the existing high-bit→RGB→HSV path is `rgb_to_hsv_row` over the
// **8-bit** `p_n_444*_to_rgb_row` output — so these are byte-identical to
// `rgb_to_hsv_row(p_n_444*_to_rgb_row(...))` with no RGB intermediate.
// The 16-bit member splits to `p_n_444_16_to_hsv_row` (the BITS-generic
// i32 path is pinned to {10, 12}, mirroring the 4:2:0 `p_n_to_hsv_row` /
// `p16_to_hsv_row` split). The SIMD backends mirror this via a small
// reused 8-bit-RGB chunk filled by the existing SIMD `p_n_444*_to_rgb_row`
// plus the SIMD `rgb_to_hsv_row`.

/// High-bit-packed semi-planar 4:4:4 (P410/P412) → planar HSV bytes
/// (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`). Const-generic over `BITS ∈ {10, 12}` and `BE`
/// (source byte order), exactly like [`p_n_444_to_rgb_row`]. Chroma is
/// full-width interleaved `U, V` — one pair per pixel, no upsampling.
///
/// Byte-identical to `rgb_to_hsv_row(p_n_444_to_rgb_row::<BITS, BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`, and each of
///   `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_444_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  for x in 0..width {
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_full[x * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    let (h, s, v) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = v;
  }
}

/// P416 (semi-planar 4:4:4, 16-bit, full UV) → planar HSV bytes (OpenCV
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). `BE` selects the source
/// byte order, exactly like [`p_n_444_16_to_rgb_row`]. Chroma is
/// full-width interleaved `U, V` — one pair per pixel.
///
/// Byte-identical to `rgb_to_hsv_row(p_n_444_16_to_rgb_row::<BE>(...))` —
/// the 8-bit RGB intermediate the existing P416 HSV path uses (i32 Q15;
/// only the u16-output P416 RGB needs the i64 chroma multiply).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `uv_full.len() >= 2 * width`, and each of
///   `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_444_16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  for x in 0..width {
    let u_sample = load_u16::<BE>(uv_full[x * 2]);
    let v_sample = load_u16::<BE>(uv_full[x * 2 + 1]);
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(load_u16::<BE>(y[x]) as i32 - y_off, y_scale);
    let (h, s, v) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = v;
  }
}

// ---- High-bit-packed semi-planar 4:2:0 (P010/P012) → HSV (direct) ------
//
// The display-referred twin of [`p_n_to_rgb_row`], fused with the OpenCV
// HSV quantizer. It shares the EXACT per-pixel **8-bit-output** Q15
// decode (`Coefficients::for_matrix` + `range_params_n::<BITS, 8>` + the
// `>> (16 - BITS)` high-bit de-pack + the interleaved-UV 1→2 upsampling
// shape) as its `_to_rgb` sibling, then feeds the decoded `(r, g, b)`
// straight into [`rgb_to_hsv_pixel`] and scatters to the H/S/V planes —
// never materializing a packed-RGB row. The HSV output is 8-bit
// (`H ∈ [0, 179]`, `S, V ∈ [0, 255]`) regardless of source depth,
// because the existing high-bit HSV path is `rgb_to_hsv_row` applied to
// the **8-bit** `p_n_to_rgb_row` output. This kernel is therefore
// byte-identical to `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))`
// but allocates no RGB intermediate. P016 (BITS = 16) has its own
// [`p16_to_hsv_row`] (the i32 path would overflow before clamp at 16
// bits). The SIMD backends mirror this via a small reused 8-bit-RGB
// chunk filled by the existing SIMD `p_n_to_rgb_row` plus the SIMD
// `rgb_to_hsv_row`.

/// High-bit-packed semi-planar 4:2:0 (P010/P012) → planar HSV bytes
/// (OpenCV `cv2.COLOR_RGB2HSV` encoding: `H ∈ [0, 179]`,
/// `S, V ∈ [0, 255]`). Const-generic over `BITS ∈ {10, 12}` and `BE`
/// (source byte order), exactly like [`p_n_to_rgb_row`]. Chroma is
/// half-width interleaved `U, V`, nearest-neighbor 1→2 upsampled per
/// pixel pair.
///
/// Byte-identical to `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `uv_half.len() >= width`, and each of
///   `h_out` / `s_out` / `v_out` `>= width`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_n_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Same compile-time guard as `p_n_to_rgb_or_rgba_row`: the i32 8-bit
  // Q15 pipeline is valid for {10, 12}; 16 lives in `p16_to_hsv_row`.
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<BITS, 8>(full_range);
  let bias = chroma_bias::<BITS>();

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2]));
    let v_sample = depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(uv_half[c_idx * 2 + 1]));
    let u_d = q15_scale(u_sample as i32 - bias, c_scale);
    let v_d = q15_scale(v_sample as i32 - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y0 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x])) as i32 - y_off,
      y_scale,
    );
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h0;
    s_out[x] = s0;
    v_out[x] = v0;

    let y1 = q15_scale(
      depack_pn::<BITS, LOW_PACKED>(load_u16::<BE>(y[x + 1])) as i32 - y_off,
      y_scale,
    );
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1 + r_chroma) as i32,
      clamp_u8(y1 + g_chroma) as i32,
      clamp_u8(y1 + b_chroma) as i32,
    );
    h_out[x + 1] = h1;
    s_out[x + 1] = s1;
    v_out[x + 1] = v1;

    x += 2;
  }
}

/// NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) → planar HSV
/// bytes (OpenCV encoding). The low-bit twin of [`p_n_to_hsv_row`]
/// (`LOW_PACKED = true`, `BITS = 10`). Byte-identical to
/// `rgb_to_hsv_row(nv20_to_rgb_row::<BE>(...))`. Thin wrapper over
/// [`p_n_to_hsv_row`] with `LOW_PACKED = true`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv20_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  p_n_to_hsv_row::<10, BE, true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
}

// ---- High-bit-packed semi-planar P-format → native luma ---------------
//
// For the high-bit semi-planar P-format family the luma plane is the Y
// plane **range-reduced to 8 bits** — NOT a verbatim copy (unlike 8-bit
// NV). P010/P012/P016 all store their active value in the **high**
// `BITS` of each wire `u16`, so the top byte (`>> 8`) is the
// range-expanded 8-bit luma for every member: for P010 (`value << 6`)
// the de-pack-then-downshift `(value >> (16 - 10)) >> (10 - 8)`
// collapses to `>> 8`; for P012 (`value << 4`) `(>> 4) >> 4` collapses
// to `>> 8`; for P016 (`>> 0`) `>> 8` is the top byte directly. So a
// single `(logical >> 8) as u8` serves the whole family, matching the
// `y2xx_n_to_luma_row` (`extract BITS-bit value, then >> (BITS - 8)`)
// contract and bit-for-bit reproducing the sink's former inline native-Y
// luma loop. The P-format sink exposes no `luma_u16` output (it is
// always `&mut None`), so there is no u16 luma variant. Y is a
// contiguous plane here — like
// [`y_plane_to_luma_u16_row`](super::y_plane_to_luma_u16::y_plane_to_luma_u16_row),
// a trivial per-element shift with no SIMD (the auto-vectorizer handles
// it); only the packed-Y families (Y2xx, V210, …) need a SIMD luma
// deinterleave.

/// High-bit-packed semi-planar P-format (P010/P012/P016) → 8-bit native
/// luma: the Y plane's high byte. Const-generic over `BE` (source byte
/// order). Each wire `u16` is normalized to host-native, then
/// `>> 8` extracts the range-reduced 8-bit luma. Bit-identical to the
/// P-format sink's former inline native-Y loop and to
/// `y2xx_n_to_luma_row::<BITS, BE>` (the `>> (16 - BITS)` de-pack and
/// the `>> (BITS - 8)` downshift compose to `>> 8` for any
/// `BITS ∈ {10, 12, 16}`). `BITS` is accepted for API symmetry with the
/// RGB / HSV kernels; the result does not depend on it.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width` and `luma_out.len() >= width`.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn p_n_to_luma_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  const { assert!(BITS == 10 || BITS == 12 || BITS == 16) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  for (d, &s) in luma_out[..width].iter_mut().zip(y[..width].iter()) {
    *d = (load_u16::<BE>(s) >> 8) as u8;
  }
}

/// NV20 (semi-planar 4:2:2, 10-bit, **low-bit-packed**) → 8-bit native
/// luma: the de-packed Y range-reduced to 8 bits. The low-bit twin of
/// [`p_n_to_luma_row`] — because NV20's active bits live in the **low**
/// 10, the high-byte (`>> 8`) shortcut does NOT apply; the de-pack
/// (`& 0x03FF`) followed by the `(BITS - 8)`-bit downshift is
/// `(logical & 0x03FF) >> 2`. Bit-identical to
/// `y2xx_n_to_luma_row::<10, BE>` and to the `binned_Y >> (BITS - 8)`
/// luma contract the NV20 sink's resample tiers use. Const-generic over
/// `BE` (source byte order). Y is a contiguous plane, so no SIMD variant
/// (the auto-vectorizer handles the per-element mask + shift).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width` and `luma_out.len() >= width`.
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn nv20_to_luma_row<const BE: bool>(y: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  for (d, &s) in luma_out[..width].iter_mut().zip(y[..width].iter()) {
    *d = (depack_pn::<10, true>(load_u16::<BE>(s)) >> 2) as u8;
  }
}
