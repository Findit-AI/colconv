# UNRELEASED

## Ship 6 — Yuv422p / Yuv444p at 8/10/12/14/16 bit

All three priorities landed in a single PR:
- **A (HW→SW gap)** — `Yuv444p16` (NVDEC / CUDA 4:4:4 HDR download target)
- **B (Pro video)** — `Yuv422p10/12/14`, `Yuv444p10/12/14` (ProRes, DNxHD)
- **C (Common SW)** — `Yuv422p`, `Yuv444p` 8-bit (libx264 defaults)

### New formats

- **`Yuv422p`** — 4:2:2 planar, 8-bit. New `Yuv422pFrame` + marker +
  walker + `MixedSinker<Yuv422p>` impl. Per-row kernel reused from
  `Yuv420p` verbatim (4:2:0 vs 4:2:2 differs only in the vertical
  walker). No new SIMD kernels.
- **`Yuv422p10` / `Yuv422p12` / `Yuv422p14`** — 4:2:2 planar at 10 /
  12 / 14 bit. Const-generic `Yuv422pFrame16<BITS>` with aliases.
  Per-row kernels reused from the `Yuv420p_n<BITS>` family.
- **`Yuv422p16`** — 4:2:2 planar at 16 bit. Alias over
  `Yuv422pFrame16<'_, 16>`. Per-row kernels reused from the parallel
  i64-chroma `yuv_420p16_to_rgb_*` family.
- **`Yuv444p`** — 4:4:4 planar, 8-bit. New `Yuv444pFrame` + marker +
  walker + `MixedSinker<Yuv444p>` + dedicated `yuv_444_to_rgb_row`
  kernel family. No width parity constraint (4:4:4 chroma is 1:1
  with Y, not paired).
- **`Yuv444p10` / `Yuv444p12` / `Yuv444p14`** — 4:4:4 planar at 10 /
  12 / 14 bit. Const-generic `Yuv444pFrame16<BITS>` with aliases.
  New const-generic `yuv_444p_n_to_rgb_row<BITS>` +
  `yuv_444p_n_to_rgb_u16_row<BITS>` kernel family.
- **`Yuv444p16`** — 4:4:4 planar at 16 bit. Alias over
  `Yuv444pFrame16<'_, 16>`. Dedicated parallel i64-chroma kernel
  family `yuv444p16_to_rgb_*` (same rationale as `Yuv420p16` — the
  blue coefficient overflows i32 at 16 bits).
- New `RowSlice` variants for the full-width 4:4:4 chroma rows:
  `UFull`, `VFull`, `UFull10/12/14`, `VFull10/12/14`.

### SIMD

Every new 4:4:4 kernel ships native SIMD on every backend — no
scalar fallbacks or cross-tier delegations:

| Kernel family                     | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `yuv_444_to_rgb_row` (8-bit)      |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_row<BITS>`     |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_u16_row<BITS>` |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p16_to_rgb_row`           |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p16_to_rgb_u16_row`       |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |

Yuv422p family reuses Yuv420p kernels (4:2:2 differs only in the
vertical walker):

| Yuv422p kernel dispatch                                      | NEON | SSE4.1 | AVX2 | AVX-512 | wasm |
| ------------------------------------------------------------ | :--: | :----: | :--: | :-----: | :--: |
| `yuv_420_to_rgb_row` (via `Yuv422p`)                         |  ✅  |   ✅   |  ✅  |   ✅    |  ✅  |
| `yuv420p{10,12,14,16}_to_rgb_*` (via `Yuv422p{10,12,14,16}`) |  ✅  |   ✅   |  ✅  |   ✅    |  ✅  |

Block sizes (u8 output): 16 pixels (NEON / SSE4.1 / wasm), 32
pixels (AVX2), 64 pixels (AVX-512). The 16-bit u16-output variants
run at 8 pixels per iter on SSE4.1 and wasm (i64-lane width), 16 on
AVX2, 32 on AVX-512.

### Bonus: native 16-bit u16 kernels on AVX2 + wasm (resolves Ship 4c leftover)

This PR also replaces the **three residual u16-output delegations**
from Ship 4b/4c — `yuv_420p16_to_rgb_u16_row`, `p16_to_rgb_u16_row`,
and the newly added `yuv_444p16_to_rgb_u16_row` — with native
implementations on AVX2 and wasm simd128:

- **AVX2**: all three previously delegated to SSE4.1. The delegation
  was rational when `_mm256_srai_epi64` was unavailable, but the
  `srai64_15` bias trick scales cleanly to 256 bits via
  `_mm256_srli_epi64` + offset. New AVX2 kernels process 16 pixels
  per iter — 2× the SSE4.1 rate.
- **wasm simd128**: all three previously fell through to scalar. The
  "no native i64 arithmetic shift" rationale became stale once
  `i64x2_shr_s` stabilized. New wasm kernels use `i64x2_mul` +
  `i64x2_shr` at 8 pixels per iter.

Every 16-bit u16-output path is now native on every backend.

### Tests

37 new tests total:
- 11 `MixedSinker` integration tests (10 `gray → gray` sanity checks
  covering every new format × u8/u16 output, plus a `yuv422p ↔
  yuv420p` equivalence check that pins the shared-row-kernel
  contract).
- 6 NEON arch equivalence tests for `yuv_444p_n` and `yuv_444p16`
  across all six matrices, full/limited range, and odd-width tails
  (1, 3, 15, 17, 32, 33, 1920, 1921).
- 10 per-arch `yuv_444_to_rgb_row` scalar-equivalence tests (2 per
  backend × 5 backends).
- 10 per-arch `yuv_444p_n<BITS>` scalar-equivalence tests on x86 +
  wasm (4 kernels × SSE4.1 / AVX2 / AVX-512 / wasm, covering 10/12/14
  and widths straddling each backend's block size).

Total suite: **273 passed on aarch64** (up from 254 at v0.5). x86
and wasm tests run in CI on their respective targets.

### Benches

10 new benches:
- `yuv_422p_to_rgb`, `yuv_422p10_to_rgb`, `yuv_422p12_to_rgb`,
  `yuv_422p14_to_rgb`, `yuv_422p16_to_rgb` — reuse the 4:2:0 row
  primitives; output numerically identical to the 4:2:0 benches at
  the same width.
- `yuv_444p_to_rgb`, `yuv_444p10_to_rgb`, `yuv_444p12_to_rgb`,
  `yuv_444p14_to_rgb`, `yuv_444p16_to_rgb` — dedicated 4:4:4
  kernels. NEON 4× over scalar on the 8-bit kernel (~1.6 GiB/s
  scalar → ~6.4 GiB/s NEON at 1080p).

## Ship 5 — NV16 / NV24 / NV42

### New formats

- **`Nv16`** — 4:2:2 semi-planar, UV-ordered. New `Nv16Frame` type
  plus `MixedSinker<Nv16>` impl. Per-row kernel is shared with
  `Nv12` (the 4:2:0 → 4:2:2 difference is purely in the vertical
  walker — one UV row per Y row instead of one per two).
- **`Nv24`** — 4:4:4 semi-planar, UV-ordered. New `Nv24Frame` type,
  `MixedSinker<Nv24>` impl, and a dedicated kernel family
  (`nv24_to_rgb_row`). No width parity constraint (4:4:4 chroma is
  1:1 with Y).
- **`Nv42`** — 4:4:4 semi-planar, VU-ordered. Shares kernels with
  `Nv24` via a `SWAP_UV` const generic (mirrors the `Nv21` / `Nv12`
  pairing).
- New `RowSlice::UvFull` / `RowSlice::VuFull` variants for the
  full-width chroma rows.

### SIMD

Native NV24 / NV42 kernels across all five arches:

| Backend   | Block (Y × UV bytes) | Relative to SSE4.1 |
| --------- | -------------------- | ------------------ |
| NEON      | 16 × 32              | 1×                 |
| SSE4.1    | 16 × 32              | 1× (baseline)      |
| AVX2      | 32 × 64              | ~2×                |
| AVX-512   | 64 × 128             | ~4×                |
| wasm      | 16 × 32              | 1×                 |

The 4:4:4 layout simplifies the main loop vs NV12/NV21 — no
horizontal chroma duplication since UV is 1:1 with Y.

### AVX-512 16-bit u16 native kernels (follow-up to Ship 4b)

- `yuv_420p16_to_rgb_u16_row` and `p16_to_rgb_u16_row` on AVX-512
  now run a native 32-pixel-per-iter kernel using
  `_mm512_srai_epi64` + `_mm512_mul_epi32` +
  `_mm512_permutex2var_epi32` reassembly, replacing the 8-pixel
  SSE4.1 delegation that Ship 4b shipped. ~4× throughput
  improvement on AVX-512 CPUs.
- AVX2 u16 paths still delegate to SSE4.1 (AVX2 lacks
  `_mm256_srai_epi64`; reimplementing the `srai64_15` bias trick at
  256 bits would have marginal gain).

### Tests

34 new tests total:
- 15 frame validation tests for `Nv16Frame` / `Nv24Frame` /
  `Nv42Frame` including odd-width + odd-height acceptance for 4:4:4,
  `u32` overflow on the `2 × width` chroma stride.
- 13 MixedSinker integration tests including the cross-format
  parity checks `nv16_matches_nv12_mixed_sinker_with_duplicated_chroma`
  and `nv42_matches_nv24_mixed_sinker_with_swapped_chroma` (the
  latter uses width 33 to exercise the no-parity contract), plus
  error-path tests mirroring the NV12 suite.
- 6 NEON arch equivalence tests across 6 matrices × full/limited
  range × odd-width tails (1, 3, 15, 17, 33).

Total suite: 254 passed on aarch64 (up from 204 at v0.4b).

### Benches

`nv16_to_rgb`, `nv24_to_rgb`, `nv42_to_rgb` — same `simd` /
`scalar` split and 720p / 1080p / 4K widths as the rest of the
family.

# 0.1.2 (January 6th, 2022)

FEATURES


