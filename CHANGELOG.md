# UNRELEASED

## Ship 7 — u16 semi-planar 4:2:2 / 4:4:4 (P210 / P212 / P216 / P410 / P412 / P416)

Six new high-bit-packed semi-planar formats from the FFmpeg HW-decode
download space (CUDA / NVDEC / QSV emit these for HDR 4:2:2 and 4:4:4
content).

### New formats

- **`P210`** / **`P212`** / **`P216`** — 4:2:2 semi-planar at 10 / 12 /
  16 bits. Const-generic `PnFrame422<BITS>` with aliases. Per-row
  layout is identical to P010/P012/P016 (half-width interleaved UV =
  `width` u16 elements per row); only the walker reads chroma row
  `r` instead of `r / 2` (4:2:2 vs 4:2:0). MixedSinker impls reuse
  the existing `p010_to_rgb_*` / `p012_to_rgb_*` / `p016_to_rgb_*`
  row primitives — **zero new SIMD code** for 4:2:2.
- **`P410`** / **`P412`** / **`P416`** — 4:4:4 semi-planar at 10 / 12 /
  16 bits. Const-generic `PnFrame444<BITS>` with aliases. UV is
  full-width (`2 * width` u16 elements per row, one `U, V` pair per
  pixel — no horizontal chroma subsampling). New row-primitive
  family `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}, Q15 i32 pipeline)
  + dedicated `p_n_444_16_to_rgb_*` (16-bit, parallel i64-chroma path
  for u16 output).

Frame error type: `PnFrameError` extended with the same variants for
both new families. The `OddWidth` variant message kept the
4:2:0-specific wording (it applies to PnFrame422 too — 4:2:2 also
requires even width).

### SIMD coverage (4:4:4 family)

| Kernel                                  | NEON | SSE4.1 | AVX2     | AVX-512  | wasm simd128 |
| --------------------------------------- | :--: | :----: | :------: | :------: | :----------: |
| `p_n_444_to_rgb_row<BITS>`              |  ✅  |   ✅   | (sse4.1) | (sse4.1) |   (scalar)   |
| `p_n_444_to_rgb_u16_row<BITS>`          |  ✅  |   ✅   | (sse4.1) | (sse4.1) |   (scalar)   |
| `p_n_444_16_to_rgb_row`                 |  ✅  |   ✅   | (sse4.1) | (sse4.1) |   (scalar)   |
| `p_n_444_16_to_rgb_u16_row`             |  ✅  |   ✅   | (sse4.1) | (sse4.1) |   (scalar)   |

NEON and SSE4.1 ship native kernels validated against the scalar
reference. AVX2 / AVX-512 currently delegate to SSE4.1 (correct
semantics, half / quarter the throughput of a hypothetical native
kernel). wasm simd128 delegates to scalar pending a native simd128
implementation. Both are tracked as TODOs in the per-arch source —
follow-up commits in this branch will replace the delegations with
native wider-lane / native-instruction implementations:

- AVX2 native: 32 Y pixels per iter via 256-bit vectors, deinterleave
  via `_mm256_shuffle_epi8` + `_mm256_permute2x128_si256`.
- AVX-512 native: 64 Y pixels per iter; `_mm512_srai_epi64` enables a
  cleaner i64-chroma path than the SSE4.1 `srai64_15` bias trick.
- wasm simd128 native: 16 Y pixels per iter; `i64x2_shr_s` for the
  16-bit u16 path (no bias trick needed).

### MixedSinker integration

6 new `MixedSinker<F>` impls (P210 / P212 / P216 / P410 / P412 /
P416). New `RowSlice` variants for the 4:4:4 chroma rows:
`UvFull10`, `UvFull12`, `UvFull16`. The 4:2:2 impls reuse the
existing `UvHalf10/12/16` variants since the per-row layout is
identical to 4:2:0.

### Tests

- 6 new sanity gray-to-gray `MixedSinker` integration tests.
- 3 new walker-level SIMD-vs-scalar equivalence tests for P410 / P412
  / P416 at width 1922 (forces tail handling), pseudo-random chroma,
  full + limited range, all matrices.
- **Total suite: 313 passed on aarch64** (up from 304 at Ship 6b).

## Ship 6b — 9-bit family + 4:4:0 family (Tier 1 completion)

Closes the remaining FFmpeg `AVPixelFormat` Tier 1 gap. Six new
formats, all reusing existing kernel families:

### New formats

- **`Yuv420p9` / `Yuv422p9` / `Yuv444p9`** — 9-bit planar at 4:2:0 /
  4:2:2 / 4:4:4. Aliases over `Yuv420pFrame16<9>` /
  `Yuv422pFrame16<9>` / `Yuv444pFrame16<9>`. Reuses the const-generic
  `yuv_420p_n_to_rgb_*<BITS>` and `yuv_444p_n_to_rgb_*<BITS>` kernel
  families — only the AND mask (`0x1FF`) and the Q15 scale change at
  `BITS = 9`. Niche format (AVC High 9 profile only); no HEVC / VP9 /
  AV1 producers.
- **`Yuv440p`** — 4:4:0 planar at 8 bits (`AV_PIX_FMT_YUV440P` /
  `AV_PIX_FMT_YUVJ440P`). Full-width chroma, half-height — the
  axis-flipped twin of `Yuv422p`. Reuses `yuv_444_to_rgb_row`
  verbatim; only the walker reads chroma row `r / 2`. Mostly seen
  from JPEG decoders that subsample vertically only.
- **`Yuv440p10` / `Yuv440p12`** — 4:4:0 planar at 10 / 12 bits.
  `Yuv440pFrame16<BITS>` with aliases. Reuses the const-generic
  `yuv_444p_n_to_rgb_*<BITS>` family. No 9 / 14 / 16-bit variants
  exist in FFmpeg, so `try_new` rejects them.
- New `RowSlice` variants for the 9-bit shape rows: `Y9`, `UHalf9`,
  `VHalf9`, `UFull9`, `VFull9`.

### SIMD

All 6 new formats inherit native SIMD coverage from the underlying
const-generic kernel families. No new SIMD code paths — only the
compile-time `BITS` validators were widened from `{10, 12, 14}` to
`{9, 10, 12, 14}` across scalar + 5 backends.

| Kernel dispatch                                       | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| ----------------------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `yuv_420p_n_to_rgb_*<9>` (4:2:0 / 4:2:2)              |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_*<9>` (4:4:4)                      |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444_to_rgb_row` (via `Yuv440p`)                  |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_*<10/12>` (via `Yuv440p10/12`)     |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |

### Notes

- 4:4:0 is rare in modern codecs (mostly JPEG vertical-only
  subsampling) but ships as a first-class citizen for completeness.
- 9-bit is niche but trivially cheap to add (zero new kernels);
  shipping it closes the Tier 1 row in the format matrix.
- Skipped: `Yuv411p` / `Yuv410p` (legacy DV / Cinepak — uncommon
  enough that adding them now would be speculative work).

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


