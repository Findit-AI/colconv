# UNRELEASED

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


