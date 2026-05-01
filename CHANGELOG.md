# CHANGELOG

## 0.13.0 — Ship 12a (Tier 5 V410, first tranche)

- Add `V410Frame` (10-bit packed YUV 4:4:4 in 32-bit words; FFmpeg
  `AV_PIX_FMT_V410` = XV30 alias).
- 5-backend SIMD: NEON (4 px/iter), SSE4.1 (8 px/iter), AVX2
  (8 px/iter), AVX-512 (16 px/iter), wasm-simd128 (4 px/iter).
- `MixedSinker<V410>` with `with_rgb` / `with_rgba` / `with_rgb_u16` /
  `with_rgba_u16` / `with_luma` / `with_hsv`. (`with_luma_u16`
  deferred — no library consumer ask yet.)
- Cross-tranche infrastructure: 4 new `RowSlice` variants (`V410Packed`,
  `Xv36Packed`, `VuyaPacked`, `Ayuv64Packed`) added eagerly so Ship
  12b/c/d don't touch `MixedSinker` mod.rs.
- V410 ↔ Yuv444p10 planar parity oracle validates the SIMD path
  byte-for-byte against the established planar 4:4:4 reference.
- Opens Tier 5 (packed YUV 4:4:4); remaining tranches are Ship 12b
  (XV36, 12-bit), 12c (VUYA, 8-bit + α), 12d (AYUV64, 16-bit + α).

---

## 0.12.0 — Ship 11d (Tier 4 Y216, closes Tier 4)

- Add `Y216Frame` (16-bit packed YUV 4:2:2, full-range u16 samples).
- Parallel `y216_*` kernel family separate from `y2xx_n_to_*<BITS>` for the
  i64 chroma u16 path (BITS=16 overflows i32 on Q15 chroma sums).
- 5-backend SIMD: NEON, SSE4.1, AVX2, AVX-512, wasm-simd128.
- `MixedSinker<Y216>` with `with_rgb` / `with_rgba` / `with_rgb_u16` /
  `with_rgba_u16` / `with_luma` (generic) / `with_luma_u16` / `with_hsv`.
- Closes Tier 4 (packed YUV 4:2:2 high-bit-depth: v210, Y210, Y212, Y216).

---

# UNRELEASED

## Tier 14 (in progress) — Bayer demosaic + WB + CCM

New RAW source family for camera-RAW pipelines (RED R3D, Blackmagic
BRAW, Nikon NRAW, FFmpeg `bayer_*`). `colconv` covers demosaic
onwards: vendor SDKs decode the camera bitstream into a Bayer plane,
`colconv` runs bilinear demosaic + per-channel white balance + 3×3
color-correction in a single per-row kernel.

### New types (all in `colconv::raw`)

- `BayerPattern` — `enum { Bggr, Rggb, Grbg, Gbrg }`,
  `#[non_exhaustive]`, `IsVariant`-derived.
- `BayerDemosaic` — `enum { Bilinear }`, `#[non_exhaustive]`,
  `Default = Bilinear`. Future variants (Malvar-He-Cutler, etc.)
  will land without a breaking change.
- `WhiteBalance { r, g, b: f32 }` — per-channel gain newtype with
  `::try_new` (validating: rejects NaN / ±∞ / negative via
  [`WhiteBalanceError`]), panicking `::new`, `::neutral`,
  accessors, `Default = neutral()`. `WbChannel` enum names which
  channel failed validation.
- `ColorCorrectionMatrix` — 3×3 newtype with `::try_new`
  (validating: rejects any non-finite element via
  [`ColorCorrectionMatrixError`]; negative entries are allowed
  because real CCMs subtract crosstalk), panicking `::new`,
  `::identity`, `as_array`, `Default = identity()`.

### New frame types (in `colconv::frame`)

- `BayerFrame<'a>` — single `&[u8]` plane. Odd widths and heights
  are accepted (cropped Bayer planes are real workflow output; the
  walker / kernel handle partial 2×2 tiles via edge clamping).
- `BayerFrame16<'a, const BITS: u32>` — `&[u16]` **low-packed** at
  `BITS` ∈ {10, 12, 14, 16} (active samples in the low `BITS` bits,
  valid range `[0, (1 << BITS) - 1]`). Matches the planar
  `Yuv420p10/12/14/16` convention in packing; diverges in
  validation: `BayerFrame16::try_new` validates **every active
  sample's range** as part of construction (returning
  `BayerFrame16Error::SampleOutOfRange` for out-of-range data),
  not just geometry. RAW pipelines often surface trusted-but-
  mispacked input from sensor SDKs, and the demosaic kernel has no
  well-defined behavior on out-of-range samples; mandatory
  validation makes the `bayer16_to` walker fully fallible — no
  data-dependent panic surface. Aliases: `Bayer10Frame` /
  `Bayer12Frame` / `Bayer14Frame` / `Bayer16Frame`. Odd dimensions
  accepted.
- `BayerFrameError` / `BayerFrame16Error` — structured error enums,
  `#[non_exhaustive]`, `IsVariant`-derived.

### New walkers / kernels

- `raw::bayer_to(src, pattern, demosaic, wb, ccm, sink)` and
  `raw::bayer16_to::<BITS, _>(...)` walkers — zero per-row and
  per-frame allocation. Walker fuses `M = CCM · diag(wb)` once at
  entry; row scratch is the source plane itself (`above` / `mid` /
  `below` row borrows with **mirror-by-2** boundary handling at
  top / bottom edges — `row 0 → above = row 1`, `row h-1 → below =
  row h-2`; replicate fallback only when `height < 2`). Same
  contract surfaces through `BayerRow::above()` / `below()` so
  custom sinks see the mirror borrows directly.
- Public dispatchers: `row::bayer_to_rgb_row`,
  `row::bayer16_to_rgb_row<BITS>`,
  `row::bayer16_to_rgb_u16_row<BITS>`. Each runs release-mode
  preflight (`above` / `below` length match `mid`, `rgb_out >=
  3 * width` via `checked_mul`, `BITS` const-asserted), matching
  the `yuv_*_to_rgb_row` boundary contract so future unsafe SIMD
  kernels inherit a hardened entry point. `use_simd` is currently
  a no-op — per-arch SIMD ships in a follow-up PR.
- Sink subtraits: `BayerSink`, `BayerSink16<BITS>`. Source markers:
  `Bayer`, `Bayer16<BITS>` plus `Bayer10` / `Bayer12` / `Bayer14` /
  `Bayer16Bit` aliases.

### SIMD coverage

| Kernel                                  | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `bayer_to_rgb_row` (8-bit)              |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |
| `bayer16_to_rgb_row<BITS>` (→u8)        |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |
| `bayer16_to_rgb_u16_row<BITS>` (→u16)   |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |

Scalar reference path lands first; per-arch SIMD backends are
scheduled as a dedicated follow-up PR (`feat/bayer-simd`).

### MixedSinker integration

- `MixedSinker<Bayer>` and `MixedSinker<Bayer16<BITS>>` impls — both
  expose `with_rgb` / `with_luma` / `with_hsv`; the `Bayer16<BITS>`
  variant additionally exposes `with_rgb_u16` for native-depth RGB
  output. RGB scratch buffer is grown lazily for the HSV-without-RGB
  path (mirrors the YUV impls).
- **Luma colorimetry is caller-configurable** via the new
  `LumaCoefficients` enum (`Bt709` / `Bt2020` / `Bt601` / `DciP3` /
  `AcesAp1` / `Custom(CustomLumaCoefficients)`, `#[non_exhaustive]`).
  YUV source impls memcpy luma directly off the Y plane and are
  unaffected; Bayer impls *derive* luma from the demosaiced RGB and
  therefore need to know which gamut weights to apply. Choose the
  preset matching the gamut your `ColorCorrectionMatrix` targets:
  passing a Rec.2020 CCM and using BT.709 luma weights produces a
  valid-shaped but numerically wrong luma plane for non-grayscale
  content (downstream scene-cut detectors, brightness thresholders
  and perceptual-diff tools see the wrong values; uniform gray is
  invariant — every preset agrees on gray, which is what made the
  hard-coded BT.709 path go undetected by uniform-gray tests).
  Default is `Bt709` to match the implicit weights every YUV → RGB →
  luma pipeline uses. API:
  `MixedSinker::<Bayer>::new(w, h).with_luma_coefficients(LumaCoefficients::Bt2020)`.
  `Custom` wraps the validated `CustomLumaCoefficients` newtype
  (private fields, mirrors the `WhiteBalance` / `ColorCorrectionMatrix`
  pattern): construct via `LumaCoefficients::try_custom(r, g, b)` /
  `CustomLumaCoefficients::try_new(r, g, b)` which return
  `Result<_, LumaCoefficientsError>` after rejecting NaN / ±∞ /
  negative / `> MAX_COEFFICIENT (10.0)` inputs. The bound is much
  tighter than `WhiteBalance::MAX_GAIN (1e6)` because the luma
  kernel multiplies into a `u32` accumulator (not `f32` as in
  WB/CCM) — `1e6` would overflow the per-row sum, `10.0` keeps it
  six orders of magnitude clear of `u32::MAX`. Custom weights are
  not normalized to sum to 1.0 — caller is responsible (otherwise
  the luma plane is brightness-scaled). All five published presets
  resolve to Q8 triples summing to exactly 256 so the kernel's
  `>> 8` divisor is exact (the published ACES AP1 weights round
  naïvely to `(70, 173, 14) = 257`; `cg` is shaved by 1 LSB to
  make the triple sum to 256 with the smallest perceptual error).

### Tests

- Frame-validation tests (8-bit + high-bit-depth, including
  `BayerFrame16::try_new` rejecting samples whose value exceeds
  `(1 << BITS) - 1` under the low-packed convention; both above-
  max and the common MSB-aligned packing-mismatch case).
- 5 type-helper tests (WB / CCM defaults, fuse arithmetic).
- 11 end-to-end walker + kernel tests (8-bit + 12-bit, solid R / G
  / B channels, uniform-byte invariant, pattern swap RGGB↔BGGR,
  walker row-count). Solid-channel assertions cover the **full
  frame** including borders — boundary handling uses mirror-by-2
  (`row -1 → row 1`, `row h → row h-2`, same on columns) which
  preserves CFA parity, so a constant-channel Bayer mosaic stays
  constant everywhere instead of bleeding wrong-color samples into
  the missing-channel averages at edges.
- 6 luma-coefficient tests covering both Bayer and Bayer16 paths:
  solid-red rows produce distinct luma values for each preset (54
  / 67 / 77 / 59 / 70 for BT.709 / BT.2020 / BT.601 / DCI-P3 /
  ACES AP1 — guards against silent collapse to one preset);
  `try_custom(1.0, 0.0, 0.0)` round-trips the red channel back to
  255; default is `Bt709`; uniform gray is invariant across all
  presets (regression-pin for the original
  `*_with_luma_uniform_byte` semantics); preset Q8 triples each
  sum to exactly 256.
- 8 `CustomLumaCoefficients` validation tests: accepts standard
  weights / zeroes / `MAX_COEFFICIENT` boundary; rejects NaN /
  ±∞ / negative / `MAX_COEFFICIENT + 1.0` / `1e9` per channel
  with the matching `LumaCoefficientsError` variant; `try_custom`
  routes errors through; `::new` panics loudly on hostile input;
  end-to-end "all three channels at MAX_COEFFICIENT, all pixels
  255" stays inside the `u32` accumulator and clamps to 255.

## Ship 8b — source-side YUVA (alpha-preserving RGBA output)

The follow-up to Ship 8: source-side alpha. Where Ship 8 padded the
output alpha lane to `0xFF` / `(1 << BITS) - 1` regardless of source,
Ship 8b adds **YUVA source types** that carry an alpha plane through
to the RGBA output. The first vertical slice ships `Yuva444p10`
(ProRes 4444 + α territory — the highest-value VFX format from the
Format Share table § 2a-1 row 10).

### Strategy B (forked kernels) over Strategy A (separate splice)

Two implementation strategies were considered:

- **Strategy A** (deferred) — run the existing RGBA kernel (alpha =
  opaque), then a second-pass helper reads source alpha + overwrites
  the alpha byte. Memory traffic 6W per pixel; ~50 LOC + 1 helper.
- **Strategy B** (adopted) — extend each kernel's const-`ALPHA`
  template with a third `ALPHA_SRC: bool` generic. Source-alpha is
  loaded inside the kernel, masked, and stored straight into the
  alpha lane in the same pass. Memory traffic 5W per pixel (single
  pass); ~3,000 LOC across 30+ kernels for an L1-noise ~10% perf
  win in the alpha-present case.

Strategy B was picked for best alpha-present throughput on the
high-bandwidth 4:4:4 + α format that motivated the work. Existing
`*_to_rgb_*` and `*_to_rgba_*` public wrappers are backward-compat
shims passing `ALPHA_SRC = false` and `None` to the templates — zero
overhead when alpha-source is off; existing call sites compile
unchanged.

### Vertical slice 1: `Yuva444p10` (3 PRs)

The first format follows the same staging pattern as Ship 8 high-bit
tranches (5/6/7): scalar prep first (call-site stable), then u8 SIMD,
then u16 SIMD.

| # | Tranche | Status |
|---|---|---|
| 1 | scalar prep + Frame + walker + dispatchers + sinker integration | ✅ shipped (PR #32) — `Yuva444pFrame16<BITS=10>`, `Yuva444p10Frame` alias, `yuva444p10_to` walker, `MixedSinker<Yuva444p10>`, scalar tests |
| 1b | u8 RGBA SIMD across all 5 backends | ✅ shipped (PR #33) |
| 1c | u16 RGBA SIMD across all 5 backends | ✅ shipped (PR #34) |

### Surface added

- **`Yuva444pFrame16<'a, const BITS: u32>`** — mirrors `Yuv444pFrame16`
  with an extra `a` slice + `a_stride`. Const-asserted `BITS == 10`
  in this slice; other bit depths land in subsequent vertical slices.
  `try_new` validates dimensions + plane lengths; `try_new_checked`
  additionally validates every active sample range.
- **`Yuva444p10Frame<'a>`** type alias.
- **`Yuva444p10`** marker + `Yuva444p10Row<'a>` (carries `a` slice)
  + `Yuva444p10Sink` trait + `yuva444p10_to` walker.
- **`MixedSinker<Yuva444p10>`** with `with_rgba` / `set_rgba` (u8) +
  `with_rgba_u16` / `set_rgba_u16` (u16) per-format builders, plus
  `with_rgb` / `with_rgb_u16` / `with_luma` / `with_hsv` alpha-drop
  paths that reuse the `Yuv444p10` row dispatchers verbatim.
- **Public dispatchers** in `colconv::row`: `yuva444p10_to_rgba_row`
  and `yuva444p10_to_rgba_u16_row` — same SIMD-via-`use_simd` shape
  as `yuv444p10_to_rgba_*`.

### Strategy B template extension

The four 4:4:4 const-`ALPHA` templates gained the `ALPHA_SRC` third
generic in this slice (only the BITS-generic planar variant is in
scope for this vertical slice; other 4:4:4 variants land later):

- `scalar::yuv_444p_n_to_rgb_or_rgba_row<BITS, ALPHA, ALPHA_SRC>` (u8)
- `scalar::yuv_444p_n_to_rgb_or_rgba_u16_row<BITS, ALPHA, ALPHA_SRC>` (u16)
- Same SIMD templates × 5 backends (NEON / SSE4.1 / AVX2 / AVX-512 /
  wasm simd128) — refactor in PRs #33 (u8) and #34 (u16).

Per-pixel store branched on three combinations:

| `ALPHA` | `ALPHA_SRC` | Per-pixel alpha |
|---|---|---|
| false | false | RGB-only (no alpha lane) |
| true | false | RGBA, alpha = `0xFF` u8 / `(1 << BITS) - 1` u16 (existing path) |
| true | true | RGBA, alpha = `(a_src[x] & bits_mask::<BITS>())` from source plane; depth-converted via `>> (BITS - 8)` for u8 output, native depth for u16 output |

`!ALPHA_SRC || ALPHA` const-asserted at every template top.

### Hardenings (Codex review fixes)

- **Source alpha is masked with `bits_mask::<BITS>()` before depth
  conversion** — `Yuva444p10Frame::try_new` accepts unchecked u16
  samples; without masking an overrange `1024` at BITS=10 would shift
  to `256` and cast to u8 zero, silently turning over-range alpha
  into transparent output. Same masking pattern that Y/U/V already
  use. Pinned by 2 regression tests at the sinker layer.
- **`MixedSinker<Yuva444p10>` wires alpha-drop paths** for `with_rgb`
  / `with_rgb_u16` / `with_luma` / `with_hsv` (declared on the
  generic `MixedSinker<F>` impl) — initial implementation only wrote
  RGBA buffers, leaving the others as silent stale-buffer bugs.
  Pinned by 4 cross-format byte-equivalence tests against
  `MixedSinker<Yuv444p10>`.

### Tests

- **Per-backend SIMD equivalence tests**: 30 per backend × 5 backends
  for `Yuva444p10` (5 u8 added in PR #33 + 5 u16 added in PR #34).
  Solid-alpha + random-alpha + tail-width coverage. All x86 tests
  carry `is_x86_feature_detected!` early-return guards.
- **Sinker integration tests**: 17 (PR #32 added 7 covering alpha
  pass-through / opacity contracts / buffer-too-short error paths;
  PR #32 review-fix added 7 covering alpha-drop paths + Strategy A
  combine; PR #32 review-fix added 2 covering overrange-alpha
  masking).
- **Test count growth**: 578 → 588 on aarch64-darwin host (583 after
  PR #33, 588 after PR #34); +5 NEON tests run at each tranche; the
  +20 x86/wasm tests fire on their respective CI runners.

### Notes

- **Sink-side YUVA + Ship 8 sinks are now end-to-end for the format**:
  with `Yuva444p10Frame` source and `MixedSinker<Yuva444p10>` sink,
  the alpha plane flows through to `with_rgba` / `with_rgba_u16`
  output. `with_rgb` / `with_rgb_u16` / `with_luma` / `with_hsv`
  are alpha-drop (reuse `Yuv444p10` row kernels).
- **Subsequent vertical slices (Ship 8b‑2 onward)** will mass-apply
  the established Strategy B template to other Yuva format families:
  `Yuva420p*` (4:2:0 with α — `yuva420p`, `yuva420p9/10/16`),
  `Yuva422p*` (4:2:2 with α — `yuva422p`, `yuva422p9/10/16`), and
  the remaining `Yuva444p*` variants (8-bit, 9-bit, 16-bit). The
  template's third generic + per-backend wrapper pattern is now
  proven; subsequent slices reuse it mechanically.

## Ship 8 — alpha + RGBA output (`with_rgba` / `with_rgba_u16`)

Adds packed RGBA output across the YUV format inventory. Every YUV
source is now sinkable to packed `R, G, B, A` u8 (alpha = `0xFF`) and,
for native-depth high-bit-depth sources, to packed u16 RGBA (alpha =
`(1 << BITS) - 1` for BITS-generic kernels, `0xFFFF` for the
dedicated 16-bit kernel family). The sink-side RGBA gap was the
single biggest unmet ask — image rendering, masking, and
alpha-aware composition all consume packed RGBA, and every
downstream of `colconv` benefits.

### Surface added

- **Per-format builders** on `MixedSinker<F>`: `with_rgba` /
  `set_rgba` (u8) for every wired format; `with_rgba_u16` /
  `set_rgba_u16` for the high-bit-depth families. Attaching RGBA
  to a sink that doesn't write it is a **compile error** (no
  silent stale-buffer bug) — each format's builder lives on its
  format-specific impl block, only added once `process` is wired.
- **Per-format public dispatchers** in `colconv::row`: `*_to_rgba_row`
  + `*_to_rgba_u16_row` siblings of every `*_to_rgb_*` dispatcher.
  Same SIMD-via-`use_simd` shape; same scalar reference contract.
- **Strategy A combine**: when both `with_rgb` and `with_rgba` are
  attached, `process` runs the YUV→RGB kernel once and fans out to
  RGBA via `expand_rgb_to_rgba_row` / `expand_rgb_u16_to_rgba_u16_row<BITS>`
  (memory-bound copy + alpha pad, ~7W bytes/row) instead of running
  the YUV math twice. ~2× speedup for the both-buffers caller.

### Mass-apply tracker

Each tranche shipped as a separate PR (or sub-PR series) to keep
review weight tractable. **All RGBA work is staged so the const-ALPHA
template lands per-format with a stable public-API signature; SIMD
backends are wired in follow-up sub-PRs without breaking call sites.**

| # | Tranche | Formats | Status |
|---|---|---|---|
| 1 | 4:2:0 planar | `Yuv420p` | ✅ shipped (PR #16) |
| 2 | 4:2:0 semi-planar | `Nv12`, `Nv21` | ✅ shipped (PR #17) — shared `<SWAP_UV, ALPHA>` template |
| 3 | 4:2:2 planar + semi-planar | `Yuv422p`, `Nv16` | ✅ shipped (PR #18) — wiring-only, reuses tranche-1+2 kernels |
| 4a | 4:4:4 planar | `Yuv444p` | ✅ shipped (PR #19) — kernel refactor across all 5 backends |
| 4b | 4:4:4 semi-planar | `Nv24`, `Nv42` | ✅ shipped (PR #20) — `<SWAP_UV, ALPHA>` template + Strategy A combine retro-applied to all 8 wired families |
| 4c | 4:4:0 planar | `Yuv440p` | ✅ shipped (PR #22) — wiring-only (reuses `yuv_444_to_rgba_row`) |
| 5 | High-bit 4:2:0 | `Yuv420p9/10/12/14/16`, `P010/P012/P016` | ✅ shipped — **5** scalar prep + dispatchers (PR #24); **5a** u8 SIMD across all 5 backends (PR #25); **5b** u16 SIMD + sinker integration (PR #26) |
| 6 | High-bit 4:2:2 | `Yuv422p9/10/12/14/16`, `P210/P212/P216` | ✅ shipped (PR #28) — sinker-only; reuses tranche-5 row kernels via the established 4:2:2 → 4:2:0 dispatcher pattern. (`Yuv440p10/12` deferred to tranche 7 alongside the 4:4:4 work it depends on.) |
| 7 | High-bit 4:4:4 + 4:4:0 | `Yuv444p9/10/12/14/16`, `P410/P412/P416`, `Yuv440p10/12` | ✅ shipped — **7** scalar prep + dispatchers (PR #29); **7b** u8 SIMD across all 5 backends (PR #30); **7c** u16 SIMD + sinker integration incl. `Yuv440p10/12` reusing 4:4:4 dispatchers (PR #31) |
| 8 | RAW | `Bayer`, `Bayer16<BITS>` | (deferred — RAW already has `with_luma_coefficients`) |

### SIMD coverage

**All 7 tranches (Ship 8 complete)**: 5 backends (NEON, SSE4.1, AVX2,
AVX-512, wasm simd128) have the const-ALPHA `<…, ALPHA>` template
wired for both u8 and u16 RGBA paths across every high-bit kernel
family (4:2:0 in tranche 5; 4:4:4 + Pn-444 in tranche 7). 4:2:2 and
4:4:0 sinkers reuse 4:2:0 / 4:4:4 dispatchers respectively — no new
SIMD code needed for those subsampling families. Per-arch RGBA store
helpers added in tranche 5: `vst4q_u8` / `vst4q_u16` (NEON),
`write_rgba_16` / `write_rgba_u16_8` (SSE4.1, AVX2 via re-export),
`write_rgba_64` / `write_rgba_u16_32` + `write_quarter_rgba`
(AVX-512), `u8x16_splat` / `i16x8_shuffle`-based `write_rgba_u16_8`
(wasm). Reused verbatim across tranches 5–7.

### Cleanup PRs

- **PR #21** — refactored inline `mod tests` blocks out of per-arch
  backend source files into sibling `tests.rs` files (NEON / SSE4.1 /
  AVX2 / AVX-512 / wasm simd128 + scalar + sinker/mixed). Pure
  layout reorg, no behavior change.
- **PR #23** — narrowed visibility of internal helpers and tightened
  module boundaries surfaced by the Strategy A retroactive refactor.
- **PR #27** — split the remaining inline `mod tests` blocks
  (`src/frame.rs`, `src/raw/types.rs`, `src/raw/bayer.rs`,
  `src/raw/bayer16.rs`) into sibling files. Same shape as PR #21.

### Tests (cumulative through PR #31, Ship 8 complete)

- **534 tests pass on aarch64-darwin** (host) at Ship 8 close;
  trajectory: 507 (PR #28, 4:2:2 sinker) → 513 (PR #29, 4:4:4 scalar
  prep) → 519 (PR #30, 4:4:4 u8 SIMD) → 534 (PR #31, 4:4:4 u16 SIMD
  + sinker).
- Per-arch RGBA equivalence tests: ~30 per high-bit family across all
  5 backends — tranche 5 added 4:2:0 (u8 + u16, BITS=9/10/12/14 + 16
  + Pn); tranche 7b/7c added 4:4:4 (u8 + u16, BITS=9/10/12/14 + 16 +
  Pn-444). All matrices × ranges × natural-block + tail widths.
- Sinker integration tests: 8 in PR #26 (4:2:0), 8 in PR #28 (4:2:2),
  6 in PR #29 (4:4:4 scalar), 9 in PR #31 (4:4:4 + Yuv440p10 cross-
  family kernel-reuse proof). Cover standalone-RGBA, Strategy A
  combine, and buffer-too-short error variants.
- All x86 `#[test]` functions exercising new SIMD kernels include
  `is_x86_feature_detected!` early-return guards (per the PR #25 CI
  fallout — without them, ASAN sanitizer saw `SIGILL` and Miri
  reported UB on runners lacking the feature).

### Notes

- **Strategy B deferred**: a third const generic on every kernel
  (`<SWAP_UV, RGB_OUT, RGBA_OUT>`) eliminating the L1-hot RGB readback
  in the Strategy A path was considered and rejected as ~2,500 LOC
  for L1-noise improvement. See `docs/color-conversion-functions.md`
  § Ship 8 → Combined RGB + RGBA path for the design notes.
- **Source-side YUVA** (Ship 8b — separate follow-up): not part of
  this Ship. Adds YUVA frame types (`Yuv420pAFrame`, etc.) so the
  alpha plane flows through to RGBA output instead of being padded
  to opaque. Ship 8 only addresses the sink-side RGBA gap.

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
both new families. The `OddWidth` variant message was reworded
format-agnostically (`"horizontally-subsampled chroma requires even
width"`) since it now surfaces from both `PnFrame::try_new` (4:2:0)
and `PnFrame422::try_new` (4:2:2). `PnFrame444` has no parity
constraint and never emits this variant.

### SIMD coverage (4:4:4 family)

| Kernel                                  | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `p_n_444_to_rgb_row<BITS>`              |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_to_rgb_u16_row<BITS>`          |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_16_to_rgb_row`                 |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_16_to_rgb_u16_row`             |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |

**Native SIMD on every supported backend** for both u8 and u16 output.
Block sizes per iteration:

| Backend       | u8 / u16-low | u16 i64 (P416) |
| ------------- | :----------: | :------------: |
| NEON          | 16 px        | 8 px           |
| SSE4.1        | 16 px        | 8 px           |
| AVX2          | 32 px        | 16 px          |
| AVX-512       | 64 px        | 32 px          |
| wasm simd128  | 16 px        | 8 px           |

UV deinterleave per-arch: `vld2q_u16` (NEON), `_mm_shuffle_epi8` +
permutes (SSE4.1), `_mm256_shuffle_epi8` + `_mm256_permute2x128_si256`
(AVX2), `_mm512_shuffle_epi8` + `_mm512_permutexvar_epi64` (AVX-512),
`u8x16_swizzle` (wasm simd128).

The 16-bit u16-output i64 chroma path uses **native `_mm512_srai_epi64`
on AVX-512** and **native `i64x2_shr` on wasm** — no bias trick. AVX2
and SSE4.1 use the `srai64_15_x4` / `srai64_15` bias trick (those ISAs
lack arithmetic i64 right shift). NEON uses native `vshrq_n_s64`.

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
- 25 new per-arch SIMD scalar-equivalence tests for the new
  `p_n_444_to_rgb_*<BITS>` and `p_n_444_16_to_rgb_*` kernels —
  5 tests × 5 backends (NEON, SSE4.1, AVX2, AVX-512, wasm simd128).
  Cover all 6 ColorMatrix variants × full + limited range at the
  backend's natural width, plus tail widths {1, 3, 7, 8, 9, 15, 16,
  17, 31, 33, 47, 63, 65, 95, 127, 129, 1920, 1921} forcing
  scalar-tail fallback at every block-size boundary.
- **Total suite: 318 passed on aarch64** (up from 304 at Ship 6b);
  +20 tests fire on x86_64 (15 SSE4.1 / AVX2 / AVX-512) / wasm32 (5)
  CI runners.

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


