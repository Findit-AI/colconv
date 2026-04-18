# `colconv` — Color Conversion Function Inventory (Design)

> **Scope.** `colconv` provides SIMD-dispatched per-row color-conversion kernels covering the full `AVPixelFormat` space FFmpeg can decode to: mainstream consumer, pro video, HDR, DCP, RAW, and legacy rawvideo.
>
> **Consumer.** FinDIT's indexing / thumbnail / scene-analysis pipelines consume these kernels. Every decoded frame eventually needs zero or more of `{BGR, Luma, HSV}` (plus possibly application-defined reductions like histograms). `colconv` is the shared kernel layer that makes producing those outputs cheap.

---

## 0. Design premises

1. **Sink-based API, one traversal of source.** Kernels walk the source pixel format exactly once and hand rows to a caller-provided `Sink`. The Sink decides what to derive and what to store — luma only, BGR only, triple output, inline histogram, whatever. This replaces the "fused triple output" signature we originally considered (see § 0a for why).
2. **Partition by pixel-format family, not by codec.** Same layout + same subsampling + same bit depth → one kernel.
3. **Integer and float paths are separate.** SIMD templates don't share meaningfully.
4. **Little/big endian is a runtime parameter**, not a separate function.
5. **Integer bit depth is parameterized** (9/10/12/14/16). Internally normalize to `u16` for processing.
6. **YUVA reuses YUV.** Alpha is ignored; matte / compositing indexing is a future hook — don't branch on it now.
7. **Color matrix, gamut, full/limited range are parameters** read from `AVFrame.colorspace` and `AVFrame.color_range`. **Never hardcode BT.601.**
8. **Stride-aware.** Every kernel reads `AVFrame.linesize[]`; never infer from width. FFmpeg adds padding, and some HW decode paths emit negative linesize (vertical flip).

### 0a. Why Sink instead of a fused `<src>_to_bgr_luma_hsv(...)` signature

A fused-triple signature assumes every caller wants all three outputs. In practice they don't:

- Thumbnails want BGR only.
- Motion analysis wants luma only — and for YUV sources, luma **is** the Y plane, so producing it should cost one `memcpy` per row, not a full YUV→BGR→Luma pipeline.
- Scene detection in `scenesdetect` wants luma + HSV, but not BGR.
- Histogram accumulation wants no stored output at all — just counts.

A Sink lets the kernel handle the source-format traversal (stride, chroma upsampling, deinterleave, bit-depth normalization) *once*, and the Sink decides what arithmetic to run per row. When the Sink is narrow (only wants luma from YUV), the kernel has nothing to compute — specialization falls out of monomorphization, not runtime flags.

What we give up: the kernel no longer produces BGR / HSV / Luma directly. A Sink that wants both BGR and HSV calls `bgr_to_hsv_row` on the BGR row *it* just wrote — that's technically two passes over the same row. But the row is ≤ width bytes, freshly written, sitting in L1. The "fused kernel" rule was really about not re-reading source memory, which Sinks still guarantee.

---

## 1. Function inventory

### 1.1 Kernel signatures

Naming convention: `<src>_to<S: <Src>Sink>(src: <SrcFrame>, sink: &mut S)`. One kernel per source family; one Sink trait per source family (the trait's method signature reflects what a row of that format actually contains).

```rust
// Planar YUV — the kernel hands the Sink a row struct carrying the
// Y row (full width) plus the *half-width* U / V rows. Chroma
// upsampling happens inside whichever kernel the Sink delegates to
// (scalar / NEON / SSE4.1 / AVX2 / AVX-512 / wasm simd128) — there's
// no intermediate full-width chroma buffer.
pub struct Yuv420pRow<'a> {
    y: &'a [u8],
    u_half: &'a [u8],
    v_half: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
}
pub trait Yuv420pSink: for<'a> PixelSink<Input<'a> = Yuv420pRow<'a>> {}

pub fn yuv420p_to<S: Yuv420pSink>(
    src: &Yuv420pFrame<'_>,
    full_range: bool,
    matrix: ColorMatrix,
    sink: &mut S,
);

// Semi-planar — same pattern, interleaved UV (also half-width in 4:2:0).
pub struct Nv12Row<'a> { y: &'a [u8], uv_half: &'a [u8], row: usize, /* .. */ }
pub trait Nv12Sink: for<'a> PixelSink<Input<'a> = Nv12Row<'a>> {}
pub fn nv12_to<S: Nv12Sink>(
    src: &Nv12Frame<'_>, full_range: bool, matrix: ColorMatrix, sink: &mut S,
);

// Packed RGB — the kernel is essentially a stride-aware row walker.
pub struct Rgb24Row<'a> { rgb: &'a [u8], row: usize }
pub trait Rgb24Sink: for<'a> PixelSink<Input<'a> = Rgb24Row<'a>> {}
pub fn rgb24_to<S: Rgb24Sink>(src: &RgbFrame<'_>, sink: &mut S);
```

### 1.2 The 48 dispatch entries

Same function inventory as the previous design; only the signatures change to the Sink pattern above.

#### Tier 0 — HW frame entry (dispatcher glue, not a color conversion)

| # | Function | Purpose |
|---|---|---|
| 1 | `hwframe_download_and_dispatch(frame, sink)` | Calls `av_hwframe_transfer_data()` to copy to system memory, then dispatches by the returned SW pix_fmt to the appropriate kernel below. |

**HW → SW pix_fmt mapping** (the dispatch layer maintains):

| HW context | Typical SW download format |
|---|---|
| VideoToolbox | `nv12`, `p010`, `p016` |
| VAAPI | `nv12`, `p010`, `yuv420p` |
| CUDA / NVDEC | `nv12`, `p010`, `p016`, `yuv444p16` |
| D3D11VA / DXVA2 | `nv12`, `p010` |
| QSV | `nv12`, `p010`, `p012` |
| DRM_PRIME | driver-dependent |
| MediaCodec (Android) | `nv12`, `nv21`, vendor-specific |
| Vulkan / OpenCL | depends on import path |

#### Tier 1 — Planar YUV (mainline; ~90% of real decoded output)

| # | Function | Covers `AV_PIX_FMT_*` |
|---|---|---|
| 2 | `yuv420p_to<S: Yuv420pSink>(..)` | `yuv420p`, `yuvj420p`, `yuv420p9/10/12/14/16`, `yuva420p*` |
| 3 | `yuv422p_to<S: Yuv422pSink>(..)` | `yuv422p`, `yuvj422p`, `yuv422p9/10/12/14/16`, `yuva422p*` |
| 4 | `yuv444p_to<S: Yuv444pSink>(..)` | `yuv444p`, `yuvj444p`, `yuv444p9/10/12/14/16`, `yuva444p*` |
| 5 | `yuv440p_to<S: Yuv440pSink>(..)` | `yuv440p`, `yuvj440p`, `yuv440p10/12` |
| 6 | `yuv411p_to<S: Yuv411pSink>(..)` | `yuv411p` — DV-NTSC |
| 7 | `yuv410p_to<S: Yuv410pSink>(..)` | `yuv410p` — legacy, optional |

#### Tier 2 — Semi-planar YUV

| # | Function | Covers |
|---|---|---|
| 8 | `nv12_to<S: Nv12Sink>(..)` | 4:2:0 8-bit |
| 9 | `nv21_to<S: Nv21Sink>(..)` | 4:2:0 8-bit, VU swapped |
| 10 | `nv16_to<S: Nv16Sink>(..)` | 4:2:2 8-bit |
| 11 | `nv24_to<S: Nv24Sink>(..)` | 4:4:4 8-bit |
| 12 | `nv42_to<S: Nv42Sink>(..)` | 4:4:4 8-bit, VU swapped |
| 13 | `p01x_to<S: P01xSink>(layout, ..)` | `layout ∈ {p010, p012, p016, p210, p216, p410, p416}` |

#### Tier 3 — Packed YUV 4:2:2 (8-bit)

| # | Function | Covers |
|---|---|---|
| 14 | `yuyv422_to<S: Yuyv422Sink>(..)` | YUY2 |
| 15 | `uyvy422_to<S: Uyvy422Sink>(..)` | UYVY |
| 16 | `yvyu422_to<S: Yvyu422Sink>(..)` | YVYU |

#### Tier 4 — Packed YUV 4:2:2 (10 / 12 / 16-bit, pro video) ⭐

| # | Function | Notes |
|---|---|---|
| 17 | `v210_to<S: V210Sink>(..)` | 10-bit in a custom 32-bit word packing. De-facto standard in BMD / DIT / ProRes intermediate workflows. **Not the same as p210** — kernel is entirely different. |
| 18 | `y210_to<S: Y210Sink>(..)` | 10-bit MSB-aligned in a 16-bit word |
| 19 | `y212_to<S: Y212Sink>(..)` | 12-bit |
| 20 | `y216_to<S: Y216Sink>(..)` | 16-bit |

#### Tier 5 — Packed YUV 4:4:4

| # | Function | Notes |
|---|---|---|
| 21 | `v410_to<S: V410Sink>(..)` | 10-bit 4:4:4, also known as XV30 |
| 22 | `xv36_to<S: Xv36Sink>(..)` | 12-bit 4:4:4 |
| 23 | `vuya_to<S: VuyaSink>(..)` | 8-bit 4:4:4+α; covers `vuyx` too (α interpreted as padding) |
| 24 | `ayuv64_to<S: Ayuv64Sink>(..)` | 16-bit 4:4:4+α |
| 25 | `uyyvyy411_to<S: Uyyvyy411Sink>(..)` | DV 4:1:1 packed |

#### Tier 6 — Packed RGB/BGR (8-bit)

| # | Function | Notes |
|---|---|---|
| 26 | `bgr24_to<S: Bgr24Sink>(..)` | identity row walker |
| 27 | `rgb24_to<S: Rgb24Sink>(..)` | identity row walker (byte order differs from bgr24) |
| 28 | `bgra_to<S: BgraSink>(..)` | |
| 29 | `rgba_to<S: RgbaSink>(..)` | |
| 30 | `argb_to<S: ArgbSink>(..)` | |
| 31 | `abgr_to<S: AbgrSink>(..)` | |
| 32 | `rgb_padding_to<S: RgbPaddingSink>(order, ..)` | `order ∈ {0rgb, rgb0, 0bgr, bgr0}`. Fourth channel is **padding, not alpha** — kept separate to prevent it being treated as α. |

#### Tier 7 — Packed RGB/BGR (legacy low-bit)

| # | Function | Notes |
|---|---|---|
| 33 | `rgb565_to<S>(order, ..)` | `order ∈ {rgb565, bgr565}` |
| 34 | `rgb555_to<S>(order, ..)` | `order ∈ {rgb555, bgr555}` |
| 35 | `rgb444_to<S>(order, ..)` | `order ∈ {rgb444, bgr444}` |

#### Tier 8 — Packed RGB/BGR (high bit-depth)

| # | Function | Notes |
|---|---|---|
| 36 | `rgb48_to<S>(order, has_alpha, ..)` | 16-bit; `order ∈ {rgb, bgr}`; `has_alpha` covers `rgba64` / `bgra64` |
| 37 | `x2rgb10_to<S>(order, ..)` | 10-bit packed + 2-bit padding (HDR10 RGB path); `order ∈ {x2rgb10, x2bgr10}` |

#### Tier 9 — Float RGB

| # | Function | Notes |
|---|---|---|
| 38 | `rgbf16_to<S>(has_alpha, ..)` | half-float; ACES / EXR adjacency |
| 39 | `rgbf32_to<S>(has_alpha, ..)` | single-precision float |

#### Tier 10 — Planar RGB (GBR)

| # | Function | Covers |
|---|---|---|
| 40 | `gbrp_int_to<S>(depth, has_alpha, ..)` | `gbrp`, `gbrap`, `gbrp9/10/12/14/16`, `gbrap10/12/16` |
| 41 | `gbrp_float_to<S>(has_alpha, ..)` | `gbrpf32`, `gbrapf32` (separate — don't tightly couple with integer) |

#### Tier 11 — Gray

| # | Function | Notes |
|---|---|---|
| 42 | `gray_int_to<S>(depth, ..)` | `gray8`, `gray9/10/12/14/16`. Luma path is a memcpy/up-sample; **bypass BGR→Luma derivation**. |
| 43 | `grayf32_to<S>(..)` | float gray |
| 44 | `ya_to<S>(depth, ..)` | `ya8`, `ya16` — gray + α |

#### Tier 12 — DCP (XYZ)

| # | Function | Notes |
|---|---|---|
| 45 | `xyz12_to<S>(..)` | 12-bit CIE XYZ — DCP-only. Full color-science path: XYZ → linear RGB (Rec.709 or Rec.2020) → gamma → BGR. **Do not** share a kernel with ordinary RGB. |

#### Tier 13 — Palette

| # | Function | Notes |
|---|---|---|
| 46 | `pal8_to<S>(..)` | palette lookup + derived |

#### Tier 14 — Bayer RAW (enable only when R3D / BRAW / NRAW ingest lands)

| # | Function | Notes |
|---|---|---|
| 47 | `bayer_to<S>(pattern, depth, wb, ccm, ..)` | `pattern ∈ {bggr, rggb, grbg, gbrg}`, `depth ∈ {8, 16}`. Includes demosaic + WB + CCM. Demosaic algorithm is a design choice (bilinear vs. better). |

#### Tier 15 — Very legacy (prefer letting swscale fall through)

| # | Function | Notes |
|---|---|---|
| 48 | `mono1bit_to<S>(polarity, ..)` | `monoblack` / `monowhite` |

---

## 2. Priority tiers

| Tier | Scope | Entries | Count |
|---|---|---|---|
| **P0** | Mainstream H.264 / HEVC / AV1 / VP9 / ProRes source | 1, 2, 3, 4, 8, 9, 13, 14, 15, 26, 27, 28, 29, 42 | 14 |
| **P1** | Pro video / HDR / DCP (director / DIT asset libraries) | 17, 18, 19, 20, 21, 22, 23, 24, 36, 37, 45 | 11 |
| **P2** | Completeness (rare but real) | 5, 10, 11, 12, 16, 30, 31, 32, 38, 39, 40, 41, 43, 44, 46 | 15 |
| **P3** | Legacy / RAW / last-resort fallback | 6, 7, 25, 33, 34, 35, 47, 48 | 8 |

**Total: 48 dispatch entries.**

---

## 3. Dispatch-layer implementation rules

### 3.1 Stride-aware

Every kernel reads `AVFrame.linesize[]`. Never derive from width alone.

- FFmpeg may pad rows.
- Some HW decode paths emit **negative linesize** (vertically flipped frames).

### 3.2 Bit-depth normalization

All integer source kernels normalize internally to **`u16`** (left-shift to MSB-align where needed) before handing rows to the Sink. Avoids writing separate 9 / 10 / 12 / 14-bit kernels.

### 3.3 YUV → RGB color matrix is a parameter

```
matrix ∈ { BT.601, BT.709, BT.2020-NCL, SMPTE240M, FCC }
```

Read `matrix` from `AVFrame.colorspace` and `full_range` from `AVFrame.color_range`. **Do not hardcode BT.601.** The kernel does not perform YUV→RGB arithmetic itself — it hands rows to the Sink, and the Sink calls the row-level `yuv_to_bgr_row(..)` primitive (see § 4) with the same matrix/range.

### 3.4 Lock in the HSV definition

Must be committed to, explicitly, in the crate root:

- **OpenCV style** — `H ∈ [0, 180)`, `S`, `V ∈ [0, 255]`
- **Standard HSV** — `H ∈ [0, 360)`, `S`, `V ∈ [0, 1]` or `[0, 100]`

Downstream histogram consumers must match this convention. **Pick one now** and document it as crate-wide policy.

### 3.5 SIMD strategy

Runtime-dispatched per-backend, matching the pattern already used in `scenesdetect::arch`:

| Target | Backend |
|---|---|
| aarch64 | NEON (compile-time; base ARMv8-A ISA) |
| x86 / x86_64 with `std` | Runtime `is_x86_feature_detected!`: AVX2 → SSSE3 → scalar |
| x86 / x86_64 without `std` | Compile-time `target_feature` gating |
| wasm32 with `simd128` | wasm SIMD |
| Everything else | Scalar fallback |

Priority per-kernel (hot paths first):

| Path | Recommendation |
|---|---|
| `yuv420p`, `nv12`, `yuyv422`, `v210` | Hot. Hand-written AVX2 + NEON both. |
| Everything else | Scalar or compiler auto-vectorization. Revisit based on profile data. |

SSE4.1, AVX-512 intentionally not added — fragmented CPU matrix, marginal benefit for byte-plane workloads. Revisit only if profiling demands.

### 3.6 Buffer management

The Sink owns output buffer policy entirely — pool reuse, alignment, lifetimes are all caller concerns. `colconv` itself never allocates output buffers; it writes into Sink-supplied row slices via the trait methods.

---

## 4. Row-level primitives Sinks call

To keep Sinks ergonomic, `colconv` exposes a set of SIMD-dispatched row-level conversion primitives. Common Sinks compose these; custom Sinks can too.

```rust
// YUV → BGR for a single (already-chroma-upsampled) row.
pub fn yuv_to_bgr_row(
    y: &[u8], u: &[u8], v: &[u8],
    bgr_out: &mut [u8],
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
);

// BGR → BT.601 luma (weighted sum).
pub fn bgr_to_luma_row(bgr: &[u8], luma_out: &mut [u8], width: usize);

// BGR → three planar HSV bytes (OpenCV 8-bit encoding).
pub fn bgr_to_hsv_row(
    bgr: &[u8],
    h_out: &mut [u8], s_out: &mut [u8], v_out: &mut [u8],
    row: usize, width: usize,
);

// Future: yuv_to_luma_row (identity for YUV — just memcpy the Y plane).
// Future: bgr_to_gray_row_weighted(matrix, ...) — luma parameterized on matrix.
```

Each primitive is stride-naive (tight-packed row input/output) and SIMD-dispatched to NEON / SSSE3 / wasm as appropriate. Kernels pass tight rows to the Sink; the Sink calls these primitives or does its own arithmetic.

---

## 5. Common Sinks shipped by the crate

```rust
// Just luma. LumaSinker on a YUV source is a memcpy of the Y plane —
// no conversion work. On BGR, it's one `bgr_to_luma_row` per row.
pub struct LumaSinker<'a> { pub out: &'a mut [u8], pub width: usize }

// Just BGR. Identity row walker for BGR sources; full conversion for YUV.
pub struct BgrSinker<'a> { pub out: &'a mut [u8], pub width: usize }

// Just HSV. For YUV sources goes YUV→BGR→HSV internally; for BGR sources
// just HSV conversion.
pub struct HsvSinker<'a> { pub h: &'a mut [u8], pub s: &'a mut [u8], pub v: &'a mut [u8], pub width: usize }

// All three outputs — direct equivalent of the old "fused triple" API.
pub struct MixedSinker<'a> {
    pub bgr:  &'a mut [u8], pub luma: &'a mut [u8],
    pub hsv_h: &'a mut [u8], pub hsv_s: &'a mut [u8], pub hsv_v: &'a mut [u8],
    pub width: usize,
}
```

Each of these impls the relevant per-format Sink trait (`Yuv420pSink`, `Nv12Sink`, `Bgr24Sink`, …). The impls are where format-specific specialization lives — e.g. `LumaSinker::process_row` on a Yuv420pSink is one line of `copy_from_slice`; on a Bgr24Sink it's one call to `bgr_to_luma_row`.

Custom Sinks for histogram binning, downsample-as-you-go, write-to-GPU-staging, etc., are application code and don't live in `colconv`.

---

## 6. Explicit non-goals

- ❌ `hsv_to_luma*` — no use case.
- ❌ A public `yuv_to_bgr` + `bgr_to_hsv` whole-frame slow path — it would get misused. Row-level primitives (§ 4) are the composable unit.
- ❌ Separate `yuva*` kernel family — reuse `yuv*` and drop α.
- ❌ LE/BE function variants — parameterize at runtime.
- ❌ Per-bit-depth function variants — parameterize `depth`.
- ❌ `dyn Sink` trait objects on kernels — the Sink must be concrete at kernel-call time for monomorphization to specialize. `Box<dyn AnySink>` loses the "LumaSinker on YUV is a memcpy" optimization.

---

## 7. Prior art: `scenesdetect::arch`

The `scenesdetect` crate's internal `arch` module already ships working SIMD kernels for a narrow slice of this design (specifically the BGR→{luma, hsv} leg of Tier 6 #26). They're not re-framed as Sinks but the kernels themselves are directly portable to row-level primitives here:

| `scenesdetect` primitive | Maps to `colconv` | Status |
|---|---|---|
| `frame::convert::bgr_to_hsv_planes` | `bgr_to_hsv_row` (§ 4), called per-row | Direct port. NEON · SSSE3 · AVX2 · wasm. |
| `frame::convert::bgr_to_luma` | `bgr_to_luma_row` (§ 4) | Direct port. NEON · SSSE3 · wasm. |

The established SIMD scaffolding transfers verbatim:

- **3-channel packed deinterleave**: NEON `vld3q_u8`, SSSE3 nine-mask `PSHUFB`, wasm `u8x16_swizzle`.
- **Weighted u8 sum**: NEON `vmull_u8 + vmlal_u8`, SSSE3 `PMULLW` + PADDW, wasm `i16x8_mul + i16x8_add`.
- **u8 horizontal sum / count**: NEON `vaddlvq_u8`, SSSE3 `PSADBW` (SAD trick), wasm `u16x8_extadd_pairwise_u8x16`.
- **3×3 stencil with stride-aware row loads**: NEON `vld1_u8` × 9 + widen to i16x8, SSSE3 `_mm_loadl_epi64` × 9 + `_mm_unpacklo_epi8`, wasm `v128_load64_zero` × 9 + `u16x8_extend_low_u8x16`.
- **Runtime dispatch**: `is_x86_feature_detected!` under `std`, `target_feature` cfg gating in no_std, `not(miri)` gate on every SIMD module.
- **Testing pattern**: scalar reference + per-backend scalar-equivalence tests at 4 dim configs (main-loop-only, tail, stride-padded, large).

Once `colconv` reaches feature parity with this subset, `scenesdetect` becomes a consumer — deleting its internal `arch::bgr_to_hsv_planes` / `arch::bgr_to_luma` in favour of `colconv`'s `bgr_to_hsv_row` / `bgr_to_luma_row`.

---

## 8. Rollout order

Filtered from the P0 / P1 list in § 2, weighted by "most common real-world input" plus "cost of groundwork already laid":

1. **Row-level primitives** (§ 4): `yuv_to_bgr_row`, `bgr_to_luma_row`, `bgr_to_hsv_row`. Port from `scenesdetect::arch` for the BGR→ pair; write fresh for `yuv_to_bgr_row`. Gate on matrix / range parameterization — must be plumbed through from day one.
2. **Per-format Sink traits** — `Yuv420pSink`, `Nv12Sink`, `Bgr24Sink` at minimum for the P0 launch.
3. **Common Sinks**: `LumaSinker`, `BgrSinker`, `HsvSinker`, `MixedSinker`. Impl each of the three traits above.
4. **Mainline kernels** in priority order:
   - `yuv420p_to` (entry #2) — the single most common decoder output. Gates the matrix/range plumbing.
   - `nv12_to` (entry #8) — every HW-accelerated decode path.
   - `yuv422p_to` (entry #3), `yuv444p_to` (entry #4).
   - `yuyv422_to` / `uyvy422_to` (entries #14, #15) — packed 4:2:2.
   - `p01x_to` (entry #13) — 10/12/16-bit semi-planar, brings the u16 MSB-align pattern.
   - `rgb24_to`, `bgra_to`, `rgba_to`, `argb_to`, `abgr_to` (entries #27–31) — direct extensions of bgr24 scaffolding.
5. **Pro-video / HDR kernels** (P1 tier) as needed: `v210`, `v410`, `rgb48`, `x2rgb10`, `xyz12`.
6. **Bayer RAW** (P3, #47) only when R3D / BRAW / NRAW ingest comes online.
7. Every kernel gets a golden-frame + pixel-level diff test against swscale as reference. Scalar-equivalence tests compare the SIMD path to a scalar reference across 4 dim configs (main-loop, tail, stride-padded, large).
