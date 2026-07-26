<div align="center">
<h1>pixon</h1>
</div>
<div align="center">

SIMD-dispatched color-conversion kernels covering the FFmpeg `AVPixelFormat` space, with a Sink-based API so consumers pick which derived outputs (RGB / Luma / HSV / custom) they want without paying for the ones they don't.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/pixon-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fpixon" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/pixon/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-studio/pixon?style=for-the-badge&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-pixon-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/pixon?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/pixon?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCwxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTZjLTEyLjExOC0xNy45MzctMzIuMjYyLTI4LjY0NS01My44ODItMjguNjQ1IGMtMzUuODMzLDAtNjQuOTg1LDI5LjE1Mi02NC45ODUsNjQuOTg2czI5LjE1Miw2NC45ODYsNjQuOTg1LDY0Ljk4NmMyMi4yODEsMCw0Mi43NTktMTEuMjE4LDU0Ljc3OC0zMC4wMDkgQzIwMC4yMDgsMTY5LjE0NywxOTguOTg1LDE2My41ODIsMTk0Ljc5OCwxNjAuOTAzeiIvPiA8L2c+IDwvZz4KDTwvc3ZnPg==" height="22">

</div>

## What it is

pixon is a header-light, `no_std`-friendly color-conversion library for
video pipelines:

- **One-call decode** — the `Convert` builder turns a validated source
  frame into any subset of RGB / RGBA / Luma / HSV with zero redundant
  parameters, an optional fused resize, and a single fallible call. See
  [Quick start](#quick-start).
- **SIMD-dispatched** — every kernel ships NEON, SSE4.1, AVX2, AVX-512,
  and wasm-simd128 backends plus a scalar fallback, selected once at
  startup (no per-row branching). The scalar path is the reference
  implementation every other tier is equivalence-tested against.
- **Sink-based output** — pick exactly which derived outputs a frame
  produces (`RGB`, `RGBA`, `Luma`, `HSV`, …); unused kernels don't
  compile, so binaries don't carry dead code.
- **FFmpeg-coverage** — source-format markers track the FFmpeg
  `AVPixelFormat` space (YUV planar / semi-planar / packed / 4:4:4
  packed, Y-series, V210, RGB 8/10/16/F16/F32, GBR, Gray, XYZ, Bayer,
  PAL8 / Mono). Pair pixon with [mediaframe][mediaframe-url] for the
  pixel-data + `SourceFormat` traits the kernels are generic over.
- **Colorimetry-honest** — every H.273 `ColorMatrix` decodes (including
  the non-affine ICtCp, IPT-C2, SMPTE ST 2085 and chroma-derived
  variants), `ChromaLocation` drives siting-aware chroma upsampling
  (center / co-sited, top / bottom variants included), and SMPTE ST 428-1
  primaries can be interpreted per FFmpeg's tabulated values or as true
  CIE XYZ.
- **Fused resampling** — downscale *while* converting in a single pass:
  a resampler splices into the convert pipeline at the earliest stage
  that realizes its averaging domain, so the frame is binned once and
  converted once. Box-coverage (`cv2.INTER_AREA`), separable filter
  kernels (PIL-byte-exact for `u8`), and swscale `BICUBLIN` are all
  available, with optional gamma-correct (linear-light) and
  scene-referred averaging. See [Resampling](#resampling).

## Quick start

```rust
use pixon::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat};
use pixon::frame::Yuv420pFrame;

fn decode(y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32) -> Result<Vec<u8>, pixon::Error> {
    let frame = Yuv420pFrame::new(y, u, v, w, h, w, w / 2, w / 2);
    let spec = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, ColorMatrix::Bt709);

    let mut rgb = vec![0u8; 1280 * 720 * 3];
    Convert::from(&frame)      // dimensions + format come from the frame
        .spec(spec)            // one colorimetric truth; per-format knobs derive from it
        .resize(1280, 720)     // optional fused downscale
        .rgb(&mut rgb)         // attach any subset of outputs
        .run()?;               // the only fallible call
    Ok(rgb)
}
```

`Convert` is the golden path for every supported source format. When you
need streaming, a custom sink, custom output geometry, or Bayer, drop one
tier down: assemble a [`MixedSinker`][doc-url] and drive it with the
matching `{format}_to` walker — the same machinery `Convert` runs on. (The
per-row kernel functions are internal as of 0.2.)

## Resampling

When a sink is configured with a resampler (instead of the default
`NoopResampler`), pixon plans the resample once and **fuses it into the
convert walk** — the source is binned/filtered and converted to the chosen
outputs in one pass, never materializing an intermediate full-resolution
frame. Resampling requires the `alloc` (or `std`) tier.

Pick a resampler by the convention you calibrate against:

| resampler | convention | notes |
|---|---|---|
| `NoopResampler` | — | identity (the default) |
| `AreaResampler` | `cv2.INTER_AREA` | exact integer box-coverage spans, fractional ratios; downscale-only |
| `FilteredResampler<K>` | PIL `Image.resize` | separable kernel `K`; the `u8` path is **byte-exact** to Pillow |
| `Bicublin` | swscale `BICUBLIN` | cubic luma + bilinear chroma |

Filter kernels (`K: FilterKernel`): `Triangle` (PIL `BILINEAR`),
`CatmullRom` (PIL `BICUBIC`), `Lanczos3` (PIL `LANCZOS`), `Lanczos4`,
`Mitchell`, `CubicBSpline`, `Gaussian`, `BlackmanSinc`, `Spline16` /
`Spline36` / `Spline64`, `OpenCvCubic` (`cv2.INTER_CUBIC`), and
`SwscaleBicubic`.

The **averaging domain** controls *where* in the pipeline the binning
happens, trading speed for colorimetric correctness:

- `AveragingDomain::Encoded` (default) — average in the encoded (gamma)
  space, the cv2/swscale convention; fastest.
- `AveragingDomain::Linear` — decode to linear light, average, re-encode
  (gamma-correct resize). The transfer function is caller-configurable
  (`TransferFunction::{Srgb, Bt1886, Gamma22, …}`, defaulting per the
  source's color matrix), and `LinearMode::SceneReferred` averages an
  unclamped-`f32` decode to preserve out-of-gamut excursions.

## Installation

```toml
[dependencies]
pixon = "0.2"
```

## Examples

Runnable, self-contained walkthroughs of every functional area live in
[`examples/`](examples/):

```sh
cargo run --example quickstart        # the Tier-0 golden call
cargo run --example formats_tour      # one call, every source family
cargo run --example multi_output      # rgb + rgba + luma + hsv in one pass
cargo run --example colorimetry       # matrices, ranges, ST 428-1
cargo run --example chroma_siting     # every ChromaLocation honored
cargo run --example resize_area       # fused INTER_AREA downscale
cargo run --example resize_filtered   # filter kernels, upscale included
cargo run --example tier1_resampling  # linear-light averaging + BICUBLIN
cargo run --example tier1_sinker      # manual MixedSinker + walker
cargo run --example bayer_demosaic    # raw pipeline: WB + CCM + demosaic
```

## Feature flags

Capability tiers (additive):

| flag | role |
|---|---|
| _none_ (`--no-default-features`) | `no_std + no_alloc` |
| `alloc` | `no_std + alloc` (pulls `libm` for the scalar path) |
| `std` (default) | `std` |

Format-family gates (opt out of kernels you don't ship):

| flag | enables |
|---|---|
| `frame` (default) | umbrella — every family below |
| `yuv-planar` | YUV planar 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 (8/10/12/14/16-bit) |
| `yuv-semi-planar` | NV-family (NV12 / NV16 / NV21 / NV24 / NV42) |
| `yuva` | YUVA 4:2:0 / 4:2:2 / 4:4:4 (auto-enables `yuv-planar`) |
| `yuv-packed` | packed YUV (UYVY / YUYV / 4:1:1) |
| `yuv-444-packed` | packed YUV 4:4:4 (AYUV64 / VUYA / Y410 / Y412 / XV30 / XV36) |
| `y2xx` | Y2-family packed 4:2:2 (Y210 / Y212 / Y216 / V210) |
| `v210` | V210 packed 4:2:2 (10-bit, 6-pixels-per-block) |
| `rgb` | packed RGB 8/16-bit (RGB / RGBA / BGR / BGRA / RGB48 / RGBA64) |
| `rgb-float` | packed RGB f16 / f32 |
| `rgb-legacy` | legacy packed RGB (RGB565 / RGB555 / RGB444 + BGR variants) |
| `gbr` | planar GBR / GBRA (8-bit, high-bit, f16, f32) |
| `gray` | gray Y8 / Y16 / YF16 / YF32 / YA8 / YA16 |
| `bayer` | Bayer 8/16-bit (RGGB / GRBG / GBRG / BGGR) |
| `xyz` | XYZ 12-bit (DCDM / DCP) |
| `mono` | 1-bit mono + PAL8 palette |

Each family gate forwards to the matching `mediaframe/<family>` so the
source-format markers and `Frame` types compile in lockstep.

## License

`pixon` is licensed under the GNU General Public License v3.0 or
later (GPL-3.0-or-later).

See [LICENSE](LICENSE) for the full text, or
<https://www.gnu.org/licenses/gpl-3.0.html>.

Copyright (C) 2026 FinDIT Studio authors.

[Github-url]: https://github.com/findit-studio/pixon/
[CI-url]: https://github.com/findit-studio/pixon/actions/workflows/ci.yml
[doc-url]: https://docs.rs/pixon
[crates-url]: https://crates.io/crates/pixon
[codecov-url]: https://app.codecov.io/gh/findit-studio/pixon/
[mediaframe-url]: https://crates.io/crates/mediaframe
