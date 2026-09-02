# Changelog

All notable changes to pixon are documented here. `pixon`'s own version
history starts at 0.1.0. The 0.1.0–0.2.1 entries below it were published
under the crate's former name, `colconv`, and are kept for provenance —
they record APIs that shipped as `colconv` and keep the paths they shipped
with. Note the two series are independent: `pixon` 0.1.0 is the code
`colconv` 0.2.1 shipped, and `pixon` 0.2.0 has no relation to `colconv`
0.2.x.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html); pre-1.0
breaking changes bump the `x` in `0.x.y`.

## 0.5.0 — 2026-09-02

**Breaking**, on one count: the public dependency `mediaframe` crosses
0.9 → 0.10, carrying a fix to the `other()` / `FromStr` relationship
through every vocabulary pixon re-exports. No pixon-authored API
changes shape or behavior in this release.

### Changed

- **`mediaframe` 0.9 → 0.10.** One upstream major, entirely mediaframe's
  own 0.10.0 — the prior `"0.9"` pin already spanned 0.9.0–0.9.2 (a
  `lang`-registry internal relayout and two additive codec households,
  neither reachable from pixon's surface), so 0.10.0 is the whole delta.
  Two changes, checked against mediaframe's own 0.10.0 CHANGELOG entry,
  not assumed:
  - **A case-sensitivity axis for every vocabulary's parse table**
    (`Insensitive` / `Sensitive`, `pub(crate)`). mediaframe's own notes
    declare this a "zero behaviour change" release-over-release: all
    twenty-two `Other(SmolStr)` households — `PixelFormat` and the six
    `color` vocabularies pixon re-exports among them — declare
    `Insensitive`, the fold every one of them already matched by. A
    no-op here.
  - **`other()` now delegates to `FromStr`'s own match table instead of
    wrapping unconditionally, and a stranger's spelling is preserved
    verbatim in `Other` rather than ASCII-folded to lowercase.** A real
    behavior change on the households themselves, so this crate's own
    call sites were censused rather than assumed clear: no `.other(`,
    `.parse(`, `from_str(` or direct `::Other(` construction exists
    anywhere in `src/` or `examples/` — pixon consumes `PixelFormat`
    and the `color` vocabularies (`ColorMatrix`/`Matrix`, `Primaries`,
    `Transfer`, `DynamicRange`, `ChromaLocation`, `DcpTargetGamut`)
    exclusively as typed, named variants, never through string parsing.
    (`frame::{Rotation, FieldOrder, StereoMode}` ride along in the
    wholesale `pub use mediaframe::frame` re-export and see no pixon
    code at all.) The one place this crate *does* call `.other(...)` is
    its own test suite: 43 call sites across five types (`ChromaLocation`
    × 32, `DynamicRange` × 5, `Transfer` × 3, `ColorMatrix` × 2,
    `DcpTargetGamut` × 1 — chroma-siting, colorimetry and
    resample-strategy coverage), all built from six synthetic,
    already-lowercase slugs (`"unassigned-7"`, `"unassigned-9"`,
    `"unassigned-42"`, `"unassigned-99"`, `"acescct"`, `"acescg"`) that
    name no roster member of the type they're called on — checked
    against mediaframe's own `FromStr` match tables for all five types,
    not assumed. ASCII-fold and verbatim are byte-identical for an
    already-lowercase input, and none of the six collide with a named
    variant, so every one of the 43 sites still lands in `Other`, with
    the same stored spelling 0.9 gave it. Production dispatch *does*
    pattern-match several of these households directly on their named
    variants — not exclusively through a closed `KernelMatrix` /
    `KernelGamut` selector: separate `matches!(tag, ColorMatrix::Ictcp)`
    / `ChromaDerivedCl` / `IptC2` / `Smpte2085` checks each gate a
    non-affine kernel path in `row/dispatch/yuv444/yuv444p12.rs` and
    `sinker/mixed/subsampled_4_4_4_high_bit/yuv444p.rs`; `Primaries` ×
    `Transfer` resolve `ClSystem` in `row/scalar/cl/mod.rs`;
    `ChromaLocation` drives the siting predicates in
    `sinker/mixed/mod.rs`; `DynamicRange::Full` gates a branch in
    `walker/mod.rs`. Every one of those matches is on the closed set of
    *named* variants, though, with `Other` falling through an
    unhandled/default arm: a census of every match arm across `src/`
    found none that destructures `Other`'s payload (no `Other(s)` /
    `Other(slug)` binding anywhere), so no dispatch decision reads an
    `Other` value's spelling — only which variant a value is, which the
    0.10 crossing does not change for any input pixon's own code ever
    constructs, string or otherwise. Wire and public contracts are
    unchanged for every roster value pixon's API surface carries.

### Fixed

- **`lib.rs`'s "Supported source formats" table documented roughly a third
  of the shipped roster.** Rebuilt mechanically from the crate's own
  `Source`-trait exhaustiveness proof (`src/convert/source_coverage.rs`):
  143 shipped formats now have a table row (up from 49), adding the RGB /
  RGB-legacy / RGB-float / Gray / Mono families (previously undocumented
  entirely), the GBR high-bit-depth tier, native YUVA, the packed 4:2:2
  8-bit family, and the stray gaps `Yuv410p` / `Yuv444p10Msb` /
  `Yuv444p12Msb` / `Nv20` / `Ayuv` / `Uyva` / `Vyu444` / `Xv48`.
- **The "YUVA sources (alpha-drop)" section described already-shipped work
  as a future item** ("RGBA pass-through ... is the dedicated Ship 8 work
  item"). `with_rgba` / `with_rgba_u16` (real alpha pass-through on
  `MixedSinker`) and the native `Yuva420p` / `Yuva422p` / `Yuva444p` frame
  families have shipped since 0.2.0; the section now documents them as
  present, keeps alpha-drop as a still-valid lighter-weight alternative for
  callers who skip the `yuva` feature, and no longer hand-duplicates a
  format list that had already drifted (it was missing every 12-bit YUVA
  variant).
- Docs-only: no code, no public API and no behavior changed. `cargo fmt`,
  `cargo +stable doc --all-features` (zero warnings traceable to `lib.rs`),
  doctests and the full `--all-features` test suite (7215 tests) are all
  green, unchanged from before this commit.

## 0.4.0 — 2026-08-30

**Breaking**, on one count: the public dependency `mediaframe` crosses
0.7 → 0.9, two more upstream majors on top of the 0.4 → 0.7 crossing
0.3.0 already carries. No pixon-authored API changes shape or behavior
in this release.

### Changed

- **`mediaframe` 0.7 → 0.9.** Two upstream majors, neither touching
  `pixel_format` or `color` — the only two mediaframe modules pixon's
  public API is built on. Checked against mediaframe's own commit
  history, not assumed: zero commits touch either module between the
  0.7.0 and 0.9.0 tags, so this is a re-pin, not a migration.
  - **0.8.0** adds `image::Format` (a new still-image vocabulary
    household) and an `extensions()` face on `container::Format` /
    `audio::ContainerFormat`. pixon re-exports none of `image`,
    `container` or `audio` — a no-op here. (This half of the crossing
    already sat on `main` unreleased, with no changelog entry of its
    own; this release is the first to publish it.)
  - **0.9.0** retires the lossy `lang::Language` wrapper for a
    four-type `lang` household (`Language`, `ScriptSubtag`, `Region`,
    `LanguageId`) and swaps `icu_locale_core` for `smol-bytes` /
    `simdutf8` internally. pixon has no `lang` surface — a no-op here
    too.

## 0.3.0 — 2026-08-28

**Breaking**, on one count: the public dependency `mediaframe` crosses
0.4 → 0.7, carrying three majors of mediaframe's own breaking changes
into the vocabularies pixon re-exports. No pixon-authored API changes
shape or behavior in this release.

### Changed

- **`mediaframe` 0.4 → 0.7.** Three upstream majors; both API-visible
  changes below land in the 0.4 → 0.5 crossing. 0.6.0's changes are all
  `audio::*` / `subtitle::*` (pixon has no audio or subtitle surface and
  re-exports none of it); 0.7.0's sole change is `mediatime` 0.3 → 0.4,
  additive only per mediaframe's own changelog (zero fallout in its own
  source) — pixon re-exports none of mediaframe's `mediatime`-typed
  surface (`frame::TimestampedFrame`'s `pts` / `duration`) either, so it
  is a no-op here too:
  - **`FromStr::Err` is now `Infallible`** for every all-tier vocabulary
    pixon re-exports — `PixelFormat`, `ColorMatrix` (mediaframe's
    `Matrix`), `Primaries`, `Transfer`, `DynamicRange`, `ChromaLocation`,
    `DcpTargetGamut`. The `Parse*Error` types themselves are untouched
    and still exported by mediaframe; what moves is the associated `Err`
    type a caller's own `s.parse::<pixon::PixelFormat>()` resolves to.
    pixon names none of these error types in its own source, so nothing
    here needed a code change — a downstream `match` arm, `From` impl,
    or turbofish that does name one of the old `Parse*Error` types may
    need to drop it.
  - **`raw::BayerPattern`** (a re-export of mediaframe's
    `frame::BayerPattern`) **is no longer `#[non_exhaustive]`.** The four
    named patterns (`Rggb` / `Bggr` / `Grbg` / `Gbrg`) were always the
    whole geometric closure — a 2x2 Bayer tile admits exactly four
    top-left phases. pixon's own `pattern_phases` kernel and its test
    twin carried a trailing `_ => unreachable!("invalid BayerPattern")`
    arm only because the re-exported enum was non-exhaustive; with it
    closed, both matches are now exhaustive over the four variants and
    the dead arm is removed (it would otherwise be a compile error under
    this repo's `-D warnings` clippy gate: `unreachable_patterns`).

## 0.2.0 — 2026-08-20

**Breaking**, on one count with a wide blast radius: the public dependency
`mediaframe` crosses 0.1 → 0.4, and with it the colour vocabularies pixon
re-exports *and* the shape of every `{fmt}_to` walker. Everything below follows
from that crossing. Two headlines: a correctness fix — **a colour matrix pixon
cannot decode is now refused instead of silently decoded as BT.709** — and a
seam move — **the walkers no longer take a matrix or a gamut beside the sink;
they ask the sink**.

### Added

- **`ColorSpec::kernel_matrix`** — the one door where the open colour-matrix
  descriptor is exchanged for the closed coefficient selector, and the one
  place an unconvertible matrix is refused. The table is exhaustive and
  documented on the method:

  | `ColorMatrix` | result |
  | --- | --- |
  | `Bt601` `Bt709` `Unspecified` `Fcc` `Bt470Bg` `Smpte170M` `Smpte240m` `YCgCo` `Bt2020Ncl` `ChromaDerivedNcl` | the matching `KernelMatrix` |
  | `Ictcp` `ChromaDerivedCl` `IptC2` `Smpte2085` | `KernelMatrix::Bt709` — pixon decodes all four **non-affinely** from the signalled transfer / primaries; the affine selector is reached only where that tag is underdetermined (an ICtCp source naming neither PQ nor HLG), and resolving underdetermined colorimetry to BT.709 is the same prerogative exercised for `Unspecified` |
  | `Rgb` | `KernelMatrix::Bt709` — the GBR identity names no YCbCr basis; the luma derivation is pinned to BT.709 regardless |
  | `Bt2020Cl` `YCgCoRe` `YCgCoRo` `Other(_)` | **`UnsupportedKernelMatrixError`** |

- **`ColorSpec::of_matrix(KernelMatrix)`** — the casual affine path. Colour
  intent has exactly one carrier in pixon (`ColorSpec`), so a caller who knows
  only "these are BT.601 samples" still says so with a spec instead of reaching
  for a second door into `YuvOptions`. Everything unnamed stays at the
  unspecified posture — `PixelFormat::None` (so **no** format-pinned range: use
  `resolve` / `from_info` for a `yuvj*` source), limited range, `Unspecified`
  primaries / transfer / siting. It takes the *closed* selector, so it can only
  describe an affine source and `kernel_matrix` on the result never fails; the
  four non-affine decodes need a transfer and primaries and so still build a
  full spec.
- `pixon::KernelMatrix`, `pixon::KernelGamut`, `pixon::UnsupportedKernelMatrixError`
  and `pixon::UnsupportedKernelGamutError`, re-exported from mediaframe.
- **`MixedSinker::with_target_gamut(&DcpTargetGamut)` / `set_target_gamut`** —
  the same exchange for the XYZ12 output gamut, refusing a gamut with no
  tabulated XYZ → RGB matrix. It lives on the sink because a gamut describes
  the *output* and the sink **is** the output; `xyz12_to` reads it from there.
- **`MixedSinker::kernel_matrix()` / `target_gamut()`** — what the walkers ask
  for, once per walk. Both are set only from a description (`ColorSpec`,
  `DcpTargetGamut`); neither has a setter that names a closed selector.
- **`Convert::target_gamut(DcpTargetGamut)`** — the Tier-0 XYZ12 gamut door.
  It exists only on an XYZ12 builder (a type-constrained `impl`), so it cannot
  be set on a source that would ignore it.
- `MixedSinkerError::UnsupportedColorMatrix` and
  `MixedSinkerError::UnsupportedTargetGamut` — how the two refusals surface out
  of the `Convert` tier, raised before a single row is read.

### Changed

- **Breaking: a refused matrix is an error, not BT.709 pixels.** Through 0.1
  the coefficient tables (`Coefficients::for_matrix`, `luma_coefficients_q15`)
  and the XYZ gamut table ended in a `_ => BT.709` / `_ => DCI-P3` wildcard, so
  a `Bt2020Cl`, `YCgCoRe`, `YCgCoRo` or unrecognised matrix decoded as BT.709
  and returned a wrong picture with no diagnostic. Those three wildcard arms
  are **deleted**: the tables now match exhaustively over the closed
  `KernelMatrix` / `KernelGamut` vocabularies, so a coefficient set added
  upstream is a compile error here rather than a silent BT.709 frame, and the
  matrices with no pixon decode are refused at `ColorSpec::kernel_matrix`.
  The two arms that *do* resolve to BT.709 — `Unspecified` (the vocabulary's
  own default) and `ChromaDerivedNcl` at the primaries-blind entry point — are
  now spelled out with their rationale instead of riding a fallback.

- **Breaking: the kernels take `KernelMatrix`, not `ColorMatrix`.** A walker
  row now carries the closed selector, and every kernel, row and sinker
  signature takes `KernelMatrix` (`KernelGamut` for XYZ12). `ColorSpec` keeps
  the **open** `ColorMatrix` descriptor — it describes a stream, and the four
  non-affine decodes read their tag from it.

- **Breaking: the walkers ask the sink for the selector.** mediaframe 0.4
  removed the matrix from every `{fmt}_to` signature and the gamut from
  `xyz12_to`, and reads them off `PixelSink::kernel_matrix` /
  `Xyz12Sink::target_gamut` once per walk — two doors onto one fact being a bug
  generator. pixon consumes that shape: `MixedSinker` answers both from the
  description it was configured with, `YuvOptions` carries only `full_range`,
  and `Xyz12Options` is **removed** (its one field was the gamut, so the XYZ12
  walk options are now the unit `()`). Migration:

  ```rust
  - yuv420p_to(&frame, spec.full_range(), spec.kernel_matrix()?, &mut sink)?;
  + let mut sink = sink.with_color_spec(&spec)?;   // the sink already knew
  + yuv420p_to(&frame, spec.full_range(), &mut sink)?;

  - xyz12_to(&frame, KernelGamut::Rec709, &mut sink)?;
  + let mut sink = sink.with_target_gamut(&DcpTargetGamut::Rec709)?;
  + xyz12_to(&frame, &mut sink)?;
  ```

  A sink that names no matrix decodes limited-range BT.709 — the posture the
  walker argument defaulted to. **A caller who passed a non-BT.709 matrix
  positionally and never gave the sink a description must now give it one**:
  that is the one shape where the crossing changes a picture, and it changes it
  from "whatever the caller passed" to "whatever the sink was built with",
  which is the only answer that cannot disagree with itself.

- **Breaking: `*Row::new` is `pub(crate)` upstream.** An out-of-tree
  kernel-parity suite that drove a single row without a frame uses mediaframe's
  `#[doc(hidden)] *Row::for_tests` door instead; it takes the identical
  parameter list and carries no stability promise.

- **Breaking: the closed selectors are not settable — colour intent has one
  source.** `YuvOptions::with_matrix(KernelMatrix)` is **removed** and no
  replacement names a `KernelMatrix` directly. A `KernelMatrix` enters a walk
  only out of `ColorSpec::kernel_matrix`, through
  `MixedSinker::with_color_spec`; a `KernelGamut` only out of
  `MixedSinker::with_target_gamut(&DcpTargetGamut)`. Each closed vocabulary
  therefore has exactly one exchange point with the open descriptor that was
  supposed to select it, and neither can be conjured alongside — or in
  contradiction of — the description the sink is decoding against. Migration:

  ```rust
  - let opts = YuvOptions::new().with_matrix(KernelMatrix::Bt601);
  + let sink = sink.with_color_spec(&ColorSpec::of_matrix(KernelMatrix::Bt601))?;
  // …or, when a fuller description is at hand, straight from it:
  + let sink = sink.with_color_spec(&spec)?;
  ```

  `full_range` is a quantisation fact about the frame rather than colour intent
  the sink owns, so it stays a walker parameter and keeps its own setters.

- **Breaking: `MixedSinker::with_color_spec` / `set_color_spec` are fallible**,
  take `&ColorSpec`, and are no longer gated to `yuv-planar`. One call now
  configures the whole decode — the row's `KernelMatrix`, the siting, the
  primaries, the transfer and the open matrix descriptor the non-affine gate
  reads — and refuses a matrix pixon cannot decode right there, before a row is
  read. `YuvOptions::from_color_spec` carries only the range and is infallible;
  `FromSpec::from_spec` keeps its `Result` shape.

- **Breaking: `Convert::format_options` overrides quantisation, not colour.**
  It could previously carry a `YuvOptions` derived from a *different* spec and
  put a different matrix in front of each reader — the one place the
  single-source rule was deliberately set aside. The options carry no colour
  value any more, so the exception is gone and `Convert::spec` is the only
  matrix door.

- **Breaking: `ColorSpec` is no longer `Copy`** (`Clone` only), and neither are
  the re-exported `ColorMatrix`, `Primaries`, `Transfer`, `ChromaLocation`,
  `DynamicRange`, `ColorInfo` and `PixelFormat` — mediaframe 0.3 gave each an
  owned `Other(..)` escape. `ColorSpec::resolve` / `from_info` and the
  accessors that return these types are no longer `const`, and
  `TransferFunction::for_matrix` takes `&ColorMatrix`.

- **Breaking: the numeric `Unknown(u32)` escape is gone** from every re-exported
  colour vocabulary; mediaframe 0.3 replaced it with `Other(slug)`. Code
  spelling `ColorMatrix::Unknown(9)` becomes `ColorMatrix::other("...")`, and
  `from_u32` now returns `Option<Self>`.

- Public dependency `mediaframe` 0.1 → 0.4. The retirement of
  `VideoCodec::{V308,V408,V410}` in mediaframe 0.4 (FFmpeg n9.0 dropped them)
  does not reach pixon: this crate is pixel-domain and names no codec
  vocabulary at all.

### Fixed

- `Bt2020Cl`, `YCgCoRe`, `YCgCoRo` and unnamed matrices no longer decode as
  BT.709. pixon implements none of them; a constant-luminance or
  reversible-YCgCo source decoded through the BT.709 affine matrix is a wrong,
  silently-wrong picture. (`ChromaDerivedCl` — H.273 code 13 — *is* pixon's
  constant-luminance decode and is unaffected; `Bt2020Cl` is code 10, which
  pixon never implemented.)

## 0.1.0 — 2026-07-26

### Changed

- **The crate is renamed `colconv` → `pixon`.** The package name, the library
  name and therefore every import path change; nothing else does. The API is
  byte-for-byte the one `colconv` 0.2.1 shipped — no type, function, trait,
  feature flag or behavior was added, removed or altered.

  **`pixon` starts its version history at 0.1.0** because it is a new name on
  crates.io with no prior releases: a first and only version of `0.2.2` would
  imply a 0.1.x and 0.2.0–0.2.1 that never existed under this name. The number
  therefore carries no maturity claim relative to `colconv` — this is `colconv`
  0.2.1's code, not an earlier or reduced one. `colconv` 0.2.1 remains on
  crates.io and is frozen; it will receive no further releases.

  Migration is two mechanical edits:

  ```toml
  # Cargo.toml
  - colconv = "0.2"
  + pixon = "0.1"
  ```

  ```rust
  // every module
  - use colconv::{Convert, ColorSpec};
  + use pixon::{Convert, ColorSpec};
  ```

  A crate-wide `s/colconv/pixon/` over your own sources is sufficient; there
  are no renamed items to reconcile beyond the crate root itself.

- Repository-internal dispatch-override cfgs are renamed to match the crate:
  `colconv_force_scalar` → `pixon_force_scalar`, `colconv_disable_avx512` →
  `pixon_disable_avx512`, `colconv_disable_avx2` → `pixon_disable_avx2`. These
  are testing/coverage helpers set via `RUSTFLAGS` by this repository's own CI
  and carry no stability guarantee.

## 0.2.0 & 0.2.1 — 2026-07-11

### Added

- **Tier 0 `Convert` — the golden one-call decode.** `Convert::from(&frame)`
  decodes a validated source frame to any subset of RGB / RGBA / Luma / HSV
  outputs with zero redundant parameters: dimensions come from the frame,
  colorimetry from a single `ColorSpec`, and each per-format walk knob is
  *derived* from that spec (overridable via `format_options`). Every setter is
  infallible; `run()` is the only fallible call. Optional area (`resize`) /
  filtered (`resize_with`) downscale rides the same call. Generic over the
  source marker — no `dyn`, no allocation beyond what `MixedSinker` already
  performs; the sealed `Source` / `FromSpec` traits are emitted per format by
  the same `walker!` table that drives the `Walker` tier, so the two never
  drift. `no_std` (`alloc`) compatible. (RFC #392, additive in this release.)
- `colconv::Error` — the canonical crate-level error name, a re-export alias of
  `sinker::MixedSinkerError` (which remains available under its original path).
- `unstable-bench-internals` feature — a semver-**exempt**, repository-internal
  tier that exposes a `#[doc(hidden)]` `colconv::bench_internals` shim over the
  now-private `row` kernels so this repo's own Criterion benches can measure
  them. Not public API; carries no stability guarantee.
- **Chroma-siting-aware upsampling — every `ChromaLocation` is now honored**
  (#302). Horizontal center vs. co-sited reconstruction lands for the packed
  4:2:2 Y-series (Y210 / Y212 / Y216) and V210 and for the 4:1:1 / 4:1:0
  families (a new 1→4 centered kernel); vertical `Bottom` / `BottomLeft` for
  4:2:0; and `Top` / `TopLeft` — previously rejected — across Yuv420p
  (8-bit and 9–16-bit), NV12 / NV21, P010 / P012 / P016, Yuv440p (4:4:0) and
  Yuva420p, via a new forward-lookahead one-row delay in the streaming sink
  (the final `process` call flushes the held row retry-atomically). Siting is
  frozen per frame: flipping `set_chroma_location` mid-frame now returns the
  typed `ChromaSitingChanged` instead of silently mixing phases.
- `ColorMatrix::IptC2` and `ColorMatrix::Smpte2085` decode support (ITU-T
  H.273 codes 15 and 11, spec-exact coefficients; scalar tier like the other
  non-affine PQ matrices), completing the H.273 matrix coverage (#303).
- `St428Interpretation` on `MixedSinker` (#310): SMPTE ST 428-1 primaries
  decode per FFmpeg's tabulated D-Cinema values (`FfmpegTabulated`, the
  default — byte-identical to before) or as true CIE XYZ (`CieXyz`, opt-in),
  where a `ChromaDerivedNcl` matrix over ST 428-1 is rejected with the typed
  `St428CieXyzUnsupported` rather than deriving a colorimetrically meaningless
  YCbCr matrix from RGB-tabulated primaries. Consumes mediaframe 0.1.9's
  `Primaries::is_cie_xyz()`.
- Direct **YUV → HSV** row kernels for every YUV source family, so a
  `MixedSinker` with `with_hsv()` (and no RGB / RGBA attached) converts
  straight from YUV to HSV on the direct and native fast tiers — skipping
  the YUV → RGB intermediate and the source-width RGB scratch it required
  (`with_luma()` + `with_hsv()` on a YUV source now allocates no RGB
  buffer). Each fused kernel is bit-identical, per SIMD tier, to the prior
  `rgb_to_hsv_row(yuv_to_rgb_row(...))` path. Coverage — scalar plus all
  five SIMD backends (NEON / SSE4.1 / AVX2 / AVX-512 / wasm-simd128) each:
  - planar 8-bit (4:2:0 / 4:2:2 / 4:4:4 / 4:4:0 / 4:1:0 / 4:1:1);
  - planar high-bit (9 / 10 / 12 / 14 / 16-bit, little- and big-endian);
  - semi-planar NV12 / NV16 / NV21 / NV24 / NV42 and high-bit P010 / P012
    / P016;
  - packed UYVY / YUYV / YVYU and 4:1:1 (UYYVYY411);
  - 4:4:4-packed AYUV64 / VUYA / VUYX / XV36;
  - Y210 / Y212 / Y216;
  - YUVA 4:2:0 / 4:2:2 / 4:4:4 (reusing the planar kernels — the alpha
    plane is independent of HSV).

  High-bit sources convert through an 8-bit RGB intermediate, bit-identical
  to the existing high-bit HSV path (HSV is `H` in `[0, 179]`, `S` / `V` in
  `[0, 255]`).
- **RGB-free HSV-only row-stage resample** (8-bit planar YUV): when a
  `MixedSinker` requests `with_hsv()` and no RGB / RGBA output, the
  row-stage area-resample bins Y / U / V and converts to HSV at output
  width instead of staging a source-width RGB row — no RGB scratch is
  allocated, the same RGB-free contract the direct and native fast tiers
  honor. HSV-only row-stage output therefore averages in the **YUV
  domain** (bit-identical to the native fast tier, which bins the codes
  then converts) rather than the RGB domain; a sink that also attaches
  RGB / RGBA keeps the RGB-staged path (convert once, derive HSV). Covers
  `Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p` / `Yuv410p` / `Yuv411p`;
  the chroma-subsampled formats weight partial trailing chroma rows /
  columns (non-multiple-of-2 / -4 dimensions) by their true luma
  coverage. The filter-resampler twin, the semi-planar, and the high-bit
  / packed row-stage paths stage RGB for HSV — a future follow-up.
- SIMD acceleration for the fused-downscale engine (NEON, SSE4.1, AVX2,
  AVX-512, wasm-simd128): the area H-pass consumes a plan-time zero-padded u16
  weight arena — each span padded to a multiple of 8, so the kernels
  run pure wide loads with zero lanes annihilating samples past the
  last tap — and the V-pass AXPY widens through exact u64 lanes. Both
  are bit-identical to the scalar reference (pinned by a pre-divide
  differential across chunk-boundary, multi-chunk, row-end and
  u16-fallback geometries) and route through the sinkers' existing
  `with_simd` switch. Gate numbers (1080p -> 336x189, Apple Silicon):
  native rgb+hsv 3.7ms -> 2.1ms, row-stage rgb+hsv 5.6ms -> 2.5ms,
  luma-only 1.9ms -> 0.95ms, fused `Rgb24` 4.5ms -> 1.6ms. On x86 the
  highest available tier wins (AVX-512 -> AVX2 -> SSE4.1); the wider
  tiers widen within a span (16 / 32 taps per step) and outpace SSE4.1
  past 16-tap spans (16x-plus downscales), falling to the shared
  128-bit step below that.
- `MixedSinker` gains a resampling-strategy type parameter
  (`R = NoopResampler`) and a `with_resampler(width, height, resampler)`
  constructor: the strategy's plan fixes the sinker's **output
  geometry** once, before any buffer attaches. Output buffers
  (`with_rgb` / `with_luma` / `with_hsv` / per-format `with_rgba`,
  `with_luma_u16`, …) now validate against the output geometry
  (`out_width()` / `out_height()`); `begin_frame` keeps validating
  walkers against the source geometry. Under the default
  `NoopResampler` (identity plan) behavior is unchanged.
- cv2 `INTER_AREA` parity harness: golden outputs generated by
  `ci/gen_cv2_goldens.py` (sources synthesized by a shared LCG on both
  sides, so only cv2's outputs are committed); the exact-integer engine
  matches within ±1 LSB across integer and fractional ratios, gray and
  interleaved RGB.
- The native decimation tier for `Yuv420p` (`with_native`, default
  on): Y, U and V are binned at full output resolution on their own
  grids and converted once per output row through the 4:4:4 kernels,
  so no alignment constraint applies to the output geometry and
  luma-only sinks never read the chroma planes. Color semantics:
  native averages in the YUV domain then converts (libswscale-class
  fused semantics; luma bit-identical, in-gamut color within rounding;
  out-of-gamut content diverges as far as it sits outside the gamut —
  measured examples of 34/255 and 117/255 are pinned by regression);
  `with_native(false)` selects strict RGB-domain `INTER_AREA`
  semantics instead. `Rgb24` routes the
  fused path with no conversion step at all (binning the packed row
  IS the work). Gate numbers (1080p -> 336x189, scalar engine):
  native rgb+hsv 3.7ms vs row-stage 5.7ms; luma-only 1.9ms either
  tier; full-res conversion baseline 0.83ms — superseded by the
  engine-SIMD entry above.
- Fused downscale now runs end-to-end for `Yuv420p`: the row-stage
  streaming engine area-averages each converted source row into the
  output geometry with exact `u64` integer arithmetic (round-half-up
  by `src_w * src_h`), emitting every output row on its last
  contributing source row. All output channels participate (`RGB`,
  `RGBA`, `Luma`, `Luma u16`, `HSV` — HSV/RGBA derive from the
  resampled RGB row); luma-only sinks touch just the Y plane.
- New `resample` module: sealed `Resampler` trait, `NoopResampler`,
  `AreaResampler` (exact `cv2.INTER_AREA`-convention area span plans —
  per-axis integer coverage weights, fractional ratios included),
  `ResamplePlan`, and structured `ResampleError` (wrapped by the new
  `MixedSinkerError::Resample` variant). Groundwork for the fused
  downscale-first walk (#123, #125).

### Changed

- **The chroma-upsample reconstruct kernels are now SIMD** across all five
  backends (NEON / SSE4.1 / AVX2 / AVX-512 / wasm-simd128): the 1→2 centered
  horizontal, the 4:2:0 bottom-sited vertical, the 4:4:0 vertical and the new
  1→4 centered families, for u8 / u16 / semi-planar element types — closing
  the last scalar hot-spot in siting-aware RGB decode. Byte-identical to the
  scalar reference per backend; `with_simd(false)` still forces scalar.

### Fixed

- Direct (identity) decode paths across the sited formats now enforce the
  same per-frame chroma-siting freeze the resample tiers always had. Before,
  a mid-frame `set_chroma_location` flip on ~30 identity paths (packed 4:2:2,
  planar and high-bit 4:2:0 / 4:2:2, NV, P0xx / P2xx, YUVA) silently mixed
  centered and co-sited phases in one frame — with a stale vertical-lookback
  in the 4:2:0 cases — instead of rejecting with `ChromaSitingChanged`.

### Removed

- **The `row` module is now `pub(crate)` (breaking).** The entire per-row
  kernel free-function surface (531 `pub fn`s, up to 10 positional parameters
  each) was kernel plumbing consumed internally by `MixedSinker` and the
  `{fmt}_to` walkers; as public API it was unreviewable and could never be made
  ergonomic. It is gone from the public surface.

  Migration (≤ 3 lines in the common case): decode through the Tier-0
  `Convert` builder, or assemble a `MixedSinker` and drive it with the matching
  `{fmt}_to` walker — both are byte-identical to the old row-kernel path. If you
  called `row` kernels directly for a use case those tiers do not cover, please
  open an issue describing it: a curated, struct-parameter row API can return
  for a real external consumer.

## 0.1.0 — 2026-06-08

Initial public release. colconv is a `no_std`-friendly SIMD-dispatched
color-conversion library covering the FFmpeg `AVPixelFormat` space.

### Architecture

- **Runtime SIMD dispatch.** Every kernel ships AVX-512, AVX2, SSE4.1,
  and a scalar fallback. Backend is selected once at startup via
  `is_x86_feature_detected!` with no per-row branching. The scalar
  path is the reference implementation every other tier is
  equivalence-tested against.
- **Sink-based output API.** Consumers pick which derived outputs a
  source frame produces (`RGB`, `RGBA`, `Luma`, `HSV`, custom);
  kernels for unselected outputs don't compile. Cross-format
  helpers (Q15 ranges, `rgb_expand`, HSV conversion, raw type
  surface, alpha extraction, Y-plane to luma) are always available.
- **Per-format opt-in.** 18 format-family feature gates forward to
  the matching `mediaframe/<family>` so source-format markers and
  `Frame` types compile in lockstep with the kernel code.
- **`no_std` + `alloc`.** Capability tiers are additive — depend on
  `no_std + no_alloc`, `no_std + alloc` (pulls `libm` for the scalar
  path), or full `std` (default).

### Format coverage

| feature | source formats |
|---|---|
| `yuv-planar` | YUV planar 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8/10/12/14/16-bit |
| `yuv-semi-planar` | NV12 / NV16 / NV21 / NV24 / NV42 + 16-bit P210/P212/P216/P410/P412/P416 |
| `yuva` | YUVA 4:2:0 / 4:2:2 / 4:4:4 (auto-enables `yuv-planar`) |
| `yuv-packed` | UYVY / YUYV / 4:1:1 packed |
| `yuv-444-packed` | AYUV64 / VUYA / VUYX / Y410 / V410 / V30X / XV30 / XV36 |
| `y2xx` | Y210 / Y212 / Y216 (10/12/16-bit packed 4:2:2) |
| `v210` | V210 (10-bit packed 4:2:2, 6-pixels-per-block) |
| `rgb` | packed RGB / RGBA / BGR / BGRA 8-bit + RGB48 / BGR48 / RGBA64 / BGRA64 16-bit |
| `rgb-float` | packed RGB f16 / f32 |
| `rgb-legacy` | RGB565 / BGR565 / RGB555 / BGR555 / RGB444 / BGR444 |
| `gbr` | planar GBR / GBRA at 8-bit, 9/10/12/14/16-bit, f16, f32 |
| `gray` | Y8 / Y16 / Yf16 / Yf32 / Ya8 / Ya16 |
| `bayer` | Bayer 8/16-bit (RGGB / GRBG / GBRG / BGGR) with optional WB + CCM |
| `xyz` | XYZ12 (DCDM / DCP) |
| `mono` | 1-bit mono (monoblack / monowhite) + PAL8 palette |

### Configuration

- `ColorMatrix` / range / endianness handled per-kernel; BT.709 and
  BT.601 (limited and full) lane orderings supported across every
  family.
- Bayer family supports optional white balance and color correction
  matrices, with validated `try_new` constructors that reject
  finite-but-extreme inputs that would overflow during the per-pixel
  matmul.

### Quality

- Scalar reference implementation drives equivalence tests across
  every SIMD backend (AVX-512 / AVX2 / SSE4.1 / NEON / WASM SIMD-128).
- Big-endian parity coverage for every `*LE` host-endian-sensitive
  format.
- Comprehensive sinker-layer fixture and dispatcher-layer regression
  suite.
