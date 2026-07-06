#!/bin/bash
set -ex

export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"

# Give the test harness's worker threads an 8 MiB stack. The sanitizer
# (debug + instrumented) builds use markedly larger stack frames, so a
# constrained worker-thread stack can overflow even on output that the
# release build runs comfortably — defence-in-depth alongside the bounded
# `MixedSinker` inline size (see `mixed_sinker_inline_size_stays_small`).
export RUST_MIN_STACK=8388608

TARGET="x86_64-unknown-linux-gnu"

# Selector for which sanitizer(s) to run and, for the build-std ones, which
# feature-group subset to compile.
#
#   $1  which : asan-lsan | msan | tsan | all   (default: all)
#   $2  feature-group : optional space-separated format features
#
# ASan / LSan instrument only the crate (no `-Zbuild-std`), so the whole
# `--all-features` suite fits in one job. MSan / TSan additionally recompile
# an instrumented `std` via `-Zbuild-std`; instrumenting std *and* the whole
# feature set at once exceeds the runner's memory (the crate compile is
# OOM-killed), so those two are sharded by feature group — mirroring the miri
# jobs. Each shard compiles `std` plus one group, keeping the peak well under
# the runner limit. Coverage note: each MSan shard enables yuv-planar, the
# conversion hub that most of the crate's cross-format cfg(all(...)) paths pair
# with, so those conversions are MSan-instrumented in the shard carrying the
# other family; a final residual shard adds the YUV-to-RGB combos whose two
# non-yuv-planar sides fall in different groups (semi-planar / yuva / packed-YUV
# crossed with rgb), so no cross-format conversion path is left uninstrumented
# by MSan. TSan shards are one-group-per-shard — these conversions are single-
# threaded pure computation, so no data race is missed regardless. ASan/LSan
# still run the full `--all-features` suite, covering out-of-bounds,
# use-after-free and leaks across every cross-feature combination.
WHICH="${1:-all}"
FEATURE_GROUP="${2:-}"

if [ -n "$FEATURE_GROUP" ]; then
  BUILD_STD_FEATURES=(--no-default-features --features "std $FEATURE_GROUP")
else
  BUILD_STD_FEATURES=(--all-features)
fi

run_asan_lsan() {
  RUSTFLAGS="-Z sanitizer=address" \
    cargo test --tests --target "$TARGET" --all-features
  RUSTFLAGS="-Z sanitizer=leak" \
    cargo test --tests --target "$TARGET" --all-features
}

run_msan() {
  RUSTFLAGS="-Z sanitizer=memory" \
    cargo -Zbuild-std test --tests --target "$TARGET" "${BUILD_STD_FEATURES[@]}"
}

run_tsan() {
  RUSTFLAGS="-Z sanitizer=thread" \
    cargo -Zbuild-std test --tests --target "$TARGET" "${BUILD_STD_FEATURES[@]}"
}

case "$WHICH" in
  asan-lsan) run_asan_lsan ;;
  msan) run_msan ;;
  tsan) run_tsan ;;
  all) run_asan_lsan; run_msan; run_tsan ;;
  *) echo "unknown sanitizer selector: $WHICH" >&2; exit 1 ;;
esac
