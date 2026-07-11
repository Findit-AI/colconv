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
# MSan / TSan recompile an instrumented `std` via `-Zbuild-std`; instrumenting
# std *and* the whole feature set at once exceeds the runner's memory (the
# crate compile is OOM-killed), so they are sharded into memory-bounded feature
# groups — each shard compiles `std` plus one group, keeping the peak well
# under the runner limit. ASan / LSan instrument only the crate (no
# `-Zbuild-std`) and have no such memory ceiling — one `--all-features` job
# would fit — but its single long compile is exactly the window the CI pool's
# runner reclamation keeps killing (exit 143), so ASan/LSan also shard, purely
# for shorter per-job durations.
#
# The shard lists live in `.github/workflows/ci.yml`: each sanitizer's shards
# are chosen so their `cargo test --tests -- --list` outputs union to the whole
# `--all-features` inventory (supplementary shards close the combination-gated
# conjunctions a bare per-family split would leave in zero shards). This script
# just runs the selector `$1` over the feature group `$2`.
WHICH="${1:-all}"
FEATURE_GROUP="${2:-}"

if [ -n "$FEATURE_GROUP" ]; then
  FEATURE_FLAGS=(--no-default-features --features "std $FEATURE_GROUP")
else
  FEATURE_FLAGS=(--all-features)
fi

run_asan_lsan() {
  RUSTFLAGS="-Z sanitizer=address" \
    cargo test --tests --target "$TARGET" "${FEATURE_FLAGS[@]}"
  RUSTFLAGS="-Z sanitizer=leak" \
    cargo test --tests --target "$TARGET" "${FEATURE_FLAGS[@]}"
}

run_msan() {
  RUSTFLAGS="-Z sanitizer=memory" \
    cargo -Zbuild-std test --tests --target "$TARGET" "${FEATURE_FLAGS[@]}"
}

run_tsan() {
  RUSTFLAGS="-Z sanitizer=thread" \
    cargo -Zbuild-std test --tests --target "$TARGET" "${FEATURE_FLAGS[@]}"
}

case "$WHICH" in
  asan-lsan) run_asan_lsan ;;
  msan) run_msan ;;
  tsan) run_tsan ;;
  all) run_asan_lsan; run_msan; run_tsan ;;
  *) echo "unknown sanitizer selector: $WHICH" >&2; exit 1 ;;
esac
