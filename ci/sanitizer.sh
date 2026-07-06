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
# the runner limit. Coverage note: because each MSan/TSan shard enables only one
# feature group, a path gated on features from two different groups (a
# cross-format conversion) is not MSan/TSan instrumented in any single shard —
# the same trade-off the sharded miri jobs make. ASan/LSan still run the full
# `--all-features` suite, so out-of-bounds, use-after-free and leaks ARE covered
# across every cross-feature combination; only MSan's uninitialised-read and
# TSan's data-race checks are per-group (and these conversions are single-
# threaded pure computation, leaving TSan no cross-thread state to miss).
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
