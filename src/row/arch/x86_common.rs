//! Shared helpers for the x86_64 SIMD backends.
//!
//! Items here use only SSE2 + SSSE3 intrinsics, so they're safe to
//! call from any x86 backend at SSSE3 or above (currently SSE4.1 and
//! AVX2; AVX‑512 will reuse them too). `#[inline(always)]` guarantees
//! they inline into the caller, inheriting its `#[target_feature]`
//! context.

use core::arch::x86_64::{
  __m128i, _mm_or_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
};

/// Writes 16 pixels of packed BGR (48 bytes) from three u8x16 channel
/// vectors.
///
/// Three output blocks of 16 bytes each interleave B, G, R triples.
/// Each channel contributes specific bytes to each block; the shuffle
/// masks below assign those bytes (with `-1` = 0x80 = "zero the lane,
/// to be OR'd in by another channel's contribution").
///
/// Conceptually, block 0 (bytes 0..16) takes:
/// `B0, G0, R0, B1, G1, R1, B2, G2, R2, B3, G3, R3, B4, G4, R4, B5`.
/// Block 1 (bytes 16..32):
/// `G5, R5, B6, G6, R6, B7, G7, R7, B8, G8, R8, B9, G9, R9, B10, G10`.
/// Block 2 (bytes 32..48):
/// `R10, B11, G11, R11, ..., B15, G15, R15`.
///
/// Each of the three 16‑byte stores is the OR of three shuffles of
/// the B, G, R inputs. This is the well‑known SSSE3 3‑way interleave
/// pattern from libyuv / OpenCV.
///
/// # Safety
///
/// - `ptr` must point to at least 48 writable, properly aligned (or
///   unaligned‑tolerated via the `storeu` variant) bytes.
/// - The calling function must have SSSE3 available (either through
///   `#[target_feature(enable = "ssse3")]` / a superset feature like
///   `"sse4.1"` or `"avx2"`, or via the target's default feature set).
#[inline(always)]
pub(super) unsafe fn write_bgr_16(b: __m128i, g: __m128i, r: __m128i, ptr: *mut u8) {
  unsafe {
    // Shuffle masks for block 0 (first 16 output bytes).
    //   dst byte i gets source byte mask[i] from the corresponding
    //   input channel (B for b_mask, G for g_mask, R for r_mask).
    //   0x80 (`-1` as i8) zeroes that output lane.
    let b0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let r0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(b, b0), _mm_shuffle_epi8(g, g0)),
      _mm_shuffle_epi8(r, r0),
    );

    // Block 1 (bytes 16..32).
    let b1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let r1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(b, b1), _mm_shuffle_epi8(g, g1)),
      _mm_shuffle_epi8(r, r1),
    );

    // Block 2 (bytes 32..48).
    let b2 = _mm_setr_epi8(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = _mm_setr_epi8(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let r2 = _mm_setr_epi8(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(b, b2), _mm_shuffle_epi8(g, g2)),
      _mm_shuffle_epi8(r, r2),
    );

    _mm_storeu_si128(ptr.cast(), out0);
    _mm_storeu_si128(ptr.add(16).cast(), out1);
    _mm_storeu_si128(ptr.add(32).cast(), out2);
  }
}
