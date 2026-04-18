//! Shared helpers for the x86_64 SIMD backends.
//!
//! Items here use only SSE2 + SSSE3 intrinsics, so they're safe to
//! call from any x86 backend at SSSE3 or above (currently SSE4.1 and
//! AVX2; AVX‑512 will reuse them too). `#[inline(always)]` guarantees
//! they inline into the caller, inheriting its `#[target_feature]`
//! context.

use core::arch::x86_64::{
  __m128i, _mm_loadu_si128, _mm_or_si128, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storeu_si128,
};

/// Writes 16 pixels of packed RGB (48 bytes) from three u8x16 channel
/// vectors.
///
/// Three output blocks of 16 bytes each interleave R, G, B triples.
/// Each channel contributes specific bytes to each block; the shuffle
/// masks below assign those bytes (with `-1` = 0x80 = "zero the lane,
/// to be OR'd in by another channel's contribution").
///
/// Conceptually, block 0 (bytes 0..16) takes:
/// `R0, G0, B0, R1, G1, B1, R2, G2, B2, R3, G3, B3, R4, G4, B4, R5`.
/// Block 1 (bytes 16..32):
/// `G5, B5, R6, G6, B6, R7, G7, B7, R8, G8, B8, R9, G9, B9, R10, G10`.
/// Block 2 (bytes 32..48):
/// `B10, R11, G11, B11, ..., R15, G15, B15`.
///
/// Each of the three 16‑byte stores is the OR of three shuffles of
/// the R, G, B inputs. This is the well‑known SSSE3 3‑way interleave
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
pub(super) unsafe fn write_rgb_16(r: __m128i, g: __m128i, b: __m128i, ptr: *mut u8) {
  unsafe {
    // Shuffle masks for block 0 (first 16 output bytes).
    //   dst byte i gets source byte mask[i] from the corresponding
    //   input channel (R for r_mask, G for g_mask, B for b_mask).
    //   0x80 (`-1` as i8) zeroes that output lane.
    let r0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r0), _mm_shuffle_epi8(g, g0)),
      _mm_shuffle_epi8(b, b0),
    );

    // Block 1 (bytes 16..32).
    let r1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r1), _mm_shuffle_epi8(g, g1)),
      _mm_shuffle_epi8(b, b1),
    );

    // Block 2 (bytes 32..48).
    let r2 = _mm_setr_epi8(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = _mm_setr_epi8(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let b2 = _mm_setr_epi8(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r2), _mm_shuffle_epi8(g, g2)),
      _mm_shuffle_epi8(b, b2),
    );

    _mm_storeu_si128(ptr.cast(), out0);
    _mm_storeu_si128(ptr.add(16).cast(), out1);
    _mm_storeu_si128(ptr.add(32).cast(), out2);
  }
}

/// Swaps the outer two channels of 16 packed 3‑byte pixels (48 bytes
/// in, 48 bytes out). Drives both BGR→RGB and RGB→BGR conversions
/// since the transformation is self‑inverse.
///
/// Uses the SSSE3 `_mm_shuffle_epi8` 3‑way gather pattern: each 16‑byte
/// output chunk is built from shuffles of the three adjacent input
/// chunks, combined with `_mm_or_si128`. 7 shuffles + 4 ORs per 16
/// pixels. Mask values verified byte‑by‑byte against the scalar
/// reference (see the equivalence tests in `neon`/x86 backends).
///
/// # Safety
///
/// - `input_ptr` must point to at least 48 readable bytes.
/// - `output_ptr` must point to at least 48 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - The calling function must have SSSE3 available (either through
///   `#[target_feature(enable = "ssse3")]` / a superset feature like
///   `"sse4.1"` / `"avx2"` / `"avx512bw"`, or the target's defaults).
#[inline(always)]
pub(super) unsafe fn swap_rb_16_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());

    // Output chunk 0 (abs bytes 0..16): 15 bytes from chunk 0, byte 15
    // (= R5) pulled from chunk 1 local position 1.
    let m00 = _mm_setr_epi8(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
    let m01 = _mm_setr_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    );
    let out0 = _mm_or_si128(_mm_shuffle_epi8(in0, m00), _mm_shuffle_epi8(in1, m01));

    // Output chunk 1 (abs bytes 16..32): most from chunk 1, byte 17
    // (= B5) from chunk 0, byte 30 (= R10) from chunk 2.
    let m10 = _mm_setr_epi8(
      -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m11 = _mm_setr_epi8(0, -1, 4, 3, 2, 7, 6, 5, 10, 9, 8, 13, 12, 11, -1, 15);
    let m12 = _mm_setr_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1,
    );
    let out1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(in0, m10), _mm_shuffle_epi8(in1, m11)),
      _mm_shuffle_epi8(in2, m12),
    );

    // Output chunk 2 (abs bytes 32..48): 15 bytes from chunk 2, byte
    // 32 (= B10) pulled from chunk 1 local position 14.
    let m20 = _mm_setr_epi8(
      14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m21 = _mm_setr_epi8(-1, 3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10, 15, 14, 13);
    let out2 = _mm_or_si128(_mm_shuffle_epi8(in1, m20), _mm_shuffle_epi8(in2, m21));

    _mm_storeu_si128(output_ptr.cast(), out0);
    _mm_storeu_si128(output_ptr.add(16).cast(), out1);
    _mm_storeu_si128(output_ptr.add(32).cast(), out2);
  }
}
