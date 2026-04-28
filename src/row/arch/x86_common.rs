//! Shared helpers for the x86_64 SIMD backends.
//!
//! Items here use SSE2 + SSSE3 + SSE4.1 intrinsics (e.g. `_mm_blendv_ps`,
//! `_mm_packus_epi32`), so they're safe to call from any x86 backend at
//! SSE4.1 or above (currently SSE4.1, AVX2, and AVX‑512).
//! `#[inline(always)]` guarantees they inline into the caller,
//! inheriting its `#[target_feature]` context.

use core::arch::x86_64::*;

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

/// Writes 16 pixels of packed RGBA (64 bytes) from four u8x16 channel
/// vectors. Mirrors [`write_rgb_16`] for the 4-channel output path.
///
/// The 4-byte stride aligns cleanly with the 16-byte register width:
/// each output block holds exactly 4 RGBA quads (16 bytes), with R,
/// G, B, A interleaved at positions `(0, 1, 2, 3)`, `(4, 5, 6, 7)`,
/// etc. The shuffle masks are simpler than the 3-channel pattern
/// because a single source byte goes to a single output byte (no
/// channel "split across blocks" boundary).
///
/// Conceptually:
/// - Block 0 (bytes 0..16): R0,G0,B0,A0, R1,G1,B1,A1, R2,G2,B2,A2,
///   R3,G3,B3,A3
/// - Block 1 (bytes 16..32): pixels 4..7
/// - Block 2 (bytes 32..48): pixels 8..11
/// - Block 3 (bytes 48..64): pixels 12..15
///
/// Each block is the OR of four `_mm_shuffle_epi8` gathers — one per
/// channel — with `0x80` (`-1` as i8) zeroing lanes that another
/// channel's shuffle will fill.
///
/// # Safety
///
/// - `ptr` must point to at least 64 writable bytes.
/// - The calling function must have SSSE3 available (via
///   `#[target_feature(enable = "ssse3")]` or a superset).
#[inline(always)]
pub(super) unsafe fn write_rgba_16(r: __m128i, g: __m128i, b: __m128i, a: __m128i, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): pixels 0..3, source bytes 0..3 from
    // each channel placed at output positions
    // (0, 1, 2, 3) for pixel 0, (4, 5, 6, 7) for pixel 1, etc.
    let r0 = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
    let g0 = _mm_setr_epi8(-1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1);
    let b0 = _mm_setr_epi8(-1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1);
    let a0 = _mm_setr_epi8(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);
    let out0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r0), _mm_shuffle_epi8(g, g0)),
      _mm_or_si128(_mm_shuffle_epi8(b, b0), _mm_shuffle_epi8(a, a0)),
    );

    // Block 1 (bytes 16..32): pixels 4..7, source bytes 4..7.
    let r1 = _mm_setr_epi8(4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);
    let g1 = _mm_setr_epi8(-1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1);
    let b1 = _mm_setr_epi8(-1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1);
    let a1 = _mm_setr_epi8(-1, -1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7);
    let out1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r1), _mm_shuffle_epi8(g, g1)),
      _mm_or_si128(_mm_shuffle_epi8(b, b1), _mm_shuffle_epi8(a, a1)),
    );

    // Block 2 (bytes 32..48): pixels 8..11, source bytes 8..11.
    let r2 = _mm_setr_epi8(8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1);
    let g2 = _mm_setr_epi8(-1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1);
    let b2 = _mm_setr_epi8(-1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1);
    let a2 = _mm_setr_epi8(-1, -1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11);
    let out2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r2), _mm_shuffle_epi8(g, g2)),
      _mm_or_si128(_mm_shuffle_epi8(b, b2), _mm_shuffle_epi8(a, a2)),
    );

    // Block 3 (bytes 48..64): pixels 12..15, source bytes 12..15.
    let r3 = _mm_setr_epi8(
      12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1,
    );
    let g3 = _mm_setr_epi8(
      -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1,
    );
    let b3 = _mm_setr_epi8(
      -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1,
    );
    let a3 = _mm_setr_epi8(
      -1, -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15,
    );
    let out3 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r3), _mm_shuffle_epi8(g, g3)),
      _mm_or_si128(_mm_shuffle_epi8(b, b3), _mm_shuffle_epi8(a, a3)),
    );

    _mm_storeu_si128(ptr.cast(), out0);
    _mm_storeu_si128(ptr.add(16).cast(), out1);
    _mm_storeu_si128(ptr.add(32).cast(), out2);
    _mm_storeu_si128(ptr.add(48).cast(), out3);
  }
}

/// Writes 8 pixels of packed **`u16`** RGB (48 bytes = 24 `u16`)
/// from three `u16x8` channel vectors. Drives the SSE4.1 / AVX2 /
/// AVX‑512 high‑bit‑depth kernels' u16 output path.
///
/// Three output blocks of 16 bytes (8 `u16`) each hold:
/// - Block 0: `R0, G0, B0, R1, G1, B1, R2, G2` (u16 indices 0..7)
/// - Block 1: `B2, R3, G3, B3, R4, G4, B4, R5`
/// - Block 2: `G5, B5, R6, G6, B6, R7, G7, B7`
///
/// Each block is the OR of three `_mm_shuffle_epi8` gathers — one
/// from each of R, G, B — with the byte mask picking a pair of
/// adjacent bytes (lo, hi) for every `u16` sourced from that
/// channel. 0x80 (`-1` as i8) zeros the lane, to be OR'd in by
/// another channel's contribution.
///
/// # Safety
///
/// - `ptr` must point to at least 48 writable bytes (aligned or
///   unaligned — we use `storeu`).
/// - The calling function must have SSSE3 available (via SSE4.1 or
///   a superset like AVX2 / AVX‑512BW).
#[inline(always)]
pub(super) unsafe fn write_rgb_u16_8(r: __m128i, g: __m128i, b: __m128i, ptr: *mut u16) {
  unsafe {
    // Block 0 = [R0 G0 B0 R1 G1 B1 R2 G2] — 8 `u16` = 16 bytes.
    // R contributes pairs (0,1), (6,7), (12,13); G pairs (2,3), (8,9),
    // (14,15); B pairs (4,5), (10,11).
    let r0 = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    let g0 = _mm_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    let b0 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    let out0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r0), _mm_shuffle_epi8(g, g0)),
      _mm_shuffle_epi8(b, b0),
    );

    // Block 1 = [B2 R3 G3 B3 R4 G4 B4 R5]. R pairs (6,7), (8,9),
    // (10,11); G pairs (6,7), (8,9); B pairs (4,5), (6,7), (8,9).
    let r1 = _mm_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    let g1 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    let b1 = _mm_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    let out1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r1), _mm_shuffle_epi8(g, g1)),
      _mm_shuffle_epi8(b, b1),
    );

    // Block 2 = [G5 B5 R6 G6 B6 R7 G7 B7]. R pairs (12,13), (14,15);
    // G pairs (10,11), (12,13), (14,15); B pairs (10,11), (12,13),
    // (14,15).
    let r2 = _mm_setr_epi8(
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
    );
    let g2 = _mm_setr_epi8(
      10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
    );
    let b2 = _mm_setr_epi8(
      -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
    );
    let out2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(r, r2), _mm_shuffle_epi8(g, g2)),
      _mm_shuffle_epi8(b, b2),
    );

    _mm_storeu_si128(ptr.cast(), out0);
    _mm_storeu_si128(ptr.add(8).cast(), out1);
    _mm_storeu_si128(ptr.add(16).cast(), out2);
  }
}

/// Interleaves 8 R/G/B/A `u16` samples into packed RGBA quads (32
/// `u16` = 64 bytes). Two 16-bit unpack stages followed by two 32-bit
/// unpack stages produce four 16-byte chunks of `[R, G, B, A]` quads,
/// stored back-to-back via `_mm_storeu_si128`.
///
/// # Safety
///
/// - `ptr` must point to at least 64 writable bytes (aligned or
///   unaligned — we use `storeu`).
/// - The calling function must have SSE2 available (the unpack +
///   `storeu_si128` intrinsics; SSE4.1 / AVX2 / AVX-512 supersets all
///   satisfy this).
#[inline(always)]
pub(super) unsafe fn write_rgba_u16_8(
  r: __m128i,
  g: __m128i,
  b: __m128i,
  a: __m128i,
  ptr: *mut u16,
) {
  unsafe {
    let rg_lo = _mm_unpacklo_epi16(r, g);
    let rg_hi = _mm_unpackhi_epi16(r, g);
    let ba_lo = _mm_unpacklo_epi16(b, a);
    let ba_hi = _mm_unpackhi_epi16(b, a);
    let q0 = _mm_unpacklo_epi32(rg_lo, ba_lo);
    let q1 = _mm_unpackhi_epi32(rg_lo, ba_lo);
    let q2 = _mm_unpacklo_epi32(rg_hi, ba_hi);
    let q3 = _mm_unpackhi_epi32(rg_hi, ba_hi);
    _mm_storeu_si128(ptr.cast(), q0);
    _mm_storeu_si128(ptr.add(8).cast(), q1);
    _mm_storeu_si128(ptr.add(16).cast(), q2);
    _mm_storeu_si128(ptr.add(24).cast(), q3);
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

/// Swaps R↔B in 4 packed BGRA pixels (16 bytes), preserving alpha.
/// One `_mm_shuffle_epi8` per 16-byte vector — within each 4-byte
/// pixel, byte 0 ↔ byte 2 (alpha at byte 3 unchanged). Self-inverse,
/// so the same helper can be used for `RGBA→BGRA` (not currently
/// wired, but the semantics are identical).
///
/// # Safety
///
/// - `input_ptr` must point to at least 16 readable bytes.
/// - `output_ptr` must point to at least 16 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - The calling function must have SSSE3 available (via
///   `#[target_feature(enable = "ssse3")]` or a superset feature
///   like `"sse4.1"` / `"avx2"` / `"avx512bw"`, or the target's
///   defaults).
#[inline(always)]
pub(super) unsafe fn swap_rb_alpha_4_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let mask = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    let in_v = _mm_loadu_si128(input_ptr.cast());
    let out_v = _mm_shuffle_epi8(in_v, mask);
    _mm_storeu_si128(output_ptr.cast(), out_v);
  }
}

/// Drops the alpha byte from 16 packed RGBA pixels (64 input bytes,
/// 48 output bytes). Four input vectors of 4 pixels each compact into
/// three output vectors. Each output block ORs two shuffles drawing
/// from neighbouring input vectors — the same "lane straddle" pattern
/// as [`swap_rb_16_pixels`] for the 3-channel target.
///
/// # Safety
///
/// - `input_ptr` must point to at least 64 readable bytes.
/// - `output_ptr` must point to at least 48 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn drop_alpha_16_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());
    let in3 = _mm_loadu_si128(input_ptr.add(48).cast());

    // out0 (bytes 0..16): pixels 0..3 from in0, pixel 4 + R5 from in1.
    let m00 = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let m01 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 4);
    let out0 = _mm_or_si128(_mm_shuffle_epi8(in0, m00), _mm_shuffle_epi8(in1, m01));

    // out1 (bytes 16..32): G5 B5 R6..R9 G9 B9 R10 G10 from in1+in2.
    let m11 = _mm_setr_epi8(5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9);
    let out1 = _mm_or_si128(_mm_shuffle_epi8(in1, m11), _mm_shuffle_epi8(in2, m12));

    // out2 (bytes 32..48): B10 + pixel 11 from in2, pixels 12..15 from in3.
    let m22 = _mm_setr_epi8(
      10, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14);
    let out2 = _mm_or_si128(_mm_shuffle_epi8(in2, m22), _mm_shuffle_epi8(in3, m23));

    _mm_storeu_si128(output_ptr.cast(), out0);
    _mm_storeu_si128(output_ptr.add(16).cast(), out1);
    _mm_storeu_si128(output_ptr.add(32).cast(), out2);
  }
}

/// Drops the leading alpha byte from 16 packed ARGB pixels (64
/// input bytes → 48 output bytes). Same compaction shape as
/// [`drop_alpha_16_pixels`], but each pixel triple is read at
/// offsets `(+1, +2, +3)` (R, G, B) since alpha is at the **leading**
/// position. Used by the Ship 9c [`Argb`](crate::yuv::Argb)
/// `with_rgb` / `with_luma` / `with_hsv` paths.
///
/// # Safety
///
/// - `input_ptr` must point to at least 64 readable bytes.
/// - `output_ptr` must point to at least 48 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn argb_to_rgb_16_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());
    let in3 = _mm_loadu_si128(input_ptr.add(48).cast());

    let m00 = _mm_setr_epi8(1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -1, -1, -1, -1);
    let m01 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 5);
    let out0 = _mm_or_si128(_mm_shuffle_epi8(in0, m00), _mm_shuffle_epi8(in1, m01));

    let m11 = _mm_setr_epi8(6, 7, 9, 10, 11, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 5, 6, 7, 9, 10);
    let out1 = _mm_or_si128(_mm_shuffle_epi8(in1, m11), _mm_shuffle_epi8(in2, m12));

    let m22 = _mm_setr_epi8(11, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m23 = _mm_setr_epi8(-1, -1, -1, -1, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15);
    let out2 = _mm_or_si128(_mm_shuffle_epi8(in2, m22), _mm_shuffle_epi8(in3, m23));

    _mm_storeu_si128(output_ptr.cast(), out0);
    _mm_storeu_si128(output_ptr.add(16).cast(), out1);
    _mm_storeu_si128(output_ptr.add(32).cast(), out2);
  }
}

/// Drops leading alpha and reverses the inner three bytes for 16
/// packed ABGR pixels (64 input bytes → 48 output bytes). Each
/// pixel triple is read at offsets `(+3, +2, +1)` (R, G, B from
/// `A, B, G, R` input). Used by the Ship 9c
/// [`Abgr`](crate::yuv::Abgr) `with_rgb` / `with_luma` / `with_hsv`
/// paths.
///
/// # Safety
///
/// - `input_ptr` must point to at least 64 readable bytes.
/// - `output_ptr` must point to at least 48 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn abgr_to_rgb_16_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());
    let in3 = _mm_loadu_si128(input_ptr.add(48).cast());

    let m00 = _mm_setr_epi8(3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, -1, -1, -1, -1);
    let m01 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 1, 7);
    let out0 = _mm_or_si128(_mm_shuffle_epi8(in0, m00), _mm_shuffle_epi8(in1, m01));

    let m11 = _mm_setr_epi8(6, 5, 11, 10, 9, 15, 14, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 1, 7, 6, 5, 11, 10);
    let out1 = _mm_or_si128(_mm_shuffle_epi8(in1, m11), _mm_shuffle_epi8(in2, m12));

    let m22 = _mm_setr_epi8(9, 15, 14, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m23 = _mm_setr_epi8(-1, -1, -1, -1, 3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13);
    let out2 = _mm_or_si128(_mm_shuffle_epi8(in2, m22), _mm_shuffle_epi8(in3, m23));

    _mm_storeu_si128(output_ptr.cast(), out0);
    _mm_storeu_si128(output_ptr.add(16).cast(), out1);
    _mm_storeu_si128(output_ptr.add(32).cast(), out2);
  }
}

/// Rotates leading alpha to trailing position in 4 packed ARGB
/// pixels (16 bytes). One `_mm_shuffle_epi8` per 16-byte vector —
/// within each 4-byte pixel: `(A, R, G, B) → (R, G, B, A)`.
///
/// # Safety
///
/// - `input_ptr` must point to at least 16 readable bytes.
/// - `output_ptr` must point to at least 16 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn argb_to_rgba_4_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let mask = _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    let in_v = _mm_loadu_si128(input_ptr.cast());
    let out_v = _mm_shuffle_epi8(in_v, mask);
    _mm_storeu_si128(output_ptr.cast(), out_v);
  }
}

/// Reverses byte order in 4 packed ABGR pixels (16 bytes). One
/// `_mm_shuffle_epi8` per 16-byte vector — within each 4-byte
/// pixel: `(A, B, G, R) → (R, G, B, A)`. Self-inverse.
///
/// # Safety
///
/// - `input_ptr` must point to at least 16 readable bytes.
/// - `output_ptr` must point to at least 16 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn abgr_to_rgba_4_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let mask = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    let in_v = _mm_loadu_si128(input_ptr.cast());
    let out_v = _mm_shuffle_epi8(in_v, mask);
    _mm_storeu_si128(output_ptr.cast(), out_v);
  }
}

/// Swaps R↔B and drops alpha for 16 packed BGRA pixels (64 input
/// bytes → 48 output bytes). Same shape as [`drop_alpha_16_pixels`]
/// but each pixel triple is read with the channel order reversed
/// (`B, G, R, A` → `R, G, B`). Within each 4-byte input pixel:
/// R is at offset +2, G at +1, B at +0; A at +3 is discarded.
///
/// # Safety
///
/// - `input_ptr` must point to at least 64 readable bytes.
/// - `output_ptr` must point to at least 48 writable bytes.
/// - `input_ptr` / `output_ptr` ranges must not alias.
/// - SSSE3 must be available in the caller's `target_feature` context.
#[inline(always)]
pub(super) unsafe fn bgra_to_rgb_16_pixels(input_ptr: *const u8, output_ptr: *mut u8) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());
    let in3 = _mm_loadu_si128(input_ptr.add(48).cast());

    let m00 = _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
    let m01 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0, 6);
    let out0 = _mm_or_si128(_mm_shuffle_epi8(in0, m00), _mm_shuffle_epi8(in1, m01));

    let m11 = _mm_setr_epi8(5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1);
    let m12 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9);
    let out1 = _mm_or_si128(_mm_shuffle_epi8(in1, m11), _mm_shuffle_epi8(in2, m12));

    let m22 = _mm_setr_epi8(
      8, 14, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m23 = _mm_setr_epi8(-1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12);
    let out2 = _mm_or_si128(_mm_shuffle_epi8(in2, m22), _mm_shuffle_epi8(in3, m23));

    _mm_storeu_si128(output_ptr.cast(), out0);
    _mm_storeu_si128(output_ptr.add(16).cast(), out1);
    _mm_storeu_si128(output_ptr.add(32).cast(), out2);
  }
}

// ---- RGB → HSV support --------------------------------------------------
//
// Matches the scalar `rgb_to_hsv_row` within ±1 LSB. Every op mirrors
// the scalar: f32 max/min preserves the same channel selection, and the
// branch cascade uses `_mm_blendv_ps` in the same
// `delta == 0 → v == r → v == g → v == b` priority as the scalar.
// For division we use `_mm_rcp_ps` followed by one Newton‑Raphson
// refinement step (`rcp * (2 - v * rcp)`) — ~3× faster than true
// `_mm_div_ps` at the cost of ±1 LSB in S/H. `#[inline(always)]`
// guarantees each helper inlines into its caller, so the
// SSSE3+SSE4.1 intrinsics execute in whatever `target_feature` context
// (sse4.1 / avx2 / avx512) the outer kernel declares.

/// Deinterleaves 48 bytes of packed RGB into three u8x16 channel
/// vectors (R, G, B). 9 shuffles + 6 ORs — mirror of the swap pattern.
///
/// # Safety
///
/// `input_ptr` must point to at least 48 readable bytes. Caller's
/// `target_feature` must include SSSE3 (via sse4.1 or higher).
#[inline(always)]
pub(super) unsafe fn deinterleave_rgb_16(input_ptr: *const u8) -> (__m128i, __m128i, __m128i) {
  unsafe {
    let in0 = _mm_loadu_si128(input_ptr.cast());
    let in1 = _mm_loadu_si128(input_ptr.add(16).cast());
    let in2 = _mm_loadu_si128(input_ptr.add(32).cast());

    // R bytes live at absolute positions 3k for k=0..15; in chunk 0
    // that's local [0,3,6,9,12,15] (6 values), chunk 1 [2,5,8,11,14]
    // (5 values), chunk 2 [1,4,7,10,13] (5 values).
    let mr0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mr1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let mr2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let r = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(in0, mr0), _mm_shuffle_epi8(in1, mr1)),
      _mm_shuffle_epi8(in2, mr2),
    );

    // G bytes at positions 3k+1: chunk 0 [1,4,7,10,13], chunk 1
    // [0,3,6,9,12,15], chunk 2 [2,5,8,11,14].
    let mg0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mg1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let mg2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let g = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(in0, mg0), _mm_shuffle_epi8(in1, mg1)),
      _mm_shuffle_epi8(in2, mg2),
    );

    // B bytes at positions 3k+2: chunk 0 [2,5,8,11,14], chunk 1
    // [1,4,7,10,13], chunk 2 [0,3,6,9,12,15].
    let mb0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mb1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let mb2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    let b = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(in0, mb0), _mm_shuffle_epi8(in1, mb1)),
      _mm_shuffle_epi8(in2, mb2),
    );

    (r, g, b)
  }
}

/// Widens a u8x16 to four f32x4 groups (lanes 0..3, 4..7, 8..11,
/// 12..15). Zero‑extends via `_mm_cvtepu8_epi32` (SSE4.1) then converts
/// to f32.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: __m128i) -> (__m128, __m128, __m128, __m128) {
  unsafe {
    let i0 = _mm_cvtepu8_epi32(v);
    let i1 = _mm_cvtepu8_epi32(_mm_srli_si128::<4>(v));
    let i2 = _mm_cvtepu8_epi32(_mm_srli_si128::<8>(v));
    let i3 = _mm_cvtepu8_epi32(_mm_srli_si128::<12>(v));
    (
      _mm_cvtepi32_ps(i0),
      _mm_cvtepi32_ps(i1),
      _mm_cvtepi32_ps(i2),
      _mm_cvtepi32_ps(i3),
    )
  }
}

/// Packs four f32x4 vectors (16 values in [0, 255]) to one u8x16.
/// Truncates f32 → i32 via `_mm_cvttps_epi32`, matches scalar `as u8`
/// (values are pre‑clamped so saturation on the narrowing steps is
/// a no‑op).
#[inline(always)]
fn f32x4_quad_to_u8x16(a: __m128, b: __m128, c: __m128, d: __m128) -> __m128i {
  unsafe {
    let ai = _mm_cvttps_epi32(a);
    let bi = _mm_cvttps_epi32(b);
    let ci = _mm_cvttps_epi32(c);
    let di = _mm_cvttps_epi32(d);
    let ab = _mm_packus_epi32(ai, bi); // i32x4 × 2 → u16x8
    let cd = _mm_packus_epi32(ci, di);
    _mm_packus_epi16(ab, cd) // u16x8 × 2 → u8x16
  }
}

/// Computes HSV for 4 pixels. Mirrors the scalar
/// `rgb_to_hsv_pixel` op‑for‑op. Returns `(h_quant, s_quant, v_quant)`
/// as f32x4 — already clamped to the scalar output ranges, still f32
/// awaiting the truncating cast in the caller.
#[inline(always)]
fn hsv_group(r: __m128, g: __m128, b: __m128) -> (__m128, __m128, __m128) {
  unsafe {
    let zero = _mm_setzero_ps();
    let half = _mm_set1_ps(0.5);
    let sixty = _mm_set1_ps(60.0);
    let one_twenty = _mm_set1_ps(120.0);
    let two_forty = _mm_set1_ps(240.0);
    let three_sixty = _mm_set1_ps(360.0);
    let one_seventy_nine = _mm_set1_ps(179.0);
    let two_fifty_five = _mm_set1_ps(255.0);

    let two = _mm_set1_ps(2.0);

    // V = max(r, g, b); min = min(r, g, b); delta = V - min.
    let v = _mm_max_ps(_mm_max_ps(r, g), b);
    let min_rgb = _mm_min_ps(_mm_min_ps(r, g), b);
    let delta = _mm_sub_ps(v, min_rgb);

    // Replace `_mm_div_ps` with 11‑bit reciprocal + one Newton‑Raphson
    // refinement step. On Skylake+/Zen4 `_mm_rcp_ps` is ~4 cycles vs
    // `_mm_div_ps` at ~13, and the refinement (`rcp * (2 - v * rcp)`)
    // adds ~7 cycles but brings precision to ~23 bits — more than
    // enough for u8 HSV output. Net ~20% throughput improvement on
    // x86 vs the f32 divide path. Output remains within ±1 LSB of the
    // scalar LUT reference.
    //
    // v = 0 / delta = 0 inputs would produce NaN through the Newton
    // step but are masked to 0 / 0 in the cascade below, so the NaNs
    // are always discarded before quantization.
    let v_rcp0 = _mm_rcp_ps(v);
    let v_rcp = _mm_mul_ps(v_rcp0, _mm_sub_ps(two, _mm_mul_ps(v, v_rcp0)));
    let delta_rcp0 = _mm_rcp_ps(delta);
    let delta_rcp = _mm_mul_ps(delta_rcp0, _mm_sub_ps(two, _mm_mul_ps(delta, delta_rcp0)));

    // S = if v == 0 { 0 } else { 255 * delta * rcp(v) }.
    let mask_v_zero = _mm_cmpeq_ps(v, zero);
    let s_nonzero = _mm_mul_ps(_mm_mul_ps(two_fifty_five, delta), v_rcp);
    let s = _mm_blendv_ps(s_nonzero, zero, mask_v_zero);

    // Hue branches.
    let mask_delta_zero = _mm_cmpeq_ps(delta, zero);
    let mask_v_is_r = _mm_cmpeq_ps(v, r);
    let mask_v_is_g = _mm_cmpeq_ps(v, g);

    // h_r = 60 * (g - b) * rcp(delta); wrap negatives by +360.
    let h_r_raw = _mm_mul_ps(_mm_mul_ps(sixty, _mm_sub_ps(g, b)), delta_rcp);
    let mask_neg = _mm_cmplt_ps(h_r_raw, zero);
    let h_r = _mm_blendv_ps(h_r_raw, _mm_add_ps(h_r_raw, three_sixty), mask_neg);

    // h_g = 60 * (b - r) * rcp(delta) + 120.
    let h_g = _mm_add_ps(
      _mm_mul_ps(_mm_mul_ps(sixty, _mm_sub_ps(b, r)), delta_rcp),
      one_twenty,
    );
    // h_b = 60 * (r - g) * rcp(delta) + 240.
    let h_b = _mm_add_ps(
      _mm_mul_ps(_mm_mul_ps(sixty, _mm_sub_ps(r, g)), delta_rcp),
      two_forty,
    );

    // Cascade priority: delta == 0 → 0; v == r → h_r; v == g → h_g;
    // else → h_b. Same as scalar's `else if` chain.
    let h_g_or_b = _mm_blendv_ps(h_b, h_g, mask_v_is_g);
    let h_nonzero = _mm_blendv_ps(h_g_or_b, h_r, mask_v_is_r);
    let hue = _mm_blendv_ps(h_nonzero, zero, mask_delta_zero);

    // Quantize to scalar output ranges.
    //   h = clamp(hue * 0.5 + 0.5, 0, 179)
    //   s = clamp(s + 0.5, 0, 255)
    //   v = clamp(v + 0.5, 0, 255)
    let h_quant = _mm_min_ps(
      _mm_max_ps(_mm_add_ps(_mm_mul_ps(hue, half), half), zero),
      one_seventy_nine,
    );
    let s_quant = _mm_min_ps(_mm_max_ps(_mm_add_ps(s, half), zero), two_fifty_five);
    let v_quant = _mm_min_ps(_mm_max_ps(_mm_add_ps(v, half), zero), two_fifty_five);

    (h_quant, s_quant, v_quant)
  }
}

/// Converts 16 RGB pixels to planar HSV (OpenCV 8‑bit encoding).
/// Reads 48 bytes from `input_ptr`, writes 16 bytes each to `h_ptr`,
/// `s_ptr`, `v_ptr`.
///
/// # Safety
///
/// - `input_ptr` must point to at least 48 readable bytes.
/// - Each of `h_ptr`, `s_ptr`, `v_ptr` must point to at least 16
///   writable bytes.
/// - No aliasing between input and output.
/// - Caller's `target_feature` must include SSE4.1 (or a superset:
///   avx2, avx512bw).
#[inline(always)]
pub(super) unsafe fn rgb_to_hsv_16_pixels(
  input_ptr: *const u8,
  h_ptr: *mut u8,
  s_ptr: *mut u8,
  v_ptr: *mut u8,
) {
  unsafe {
    let (r_u8, g_u8, b_u8) = deinterleave_rgb_16(input_ptr);

    // Widen each channel to 4 × f32x4 groups (16 pixels → 4 groups of
    // 4 lanes each).
    let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);
    let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
    let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);

    // HSV compute per group.
    let (h0, s0, v0) = hsv_group(r0, g0, b0);
    let (h1, s1, v1) = hsv_group(r1, g1, b1);
    let (h2, s2, v2) = hsv_group(r2, g2, b2);
    let (h3, s3, v3) = hsv_group(r3, g3, b3);

    // Pack each planar f32 quad back to u8x16 and store.
    _mm_storeu_si128(h_ptr.cast(), f32x4_quad_to_u8x16(h0, h1, h2, h3));
    _mm_storeu_si128(s_ptr.cast(), f32x4_quad_to_u8x16(s0, s1, s2, s3));
    _mm_storeu_si128(v_ptr.cast(), f32x4_quad_to_u8x16(v0, v1, v2, v3));
  }
}
