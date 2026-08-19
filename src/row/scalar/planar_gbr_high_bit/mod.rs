//! Scalar reference kernels for high-bit-depth planar GBR sources
//! (Tier 10b — `AV_PIX_FMT_GBRP{9,10,12,14,16}LE/BE` /
//! `AV_PIX_FMT_GBRAP{10,12,14,16}LE/BE`).
//!
//! `gbr_*` kernels (3-plane, no α) are const-generic over
//! `BITS ∈ {9, 10, 12, 14, 16}` **and** `BE: bool` (endianness of the
//! source planes). `gbra_*` kernels (4-plane, with α) are const-generic
//! over `BITS ∈ {10, 12, 14, 16}` — FFmpeg has no `GBRAP9` variant;
//! only the 3-plane `GBRP9` exists at 9 bits.
//! No runtime branching on `BITS` — every `BITS - 8` shift is a
//! const-eval expression resolved at monomorphisation.  The `BE` branch is
//! also const-folded away at monomorphisation time.
//!
//! # Output variants
//!
//! | Suffix             | Element type | Alpha         |
//! |--------------------|-------------|---------------|
//! | `rgb_high_bit`     | `u8`        | none          |
//! | `rgb_u16_high_bit` | `u16`       | none          |
//! | `rgba_opaque_*`    | `u8`/`u16`  | opaque const  |
//! | `gbra_to_rgba_*`   | `u8`/`u16`  | source plane  |
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the
//! packed output convention is **R, G, B** (matching FFmpeg
//! `AV_PIX_FMT_RGB24`). Every kernel performs this reorder.
//!
//! # u8 downshift
//!
//! u8-output kernels apply `>> (BITS - 8)` per sample (plain truncation,
//! matching FFmpeg `swscale` behaviour). For `BITS == 16` this is `>> 8`;
//! for `BITS == 9` it is `>> 1`.
//!
//! # Opaque alpha constants
//!
//! - u8: `0xFF`
//! - u16: `(1u16 << BITS) - 1` (i.e., `511`, `1023`, `4095`, …)
//!
//! # Big-endian (`BE = true`) mode
//!
//! When `BE = true` each u16 sample is byte-swapped before masking and
//! arithmetic.  The swap is a compile-time branch: the `BE = false` path
//! compiles to a no-op and the call overhead is zero.

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B`
/// **bytes**, downshifting each sample by `BITS - 8`.
///
/// Output order is **R, G, B** per pixel (FFmpeg `RGB24` convention).
///
/// When `BE = true` each source element is byte-swapped before processing
/// (big-endian wire format → host-native arithmetic value).
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_out` has at least `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let dst = x * 3;
    rgb_out[dst] = (r_val >> shift) as u8;
    rgb_out[dst + 1] = (g_val >> shift) as u8;
    rgb_out[dst + 2] = (b_val >> shift) as u8;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B`
/// **`u16`** samples. Copies samples directly without shifting —
/// output values are in `[0, (1 << BITS) - 1]`.
///
/// When `BE = true` each source element is byte-swapped before processing.
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_u16_out` has at least `width * 3` samples.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let dst = x * 3;
    rgb_u16_out[dst] = r_val;
    rgb_u16_out[dst + 1] = g_val;
    rgb_u16_out[dst + 2] = b_val;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A`
/// **bytes** with a constant **opaque** alpha (`0xFF`). Used for
/// `Gbrp*` sources (no alpha plane) when `with_rgba` is requested.
///
/// Each sample is downshifted by `BITS - 8`.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let dst = x * 4;
    rgba_out[dst] = (r_val >> shift) as u8;
    rgba_out[dst + 1] = (g_val >> shift) as u8;
    rgba_out[dst + 2] = (b_val >> shift) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A`
/// **`u16`** samples with a constant **opaque** alpha
/// (`(1u16 << BITS) - 1`). Used for `Gbrp*` sources (no alpha plane)
/// when `with_rgba_u16` is requested. Copies samples directly.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let opaque: u16 = mask;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let dst = x * 4;
    rgba_u16_out[dst] = r_val;
    rgba_u16_out[dst + 1] = g_val;
    rgba_u16_out[dst + 2] = b_val;
    rgba_u16_out[dst + 3] = opaque;
  }
}

/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **bytes**. Alpha is sourced from the `a` plane (real per-pixel α).
/// Each sample (including α) is downshifted by `BITS - 8`.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra_to_rgba_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let a_raw = if BE {
      u16::from_be(a[x])
    } else {
      u16::from_le(a[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let a_val = a_raw & mask;
    let dst = x * 4;
    rgba_out[dst] = (r_val >> shift) as u8;
    rgba_out[dst + 1] = (g_val >> shift) as u8;
    rgba_out[dst + 2] = (b_val >> shift) as u8;
    rgba_out[dst + 3] = (a_val >> shift) as u8;
  }
}

/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **`u16`** samples. Alpha is sourced from the `a` plane at native
/// depth (no shift). Copies all four channels directly.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra_to_rgba_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let r_raw = if BE {
      u16::from_be(r[x])
    } else {
      u16::from_le(r[x])
    };
    let g_raw = if BE {
      u16::from_be(g[x])
    } else {
      u16::from_le(g[x])
    };
    let b_raw = if BE {
      u16::from_be(b[x])
    } else {
      u16::from_le(b[x])
    };
    let a_raw = if BE {
      u16::from_be(a[x])
    } else {
      u16::from_le(a[x])
    };
    let r_val = r_raw & mask;
    let g_val = g_raw & mask;
    let b_val = b_raw & mask;
    let a_val = a_raw & mask;
    let dst = x * 4;
    rgba_u16_out[dst] = r_val;
    rgba_u16_out[dst + 1] = g_val;
    rgba_u16_out[dst + 2] = b_val;
    rgba_u16_out[dst + 3] = a_val;
  }
}

/// Derives luma (Y') from three planar G/B/R `u16` rows directly at
/// native bit depth, avoiding the 256-level banding that would result
/// from staging through u8.
///
/// Uses i64 intermediates throughout so the BITS=16 case
/// (`max R = 65535`, product ≈ 1.54 B) does not overflow. The
/// performance cost relative to a separate i32 path for lower
/// bit-depths is negligible at the per-row level.
///
/// `full_range = true` → Y' ∈ `[0, (1 << BITS) - 1]` (full).
/// `full_range = false` → Y' ∈ `[16 << (BITS - 8), 235 << (BITS - 8)]`
/// (limited / studio swing). The limited-range formula mirrors
/// `rgb_to_luma_row` but scaled to native depth.
/// When `BE = true` each source element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_luma_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: crate::KernelMatrix,
  full_range: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (k_r, k_g, k_b) = super::luma_coefficients_q15(matrix);
  let k_r = k_r as i64;
  let k_g = k_g as i64;
  let k_b = k_b as i64;
  const RND: i64 = 1 << 14;
  let native_max: u16 = ((1u32 << BITS) - 1) as u16;
  let mask: u16 = native_max;

  if full_range {
    for x in 0..width {
      let r_raw = if BE {
        u16::from_be(r[x])
      } else {
        u16::from_le(r[x])
      };
      let g_raw = if BE {
        u16::from_be(g[x])
      } else {
        u16::from_le(g[x])
      };
      let b_raw = if BE {
        u16::from_be(b[x])
      } else {
        u16::from_le(b[x])
      };
      let rv = (r_raw & mask) as i64;
      let gv = (g_raw & mask) as i64;
      let bv = (b_raw & mask) as i64;
      let y = ((k_r * rv + k_g * gv + k_b * bv + RND) >> 15) as i32;
      luma_out[x] = y.clamp(0, native_max as i32) as u16;
    }
  } else {
    // Limited-range luma at native depth:
    //   Y_lim = Y_off + Y_full_clamped * range / native_max
    // where:
    //   Y_off       = 16  << (BITS - 8)        (native limited black)
    //   range       = 219 << (BITS - 8)        (native limited span)
    //   native_max  = (1 << BITS) - 1          (full-range upper bound)
    //
    // The naive 8-bit `LIMITED_SCALE_Q15 = round(219/255 x 32768)` ratio
    // is wrong here because it scales Y_full by `219/255 ≈ 0.85882`
    // when the correct native ratio is `range / native_max ≈ 0.85546`
    // at BITS=16. The ~0.4% overshoot makes the top ~250 input codes
    // collapse onto the y_max clamp, destroying highlight gradation.
    // The exact form below uses i64 throughout —
    // `range x native_max < 2^32` for BITS ≤ 16 — and a +native_max/2
    // bias for round-half-up semantics.
    let y_off = (16i64) << (BITS - 8);
    let range = (219i64) << (BITS - 8);
    let native_max_i64 = native_max as i64;
    let y_max = (235i64) << (BITS - 8);
    let y_min = y_off;
    for x in 0..width {
      let r_raw = if BE {
        u16::from_be(r[x])
      } else {
        u16::from_le(r[x])
      };
      let g_raw = if BE {
        u16::from_be(g[x])
      } else {
        u16::from_le(g[x])
      };
      let b_raw = if BE {
        u16::from_be(b[x])
      } else {
        u16::from_le(b[x])
      };
      let rv = (r_raw & mask) as i64;
      let gv = (g_raw & mask) as i64;
      let bv = (b_raw & mask) as i64;
      let y_full = (k_r * rv + k_g * gv + k_b * bv + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, native_max_i64);
      let y_lim = y_off + (y_full_clamped * range + native_max_i64 / 2) / native_max_i64;
      luma_out[x] = y_lim.clamp(y_min, y_max) as u16;
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
