//! Dispatch layer for Monoblack and Monowhite kernels.
//!
//! Selects the highest available SIMD backend (avx512 → avx2 → sse4.1 →
//! neon → wasm-simd128) and falls back to scalar.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::row::{
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::mono1bit as scalar,
};

// ---- Monoblack dispatch ------------------------------------------------------

pub(crate) fn monoblack_to_rgb_or_rgba_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  let out_min = if ALPHA {
    rgba_row_bytes(width)
  } else {
    rgb_row_bytes(width)
  };
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= out_min,
    "out too short: need >= {out_min}, got {}",
    out.len()
  );
  if !use_simd {
    if ALPHA {
      return scalar::monoblack_to_rgba_row(data, out, width);
    } else {
      return scalar::monoblack_to_rgb_row(data, out, width);
    }
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          if ALPHA {
            arch::neon::mono1bit::monoblack_to_rgba_row(data, out, width);
          } else {
            arch::neon::mono1bit::monoblack_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx512::monoblack_to_rgba_row(data, out, width);
          } else {
            arch::x86_avx512::monoblack_to_rgb_row(data, out, width);
          }
        }
        return;
      }
      if avx2_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx2::monoblack_to_rgba_row(data, out, width);
          } else {
            arch::x86_avx2::monoblack_to_rgb_row(data, out, width);
          }
        }
        return;
      }
      if sse41_available() {
        unsafe {
          if ALPHA {
            arch::x86_sse41::monoblack_to_rgba_row(data, out, width);
          } else {
            arch::x86_sse41::monoblack_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          if ALPHA {
            arch::wasm_simd128::monoblack_to_rgba_row(data, out, width);
          } else {
            arch::wasm_simd128::monoblack_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    _ => {}
  }
  if ALPHA {
    scalar::monoblack_to_rgba_row(data, out, width);
  } else {
    scalar::monoblack_to_rgb_row(data, out, width);
  }
}

pub(crate) fn monoblack_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  let out_min = if ALPHA {
    rgba_row_elems(width)
  } else {
    rgb_row_elems(width)
  };
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= out_min,
    "out too short: need >= {out_min}, got {}",
    out.len()
  );
  if !use_simd {
    if ALPHA {
      return scalar::monoblack_to_rgba_u16_row(data, out, width);
    } else {
      return scalar::monoblack_to_rgb_u16_row(data, out, width);
    }
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          if ALPHA {
            arch::neon::mono1bit::monoblack_to_rgba_u16_row(data, out, width);
          } else {
            arch::neon::mono1bit::monoblack_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx512::monoblack_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_avx512::monoblack_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
      if avx2_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx2::monoblack_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_avx2::monoblack_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
      if sse41_available() {
        unsafe {
          if ALPHA {
            arch::x86_sse41::monoblack_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_sse41::monoblack_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          if ALPHA {
            arch::wasm_simd128::monoblack_to_rgba_u16_row(data, out, width);
          } else {
            arch::wasm_simd128::monoblack_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    _ => {}
  }
  if ALPHA {
    scalar::monoblack_to_rgba_u16_row(data, out, width);
  } else {
    scalar::monoblack_to_rgb_u16_row(data, out, width);
  }
}

pub(crate) fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize, use_simd: bool) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= width,
    "out too short: need >= {width}, got {}",
    out.len()
  );
  if !use_simd {
    return scalar::monoblack_to_luma_row(data, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monoblack_to_luma_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monoblack_to_luma_row(data, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monoblack_to_luma_row(data, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monoblack_to_luma_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monoblack_to_luma_row(data, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monoblack_to_luma_row(data, out, width);
}

pub(crate) fn monoblack_to_luma_u16_row(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= width,
    "out too short: need >= {width}, got {}",
    out.len()
  );
  if !use_simd {
    return scalar::monoblack_to_luma_u16_row(data, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monoblack_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monoblack_to_luma_u16_row(data, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monoblack_to_luma_u16_row(data, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monoblack_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monoblack_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monoblack_to_luma_u16_row(data, out, width);
}

pub(crate) fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    h.len() >= width,
    "h too short: need >= {width}, got {}",
    h.len()
  );
  assert!(
    s.len() >= width,
    "s too short: need >= {width}, got {}",
    s.len()
  );
  assert!(
    v.len() >= width,
    "v too short: need >= {width}, got {}",
    v.len()
  );
  if !use_simd {
    return scalar::monoblack_to_hsv_row(data, h, s, v, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monoblack_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monoblack_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monoblack_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monoblack_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monoblack_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monoblack_to_hsv_row(data, h, s, v, width);
}

// ---- Monowhite dispatch ------------------------------------------------------

pub(crate) fn monowhite_to_rgb_or_rgba_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  let out_min = if ALPHA {
    rgba_row_bytes(width)
  } else {
    rgb_row_bytes(width)
  };
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= out_min,
    "out too short: need >= {out_min}, got {}",
    out.len()
  );
  if !use_simd {
    if ALPHA {
      return scalar::monowhite_to_rgba_row(data, out, width);
    } else {
      return scalar::monowhite_to_rgb_row(data, out, width);
    }
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          if ALPHA {
            arch::neon::mono1bit::monowhite_to_rgba_row(data, out, width);
          } else {
            arch::neon::mono1bit::monowhite_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx512::monowhite_to_rgba_row(data, out, width);
          } else {
            arch::x86_avx512::monowhite_to_rgb_row(data, out, width);
          }
        }
        return;
      }
      if avx2_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx2::monowhite_to_rgba_row(data, out, width);
          } else {
            arch::x86_avx2::monowhite_to_rgb_row(data, out, width);
          }
        }
        return;
      }
      if sse41_available() {
        unsafe {
          if ALPHA {
            arch::x86_sse41::monowhite_to_rgba_row(data, out, width);
          } else {
            arch::x86_sse41::monowhite_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          if ALPHA {
            arch::wasm_simd128::monowhite_to_rgba_row(data, out, width);
          } else {
            arch::wasm_simd128::monowhite_to_rgb_row(data, out, width);
          }
        }
        return;
      }
    }
    _ => {}
  }
  if ALPHA {
    scalar::monowhite_to_rgba_row(data, out, width);
  } else {
    scalar::monowhite_to_rgb_row(data, out, width);
  }
}

pub(crate) fn monowhite_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  let out_min = if ALPHA {
    rgba_row_elems(width)
  } else {
    rgb_row_elems(width)
  };
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= out_min,
    "out too short: need >= {out_min}, got {}",
    out.len()
  );
  if !use_simd {
    if ALPHA {
      return scalar::monowhite_to_rgba_u16_row(data, out, width);
    } else {
      return scalar::monowhite_to_rgb_u16_row(data, out, width);
    }
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          if ALPHA {
            arch::neon::mono1bit::monowhite_to_rgba_u16_row(data, out, width);
          } else {
            arch::neon::mono1bit::monowhite_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx512::monowhite_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_avx512::monowhite_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
      if avx2_available() {
        unsafe {
          if ALPHA {
            arch::x86_avx2::monowhite_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_avx2::monowhite_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
      if sse41_available() {
        unsafe {
          if ALPHA {
            arch::x86_sse41::monowhite_to_rgba_u16_row(data, out, width);
          } else {
            arch::x86_sse41::monowhite_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          if ALPHA {
            arch::wasm_simd128::monowhite_to_rgba_u16_row(data, out, width);
          } else {
            arch::wasm_simd128::monowhite_to_rgb_u16_row(data, out, width);
          }
        }
        return;
      }
    }
    _ => {}
  }
  if ALPHA {
    scalar::monowhite_to_rgba_u16_row(data, out, width);
  } else {
    scalar::monowhite_to_rgb_u16_row(data, out, width);
  }
}

pub(crate) fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize, use_simd: bool) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= width,
    "out too short: need >= {width}, got {}",
    out.len()
  );
  if !use_simd {
    return scalar::monowhite_to_luma_row(data, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monowhite_to_luma_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monowhite_to_luma_row(data, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monowhite_to_luma_row(data, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monowhite_to_luma_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monowhite_to_luma_row(data, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monowhite_to_luma_row(data, out, width);
}

pub(crate) fn monowhite_to_luma_u16_row(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    out.len() >= width,
    "out too short: need >= {width}, got {}",
    out.len()
  );
  if !use_simd {
    return scalar::monowhite_to_luma_u16_row(data, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monowhite_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monowhite_to_luma_u16_row(data, out, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monowhite_to_luma_u16_row(data, out, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monowhite_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monowhite_to_luma_u16_row(data, out, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monowhite_to_luma_u16_row(data, out, width);
}

pub(crate) fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let packed_min = width.div_ceil(8);
  assert!(
    data.len() >= packed_min,
    "packed too short: need >= {packed_min}, got {}",
    data.len()
  );
  assert!(
    h.len() >= width,
    "h too short: need >= {width}, got {}",
    h.len()
  );
  assert!(
    s.len() >= width,
    "s too short: need >= {width}, got {}",
    s.len()
  );
  assert!(
    v.len() >= width,
    "v too short: need >= {width}, got {}",
    v.len()
  );
  if !use_simd {
    return scalar::monowhite_to_hsv_row(data, h, s, v, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        unsafe {
          arch::neon::mono1bit::monowhite_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    target_arch = "x86_64" => {
      if avx512_available() {
        unsafe {
          arch::x86_avx512::monowhite_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
      if avx2_available() {
        unsafe {
          arch::x86_avx2::monowhite_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
      if sse41_available() {
        unsafe {
          arch::x86_sse41::monowhite_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    target_arch = "wasm32" => {
      if simd128_available() {
        unsafe {
          arch::wasm_simd128::monowhite_to_hsv_row(data, h, s, v, width);
        }
        return;
      }
    }
    _ => {}
  }
  scalar::monowhite_to_hsv_row(data, h, s, v, width);
}
