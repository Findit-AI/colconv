use super::*;

mod high_bit_4_2_0;
mod high_bit_4_4_4_and_pn;
mod planar_8bit_and_nv;
mod yuva;

// ---- Shared test helpers (used across submodule tests) -------------

pub(super) fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
  let pairs = u.len();
  debug_assert_eq!(u.len(), v.len());
  let mut out = std::vec::Vec::with_capacity(pairs * 2);
  for i in 0..pairs {
    out.push(u[i]);
    out.push(v[i]);
  }
  out
}

pub(super) fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  (0..n)
    .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
    .collect()
}

pub(super) fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
    .collect()
}

pub(super) fn p16_plane_avx2(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

pub(super) fn high_bit_plane_avx2<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

pub(super) fn interleave_uv_avx2(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}
