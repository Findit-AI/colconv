//! Fused-downscale coverage for the 10-bit packed RGB family
//! (`X2Rgb10` / `X2Bgr10`): the packed 10-bit wire row is unpacked to a
//! source-width host u16 RGB row (channels `0..=1023`), binning runs at
//! native 10-bit depth, the native-depth `rgb_u16` output is the exact
//! area mean, `rgba_u16` is that same area mean expanded with opaque
//! alpha `1023`, and the u8 / `luma_u16` outputs derive from a single
//! `>> 2` narrowing — the same source-of-truth ordering the direct path
//! uses. `luma_u16` is the narrowed variant (no native 10-bit luma
//! kernel for this padding-source), byte-identical to the direct path.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{X2Bgr10, X2Rgb10, X2Rgb10Row, x2bgr10_to, x2rgb10_to},
};
use mediaframe::frame::{X2Bgr10Frame, X2Rgb10Frame};

const SRC: usize = 8;
const OUT: usize = 4;
const MATRIX: KernelMatrix = KernelMatrix::Bt709;

/// X2RGB10 LE word: `(MSB) 2X | 10R | 10G | 10B (LSB)`.
fn pack_x2rgb10(r10: u32, g10: u32, b10: u32) -> u32 {
  ((r10 & 0x3FF) << 20) | ((g10 & 0x3FF) << 10) | (b10 & 0x3FF)
}

/// X2BGR10 LE word: R at low 10, G mid, B high.
fn pack_x2bgr10(r10: u32, g10: u32, b10: u32) -> u32 {
  ((b10 & 0x3FF) << 20) | ((g10 & 0x3FF) << 10) | (r10 & 0x3FF)
}

/// Encode a sequence of packed words as LE wire bytes (`4 * n`).
fn x2_wire_bytes<F: Fn(usize) -> u32>(n: usize, word_at: F) -> Vec<u8> {
  let mut buf = vec![0u8; n * 4];
  for (i, chunk) in buf.chunks_mut(4).enumerate() {
    chunk.copy_from_slice(&word_at(i).to_le_bytes());
  }
  buf
}

/// Per-pixel native 10-bit `(r, g, b)` ramp; interior values so the
/// derived luma / HSV kernels see real math and the wide accumulator
/// carries bits a u8 path would drop. Masked to `0..=1023`.
fn rgb_px(i: usize) -> [u32; 3] {
  let r = (40 + (i as u32) * 17) & 0x3FF;
  let g = 1023u32.wrapping_sub((i as u32) * 21) & 0x3FF;
  let b = (100 + (i as u32 % 8) * 99) & 0x3FF;
  [r, g, b]
}

/// Exact 2x2 block mean with round-half-up over native u16 values — the
/// integer-area-mean contract for a 2:1 downscale at native 10-bit depth.
fn expected_block_mean(rgb: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

/// Source-width packed native-u16 RGB ramp (`SRC * SRC * 3` elements) —
/// the host-side oracle for the block mean.
fn host_rgb_ramp() -> Vec<u16> {
  let mut buf = vec![0u16; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    let [r, g, b] = rgb_px(i);
    px.copy_from_slice(&[r as u16, g as u16, b as u16]);
  }
  buf
}

#[test]
fn x2rgb10_downscale_rgb_u16_is_exact_native_area_mean() {
  let host = host_rgb_ramp();
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  let src = X2Rgb10Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    x2rgb10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&host, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn x2rgb10_derived_outputs_come_from_binned_rgb() {
  // Every attached output — native-depth u16 and narrowed u8 (incl. HSV
  // and luma_u16) — must be exactly what the direct full-res X2Rgb10 sink
  // produces over a frame that already holds the binned 10-bit RGB.
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  let src = X2Rgb10Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    x2rgb10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }

  // The resampled rgb_u16 IS the exact native block mean; re-pack it as a
  // binned X2RGB10 wire frame and drive the direct oracle from it.
  let binned_wire = x2_wire_bytes(OUT * OUT, |i| {
    pack_x2rgb10(
      rgb_u16[i * 3] as u32,
      rgb_u16[i * 3 + 1] as u32,
      rgb_u16[i * 3 + 2] as u32,
    )
  });
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned =
      X2Rgb10Frame::try_new(&binned_wire, OUT as u32, OUT as u32, (OUT * 4) as u32).unwrap();
    let mut sink = MixedSinker::<X2Rgb10>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    x2rgb10_to(&binned, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb (narrowed >> 2)");
  assert_eq!(rgba, ref_rgba, "rgba (narrowed, alpha forced 0xFF)");
  assert_eq!(luma, ref_luma, "luma (narrowed)");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16 (narrowed, zero-extended)");
  assert_eq!(h, ref_h, "hsv H");
  assert_eq!(s_, ref_s, "hsv S");
  assert_eq!(v_, ref_v, "hsv V");

  // rgba_u16 is the exact native block mean (== rgb_u16) expanded to RGBA
  // with opaque alpha (1 << 10) - 1 = 1023 — no source alpha (padding).
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  for (px, out) in ref_rgba_u16.chunks_exact_mut(4).enumerate() {
    out.copy_from_slice(&[
      rgb_u16[px * 3],
      rgb_u16[px * 3 + 1],
      rgb_u16[px * 3 + 2],
      1023,
    ]);
  }
  assert_eq!(
    rgba_u16, ref_rgba_u16,
    "rgba_u16 (native, alpha forced 1023)"
  );
}

#[test]
fn x2rgb10_identity_plan_matches_new_sink() {
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  let src = X2Rgb10Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<X2Rgb10>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    x2rgb10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    x2rgb10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
fn x2rgb10_resample_no_outputs_is_a_no_op() {
  // A resampling sink with no attached outputs is the documented legal
  // no-op: it walks every row and returns Ok without touching any caller
  // buffer (there is none to touch).
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  let src = X2Rgb10Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut sink =
    MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  x2rgb10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
}

#[test]
fn x2rgb10_reuses_stream_across_frames() {
  // begin_frame resets the u16 area stream + frozen output set, so frame
  // 2's row 0 is accepted (not rejected as out-of-sequence) and the
  // output reflects frame 2's input. Both frames share one output buffer;
  // only the input data changes.
  let wire1 = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  // Frame 2: complement each channel within 0..=1023.
  let host2: Vec<u16> = host_rgb_ramp().iter().map(|&p| 1023 - p).collect();
  let wire2 = x2_wire_bytes(SRC * SRC, |i| {
    pack_x2rgb10(
      host2[i * 3] as u32,
      host2[i * 3 + 1] as u32,
      host2[i * 3 + 2] as u32,
    )
  });

  let mut out = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut out)
        .unwrap();
    let f1 = X2Rgb10Frame::try_new(&wire1, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    let f2 = X2Rgb10Frame::try_new(&wire2, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    x2rgb10_to(&f1, true, sink.set_kernel_matrix(MATRIX)).unwrap();
    x2rgb10_to(&f2, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }

  let mut expected = vec![0u16; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        expected[(oy * OUT + ox) * 3 + c] = expected_block_mean(&host2, ox, oy, c);
      }
    }
  }
  assert_eq!(out, expected, "frame 2 output must area-downscale frame 2");
}

#[test]
fn x2rgb10_contracts_hold_on_the_fused_path() {
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2rgb10(r, g, b)
  });
  let row0 = &wire[..SRC * 4];

  // Out-of-order direct process: row 3 before row 0.
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    let err = sink
      .process(X2Rgb10Row::for_tests(row0, 3, MATRIX, true))
      .unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ));
  }

  // Mid-frame output change is rejected and leaves the new buffer untouched.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<X2Rgb10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(X2Rgb10Row::for_tests(row0, 0, MATRIX, true))
      .unwrap();
    sink.set_luma(&mut luma).unwrap();
    let err = sink
      .process(X2Rgb10Row::for_tests(
        &wire[SRC * 4..SRC * 8],
        1,
        MATRIX,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(luma.iter().all(|&l| l == 0));
}

#[test]
fn x2bgr10_downscale_rgb_u16_is_exact_native_area_mean() {
  // X2BGR10 stores B, G, R; the unpacked host RGB resolves channel order
  // back to (R, G, B), so the binned rgb_u16 equals the host RGB block
  // mean directly (no swap on the oracle).
  let host = host_rgb_ramp();
  let wire = x2_wire_bytes(SRC * SRC, |i| {
    let [r, g, b] = rgb_px(i);
    pack_x2bgr10(r, g, b)
  });
  let src = X2Bgr10Frame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<X2Bgr10, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    x2bgr10_to(&src, true, sink.set_kernel_matrix(MATRIX)).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&host, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}
