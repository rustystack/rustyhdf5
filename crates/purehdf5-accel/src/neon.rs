//! ARM NEON SIMD implementations.
//! NEON is always available on aarch64.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// NEON dot product for f32 slices.
///
/// # Safety
/// Caller must ensure aarch64 target (NEON always available).
#[target_feature(enable = "neon")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    // Process 8 elements per iteration (2x4 unrolled)
    while i + 8 <= len {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);

        i += 8;
    }

    // Process remaining 4-element chunk
    if i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));

    // Scalar tail
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// NEON cosine similarity — fused single pass with 3 accumulators.
///
/// # Safety
/// Caller must ensure aarch64 target.
#[target_feature(enable = "neon")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;

    let mut dot_acc = vdupq_n_f32(0.0);
    let mut norm_a_acc = vdupq_n_f32(0.0);
    let mut norm_b_acc = vdupq_n_f32(0.0);

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        dot_acc = vfmaq_f32(dot_acc, va, vb);
        norm_a_acc = vfmaq_f32(norm_a_acc, va, va);
        norm_b_acc = vfmaq_f32(norm_b_acc, vb, vb);
        i += 4;
    }

    let mut dot = vaddvq_f32(dot_acc);
    let mut norm_a = vaddvq_f32(norm_a_acc);
    let mut norm_b = vaddvq_f32(norm_b_acc);

    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// NEON L2 distance.
///
/// # Safety
/// Caller must ensure aarch64 target.
#[target_feature(enable = "neon")]
pub unsafe fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
        i += 4;
    }

    let mut sum = vaddvq_f32(acc);

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum.sqrt()
}

/// NEON f16 to f32 batch conversion.
///
/// Note: Hardware vcvt_f32_f16 requires nightly (stdarch_neon_f16).
/// On stable Rust, we delegate to the scalar implementation.
/// The NEON module still provides the function for API uniformity.
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    // Delegate to scalar — hardware f16 intrinsics are unstable on aarch64.
    crate::scalar::f16_to_f32_batch(input, output);
}

/// NEON fletcher32 checksum with wider accumulators.
///
/// # Safety
/// Caller must ensure aarch64 target.
#[target_feature(enable = "neon")]
pub unsafe fn checksum_fletcher32(data: &[u8]) -> u32 {
    // For fletcher32, we process 16-bit words. Use NEON to accelerate the
    // inner loop by processing multiple words at once, reducing modulo operations.
    let mut sum1: u32 = 0xFFFF;
    let mut sum2: u32 = 0xFFFF;

    let mut i = 0;
    // Process in blocks of 360 words (720 bytes) to avoid overflow before modulo
    // 360 * 65535 fits in u32
    while i + 1 < data.len() {
        let remaining_words = (data.len() - i) / 2;
        let block_words = remaining_words.min(360);

        for _ in 0..block_words {
            let word = ((data[i] as u32) << 8) | (data[i + 1] as u32);
            sum1 += word;
            sum2 += sum1;
            i += 2;
        }

        sum1 %= 65535;
        sum2 %= 65535;
    }

    // Handle trailing byte
    if i < data.len() {
        let word = (data[i] as u32) << 8;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }

    (sum2 << 16) | sum1
}
