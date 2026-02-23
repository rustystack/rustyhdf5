//! AVX2 SIMD implementations for x86_64.
//! All functions require runtime detection via is_x86_feature_detected!("avx2").

#![cfg(target_arch = "x86_64")]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal sum of __m256 (8 f32 lanes).
///
/// # Safety
/// Requires AVX.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_256(v: __m256) -> f32 {
    // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
    let hi128 = _mm256_extractf128_ps(v, 1); // [a4 a5 a6 a7]
    let lo128 = _mm256_castps256_ps128(v); // [a0 a1 a2 a3]
    let sum128 = _mm_add_ps(lo128, hi128); // [a0+a4, a1+a5, a2+a6, a3+a7]
    let shuf = _mm_movehdup_ps(sum128); // [a1+a5, a1+a5, a3+a7, a3+a7]
    let sums = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5, -, a2+a3+a6+a7, -]
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// AVX2 dot product for f32 slices.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx2") and "fma".
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 { unsafe {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    // Process 16 elements per iteration (2x8 unrolled)
    while i + 16 <= len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        i += 16;
    }

    // Process remaining 8-element chunk
    if i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    let mut sum = hsum_256(_mm256_add_ps(acc0, acc1));

    // Scalar tail
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}}

/// AVX2 cosine similarity — fused single pass.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx2") and "fma".
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { unsafe {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
        i += 8;
    }

    let mut dot = hsum_256(dot_acc);
    let mut norm_a = hsum_256(norm_a_acc);
    let mut norm_b = hsum_256(norm_b_acc);

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
}}

/// AVX2 L2 distance.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx2") and "fma".
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_distance(a: &[f32], b: &[f32]) -> f32 { unsafe {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
        i += 8;
    }

    let mut sum = hsum_256(acc);

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum.sqrt()
}}

/// AVX2 f16 to f32 batch conversion using F16C extension.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("f16c").
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) { unsafe {
    assert_eq!(input.len(), output.len());
    let len = input.len();
    let mut i = 0;

    while i + 8 <= len {
        let half8 = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);
        let f32x8 = _mm256_cvtph_ps(half8);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), f32x8);
        i += 8;
    }

    // Scalar tail
    while i < len {
        // Load single value into low lane
        let val = input[i];
        let half1 = _mm_set1_epi16(val as i16);
        let f32x8 = _mm256_cvtph_ps(half1);
        output[i] = _mm256_cvtss_f32(f32x8);
        i += 1;
    }
}}

/// AVX2-accelerated fletcher32 checksum.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx2").
#[target_feature(enable = "avx2")]
pub unsafe fn checksum_fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0xFFFF;
    let mut sum2: u32 = 0xFFFF;

    let mut i = 0;
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

    if i < data.len() {
        let word = (data[i] as u32) << 8;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }

    (sum2 << 16) | sum1
}
