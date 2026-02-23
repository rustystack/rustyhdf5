//! AVX-512 SIMD implementations for x86_64.
//! Gated behind the "avx512" feature flag.
//! All functions require runtime detection via is_x86_feature_detected!("avx512f").

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 dot product for f32 slices.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx512f").
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    // Process 32 elements per iteration (2x16 unrolled)
    while i + 32 <= len {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        acc1 = _mm512_fmadd_ps(va1, vb1, acc1);

        i += 32;
    }

    if i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// AVX-512 cosine similarity — fused single pass.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx512f").
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;

    let mut dot_acc = _mm512_setzero_ps();
    let mut norm_a_acc = _mm512_setzero_ps();
    let mut norm_b_acc = _mm512_setzero_ps();

    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        dot_acc = _mm512_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm512_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm512_fmadd_ps(vb, vb, norm_b_acc);
        i += 16;
    }

    let mut dot = _mm512_reduce_add_ps(dot_acc);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_acc);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_acc);

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

/// AVX-512 L2 distance.
///
/// # Safety
/// Caller must verify is_x86_feature_detected!("avx512f").
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_ps();

    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum.sqrt()
}
