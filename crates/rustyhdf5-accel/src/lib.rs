//! SIMD-accelerated operations for rustyhdf5.
//!
//! This crate provides runtime-dispatched SIMD acceleration for common
//! vector operations used in HDF5 processing: dot products, cosine similarity,
//! L2 distance, f16 conversion, and checksums.
//!
//! All public functions automatically select the best available SIMD backend
//! at runtime. Every operation has a portable scalar fallback.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod avx512;

pub mod checksum;
pub mod convert;

// ---------------------------------------------------------------------------
// Cache-line size detection (TVL — Tensor Virtualization Layout)
// ---------------------------------------------------------------------------

/// Cache line size in bytes for the target architecture.
///
/// ARM64 (Apple M-series, Cortex-A76+) uses 128-byte cache lines.
/// x86_64 uses 64-byte cache lines. Other architectures default to 64.
#[cfg(target_arch = "aarch64")]
pub const CACHE_LINE_SIZE: usize = 128;

#[cfg(target_arch = "x86_64")]
pub const CACHE_LINE_SIZE: usize = 64;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const CACHE_LINE_SIZE: usize = 64;

/// Round `size` up to the next multiple of [`CACHE_LINE_SIZE`].
#[inline]
pub fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// Available SIMD backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// ARM NEON (always available on aarch64)
    Neon,
    /// x86_64 AVX2 + FMA
    Avx2,
    /// x86_64 AVX-512F
    Avx512,
    /// x86_64 SSE4.1
    Sse4,
    /// WebAssembly SIMD128
    WasmSimd128,
    /// Portable scalar fallback
    Scalar,
}

/// Detect the best available SIMD backend at runtime.
pub fn detect_backend() -> Backend {
    #[cfg(target_arch = "aarch64")]
    {
        return Backend::Neon; // Always available on aarch64
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Backend::Avx512;
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Backend::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return Backend::Sse4;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        return Backend::WasmSimd128;
    }

    #[allow(unreachable_code)]
    Backend::Scalar
}

// ---------------------------------------------------------------------------
// Public API — auto-dispatched
// ---------------------------------------------------------------------------

/// Compute the dot product of two f32 slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    match detect_backend() {
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::dot_product(a, b) },

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Backend::Avx512 => unsafe { avx512::dot_product(a, b) },

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { avx2::dot_product(a, b) },

        _ => scalar::dot_product(a, b),
    }
}

/// Compute the L2 norm (magnitude) of a vector.
pub fn vector_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Compute cosine similarity between two vectors (fused single-pass).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    match detect_backend() {
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::cosine_similarity(a, b) },

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Backend::Avx512 => unsafe { avx512::cosine_similarity(a, b) },

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { avx2::cosine_similarity(a, b) },

        _ => scalar::cosine_similarity(a, b),
    }
}

/// Compute cosine similarity between a query and multiple vectors.
///
/// Results are stored as `(index, similarity)` pairs.
pub fn batch_cosine(query: &[f32], vectors: &[&[f32]], results: &mut [(usize, f32)]) {
    assert!(results.len() >= vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        results[i] = (i, cosine_similarity(query, v));
    }
}

/// Compute cosine similarity with pre-normalized query vector.
///
/// `query_normed` must already be unit-length. `norms` contains the L2 norms
/// of each vector in `vectors`.
pub fn batch_cosine_prenorm(
    query_normed: &[f32],
    vectors: &[&[f32]],
    norms: &[f32],
    results: &mut [(usize, f32)],
) {
    assert!(results.len() >= vectors.len());
    assert!(norms.len() >= vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        let dot = dot_product(query_normed, v);
        let sim = if norms[i] == 0.0 { 0.0 } else { dot / norms[i] };
        results[i] = (i, sim);
    }
}

/// Compute L2 (Euclidean) distance between two vectors.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    match detect_backend() {
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::l2_distance(a, b) },

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Backend::Avx512 => unsafe { avx512::l2_distance(a, b) },

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { avx2::l2_distance(a, b) },

        _ => scalar::l2_distance(a, b),
    }
}

/// Compute L2 norms for a batch of vectors.
pub fn batch_norms(vectors: &[&[f32]], norms: &mut [f32]) {
    assert!(norms.len() >= vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        norms[i] = vector_norm(v);
    }
}

/// Convert a batch of f16 values (as raw u16 bits) to f32.
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    convert::f16_to_f32_batch(input, output);
}

/// Compute Fletcher-32 checksum.
pub fn checksum_fletcher32(data: &[u8]) -> u32 {
    checksum::checksum_fletcher32(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // -----------------------------------------------------------------------
    // Backend detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_backend_returns_valid() {
        let backend = detect_backend();
        match backend {
            Backend::Neon
            | Backend::Avx2
            | Backend::Avx512
            | Backend::Sse4
            | Backend::WasmSimd128
            | Backend::Scalar => {}
        }
    }

    #[test]
    fn test_detect_backend_consistent() {
        let b1 = detect_backend();
        let b2 = detect_backend();
        assert_eq!(b1, b2);
    }

    // -----------------------------------------------------------------------
    // Dot product
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_product_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot_product(&a, &b);
        assert!(approx_eq(result, 70.0, EPSILON), "got {result}");
    }

    #[test]
    fn test_dot_product_zero_vectors() {
        let a = [0.0f32; 16];
        let b = [1.0f32; 16];
        assert!(approx_eq(dot_product(&a, &b), 0.0, EPSILON));
    }

    #[test]
    fn test_dot_product_unit_vectors() {
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        a[0] = 1.0;
        b[0] = 1.0;
        assert!(approx_eq(dot_product(&a, &b), 1.0, EPSILON));
    }

    #[test]
    fn test_dot_product_large_random() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();
        let scalar_result = scalar::dot_product(&a, &b);
        let simd_result = dot_product(&a, &b);
        assert!(
            approx_eq(scalar_result, simd_result, 0.1),
            "scalar={scalar_result} simd={simd_result}"
        );
    }

    #[test]
    fn test_dot_product_negative_values() {
        let a = [-1.0, -2.0, -3.0];
        let b = [1.0, 2.0, 3.0];
        assert!(approx_eq(dot_product(&a, &b), -14.0, EPSILON));
    }

    #[test]
    fn test_dot_product_single_element() {
        assert!(approx_eq(dot_product(&[3.0], &[4.0]), 12.0, EPSILON));
    }

    #[test]
    fn test_dot_product_empty() {
        assert!(approx_eq(dot_product(&[], &[]), 0.0, EPSILON));
    }

    #[test]
    fn test_dot_product_scalar_vs_dispatch() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();
        let s = scalar::dot_product(&a, &b);
        let d = dot_product(&a, &b);
        assert!(
            approx_eq(s, d, 0.01),
            "scalar={s} dispatched={d}"
        );
    }

    // -----------------------------------------------------------------------
    // Vector norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_norm_unit() {
        let v = [1.0, 0.0, 0.0];
        assert!(approx_eq(vector_norm(&v), 1.0, EPSILON));
    }

    #[test]
    fn test_vector_norm_345() {
        let v = [3.0, 4.0];
        assert!(approx_eq(vector_norm(&v), 5.0, EPSILON));
    }

    #[test]
    fn test_vector_norm_zero() {
        let v = [0.0f32; 10];
        assert!(approx_eq(vector_norm(&v), 0.0, EPSILON));
    }

    // -----------------------------------------------------------------------
    // Cosine similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_identical_is_one() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(cosine_similarity(&v, &v), 1.0, EPSILON));
    }

    #[test]
    fn test_cosine_opposite_is_neg_one() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        assert!(approx_eq(cosine_similarity(&a, &b), -1.0, EPSILON));
    }

    #[test]
    fn test_cosine_orthogonal_is_zero() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0, EPSILON));
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0f32; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0, EPSILON));
    }

    #[test]
    fn test_cosine_scalar_vs_dispatch() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.7).cos()).collect();
        let s = scalar::cosine_similarity(&a, &b);
        let d = cosine_similarity(&a, &b);
        assert!(
            approx_eq(s, d, 1e-4),
            "scalar={s} dispatched={d}"
        );
    }

    // -----------------------------------------------------------------------
    // Batch cosine
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_cosine_ranking_order() {
        let query = [1.0, 0.0, 0.0];
        let v0: Vec<f32> = vec![0.0, 1.0, 0.0]; // orthogonal = 0
        let v1: Vec<f32> = vec![1.0, 0.0, 0.0]; // identical = 1
        let v2: Vec<f32> = vec![0.5, 0.5, 0.0]; // in between
        let vectors: Vec<&[f32]> = vec![&v0, &v1, &v2];
        let mut results = vec![(0usize, 0.0f32); 3];
        batch_cosine(&query, &vectors, &mut results);

        // v1 should have highest similarity
        assert!(results[1].1 > results[2].1);
        assert!(results[2].1 > results[0].1);
    }

    #[test]
    fn test_batch_cosine_scalar_vs_dispatch() {
        let query: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let v0: Vec<f32> = (0..32).map(|i| (i as f32).cos()).collect();
        let v1: Vec<f32> = (0..32).map(|i| (i as f32 * 2.0).sin()).collect();
        let vectors: Vec<&[f32]> = vec![&v0, &v1];

        let mut scalar_results = vec![(0usize, 0.0f32); 2];
        scalar::batch_cosine(&query, &vectors, &mut scalar_results);

        let mut simd_results = vec![(0usize, 0.0f32); 2];
        batch_cosine(&query, &vectors, &mut simd_results);

        for i in 0..2 {
            assert!(
                approx_eq(scalar_results[i].1, simd_results[i].1, 1e-4),
                "mismatch at {i}: scalar={} simd={}",
                scalar_results[i].1,
                simd_results[i].1
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batch cosine prenorm
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_cosine_prenorm() {
        let query = [1.0, 0.0, 0.0]; // already unit-length
        let v0: Vec<f32> = vec![3.0, 4.0, 0.0];
        let v1: Vec<f32> = vec![0.0, 0.0, 5.0];
        let vectors: Vec<&[f32]> = vec![&v0, &v1];
        let norms = [5.0, 5.0];
        let mut results = vec![(0usize, 0.0f32); 2];
        batch_cosine_prenorm(&query, &vectors, &norms, &mut results);
        // dot(query, v0) = 3.0, sim = 3.0/5.0 = 0.6
        assert!(approx_eq(results[0].1, 0.6, EPSILON));
        // dot(query, v1) = 0.0, sim = 0.0
        assert!(approx_eq(results[1].1, 0.0, EPSILON));
    }

    // -----------------------------------------------------------------------
    // L2 distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_distance_same_is_zero() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(l2_distance(&v, &v), 0.0, EPSILON));
    }

    #[test]
    fn test_l2_distance_known_triangle() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!(approx_eq(l2_distance(&a, &b), 5.0, EPSILON));
    }

    #[test]
    fn test_l2_distance_unit_axes() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(approx_eq(l2_distance(&a, &b), 2.0f32.sqrt(), EPSILON));
    }

    #[test]
    fn test_l2_distance_scalar_vs_dispatch() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();
        let s = scalar::l2_distance(&a, &b);
        let d = l2_distance(&a, &b);
        assert!(
            approx_eq(s, d, 0.01),
            "scalar={s} dispatched={d}"
        );
    }

    // -----------------------------------------------------------------------
    // Batch norms
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_norms() {
        let v0: Vec<f32> = vec![3.0, 4.0];
        let v1: Vec<f32> = vec![0.0, 0.0];
        let v2: Vec<f32> = vec![1.0, 0.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&v0, &v1, &v2];
        let mut norms = vec![0.0f32; 3];
        batch_norms(&vectors, &mut norms);
        assert!(approx_eq(norms[0], 5.0, EPSILON));
        assert!(approx_eq(norms[1], 0.0, EPSILON));
        assert!(approx_eq(norms[2], 1.0, EPSILON));
    }

    // -----------------------------------------------------------------------
    // f16 conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_f16_to_f32_known_values() {
        // f16 representation of 1.0 = 0x3C00
        let input = [0x3C00u16, 0x4000, 0x0000]; // 1.0, 2.0, 0.0
        let mut output = [0.0f32; 3];
        f16_to_f32_batch(&input, &mut output);
        assert!(approx_eq(output[0], 1.0, EPSILON), "got {}", output[0]);
        assert!(approx_eq(output[1], 2.0, EPSILON), "got {}", output[1]);
        assert!(approx_eq(output[2], 0.0, EPSILON), "got {}", output[2]);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        // f16 -1.0 = 0xBC00
        let input = [0xBC00u16];
        let mut output = [0.0f32; 1];
        f16_to_f32_batch(&input, &mut output);
        assert!(approx_eq(output[0], -1.0, EPSILON), "got {}", output[0]);
    }

    #[test]
    fn test_f16_to_f32_batch_larger() {
        // Test with a larger batch to exercise SIMD paths
        let input: Vec<u16> = (0..32).map(|_| 0x3C00u16).collect(); // all 1.0
        let mut output = vec![0.0f32; 32];
        f16_to_f32_batch(&input, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 1.0, EPSILON), "mismatch at {i}: {v}");
        }
    }

    #[test]
    fn test_f16_to_f32_round_trip_accuracy() {
        // Test several known f16 bit patterns
        let cases: Vec<(u16, f32)> = vec![
            (0x3C00, 1.0),
            (0x4000, 2.0),
            (0x3800, 0.5),
            (0x4200, 3.0),
            (0x4400, 4.0),
            (0x0000, 0.0),
            (0x8000, -0.0),
        ];
        let input: Vec<u16> = cases.iter().map(|(bits, _)| *bits).collect();
        let mut output = vec![0.0f32; cases.len()];
        f16_to_f32_batch(&input, &mut output);
        for (i, (_, expected)) in cases.iter().enumerate() {
            assert!(
                approx_eq(output[i], *expected, EPSILON),
                "f16 0x{:04X}: expected {expected}, got {}",
                input[i],
                output[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Fletcher-32 checksum
    // -----------------------------------------------------------------------

    #[test]
    fn test_fletcher32_empty() {
        let result = checksum_fletcher32(&[]);
        // Both sums remain 0xFFFF
        assert_eq!(result, 0xFFFF_FFFF);
    }

    #[test]
    fn test_fletcher32_known() {
        let data = [0x00u8, 0x01, 0x00, 0x02];
        let result = checksum_fletcher32(&data);
        let scalar = scalar::checksum_fletcher32(&data);
        assert_eq!(result, scalar);
    }

    #[test]
    fn test_fletcher32_scalar_vs_dispatch() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let s = scalar::checksum_fletcher32(&data);
        let d = checksum_fletcher32(&data);
        assert_eq!(s, d);
    }

    // -----------------------------------------------------------------------
    // Performance sanity check
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_product_384_dim_perf() {
        use std::time::Instant;
        let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();

        // Warm up
        for _ in 0..100 {
            let _ = dot_product(&a, &b);
        }

        let start = Instant::now();
        let iterations = 10_000;
        let mut sum = 0.0f32;
        for _ in 0..iterations {
            sum += dot_product(&a, &b);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;
        // Prevent optimization
        assert!(sum.abs() >= 0.0);

        // In release mode, 384-dim dot product should be < 1µs.
        // In debug mode, allow up to 20µs (no optimizations).
        let limit_ns = if cfg!(debug_assertions) { 20_000 } else { 1_000 };
        assert!(
            per_call.as_nanos() < limit_ns,
            "dot product too slow: {per_call:?} per call (limit {limit_ns}ns)"
        );
    }

    // -----------------------------------------------------------------------
    // Cache-line alignment (TVL)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_line_size_is_power_of_two() {
        assert!(CACHE_LINE_SIZE.is_power_of_two());
    }

    #[test]
    fn test_cache_line_size_platform() {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(CACHE_LINE_SIZE, 128);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(CACHE_LINE_SIZE, 64);
    }

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE + 1), CACHE_LINE_SIZE * 2);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE * 3), CACHE_LINE_SIZE * 3);
    }

    #[test]
    fn test_align_to_cache_line_64_and_128() {
        // Both 64 and 128 alignment scenarios
        let val = align_to_cache_line(100);
        assert_eq!(val % CACHE_LINE_SIZE, 0);
        assert!(val >= 100);
        assert!(val < 100 + CACHE_LINE_SIZE);
    }

    // -----------------------------------------------------------------------
    // Edge cases / additional coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_product_non_aligned_length() {
        // Test with lengths that don't align to SIMD widths (not multiple of 4, 8, 16)
        for len in [1, 3, 5, 7, 9, 13, 17, 31, 33] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5).collect();
            let s = scalar::dot_product(&a, &b);
            let d = dot_product(&a, &b);
            assert!(
                approx_eq(s, d, 0.01),
                "len={len}: scalar={s} dispatched={d}"
            );
        }
    }

    #[test]
    fn test_cosine_non_aligned_length() {
        for len in [1, 3, 5, 7, 9, 13, 17, 31, 33] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 2.0).collect();
            let s = scalar::cosine_similarity(&a, &b);
            let d = cosine_similarity(&a, &b);
            assert!(
                approx_eq(s, d, 1e-4),
                "len={len}: scalar={s} dispatched={d}"
            );
        }
    }

    #[test]
    fn test_l2_distance_non_aligned_length() {
        for len in [1, 3, 5, 7, 9, 13, 17, 31, 33] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
            let s = scalar::l2_distance(&a, &b);
            let d = l2_distance(&a, &b);
            assert!(
                approx_eq(s, d, 0.01),
                "len={len}: scalar={s} dispatched={d}"
            );
        }
    }
}
