//! Integration tests for GPU-accelerated vector operations.
//!
//! These tests require a GPU. They are skipped gracefully if no GPU is available.

#[cfg(feature = "gpu-wgpu")]
mod tests {
    use rustyhdf5_gpu::{GpuAccelerator, GpuError};

    fn skip_if_no_gpu() -> Option<GpuAccelerator> {
        match GpuAccelerator::new() {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                eprintln!("SKIPPED: no GPU available");
                None
            }
        }
    }

    /// Helper: compute cosine similarity on CPU for validation.
    fn cpu_cosine(query: &[f32], vectors: &[f32], dim: usize) -> Vec<f32> {
        let n = vectors.len() / dim;
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        (0..n)
            .map(|i| {
                let base = i * dim;
                let dot: f32 = (0..dim).map(|d| query[d] * vectors[base + d]).sum();
                let v_norm: f32 = (0..dim)
                    .map(|d| vectors[base + d] * vectors[base + d])
                    .sum::<f32>()
                    .sqrt();
                let denom = q_norm * v_norm;
                if denom > 0.0 {
                    dot / denom
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Helper: compute L2 distance on CPU.
    fn cpu_l2(query: &[f32], vectors: &[f32], dim: usize) -> Vec<f32> {
        let n = vectors.len() / dim;
        (0..n)
            .map(|i| {
                let base = i * dim;
                (0..dim)
                    .map(|d| {
                        let diff = query[d] - vectors[base + d];
                        diff * diff
                    })
                    .sum()
            })
            .collect()
    }

    fn compute_norms(vectors: &[f32], dim: usize) -> Vec<f32> {
        let n = vectors.len() / dim;
        (0..n)
            .map(|i| {
                let base = i * dim;
                (0..dim)
                    .map(|d| vectors[base + d] * vectors[base + d])
                    .sum::<f32>()
                    .sqrt()
            })
            .collect()
    }

    // ── Test 1: GPU availability detection ──

    #[test]
    fn test_gpu_availability_detection() {
        // Should not panic regardless of GPU presence
        let available = GpuAccelerator::is_available();
        eprintln!("GPU available: {available}");
    }

    // ── Test 2: Device info reporting ──

    #[test]
    fn test_device_info_reporting() {
        let Some(gpu) = skip_if_no_gpu() else {
            return;
        };
        let info = gpu.device_info();
        assert!(!info.name.is_empty());
        assert!(!info.backend.is_empty());
        assert!(info.max_buffer_size > 0);
        eprintln!("Device: {info}");
    }

    // ── Test 3: Upload vectors basic ──

    #[test]
    fn test_upload_vectors() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 128;
        let n = 100;
        let vectors = vec![1.0f32; n * dim];
        gpu.upload_vectors(&vectors, dim).unwrap();
        assert_eq!(gpu.vector_count(), n);
        assert_eq!(gpu.dimension(), dim);
    }

    // ── Test 4: Upload dimension mismatch ──

    #[test]
    fn test_upload_dimension_mismatch() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        // 10 elements doesn't divide evenly into dim=3
        let result = gpu.upload_vectors(&[1.0; 10], 3);
        assert!(result.is_err());
    }

    // ── Test 5: Cosine search correctness ──

    #[test]
    fn test_cosine_search_matches_cpu() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 64;
        let n = 500;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(((i * dim + d) as f32).sin());
            }
        }
        let norms = compute_norms(&vectors, dim);
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).cos()).collect();

        let cpu_scores = cpu_cosine(&query, &vectors, dim);
        let mut cpu_ranked: Vec<(usize, f32)> =
            cpu_scores.iter().copied().enumerate().collect();
        cpu_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        cpu_ranked.truncate(10);

        gpu.upload_vectors(&vectors, dim).unwrap();
        gpu.upload_norms(&norms).unwrap();
        let gpu_results = gpu.cosine_search(&query, 10).unwrap();

        assert_eq!(gpu_results.len(), 10);
        // Top result should match
        assert_eq!(gpu_results[0].0, cpu_ranked[0].0);
        // Scores should be close
        for (gpu_r, cpu_r) in gpu_results.iter().zip(cpu_ranked.iter()) {
            assert!(
                (gpu_r.1 - cpu_r.1).abs() < 1e-3,
                "GPU score {} vs CPU score {} for index gpu={} cpu={}",
                gpu_r.1,
                cpu_r.1,
                gpu_r.0,
                cpu_r.0
            );
        }
    }

    // ── Test 6: Cosine search ranking matches CPU ──

    #[test]
    fn test_cosine_ranking_order() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 32;
        let n = 200;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        let norms = compute_norms(&vectors, dim);
        // Query aligned with dimension 0
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;

        gpu.upload_vectors(&vectors, dim).unwrap();
        gpu.upload_norms(&norms).unwrap();
        let results = gpu.cosine_search(&query, 5).unwrap();

        // All top results should have index % dim == 0
        assert_eq!(results[0].0 % dim, 0);
        // Scores should be descending
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ── Test 7: L2 search correctness ──

    #[test]
    fn test_l2_search_matches_cpu() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 64;
        let n = 500;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(((i * dim + d) as f32).sin());
            }
        }
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).cos()).collect();

        let cpu_dists = cpu_l2(&query, &vectors, dim);
        let mut cpu_ranked: Vec<(usize, f32)> =
            cpu_dists.iter().copied().enumerate().collect();
        cpu_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cpu_ranked.truncate(10);

        gpu.upload_vectors(&vectors, dim).unwrap();
        let gpu_results = gpu.l2_search(&query, 10).unwrap();

        assert_eq!(gpu_results.len(), 10);
        assert_eq!(gpu_results[0].0, cpu_ranked[0].0);
        for (gpu_r, cpu_r) in gpu_results.iter().zip(cpu_ranked.iter()) {
            assert!(
                (gpu_r.1 - cpu_r.1).abs() < 1e-2,
                "GPU dist {} vs CPU dist {}",
                gpu_r.1,
                cpu_r.1
            );
        }
    }

    // ── Test 8: L2 ranking order ──

    #[test]
    fn test_l2_ranking_order() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 16;
        let n = 100;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(i as f32 + d as f32 * 0.01);
            }
        }
        let query: Vec<f32> = (0..dim).map(|d| 50.0 + d as f32 * 0.01).collect();

        gpu.upload_vectors(&vectors, dim).unwrap();
        let results = gpu.l2_search(&query, 5).unwrap();

        // Distances should be ascending
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
        // Closest should be vector 50
        assert_eq!(results[0].0, 50);
    }

    // ── Test 9: Batch cosine search ──

    #[test]
    fn test_batch_cosine_search() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 32;
        let n = 100;
        let vectors = vec![1.0f32; n * dim];
        let norms = compute_norms(&vectors, dim);

        gpu.upload_vectors(&vectors, dim).unwrap();
        gpu.upload_norms(&norms).unwrap();

        let queries = vec![vec![1.0f32; dim]; 3];
        let results = gpu.batch_cosine_search(&queries, 5).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 5);
        }
    }

    // ── Test 10: Compute norms on GPU ──

    #[test]
    fn test_compute_norms() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 64;
        let n = 200;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(((i + d) as f32 * 0.5).sin());
            }
        }
        let cpu_norms = compute_norms(&vectors, dim);

        gpu.upload_vectors(&vectors, dim).unwrap();
        let gpu_norms = gpu.compute_norms().unwrap();

        assert_eq!(gpu_norms.len(), n);
        for (i, (g, c)) in gpu_norms.iter().zip(cpu_norms.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-3,
                "Norm mismatch at {i}: GPU={g}, CPU={c}"
            );
        }
    }

    // ── Test 11: Batch dot product ──

    #[test]
    fn test_batch_dot_product() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 16;
        let n = 50;
        let q = 3;

        // Identity-like vectors
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            vectors[i * dim + (i % dim)] = 1.0;
        }

        let mut queries = vec![0.0f32; q * dim];
        for qi in 0..q {
            queries[qi * dim + qi] = 1.0;
        }

        gpu.upload_vectors(&vectors, dim).unwrap();
        let scores = gpu.batch_dot_product(&queries, q).unwrap();
        assert_eq!(scores.len(), q * n);

        // Query 0 has 1.0 at dim 0, so dot with vector i is vectors[i*dim+0]
        for ni in 0..n {
            let expected = vectors[ni * dim]; // dim 0
            assert!(
                (scores[ni] - expected).abs() < 1e-5,
                "Mismatch at q=0, n={ni}"
            );
        }
    }

    // ── Test 12: f16 conversion ──

    #[test]
    fn test_f16_conversion() {
        let Some(gpu) = skip_if_no_gpu() else {
            return;
        };
        use half::f16;

        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 3.125, 100.0, -0.001, 65504.0];
        let f16_bits: Vec<u16> = values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

        let gpu_f32 = gpu.f16_to_f32_batch(&f16_bits).unwrap();
        assert_eq!(gpu_f32.len(), values.len());

        for (i, (&gpu_val, &orig)) in gpu_f32.iter().zip(values.iter()).enumerate() {
            let expected = f16::from_f32(orig).to_f32();
            assert!(
                (gpu_val - expected).abs() < 1e-3,
                "f16 conversion mismatch at {i}: GPU={gpu_val}, expected={expected}"
            );
        }
    }

    // ── Test 13: Error - no vectors uploaded ──

    #[test]
    fn test_error_no_vectors() {
        let Some(gpu) = skip_if_no_gpu() else {
            return;
        };
        let query = vec![1.0f32; 64];
        let result = gpu.cosine_search(&query, 5);
        assert!(matches!(result, Err(GpuError::NoVectors)));
    }

    // ── Test 14: Error - no norms uploaded ──

    #[test]
    fn test_error_no_norms() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 32;
        gpu.upload_vectors(&vec![1.0; 100 * dim], dim).unwrap();
        let result = gpu.cosine_search(&vec![1.0; dim], 5);
        assert!(matches!(result, Err(GpuError::NoNorms)));
    }

    // ── Test 15: Error - k exceeds n ──

    #[test]
    fn test_error_k_exceeds_n() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 16;
        let n = 5;
        gpu.upload_vectors(&vec![1.0; n * dim], dim).unwrap();
        let norms = vec![1.0f32; n];
        gpu.upload_norms(&norms).unwrap();
        let result = gpu.cosine_search(&vec![1.0; dim], 100);
        assert!(matches!(result, Err(GpuError::KExceedsN { .. })));
    }

    // ── Test 16: Error - dimension mismatch on search ──

    #[test]
    fn test_error_query_dim_mismatch() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        gpu.upload_vectors(&vec![1.0; 100 * 32], 32).unwrap();
        gpu.upload_norms(&vec![1.0; 100]).unwrap();
        let result = gpu.cosine_search(&vec![1.0; 64], 5);
        assert!(matches!(result, Err(GpuError::DimensionMismatch { .. })));
    }

    // ── Test 17: Graceful fallback when no GPU ──

    #[test]
    fn test_graceful_no_gpu_fallback() {
        // This test just demonstrates the pattern — it always passes
        match GpuAccelerator::new() {
            Ok(gpu) => {
                eprintln!("GPU found: {}", gpu.device_info());
            }
            Err(e) => {
                eprintln!("No GPU, fallback to CPU: {e}");
                // In real code, you'd use CPU SIMD here
            }
        }
    }

    // ── Test 18: Large dataset (1K vectors) ──

    #[test]
    fn test_larger_dataset() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 128;
        let n = 1000;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(((i * 7 + d * 13) as f32 * 0.01).sin());
            }
        }
        let norms = compute_norms(&vectors, dim);
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).cos()).collect();

        gpu.upload_vectors(&vectors, dim).unwrap();
        gpu.upload_norms(&norms).unwrap();
        let results = gpu.cosine_search(&query, 20).unwrap();
        assert_eq!(results.len(), 20);
        // Scores descending
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1 - 1e-6);
        }
    }

    // ── Test 19: f16 conversion odd length ──

    #[test]
    fn test_f16_conversion_odd_length() {
        let Some(gpu) = skip_if_no_gpu() else {
            return;
        };
        use half::f16;

        let values: Vec<f32> = vec![1.0, 2.0, 3.0]; // odd count
        let f16_bits: Vec<u16> = values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        let gpu_f32 = gpu.f16_to_f32_batch(&f16_bits).unwrap();
        assert_eq!(gpu_f32.len(), 3);
        for (i, (&gpu_val, &orig)) in gpu_f32.iter().zip(values.iter()).enumerate() {
            let expected = f16::from_f32(orig).to_f32();
            assert!(
                (gpu_val - expected).abs() < 1e-3,
                "Mismatch at {i}: {gpu_val} vs {expected}"
            );
        }
    }

    // ── Test 20: Upload then re-upload ──

    #[test]
    fn test_re_upload_vectors() {
        let Some(mut gpu) = skip_if_no_gpu() else {
            return;
        };
        let dim = 16;
        gpu.upload_vectors(&vec![1.0; 50 * dim], dim).unwrap();
        assert_eq!(gpu.vector_count(), 50);
        gpu.upload_vectors(&vec![2.0; 100 * dim], dim).unwrap();
        assert_eq!(gpu.vector_count(), 100);
    }
}

/// Test that compiles even without GPU feature.
#[cfg(not(feature = "gpu-wgpu"))]
mod no_gpu_tests {
    use rustyhdf5_gpu::GpuAccelerator;

    #[test]
    fn test_no_gpu_stub() {
        assert!(!GpuAccelerator::is_available());
        assert!(GpuAccelerator::new().is_err());
    }
}
