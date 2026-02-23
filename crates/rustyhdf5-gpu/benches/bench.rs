use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "gpu-wgpu")]
mod gpu_benches {
    use super::*;
    use rustyhdf5_gpu::GpuAccelerator;

    fn make_vectors(n: usize, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                v.push(((i * 7 + d * 13) as f32 * 0.01).sin());
            }
        }
        v
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

    fn cpu_cosine_search(
        query: &[f32],
        vectors: &[f32],
        norms: &[f32],
        dim: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let n = vectors.len() / dim;
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scores: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let base = i * dim;
                let dot: f32 = (0..dim).map(|d| query[d] * vectors[base + d]).sum();
                let denom = q_norm * norms[i];
                let s = if denom > 0.0 { dot / denom } else { 0.0 };
                (i, s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }

    pub fn bench_cosine(c: &mut Criterion) {
        let mut gpu = match GpuAccelerator::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU benchmarks: {e}");
                return;
            }
        };
        eprintln!("GPU: {}", gpu.device_info());

        let dim = 128;
        let k = 10;
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).cos()).collect();

        let mut group = c.benchmark_group("cosine_search");

        for &n in &[1_000, 10_000, 100_000] {
            let vectors = make_vectors(n, dim);
            let norms = compute_norms(&vectors, dim);

            // CPU benchmark
            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
                b.iter(|| cpu_cosine_search(&query, &vectors, &norms, dim, k));
            });

            // GPU upload benchmark
            group.bench_with_input(BenchmarkId::new("gpu_upload", n), &n, |b, _| {
                b.iter(|| {
                    gpu.upload_vectors(&vectors, dim).unwrap();
                    gpu.upload_norms(&norms).unwrap();
                });
            });

            // GPU search benchmark (pre-uploaded)
            gpu.upload_vectors(&vectors, dim).unwrap();
            gpu.upload_norms(&norms).unwrap();
            group.bench_with_input(BenchmarkId::new("gpu_search", n), &n, |b, _| {
                b.iter(|| gpu.cosine_search(&query, k).unwrap());
            });
        }

        group.finish();
    }

    pub fn bench_l2(c: &mut Criterion) {
        let mut gpu = match GpuAccelerator::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping L2 benchmarks: {e}");
                return;
            }
        };

        let dim = 128;
        let k = 10;
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).cos()).collect();

        let mut group = c.benchmark_group("l2_search");

        for &n in &[1_000, 10_000, 100_000] {
            let vectors = make_vectors(n, dim);
            gpu.upload_vectors(&vectors, dim).unwrap();

            group.bench_with_input(BenchmarkId::new("gpu_search", n), &n, |b, _| {
                b.iter(|| gpu.l2_search(&query, k).unwrap());
            });
        }

        group.finish();
    }

    pub fn bench_norms(c: &mut Criterion) {
        let mut gpu = match GpuAccelerator::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping norms benchmarks: {e}");
                return;
            }
        };

        let dim = 128;
        let mut group = c.benchmark_group("compute_norms");

        for &n in &[1_000, 10_000, 100_000] {
            let vectors = make_vectors(n, dim);
            gpu.upload_vectors(&vectors, dim).unwrap();

            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
                b.iter(|| gpu.compute_norms().unwrap());
            });
        }

        group.finish();
    }
}

#[cfg(feature = "gpu-wgpu")]
criterion_group!(
    benches,
    gpu_benches::bench_cosine,
    gpu_benches::bench_l2,
    gpu_benches::bench_norms,
);

#[cfg(not(feature = "gpu-wgpu"))]
fn no_gpu_bench(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled, skipping benchmarks");
}

#[cfg(not(feature = "gpu-wgpu"))]
criterion_group!(benches, no_gpu_bench);

criterion_main!(benches);
