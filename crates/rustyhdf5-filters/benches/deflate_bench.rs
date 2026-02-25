//! Benchmark: deflate compression/decompression across backends.
//!
//! Both the active backend (zlib-ng or apple-compression) and the pure-Rust
//! miniz_oxide baseline are tested in each run for direct comparison.
//!
//! Run:
//!   cargo bench -p rustyhdf5-filters -- deflate

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_sine_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8).collect()
}

/// Generate 1M f64 values as raw bytes — matches the purehdf5-format bench pattern.
fn generate_f64_data(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (i as f64).to_le_bytes())
        .collect()
}

fn bench_compress_1mb(c: &mut Criterion) {
    let data = generate_sine_data(1_000_000);
    let backend = rustyhdf5_filters::deflate_backend();

    c.bench_function(&format!("deflate_compress_1MB ({backend})"), |b| {
        b.iter(|| rustyhdf5_filters::deflate_compress(black_box(&data), 6).unwrap())
    });

    c.bench_function("deflate_compress_1MB (miniz_oxide)", |b| {
        b.iter(|| rustyhdf5_filters::deflate_compress_miniz(black_box(&data), 6).unwrap())
    });
}

fn bench_decompress_1mb(c: &mut Criterion) {
    let data = generate_sine_data(1_000_000);
    let compressed = rustyhdf5_filters::deflate_compress_miniz(&data, 6).unwrap();
    let backend = rustyhdf5_filters::deflate_backend();

    c.bench_function(&format!("deflate_decompress_1MB ({backend})"), |b| {
        b.iter(|| {
            rustyhdf5_filters::deflate_decompress(black_box(&compressed), data.len()).unwrap()
        })
    });

    c.bench_function("deflate_decompress_1MB (miniz_oxide)", |b| {
        b.iter(|| rustyhdf5_filters::deflate_decompress_miniz(black_box(&compressed)).unwrap())
    });
}

fn bench_compress_f64(c: &mut Criterion) {
    let data = generate_f64_data(1_000_000);
    let backend = rustyhdf5_filters::deflate_backend();

    c.bench_function(&format!("deflate_compress_8MB_f64 ({backend})"), |b| {
        b.iter(|| rustyhdf5_filters::deflate_compress(black_box(&data), 6).unwrap())
    });

    c.bench_function("deflate_compress_8MB_f64 (miniz_oxide)", |b| {
        b.iter(|| rustyhdf5_filters::deflate_compress_miniz(black_box(&data), 6).unwrap())
    });
}

fn bench_decompress_f64(c: &mut Criterion) {
    let data = generate_f64_data(1_000_000);
    let compressed = rustyhdf5_filters::deflate_compress_miniz(&data, 6).unwrap();
    let backend = rustyhdf5_filters::deflate_backend();

    c.bench_function(&format!("deflate_decompress_8MB_f64 ({backend})"), |b| {
        b.iter(|| {
            rustyhdf5_filters::deflate_decompress(black_box(&compressed), data.len()).unwrap()
        })
    });

    c.bench_function("deflate_decompress_8MB_f64 (miniz_oxide)", |b| {
        b.iter(|| rustyhdf5_filters::deflate_decompress_miniz(black_box(&compressed)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_compress_1mb,
    bench_decompress_1mb,
    bench_compress_f64,
    bench_decompress_f64,
);
criterion_main!(benches);
