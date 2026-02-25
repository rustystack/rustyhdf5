//! Benchmark: all compression backends — deflate, LZ4, zstd.
//!
//! Run with specific features:
//!   cargo bench -p rustyhdf5-filters --features lz4,zstd --bench compression_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Generate test data simulating 1M f64 values with a sin() pattern.
fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8)
        .collect()
}

fn bench_deflate(c: &mut Criterion) {
    let data = generate_test_data(8_000_000); // 1M f64 = 8MB

    c.bench_function("deflate_compress_level6_8MB", |b| {
        b.iter(|| rustyhdf5_filters::deflate_compress(black_box(&data), 6).unwrap())
    });

    let compressed = rustyhdf5_filters::deflate_compress(&data, 6).unwrap();
    c.bench_function("deflate_decompress_8MB", |b| {
        b.iter(|| {
            rustyhdf5_filters::deflate_decompress(black_box(&compressed), data.len()).unwrap()
        })
    });
}

#[cfg(feature = "lz4")]
fn bench_lz4(c: &mut Criterion) {
    let data = generate_test_data(8_000_000);

    c.bench_function("lz4_compress_8MB", |b| {
        b.iter(|| rustyhdf5_filters::lz4_compress(black_box(&data)).unwrap())
    });

    let compressed = rustyhdf5_filters::lz4_compress(&data).unwrap();
    c.bench_function("lz4_decompress_8MB", |b| {
        b.iter(|| rustyhdf5_filters::lz4_decompress(black_box(&compressed)).unwrap())
    });
}

#[cfg(feature = "zstd")]
fn bench_zstd(c: &mut Criterion) {
    let data = generate_test_data(8_000_000);

    c.bench_function("zstd_compress_level1_8MB", |b| {
        b.iter(|| rustyhdf5_filters::zstd_compress(black_box(&data), 1).unwrap())
    });

    c.bench_function("zstd_compress_level3_8MB", |b| {
        b.iter(|| rustyhdf5_filters::zstd_compress(black_box(&data), 3).unwrap())
    });

    let compressed = rustyhdf5_filters::zstd_compress(&data, 3).unwrap();
    c.bench_function("zstd_decompress_8MB", |b| {
        b.iter(|| rustyhdf5_filters::zstd_decompress(black_box(&compressed)).unwrap())
    });
}

fn bench_parallel_deflate(c: &mut Criterion) {
    // 10 chunks of ~800KB each
    let chunks: Vec<Vec<u8>> = (0..10)
        .map(|i| generate_test_data(800_000 + i * 1000))
        .collect();

    c.bench_function("deflate_sequential_10chunks", |b| {
        b.iter(|| {
            let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
            rustyhdf5_filters::compress_chunks(black_box(&refs), |data| {
                rustyhdf5_filters::deflate_compress(data, 6)
            })
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_deflate,
    bench_parallel_deflate,
);

#[cfg(feature = "lz4")]
criterion_group!(lz4_benches, bench_lz4);

#[cfg(feature = "zstd")]
criterion_group!(zstd_benches, bench_zstd);

// Combine all benchmark groups
#[cfg(all(feature = "lz4", feature = "zstd"))]
criterion_main!(benches, lz4_benches, zstd_benches);

#[cfg(all(feature = "lz4", not(feature = "zstd")))]
criterion_main!(benches, lz4_benches);

#[cfg(all(not(feature = "lz4"), feature = "zstd"))]
criterion_main!(benches, zstd_benches);

#[cfg(not(any(feature = "lz4", feature = "zstd")))]
criterion_main!(benches);
