//! Benchmark: miniz_oxide vs libdeflater decompression.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i as f64 * 0.01).sin() * 127.0 + 128.0) as u8).collect()
}

fn bench_compress(c: &mut Criterion) {
    let data = generate_test_data(1_000_000);

    c.bench_function("deflate_compress", |b| {
        b.iter(|| purehdf5_filters::deflate_compress(black_box(&data), 6).unwrap())
    });

    c.bench_function("deflate_compress_miniz", |b| {
        b.iter(|| purehdf5_filters::deflate_compress_miniz(black_box(&data), 6).unwrap())
    });
}

fn bench_decompress(c: &mut Criterion) {
    let data = generate_test_data(1_000_000);
    let compressed = purehdf5_filters::deflate_compress_miniz(&data, 6).unwrap();

    c.bench_function("deflate_decompress", |b| {
        b.iter(|| {
            purehdf5_filters::deflate_decompress(black_box(&compressed), data.len()).unwrap()
        })
    });

    c.bench_function("deflate_decompress_miniz", |b| {
        b.iter(|| purehdf5_filters::deflate_decompress_miniz(black_box(&compressed)).unwrap())
    });
}

criterion_group!(benches, bench_compress, bench_decompress);
criterion_main!(benches);
