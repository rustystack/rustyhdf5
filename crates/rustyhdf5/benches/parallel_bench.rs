//! Benchmark: File::open vs MmapFile::open, sequential vs parallel chunk decompression.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustyhdf5::FileBuilder;

fn generate_f64_file(num_elements: usize) -> Vec<u8> {
    let values: Vec<f64> = (0..num_elements).map(|i| i as f64 * 0.001).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(&values);
    b.finish().unwrap()
}

fn bench_file_open(c: &mut Criterion) {
    let data = generate_f64_file(1_000_000);
    let dir = std::env::temp_dir();
    let path = dir.join("rustyhdf5_bench_parallel.h5");
    std::fs::write(&path, &data).unwrap();

    c.bench_function("File::open 1M f64", |b| {
        b.iter(|| {
            let file = rustyhdf5::File::open(black_box(&path)).unwrap();
            let ds = file.dataset("data").unwrap();
            ds.read_f64().unwrap()
        })
    });

    #[cfg(feature = "mmap")]
    c.bench_function("MmapFile::open 1M f64", |b| {
        b.iter(|| {
            let file = rustyhdf5::MmapFile::open(black_box(&path)).unwrap();
            let ds = file.dataset("data").unwrap();
            ds.read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

criterion_group!(benches, bench_file_open);
criterion_main!(benches);
