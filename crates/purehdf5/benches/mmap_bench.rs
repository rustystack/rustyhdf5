//! Benchmarks comparing FileReader vs MmapReader and Eager vs Lazy file opening.

use criterion::{criterion_group, criterion_main, Criterion};
use purehdf5::{FileBuilder, LazyFile};
use purehdf5_io::{FileReader, MemoryReader, MmapReader};

use std::path::Path;

const N: usize = 1_000_000;

fn make_f64_data() -> Vec<f64> {
    (0..N).map(|i| i as f64).collect()
}

/// Write a contiguous f64 dataset to a temp file, return the path.
fn write_contiguous_file(path: &Path) {
    let data = make_f64_data();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[N as u64]);
    b.write(path).unwrap();
}

/// Write a chunked f64 dataset to a temp file, return the path.
fn write_chunked_file(path: &Path) {
    let data = make_f64_data();
    let mut b = FileBuilder::new();
    b.create_dataset("data")
        .with_f64_data(&data)
        .with_shape(&[N as u64])
        .with_chunks(&[10_000]);
    b.write(path).unwrap();
}

/// Write a file with 100 datasets.
fn write_100_dataset_file(path: &Path) {
    let mut b = FileBuilder::new();
    for i in 0..100 {
        b.create_dataset(&format!("ds_{i:04}"))
            .with_f64_data(&[i as f64; 10]);
    }
    b.write(path).unwrap();
}

// ===========================================================================
// Bench 1: FileReader vs MmapReader for 1M f64 contiguous
// ===========================================================================

fn bench_filereader_contiguous(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_filereader_contig.h5");
    write_contiguous_file(&path);

    c.bench_function("filereader_1M_f64_contiguous", |b| {
        b.iter(|| {
            let reader = FileReader::open(&path).unwrap();
            let file = purehdf5::File::from_bytes(reader.into_inner()).unwrap();
            file.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_mmapreader_contiguous(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_mmapreader_contig.h5");
    write_contiguous_file(&path);

    c.bench_function("mmapreader_1M_f64_contiguous", |b| {
        b.iter(|| {
            let reader = MmapReader::open(&path).unwrap();
            let lazy = LazyFile::open(reader).unwrap();
            lazy.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

// ===========================================================================
// Bench 2: FileReader vs MmapReader for 1M f64 chunked
// ===========================================================================

fn bench_filereader_chunked(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_filereader_chunked.h5");
    write_chunked_file(&path);

    c.bench_function("filereader_1M_f64_chunked", |b| {
        b.iter(|| {
            let reader = FileReader::open(&path).unwrap();
            let file = purehdf5::File::from_bytes(reader.into_inner()).unwrap();
            file.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_mmapreader_chunked(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_mmapreader_chunked.h5");
    write_chunked_file(&path);

    c.bench_function("mmapreader_1M_f64_chunked", |b| {
        b.iter(|| {
            let reader = MmapReader::open(&path).unwrap();
            let lazy = LazyFile::open(reader).unwrap();
            lazy.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

// ===========================================================================
// Bench 3: Eager vs Lazy file opening (100 datasets, read only 1)
// ===========================================================================

fn bench_eager_open_100ds(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_eager_100ds.h5");
    write_100_dataset_file(&path);

    c.bench_function("eager_open_100ds_read_1", |b| {
        b.iter(|| {
            let file = purehdf5::File::open(&path).unwrap();
            file.dataset("ds_0050").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_lazy_open_100ds(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_lazy_100ds.h5");
    write_100_dataset_file(&path);

    c.bench_function("lazy_open_100ds_read_1", |b| {
        b.iter(|| {
            let reader = MmapReader::open(&path).unwrap();
            let lazy = LazyFile::open(reader).unwrap();
            lazy.dataset("ds_0050").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

// ===========================================================================
// Bench 4: Sequential chunk read with vs without prefetch
// ===========================================================================

fn bench_sequential_no_prefetch(c: &mut Criterion) {
    let dir = std::env::temp_dir();
    let path = dir.join("bench_seq_no_prefetch.h5");
    write_chunked_file(&path);

    c.bench_function("sequential_chunked_no_prefetch", |b| {
        b.iter(|| {
            let reader = MemoryReader::new(std::fs::read(&path).unwrap());
            let lazy = LazyFile::open(reader).unwrap();
            lazy.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

fn bench_sequential_with_prefetch(c: &mut Criterion) {
    use purehdf5_io::prefetch::PrefetchReader;

    let dir = std::env::temp_dir();
    let path = dir.join("bench_seq_prefetch.h5");
    write_chunked_file(&path);

    // 10000 elements * 8 bytes = 80KB per chunk
    let chunk_bytes = 10_000 * 8;

    c.bench_function("sequential_chunked_with_prefetch", |b| {
        b.iter(|| {
            let reader = MemoryReader::new(std::fs::read(&path).unwrap());
            let prefetch = PrefetchReader::with_defaults(reader, chunk_bytes);
            let lazy = LazyFile::open(prefetch).unwrap();
            lazy.dataset("data").unwrap().read_f64().unwrap()
        })
    });

    std::fs::remove_file(&path).ok();
}

criterion_group!(
    benches,
    bench_filereader_contiguous,
    bench_mmapreader_contiguous,
    bench_filereader_chunked,
    bench_mmapreader_chunked,
    bench_eager_open_100ds,
    bench_lazy_open_100ds,
    bench_sequential_no_prefetch,
    bench_sequential_with_prefetch,
);
criterion_main!(benches);
