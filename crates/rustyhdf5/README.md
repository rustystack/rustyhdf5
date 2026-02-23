# rustyhdf5

[![crates.io](https://img.shields.io/crates/v/rustyhdf5.svg)](https://crates.io/crates/rustyhdf5)
[![docs.rs](https://docs.rs/rustyhdf5/badge.svg)](https://docs.rs/rustyhdf5)

Pure-Rust HDF5 reader/writer — no C dependencies.

## Features

- Read and write HDF5 files entirely in Rust
- Memory-mapped I/O for large files (`mmap` feature, enabled by default)
- Parallel chunk reads via Rayon (`parallel` feature)
- Lazy dataset access for minimal memory usage
- h5py-compatible file output

## Usage

```rust
use rustyhdf5::File;

let file = File::open("data.h5").unwrap();
let dataset = file.dataset("/group/data").unwrap();
let values: Vec<f64> = dataset.read_1d().unwrap();
```

## License

MIT
