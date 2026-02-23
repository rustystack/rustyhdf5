# rustyhdf5-format

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-format.svg)](https://crates.io/crates/rustyhdf5-format)
[![docs.rs](https://docs.rs/rustyhdf5-format/badge.svg)](https://docs.rs/rustyhdf5-format)

Pure-Rust HDF5 binary format parsing and writing — no C dependencies.

## Features

- Zero-copy superblock, object header, and B-tree parsing
- Chunked dataset read/write with filter pipelines
- `no_std` support (disable `std` feature)
- Optional parallel reads via Rayon
- SHA-256 provenance tracking

## Usage

```rust
use rustyhdf5_format::Superblock;

let data = std::fs::read("data.h5").unwrap();
let sb = Superblock::from_bytes(&data).unwrap();
println!("HDF5 version {}.{}", sb.version_major(), sb.version_minor());
```

## License

MIT
