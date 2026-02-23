# rustyhdf5-io

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-io.svg)](https://crates.io/crates/rustyhdf5-io)
[![docs.rs](https://docs.rs/rustyhdf5-io/badge.svg)](https://docs.rs/rustyhdf5-io)

I/O abstraction layer for rustyhdf5.

## Features

- Memory-mapped file access (`mmap` feature)
- Async I/O via Tokio (`async` feature)
- HSDS remote access (`hsds` feature)
- Prefetching and sweep optimizations

## Usage

```rust
use rustyhdf5_io::MmapReader;

let reader = MmapReader::open("data.h5").unwrap();
```

## License

MIT
