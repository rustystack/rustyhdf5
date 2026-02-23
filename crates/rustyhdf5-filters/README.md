# rustyhdf5-filters

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-filters.svg)](https://crates.io/crates/rustyhdf5-filters)
[![docs.rs](https://docs.rs/rustyhdf5-filters/badge.svg)](https://docs.rs/rustyhdf5-filters)

Filter and compression pipeline for rustyhdf5.

## Features

- DEFLATE compression/decompression
- Fast deflate via zlib-ng (`fast-deflate` feature)
- Apple Compression framework support (`apple-compression` feature)

## Usage

```rust
use rustyhdf5_filters::{deflate_decode, deflate_encode};

let compressed = deflate_encode(&data, 6).unwrap();
let decompressed = deflate_decode(&compressed).unwrap();
```

## License

MIT
