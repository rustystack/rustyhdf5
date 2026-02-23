# rustyhdf5-derive

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-derive.svg)](https://crates.io/crates/rustyhdf5-derive)
[![docs.rs](https://docs.rs/rustyhdf5-derive/badge.svg)](https://docs.rs/rustyhdf5-derive)

Derive macros for rustyhdf5 HDF5 traits.

## Features

- `#[derive(HDF5Type)]` for automatic HDF5 datatype mapping
- Struct-to-compound-type derivation

## Usage

```rust
use rustyhdf5_derive::HDF5Type;

#[derive(HDF5Type)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
```

## License

MIT
