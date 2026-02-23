# rustyhdf5-accel

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-accel.svg)](https://crates.io/crates/rustyhdf5-accel)
[![docs.rs](https://docs.rs/rustyhdf5-accel/badge.svg)](https://docs.rs/rustyhdf5-accel)

SIMD-accelerated operations for rustyhdf5.

## Features

- AVX2 and NEON SIMD acceleration
- AVX-512 support (`avx512` feature)
- Float16 conversion (`float16` feature)
- CRC32 checksum acceleration

## Usage

```rust
use rustyhdf5_accel::checksum::crc32_simd;

let crc = crc32_simd(&data);
```

## License

MIT
