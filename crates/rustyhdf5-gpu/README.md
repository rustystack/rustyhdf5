# rustyhdf5-gpu

[![crates.io](https://img.shields.io/crates/v/rustyhdf5-gpu.svg)](https://crates.io/crates/rustyhdf5-gpu)
[![docs.rs](https://docs.rs/rustyhdf5-gpu/badge.svg)](https://docs.rs/rustyhdf5-gpu)

GPU-accelerated vector operations for rustyhdf5 using wgpu compute shaders.

## Features

- GPU-accelerated distance computations (L2, cosine)
- wgpu-based compute shaders for cross-platform GPU support
- Float16 support via `half` crate

## Usage

```rust
use rustyhdf5_gpu::GpuAccelerator;

let accel = GpuAccelerator::new().unwrap();
let distances = accel.l2_distances(&query, &vectors).unwrap();
```

## License

MIT
