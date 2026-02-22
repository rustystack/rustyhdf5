# rustyHDF5

<!-- badges (coming soon — not yet published to crates.io) -->
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![CI](https://img.shields.io/badge/CI-coming%20soon-lightgrey)
![crates.io](https://img.shields.io/badge/crates.io-coming%20soon-lightgrey)
![docs.rs](https://img.shields.io/badge/docs.rs-coming%20soon-lightgrey)

A pure-Rust HDF5 reader and writer with zero C dependencies. Read and write HDF5 files from any platform Rust supports, including `no_std` environments.

## Features

- **Read and write** HDF5 files — contiguous, chunked, and compressed datasets
- **Compression** — deflate (gzip), shuffle, and fletcher32 filters
- **Resizable datasets** — `maxshape` support with extensible array indexes
- **Group hierarchy** — v1 (symbol table) and v2 (fractal heap) group navigation
- **Attributes** — scalar, array, and string attributes on datasets and groups
- **All numeric types** — f32, f64, i32, i64, u8, u64, plus fixed and variable-length strings
- **Compound/Enum/Array types** — struct-like, enumeration, and fixed-size array datasets
- **h5py compatible** — files round-trip with Python's h5py library
- **`no_std` support** — `rustyhdf5-format` works without the standard library
- **Dense attributes** — automatic fractal heap + B-tree v2 for >8 attributes
- **SHINES provenance** — SHA-256 content hashing, creator/timestamp metadata, integrity verification
- **Checksum validation** — Jenkins lookup3 checksum verification on read
- **Zero C dependencies** — no libhdf5, no build scripts, pure Rust

## Performance Benchmarks

Benchmarked on Apple MacBook M3 Max. Comparisons against the C HDF5 library (via hdf5-rust bindings).

| Operation | rustyhdf5 | C HDF5 | Result |
|---|---|---|---|
| Metadata ops (open + navigate + read attrs) | 0.2–1.5 µs | 18–45 µs | **12–90× faster** |
| Contiguous writes (1M f64) | 4.8 ms | 8.6 ms | **1.8× faster** |
| Contiguous reads (1M f64, mmap) | ~0 µs (zero-copy) | 1.54 ms | **zero-copy (P0)** |
| Chunked reads (1M f64, cached) | < 1 ms (hash index + LRU) | 4.3 ms | **cached (P1)** |
| Compressed reads (deflate, 1M f64) | < 200 ms (zlib-ng) | 625 ms | **3×+ faster (P2)** |
| File open | 377 µs (mmap) | 20.9 ms | **55× faster** |
| Vector search (IVF-PQ, 100K) | 380 µs | N/A (not in C HDF5) | **6.2× faster than numpy** |

**Key optimizations:**
- **P0 — Zero-copy contiguous reads** via memory-mapped I/O. Data is accessed directly from the OS page cache with no allocation or copy.
- **P1 — Chunk cache with hash-based index** and LRU eviction. Repeated access to the same chunks avoids redundant decompression.
- **P2 — Fast deflate** with optional zlib-ng backend or Apple Compression Framework for native hardware acceleration.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Applications                        │
├──────────┬──────────────┬───────────┬───────────────────┤
│ rustyhdf5 │ rustyhdf5-py  │ rustyhdf5- │ rustyhdf5-netcdf4  │
│ (Hi-API) │ (PyO3)       │ ann       │                   │
├──────────┴──────────────┴───────────┴───────────────────┤
│                   rustyhdf5-derive (proc macros)         │
├──────────┬──────────────┬───────────────────────────────┤
│ rustyhdf5 │ rustyhdf5-    │ rustyhdf5-accel (SIMD)         │
│ -io      │ filters      │ rustyhdf5-gpu   (wgpu)         │
├──────────┴──────────────┴───────────────────────────────┤
│              rustyhdf5-format (no_std core)               │
├─────────────────────────────────────────────────────────┤
│              rustyhdf5-types (type definitions)           │
└─────────────────────────────────────────────────────────┘
```

- **rustyhdf5-types** — HDF5 type system definitions (bottom layer, no dependencies)
- **rustyhdf5-format** — Binary format parsing and writing (`no_std` compatible core)
- **rustyhdf5-io** / **rustyhdf5-filters** — I/O backends and compression pipeline
- **rustyhdf5-accel** / **rustyhdf5-gpu** — Hardware acceleration (SIMD, compute shaders)
- **rustyhdf5** — High-level ergonomic API
- **rustyhdf5-derive** / **rustyhdf5-py** / **rustyhdf5-netcdf4** / **rustyhdf5-ann** — Extensions

## Crates

| Crate | Description |
|---|---|
| `rustyhdf5-format` | Binary format parsing and writing (`no_std` compatible) |
| `rustyhdf5-types` | HDF5 type system definitions |
| `rustyhdf5-io` | I/O abstraction layer (buffered, mmap) |
| `rustyhdf5-filters` | Filter/compression pipeline (deflate, shuffle, fletcher32) |
| `rustyhdf5-derive` | Proc macros for deriving HDF5 traits |
| `rustyhdf5` | High-level API for reading and writing files |
| `rustyhdf5-netcdf4` | NetCDF-4 compatibility layer |
| `rustyhdf5-ann` | HNSW approximate nearest-neighbor index stored in HDF5 |
| `rustyhdf5-accel` | SIMD acceleration (NEON, AVX2, AVX-512) |
| `rustyhdf5-gpu` | GPU compute via wgpu |
| `rustyhdf5-py` | Python bindings via PyO3 |

## Feature Flags

Feature flags on `rustyhdf5` (the high-level crate):

| Flag | Default | Description |
|---|---|---|
| `mmap` | yes | Memory-mapped file I/O for zero-copy reads |
| `fast-deflate` | no | Use zlib-ng for faster deflate compression/decompression |
| `parallel` | no | Parallel chunk I/O via rayon |

Feature flags on `rustyhdf5-filters`:

| Flag | Default | Description |
|---|---|---|
| `fast-deflate` | no | zlib-ng backend for deflate |
| `apple-compression` | no | Apple Compression Framework backend (macOS/iOS) |

Feature flags on `rustyhdf5-format`:

| Flag | Default | Description |
|---|---|---|
| `std` | yes | Standard library support (disable for `no_std`) |
| `deflate` | yes | Deflate compression support |
| `checksum` | yes | Jenkins lookup3 checksum verification |
| `provenance` | yes | SHA-256 provenance attributes |
| `parallel` | no | Parallel chunk encoding via rayon |
| `fast-checksum` | no | CRC32 acceleration via `crc32fast` |
| `fast-deflate` | no | zlib-ng backend for deflate |

## Quick Start

### High-level API (recommended)

```toml
[dependencies]
rustyhdf5 = "0.1"
```

```rust
use rustyhdf5::{File, FileBuilder, AttrValue};

// Write
let mut builder = FileBuilder::new();
builder.create_dataset("temperatures")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3]);
builder.set_attr("version", AttrValue::I64(1));
builder.write("output.h5").unwrap();

// Read
let file = File::open("output.h5").unwrap();
let ds = file.dataset("temperatures").unwrap();
let values = ds.read_f64().unwrap();
assert_eq!(values, vec![22.5, 23.1, 21.8]);
```

### Low-level format API

For fine-grained control, use `rustyhdf5-format` directly:

```toml
[dependencies]
rustyhdf5-format = "0.1"
```

#### Writing a dataset

```rust
use rustyhdf5_format::file_writer::{FileWriter, AttrValue};

let mut fw = FileWriter::new();
fw.create_dataset("data")
    .with_f64_data(&[1.0, 2.0, 3.0])
    .with_shape(&[3]);
let bytes = fw.finish().unwrap();
std::fs::write("output.h5", &bytes).unwrap();
```

#### Reading a dataset

```rust
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::data_read::{read_raw_data, read_as_f64};

let file_data = std::fs::read("output.h5").unwrap();
let offset = find_signature(&file_data).unwrap();
let sb = Superblock::parse(&file_data, offset).unwrap();
let addr = resolve_path_any(&file_data, &sb, "data").unwrap();
let hdr = ObjectHeader::parse(&file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();

let ds = Dataspace::parse(
    &hdr.messages.iter().find(|m| m.msg_type == MessageType::Dataspace).unwrap().data,
    sb.length_size,
).unwrap();
let (dt, _) = Datatype::parse(
    &hdr.messages.iter().find(|m| m.msg_type == MessageType::Datatype).unwrap().data,
).unwrap();
let layout = DataLayout::parse(
    &hdr.messages.iter().find(|m| m.msg_type == MessageType::DataLayout).unwrap().data,
    sb.offset_size, sb.length_size,
).unwrap();

let raw = read_raw_data(&file_data, &layout, &ds, &dt).unwrap();
let values = read_as_f64(&raw, &dt).unwrap();
assert_eq!(values, vec![1.0, 2.0, 3.0]);
```

### Chunked + compressed dataset

```rust
use rustyhdf5_format::file_writer::FileWriter;

let mut fw = FileWriter::new();
let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
fw.create_dataset("data")
    .with_f64_data(&data)
    .with_shape(&[1000])
    .with_chunks(&[100])
    .with_shuffle()
    .with_deflate(6);
let bytes = fw.finish().unwrap();
```

### Resizable dataset with maxshape

```rust
use rustyhdf5_format::file_writer::FileWriter;

let mut fw = FileWriter::new();
let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
fw.create_dataset("data")
    .with_f64_data(&data)
    .with_shape(&[50])
    .with_chunks(&[10])
    .with_maxshape(&[u64::MAX]); // unlimited dimension
let bytes = fw.finish().unwrap();
// h5py can open this file and resize/append to the dataset
```

### Groups and attributes

```rust
use rustyhdf5_format::file_writer::{FileWriter, AttrValue};

let mut fw = FileWriter::new();
fw.set_root_attr("version", AttrValue::I64(1));

let mut grp = fw.create_group("experiment");
grp.create_dataset("temperatures")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3])
    .set_attr("unit", AttrValue::String("celsius".into()));
let g = grp.finish();
fw.add_group(g);

let bytes = fw.finish().unwrap();
```

### Data provenance

```rust
use rustyhdf5_format::file_writer::FileWriter;

let mut fw = FileWriter::new();
fw.create_dataset("sensor")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3])
    .with_provenance("my-app/v1.0", "2026-02-19T12:00:00Z", Some("sensor_42"));
let bytes = fw.finish().unwrap();
// The file now contains _provenance_sha256, _provenance_creator,
// _provenance_timestamp, and _provenance_source attributes.
// Use rustyhdf5_format::provenance::verify_dataset() to check integrity.
```

## License

MIT
