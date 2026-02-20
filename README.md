# purehdf5-rs

A pure-Rust HDF5 reader and writer with zero C dependencies. Read and write HDF5 files from any platform Rust supports, including `no_std` environments.

## Features

- **Read and write** HDF5 files — contiguous, chunked, and compressed datasets
- **Compression** — deflate (gzip), shuffle, and fletcher32 filters
- **Resizable datasets** — `maxshape` support with extensible array indexes
- **Group hierarchy** — v1 (symbol table) and v2 (fractal heap) group navigation
- **Attributes** — scalar, array, and string attributes on datasets and groups
- **All numeric types** — f32, f64, i32, i64, u8, u64, plus fixed and variable-length strings
- **h5py compatible** — files round-trip with Python's h5py library
- **`no_std` support** — `purehdf5-format` works without the standard library
- **Dense attributes** — automatic fractal heap + B-tree v2 for >8 attributes
- **Compound/Enum/Array types** — struct-like, enumeration, and fixed-size array datasets
- **SHINES provenance** — SHA-256 content hashing, creator/timestamp metadata, integrity verification
- **Checksum validation** — Jenkins lookup3 checksum verification on read
- **Zero C dependencies** — no libhdf5, no build scripts, pure Rust

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
purehdf5-format = "0.1"
```

### Writing a dataset

```rust
use purehdf5_format::file_writer::{FileWriter, AttrValue};

let mut fw = FileWriter::new();
fw.create_dataset("data")
    .with_f64_data(&[1.0, 2.0, 3.0])
    .with_shape(&[3]);
let bytes = fw.finish().unwrap();
std::fs::write("output.h5", &bytes).unwrap();
```

### Reading a dataset

```rust
use purehdf5_format::signature::find_signature;
use purehdf5_format::superblock::Superblock;
use purehdf5_format::object_header::ObjectHeader;
use purehdf5_format::group_v2::resolve_path_any;
use purehdf5_format::message_type::MessageType;
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read::{read_raw_data, read_as_f64};

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
use purehdf5_format::file_writer::FileWriter;

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
use purehdf5_format::file_writer::FileWriter;

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
use purehdf5_format::file_writer::{FileWriter, AttrValue};

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
use purehdf5_format::file_writer::FileWriter;

let mut fw = FileWriter::new();
fw.create_dataset("sensor")
    .with_f64_data(&[22.5, 23.1, 21.8])
    .with_shape(&[3])
    .with_provenance("my-app/v1.0", "2026-02-19T12:00:00Z", Some("sensor_42"));
let bytes = fw.finish().unwrap();
// The file now contains _provenance_sha256, _provenance_creator,
// _provenance_timestamp, and _provenance_source attributes.
// Use purehdf5_format::provenance::verify_dataset() to check integrity.
```

## Crates

| Crate | Description |
|---|---|
| `purehdf5-format` | Binary format parsing and writing (`no_std` compatible) |
| `purehdf5-types` | HDF5 type system definitions |
| `purehdf5-io` | I/O abstraction layer |
| `purehdf5-filters` | Filter/compression pipeline |
| `purehdf5-derive` | Proc macros for deriving HDF5 traits |
| `purehdf5` | High-level API |

## License

MIT
