# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-19

### Added

- **File writing** — `FileWriter` API for creating HDF5 files with v3 superblock
  and v2 object headers
- **Contiguous datasets** — f64, f32, i32, i64, u8, u64 data types
- **Chunked datasets** — B-tree v1 chunk indexing with configurable chunk dimensions
- **Compression** — deflate (gzip), shuffle, and fletcher32 filters
- **Resizable datasets** — `maxshape` / extensible array index support
- **Groups** — hierarchical group creation with link messages
- **Attributes** — scalar, array, and string attributes on datasets and groups
- **Dense attributes** — automatic switch to fractal heap + B-tree v2 (type 8)
  storage when attribute count exceeds 8
- **Compound types** — struct-like datasets with named fields
- **Enum types** — i32 and u8 based enumerations
- **Array types** — fixed-size array elements
- **Variable-length strings** — global heap based VL string reading
- **SOHM** — shared object header message table parsing and resolution
- **SHINES provenance** — SHA-256 content hashing, creator/timestamp attributes,
  and `verify_dataset()` integrity checking
- **File reading** — full parsing of HDF5 v1/v2/v3 superblocks, v1/v2 object
  headers, all standard message types
- **Group traversal** — v1 (symbol table + B-tree v1) and v2 (fractal heap +
  B-tree v2) group navigation with `resolve_path_any()`
- **Checksum validation** — Jenkins lookup3 checksums for superblock, object
  headers, B-tree v2 headers/leaves, and fractal heap headers
- **Robustness** — bounds-checked parsers, recursion depth limits, graceful error
  handling on malformed input
- **`no_std` support** — core crate works without the standard library
- **h5py compatibility** — 26+ round-trip integration tests with Python's h5py
- **Benchmarks** — 14 Criterion benchmarks covering read, write, compression,
  dense attributes, provenance, and checksum operations
