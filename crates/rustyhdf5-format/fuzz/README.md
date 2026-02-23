# Fuzz Testing for rustyhdf5-format

Uses [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer) to test parser robustness against malformed inputs.

## Prerequisites

```bash
cargo install cargo-fuzz
rustup toolchain install nightly
```

## Fuzz Targets

| Target | Parser | Description |
|--------|--------|-------------|
| `fuzz_superblock` | `Superblock::parse` | Superblock parsing (v0-v3) with signature search |
| `fuzz_object_header` | `ObjectHeader::parse` | Object header v1/v2 with various offset/length sizes |
| `fuzz_datatype` | `Datatype::parse` | All 12 HDF5 datatype classes (recursive) |
| `fuzz_dataspace` | `Dataspace::parse` | Dataspace messages with various length sizes |
| `fuzz_fractal_heap` | `FractalHeapHeader::parse` | Fractal heap header parsing |
| `fuzz_btree_v2` | `BTreeV2Header::parse` | B-tree v2 header parsing |
| `fuzz_filter_pipeline` | `FilterPipeline::parse` | Filter pipeline messages (v1/v2) |
| `fuzz_full_file` | signature + superblock + root group | End-to-end file parsing chain |

## Running

Run a single target (runs indefinitely until stopped or a crash is found):

```bash
cd crates/rustyhdf5-format
cargo +nightly fuzz run fuzz_datatype
```

Run with a time limit (seconds):

```bash
cargo +nightly fuzz run fuzz_datatype -- -max_total_time=60
```

Run all targets for 30 seconds each:

```bash
for target in fuzz_superblock fuzz_object_header fuzz_datatype fuzz_dataspace \
              fuzz_fractal_heap fuzz_btree_v2 fuzz_filter_pipeline fuzz_full_file; do
    echo "=== $target ==="
    cargo +nightly fuzz run "$target" -- -max_total_time=30 -max_len=4096
done
```

## Reproducing Crashes

If a crash is found, the input is saved to `fuzz/artifacts/<target>/`. Reproduce with:

```bash
cargo +nightly fuzz run fuzz_datatype fuzz/artifacts/fuzz_datatype/crash-<hash>
```

Minimize the crashing input:

```bash
cargo +nightly fuzz tmin fuzz_datatype fuzz/artifacts/fuzz_datatype/crash-<hash>
```

## Design

Each fuzz target feeds arbitrary bytes directly to a parser entry point. The parsers must never panic on any input -- they should return `Err(FormatError)` for malformed data. Any panic found by fuzzing is a bug that should be fixed with proper bounds checks and error returns.
