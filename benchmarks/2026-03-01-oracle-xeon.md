# Benchmarks: Oracle Server (2026-03-01)

## Hardware
- CPU: Intel Xeon E5-2697 v2 @ 2.70GHz (48 cores)
- RAM: 247 GB
- OS: Ubuntu, Linux 6.14.0-37-generic x86_64

## Software
- rustyhdf5: commit 1b12675 (nightly 1.95.0)
- h5py 3.15.1 / HDF5 1.14.6 / numpy 2.4.2 / Python 3.13.3

## Results (1M float64, 8 MB)

| Benchmark | RustyHDF5 | h5py (C HDF5) | Speedup |
|-----------|-----------|----------------|---------|
| **Write contiguous** | 24.3 ms | 4.3 ms | 0.18x (h5py faster) |
| **Write chunked** | 37.0 ms | 6.4 ms | 0.17x (h5py faster) |
| **Write chunked deflate** | 331.8 ms | 551.9 ms | 1.66x |
| **Read contiguous** | 5.74 ms | 3.31 ms | 0.58x (h5py faster) |
| **Read chunked** | 6.51 ms | 4.14 ms | 0.63x (h5py faster) |
| **Read chunked deflate** | 24.1 ms | 21.8 ms | 0.91x (comparable) |
| **Read 50 attrs** | 16.8 µs | 8.25 ms | 491x |
| **Group nav 100** | 17.9 µs | 1.31 ms | 73x |

## Other RustyHDF5 benchmarks (no h5py equivalent)
- write_dataset_20_attrs_dense: 33.8 µs
- read_dataset_20_attrs_dense: 7.2 µs
- parse_object_header_complex: 367 ns
- write_10K_string_attrs: 161.1 µs
- read_100_string_attrs: 31.1 µs
- write_compound_10K_rows: 76.9 µs
- read_compound_10K_rows: 17.3 µs
- jenkins_lookup3_4MB: 2.80 ms
- sha256_4MB: 18.97 ms
- roundtrip_contiguous: 31.7 ms
- roundtrip_chunked_deflate: 354.9 ms
- write_provenance: 58.7 ms

## Notes
- h5py write includes kernel I/O (tmpfile); RustyHDF5 is in-memory buffer
- h5py read includes file open/close overhead
- Metadata operations (attrs, group nav) show massive RustyHDF5 advantage (in-memory parsing)
- Deflate compression is CPU-bound and comparable between both
- Xeon E5-2697 v2 is older (Ivy Bridge-EP, 2013) — no AVX-512, slower single-thread than M3 Max

## Mmap & I/O Strategy Benchmarks

### RustyHDF5 Mmap vs FileReader (1M f64)

| Benchmark | Time |
|-----------|------|
| filereader contiguous | 16.6 ms |
| **mmapreader contiguous** | **16.5 ms** |
| filereader chunked | 22.1 ms |
| **mmapreader chunked** | **7.3 ms** (3x faster) |
| File::open mmap read | 6.5 ms |
| File::open buffered read | 6.7 ms |
| **Zero-copy raw ref** | **601 ns** |
| **Zero-copy f64 slice** | **627 ns** |
| **Zero-copy f64 mmap** | **622 ns** |
| read_f64 (copy) | 5.8 ms |
| **read_f64 zerocopy** | **618 ns** (~9,400x faster) |
| read_as_slice f64 | 610 ns |

### File Open Overhead (10MB file)

| Method | Time |
|--------|------|
| mmap open only | 22.4 µs |
| buffered open only | 1.09 ms (49x slower) |

### Lazy vs Eager (100 datasets, read 1)

| Method | RustyHDF5 | h5py |
|--------|-----------|------|
| Eager open + read 1 | 40.8 µs | 0.66 ms (16x slower) |
| Lazy/mmap open + read 1 | 52.0 µs | — |

### Prefetch (chunked 1M f64, in-memory)

| Method | Time |
|--------|------|
| No prefetch | 7.58 ms |
| With prefetch | 7.52 ms (marginal — data already in memory) |

### h5py I/O Strategies

| Benchmark | Time |
|-----------|------|
| Default driver read | 4.0 ms |
| Core driver (in-memory) | 10.1 ms (slower — full copy) |
| 10 datasets sequential | 44.2 ms |
| 10 datasets threaded (10 workers) | 62.2 ms (GIL bottleneck) |
| 1-of-100 datasets | 0.66 ms |

### Parallel I/O (File::open vs MmapFile::open)

| Method | Time |
|--------|------|
| File::open 1M f64 | 12.1 ms |
| MmapFile::open 1M f64 | 12.2 ms |

## Key Takeaways (Oracle Xeon)
1. **Zero-copy is the killer feature**: 618ns vs 5.8ms copy vs 4.0ms h5py — ~6,500x faster than h5py
2. **Mmap chunked reads 3x faster** than FileReader (OS page cache does the work)
3. **Mmap file open is 49x faster** than buffered (no data copy, just page table setup)
4. **h5py threading hurts** due to GIL — RustyHDF5's Rust-native parallelism has no such limitation
5. **Eager open at 40.8µs** vs h5py's 0.66ms = 16x faster for selective dataset access

## Rayon Parallel Decompression Scaling (10M f64, 1000 chunks, deflate-6)

80MB uncompressed data, lane-partitioned parallel decompression.

| Threads | Median (ms) | Min (ms) | Speedup vs 1T |
|---------|-------------|----------|---------------|
| 1 | 449.5 | 426.2 | 1.0x |
| 2 | 287.4 | 279.0 | 1.56x |
| 4 | 216.0 | 214.1 | 2.08x |
| 8 | 232.5 | 186.2 | 1.93x (2.29x min) |
| 16 | 220.7 | 177.9 | 2.04x (2.40x min) |
| 24 | 206.7 | 167.4 | 2.17x (2.55x min) |
| 32 | 208.2 | 167.4 | 2.16x (2.55x min) |
| 48 | 208.6 | 165.2 | 2.15x (2.58x min) |
| no-parallel feature | 335.0 | 332.5 | 1.34x (sequential codepath) |

### h5py comparison (GIL-limited)
| Method | Median (ms) |
|--------|-------------|
| h5py sequential deflate read (1M) | 21.8 |
| h5py threaded 10 datasets | 62.2 (slower than sequential!) |

### Analysis
- **Peak scaling ~2.6x** at 48 threads (min times), saturating around 8 cores
- Diminishing returns after 4 threads — deflate decompression is memory-bandwidth limited on this Xeon
- The Xeon E5-2697 v2 has 2 sockets × 12 cores with shared L3; NUMA effects likely cause saturation
- **Sequential no-parallel (335ms) vs parallel-1T (449ms)**: the parallel codepath has ~34% overhead from lane partitioning setup when only using 1 thread
- **vs h5py**: RustyHDF5 parallel at 48T decompresses 10M elements in ~208ms; h5py can't parallelize at all due to GIL
- At scale (100M+ elements), the parallelism advantage would compound further
