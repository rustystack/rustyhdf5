//! Integration tests for mmap, parallel decompression, fast compression,
//! and SIMD checksum features.

use purehdf5::{AttrValue, File, FileBuilder};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("purehdf5_test_{name}"))
}

fn make_contiguous_f64_file(values: &[f64]) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("data").with_f64_data(values);
    b.finish().unwrap()
}

fn make_grouped_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    let mut g = b.create_group("sensors");
    g.create_dataset("temperature")
        .with_f64_data(&[22.5, 23.1, 21.8]);
    g.create_dataset("humidity").with_i32_data(&[45, 50, 55]);
    g.set_attr("location", AttrValue::String("lab".into()));
    let finished = g.finish();
    b.add_group(finished);
    b.finish().unwrap()
}

fn write_to_temp(name: &str, data: &[u8]) -> PathBuf {
    let path = temp_file(name);
    std::fs::write(&path, data).unwrap();
    path
}

// ===========================================================================
// Part 1: MmapFile tests
// ===========================================================================

#[cfg(feature = "mmap")]
mod mmap_tests {
    use super::*;
    use purehdf5::MmapFile;

    // Test 1: MmapFile reads same data as File for contiguous dataset
    #[test]
    fn mmap_reads_same_as_file_contiguous() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 1.1).collect();
        let bytes = make_contiguous_f64_file(&values);
        let path = write_to_temp("mmap_contig.h5", &bytes);

        let file = File::open(&path).unwrap();
        let mmap = MmapFile::open(&path).unwrap();

        let file_data = file.dataset("data").unwrap().read_f64().unwrap();
        let mmap_data = mmap.dataset("data").unwrap().read_f64().unwrap();
        assert_eq!(file_data, mmap_data);
        assert_eq!(mmap_data, values);

        std::fs::remove_file(&path).ok();
    }

    // Test 2: MmapFile reads same data as File for grouped dataset
    #[test]
    fn mmap_reads_same_as_file_grouped() {
        let bytes = make_grouped_file();
        let path = write_to_temp("mmap_grouped.h5", &bytes);

        let file = File::open(&path).unwrap();
        let mmap = MmapFile::open(&path).unwrap();

        let file_temps = file
            .dataset("sensors/temperature")
            .unwrap()
            .read_f64()
            .unwrap();
        let mmap_temps = mmap
            .dataset("sensors/temperature")
            .unwrap()
            .read_f64()
            .unwrap();
        assert_eq!(file_temps, mmap_temps);

        let file_hum = file
            .dataset("sensors/humidity")
            .unwrap()
            .read_i32()
            .unwrap();
        let mmap_hum = mmap
            .dataset("sensors/humidity")
            .unwrap()
            .read_i32()
            .unwrap();
        assert_eq!(file_hum, mmap_hum);

        std::fs::remove_file(&path).ok();
    }

    // Test 3: MmapFile zero-copy: verify contiguous read returns slice from mmap
    #[test]
    fn mmap_zero_copy_contiguous() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let bytes = make_contiguous_f64_file(&values);
        let path = write_to_temp("mmap_zerocopy.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let ds = mmap.dataset("data").unwrap();
        let slice = ds.read_raw_slice().unwrap();

        // Contiguous dataset should return Some (zero-copy slice)
        assert!(slice.is_some());
        let raw = slice.unwrap();
        assert_eq!(raw.len(), 5 * 8); // 5 f64 values

        // Verify the slice points into the mmap'd region
        let mmap_bytes = mmap.as_bytes();
        let raw_ptr = raw.as_ptr() as usize;
        let mmap_start = mmap_bytes.as_ptr() as usize;
        let mmap_end = mmap_start + mmap_bytes.len();
        assert!(raw_ptr >= mmap_start && raw_ptr < mmap_end);

        std::fs::remove_file(&path).ok();
    }

    // Test 4: MmapFile group navigation
    #[test]
    fn mmap_group_navigation() {
        let bytes = make_grouped_file();
        let path = write_to_temp("mmap_groups.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let root = mmap.root();
        let groups = root.groups().unwrap();
        assert_eq!(groups.len(), 1);

        let sensors = mmap.group("sensors").unwrap();
        let mut datasets = sensors.datasets().unwrap();
        datasets.sort();
        assert_eq!(datasets, vec!["humidity", "temperature"]);

        std::fs::remove_file(&path).ok();
    }

    // Test 5: MmapFile attributes
    #[test]
    fn mmap_group_attrs() {
        let bytes = make_grouped_file();
        let path = write_to_temp("mmap_attrs.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let sensors = mmap.group("sensors").unwrap();
        let attrs = sensors.attrs().unwrap();
        assert!(
            matches!(attrs.get("location"), Some(AttrValue::String(s)) if s == "lab")
        );

        std::fs::remove_file(&path).ok();
    }

    // Test 6: MmapFile dataset shape and dtype
    #[test]
    fn mmap_dataset_shape_dtype() {
        let values: Vec<f64> = vec![1.0, 2.0, 3.0];
        let bytes = make_contiguous_f64_file(&values);
        let path = write_to_temp("mmap_shape.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let ds = mmap.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![3]);
        assert_eq!(ds.dtype().unwrap(), purehdf5::DType::F64);

        std::fs::remove_file(&path).ok();
    }

    // Test 7: MmapFile debug formatting
    #[test]
    fn mmap_debug_format() {
        let bytes = make_contiguous_f64_file(&[1.0]);
        let path = write_to_temp("mmap_debug.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let debug = format!("{mmap:?}");
        assert!(debug.contains("MmapFile"));
        assert!(debug.contains("size"));

        std::fs::remove_file(&path).ok();
    }

    // Test 8: File::open_mmap convenience constructor
    #[test]
    fn file_open_mmap() {
        let values = vec![42.0f64];
        let bytes = make_contiguous_f64_file(&values);
        let path = write_to_temp("mmap_convenience.h5", &bytes);

        let mmap = File::open_mmap(&path).unwrap();
        let ds = mmap.dataset("data").unwrap();
        let data = ds.read_f64().unwrap();
        assert_eq!(data, vec![42.0]);

        std::fs::remove_file(&path).ok();
    }

    // Test 9: MmapFile error on nonexistent file
    #[test]
    fn mmap_nonexistent_file() {
        let result = MmapFile::open("/tmp/purehdf5_nonexistent_12345.h5");
        assert!(result.is_err());
    }

    // Test 10: MmapFile read i32 dataset
    #[test]
    fn mmap_read_i32() {
        let mut b = FileBuilder::new();
        b.create_dataset("counts").with_i32_data(&[10, 20, 30]);
        let bytes = b.finish().unwrap();
        let path = write_to_temp("mmap_i32.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let data = mmap.dataset("counts").unwrap().read_i32().unwrap();
        assert_eq!(data, vec![10, 20, 30]);

        std::fs::remove_file(&path).ok();
    }

    // Test 11: MmapFile group dataset access
    #[test]
    fn mmap_group_dataset_access() {
        let bytes = make_grouped_file();
        let path = write_to_temp("mmap_grp_ds.h5", &bytes);

        let mmap = MmapFile::open(&path).unwrap();
        let sensors = mmap.group("sensors").unwrap();
        let ds = sensors.dataset("temperature").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![22.5, 23.1, 21.8]);

        std::fs::remove_file(&path).ok();
    }
}

// ===========================================================================
// Part 2: Parallel chunk decompression tests
// ===========================================================================

mod parallel_tests {
    // Test 12: Parallel decompression produces identical output to sequential
    #[test]
    fn parallel_matches_sequential() {
        use purehdf5_format::chunked_read::ChunkInfo;
        use purehdf5_format::filter_pipeline::{FilterDescription, FilterPipeline, FILTER_DEFLATE};
        use purehdf5_format::filters::compress_chunk;

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };

        // Create 10 chunks of compressed data
        let mut compressed_chunks: Vec<Vec<u8>> = Vec::new();
        let mut chunk_infos: Vec<ChunkInfo> = Vec::new();
        let elem_size = 8u32;
        let chunk_elems = 100usize;
        let chunk_bytes = chunk_elems * elem_size as usize;

        for i in 0..10 {
            let raw: Vec<u8> = (0..chunk_elems)
                .flat_map(|j| ((i * chunk_elems + j) as f64).to_le_bytes())
                .collect();
            let compressed = compress_chunk(&raw, &pipeline, elem_size).unwrap();
            compressed_chunks.push(compressed);
        }

        // Build a synthetic file with all chunks placed sequentially
        let mut file_data = vec![0u8; 1024 * 1024];
        let mut offset = 0x1000usize;
        for (i, chunk) in compressed_chunks.iter().enumerate() {
            file_data[offset..offset + chunk.len()].copy_from_slice(chunk);
            chunk_infos.push(ChunkInfo {
                chunk_size: chunk.len() as u32,
                filter_mask: 0,
                offsets: vec![(i * chunk_elems) as u64, 0],
                address: offset as u64,
            });
            offset += chunk.len() + 16;
        }

        // Sequential decompression
        let sequential: Vec<Vec<u8>> = chunk_infos
            .iter()
            .map(|ci| {
                let addr = ci.address as usize;
                let size = ci.chunk_size as usize;
                let raw = &file_data[addr..addr + size];
                purehdf5_format::filters::decompress_chunk(
                    raw,
                    &pipeline,
                    chunk_bytes,
                    elem_size,
                )
                .unwrap()
            })
            .collect();

        // Parallel decompression (if feature enabled)
        #[cfg(feature = "parallel")]
        {
            let parallel =
                purehdf5_format::parallel_read::decompress_chunks_parallel(
                    &file_data,
                    &chunk_infos,
                    &pipeline,
                    chunk_bytes,
                    elem_size,
                )
                .unwrap();
            assert_eq!(parallel.len(), sequential.len());
            for (p, s) in parallel.iter().zip(sequential.iter()) {
                assert_eq!(p, s);
            }
        }
    }

    // Test 13: Parallel threshold check
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_threshold() {
        assert!(!purehdf5_format::parallel_read::should_use_parallel(1));
        assert!(!purehdf5_format::parallel_read::should_use_parallel(4));
        assert!(purehdf5_format::parallel_read::should_use_parallel(5));
        assert!(purehdf5_format::parallel_read::should_use_parallel(100));
    }
}

// ===========================================================================
// Part 3: Fast compression tests
// ===========================================================================

mod fast_compression_tests {
    // Test 14: libdeflater decompression matches miniz_oxide byte-for-byte
    #[test]
    fn decompress_matches_miniz() {
        let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let compressed = purehdf5_filters::deflate_compress_miniz(&data, 6).unwrap();

        let miniz_result = purehdf5_filters::deflate_decompress_miniz(&compressed).unwrap();
        let current_result =
            purehdf5_filters::deflate_decompress(&compressed, data.len()).unwrap();

        assert_eq!(miniz_result, data);
        assert_eq!(current_result, data);
        assert_eq!(current_result, miniz_result);
    }

    // Test 15: Cross-backend compress/decompress
    #[test]
    fn cross_backend_roundtrip() {
        let data: Vec<u8> = (0..5000).map(|i| ((i as f64 * 0.1).sin() * 127.0 + 128.0) as u8).collect();

        // Compress with current backend, decompress with miniz
        let compressed = purehdf5_filters::deflate_compress(&data, 6).unwrap();
        let decompressed = purehdf5_filters::deflate_decompress_miniz(&compressed).unwrap();
        assert_eq!(decompressed, data);

        // Compress with miniz, decompress with current backend
        let compressed2 = purehdf5_filters::deflate_compress_miniz(&data, 6).unwrap();
        let decompressed2 =
            purehdf5_filters::deflate_decompress(&compressed2, data.len()).unwrap();
        assert_eq!(decompressed2, data);
    }

    // Test 16: Large data roundtrip
    #[test]
    fn large_data_roundtrip() {
        let data: Vec<u8> = (0..1_000_000)
            .map(|i| ((i as f64 * 0.001).cos() * 127.0 + 128.0) as u8)
            .collect();
        let compressed = purehdf5_filters::deflate_compress(&data, 6).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed =
            purehdf5_filters::deflate_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    // Test 17: Empty data roundtrip
    #[test]
    fn empty_roundtrip() {
        let compressed = purehdf5_filters::deflate_compress(&[], 6).unwrap();
        let decompressed =
            purehdf5_filters::deflate_decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    // Test 18: Different compression levels
    #[test]
    fn compression_levels() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        for level in [1, 6, 9] {
            let compressed = purehdf5_filters::deflate_compress(&data, level).unwrap();
            let decompressed =
                purehdf5_filters::deflate_decompress(&compressed, data.len()).unwrap();
            assert_eq!(decompressed, data, "failed at level {level}");
        }
    }
}

// ===========================================================================
// Part 4: SIMD checksum tests
// ===========================================================================

mod checksum_tests {
    use purehdf5_format::checksum;

    // Test 19: Fast CRC32 matches software CRC32
    #[test]
    fn crc32_fast_matches_software() {
        let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let fast = checksum::crc32(&data);
        let software = checksum::crc32_software(&data);
        assert_eq!(fast, software);
    }

    // Test 20: CRC32 known value
    #[test]
    fn crc32_known_value() {
        assert_eq!(checksum::crc32(b"123456789"), 0xCBF43926);
    }

    // Test 21: CRC32 empty
    #[test]
    fn crc32_empty() {
        assert_eq!(checksum::crc32(b""), 0);
    }

    // Test 22: Fletcher32 optimized matches original behavior
    #[test]
    fn fletcher32_roundtrip_via_filters() {
        use purehdf5_format::filter_pipeline::{
            FilterDescription, FilterPipeline, FILTER_FLETCHER32,
        };
        use purehdf5_format::filters::{compress_chunk, decompress_chunk};

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_FLETCHER32,
                name: None,
                flags: 0,
                client_data: vec![],
            }],
        };

        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let with_checksum = compress_chunk(&data, &pipeline, 1).unwrap();
        let verified = decompress_chunk(&with_checksum, &pipeline, data.len(), 1).unwrap();
        assert_eq!(verified, data);
    }

    // Test 23: Fletcher32 optimized large data
    #[test]
    fn fletcher32_large_data() {
        use purehdf5_format::filter_pipeline::{
            FilterDescription, FilterPipeline, FILTER_FLETCHER32,
        };
        use purehdf5_format::filters::{compress_chunk, decompress_chunk};

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_FLETCHER32,
                name: None,
                flags: 0,
                client_data: vec![],
            }],
        };

        // Large data to exercise the block optimization path (>720 bytes)
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let with_checksum = compress_chunk(&data, &pipeline, 1).unwrap();
        let verified = decompress_chunk(&with_checksum, &pipeline, data.len(), 1).unwrap();
        assert_eq!(verified, data);
    }

    // Test 24: CRC32 deterministic
    #[test]
    fn crc32_deterministic() {
        let data = b"hello world, this is a test of crc32";
        let h1 = checksum::crc32(data);
        let h2 = checksum::crc32(data);
        assert_eq!(h1, h2);
    }

    // Test 25: CRC32 different inputs differ
    #[test]
    fn crc32_different_inputs() {
        let h1 = checksum::crc32(b"hello");
        let h2 = checksum::crc32(b"world");
        assert_ne!(h1, h2);
    }
}

// ===========================================================================
// Feature-gated compilation tests
// ===========================================================================

// Test 26: Compiles without mmap feature (implicit — this file compiles)
#[test]
fn compiles_without_optional_features() {
    // This test just verifies that the core API works without optional features
    let mut b = FileBuilder::new();
    b.create_dataset("x").with_f64_data(&[1.0]);
    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let data = file.dataset("x").unwrap().read_f64().unwrap();
    assert_eq!(data, vec![1.0]);
}
